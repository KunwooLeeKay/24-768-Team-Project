from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict, deque

# Load YOLO models
model_detect = YOLO("yolov8n.pt")  # Detection and tracking
model_pose = YOLO("yolov8n-pose.pt")  # Pose estimation for torso detection

input_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
output_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/YOLO_track_interpolated.mp4"

# USER INPUT: Team uniform colors
TEAM_1_COLOR = "green"
TEAM_2_COLOR = "white"

# Open video
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output directory and video writer
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Multi-frame interpolation setup
HISTORY_FRAMES = 5
track_history = defaultdict(lambda: deque(maxlen=HISTORY_FRAMES))

# Team classification setup
track_teams = {}
team_colors_display = {1: (0, 255, 0), 2: (255, 255, 255)}
team_vote_history = defaultdict(lambda: deque(maxlen=10))

# COCO pose keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

def get_torso_region_from_pose(keypoints, bbox):
    """
    Extract torso region using pose keypoints.
    keypoints: shape (17, 3) where each row is [x, y, confidence]
    Returns: torso bbox (x1, y1, x2, y2) or None
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    
    # Extract relevant keypoints
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]
    
    # Check if keypoints are visible (confidence > 0.3)
    visible_points = []
    if left_shoulder[2] > 0.3:
        visible_points.append(left_shoulder[:2])
    if right_shoulder[2] > 0.3:
        visible_points.append(right_shoulder[:2])
    if left_hip[2] > 0.3:
        visible_points.append(left_hip[:2])
    if right_hip[2] > 0.3:
        visible_points.append(right_hip[:2])
    
    # Need at least 2 keypoints to define torso
    if len(visible_points) < 2:
        return None
    
    # Get bounding box of visible torso keypoints
    points = np.array(visible_points)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    
    # Add margin around torso
    margin_x = (x2 - x1) * 0.2
    margin_y = (y2 - y1) * 0.15
    
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = int(x2 + margin_x)
    y2 = int(y2 + margin_y)
    
    # Ensure torso region is within original bbox
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, bbox)
    x1 = max(bbox_x1, x1)
    y1 = max(bbox_y1, y1)
    x2 = min(bbox_x2, x2)
    y2 = min(bbox_y2, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return (x1, y1, x2, y2)

def get_fallback_torso_region(bbox):
    """
    Fallback method: Use center portion of bbox if pose detection fails.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    if box_width < 20 or box_height < 30:
        return None
    
    # Take center 50% width, upper 40-70% height
    y_start = int(y1 + box_height * 0.25)
    y_end = int(y1 + box_height * 0.65)
    x_start = int(x1 + box_width * 0.25)
    x_end = int(x2 - box_width * 0.25)
    
    return (x_start, y_start, x_end, y_end)

def classify_uniform_color(frame, torso_bbox, track_id=None, debug=False):
    """
    Classify uniform color from torso region.
    Returns: 1 for green team, 2 for white team, or None if uncertain
    """
    if torso_bbox is None:
        return None
    
    x1, y1, x2, y2 = torso_bbox
    
    # Extract torso region
    torso = frame[y1:y2, x1:x2]
    
    if torso.size == 0 or torso.shape[0] < 10 or torso.shape[1] < 10:
        return None
    
    # Apply slight Gaussian blur to reduce noise from numbers/logos
    torso_blurred = cv2.GaussianBlur(torso, (5, 5), 0)
    
    # Convert to HSV
    hsv = cv2.cvtColor(torso_blurred, cv2.COLOR_BGR2HSV)
    
    # Define color masks with more lenient thresholds
    # Green: Hue 35-85, Saturation > 25, Value > 20
    green_mask = cv2.inRange(hsv, np.array([35, 25, 20]), np.array([85, 255, 255]))
    
    # White: Saturation < 60, Value > 120
    white_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 60, 255]))
    
    # Filter out skin tones (Hue 0-25, Saturation 15-150, Value 50-255)
    skin_mask = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([25, 150, 255]))
    
    # Remove skin pixels from analysis
    valid_mask = cv2.bitwise_not(skin_mask)
    green_mask = cv2.bitwise_and(green_mask, valid_mask)
    white_mask = cv2.bitwise_and(white_mask, valid_mask)
    
    # Count pixels
    total_pixels = torso.shape[0] * torso.shape[1]
    valid_pixels = np.sum(valid_mask > 0)
    
    if valid_pixels < total_pixels * 0.2:  # Too much skin, unreliable
        if debug:
            print(f"  Track {track_id}: Too much skin detected ({valid_pixels}/{total_pixels})")
        return None
    
    green_pixels = np.sum(green_mask > 0)
    white_pixels = np.sum(white_mask > 0)
    
    green_ratio = green_pixels / valid_pixels
    white_ratio = white_pixels / valid_pixels
    
    if debug:
        print(f"  Track {track_id}: green={green_ratio:.2f}, white={white_ratio:.2f}, valid={valid_pixels}/{total_pixels}")
    
    # Classify based on dominant color (need at least 15% coverage)
    if green_ratio > 0.15 and green_ratio > white_ratio * 1.2:
        if debug:
            print(f"    -> GREEN TEAM")
        return 1  # Green team
    elif white_ratio > 0.15 and white_ratio > green_ratio * 1.2:
        if debug:
            print(f"    -> WHITE TEAM")
        return 2  # White team
    else:
        if debug:
            print(f"    -> UNCERTAIN")
        return None  # Uncertain

def get_team_assignment(track_id, current_classification, vote_history):
    """
    Smooth team assignment using voting over recent frames.
    """
    if current_classification is not None:
        vote_history[track_id].append(current_classification)
    
    if len(vote_history[track_id]) < 3:
        return None
    
    # Majority vote
    votes = list(vote_history[track_id])
    team_1_votes = votes.count(1)
    team_2_votes = votes.count(2)
    
    if team_1_votes > team_2_votes:
        return 1
    elif team_2_votes > team_1_votes:
        return 2
    else:
        return None

def smooth_bbox(track_id, bbox, history):
    """
    Smooth bounding box using weighted average of recent frames.
    """
    hist = history[track_id]
    hist.append(bbox)
    
    if len(hist) < 2:
        return bbox
    
    weights = np.linspace(0.5, 1.0, len(hist))
    weights = weights / weights.sum()
    
    smoothed = np.zeros(4)
    for i, (weight, box) in enumerate(zip(weights, hist)):
        smoothed += weight * np.array(box)
    
    return smoothed.astype(int)

def predict_position(track_id, history):
    """
    Predict next position based on velocity from recent frames.
    """
    hist = list(history[track_id])
    if len(hist) < 3:
        return None
    
    boxes = np.array(hist[-3:])
    vel = (boxes[-1] - boxes[0]) / 2
    predicted = boxes[-1] + vel
    return predicted.astype(int)

frame_count = 0
missing_tracks = defaultdict(int)
MAX_MISSING = 3
DEBUG_MODE = True  # Enable to see pose matching debug info
SHOW_POSE = True  # Show pose keypoints and torso regions
pose_detection_count = 0
fallback_count = 0

print(f"Team 1: {TEAM_1_COLOR} uniforms")
print(f"Team 2: {TEAM_2_COLOR} uniforms")
print("Loading models...")
print("Processing video...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Track people and ball
    results = model_detect.track(
        frame,
        persist=True,
        tracker='botsort.yaml',
        conf=0.15,
        iou=0.5,
        classes=[0, 32],
        verbose=False
    )[0]
    
    current_frame_ids = set()
    player_bboxes = {}  # Store player bboxes for pose estimation
    
    # First pass: collect detections
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            if cls == 0:  # Person
                smoothed_box = smooth_bbox(track_id, box, track_history)
                player_bboxes[track_id] = (smoothed_box, conf)
            current_frame_ids.add(track_id)
            missing_tracks[track_id] = 0
    
    # Run pose estimation on each player's cropped region
    pose_results = {}
    if len(player_bboxes) > 0:
        if DEBUG_MODE and frame_count % 30 == 0:
            print(f"Running pose on {len(player_bboxes)} players")
        
        for track_id, (bbox, _) in player_bboxes.items():
            try:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Add padding around bbox for better pose detection
                padding = 20
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(width, x2 + padding)
                y2_pad = min(height, y2 + padding)
                
                # Crop player region
                player_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_crop.size == 0:
                    if DEBUG_MODE and frame_count % 30 == 0:
                        print(f"  Track {track_id}: Empty crop")
                    continue
                
                # Run pose on cropped region
                pose_output = model_pose(player_crop, conf=0.1, verbose=False)[0]
                
                if pose_output.keypoints is not None and len(pose_output.keypoints.data) > 0:
                    # Get the first (most confident) detection
                    keypoints = pose_output.keypoints.data.cpu().numpy()[0]
                    
                    # Adjust keypoint coordinates back to full frame
                    keypoints_adjusted = keypoints.copy()
                    keypoints_adjusted[:, 0] += x1_pad  # Add x offset
                    keypoints_adjusted[:, 1] += y1_pad  # Add y offset
                    
                    pose_results[track_id] = keypoints_adjusted
                    
                    if DEBUG_MODE and frame_count % 30 == 0:
                        visible_kps = np.sum(keypoints[:, 2] > 0.3)
                        print(f"  Track {track_id}: Pose detected! {visible_kps}/17 keypoints visible")
                else:
                    if DEBUG_MODE and frame_count % 30 == 0:
                        print(f"  Track {track_id}: No pose detected in crop")
                        
            except Exception as e:
                if DEBUG_MODE and frame_count % 30 == 0:
                    print(f"  Track {track_id}: Pose error - {e}")
    
    # Second pass: classify and draw
    if DEBUG_MODE and frame_count % 30 == 0:
        print(f"\n--- Frame {frame_count} Debug ---")
    
    for track_id, (bbox, conf) in player_bboxes.items():
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get torso region from pose if available
        torso_bbox = None
        used_pose = False
        keypoints = None
        
        if track_id in pose_results:
            keypoints = pose_results[track_id]
            torso_bbox = get_torso_region_from_pose(keypoints, bbox)
            if torso_bbox is not None:
                used_pose = True
                pose_detection_count += 1
        
        # Fallback to heuristic method if pose failed
        if torso_bbox is None:
            torso_bbox = get_fallback_torso_region(bbox)
            fallback_count += 1
        
        # Classify uniform color
        if torso_bbox is not None:
            debug_this = DEBUG_MODE and frame_count % 30 == 0
            current_classification = classify_uniform_color(frame, torso_bbox, track_id, debug=debug_this)
            team = get_team_assignment(track_id, current_classification, team_vote_history)
            
            if team is not None:
                track_teams[track_id] = team
            
            # Draw torso region
            if SHOW_POSE:
                tx1, ty1, tx2, ty2 = torso_bbox
                # Green outline for pose-based, yellow for fallback
                torso_color = (0, 255, 0) if used_pose else (0, 255, 255)
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), torso_color, 1)
                
                # Draw pose keypoints if available
                if keypoints is not None and used_pose:
                    # Draw shoulder and hip keypoints
                    shoulder_hip_indices = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
                    for idx in shoulder_hip_indices:
                        kp = keypoints[idx]
                        if kp[2] > 0.3:  # confidence threshold
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)  # Magenta dots
                    
                    # Draw lines between keypoints to show torso
                    if (keypoints[LEFT_SHOULDER][2] > 0.3 and keypoints[RIGHT_SHOULDER][2] > 0.3):
                        p1 = (int(keypoints[LEFT_SHOULDER][0]), int(keypoints[LEFT_SHOULDER][1]))
                        p2 = (int(keypoints[RIGHT_SHOULDER][0]), int(keypoints[RIGHT_SHOULDER][1]))
                        cv2.line(frame, p1, p2, (255, 0, 255), 1)
                    
                    if (keypoints[LEFT_HIP][2] > 0.3 and keypoints[RIGHT_HIP][2] > 0.3):
                        p1 = (int(keypoints[LEFT_HIP][0]), int(keypoints[LEFT_HIP][1]))
                        p2 = (int(keypoints[RIGHT_HIP][0]), int(keypoints[RIGHT_HIP][1]))
                        cv2.line(frame, p1, p2, (255, 0, 255), 1)
                    
                    # Draw lines connecting shoulders to hips
                    if (keypoints[LEFT_SHOULDER][2] > 0.3 and keypoints[LEFT_HIP][2] > 0.3):
                        p1 = (int(keypoints[LEFT_SHOULDER][0]), int(keypoints[LEFT_SHOULDER][1]))
                        p2 = (int(keypoints[LEFT_HIP][0]), int(keypoints[LEFT_HIP][1]))
                        cv2.line(frame, p1, p2, (255, 0, 255), 1)
                    
                    if (keypoints[RIGHT_SHOULDER][2] > 0.3 and keypoints[RIGHT_HIP][2] > 0.3):
                        p1 = (int(keypoints[RIGHT_SHOULDER][0]), int(keypoints[RIGHT_SHOULDER][1]))
                        p2 = (int(keypoints[RIGHT_HIP][0]), int(keypoints[RIGHT_HIP][1]))
                        cv2.line(frame, p1, p2, (255, 0, 255), 1)
        
        # Draw player bbox with team color
        if track_id in track_teams:
            team = track_teams[track_id]
            color = team_colors_display[team]
            team_name = TEAM_1_COLOR if team == 1 else TEAM_2_COLOR
            label = f"T{team} ({team_name}) P{track_id}"
        else:
            color = (128, 128, 128)
            label = f"Player {track_id}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - baseline - 5),
                     (x1 + label_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw ball tracking
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls in zip(boxes, track_ids, classes):
            if cls == 32:  # Ball
                smoothed_box = smooth_bbox(track_id, box, track_history)
                x1, y1, x2, y2 = map(int, smoothed_box)
                
                color = (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Ball {track_id}", (x1, max(15, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trail
                if len(track_history[track_id]) > 1:
                    points = []
                    for hist_box in track_history[track_id]:
                        center_x = int((hist_box[0] + hist_box[2]) / 2)
                        center_y = int((hist_box[1] + hist_box[3]) / 2)
                        points.append((center_x, center_y))
                    
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 255, 0), 2)
    
    # Interpolate missing tracks
    all_tracked_ids = set(track_history.keys())
    missing_ids = all_tracked_ids - current_frame_ids
    
    for track_id in missing_ids:
        missing_tracks[track_id] += 1
        
        if missing_tracks[track_id] <= MAX_MISSING:
            predicted_box = predict_position(track_id, track_history)
            
            if predicted_box is not None:
                x1, y1, x2, y2 = map(int, predicted_box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if track_id in track_teams:
                    team = track_teams[track_id]
                    color = tuple(c // 2 for c in team_colors_display[team])
                else:
                    color = (64, 64, 64)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, f"ID {track_id} (pred)", (x1, max(15, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                track_history[track_id].append(predicted_box)
    
    # Draw legend
    legend_y = 30
    cv2.rectangle(frame, (10, 10), (280, 140), (0, 0, 0), -1)
    cv2.putText(frame, "Teams:", (20, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Team 1 ({TEAM_1_COLOR})", (20, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors_display[1], 2)
    cv2.putText(frame, f"Team 2 ({TEAM_2_COLOR})", (20, legend_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors_display[2], 2)
    
    if SHOW_POSE:
        cv2.putText(frame, "Torso: Green=Pose, Yellow=Fallback", (20, legend_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Magenta: Pose keypoints", (20, legend_y + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    out.write(frame)
    
    if frame_count % 30 == 0:
        team1_count = sum(1 for t in track_teams.values() if t == 1)
        team2_count = sum(1 for t in track_teams.values() if t == 2)
        print(f"Frame {frame_count}: Team 1={team1_count}, Team 2={team2_count}, Tracking {len(current_frame_ids)} objects")
    if frame_count == 300: break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nTracking complete!")
print(f"Saved tracked video to: {output_path}")
print(f"Total frames processed: {frame_count}")
print(f"Pose detections: {pose_detection_count}, Fallback method: {fallback_count}")
print(f"\nTeam 1 ({TEAM_1_COLOR}) players: {sorted([tid for tid, team in track_teams.items() if team == 1])}")
print(f"Team 2 ({TEAM_2_COLOR}) players: {sorted([tid for tid, team in track_teams.items() if team == 2])}")