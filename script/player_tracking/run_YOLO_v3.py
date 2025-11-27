from ultralytics import YOLO
import cv2
import os
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import KMeans

# Load YOLO model (detection only, no pose)
model = YOLO("yolov8n.pt")

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
team_vote_history = defaultdict(lambda: deque(maxlen=15))  # Increased from 10 to 15
team_color_scores = defaultdict(lambda: deque(maxlen=10))  # Store raw color scores

def get_torso_region(bbox):
    """
    Get the torso region from bbox using simple heuristic.
    Focus on center portion to avoid grass.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    if box_width < 20 or box_height < 30:
        return None
    
    # Take center 60% width, upper-middle 30-65% height
    y_start = int(y1 + box_height * 0.30)
    y_end = int(y1 + box_height * 0.65)
    x_start = int(x1 + box_width * 0.20)
    x_end = int(x2 - box_width * 0.20)
    
    return (x_start, y_start, x_end, y_end)

def classify_uniform_by_clustering(frame, bbox, debug=False):
    """
    Classify uniform color using K-means clustering to find dominant colors.
    Filters out grass (green field) and skin tones automatically.
    Returns: (team, green_score, white_score) where team is 1, 2, or None
    """
    torso_bbox = get_torso_region(bbox)
    if torso_bbox is None:
        return None, 0, 0
    
    x1, y1, x2, y2 = torso_bbox
    torso = frame[y1:y2, x1:x2]
    
    if torso.size == 0 or torso.shape[0] < 10 or torso.shape[1] < 10:
        return None, 0, 0
    
    # Reshape to list of pixels
    pixels = torso.reshape(-1, 3).astype(np.float32)
    
    # Convert to HSV for better color separation
    torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    pixels_hsv = torso_hsv.reshape(-1, 3).astype(np.float32)
    
    # Filter out grass pixels (dark green, high saturation)
    # Grass: Hue 35-85, Saturation > 60, Value 20-80
    grass_mask = (
        (pixels_hsv[:, 0] >= 35) & (pixels_hsv[:, 0] <= 85) &
        (pixels_hsv[:, 1] > 60) &
        (pixels_hsv[:, 2] >= 20) & (pixels_hsv[:, 2] <= 80)
    )
    
    # Filter out skin tones (Hue 0-25, moderate saturation)
    skin_mask = (
        (pixels_hsv[:, 0] >= 0) & (pixels_hsv[:, 0] <= 25) &
        (pixels_hsv[:, 1] >= 20) & (pixels_hsv[:, 1] <= 150) &
        (pixels_hsv[:, 2] >= 50)
    )
    
    # Keep only non-grass, non-skin pixels
    valid_mask = ~(grass_mask | skin_mask)
    valid_pixels = pixels[valid_mask]
    valid_pixels_hsv = pixels_hsv[valid_mask]
    
    if len(valid_pixels) < 50:  # Not enough valid pixels
        return None, 0, 0
    
    # Run K-means to find 3 dominant colors
    n_clusters = min(3, len(valid_pixels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=100)
    kmeans.fit(valid_pixels_hsv)
    
    # Get cluster centers and their sizes
    centers_hsv = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Find the largest cluster (likely the uniform)
    cluster_sizes = np.bincount(labels)
    dominant_clusters = np.argsort(cluster_sizes)[::-1]  # Largest first
    
    # Check top 2 clusters for uniform colors
    green_score = 0
    white_score = 0
    
    for i in range(min(2, len(dominant_clusters))):
        cluster_idx = dominant_clusters[i]
        cluster_size_ratio = cluster_sizes[cluster_idx] / len(valid_pixels)
        
        h, s, v = centers_hsv[cluster_idx]
        
        # Check if this cluster is green (bright green jersey)
        # Green: Hue 40-85, Saturation > 40, Value > 80
        is_green = (40 <= h <= 85 and s > 40 and v > 80)
        
        # Check if this cluster is white
        # White: Low saturation < 50, High value > 140
        is_white = (s < 50 and v > 140)
        
        if is_green:
            green_score += cluster_size_ratio
        elif is_white:
            white_score += cluster_size_ratio
    
    if debug:
        print(f"    Clustering: green_score={green_score:.2f}, white_score={white_score:.2f}")
    
    # Classify based on scores
    team = None
    if green_score > 0.20 and green_score > white_score * 1.3:
        team = 1  # Green team
    elif white_score > 0.20 and white_score > green_score * 1.3:
        team = 2  # White team
    
    return team, green_score, white_score

def get_team_assignment(track_id, current_classification, green_score, white_score, vote_history, score_history):
    """
    Smooth team assignment using both voting and temporal averaging of color scores.
    This prevents flickering by considering history.
    """
    # Store raw scores for temporal smoothing
    if green_score > 0 or white_score > 0:
        score_history[track_id].append((green_score, white_score))
    
    # Add classification to vote history
    if current_classification is not None:
        vote_history[track_id].append(current_classification)
    
    # Need at least 5 frames before making decision
    if len(vote_history[track_id]) < 5:
        return None
    
    # Calculate average scores over time
    if len(score_history[track_id]) >= 5:
        recent_scores = list(score_history[track_id])[-10:]  # Last 10 frames
        avg_green = np.mean([s[0] for s in recent_scores])
        avg_white = np.mean([s[1] for s in recent_scores])
        
        # Use temporal average with higher confidence threshold
        if avg_green > 0.25 and avg_green > avg_white * 1.4:
            return 1
        elif avg_white > 0.25 and avg_white > avg_green * 1.4:
            return 2
    
    # Fallback to majority vote if scores are inconclusive
    votes = list(vote_history[track_id])
    team_1_votes = votes.count(1)
    team_2_votes = votes.count(2)
    
    # Require strong majority (60%+) to assign team
    total_votes = len(votes)
    if team_1_votes > team_2_votes and team_1_votes / total_votes > 0.6:
        return 1
    elif team_2_votes > team_1_votes and team_2_votes / total_votes > 0.6:
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
DEBUG_MODE = False  # Set to True for debug output
SHOW_TORSO = False  # Set to True to show torso boxes

print(f"Team 1: {TEAM_1_COLOR} uniforms")
print(f"Team 2: {TEAM_2_COLOR} uniforms")
print("Processing video with color clustering...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Track people and ball
    results = model.track(
        frame,
        persist=True,
        tracker='botsort.yaml',
        conf=0.15,
        iou=0.5,
        classes=[0, 32],
        verbose=False
    )[0]
    
    current_frame_ids = set()
    
    if DEBUG_MODE and frame_count % 30 == 0:
        print(f"\n--- Frame {frame_count} ---")
    
    # Process detections
    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
            current_frame_ids.add(track_id)
            missing_tracks[track_id] = 0
            
            # Smooth the bounding box
            smoothed_box = smooth_bbox(track_id, box, track_history)
            x1, y1, x2, y2 = map(int, smoothed_box)
            
            if cls == 0:  # Person
                # Classify uniform using color clustering
                debug_this = DEBUG_MODE and frame_count % 30 == 0
                current_classification, green_score, white_score = classify_uniform_by_clustering(
                    frame, smoothed_box, debug=debug_this
                )
                
                # Get team assignment with temporal smoothing
                team = get_team_assignment(
                    track_id, current_classification, green_score, white_score,
                    team_vote_history, team_color_scores
                )
                
                # Once assigned, lock in the team (no more flickering)
                if team is not None:
                    if track_id not in track_teams:
                        track_teams[track_id] = team
                    # If already assigned, only change if overwhelming evidence (90%+ opposite votes)
                    elif track_id in track_teams:
                        votes = list(team_vote_history[track_id])[-10:]
                        if len(votes) >= 10:
                            opposite_team = 2 if track_teams[track_id] == 1 else 1
                            opposite_votes = votes.count(opposite_team)
                            if opposite_votes >= 9:  # 90%+ evidence to switch
                                track_teams[track_id] = team
                
                if debug_this:
                    result = "UNCERTAIN"
                    if current_classification == 1:
                        result = "GREEN"
                    elif current_classification == 2:
                        result = "WHITE"
                    final_team = track_teams.get(track_id, "NONE")
                    print(f"  Track {track_id}: current={result}, assigned=T{final_team}")
                
                # Draw torso region if enabled
                if SHOW_TORSO:
                    torso_bbox = get_torso_region(smoothed_box)
                    if torso_bbox:
                        tx1, ty1, tx2, ty2 = torso_bbox
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 1)
                
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
            
            elif cls == 32:  # Ball
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
    cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1)
    cv2.putText(frame, "Teams:", (20, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Team 1 ({TEAM_1_COLOR})", (20, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors_display[1], 2)
    cv2.putText(frame, f"Team 2 ({TEAM_2_COLOR})", (20, legend_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_colors_display[2], 2)
    
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
print(f"\nTeam 1 ({TEAM_1_COLOR}) players: {sorted([tid for tid, team in track_teams.items() if team == 1])}")
print(f"Team 2 ({TEAM_2_COLOR}) players: {sorted([tid for tid, team in track_teams.items() if team == 2])}")