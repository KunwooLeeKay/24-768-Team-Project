from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import sys
from tqdm import tqdm
import os

# --- IMPORT ROBOFLOW INFERENCE ---
try:
    from inference import get_model
except ImportError:
    print("Please run: pip install inference")
    sys.exit()

# --- CONFIGURATION ---
INPUT_PATH = '/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/video_clip.mp4'
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/tracking_localization_v6_CLEAN.mp4"

# *** MODEL SETTINGS ***
PLAYER_MODEL = "yolov8x.pt"
ROBOFLOW_MODEL_ID = "soccer-ball-detection-2/3" 
ROBOFLOW_KEY = "QXMNvrMDhPsZcdBNFfnj" 

# *** LIMIT PROCESSING ***
MAX_FRAMES = 300  # Set to None for full video

# --- LOCALIZATION CONSTANTS ---
ROTATION_SENSITIVITY = 0.0005
TILT_SENSITIVITY = 0
MINIMAP_WIDTH = 600   # Smaller, cleaner map
MINIMAP_HEIGHT = 400
FEATURE_REFRESH_THRESHOLD = 50 
RANSAC_REPROJECTION_THRESHOLD = 5.0 
# INCREASED CROP: Ignore top 40% of screen (Crowd/Stands cause drift)
FEATURE_CROP_TOP_PERCENT = 0.40 
PIVOT_OFFSET_Y = 400 
BASE_FOV_ANGLE = np.deg2rad(45)

# --- VISUALS ---
COLOR_TEAM_1 = (0, 200, 0)      # Green
COLOR_TEAM_2 = (220, 220, 220)  # White
COLOR_BALL_DOT = (0, 0, 255)    # Red
COLOR_FOV_LINES = (50, 50, 50)  # Dark Gray for view cone

def safe_int(val):
    if val is None or np.isnan(val): return 0
    return int(val)

class TrackEntity:
    def __init__(self, track_id):
        self.id = track_id
        self.frames = []
        self.bboxes = []
        self.features = []
        self.assigned_team = None
    
    def add_observation(self, frame_idx, bbox, feature=None):
        self.frames.append(frame_idx)
        self.bboxes.append(bbox)
        if feature is not None: self.features.append(feature)

    def sort_by_frame(self):
        sorted_indices = np.argsort(self.frames)
        self.frames = [self.frames[i] for i in sorted_indices]
        self.bboxes = [self.bboxes[i] for i in sorted_indices]
        if self.features:
            self.features = [self.features[i] for i in sorted_indices if i < len(self.features)]

class OfflineTracker:
    def __init__(self):
        print("--- Loading Models ---")
        self.model_players = YOLO(PLAYER_MODEL)
        
        print(f"Loading Ball Model via API: {ROBOFLOW_MODEL_ID}...")
        try:
            self.model_ball = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_KEY)
        except Exception as e:
            print(f"CRITICAL ERROR loading Roboflow model: {e}")
            sys.exit()
        
        self.tracks = {}        
        self.ball_raw = []      
        self.ball_clean = []    

    def create_soccer_field(self, width, height):
        # Professional Dark Theme Field
        field_color = (40, 45, 40)   # Dark Slate/Green
        line_color = (200, 200, 200) # Light Grey
        line_thickness = 2
        
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = field_color
        margin = 30
        w_field = width - 2 * margin; h_field = height - 2 * margin
        
        # Field markings
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), line_color, line_thickness)
        cv2.line(img, (width // 2, margin), (width // 2, height - margin), line_color, line_thickness)
        cv2.circle(img, (width // 2, height // 2), int(h_field * 0.15), line_color, line_thickness)
        cv2.circle(img, (width // 2, height // 2), 4, line_color, -1)
        
        box_h = int(h_field * 0.6); box_w = int(w_field * 0.16)
        cv2.rectangle(img, (margin, height//2 - box_h//2), (margin + box_w, height//2 + box_h//2), line_color, line_thickness)
        cv2.rectangle(img, (width - margin - box_w, height//2 - box_h//2), (width - margin, height//2 + box_h//2), line_color, line_thickness)
        return img

    def create_feature_mask(self, frame_h, frame_w):
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        top_y = int(frame_h * FEATURE_CROP_TOP_PERCENT)
        mask[top_y:frame_h, 0:frame_w] = 255
        return mask

    def get_histogram(self, img):
        if img.size == 0: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float64)

    def get_torso_crop(self, frame, box):
        x1, y1, x2, y2 = map(safe_int, box)
        tx1 = int(x1 + (x2-x1)*0.20); tx2 = int(x2 - (x2-x1)*0.20)
        ty1 = int(y1 + (y2-y1)*0.15); ty2 = int(y1 + (y2-y1)*0.60)
        tx1=max(0,tx1); ty1=max(0,ty1); tx2=min(frame.shape[1],tx2); ty2=min(frame.shape[0],ty2)
        return frame[ty1:ty2, tx1:tx2]

    # --- STEP 1: ANALYSIS ---
    def pass_1_analysis(self):
        print("\n--- Starting Analysis ---")
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if MAX_FRAMES is not None: total_frames = min(total_frames, MAX_FRAMES)
        
        frame_idx = 0
        with tqdm(total=total_frames, desc="1/4 Analysis", unit="frame") as pbar:
            while True:
                if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES: break
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                
                # A. Players
                p_results = self.model_players.track(frame, persist=True, verbose=False, classes=[0], conf=0.25)
                if p_results[0].boxes.id is not None:
                    boxes = p_results[0].boxes.xyxy.cpu().numpy()
                    ids = p_results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        if track_id not in self.tracks: self.tracks[track_id] = TrackEntity(track_id)
                        track = self.tracks[track_id]
                        crop = self.get_torso_crop(frame, box)
                        hist = self.get_histogram(crop)
                        if hist is not None: track.add_observation(frame_idx, box, hist)

                # B. Ball
                b_results = self.model_ball.infer(frame)
                predictions = []
                try:
                    if isinstance(b_results, list): predictions = b_results[0].predictions
                    else: predictions = b_results.predictions
                except: pass

                best_pred = None; best_conf = 0
                for pred in predictions:
                    if pred.confidence > 0.4 and pred.confidence > best_conf:
                        best_conf = pred.confidence; best_pred = pred
                
                if best_pred:
                    self.ball_raw.append({'frame': frame_idx, 'x': best_pred.x, 'y': best_pred.y, 'w': best_pred.width, 'h': best_pred.height})
                else:
                    self.ball_raw.append({'frame': frame_idx, 'x': 0, 'y': 0, 'w': 0, 'h': 0})
                
                pbar.update(1)
        cap.release()

    # --- STEP 2: OPTIMIZATION ---
    def optimize_tracks(self):
        print()
        # Players
        for t_id, track in tqdm(self.tracks.items(), desc="2/4 Interpolation", unit="track"):
            track.sort_by_frame()
            if len(track.frames) < 2: continue
            new_frames, new_bboxes = [], []
            for i in range(len(track.frames) - 1):
                f_cur, f_nxt = track.frames[i], track.frames[i+1]
                b_cur, b_nxt = track.bboxes[i], track.bboxes[i+1]
                new_frames.append(f_cur); new_bboxes.append(b_cur)
                gap = f_nxt - f_cur
                if 1 < gap < 20:
                    for step in range(1, gap):
                        alpha = step / gap
                        interp_box = (1 - alpha) * b_cur + alpha * b_nxt
                        new_frames.append(f_cur + step); new_bboxes.append(interp_box)
            new_frames.append(track.frames[-1]); new_bboxes.append(track.bboxes[-1])
            track.frames = new_frames; track.bboxes = new_bboxes

        # Ball
        if self.ball_raw:
            df = pd.DataFrame(self.ball_raw)
            # Safe float conversion
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Smart Interpolation: Only interpolate if gaps are small
            # (If ball disappears for 100 frames, don't draw a line across the field)
            mask = df['w'] > 0
            df.loc[~mask, ['x', 'y', 'w', 'h']] = np.nan
            df = df.interpolate(method='linear', limit_direction='both', limit=30) # Limit 30 frames
            df.fillna(0, inplace=True)
            self.ball_clean = df.to_dict('records')

        # Teams
        all_features = []
        for track in self.tracks.values():
            if len(track.features) > 0: all_features.extend(track.features)
        
        if len(all_features) > 10:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.array(all_features))
            center_0 = kmeans.cluster_centers_[0].reshape(16, 16)
            center_1 = kmeans.cluster_centers_[1].reshape(16, 16)
            score_0 = np.sum(center_0[4:8, 5:])
            score_1 = np.sum(center_1[4:8, 5:])
            green_cluster = 0 if score_0 > score_1 else 1
            
            for t_id, track in self.tracks.items():
                if not track.features: continue
                votes = kmeans.predict(np.array(track.features))
                track.assigned_team = 1 if np.mean(votes == green_cluster) > 0.5 else 2

    # --- STEP 3: RENDERING (CLEAN MODE) ---
    def pass_2_render(self):
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if MAX_FRAMES is not None: total_frames = min(total_frames, MAX_FRAMES)
        
        ret, old_frame = cap.read()
        vid_h, vid_w = old_frame.shape[:2]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        minimap_w, minimap_h = MINIMAP_WIDTH, MINIMAP_HEIGHT
        base_field = self.create_soccer_field(minimap_w, minimap_h)
        
        # Output Setup
        scale_factor = minimap_h / vid_h
        final_w = int(vid_w * scale_factor) + minimap_w
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (final_w, minimap_h))
        
        feature_roi_mask = self.create_feature_mask(vid_h, vid_w)
        feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=15, blockSize=7)
        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=feature_roi_mask, **feature_params)
        
        cam_angle, cam_tilt, cam_zoom = 0.0, 0.0, 1.0
        smooth_cam_angle, smooth_cam_tilt, smooth_cam_zoom = 0.0, 0.0, 1.0
        pivot_x = minimap_w // 2; pivot_y = minimap_h + PIVOT_OFFSET_Y
        
        frame_map = defaultdict(list)
        for t_id, track in self.tracks.items():
            for f_idx, bbox in zip(track.frames, track.bboxes): frame_map[f_idx].append((t_id, bbox))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        print()
        with tqdm(total=total_frames, desc="4/4 Rendering", unit="frame") as pbar:
            while True:
                if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES: break
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # --- 1. LOCALIZATION ---
                d_angle, d_tilt, d_zoom = 0, 0, 1.0
                if p0 is not None and len(p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    if p1 is not None:
                        good_new = p1[st == 1]; good_old = p0[st == 1]
                        if len(good_new) > 4:
                            H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
                            if H is not None and mask is not None and np.sum(mask) > 4:
                                inlier_mask = mask.ravel() == 1
                                inlier_old = good_old[inlier_mask]; inlier_new = good_new[inlier_mask]
                                m, _ = cv2.estimateAffinePartial2D(inlier_old, inlier_new)
                                if m is not None:
                                    # Dampening: Ignore extremely small movements (jitter)
                                    raw_da = m[0, 2] * ROTATION_SENSITIVITY
                                    raw_dt = -m[1, 2] * TILT_SENSITIVITY
                                    
                                    if abs(raw_da) > 0.0001: d_angle = raw_da
                                    if abs(raw_dt) > 0.1: d_tilt = raw_dt
                                    
                                    s_x = np.sqrt(m[0,0]**2 + m[0,1]**2); s_y = np.sqrt(m[1,0]**2 + m[1,1]**2)
                                    d_zoom = (s_x + s_y) / 2
                                    
                                    if len(inlier_new) < FEATURE_REFRESH_THRESHOLD:
                                         p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                                    else: p0 = inlier_new.reshape(-1, 1, 2)
                                else: p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                            else: p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                        else: p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                    else: p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                else: p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)

                cam_angle += d_angle; cam_tilt += d_tilt; cam_zoom *= d_zoom
                cam_zoom = max(0.5, min(cam_zoom, 3.0)); cam_tilt = max(-100, min(cam_tilt, 100))
                
                # Smoothing
                smooth_cam_angle = smooth_cam_angle * 0.90 + cam_angle * 0.10
                smooth_cam_tilt = smooth_cam_tilt * 0.90 + cam_tilt * 0.10
                smooth_cam_zoom = smooth_cam_zoom * 0.90 + cam_zoom * 0.10

                # --- 2. MAPPING ---
                current_fov = BASE_FOV_ANGLE / smooth_cam_zoom
                y_far = max(0, safe_int(40 + smooth_cam_tilt))
                y_near = min(minimap_h, safe_int(minimap_h - 40 + smooth_cam_tilt))
                
                theta_left = smooth_cam_angle + (current_fov / 2); theta_right = smooth_cam_angle - (current_fov / 2)
                dy_far = y_far - pivot_y; dy_near = y_near - pivot_y
                
                x_far_left = safe_int(pivot_x + dy_far * np.tan(theta_left))
                x_far_right = safe_int(pivot_x + dy_far * np.tan(theta_right))
                x_near_left = safe_int(pivot_x + dy_near * np.tan(theta_left))
                x_near_right = safe_int(pivot_x + dy_near * np.tan(theta_right))
                
                src_pts = np.float32([[0, safe_int(vid_h * FEATURE_CROP_TOP_PERCENT)], [vid_w, safe_int(vid_h * FEATURE_CROP_TOP_PERCENT)], [vid_w, vid_h], [0, vid_h]])
                dst_pts = np.float32([(x_far_left, y_far), (x_far_right, y_far), (x_near_right, y_near), (x_near_left, y_near)])
                M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # --- 3. DRAW CLEAN MAP (No Video Warp) ---
                current_minimap = base_field.copy()
                fov_poly = np.array([(x_near_left, y_near), (x_far_left, y_far), (x_far_right, y_far), (x_near_right, y_near)], np.int32)
                
                # Draw View Cone (Transparent)
                overlay = current_minimap.copy()
                cv2.fillPoly(overlay, [fov_poly], (60, 70, 60)) # Slightly lighter green
                cv2.addWeighted(overlay, 0.3, current_minimap, 0.7, 0, current_minimap)
                cv2.polylines(current_minimap, [fov_poly], True, COLOR_FOV_LINES, 1)

                # --- 4. DRAW ENTITIES ---
                points_to_transform = []; colors_for_points = []
                
                for t_id, bbox in frame_map[frame_idx]:
                    track = self.tracks[t_id]
                    x1, y1, x2, y2 = map(safe_int, bbox)
                    color = COLOR_TEAM_1 if track.assigned_team == 1 else COLOR_TEAM_2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    points_to_transform.append([(x1 + x2) / 2, y2])
                    colors_for_points.append(color)

                if frame_idx <= len(self.ball_clean):
                    b_data = self.ball_clean[frame_idx - 1]
                    if b_data['w'] > 0: # Only draw if ball exists
                        bx, by = safe_int(b_data['x']), safe_int(b_data['y'])
                        # Safety check: Is ball actually in frame?
                        if bx > 0 and by > 0:
                            cv2.circle(frame, (bx, by), 5, COLOR_BALL_DOT, -1)
                            points_to_transform.append([bx, by])
                            colors_for_points.append(COLOR_BALL_DOT)

                if len(points_to_transform) > 0:
                    pts_array = np.array([points_to_transform], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(pts_array, M_persp)[0]
                    for i, point in enumerate(transformed):
                        mx, my = safe_int(point[0]), safe_int(point[1])
                        # Bound Check
                        if 0 <= mx < minimap_w and 0 <= my < minimap_h:
                            cv2.circle(current_minimap, (mx, my), 5, colors_for_points[i], -1)

                # Combine
                resized_frame = cv2.resize(frame, (int(vid_w * scale_factor), minimap_h))
                combined = np.hstack((resized_frame, current_minimap))
                out.write(combined)
                
                old_gray = frame_gray.copy()
                pbar.update(1)

        cap.release()
        out.release()
        print(f"\nProcessing Complete! Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    tracker = OfflineTracker()
    tracker.pass_1_analysis()
    tracker.optimize_tracks()
    tracker.pass_2_render()