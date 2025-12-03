from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
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
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/tracking_dashboard_v2.mp4"

# *** MODEL SETTINGS ***
PLAYER_MODEL = "yolov8x.pt"
ROBOFLOW_MODEL_ID = "soccer-ball-detection-2/3" 

MAX_FRAMES = None

# --- PHYSICS ---
BASE_ROTATION_SENSITIVITY = 0.00025
BASE_TILT_SENSITIVITY = 0.0
BASE_FOV_ANGLE = np.deg2rad(30) 
REFERENCE_PLAYER_HEIGHT_PX = 60.0 

# --- VISUALS ---
MINIMAP_WIDTH = 600   
MINIMAP_HEIGHT = 400
PIVOT_OFFSET_Y = 600 

COLOR_TEAM_1 = (0, 200, 0)      # Green
COLOR_TEAM_2 = (220, 220, 220)  # White/Grey
COLOR_BALL_DOT = (0, 0, 255)    # Red
COLOR_TEXT = (255, 255, 255)

POSSESSION_PROXIMITY_THRESHOLD = 35

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
        try:
            self.model_ball = get_model(model_id=ROBOFLOW_MODEL_ID)
        except Exception as e:
            print(f"CRITICAL ERROR loading Roboflow model: {e}")
            sys.exit()
        
        self.tracks = {}        
        self.ball_raw = []      
        self.ball_clean = []    

    def create_soccer_field(self, width, height):
        field_color = (40, 45, 40); line_color = (100, 100, 100); line_thickness = 2
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = field_color
        margin = 30
        w_field = width - 2 * margin; h_field = height - 2 * margin
        
        # Field Outline
        cv2.rectangle(img, (margin, margin), (width - margin, height - margin), line_color, line_thickness)
        cv2.line(img, (width // 2, margin), (width // 2, height - margin), line_color, line_thickness)
        cv2.circle(img, (width // 2, height // 2), int(h_field * 0.15), line_color, line_thickness)
        
        # Action Area Dividers (Dotted feel)
        third_w = width // 3
        cv2.line(img, (third_w, margin), (third_w, height - margin), (60, 60, 60), 1)
        cv2.line(img, (third_w * 2, margin), (third_w * 2, height - margin), (60, 60, 60), 1)

        box_h = int(h_field * 0.6); box_w = int(w_field * 0.16)
        cv2.rectangle(img, (margin, height//2 - box_h//2), (margin + box_w, height//2 + box_h//2), line_color, line_thickness)
        cv2.rectangle(img, (width - margin - box_w, height//2 - box_h//2), (width - margin, height//2 + box_h//2), line_color, line_thickness)
        return img

    def create_feature_mask(self, frame_h, frame_w):
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        top_y = int(frame_h * 0.40)
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

    # --- ANALYSIS & OPTIMIZATION (Unchanged) ---
    def pass_1_analysis(self):
        print("\n--- Starting Analysis ---")
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if MAX_FRAMES: total_frames = min(total_frames, MAX_FRAMES)
        
        frame_idx = 0
        with tqdm(total=total_frames, desc="1/3 Analysis", unit="frame") as pbar:
            while True:
                if MAX_FRAMES and frame_idx >= MAX_FRAMES: break
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                
                # Players
                p_results = self.model_players.track(frame, persist=True, verbose=False, classes=[0], conf=0.25)
                if p_results[0].boxes.id is not None:
                    boxes = p_results[0].boxes.xyxy.cpu().numpy()
                    ids = p_results[0].boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        if track_id not in self.tracks: self.tracks[track_id] = TrackEntity(track_id)
                        crop = self.get_torso_crop(frame, box)
                        hist = self.get_histogram(crop)
                        if hist is not None: self.tracks[track_id].add_observation(frame_idx, box, hist)

                # Ball
                b_results = self.model_ball.infer(frame)
                predictions = []
                try:
                    if isinstance(b_results, list): predictions = b_results[0].predictions
                    else: predictions = b_results.predictions
                except: pass
                
                best = None; best_conf = 0
                for p in predictions:
                    if p.confidence > 0.4 and p.confidence > best_conf: best_conf = p.confidence; best = p
                
                if best: self.ball_raw.append({'frame': frame_idx, 'x': best.x, 'y': best.y, 'w': best.width, 'h': best.height})
                else: self.ball_raw.append({'frame': frame_idx, 'x': 0, 'y': 0, 'w': 0, 'h': 0})
                pbar.update(1)
        cap.release()

    def optimize_tracks(self):
        print("\n--- Optimizing Tracks ---")
        # Interpolate Players
        for track in self.tracks.values():
            track.sort_by_frame()
            if len(track.frames) < 2: continue
            new_frames, new_bboxes = [], []
            for i in range(len(track.frames) - 1):
                f1, f2 = track.frames[i], track.frames[i+1]
                b1, b2 = track.bboxes[i], track.bboxes[i+1]
                new_frames.append(f1); new_bboxes.append(b1)
                if 1 < (f2 - f1) < 20:
                    for s in range(1, f2-f1):
                        alpha = s / (f2 - f1)
                        new_frames.append(f1 + s); new_bboxes.append((1-alpha)*b1 + alpha*b2)
            new_frames.append(track.frames[-1]); new_bboxes.append(track.bboxes[-1])
            track.frames = new_frames; track.bboxes = new_bboxes

        # Interpolate Ball
        if self.ball_raw:
            df = pd.DataFrame(self.ball_raw).replace(0, np.nan).interpolate(limit=30).fillna(0)
            self.ball_clean = df.to_dict('records')

        # Cluster Teams
        features = [f for t in self.tracks.values() for f in t.features]
        if len(features) > 10:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(features)
            # Identify which cluster is which based on simple heuristic (usually varies, assume 0 is team A)
            # (In a real app, you'd compare to field color or jersey color)
            for track in self.tracks.values():
                if track.features:
                    votes = kmeans.predict(np.array(track.features))
                    track.assigned_team = 1 if np.mean(votes == 0) > 0.5 else 2

    # --- RENDER WITH DASHBOARD ---
    def pass_2_render(self):
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if MAX_FRAMES: total_frames = min(total_frames, MAX_FRAMES)
        
        ret, frame = cap.read()
        vid_h, vid_w = frame.shape[:2]
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Output Setup
        minimap_w, minimap_h = MINIMAP_WIDTH, MINIMAP_HEIGHT
        base_field = self.create_soccer_field(minimap_w, minimap_h)
        scale = minimap_h / vid_h
        final_w = int(vid_w * scale) + minimap_w
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (final_w, minimap_h))
        
        # Tracking & Cam Vars
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=self.create_feature_mask(vid_h, vid_w), maxCorners=300, qualityLevel=0.01, minDistance=15)
        cam_angle, cam_tilt = 0.0, 0.0
        smooth_angle, smooth_tilt, cur_zoom = 0.0, 0.0, 1.0
        pivot_x, pivot_y = minimap_w // 2, minimap_h + PIVOT_OFFSET_Y

        # Pre-process lookups
        frame_map = defaultdict(list)
        for t_id, t in self.tracks.items():
            for i, f in enumerate(t.frames): frame_map[f].append((t_id, t.bboxes[i]))

        # Metrics Containers
        stats_poss = {1: 0, 2: 0}
        stats_area = [0, 0, 0]

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="3/3 Rendering", unit="frame") as pbar:
            while True:
                if MAX_FRAMES and frame_idx >= MAX_FRAMES: break
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 1. Physics & Zoom
                heights = [b[3]-b[1] for _, b in frame_map[frame_idx] if 20 < b[3]-b[1] < 400]
                if len(heights) > 2:
                    tgt_zoom = np.median(heights) / REFERENCE_PLAYER_HEIGHT_PX
                    cur_zoom = cur_zoom * 0.9 + tgt_zoom * 0.1
                
                sens_rot = BASE_ROTATION_SENSITIVITY / cur_zoom
                sens_tilt = BASE_TILT_SENSITIVITY / cur_zoom

                # 2. Optical Flow
                if p0 is not None and len(p0) > 4:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, winSize=(21,21), maxLevel=3)
                    if p1 is not None:
                        good_new = p1[st==1]; good_old = p0[st==1]
                        if len(good_new) > 4:
                            H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                            if H is not None:
                                m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                                if m is not None:
                                    da = m[0,2] * sens_rot
                                    dt = -m[1,2] * sens_tilt
                                    cam_angle += da if abs(da) > 1e-5 else 0
                                    cam_tilt = max(-100, min(100, cam_tilt + (dt if abs(dt) > 0.1 else 0)))
                                    p0 = good_new.reshape(-1, 1, 2)
                                else: p0 = cv2.goodFeaturesToTrack(gray, mask=self.create_feature_mask(vid_h, vid_w), maxCorners=300, qualityLevel=0.01, minDistance=15)
                            else: p0 = cv2.goodFeaturesToTrack(gray, mask=self.create_feature_mask(vid_h, vid_w), maxCorners=300, qualityLevel=0.01, minDistance=15)
                        else: p0 = cv2.goodFeaturesToTrack(gray, mask=self.create_feature_mask(vid_h, vid_w), maxCorners=300, qualityLevel=0.01, minDistance=15)
                    else: p0 = cv2.goodFeaturesToTrack(gray, mask=self.create_feature_mask(vid_h, vid_w), maxCorners=300, qualityLevel=0.01, minDistance=15)

                smooth_angle = smooth_angle * 0.9 + cam_angle * 0.1
                smooth_tilt = smooth_tilt * 0.9 + cam_tilt * 0.1

                # 3. Perspective
                fov = BASE_FOV_ANGLE / cur_zoom
                y_f = max(0, safe_int(40 + smooth_tilt)); y_n = min(minimap_h, safe_int(minimap_h - 40 + smooth_tilt))
                dy_f = y_f - pivot_y; dy_n = y_n - pivot_y
                tl = smooth_angle + fov/2; tr = smooth_angle - fov/2
                pts_src = np.float32([[0, vid_h*0.4], [vid_w, vid_h*0.4], [vid_w, vid_h], [0, vid_h]])
                pts_dst = np.float32([
                    (pivot_x + dy_f*np.tan(tl), y_f), (pivot_x + dy_f*np.tan(tr), y_f),
                    (pivot_x + dy_n*np.tan(tr), y_n), (pivot_x + dy_n*np.tan(tl), y_n)
                ])
                M = cv2.getPerspectiveTransform(pts_src, pts_dst)

                # 4. Map Logic
                curr_map = base_field.copy()
                
                # Draw FOV on Map
                fov_poly = pts_dst.astype(np.int32)
                overlay = curr_map.copy()
                cv2.fillPoly(overlay, [fov_poly], (60, 70, 60))
                cv2.addWeighted(overlay, 0.3, curr_map, 0.7, 0, curr_map)
                cv2.polylines(curr_map, [fov_poly], True, (80,80,80), 1)

                # Points for Transformation
                pts_track = []; identities = [] # ('p', team) or ('b', 0)
                
                # Players
                for t_id, box in frame_map[frame_idx]:
                    x1, y1, x2, y2 = map(safe_int, box)
                    tm = self.tracks[t_id].assigned_team
                    color = COLOR_TEAM_1 if tm == 1 else COLOR_TEAM_2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    pts_track.append([(x1+x2)/2, y2])
                    identities.append(('p', tm))

                # Ball
                b_loc = None
                if frame_idx <= len(self.ball_clean):
                    bd = self.ball_clean[frame_idx-1]
                    if bd['w'] > 0:
                        bx, by = safe_int(bd['x']), safe_int(bd['y'])
                        cv2.circle(frame, (bx, by), 6, COLOR_BALL_DOT, -1)
                        pts_track.append([bx, by])
                        identities.append(('b', 0))

                # Transform & Draw
                map_players = []
                if pts_track:
                    mapped = cv2.perspectiveTransform(np.array([pts_track], dtype=np.float32), M)[0]
                    for i, (mx, my) in enumerate(mapped):
                        mx, my = safe_int(mx), safe_int(my)
                        if 0 <= mx < minimap_w and 0 <= my < minimap_h:
                            type_, team = identities[i]
                            if type_ == 'p':
                                c = COLOR_TEAM_1 if team == 1 else COLOR_TEAM_2
                                cv2.circle(curr_map, (mx, my), 5, c, -1)
                                map_players.append({'pos': np.array([mx, my]), 'team': team})
                            else:
                                cv2.circle(curr_map, (mx, my), 5, COLOR_BALL_DOT, -1)
                                b_loc = np.array([mx, my])
                
                # Metrics Calculation
                if b_loc is not None:
                    # Zones
                    z = 0 if b_loc[0] < minimap_w/3 else (1 if b_loc[0] < 2*minimap_w/3 else 2)
                    stats_area[z] += 1
                    # Possession
                    closest, c_dist = None, 9999
                    for p in map_players:
                        d = np.linalg.norm(p['pos'] - b_loc)
                        if d < c_dist: c_dist = d; closest = p['team']
                    if c_dist < POSSESSION_PROXIMITY_THRESHOLD: stats_poss[closest] += 1

                # --- VISUAL DASHBOARD (ON MINIMAP) ---
                # 1. Background Header
                cv2.rectangle(curr_map, (0, 0), (minimap_w, 65), (20, 20, 20), -1)
                
                # 2. Possession Bar
                tot_poss = stats_poss[1] + stats_poss[2]
                pct1 = stats_poss[1] / tot_poss if tot_poss > 0 else 0.5
                
                bar_x, bar_y, bar_w, bar_h = 100, 25, 400, 12
                # Text Label
                cv2.putText(curr_map, "POSSESSION", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
                # Green Part
                w1 = int(bar_w * pct1)
                cv2.rectangle(curr_map, (bar_x, bar_y), (bar_x + w1, bar_y + bar_h), COLOR_TEAM_1, -1)
                # White Part
                cv2.rectangle(curr_map, (bar_x + w1, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_TEAM_2, -1)
                # Percentages
                cv2.putText(curr_map, f"{int(pct1*100)}%", (bar_x - 35, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEAM_1, 1)
                cv2.putText(curr_map, f"{100 - int(pct1*100)}%", (bar_x + bar_w + 5, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEAM_2, 1)

                # 3. Action Areas (Simple Text Row)
                tot_area = sum(stats_area)
                if tot_area > 0:
                    ap = [int(x/tot_area*100) for x in stats_area]
                    txt = f"ACTION ZONES | Left: {ap[0]}%  -  Mid: {ap[1]}%  -  Right: {ap[2]}%"
                    cv2.putText(curr_map, txt, (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                # Combine and Save
                resized_frame = cv2.resize(frame, (int(vid_w * scale), minimap_h))
                out.write(np.hstack((resized_frame, curr_map)))
                old_gray = gray.copy()
                pbar.update(1)

        cap.release(); out.release()
        print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"Error: File not found at {INPUT_PATH}"); sys.exit()
    
    tracker = OfflineTracker()
    tracker.pass_1_analysis()
    tracker.optimize_tracks()
    tracker.pass_2_render()