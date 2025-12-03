from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import sys
from tqdm import tqdm
import os
from inference import get_model


# --- CONFIGURATION ---
INPUT_PATH = '/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/video_clip.mp4'
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/tracking_roboflow.mp4"

# *** MODEL SETTINGS ***
PLAYER_MODEL = "yolov8x.pt"
# Make sure your API Key is set in your environment variables, or pass api_key="XYZ" below
ROBOFLOW_MODEL_ID = "soccer-ball-detection-2/3" 

# *** LIMIT PROCESSING ***
MAX_FRAMES = None  # Set to None for full video

# Visuals
COLOR_TEAM_1 = (0, 200, 0)      # Green
COLOR_TEAM_2 = (220, 220, 220)  # White
COLOR_BALL_BOX = (0, 255, 0)    
COLOR_BALL_DOT = (0, 0, 255)    
COLOR_TRAIL = (0, 255, 255)     

class TrackEntity:
    """Stores history for PLAYERS only."""
    def __init__(self, track_id):
        self.id = track_id
        self.frames = []
        self.bboxes = []
        self.features = []
        self.assigned_team = None

    def add_observation(self, frame_idx, bbox, feature=None):
        self.frames.append(frame_idx)
        self.bboxes.append(bbox)
        if feature is not None:
            self.features.append(feature)

    def sort_by_frame(self):
        sorted_indices = np.argsort(self.frames)
        self.frames = [self.frames[i] for i in sorted_indices]
        self.bboxes = [self.bboxes[i] for i in sorted_indices]
        if self.features:
            self.features = [self.features[i] for i in sorted_indices if i < len(self.features)]

class OfflineTracker:
    def __init__(self):
        print("--- Loading Models ---")
        
        # 1. Player Model (Standard YOLO via Ultralytics)
        print(f"Loading Player Model: {PLAYER_MODEL}...")
        self.model_players = YOLO(PLAYER_MODEL)
        
        # 2. Ball Model (Roboflow Inference API)
        print(f"Loading Ball Model via API: {ROBOFLOW_MODEL_ID}...")
        try:
            # NOTE: If you haven't set ROBOFLOW_API_KEY in env, add api_key="YOUR_KEY" inside get_model()
            self.model_ball = get_model(model_id=ROBOFLOW_MODEL_ID)
        except Exception as e:
            print(f"Error loading Roboflow model: {e}")
            print("Did you set your ROBOFLOW_API_KEY?")
            sys.exit()
        
        # Storage
        self.tracks = {}        
        self.ball_raw = []      
        self.ball_clean = []    
        
    def get_histogram(self, img):
        if img.size == 0: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float64)

    def get_torso_crop(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2-x1, y2-y1
        tx1 = int(x1 + w*0.20)
        tx2 = int(x2 - w*0.20)
        ty1 = int(y1 + h*0.15)
        ty2 = int(y1 + h*0.60)
        tx1=max(0,tx1); ty1=max(0,ty1); tx2=min(frame.shape[1],tx2); ty2=min(frame.shape[0],ty2)
        return frame[ty1:ty2, tx1:tx2]

    # --- STEP 1: GATHER DATA ---
    def pass_1_analysis(self):
        print("\n--- Starting Dual-Model Analysis ---")
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        if MAX_FRAMES is not None:
            total_frames = min(total_frames, MAX_FRAMES)
            print(f"DEBUG: Processing first {MAX_FRAMES} frames only.")
        
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="1/4 Analysis (Hybrid)", unit="frame") as pbar:
            while True:
                if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES: break

                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                
                # --- PASS A: PLAYERS (Ultralytics) ---
                p_results = self.model_players.track(frame, persist=True, verbose=False, classes=[0], conf=0.25)
                
                if p_results[0].boxes.id is not None:
                    boxes = p_results[0].boxes.xyxy.cpu().numpy()
                    ids = p_results[0].boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id in zip(boxes, ids):
                        if track_id not in self.tracks:
                            self.tracks[track_id] = TrackEntity(track_id)
                        
                        track = self.tracks[track_id]
                        crop = self.get_torso_crop(frame, box)
                        hist = self.get_histogram(crop)
                        if hist is not None:
                            track.add_observation(frame_idx, box, hist)

                # --- PASS B: BALL (Roboflow Inference) ---
                # Roboflow inference returns a list of prediction objects
                b_results = self.model_ball.infer(frame)
                
                # Extract predictions (Note: syntax differs from Ultralytics)
                # Usually b_results[0].predictions if batch, or just .predictions
                try:
                    # Depending on library version, might be b_results[0].predictions or just b_results.predictions
                    # We assume single image inference
                    predictions = b_results[0].predictions
                except:
                    predictions = []

                best_conf = 0
                best_pred = None

                # Find best ball detection
                for pred in predictions:
                    # Roboflow usually returns confidence as float 0.0-1.0
                    if pred.confidence > 0.5:
                        if pred.confidence > best_conf:
                            best_conf = pred.confidence
                            best_pred = pred
                
                if best_pred:
                    # Roboflow returns Center X, Center Y, Width, Height
                    self.ball_raw.append({
                        'frame': frame_idx, 
                        'x': best_pred.x, 
                        'y': best_pred.y, 
                        'w': best_pred.width, 
                        'h': best_pred.height
                    })
                else:
                    self.ball_raw.append({'frame': frame_idx, 'x': 0, 'y': 0, 'w': 0, 'h': 0})
                
                pbar.update(1)
        
        cap.release()

    # --- STEP 2: OPTIMIZATION ---
    def optimize_tracks(self):
        # A. PLAYERS
        print()
        for t_id, track in tqdm(self.tracks.items(), desc="2/4 Player Interp", unit="track"):
            track.sort_by_frame()
            if len(track.frames) < 2: continue
            
            new_frames, new_bboxes = [], []
            for i in range(len(track.frames) - 1):
                f_current, f_next = track.frames[i], track.frames[i+1]
                b_current, b_next = track.bboxes[i], track.bboxes[i+1]
                new_frames.append(f_current)
                new_bboxes.append(b_current)
                
                gap = f_next - f_current
                if 1 < gap < 20: 
                    for step in range(1, gap):
                        alpha = step / gap
                        interp_box = (1 - alpha) * b_current + alpha * b_next
                        new_frames.append(f_current + step)
                        new_bboxes.append(interp_box)
            new_frames.append(track.frames[-1])
            new_bboxes.append(track.bboxes[-1])
            track.frames = new_frames
            track.bboxes = new_bboxes

        # B. BALL (Pandas)
        print("    -> Interpolating Ball...")
        if not self.ball_raw:
            self.ball_clean = []
        else:
            df = pd.DataFrame(self.ball_raw)
            df.replace(0, np.nan, inplace=True) 
            df = df.interpolate(method='linear', limit_direction='both') 
            df.fillna(0, inplace=True) 
            self.ball_clean = df.to_dict('records') 

        # C. TEAM CLUSTERING
        all_features = []
        for track in self.tracks.values():
            if len(track.features) > 0: all_features.extend(track.features)
        
        if len(all_features) < 10:
            print("Not enough people found for clustering.")
            return

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(np.array(all_features))
        
        center_0 = kmeans.cluster_centers_[0].reshape(16, 16)
        center_1 = kmeans.cluster_centers_[1].reshape(16, 16)
        score_0 = np.sum(center_0[4:8, 5:])
        score_1 = np.sum(center_1[4:8, 5:])
        green_cluster = 0 if score_0 > score_1 else 1
        
        for t_id, track in tqdm(self.tracks.items(), desc="3/4 Team Assign  ", unit="track"):
            if not track.features: continue
            votes = kmeans.predict(np.array(track.features))
            if np.mean(votes == green_cluster) > 0.5:
                track.assigned_team = 1
            else:
                track.assigned_team = 2

    # --- STEP 3: RENDERING ---
    def pass_2_render(self):
        cap = cv2.VideoCapture(INPUT_PATH)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if MAX_FRAMES is not None: total_frames = min(total_frames, MAX_FRAMES)
        
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        frame_map = defaultdict(list)
        for t_id, track in self.tracks.items():
            for f_idx, bbox in zip(track.frames, track.bboxes):
                frame_map[f_idx].append((t_id, bbox))
                
        trail_buffer = deque(maxlen=30) 

        print()
        with tqdm(total=total_frames, desc="4/4 Rendering    ", unit="frame") as pbar:
            frame_idx = 0
            while True:
                if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES: break
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                
                # DRAW PLAYERS
                active_tracks = frame_map[frame_idx]
                for t_id, bbox in active_tracks:
                    track = self.tracks[t_id]
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    if track.assigned_team == 1:
                        color = COLOR_TEAM_1
                        label = "Team 1"
                    elif track.assigned_team == 2:
                        color = COLOR_TEAM_2
                        label = "Team 2"
                    else:
                        color = (100, 100, 100)
                        label = "Unk"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{t_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # DRAW BALL
                if frame_idx <= len(self.ball_clean):
                    ball_data = self.ball_clean[frame_idx - 1] 
                    
                    if ball_data['w'] > 0: 
                        cx, cy = int(ball_data['x']), int(ball_data['y'])
                        w, h = int(ball_data['w']), int(ball_data['h'])
                        
                        top_left = (int(cx - w/2), int(cy - h/2))
                        bottom_right = (int(cx + w/2), int(cy + h/2))
                        
                        trail_buffer.append((cx, cy))
                        
                        if len(trail_buffer) > 1:
                            cv2.polylines(frame, [np.array(trail_buffer, dtype=np.int32)], False, COLOR_TRAIL, 2)

                        cv2.rectangle(frame, top_left, bottom_right, COLOR_BALL_BOX, 2)
                        cv2.circle(frame, (cx, cy), 5, COLOR_BALL_DOT, -1)

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()
        print(f"\nProcessing Complete! Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    tracker = OfflineTracker()
    tracker.pass_1_analysis()
    tracker.optimize_tracks()
    tracker.pass_2_render()