from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import sys

# --- CONFIGURATION ---
INPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/YOLO_v5_offline.mp4"
MODEL_SIZE = "m" # options: s, n

# Visuals
COLOR_TEAM_1 = (0, 200, 0)      # Green
COLOR_TEAM_2 = (220, 220, 220)  # White
COLOR_BALL = (0, 165, 255)      # Orange
COLOR_TRAIL = (0, 255, 255)     # Yellow



class TrackEntity:
    """Stores the entire history of a single Track ID."""
    def __init__(self, track_id):
        self.id = track_id
        self.frames = []        # List of frame indices
        self.bboxes = []        # List of [x1, y1, x2, y2]
        self.features = []      # List of color histograms
        self.assigned_team = None
        self.is_ball = False

    def add_observation(self, frame_idx, bbox, feature=None):
        self.frames.append(frame_idx)
        self.bboxes.append(bbox)
        if feature is not None:
            self.features.append(feature)

    def sort_by_frame(self):
        # Ensure data is chronological (YOLO usually does this, but safety first)
        sorted_indices = np.argsort(self.frames)
        self.frames = [self.frames[i] for i in sorted_indices]
        self.bboxes = [self.bboxes[i] for i in sorted_indices]
        # Features might be empty for ball, so check
        if self.features:
            self.features = [self.features[i] for i in sorted_indices if i < len(self.features)]

class OfflineTracker:
    def __init__(self):
        self.model = YOLO(f"yolov8{MODEL_SIZE}.pt")
        self.tracks = {}  # Dict {track_id: TrackEntity}
        self.team_colors = {}
        self.team_names = {}
        
    def get_histogram(self, img):
        """Extract color fingerprint (float64 for sklearn)."""
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
        print("--- PASS 1: Analysis (Reading Video & Extracting Features) ---")
        cap = cv2.VideoCapture(INPUT_PATH)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            # Progress bar
            if frame_idx % 50 == 0:
                print(f"Analyzing Frame {frame_idx}/{total_frames}...")

            # Run Tracking
            results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, classes=[0, 32])
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls in zip(boxes, ids, clss):
                    # Create TrackEntity if new
                    if track_id not in self.tracks:
                        self.tracks[track_id] = TrackEntity(track_id)
                    
                    track = self.tracks[track_id]
                    
                    if cls == 32: # Ball
                        track.is_ball = True
                        track.add_observation(frame_idx, box, None)
                    else: # Person
                        # Extract feature
                        crop = self.get_torso_crop(frame, box)
                        hist = self.get_histogram(crop)
                        if hist is not None:
                            track.add_observation(frame_idx, box, hist)
        
        cap.release()
        print("Analysis Complete. Optimizing Tracks...")

    # --- STEP 2: GLOBAL OPTIMIZATION ---
    def optimize_tracks(self):
        # 1. INTERPOLATION (Fill gaps)
        print("Interpolating missing frames...")
        for t_id, track in self.tracks.items():
            track.sort_by_frame()
            if len(track.frames) < 2: continue
            
            # Find gaps
            new_frames = []
            new_bboxes = []
            
            for i in range(len(track.frames) - 1):
                f_current = track.frames[i]
                f_next = track.frames[i+1]
                b_current = track.bboxes[i]
                b_next = track.bboxes[i+1]
                
                # Add current
                new_frames.append(f_current)
                new_bboxes.append(b_current)
                
                # If gap is small (< 20 frames), fill it
                gap = f_next - f_current
                if 1 < gap < 20:
                    for step in range(1, gap):
                        # Linear Interpolation
                        alpha = step / gap
                        interp_box = (1 - alpha) * b_current + alpha * b_next
                        new_frames.append(f_current + step)
                        new_bboxes.append(interp_box)
            
            # Add last
            new_frames.append(track.frames[-1])
            new_bboxes.append(track.bboxes[-1])
            
            track.frames = new_frames
            track.bboxes = new_bboxes

        # 2. CLUSTERING (Determine Teams)
        print("Clustering teams based on global data...")
        all_features = []
        for track in self.tracks.values():
            if not track.is_ball and len(track.features) > 0:
                all_features.extend(track.features)
        
        if len(all_features) < 10:
            print("Not enough people found.")
            return

        # Train global model
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(np.array(all_features))
        
        # Identify Green Team
        center_0 = kmeans.cluster_centers_[0].reshape(16, 16)
        center_1 = kmeans.cluster_centers_[1].reshape(16, 16)
        score_0 = np.sum(center_0[4:8, 5:])
        score_1 = np.sum(center_1[4:8, 5:])
        
        green_cluster = 0 if score_0 > score_1 else 1
        
        # 3. ASSIGN TEAMS TO TRACKS
        for t_id, track in self.tracks.items():
            if track.is_ball or not track.features: continue
            
            # Predict every single frame this person appeared in
            votes = kmeans.predict(np.array(track.features))
            
            # Global Average Vote
            avg_vote = np.mean(votes == green_cluster) # % of frames they looked Green
            
            if avg_vote > 0.5:
                track.assigned_team = 1 # Green
            else:
                track.assigned_team = 2 # White

    # --- STEP 3: RENDERING ---
    def pass_2_render(self):
        print(f"--- PASS 2: Rendering to {OUTPUT_PATH} ---")
        cap = cv2.VideoCapture(INPUT_PATH)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        
        # Pre-organize data: Frame ID -> List of (TrackID, BBox)
        frame_map = defaultdict(list)
        for t_id, track in self.tracks.items():
            for f_idx, bbox in zip(track.frames, track.bboxes):
                frame_map[f_idx].append((t_id, bbox))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            if frame_idx % 50 == 0: print(f"Rendering Frame {frame_idx}...")
            
            # Retrieve prepared tracks for this frame
            active_tracks = frame_map[frame_idx]
            
            for t_id, bbox in active_tracks:
                track = self.tracks[t_id]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Determine Color
                if track.is_ball:
                    color = COLOR_BALL
                    label = "Ball"
                else:
                    if track.assigned_team == 1:
                        color = COLOR_TEAM_1
                        label = "Team 1"
                    elif track.assigned_team == 2:
                        color = COLOR_TEAM_2
                        label = "Team 2"
                    else:
                        color = (100, 100, 100)
                        label = "Unknown"

                # Draw BBox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label
                cv2.putText(frame, f"{label} ID:{t_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw Trail (Look back in history)
                # Since we have the whole track object, we can find indices where frame < current_frame
                trail_points = []
                for f, b in zip(track.frames, track.bboxes):
                    if f < frame_idx and f > frame_idx - 30: # Last 30 frames
                        cx = int((b[0]+b[2])/2)
                        cy = int(b[3]) if not track.is_ball else int((b[1]+b[3])/2)
                        trail_points.append((cx, cy))
                
                if len(trail_points) > 1:
                    cv2.polylines(frame, [np.array(trail_points, dtype=np.int32)], False, COLOR_TRAIL if not track.is_ball else COLOR_BALL, 2)

            out.write(frame)

        cap.release()
        out.release()
        print("Processing Complete!")

if __name__ == "__main__":
    tracker = OfflineTracker()
    tracker.pass_1_analysis()
    tracker.optimize_tracks()
    tracker.pass_2_render()