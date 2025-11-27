from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, deque

# --- CONFIGURATION ---
INPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/YOLO_v5_online.mp4"

# Visualization Colors
COLOR_TRAIL = (0, 255, 255)    # Yellow
COLOR_TORSO = (255, 0, 255)    # Magenta (Debug box)

class TeamClassifier:
    def __init__(self):
        self.kmeans = None
        self.collected_samples = []
        self.trained = False
        self.team_colors = {}   # Map {0: BGR, 1: BGR}
        self.team_names = {}    # Map {0: "Green", 1: "White"}

    def get_histogram_feature(self, img):
        """
        Extracts a color fingerprint (Hue + Saturation).
        Robust against brightness changes (Shadows).
        """
        if img.size == 0: return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # We use 2 channels: Hue (0) and Saturation (1). Ignore Value (2) for shadow robustness.
        # 16 bins for Hue, 16 bins for Saturation = 256 feature vector size
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # --- FIX: Convert float32 (OpenCV default) to float64 (Sklearn requirement) ---
        return hist.flatten().astype(np.float64)

    def fit(self):
        """Train the model to distinguish the two teams based on initial footage."""
        if len(self.collected_samples) < 10: return

        print(f"[Info] Training Classifier on {len(self.collected_samples)} samples...")
        data = np.array(self.collected_samples, dtype=np.float64) # Ensure double precision
        
        # 1. Cluster into 2 groups
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.kmeans.fit(data)
        
        # 2. Figure out which cluster is "Green" and which is "White"
        center_0 = self.kmeans.cluster_centers_[0].reshape(16, 16)
        center_1 = self.kmeans.cluster_centers_[1].reshape(16, 16)
        
        # "Greenness" heuristic: Sum of pixels in the Green Hue range (approx bins 4-8 out of 16)
        # and High Saturation (bins 8-16)
        score_0 = np.sum(center_0[4:8, 5:]) 
        score_1 = np.sum(center_1[4:8, 5:])

        if score_0 > score_1:
            # Cluster 0 is Green
            self.team_names = {0: "Team 1", 1: "Team 2"}
            self.team_colors = {0: (0, 200, 0), 1: (220, 220, 220)} # Green, White
        else:
            # Cluster 1 is Green
            self.team_names = {0: "Team 2", 1: "Team 1"}
            self.team_colors = {0: (220, 220, 220), 1: (0, 200, 0)} # White, Green
            
        self.trained = True
        print(f"[Info] Training Complete. {self.team_names}")

    def predict(self, img):
        if not self.trained: return None
        feat = self.get_histogram_feature(img)
        if feat is None: return None
        # Predict expects a 2D array of float64
        return self.kmeans.predict([feat])[0]

def get_torso_crop(frame, box):
    """
    Get the upper-body region.
    Standardized to 20%-80% width, 15%-60% height of bbox.
    """
    x1, y1, x2, y2 = map(int, box)
    w, h = x2-x1, y2-y1
    
    # Safe crop logic
    tx1 = int(x1 + w*0.20)
    tx2 = int(x2 - w*0.20)
    ty1 = int(y1 + h*0.15)
    ty2 = int(y1 + h*0.60)
    
    # Clip to frame
    tx1 = max(0, tx1); ty1 = max(0, ty1)
    tx2 = min(frame.shape[1], tx2); ty2 = min(frame.shape[0], ty2)
    
    return frame[ty1:ty2, tx1:tx2], (tx1, ty1, tx2, ty2)

def run_pipeline():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(INPUT_PATH)
    
    # Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    classifier = TeamClassifier()
    
    # TRACKING STATE
    # track_history: stores path [(x,y), (x,y)...]
    track_history = defaultdict(lambda: deque(maxlen=40))
    
    # cumulative_votes: { track_id: { 0: score, 1: score } }
    # This ensures "Same Entity = Same Team" by accumulating history
    cumulative_votes = defaultdict(lambda: {0: 0, 1: 0})

    # --- PHASE 1: DATA COLLECTION (First 60 frames) ---
    print("Phase 1: Learning team colors...")
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames_buffer.append(frame)
        
        if len(frames_buffer) <= 60:
            results = model.track(frame, persist=True, verbose=False, classes=[0])
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    crop, _ = get_torso_crop(frame, box)
                    if crop.size > 0:
                        feat = classifier.get_histogram_feature(crop)
                        if feat is not None:
                            classifier.collected_samples.append(feat)
        else:
            break
            
    # Train the brain
    classifier.fit()
    
    # --- PHASE 2: PROCESSING ---
    print("Phase 2: Tracking with Continuous Voting...")
    
    # Reset stream via generator to process buffer first
    def frame_generator():
        for f in frames_buffer: yield f
        while True:
            r, f = cap.read()
            if not r: break
            yield f

    frame_count = 0
    
    for frame in frame_generator():
        frame_count += 1
        
        # Track both Person (0) and Ball (32)
        results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, classes=[0, 32])
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2-x1, y2-y1
                
                # Update Trail
                center_bottom = (int(x1 + w/2), int(y2))
                track_history[track_id].append(center_bottom)

                if cls == 32: # Ball
                    # Draw Trail
                    if len(track_history[track_id]) > 1:
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, (0, 100, 255), 2)
                    # Draw Object
                    cv2.circle(frame, (int(x1+w/2), int(y1+h/2)), 5, (0, 165, 255), -1)
                    continue

                if cls == 0: # Person
                    # 1. VISUALIZE TORSO (Debug)
                    crop, torso_coords = get_torso_crop(frame, box)
                    tx1, ty1, tx2, ty2 = torso_coords
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), COLOR_TORSO, 1)

                    # 2. CLASSIFY & VOTE
                    # We classify *every* frame to build confidence
                    predicted_cluster = classifier.predict(crop)
                    if predicted_cluster is not None:
                        cumulative_votes[track_id][predicted_cluster] += 1
                    
                    # 3. DETERMINE WINNER
                    votes_0 = cumulative_votes[track_id][0]
                    votes_1 = cumulative_votes[track_id][1]
                    total_votes = votes_0 + votes_1
                    
                    # Heuristic: Who is winning?
                    if votes_0 > votes_1:
                        winning_cluster = 0
                        confidence = votes_0 / (total_votes + 1e-5)
                    else:
                        winning_cluster = 1
                        confidence = votes_1 / (total_votes + 1e-5)
                    
                    # 4. DRAW
                    # If we have very little data (< 5 frames), don't commit yet
                    if total_votes < 5:
                        color = (128, 128, 128)
                        label = "Scanning..."
                    else:
                        color = classifier.team_colors[winning_cluster]
                        label = f"{classifier.team_names[winning_cluster]}"
                    
                    # Draw BBox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw Label with Confidence Bar
                    # Bar width based on confidence (max 50px)
                    bar_width = int(50 * confidence)
                    cv2.rectangle(frame, (x1, y1-15), (x1 + bar_width, y1-10), color, -1)
                    cv2.rectangle(frame, (x1, y1-15), (x1 + 50, y1-10), (255,255,255), 1) # Outline
                    
                    cv2.putText(frame, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw Trail (Yellow for players)
                    if len(track_history[track_id]) > 1:
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, COLOR_TRAIL, 1)

        out.write(frame)
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
        
        # # Safety break for debugging length
        # if frame_count > 600: break

    cap.release()
    out.release()
    print("Done. Saved to", OUTPUT_PATH)

if __name__ == "__main__":
    run_pipeline()