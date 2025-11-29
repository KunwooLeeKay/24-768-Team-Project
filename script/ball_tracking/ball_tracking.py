import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def interpolate_ball_positions(positions):
    """
    Post-processing magic: Fills in the gaps where detection failed.
    If the ball is detected at frame 10 and 20, but lost in 11-19,
    this draws a straight line between them.
    """
    df = pd.DataFrame(positions, columns=['frame', 'x', 'y', 'w', 'h'])
    
    # Replace zeros (misses) with NaN to allow interpolation
    df.replace(0, np.nan, inplace=True)
    
    # Interpolate missing values linearly
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Fill remaining NaNs with 0 (if lost at very start/end)
    df.fillna(0, inplace=True)
    
    return df.to_dict('records')

def run_professional_tracker(video_path, output_path):
    # 1. LOAD THE AI MODEL
    # 'yolov8x.pt' is the largest, slowest, but MOST ACCURATE model.
    # It will automatically download on first run.
    print("Loading AI Model...")
    model = YOLO('yolov8x.pt') 

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # We will save tracking data here to process later
    ball_positions = [] 

    print("Step 1: Analyzing Video (AI Detection)...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 2. RUN AI TRACKING
        # classes=[32] tells YOLO to ONLY look for "sports ball" (COCO class ID 32)
        # conf=0.25 is the confidence threshold
        # persist=True activates the internal BoT-SORT tracker
        results = model.track(frame, persist=True, verbose=False, classes=[32], conf=0.15)
        
        detected = False
        
        # Did we find a ball?
        if results[0].boxes.id is not None:
            # Get the box with the highest confidence
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Use the most confident detection
            best_idx = np.argmax(confidences)
            x, y, w, h = boxes[best_idx]
            
            ball_positions.append({'frame': frame_idx, 'x': x, 'y': y, 'w': w, 'h': h})
            detected = True
        
        if not detected:
            # Mark as missing (0) to interpolate later
            ball_positions.append({'frame': frame_idx, 'x': 0, 'y': 0, 'w': 0, 'h': 0})

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()

    # 3. INTERPOLATION (The "Robust" part)
    print("Step 2: Cleaning Data & Interpolating Gaps...")
    clean_positions = interpolate_ball_positions(ball_positions)

    # 4. RENDER FINAL VIDEO
    print("Step 3: Rendering Final Video...")
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        data = clean_positions[current_idx]
        
        # If we have valid data (not 0 after interpolation)
        if data['w'] > 0:
            x, y, w, h = int(data['x']), int(data['y']), int(data['w']), int(data['h'])
            
            # Draw center point
            center_x = int(x)
            center_y = int(y)
            
            # Draw HUD
            # Visual: Bounding Box
            top_left = (int(x - w/2), int(y - h/2))
            bottom_right = (int(x + w/2), int(y + h/2))
            
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, "AI TRACKER", (top_left[0], top_left[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        current_idx += 1

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    run_professional_tracker('soccer3.mp4', 'output_robust_3.mp4')