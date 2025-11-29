import cv2
import numpy as np

# --- Configuration ---
INPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
OUTPUT_PATH = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/field_localization.mp4"

# Sensitivity: Adjusts how fast the camera moves on the map relative to the video
MOVEMENT_SCALE = 0.5 
MINIMAP_WIDTH = 900
MINIMAP_HEIGHT = 600
FEATURE_REFRESH_THRESHOLD = 50 

def create_soccer_field(width, height):
    """
    Creates a detailed top-down image of a soccer field with penalty boxes,
    center circle, and corner arcs.
    """
    field_color = (34, 139, 34)  # Forest Green
    line_color = (255, 255, 255) # White
    line_thickness = 2
    
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = field_color

    # Margins
    margin = 40
    w_field = width - 2 * margin
    h_field = height - 2 * margin
    
    # Outer Boundary
    top_left = (margin, margin)
    bottom_right = (width - margin, height - margin)
    cv2.rectangle(img, top_left, bottom_right, line_color, line_thickness)
    
    # Center Line
    center_x = width // 2
    cv2.line(img, (center_x, margin), (center_x, height - margin), line_color, line_thickness)
    
    # Center Circle
    center_y = height // 2
    cv2.circle(img, (center_x, center_y), int(h_field * 0.15), line_color, line_thickness)
    cv2.circle(img, (center_x, center_y), 4, line_color, -1) # Center spot

    # --- Penalty Areas ---
    # Dimensions relative to field height (approximate)
    box_h = int(h_field * 0.6)
    box_w = int(w_field * 0.16)
    goal_box_h = int(h_field * 0.25)
    goal_box_w = int(w_field * 0.05)
    
    # Left Penalty Box
    cv2.rectangle(img, (margin, center_y - box_h // 2), (margin + box_w, center_y + box_h // 2), line_color, line_thickness)
    # Right Penalty Box
    cv2.rectangle(img, (width - margin - box_w, center_y - box_h // 2), (width - margin, center_y + box_h // 2), line_color, line_thickness)

    # Left Goal Box
    cv2.rectangle(img, (margin, center_y - goal_box_h // 2), (margin + goal_box_w, center_y + goal_box_h // 2), line_color, line_thickness)
    # Right Goal Box
    cv2.rectangle(img, (width - margin - goal_box_w, center_y - goal_box_h // 2), (width - margin, center_y + goal_box_h // 2), line_color, line_thickness)

    # Penalty Spots
    spot_dist = int(w_field * 0.11)
    cv2.circle(img, (margin + spot_dist, center_y), 4, line_color, -1)
    cv2.circle(img, (width - margin - spot_dist, center_y), 4, line_color, -1)

    # Corner Arcs
    corner_radius = 20
    # Top Left
    cv2.ellipse(img, (margin, margin), (corner_radius, corner_radius), 0, 0, 90, line_color, line_thickness)
    # Top Right
    cv2.ellipse(img, (width - margin, margin), (corner_radius, corner_radius), 0, 90, 180, line_color, line_thickness)
    # Bottom Right
    cv2.ellipse(img, (width - margin, height - margin), (corner_radius, corner_radius), 0, 180, 270, line_color, line_thickness)
    # Bottom Left
    cv2.ellipse(img, (margin, height - margin), (corner_radius, corner_radius), 0, 270, 360, line_color, line_thickness)

    return img

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_PATH}")
        return

    # Get video properties
    ret, old_frame = cap.read()
    if not ret:
        return
    
    vid_h, vid_w = old_frame.shape[:2]
    
    # Create the detailed base minimap
    minimap_w = MINIMAP_WIDTH
    minimap_h = MINIMAP_HEIGHT
    base_field = create_soccer_field(minimap_w, minimap_h)

    # Video Writer (Initialized later once we know the combined frame size)
    out = None

    # Initial Computer Vision Setup
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for ShiTomasi corner detection
    # We reduce minDistance slightly to capture more points on the complex center logo
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=20, blockSize=7)
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect initial points
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # State variables
    # Start at center of field (50-yard line)
    camera_x_pos = minimap_w // 2 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # --- MOTION ESTIMATION ---
                m, inliers = cv2.estimateAffinePartial2D(good_old, good_new)

                if m is not None:
                    dx = m[0, 2]
                    # Subtract dx because rightward pixel movement = leftward camera pan
                    camera_x_pos -= (dx * MOVEMENT_SCALE)

                if len(good_new) < FEATURE_REFRESH_THRESHOLD:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                else:
                    p0 = good_new.reshape(-1, 1, 2)
                    
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

        # --- DRAWING THE MINIMAP ---
        current_minimap = base_field.copy()
        
        # Clamp camera position
        cam_center_x = int(camera_x_pos)
        cam_center_x = max(0, min(minimap_w, cam_center_x))
        
        # Define the Camera View Cone (Trapezoid)
        camera_lens_pos = (cam_center_x, minimap_h - 20) 
        
        # The field of view projection
        fov_width_far = 250
        fov_width_near = 100
        
        pt_far_left = (cam_center_x - fov_width_far // 2, 40)
        pt_far_right = (cam_center_x + fov_width_far // 2, 40)
        pt_near_left = (cam_center_x - fov_width_near // 2, minimap_h - 40)
        pt_near_right = (cam_center_x + fov_width_near // 2, minimap_h - 40)
        
        fov_poly = np.array([pt_near_left, pt_far_left, pt_far_right, pt_near_right], np.int32)
        
        # Draw translucent View Cone
        overlay = current_minimap.copy()
        cv2.fillPoly(overlay, [fov_poly], (0, 255, 255)) # Yellow fill
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, current_minimap, 1 - alpha, 0, current_minimap)
        
        cv2.polylines(current_minimap, [fov_poly], True, (255, 255, 0), 2)
        
        # Draw Camera Icon
        cv2.circle(current_minimap, camera_lens_pos, 10, (0, 0, 255), -1) 
        cv2.putText(current_minimap, "CAM", (camera_lens_pos[0]-15, camera_lens_pos[1]+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # "Center Lock" Indicator
        # If camera is near the middle (within 20 pixels), light up
        if abs(cam_center_x - (minimap_w // 2)) < 20:
             cv2.putText(current_minimap, "CENTER LOCKED", (minimap_w//2 - 60, minimap_h - 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- COMBINE & DISPLAY ---
        # Resize frame to match minimap height
        scale_factor = minimap_h / vid_h
        resized_frame = cv2.resize(frame, (int(vid_w * scale_factor), minimap_h))
        
        # Stack images side-by-side
        combined_view = np.hstack((resized_frame, current_minimap))
        
        # Initialize VideoWriter if not done
        if out is None:
            h_out, w_out = combined_view.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (w_out, h_out))

        # Write the frame
        out.write(combined_view)

        # Display (optional, scaled down for screen)
        display_scale = 0.8
        h_final, w_final = combined_view.shape[:2]
        display_img = cv2.resize(combined_view, (int(w_final * display_scale), int(h_final * display_scale)))

        old_gray = frame_gray.copy()
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()