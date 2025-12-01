import cv2
import numpy as np
import math

# --- Configuration ---
# NOTE: Update these paths for your environment before running!
INPUT_PATH = r"D:\surfe\Desktop\University Applications\Carnegie Mellon University\Courses\24-678 Computer Vision for Engineers Fall 2025\Project(1)\trimmed_vid_1.mp4"
OUTPUT_PATH = r"D:\surfe\Desktop\University Applications\Carnegie Mellon University\Courses\24-678 Computer Vision for Engineers Fall 2025\Project(1)\field_localization_v4.mp4"

# --- TUNING CONSTANTS ---
# Rotation Sensitivity: Radians to rotate per pixel of optical flow translation.
ROTATION_SENSITIVITY = 0.0005
# Tilt Sensitivity: Map pixels to shift per video pixel of vertical movement.
TILT_SENSITIVITY = 0.5
# Zoom Sensitivity: Multiplier for the calculated scale factor.
ZOOM_SENSITIVITY = 1.0 

MINIMAP_WIDTH = 900
MINIMAP_HEIGHT = 600

# Feature Tracking
FEATURE_REFRESH_THRESHOLD = 50 

# RANSAC Parameters (for Homography/Outlier Rejection)
RANSAC_REPROJECTION_THRESHOLD = 5.0 # Max pixel distance for an inlier

# Perspective / FOV Setup
# Only consider features below this line (e.g., ignore sky/stands/very close foreground)
FEATURE_CROP_TOP_PERCENT = 0.30 

# Camera Geometry
PIVOT_OFFSET_Y = 400 
BASE_FOV_ANGLE = np.deg2rad(45)

def create_soccer_field(width, height):
    """
    Creates a detailed top-down image of a soccer field with line markings.
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

    # Penalty Areas (approximate dimensions)
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

    return img

def create_feature_mask(frame_h, frame_w):
    """
    Creates a simple mask to restrict feature detection to the bottom 
    part of the frame (where the field is) and avoid the sky/stands/very close foreground.
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    
    # Define ROI: bottom 70% of the screen
    top_y = int(frame_h * FEATURE_CROP_TOP_PERCENT)
    
    # Fill the ROI
    mask[top_y:frame_h, 0:frame_w] = 255
    return mask

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {INPUT_PATH}")
        return

    # Get video properties and initial frame
    ret, old_frame = cap.read()
    if not ret:
        return
    
    vid_h, vid_w = old_frame.shape[:2]
    
    # Create the detailed base minimap
    minimap_w = MINIMAP_WIDTH
    minimap_h = MINIMAP_HEIGHT
    base_field = create_soccer_field(minimap_w, minimap_h)

    # Video Writer
    out = None

    # Initial Computer Vision Setup
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Mask for feature tracking
    feature_roi_mask = create_feature_mask(vid_h, vid_w)
    
    # Feature params
    feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=15, blockSize=7)
    
    # Optical Flow params
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect initial points, restricted by the ROI mask
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=feature_roi_mask, **feature_params)

    # State variables
    camera_angle = 0.0
    smooth_camera_angle = 0.0
    camera_zoom = 1.0
    smooth_camera_zoom = 1.0
    camera_tilt = 0.0
    smooth_camera_tilt = 0.0

    # Fixed Pivot Point (Virtual Camera Location)
    pivot_x = minimap_w // 2
    pivot_y = minimap_h + PIVOT_OFFSET_Y

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- OPTICAL FLOW ---
        d_angle = 0
        d_zoom = 1.0
        d_tilt = 0
        
        if p0 is not None and len(p0) > 0:
            # 1. Track features
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) < 4:
                    # Not enough points for Homography/Affine, refresh features and skip motion update
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                    old_gray = frame_gray.copy()
                    continue

                # --- NEW ROBUST MOTION ESTIMATION (Homography + Affine) ---
                # 2. Estimate Homography with RANSAC to filter OUTLIERS (e.g., moving players)
                # This finds the best perspective transform (H) and a mask of inliers.
                H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)
                
                # Filter points to keep only RANSAC inliers
                if H is not None and mask is not None and np.sum(mask) > 4:
                    inlier_mask = mask.ravel() == 1
                    inlier_old = good_old[inlier_mask]
                    inlier_new = good_new[inlier_mask]
                    
                    # 3. Use the CLEAN inliers to re-estimate the Affine Partial matrix (m)
                    # We use Affine Partial here to easily decompose into tilt/pan/zoom
                    m, inliers_final = cv2.estimateAffinePartial2D(inlier_old, inlier_new)

                    if m is not None:
                        # Decompose 2x3 Affine matrix m:
                        # [[s*cos(th), -s*sin(th), tx],
                        #  [s*sin(th),  s*cos(th), ty]]
                        
                        # 1. Translation in x (horizontal pan) -> Rotation
                        raw_dx = m[0, 2]
                        d_angle = raw_dx * ROTATION_SENSITIVITY
                        
                        # 2. Translation in y (vertical tilt)
                        raw_dy = m[1, 2]
                        d_tilt = -raw_dy * TILT_SENSITIVITY # Negative sign ensures correct map direction
                        
                        # 3. Scale (Zoom)
                        scale_x = np.sqrt(m[0, 0]**2 + m[0, 1]**2)
                        scale_y = np.sqrt(m[1, 0]**2 + m[1, 1]**2)
                        current_scale = (scale_x + scale_y) / 2
                        d_zoom = current_scale 
                        
                        # Refresh features using only the inlier points
                        if len(inlier_new) < FEATURE_REFRESH_THRESHOLD:
                             # Re-detect new points if too few inliers remain
                             p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                        else:
                             # Use the new positions of the inliers for the next frame's tracking
                             p0 = inlier_new.reshape(-1, 1, 2)
                             
                        # Visualization of tracks (only inliers)
                        frame_display = frame.copy()
                        for i, (new, old) in enumerate(zip(inlier_new, inlier_old)):
                            a, b = new.ravel()
                            cv2.circle(frame_display, (int(a), int(b)), 2, (0, 255, 0), -1)
                        frame = frame_display # Use frame with tracks for output
                        
                    else:
                        # Failed to re-estimate Affine, refresh features
                        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
                else:
                    # Failed RANSAC (too few inliers), refresh features
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
            else:
                # Optical flow failed, refresh features
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=feature_roi_mask, **feature_params)
        
        # --- UPDATE STATE ---
        camera_angle += d_angle
        camera_tilt += d_tilt
        camera_zoom *= d_zoom
        
        # Clamp State (Avoid extreme values)
        camera_zoom = max(0.5, min(camera_zoom, 3.0))
        camera_tilt = max(-100, min(camera_tilt, 100))

        # Smoothing
        smooth_camera_angle = smooth_camera_angle * 0.90 + camera_angle * 0.10
        smooth_camera_tilt = smooth_camera_tilt * 0.90 + camera_tilt * 0.10
        smooth_camera_zoom = smooth_camera_zoom * 0.90 + camera_zoom * 0.10
        
        # --- CALCULATE ROTATED & ZOOMED VIEW CONE ---
        
        # 1. Adjust FOV based on Zoom
        current_fov = BASE_FOV_ANGLE / smooth_camera_zoom
        
        # 2. Adjust Y coordinates based on Tilt (Tilt UP decreases Y on map)
        base_y_far = 40
        base_y_near = minimap_h - 40
        
        # Apply tilt offset
        y_far = int(base_y_far + smooth_camera_tilt)
        y_near = int(base_y_near + smooth_camera_tilt)
        
        y_far = max(0, y_far)
        y_near = min(minimap_h, y_near)

        # Angles for left and right edges of the FOV
        theta_left = smooth_camera_angle + (current_fov / 2)
        theta_right = smooth_camera_angle - (current_fov / 2)
        
        # Calculate X coordinates using trigonometry relative to the pivot
        dy_far = y_far - pivot_y
        dy_near = y_near - pivot_y
        
        x_far_left = int(pivot_x + dy_far * np.tan(theta_left))
        x_far_right = int(pivot_x + dy_far * np.tan(theta_right))
        
        x_near_left = int(pivot_x + dy_near * np.tan(theta_left))
        x_near_right = int(pivot_x + dy_near * np.tan(theta_right))
        
        # Points for the trapezoid
        pt_far_left = (x_far_left, y_far)
        pt_far_right = (x_far_right, y_far)
        pt_near_left = (x_near_left, y_near)
        pt_near_right = (x_near_right, y_near)
        
        fov_poly = np.array([pt_near_left, pt_far_left, pt_far_right, pt_near_right], np.int32)

        # --- MAPPING VISUALIZATION ---
        current_minimap = base_field.copy()

        # Perspective Warp Source points: Crop from video feed
        src_top_y = int(vid_h * FEATURE_CROP_TOP_PERCENT)
        src_pts = np.float32([
            [0, src_top_y],      
            [vid_w, src_top_y],  
            [vid_w, vid_h],      
            [0, vid_h]          
        ])
        
        # Destination: The Rotated Trapezoid on the minimap
        dst_pts = np.float32([pt_far_left, pt_far_right, pt_near_right, pt_near_left])
        
        # Compute Matrix & Warp
        # Note: We use the estimated camera state here to define the DST points, 
        # which is the core of the tactical mapping.
        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_frame = cv2.warpPerspective(frame, M_persp, (minimap_w, minimap_h))
        
        # Mask & Blend
        mask = np.zeros((minimap_h, minimap_w), dtype=np.uint8)
        cv2.fillPoly(mask, [fov_poly], 255)
        
        alpha_overlay = 0.5
        roi = current_minimap[mask == 255]
        warped_roi = warped_frame[mask == 255]
        
        blended = cv2.addWeighted(roi, 1 - alpha_overlay, warped_roi, alpha_overlay, 0)
        current_minimap[mask == 255] = blended
        
        # Draw outlines
        cv2.polylines(current_minimap, [fov_poly], True, (0, 255, 255), 2)
        
        # Draw Stationary Camera Icon and Telemetry
        cam_icon_pos = (minimap_w // 2, minimap_h - 10)
        cv2.circle(current_minimap, cam_icon_pos, 10, (0, 0, 255), -1)
        cv2.putText(current_minimap, "FIXED CAM", (cam_icon_pos[0]-40, cam_icon_pos[1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.putText(current_minimap, f"Zoom: {smooth_camera_zoom:.2f}x", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(current_minimap, f"Tilt: {smooth_camera_tilt:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # --- FINAL OUTPUT ---
        scale_factor = minimap_h / vid_h
        resized_frame = cv2.resize(frame, (int(vid_w * scale_factor), minimap_h))
        
        combined_view = np.hstack((resized_frame, current_minimap))
        
        if out is None:
            h_out, w_out = combined_view.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (w_out, h_out))

        out.write(combined_view)
        
        # Update old frame for next loop
        old_gray = frame_gray.copy()
        p0 = p0 # p0 is already updated with the new inlier positions or refreshed
        
        # NOTE: cv2.imshow is commented out as it often fails in interactive environments
        # cv2.imshow('Camera Movement Inference V4', combined_view)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    # Ensure you update INPUT_PATH and OUTPUT_PATH before running
    main()