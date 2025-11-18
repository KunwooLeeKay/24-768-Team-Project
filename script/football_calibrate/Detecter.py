import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def detect_horizontal_lines(img, min_y=80, max_y=100):
    """Find horizontal lines (back and front of pitch)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    
    lines = cv2.HoughLines(edges, 1, np.pi/720, 500,
                          min_theta=min_y*np.pi/180,
                          max_theta=max_y*np.pi/180)
    
    if lines is None:
        return None, None
    
    height, width = img.shape[:2]
    back_line = front_line = None
    back_y = 0
    front_y = height
    
    # Find top and bottom horizontal lines
    for line in lines:
        rho, theta = line[0]
        y = (rho - width/2 * np.cos(theta)) / np.sin(theta)
        
        if back_y < y < height/2:
            back_y = y
            back_line = line[0]
        elif height/2 < y < front_y:
            front_y = y
            front_line = line[0]
    
    return back_line, front_line


def detect_vertical_line(img):
    """Find main vertical line (center line)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    
    lines = cv2.HoughLines(edges, 1, np.pi/360, 200,
                          min_theta=0,
                          max_theta=40*np.pi/180)
    
    return lines[0][0] if lines is not None else None


def line_intersection(line1, line2):
    """Find intersection point of two lines in polar coordinates"""
    if line1 is None or line2 is None:
        return None
    
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    if theta1 == theta2:
        return None
    
    x = (rho1*np.sin(theta2) - rho2*np.sin(theta1)) / np.sin(theta2-theta1)
    y = (rho1 - x*np.cos(theta1)) / np.sin(theta1) if theta1 != 0 else \
        (rho2 - x*np.cos(theta2)) / np.sin(theta2)
    
    return (int(x), int(y))


def draw_line(img, line, color=(0,255,0)):
    """Draw a line on the image"""
    if line is None:
        return img
    
    rho, theta = line
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*a))
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*a))
    cv2.line(img, pt1, pt2, color, 2)
    return img


def draw_point(img, point, color=(0,0,255)):
    """Draw a point on the image"""
    if point is None:
        return img
    # Commented out drawing circles
    # cv2.circle(img, point, 5, color, -1)
    return img


def detect_center_circle(img, mask_top=None, mask_bottom=None):
    """Detect the center circle of the pitch"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mask to search area (between back and front lines if known)
    if mask_top and mask_bottom:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[int(mask_top):int(mask_bottom), :] = 255
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    
    if circles is None:
        return None, []
    
    circles = np.uint16(np.around(circles[0]))
    
    # Find the largest circle (likely the center circle)
    largest = max(circles, key=lambda c: c[2])
    center = (int(largest[0]), int(largest[1]))
    radius = int(largest[2])
    
    # Calculate key circle points (left, right, top, bottom)
    circle_points = {
        'center': center,
        'radius': radius,
        'left': (center[0] - radius, center[1]),
        'right': (center[0] + radius, center[1]),
        'top': (center[0], center[1] - radius),
        'bottom': (center[0], center[1] + radius)
    }
    
    return (center, radius), circle_points


def find_central_ellipse(img, back_middle_point, front_middle_point, debug=False):
    """
    Find the central ellipse (representing the central circle) and return its parameters.
    """
    if back_middle_point is None or front_middle_point is None:
        return None

    dst = cv2.Canny(img, 20, 100, None, 3)

    if debug:
        cv2.imshow("Canny", dst)
        cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and fit ellipses
    ellipses = []
    for contour in contours:
        if len(contour) >= 5:  # Minimum points required to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)

    if not ellipses:
        return None

    # Select the ellipse closest to the center of the field
    center_approx = 0.3 * np.array(front_middle_point) + 0.7 * np.array(back_middle_point)
    best_ellipse = min(ellipses, key=lambda e: np.linalg.norm(np.array(e[0]) - center_approx))

    if debug:
        output = img.copy()
        cv2.ellipse(output, best_ellipse, (0, 255, 0), 2)
        cv2.imshow("Detected Ellipse", output)
        cv2.waitKey(0)

    return best_ellipse


def process_image(img):
    """Main processing function"""
    # Detect lines
    back_line, front_line = detect_horizontal_lines(img)
    main_line = detect_vertical_line(img)
    
    # Find key intersection points
    back_center = line_intersection(main_line, back_line)
    front_center = line_intersection(main_line, front_line)
    
    # Detect center circle (mask search area if we have horizontal lines)
    mask_top = back_center[1] if back_center else None
    mask_bottom = front_center[1] if front_center else None
    circle_data, circle_points = detect_center_circle(img, mask_top, mask_bottom)
    
    # Draw results
    result = img.copy()
    result = draw_line(result, back_line, (0,0,255))     # Red
    result = draw_line(result, front_line, (0,0,255))    # Red
    result = draw_line(result, main_line, (0,255,0))     # Green
    result = draw_point(result, back_center)
    result = draw_point(result, front_center)
    
    # Draw center circle
    if circle_data:
        center, radius = circle_data
        cv2.circle(result, center, radius, (255,0,255), 2)  # Magenta circle
        cv2.circle(result, center, 3, (255,0,255), -1)      # Center point
        
        # Draw circle edge points
        if circle_points:
            pass  # Placeholder to maintain the block structure
    
    # Find and draw central ellipse
    ellipse = find_central_ellipse(img, back_center, front_center, debug=False)
    if ellipse is not None:
        cv2.ellipse(result, ellipse, (255,255,255), 2)
    
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Detect soccer field lines")
    parser.add_argument("input", help="Image file or folder")
    args = parser.parse_args()
    
    path = Path(args.input)
    images = list(path.glob("**/*")) if path.is_dir() else [path]
    
    for img_path in images:
        if not img_path.is_file():
            continue
        
        print(f"Processing: {img_path}")
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
        
        result = process_image(img)
        cv2.imshow("Soccer Pitch Lines", result)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    
    cv2.destroyAllWindows()
