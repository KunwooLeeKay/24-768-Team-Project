import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ---------------------------
# Backend-robust video opener (Quick Fix: avoid GStreamer)
# ---------------------------
def open_video_safely(path):
    """
    Try multiple backends to avoid GStreamer (FFmpeg/MSMF/DirectShow/AVFoundation).
    """
    candidates = [
        (cv2.CAP_FFMPEG, "FFmpeg"),             # Most OpenCV pip wheels have this
        (cv2.CAP_MSMF, "MSMF"),                 # Windows
        (cv2.CAP_DSHOW, "DirectShow"),          # Windows fallback
        (cv2.CAP_AVFOUNDATION, "AVFoundation"), # macOS
        (cv2.CAP_ANY, "ANY"),                   # last resort (may pick GStreamer)
    ]
    for api, name in candidates:
        cap = cv2.VideoCapture(path, api)
        if cap.isOpened():
            print(f"[ok] Opened video with backend: {name}")
            return cap
        cap.release()
    raise RuntimeError(
        "Could not open video with available backends. "
        "Consider re-encoding the file (e.g., ffmpeg → H.264/AAC)."
    )


# ---------------------------
# Mouse interaction for point picking
# ---------------------------
class PointPicker:
    def __init__(self, image, win_name="Pick Points"):
        self.image = image.copy()
        self.base = image.copy()
        self.points = []
        self.win_name = win_name
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self._redraw()

    def _redraw(self):
        self.image = self.base.copy()
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.image, f"{i}", (x + 8, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.win_name, self.image)

    def pick(self, min_points=4):
        self._redraw()
        print("Left-click: add point | Right-click: undo | 'r': reset | 'q': finish (>=4)")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') and len(self.points) >= min_points:
                break
            if key == ord('r'):
                self.points = []
                self._redraw()
            if key == 27:  # ESC
                break
        cv2.destroyWindow(self.win_name)
        return np.array(self.points, dtype=np.float32)


# ---------------------------
# Utilities
# ---------------------------
def load_frame_from_video(video_path, frame_index):
    cap = open_video_safely(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_index = max(0, min(frame_index, max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def prompt_dst_points(src_pts, units="yards", dst_source=None):
    """
    Collect destination (field-plane) coords matching the clicked src_pts order.
    If dst_source is JSON/CSV, read from it:
      - JSON: {"points": [[x1,y1], ...]}
      - CSV:  each line 'x,y'
    Else prompt in terminal.
    """
    if dst_source:
        p = Path(dst_source)
        if not p.exists():
            raise FileNotFoundError(f"Destination points file not found: {dst_source}")
        if p.suffix.lower() == ".json":
            data = json.loads(Path(dst_source).read_text())
            dst = np.array(data["points"], dtype=np.float32)
        else:
            rows = []
            for line in Path(dst_source).read_text().strip().splitlines():
                x, y = line.strip().split(",")
                rows.append([float(x), float(y)])
            dst = np.array(rows, dtype=np.float32)
        if len(dst) != len(src_pts):
            raise ValueError("Destination point count does not match clicked source points")
        return dst

    print(f"\nEnter {len(src_pts)} destination coordinates in {units} (order must match clicks). Example: 0,0")
    dst = []
    for i in range(len(src_pts)):
        while True:
            s = input(f"  dst[{i}] (x,y): ").strip()
            try:
                x, y = [float(v) for v in s.split(",")]
                dst.append([x, y])
                break
            except Exception:
                print("  Parse error. Please enter as x,y (e.g., 35.0, 12.0)")
    return np.array(dst, dtype=np.float32)


def compute_homography(src_pts, dst_pts, ransac_thresh=3.0):
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    inliers = int(mask.sum()) if mask is not None else len(src_pts)
    return H, mask, inliers


def warp_topdown(image, H, out_size, bg_color=(255, 255, 255)):
    w, h = out_size
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    warped = cv2.warpPerspective(image, H, (w, h), dst=canvas, flags=cv2.INTER_LINEAR)
    return warped


def perspective_transform_points(pts_uv, H):
    pts = np.array(pts_uv, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, H)
    return mapped.reshape(-1, 2)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Single-camera field calibration via homography (soccer on football field).")
    ap.add_argument("--video", required=True, help="Path to video")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to calibrate on")
    ap.add_argument("--units", default="yards", choices=["yards", "meters", "unitless"],
                    help="Units for destination coordinates")
    ap.add_argument("--dst", default=None, help="Optional JSON/CSV with destination points (same order as clicks)")
    ap.add_argument("--out_prefix", default="calib/field", help="Output prefix for H and QA images")
    ap.add_argument("--ransac", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)")
    ap.add_argument("--out_w", type=int, default=1200, help="Width of rectified top-down image")
    ap.add_argument("--out_h", type=int, default=533, help="Height of rectified top-down image")
    args = ap.parse_args()

    os.makedirs(Path(args.out_prefix).parent, exist_ok=True)

    # 1) Load a frame
    frame = load_frame_from_video(args.video, args.frame)
    disp = frame.copy()
    cv2.putText(disp, "Left-click:add | Right-click:undo | 'r':reset | 'q':finish (>=4)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 50), 2)

    # 2) Pick source points (pixels)
    picker = PointPicker(disp, "Pick Points")
    src_pts = picker.pick(min_points=4)
    if len(src_pts) < 4:
        print("Need at least 4 points. Exiting.")
        return

    # 3) Destination points (same order as clicks)
    dst_pts = prompt_dst_points(src_pts, units=args.units, dst_source=args.dst)

    # 4) Compute homography
    H, mask, inliers = compute_homography(src_pts, dst_pts, ransac_thresh=args.ransac)
    if H is None:
        print("Homography failed. Try better spread points / adjust RANSAC threshold.")
        return

    # Save H
    np.save(f"{args.out_prefix}_H.npy", H)
    with open(f"{args.out_prefix}_H.txt", "w") as f:
        np.set_printoptions(suppress=True, linewidth=120)
        f.write(repr(H))

    inlier_ratio = inliers / len(src_pts)
    print(f"\nHomography computed. Inliers: {inliers}/{len(src_pts)} (ratio={inlier_ratio:.2f})")
    print(f"Saved: {args.out_prefix}_H.npy  and  {args.out_prefix}_H.txt")

    # 5) Visual check: reproject points back onto the frame (fixed inverse)
    reproj = frame.copy()
    H_inv = np.linalg.inv(H)  # <-- FIXED (use NumPy inverse)
    for i, (u, v) in enumerate(src_pts):
        xy = perspective_transform_points([(u, v)], H)[0]
        uv_back = perspective_transform_points([xy], H_inv)[0]
        cv2.circle(reproj, (int(uv_back[0]), int(uv_back[1])), 5, (0, 0, 255), -1)
        cv2.putText(reproj, f"{i}", (int(uv_back[0]) + 8, int(uv_back[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(f"{args.out_prefix}_reproj_points.png", reproj)

    # 6) Produce a rectified top-down snapshot for QA
    warped = warp_topdown(frame, H, (args.out_w, args.out_h), bg_color=(255, 255, 255))
    cv2.imwrite(f"{args.out_prefix}_rectified.png", warped)

    print(f"Saved QA images: {args.out_prefix}_reproj_points.png  and  {args.out_prefix}_rectified.png")

    # 7) Helper usage message
    print("\nTo map a pixel (u,v) to field (x,y):")
    print(f"""
import numpy as np, cv2
H = np.load('{args.out_prefix}_H.npy')
pt = np.array([[[u, v]]], dtype=np.float32)
xy = cv2.perspectiveTransform(pt, H)[0,0]
print('Field coordinates:', xy)
""")


if __name__ == "__main__":
    main()