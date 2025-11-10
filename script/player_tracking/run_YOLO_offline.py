from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import defaultdict, Counter

# ---------- Tracker code ----------

class Track:
    _next_id = 0

    def __init__(self, bbox, max_missed=2):
        # bbox = (x1, y1, x2, y2)
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.missed = 0
        self.max_missed = max_missed

    def update(self, bbox):
        # exponential smoothing so real-time pass is not super jittery
        alpha = 0.5
        self.bbox = tuple(
            int(alpha * n + (1 - alpha) * o)
            for n, o in zip(bbox, self.bbox)
        )
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    @property
    def is_dead(self):
        return self.missed > self.max_missed


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def update_tracks(tracks, detections, iou_thresh=0.3, max_missed=2):
    if len(tracks) == 0:
        return [Track(b, max_missed=max_missed) for b in detections]

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, t in enumerate(tracks):
        for d_idx, d in enumerate(detections):
            iou_matrix[t_idx, d_idx] = iou(t.bbox, d)

    matched_tracks = set()
    matched_dets = set()

    while True:
        if iou_matrix.size == 0:
            break
        t_idx, d_idx = divmod(iou_matrix.argmax(), iou_matrix.shape[1])
        if iou_matrix[t_idx, d_idx] < iou_thresh:
            break

        tracks[t_idx].update(detections[d_idx])
        matched_tracks.add(t_idx)
        matched_dets.add(d_idx)
        iou_matrix[t_idx, :] = -1
        iou_matrix[:, d_idx] = -1

    for idx, t in enumerate(tracks):
        if idx not in matched_tracks:
            t.mark_missed()

    for d_idx, d in enumerate(detections):
        if d_idx not in matched_dets:
            tracks.append(Track(d, max_missed=max_missed))

    tracks = [t for t in tracks if not t.is_dead]
    return tracks


def deduplicate_tracks(tracks, iou_thresh=0.7):
    """Remove duplicate tracks that overlap too much in the same frame."""
    to_remove = set()
    n = len(tracks)
    for i in range(n):
        for j in range(i + 1, n):
            if i in to_remove or j in to_remove:
                continue
            if iou(tracks[i].bbox, tracks[j].bbox) > iou_thresh:
                # keep older id
                if tracks[i].id < tracks[j].id:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    return [t for idx, t in enumerate(tracks) if idx not in to_remove]


# ---------- Team classification (white vs green) ----------

# team1 = white uniforms: low saturation, high value
WHITE_LOWER = np.array([0,   0, 180], dtype=np.uint8)
WHITE_UPPER = np.array([180, 60, 255], dtype=np.uint8)

# team2 = green uniforms: green hue band, reasonably saturated
GREEN_LOWER = np.array([35,  40,  40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)


def classify_team(frame, bbox):
    """
    frame: BGR image
    bbox: (x1, y1, x2, y2)
    returns: "team1", "team2", or "unknown"
    team1 = white, team2 = green
    """
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape

    # clamp to image bounds
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    # focus on central torso region (avoid grass, legs, head)
    box_w = x2 - x1
    box_h = y2 - y1
    cx1 = int(x1 + 0.25 * box_w)
    cx2 = int(x1 + 0.75 * box_w)
    cy1 = int(y1 + 0.25 * box_h)
    cy2 = int(y1 + 0.70 * box_h)

    cx1 = max(x1, cx1)
    cx2 = min(x2, cx2)
    cy1 = max(y1, cy1)
    cy2 = min(y2, cy2)

    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    white_count = int(np.count_nonzero(white_mask))
    green_count = int(np.count_nonzero(green_mask))
    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return "unknown"

    white_frac = white_count / total
    green_frac = green_count / total
    MIN_FRAC = 0.08  # minimum fraction to be confident

    if white_frac > green_frac * 1.2 and white_frac > MIN_FRAC:
        return "team1"
    elif green_frac > white_frac * 1.2 and green_frac > MIN_FRAC:
        return "team2"
    else:
        return "unknown"


# ---------- Smoothing helper ----------

def smooth_1d(values, window=7):
    """Centered moving average (uses past + future)."""
    if len(values) <= 2:
        return values
    w = min(window, len(values))
    kernel = np.ones(w, dtype=float) / w
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


# ---------- 1st pass: YOLO + tracking + per-frame team labels ----------

which_yolo = 'n' # 'n', 's', 'm', 'l', 'x'

model = YOLO(f"yolov8{which_yolo}.pt")

input_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
output_path = f"/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/YOLO_{which_yolo}_trimmed_vid_1_offline_smooth_teams.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frames_count)

# frame_idx -> list of (track_id, bbox, frame_team_label)
frame_tracks = defaultdict(list)
tracks = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640, conf=0.2)[0]

    det_boxes = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if model.names[cls_id] != "person":
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_boxes.append((x1, y1, x2, y2))

    tracks = update_tracks(tracks, det_boxes, iou_thresh=0.3, max_missed=2)
    tracks = deduplicate_tracks(tracks, iou_thresh=0.7)

    # store snapshot for this frame with per-frame team label
    for t in tracks:
        team_label = classify_team(frame, t.bbox)
        frame_tracks[frame_idx].append((t.id, t.bbox, team_label))

    frame_idx += 1

cap.release()
print("Finished first pass. Num tracks seen:", Track._next_id)

# ---------- Build per-track history & smooth (boxes + team majority) ----------

# id -> list of (frame_idx, bbox, frame_team_label)
track_history = defaultdict(list)
for f_idx, items in frame_tracks.items():
    for tid, bbox, team_label in items:
        track_history[tid].append((f_idx, bbox, team_label))

MIN_LEN = 5  # drop short tracks

# frame_idx -> list of (tid, bbox, final_team_label)
smoothed_by_frame = defaultdict(list)

for tid, seq in track_history.items():
    if len(seq) < MIN_LEN:
        continue

    seq.sort(key=lambda x: x[0])
    frames = np.array([f for f, _, _ in seq])
    x1s = np.array([b[0] for _, b, _ in seq], dtype=float)
    y1s = np.array([b[1] for _, b, _ in seq], dtype=float)
    x2s = np.array([b[2] for _, b, _ in seq], dtype=float)
    y2s = np.array([b[3] for _, b, _ in seq], dtype=float)
    labels = [lab for _, _, lab in seq if lab != "unknown"]

    # smooth coordinates using future + past
    x1s_s = smooth_1d(x1s, window=7)
    y1s_s = smooth_1d(y1s, window=7)
    x2s_s = smooth_1d(x2s, window=7)
    y2s_s = smooth_1d(y2s, window=7)

    # majority vote for team label over the whole track
    if labels:
        final_team = Counter(labels).most_common(1)[0][0]
    else:
        final_team = "unknown"

    for f, x1, y1, x2, y2 in zip(frames, x1s_s, y1s_s, x2s_s, y2s_s):
        bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
        smoothed_by_frame[f].append((tid, bbox, final_team))

print("Smoothing + team aggregation complete.")

# ---------- 2nd pass: draw smoothed, team-labeled tracks ----------

cap = cv2.VideoCapture(input_path)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for tid, bbox, team in smoothed_by_frame.get(frame_idx, []):
        x1, y1, x2, y2 = bbox

        if team == "team1":
            color = (255, 255, 255)  # white
            label = f"id {tid} team1"
        elif team == "team2":
            color = (0, 255, 0)      # green
            label = f"id {tid} team2"
        else:
            color = (0, 255, 255)    # unknown
            label = f"id {tid} ?"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved offline-smoothed, team-labeled video to: {output_path}")