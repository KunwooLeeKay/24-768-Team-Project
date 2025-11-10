import cv2
import numpy as np
import os

# ---------- Simple IoU-based tracker ----------

class Track:
    _next_id = 0

    def __init__(self, bbox, max_missed=10):
        # bbox = (x1, y1, x2, y2)
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.missed = 0
        self.max_missed = max_missed

    def update(self, bbox):
        # smooth the box to reduce jitter
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


def update_tracks(tracks, detections, iou_thresh=0.3, max_missed=10):
    if len(tracks) == 0:
        return [Track(b, max_missed=max_missed) for b in detections]

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, t in enumerate(tracks):
        for d_idx, d in enumerate(detections):
            iou_matrix[t_idx, d_idx] = iou(t.bbox, d)

    matched_tracks = set()
    matched_dets = set()

    while iou_matrix.size > 0:
        t_idx, d_idx = divmod(iou_matrix.argmax(), iou_matrix.shape[1])
        if iou_matrix[t_idx, d_idx] < iou_thresh:
            break

        tracks[t_idx].update(detections[d_idx])
        matched_tracks.add(t_idx)
        matched_dets.add(d_idx)

        iou_matrix[t_idx, :] = -1
        iou_matrix[:, d_idx] = -1

    # unmatched tracks
    for idx, t in enumerate(tracks):
        if idx not in matched_tracks:
            t.mark_missed()

    # new tracks for unmatched detections
    for d_idx, d in enumerate(detections):
        if d_idx not in matched_dets:
            tracks.append(Track(d, max_missed=max_missed))

    # remove dead tracks
    tracks = [t for t in tracks if not t.is_dead]
    return tracks


# ---------- Video + white-color-based detection ----------

input_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
output_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/color_based_trimmed_vid_1.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

tracks = []

# HSV range for "white": low saturation, high value
lower_white = np.array([0, 0, 200])      # H, S, V
upper_white = np.array([180, 60, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Convert to HSV and threshold for white
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 2) Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) Find contours → bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:   # ignore tiny white bits
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by aspect ratio to ignore long white lines on the pitch
        aspect = h / float(w + 1e-6)
        if aspect < 0.5 or aspect > 4.0:
            continue

        detections.append((x, y, x + w, y + h))

    # 4) Update tracks
    tracks = update_tracks(tracks, detections, iou_thresh=0.3, max_missed=8)

    # 5) Draw tracks
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"id {t.id}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved white-uniform tracking video to:\n{output_path}")