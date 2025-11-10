from ultralytics import YOLO
import cv2
import os
import numpy as np

# ---------- Simple IoU-based tracker ----------

class Track:
    _next_id = 0

    def __init__(self, bbox, max_missed=2):
        # bbox = (x1, y1, x2, y2)
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.missed = 0
        self.max_missed = max_missed
        self.team = None  # optional: for your team1/team2 labeling

    def update(self, bbox):
        # simple exponential smoothing of the box to reduce jitter
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
    """
    tracks: list[Track]
    detections: list[(x1, y1, x2, y2)]
    """
    if len(tracks) == 0:
        return [Track(b, max_missed=max_missed) for b in detections]

    # IoU matrix (tracks x detections)
    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t_idx, t in enumerate(tracks):
        for d_idx, d in enumerate(detections):
            iou_matrix[t_idx, d_idx] = iou(t.bbox, d)

    matched_dets = set()
    matched_tracks = set()

    # greedy matching: highest IoU pairs above threshold
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

    # tracks that didn't get matched: mark as missed
    for t_idx, t in enumerate(tracks):
        if t_idx not in matched_tracks:
            t.mark_missed()

    # detections that didn't get matched: create new tracks
    for d_idx, d in enumerate(detections):
        if d_idx not in matched_dets:
            tracks.append(Track(d, max_missed=max_missed))

    # remove dead tracks
    tracks = [t for t in tracks if not t.is_dead]
    return tracks


def deduplicate_tracks(tracks, iou_thresh=0.7):
    """
    Remove duplicate tracks that heavily overlap in the same frame.
    Keep the older (smaller id) track when IoU > iou_thresh.
    """
    to_remove = set()
    n = len(tracks)
    for i in range(n):
        for j in range(i + 1, n):
            if i in to_remove or j in to_remove:
                continue
            if iou(tracks[i].bbox, tracks[j].bbox) > iou_thresh:
                # keep older id, drop newer
                if tracks[i].id < tracks[j].id:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    return [t for idx, t in enumerate(tracks) if idx not in to_remove]


# ---------- YOLO + video saving code with tracking ----------

model = YOLO("yolov8n.pt")

input_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/input/trimmed_vid_1.mp4"
output_path = "/Users/kunwoomac/CodeSpace/24-768-Team-Project/output/YOLO_interp_trimmed_vid_1_dedup.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs(os.path.dirname(output_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

tracks = []

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

    # update + deduplicate
    tracks = update_tracks(tracks, det_boxes, iou_thresh=0.3, max_missed=2)
    tracks = deduplicate_tracks(tracks, iou_thresh=0.7)

    # draw
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"id {t.id}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved tracked (deduplicated) video to: {output_path}")