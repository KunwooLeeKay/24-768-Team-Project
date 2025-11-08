import os, json
import cv2
import numpy as np
from tqdm import tqdm
try:
    from openpose import pyopenpose as op
except:
    pass

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def run_openpose_all_keypoints(video_path, output_video_path, output_json_path, stride=1):


    params = {
        "model_folder": "./openpose/models/",
        "model_pose": "BODY_25",
        "net_resolution": "-1x368",
        "render_threshold": 0.1,
    }

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    results = []

    for frame_idx in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            datum = op.Datum()
            datum.cvInputData = frame
            datums = op.VectorDatum()
            datums.append(datum)

            opWrapper.emplaceAndPop(datums)
            output_image = datums[0].cvOutputData
            keypoints = datums[0].poseKeypoints  # shape: (num_people, 25, 3) or None

            frame_entry = {"frame": frame_idx, "people": []}
            if keypoints is not None and keypoints.ndim == 3:
                num_people = keypoints.shape[0]
                for p in range(num_people):
                    person = keypoints[p]  # (25, 3) -> x, y, score
                    frame_entry["people"].append({
                        "person_idx": p,
                        "keypoints": person[:, :2].tolist(),
                        "keypoint_scores": person[:, 2].tolist()
                    })

            results.append(frame_entry)
            out.write(output_image if output_image is not None else frame)
        else:
            # Keep the timeline consistent even when skipping processing
            out.write(frame)

    cap.release()
    out.release()

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("✅ Saved:", output_video_path)
    print("✅ Saved:", output_json_path)
    return results