import cv2
import os

video_path = "input/input_vid.mp4"

# from 7:00 - 7:40, 7:41 - 8:15, 8:22 -8:44, ...

vid = cv2.VideoCapture(video_path)
fps = vid.get(cv2.CAP_PROP_FPS)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# segment to 7:00 to 7:40
start_time = 7 * 60
end_time = start_time + 40
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

out = cv2.VideoWriter("output/trimmed_vid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for frame_num in range(start_frame, end_frame):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = vid.read()
    if not ret:
        break
    out.write(frame)

vid.release()
out.release()