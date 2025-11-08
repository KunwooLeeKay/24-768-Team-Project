from utils import *
from pdb import set_trace as st 

input_dir = "input"
video_name = "trimmed_vid_1.mp4"
video_path = f"{input_dir}/{video_name}"

output_video_path = f"output/output_openpose_{video_name}"
output_json_path = f"output/keypoints_openpose_{video_name.replace('.mp4', '.json')}"

run_openpose_all_keypoints(video_path, output_video_path, output_json_path, stride=1)

st()