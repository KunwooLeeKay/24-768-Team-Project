LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/lib/x86_64-linux-gnu/libgcc_s.so.1 \
LD_LIBRARY_PATH=~/MBL/openpose/build/lib \
PYTHONPATH=~/MBL/openpose/build/python:~/MBL/openpose/build/python/openpose \
python -m script.player_tracking.run_openpose
