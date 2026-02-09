#! /bin/bash

DATA_ROOT=$1
#python exextract_archive.py --scared_root $DATA_ROOT
#python extract_rgb_from_video.py --scared_root $DATA_ROOT
#python extract_depth_from_scene.py --scared_root $DATA_ROOT
python extract_pose_from_data.py --scared_root $DATA_ROOT