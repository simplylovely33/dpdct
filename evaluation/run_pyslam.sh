#! /bin/bash

image_folder=$1
folder_name=$2

python preprocess_pyslam.py --input_dir "$image_folder" --scene_name "$folder_name"

config_path=/home/wsco1/yyz/pyslam/config.yaml
sed -i "127s#\(base_path: ./data/videos/\).*#\1${folder_name}#" "$config_path"

cd /home/wsco1/yyz/pyslam || exit
export DISPLAY=:1
source /home/wsco/anaconda3/etc/profile.d/conda.sh
conda activate pyslam
python main_slam.py

save_path=/home/wsco1/yyz/dataset/endo/c3vd/$folder_name/slam_result
mkdir -p "$save_path"
cd /home/wsco1/yyz/pyslam/results || exit
mv metrics_*/ slam_state/ "$save_path"