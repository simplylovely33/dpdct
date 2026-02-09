# recommend checking the max_workers in classifies.py and other parallel processing scripts

DATA_ROOT=$1
#python repair_first_scene.py
python classifies.py --dataset_path $DATA_ROOT
python undisort.py --root_path $DATA_ROOT
#python video.py
python resize.py --root_path $DATA_ROOT --target_width 640 --target_height 512