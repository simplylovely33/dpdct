import os
import numpy as np
import utils.toolkit as utils
import utils.toolkit_vis as vis_utils


if __name__ == "__main__":
    device = utils.device_confirmation()
    config = utils.load_config('/home/wsco1/yyz/ssh/dpdct/configs/config.yaml')

    map_setup_json = config['PYSLAM']['map_setup_json']
    map_setup = utils.read_json_file(map_setup_json)
    if map_setup is None:
        print("Failed to read map file")
        exit(1)
    
    save_folder = config['PYSLAM']['output_folder']
    os.makedirs(save_folder, exist_ok=True)
    
    map_setup_json = config['PYSLAM']['map_setup_json']
    map_setup = utils.read_json_file(map_setup_json)
    map_data = map_setup['map']
    map_keyframes = utils.get_specific_keys(map_data['keyframes'],  config['KEYFRAMS_KEYS'])
    map_points = utils.get_specific_keys(map_data['points'], config['POINTS_KEYS'])
    
    #DEBUG: Map visualization
    vis_utils.visualize_slam_map(map_keyframes, map_points)
    vis_utils.extract_keyframs_imgs(
        map_keyframes, 
        input_folder='/home/wsco1/yyz/dataset/C3VD_EndoGSLAM/C3VD/sigmoid_t3_a/color/', 
        output_folder=os.path.join(save_folder, 'kfimg')
    )

    # DEBUG_MODE: save the all the projected depth map
    projections = utils.project_all_keyframes(map_keyframes, map_points, debug=False)
    
   

    a = 1
    

