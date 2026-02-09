import os
import cv2
import shutil
import toolkit as utils
from tqdm import tqdm

def preprocess_c3vd(input_dir):
    for folder in ['color', 'occlusion', 'depth', 'normals', 'flow']:
        os.makedirs(os.path.join(input_dir, folder), exist_ok=True)
    files = os.listdir(input_dir)
    for file in tqdm(files, desc="Working on : ", unit="file"):
        if not file.endswith(('.png', '.tiff')):
            continue
        base, ext = os.path.splitext(file)
        type_part = base.split('_')[-1] if '_' in base else ''
        if ext == '.png':
            if type_part == 'color':
                shutil.move(os.path.join(input_dir, file), input_dir + '/color/')
            elif type_part == 'occlusion':
                shutil.move(os.path.join(input_dir, file), input_dir + '/occlusion/')
        elif ext == '.tiff':
            if type_part == 'depth':
                shutil.move(os.path.join(input_dir, file), input_dir + '/depth/')
            elif type_part == 'normals':
                shutil.move(os.path.join(input_dir, file), input_dir + '/normals/')
            elif type_part == 'flow':
                shutil.move(os.path.join(input_dir, file), input_dir + '/flow/')

def restore_c3vd(input_dir):
    target_folders = ['rgb', 'occlusion', 'depth', 'normals', 'flow']
    print(f"Start restore directory structure: {input_dir}")
    
    for folder in target_folders:
        folder_path = os.path.join(input_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"Skip: {folder} not exist")
            continue
        files = os.listdir(folder_path)
        if len(files) > 0:
            for file_name in tqdm(files, desc=f"Restoring {folder}", unit="file"):
                src_path = os.path.join(folder_path, file_name)
                dst_path = os.path.join(input_dir, file_name)
                
                if os.path.isfile(src_path):
                    if not os.path.exists(dst_path):
                        shutil.move(src_path, dst_path)
                    else:
                        print(f"Warning: {file_name} already existed in root folder, skip move.")
        try:
            os.rmdir(folder_path)
            print(f"Successfully remove empty folder: {folder}")
        except OSError:
            print(f"Warning: Can't remove folder {folder_path}, it may not be empty.")
    print("Restore directory structure completed!")

def img2video(image_folder, fps=30, image_extensions=None):
    if image_extensions is None:
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    
    images = os.listdir(image_folder)
    images.sort()
    
    if not images:
        print(f"[INFO]: Can't not find any image file in {image_folder}")
        return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    
    height, width, _ = first_image.shape
    
    output_video_path = os.path.dirname(image_folder) + f'/{os.path.basename(image_folder)}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in tqdm(images, desc="Convert img2video", unit="img"):
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        
        if img is not None:
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))
            video_writer.write(img)
        else:
            print(f"[WARN]: Can't read image {image_path}")
    
    video_writer.release()
    print(f"[INFO]: Video saved to {output_video_path}")

if __name__ == "__main__":
    cfg = utils.load_config("configs/config.yaml")['PREPROCESS']
    restore_c3vd(cfg['FOLDER_ROOT'])
    #folders = os.listdir(data_root)
    #img2video(os.path.join(data_root, folders[0], 'color'))
