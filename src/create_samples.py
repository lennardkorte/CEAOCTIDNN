
import os
import h5py
import math
import numpy as np
import cv2

from pathlib import Path

from config import Config
from da_techniques import DataAugmentationTechniques


def tensor_to_numpy(image_tensor) -> np.ndarray:
    # Convert back to Numpy
    image_np = image_tensor.cpu().detach().numpy()
    image_np = image_np[:1].squeeze()
    #image_np = image_np * 5
    image_np = image_np * 65535.0 # here because rescaling is active in preprocessing -> scales to a value between 0 and 1
    image_np = image_np.astype(np.uint16)
    return image_np
    
def convert(source_path:Path, path_shape_type_set:list, config:Config) -> None:
    # Data Augmentation
    if '/' + config['cart_or_pol'] + '/' in str(source_path) and config['c2_or_c3'] in str(source_path):
        # Read data from .h5 file
        file_r = h5py.File(source_path, 'r')
        file_keys = list(file_r.keys())
        dset = file_r[file_keys[0]]
        original_image = np.array(dset)
    
        image_tensor = DataAugmentationTechniques.transform_image(original_image, config['transformations_chosen'], True)
        image_da = tensor_to_numpy(image_tensor)
        
        # Create Directory to store file, if not existent yet
        _, filename = os.path.split(source_path)
        
        # Create filename
        filename_with_directory = os.path.split(os.path.split(source_path)[0])[1] + '_' + os.path.splitext(filename)[0] + '.png'
        
        # Store .jpg versions of original images
        file_directory_jpgs = config.save_path / 'samples_jpg' / 'original'
        os.makedirs(file_directory_jpgs, exist_ok=True)
        destination_path_original = file_directory_jpgs / filename_with_directory
        cv2.imwrite(str(destination_path_original), original_image)
        
        # Store .jpg version of modified image
        if config['transformations_chosen'] != []:
            file_directory_da = config.save_path / 'samples_jpg' / 'augmented'
            os.makedirs(file_directory_da, exist_ok=True)
            destination_path_da = file_directory_da / filename_with_directory
            cv2.imwrite(str(destination_path_da), image_da)
        
        path_shape_type_set.add(('Key: ' + file_keys[0], 'Shape: ' + str(dset.shape), 'DType: ' + str(dset.dtype)))

def create_samples(config: Config):
    raw_images_dir = './data/h5s/original'
    # Calculate how many .h5 files are in directory
    file_count = sum(len([filename for filename in filenames if (filename.endswith('.h5') and '/orig/' in dirpath)]) for dirpath, _, filenames in os.walk(raw_images_dir))
    print("Total image number: " + str(file_count))
    
    # Calculate how many files to convert when dividing by input
    every_x_image = math.ceil(file_count/int(20))
    # Convert every x h5 image in directory to jpg
    i=0
    path_shape_type_set = set()
    for dirpath, _, filenames in os.walk(raw_images_dir):
        for filename in filenames:
            included_folders = ['/orig/']
            if filename.endswith('.h5') and any(folder in dirpath for folder in included_folders):
                i+=1
                if i % (every_x_image) == 0:
                    source_path = os.path.join(dirpath, filename)
                    convert(Path(source_path), path_shape_type_set, config)
                    convert(Path(source_path.replace('/orig/', '/pol/')), path_shape_type_set, config)
                
    print('Shapes and Types: ')
    for element in path_shape_type_set:
        print(element)
        
