
import os
import h5py
import math
import numpy as np
import cv2

from pathlib import Path

from config import Config
from da_techniques import DataAugmentationTechniques

def create_samples(config: Config) -> None:
    raw_images_dir = './data/datasets/001/h5s/original/' + config['c2_or_c3'] + config['cart_or_pol'] + '/'
    # Calculate how many .h5 files are in directory
    file_count = sum(len(filenames) for _, _, filenames in os.walk(raw_images_dir))
    print("Total image number: " + str(file_count))
    
    # Calculate how many files to convert when dividing by input
    every_x_image = math.ceil(file_count/int(20))
    file_dir = config.save_path / 'samples_jpg'
    i=0
    path_shape_type_set = set()
    for dirpath, _, filenames in os.walk(raw_images_dir):
        for filename in filenames:
            i+=1
            if not i % (every_x_image):
                source_path = Path(os.path.join(dirpath, filename))
                # Read data from .h5 file
                file_r = h5py.File(source_path, 'r')
                file_keys = list(file_r.keys())
                dset = file_r[file_keys[0]]
                original_image = np.array(dset)

                image_tensor = DataAugmentationTechniques.transform_image(original_image, config['transformations_chosen'], True)
                image_da = image_tensor.cpu().detach().numpy()
                image_da = image_da[:1].squeeze()
                #image_np = image_np * 5
                image_da = image_da * 65535.0 # here because rescaling is active in preprocessing -> scales to a value between 0 and 1
                image_da = image_da.astype(np.uint16)
                
                # Create new file name
                _, filename = os.path.split(source_path)
                filename_new = os.path.split(os.path.split(source_path)[0])[1] + '_' + os.path.splitext(filename)[0] + '.png'
                
                # Store .png versions of original images
                file_directory_orig = file_dir / 'original'
                os.makedirs(file_directory_orig, exist_ok=True)
                cv2.imwrite(str(file_directory_orig / filename_new), original_image)
                
                # Store .png version of modified image
                if config['transformations_chosen'] != []:
                    file_directory_da = file_dir / 'augmented'
                    os.makedirs(file_directory_da, exist_ok=True)
                    cv2.imwrite(str(file_directory_da / filename_new), image_da)
                
                path_shape_type_set.add(('Key: ' + file_keys[0], 'Shape: ' + str(dset.shape), 'DType: ' + str(dset.dtype)))
                
    print('Shapes and Types: ')
    for element in path_shape_type_set:
        print(element)
        
