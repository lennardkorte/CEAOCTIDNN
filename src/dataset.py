
import numpy as np
from torch.utils.data import Dataset
import h5py
import time
import sys
from da_techniques import DataAugmentationTechniques
from torchvision import transforms as T

class IVOCT_Dataset(Dataset):
    def __init__(self, ind_set:list, label_data, all_files_paths, config, for_train=False):
        self.for_train = for_train
        self.indices = ind_set
        self.preload = config['preload']
        self.transformations_chosen = config['transformations_chosen']
        self.datasets = [h5py.File(path, 'r')['raw'] for path in all_files_paths]
        self.label_data = label_data
        
        if self.preload:
            start_time_preload = time.time()
            self.input_data = {}
            for p, dset in enumerate(self.datasets):
                self.input_data[p] = np.squeeze(dset[:])
                if p % 200 == 0:
                    print("Pre-Load Data:", p, "/", len(self.datasets), '- Preload Duration: ', round(time.time() - start_time_preload, 1), 'seconds', end="\r")
        
            sys.stdout.write("\033[K")
            
        self.length = len(self.indices)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # index of current sample
        elem_idx = int(self.indices[idx])
        
        # Get the Input Data
        if self.preload:
            image = self.input_data[elem_idx]
        else:
            image = self.datasets[elem_idx][:]

        image_tensor = DataAugmentationTechniques.transform_image(image, self.transformations_chosen, self.for_train)
        label = self.label_data[elem_idx].astype(np.float32)
        
        return image_tensor, label
