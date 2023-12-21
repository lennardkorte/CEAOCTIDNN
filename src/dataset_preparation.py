
import os
import h5py
import numpy as np
import random
import warnings
import itertools
import torch
from sklearn.utils.class_weight import compute_class_weight
from functools import reduce

DATA_PATH_ORIGINAL = './data/h5s/original/'
    
class DatasetPreparation():
    def __init__(self, config):        
        self.all_files_paths = self._get_file_paths(DATA_PATH_ORIGINAL + config['c2_or_c3'] + config['cart_or_pol'] + '/')
        self.label_data, self.label_classes = self._get_prepared_labels(config)
        self.class_weights = self._get_class_weights()
        self.test_ind, self.train_ind_subdivision = self.set_test_set_manually() if config['define_sets_manually'] else self._set_test_set_by_percentage(config)

    def _get_file_paths(self, walking_dir):
        # Get File and Directory Paths
        all_files_paths = []
        for dirpath, _, filenames in os.walk(walking_dir):
            for filename in filenames:
                if '.h5' in filename:
                    path_str = os.path.join(dirpath, filename).__str__()
                    all_files_paths.append(path_str)
        
        assert all_files_paths, "No training data found."
        return all_files_paths
            
    def _get_prepared_labels(self, config):

        LABEL_CLASSES_C3 = ['No Plaque', 'Calcified Plaque', 'Lipid/fibrous Plaque']
        LABEL_CLASSES_C2 = ['No Plaque', 'Plaque']
    
        # extract Labels for Data Set
        label_paths = [path.replace('/orig/', '/labels/').replace('/pol/', '/labels/').replace('/im_', '/t_').replace('ims/', 'tars/') for path in self.all_files_paths]
        label_data = [int(h5py.File(path, 'r')['targets'][0]) for path in label_paths]
        
        # reassign labels 1,2,6 etc. to ind
        if '_c2' in config['c2_or_c3']:
            label_classes = LABEL_CLASSES_C2
        elif '_c3' in config['c2_or_c3']:
            label_c3_dict = {6: 0, 1: 1, 2: 2}
            label_data = [label_c3_dict[t] for t in label_data]
            label_classes = LABEL_CLASSES_C3
        return np.asarray(label_data), label_classes

    def _get_class_weights(self):
        # Determine and list all classes, counts and class-weights
        classes, counts = np.unique(self.label_data, return_counts=True)
        class_weights_comp = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.label_data)
        print('\nClasses:      ', classes)
        print('Counts:       ', counts)
        print('Class weights:', class_weights_comp)
        return torch.tensor(class_weights_comp, dtype=torch.float)
    
    # Get Indices by percentage for test
    def _set_test_set_by_percentage(self, config):
        number_images = len(self.all_files_paths)
        all_image_indices = list(range(number_images))

        if config['deterministic_training']:
            random.Random(18).shuffle(all_image_indices)
        else:
            random.shuffle(all_image_indices)
        
        split_point = int((number_images / 100) * config['set_percentage_cv'])
        
        test_ind = all_image_indices[split_point:]
        set_cv_ind = all_image_indices[:split_point]
        
        length = len(set_cv_ind)
        if config['num_cv'] >= 4:
            train_ind_subdivision = [set_cv_ind[i*length // config['num_cv']: (i+1)*length // config['num_cv']] for i in range(config['num_cv'])]
        elif config['num_cv'] == 1:
            split_point = int((length / 100) * config['set_percntage_val'])
            train_ind_subdivision = [set_cv_ind[:split_point], set_cv_ind[split_point:]]
        else:
            warnings.warn('Set num_cv to zero or min. 4 for reasonable results.') # TODO: warining or assert?
            exit()
            
        return test_ind, train_ind_subdivision

    # Separate file indices into sets and set Index for Folds, then store in cross validation array
    def get_train_valid_ind(self, cv):
        removed_validation_ind_set = self.train_ind_subdivision[:cv] + self.train_ind_subdivision[cv+1:]
        train_ind_for_cv = list(itertools.chain.from_iterable(removed_validation_ind_set))
        
        return self.train_ind_subdivision[cv], train_ind_for_cv
                 
    def set_test_set_manually(self):
        # Function only used if define_sets_manually = True

        SET_NAMES_ALL = ['set1',  'set2',  'set3',  'set4',  'set5',  'set6',  'set10', 
                         'set14', 'set15', 'set17', 'set20', 'set23', 'set24', 'set25', 
                         'set26', 'set28', 'set29', 'set30', 'set32', 'set33', 'set35',
                         'set37', 'set38', 'set39', 'set40', 'set42', 'set43', 'set44',
                         'set45', 'set47', 'set48', 'set50', 'set52', 'set54', 'set55',
                         'set57', 'set59', 'set61', 'set62', 'set63', 'set64', 'set65',
                         'set68', 'set69', 'set70', 'set72', 'set74', 'set75', 'set76'] # total: 49
        SET_NAMES_TEST = ['set1', 'set2', 'set23', 'set24', 'set37', 'set38', 'set50'] # total: 7
        
        
        # Gets sets without testsets (complement)
        sets_for_cv = [item for item in SET_NAMES_ALL if item not in SET_NAMES_TEST]

        # Divide complement into 7 groups of 6 sets each
        sets_for_cv_grouped = []
        for index in range(7):
            sets_for_cv_grouped.append(sets_for_cv[index::7])
        
        # Define validations manually:
        #sets_for_cv_grouped = []


        
        train_ind_subdivision = []
        for list_of_sets in sets_for_cv_grouped:
            train_ind_cv = []
            for set_name in list_of_sets:
                for index, file_path in enumerate(self.all_files_paths):
                    if set_name + '/' in file_path:
                        train_ind_cv.append(index)
                        
            train_ind_subdivision.append(train_ind_cv)
            
        test_ind = []
        for index, file_path in enumerate(self.all_files_paths):
            for set_name in SET_NAMES_TEST:
                if set_name + '/' in file_path:
                    test_ind.append(index)
                    
        return test_ind, train_ind_subdivision
