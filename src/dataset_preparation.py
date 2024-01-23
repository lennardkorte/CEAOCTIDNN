
import os
import h5py
import numpy as np
import random
import warnings
import itertools
from utils import comp_class_weights
#from functools import reduce

DATA_PATH_ORIGINAL = './data/h5s/original/'
    
class DatasetPreparation():
    def __init__(self, config):        
        self.all_files_paths = self._get_file_paths(DATA_PATH_ORIGINAL + config['c2_or_c3'] + config['cart_or_pol'] + '/')
        self.label_data, self.label_classes = self._get_prepared_labels(config)
        self.test_ind, self.train_ind_subdivision = self._set_test_set_manually() if config['define_sets_manually'] else self._set_test_set_by_percentage(config)

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
                 
    def _set_test_set_manually(self):
        # Function only used if define_sets_manually = True
        
        set_names_all = ['set1',  'set2',  'set3',  'set4',  'set5',  'set6',  'set10', 
                         'set14', 'set15', 'set17', 'set20', 'set23', 'set24', 'set25', 
                         'set26', 'set28', 'set29', 'set30', 'set32', 'set33', 'set35',
                         'set37', 'set38', 'set39', 'set40', 'set42', 'set43', 'set44',
                         'set45', 'set47', 'set48', 'set50', 'set52', 'set54', 'set55',
                         'set57', 'set59', 'set61', 'set62', 'set63', 'set64', 'set65',
                         'set68', 'set69', 'set70', 'set72', 'set74', 'set75', 'set76'] # total: 49
        '''
        set_names_test = ['set1', 'set2', 'set23', 'set24', 'set37', 'set38', 'set50'] # total: 7

        # Gets sets without testsets (complement)
        sets_for_cv = [item for item in set_names_all if item not in SET_NAMES_TEST]
        # Divide complement into 7 groups of 6 sets each
        sets_for_cv_grouped = []
        for index in range(7):
            sets_for_cv_grouped.append(sets_for_cv[index::7])
        '''
        
        # Define validations manually

        random.seed(5)
        random.shuffle(set_names_all)

        sets_for_cv_grouped = []
        num_cvs = 5
        for index in range(num_cvs):
            sets_for_cv_grouped.append(set_names_all[index::num_cvs])
        
        print('pullback sets (number of images in folds, positive weights):')
        train_ind_subdivision = []
        for list_of_sets in sets_for_cv_grouped:
            train_ind_cv = []
            for set_name in list_of_sets:
                image_indices = []
                for index, file_path in enumerate(self.all_files_paths):
                    if set_name + '/' in file_path:
                        image_indices.append(index)

                max_num_per_set = 1000 # TODO try if improves
                
                # Separating indices based on the values they map to
                indices_mapping_to_one = [index for index in image_indices if self.label_data[index] == 1]
                indices_mapping_to_zero = [index for index in image_indices if self.label_data[index] == 0]

                # Selecting random subsets
                selected_ones = random.sample(indices_mapping_to_one, min(len(indices_mapping_to_one), max_num_per_set // 2))
                selected_zeros = random.sample(indices_mapping_to_zero, min(len(indices_mapping_to_zero), max_num_per_set // 2))

                # Merging the two selections
                selected_indices = selected_ones + selected_zeros
                random.shuffle(selected_indices)  # Optional: shuffle the merged list

                #train_ind = random.sample(train_ind, max_num_per_set)
                #random.shuffle(train_ind)
                    
                label_sum_set = sum(self.label_data[selected_indices])
                indices_num_set = len(selected_indices)
                print(set_name, indices_num_set, label_sum_set/indices_num_set)

                train_ind_cv.extend(selected_indices)
                        
            train_ind_subdivision.append(train_ind_cv)

        

        print('Cross-validation (number of images in folds, positive weights):')
        num_pos_labels_total = 0
        for indices_cv in train_ind_subdivision:
            label_sum_cv = sum(self.label_data[indices_cv])
            num_pos_labels_total += label_sum_cv
            indices_num_cv = len(indices_cv)
            print(indices_num_cv, label_sum_cv/indices_num_cv)
            num_labels_total = sum(len(sublist) for sublist in train_ind_subdivision)
        print(num_labels_total, num_pos_labels_total/num_labels_total)  

        test_ind = train_ind_subdivision.pop(1)      
        
        all_labels = []
        for subset in train_ind_subdivision:
            all_labels.extend(self.label_data[subset])
        # Determine and list all classes, counts and class-weights
            
        comp_class_weights(all_labels)
                    
        return test_ind, train_ind_subdivision
