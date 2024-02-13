
import os
import h5py
import numpy as np
import random
import itertools
from utils import comp_class_weights
from logger import Logger
from collections import defaultdict
    
class DatasetPreparation():
    def __init__(self, config):
        match config['dataset_no']:
            case 0:
                # Get File and Directory Paths
                self.all_files_paths = []
                for dirpath, _, filenames in os.walk('./data/datasets/001/h5s/original/' + config['c2_or_c3'] + config['cart_or_pol'] + '/'):
                    for filename in filenames:
                        if '.h5' in filename:
                            path_str = os.path.join(dirpath, filename).__str__()
                            self.all_files_paths.append(path_str)
                        else:
                            assert AssertionError("Oh no")
                
                # extract Labels for Data Set
                label_paths = [path.replace('/orig/', '/labels/').replace('/pol/', '/labels/').replace('/im_', '/t_').replace('ims/', 'tars/') for path in self.all_files_paths]
                self.label_data = np.asarray([int(h5py.File(path, 'r')['targets'][0]) for path in label_paths])
                
                # reassign labels 1,2,6 etc. to class index
                if '_c2' in config['c2_or_c3']:
                    self.label_classes = ['No Plaque', 'Plaque']
                elif '_c3' in config['c2_or_c3']:
                    label_c3_dict = {6: 0, 1: 1, 2: 2}
                    self.label_data = [label_c3_dict[t] for t in self.label_data]
                    self.label_classes = ['No Plaque', 'Calcified Plaque', 'Lipid/fibrous Plaque']        
                
                self.test_ind, self.train_ind_subdivision = self._set_test_set_manually_ds1(config)

            case 1:
                # Generating the lists as required
                class_numbers = {"NORMAL": 0, "CNV": 1, "DME": 2, "DRUSEN": 3}
                all_file_paths = []
                class_indices = []

                # For handling groups and limiting samples
                group_counts = defaultdict(int)

                for root, dirs, files in os.walk('./data/datasets/002/dataset_unzipped/CellData/OCT/'):
                    for file in files:
                        if file.endswith(('.jpeg', '.jpg')):
                            full_path = os.path.join(root, file)
                            all_file_paths.append(full_path)
                            
                            # Extract class name from the path
                            cls_name = os.path.basename(os.path.dirname(full_path))
                            class_idx = class_numbers[cls_name]
                            class_indices.append(class_idx)

                            # Counting files for groups
                            group_name = file.rsplit("-", 1)[0]  # Get group name without the last part
                            group_counts[group_name] += 1

                # Group samples handling: Identify groups and limit samples
                group_sizes = list(group_counts.values())
                mean_size = np.mean(group_sizes)
                std_dev = np.std(group_sizes)
                upper_limit = mean_size + std_dev

                # Adjusting all_file_paths and class_indices based on the group limits
                adjusted_all_file_paths = []
                adjusted_class_indices = []

                for group_name, count in group_counts.items():
                    if count > upper_limit:
                        allowed_count = int(upper_limit)
                    else:
                        allowed_count = count

                    group_files = [path for path in all_file_paths if self._extract_group_name(path) == group_name]
                    group_files = random.sample(group_files, min(len(group_files), allowed_count))

                    # Adjusting lists based on allowed files
                    adjusted_all_file_paths.extend(group_files)
                    for file in group_files:
                        adjusted_class_indices.append(class_indices[all_file_paths.index(file)])

                if config['binary_class']:
                    # For the binary class index list
                    adjusted_class_indices = [0 if x == 0 else 1 for x in adjusted_class_indices]

                # Updating train and test indices for adjusted lists
                train_indices = []
                test_indices = []
                for idx, path in enumerate(adjusted_all_file_paths):
                    first_up = os.path.dirname(path)
                    second_up = os.path.dirname(first_up)
                    sub_dir_name = os.path.basename(second_up)
                    if sub_dir_name == "train":
                        train_indices.append(idx)
                    elif sub_dir_name == "test":
                        test_indices.append(idx)

                # Group train files by their group name
                train_files_groups = defaultdict(list)
                for idx in train_indices:
                    file_path = all_file_paths[idx]
                    group_name = self._extract_group_name(file_path)
                    train_files_groups[group_name].append(idx)
                # Sort groups by their size to try and balance the subsets
                sorted_groups = sorted(train_files_groups.values(), key=len, reverse=True)
                # Divide into 5 subsets
                subsets = [[] for _ in range(config['num_cv'])]
                for group in sorted_groups:
                    # Find the subset with the least number of indices and add the current group to it
                    subset_with_min_length = min(subsets, key=len)
                    subset_with_min_length.extend(group)

                self.all_files_paths = all_file_paths
                self.label_data = adjusted_class_indices
                self.label_classes = ["NORMAL", "CNV", "DME", "DRUSEN"]
                self.test_ind = test_indices
                self.train_ind_subdivision = subsets

                len(all_file_paths), len(class_indices), len(train_indices), len(test_indices), len(adjusted_all_file_paths), len(adjusted_class_indices), len(binary_class_indices)
                comp_class_weights(self.label_data)

            case idx:
                raise AssertionError(f"Dataset with index {idx} not implemented.")
        
    def _extract_group_name(file_path):
        # Extracts the part of the file name before the last "-" (assuming group names are at the start)
        return os.path.basename(file_path).rsplit("-", 1)[0]
    
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
            raise AssertionError("Set num_cv to zero or min. 4 for reasonable results.")
            
        return test_ind, train_ind_subdivision

    # Separate file indices into sets and set Index for Folds, then store in cross validation array
    def get_train_valid_ind(self, cv):
        removed_validation_ind_set = self.train_ind_subdivision[:cv] + self.train_ind_subdivision[cv+1:]
        train_ind_for_cv = list(itertools.chain.from_iterable(removed_validation_ind_set))
        return self.train_ind_subdivision[cv], train_ind_for_cv
                 
    def _set_test_set_manually_ds1(self, config):
    
        # Define subdivision for cross-validation (4x10) and testing (1x9)
        set_names_all = ['set1',  'set2',  'set3',  'set4',  'set5',  'set6',  'set10', 
                         'set14', 'set15', 'set17', 'set20', 'set23', 'set24', 'set25', 
                         'set26', 'set28', 'set29', 'set30', 'set32', 'set33', 'set35',
                         'set37', 'set38', 'set39', 'set40', 'set42', 'set43', 'set44',
                         'set45', 'set47', 'set48', 'set50', 'set52', 'set54', 'set55',
                         'set57', 'set59', 'set61', 'set62', 'set63', 'set64', 'set65',
                         'set68', 'set69', 'set70', 'set72', 'set74', 'set75', 'set76'] # total: 49
        sets_for_cv_grouped = []
        num_cvs = 5
        for index in range(num_cvs):
            sets_for_cv_grouped.append(set_names_all[index::num_cvs])
        
        with Logger(config.save_path / 'dataset_statistics.txt', 'w', terminal=False):
            print('pullback sets (number of images in folds, positive weights):')
            train_ind_subdivision = []
            for list_of_sets in sets_for_cv_grouped:
                train_ind_cv = []
                for set_name in list_of_sets:

                    # create subdivision of indices 
                    image_indices = []
                    for index, file_path in enumerate(self.all_files_paths):
                        if set_name + '/' in file_path:
                            image_indices.append(index)

                    # TODO: The following limits the number of samples for each index in each pullback / set (if > limit, random dropout)
                    limit = 1000 
                    indices_mapping_to_one = [index for index in image_indices if self.label_data[index] == 1]
                    indices_mapping_to_zero = [index for index in image_indices if self.label_data[index] == 0]
                    random.seed(5)
                    random.shuffle(set_names_all)
                    selected_ones = random.sample(indices_mapping_to_one, min(len(indices_mapping_to_one), limit // 2))
                    selected_zeros = random.sample(indices_mapping_to_zero, min(len(indices_mapping_to_zero), limit // 2))
                    selected_indices = selected_ones + selected_zeros
                    print(set_name, len(selected_indices), sum(self.label_data[selected_indices])/len(selected_indices))
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
