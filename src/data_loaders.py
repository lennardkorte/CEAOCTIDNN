
from torch.utils.data import DataLoader
from dataset import IVOCT_Dataset
from dataset_preparation import DatasetPreparation

class Dataloaders():
    @classmethod
    def setup_data_loader_testset(cls, cust_data:DatasetPreparation, config):
        print('Images for Tests:                   ', len(cust_data.test_ind))
        # For Testing
        num_workers = len(config['gpus']) * 4 # Recommended by Pytorch Docs
        cls.testInd = DataLoader(
            IVOCT_Dataset(cust_data.test_ind, cust_data.label_data, cust_data.all_files_paths, config),
            batch_size = config['batch_size'],
            num_workers = num_workers,
            pin_memory = True)

    @classmethod
    def setup_data_loaders_training(cls, train_ind_for_cv, train_eval_ind_for_cv, valid_ind_for_cv, cust_data, config):
        # TODO: The following print commands do not belong here
        print('Images for training:                ', len(train_ind_for_cv))
        print('Images for testing while training:  ', len(train_eval_ind_for_cv))
        print('Images for validation:              ', len(valid_ind_for_cv))
        print('')

        # TODO: remove the repetition in the following commands, maybe with for loop?
        # for ind_dataset, trainInd in zip():

        num_workers = len(config['gpus']) * 4 # Recommended by Pytorch Docs
        cls.trainInd = DataLoader(
            IVOCT_Dataset(train_ind_for_cv, cust_data.label_data, cust_data.all_files_paths, config, for_train=True),
            batch_size = config['batch_size'],
            shuffle = True,
            num_workers = num_workers,
            pin_memory = True, # When to use pinning?
            drop_last = True)
        
        # For train Evaluation during training
        cls.trainInd_eval = DataLoader(
            IVOCT_Dataset(train_eval_ind_for_cv, cust_data.label_data, cust_data.all_files_paths, config),
            batch_size = config['batch_size'],
            num_workers = num_workers,
            pin_memory = True)
        
        # For val
        cls.valInd = DataLoader(
            IVOCT_Dataset(valid_ind_for_cv, cust_data.label_data, cust_data.all_files_paths, config),
            batch_size = config['batch_size'],
            num_workers = num_workers,
            pin_memory = True)
    
    
