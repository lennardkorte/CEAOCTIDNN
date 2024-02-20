
import os
import torch
import random
import json

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from pathlib import Path
from collections import OrderedDict

from logger import Logger
from dataset import OCT_Dataset
from data_loaders import Dataloaders

import torch
import os
from torchvision.utils import save_image

from pathlib import Path

class Utils():
    ''' 
    # TODO: remove utils
    '''
    
    @staticmethod
    def config_torch_and_cuda(config):
        #if config['gpus'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0,1' #config['gpus']
           
        print("Indices of devices to use:", os.environ["CUDA_VISIBLE_DEVICES"])
        
        # Set location where torch stores its models
        os.environ['TORCH_HOME'] = './data/torch_pretrained_models'
        
        torch.backends.cudnn.enabled = config['use_cuda']
        
        # use deterministic training?
        if config['deterministic_training']:
            seed = 18
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if config['deterministic_batching']:
                np.random.seed(seed)
                random.seed(seed)
        
        try:
            # Check if GPU is available
            assert torch.cuda.is_available(), "No GPU available"
            if config['gpu'] == 1:
                assert 2 <= torch.cuda.device_count(), "Second GPU not available"
            print("Chosen GPU is available")
            print(torch.cuda.get_device_name(config['gpu']))

        except AssertionError as error:
            # Handle the assertion error if no GPU is available
            print(f"Assertion Error: {error}")
            raise SystemExit("Program terminated due to lack of GPU.")
        
        return torch.device(config['gpu'])
    
    @staticmethod
    def train_one_epoch(model, device, scaler, optimizer, config, class_weights):

        if config['auto_encoder']:
            loss_function = nn.MSELoss()
        else:
            loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        model.train()
        
        learning_rate_sum = 0
        loss_sum = 0
        for j, (inputs, labels) in tqdm(enumerate(Dataloaders.training), total=len(Dataloaders.training), desc='Training batches', leave=False):
            
            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):

                # Runs the forward pass under autocast.
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)

                    if config["auto_encoder"]:
                        
                        # TODO: remove if not needed
                        # first_channel = inputs[:,:1,:,:]
                        # inputs = F.interpolate(first_channel, size=(32, 32), mode='bilinear', align_corners=False)
                        
                        loss_all = loss_function(outputs, inputs)
                    else:
                        loss_all = loss_function(outputs, labels)
                
                # Scales loss and calls backward()
                # to create scaled gradients.
                #loss_all =  torch.mean(loss_each)
                scaler.scale(loss_all).backward()
                
                # Unscales gradients and calls
                # or skips optimizer.step().
                scaler.step(optimizer)
                
                # Updates the scale for next iteration.
                scaler.update()

                learning_rate_sum += optimizer.param_groups[0]['lr']
                loss_sum += loss_all
                
        #"mean_loss training": float(np.float32(loss_sum / (j + 1))) #TODO: cannot be 
        Logger.add({"mean_learning_rate": learning_rate_sum / (j + 1)}, "train_set")

    @staticmethod
    def read_json(file_name):
        file_name = Path(file_name)
        with file_name.open('rt') as handle:
            return json.load(handle, object_hook=OrderedDict)

    @staticmethod
    def write_json(content, file_name):
        file_name = Path(file_name)
        with file_name.open('wt') as handle:
            json.dump(content, handle, indent=4, sort_keys=False)

def data_loader_sampling(cust_data, path, transf_chosen, dataset_no, sample_no):
    os.makedirs(path / 'sample_images', exist_ok=True)
    sample_ind = random.sample(range(len(cust_data.label_data)), sample_no)
    
    dataset_prepro = OCT_Dataset(sample_ind, cust_data.label_data, cust_data.all_files_paths, False, False, dataset_no)
    dataset_prepro_and_aug = OCT_Dataset(sample_ind, cust_data.label_data, cust_data.all_files_paths, True, False, dataset_no)

    for i in range(sample_no):
        image_prepro, image_prepro_label = dataset_prepro[i]
        image_prepro_and_aug, image_prepro_and_aug_label = dataset_prepro_and_aug[i]

        image_prepro_rescaled = image_prepro
        image_prepro_and_aug_rescaled = image_prepro_and_aug
        
        save_image(image_prepro_rescaled, path / f'sample_images/prepro_{i+1}.png')
        save_image(image_prepro_and_aug_rescaled, path / f'sample_images/prepro_and_aug_{i+1}.png')



    