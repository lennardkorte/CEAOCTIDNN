
import os
import time
import sys
import torch
import random
import json

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import OrderedDict

from utils_wandb import Wandb
from data_loaders import Dataloaders


class Utils():
    ''' 
    '''
    
    @staticmethod
    def config_torch_and_cuda(config):
        if config['gpus'] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
            
        print("Indices of devices to use:          ", os.environ["CUDA_VISIBLE_DEVICES"])
        
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
                
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def check_cv_status(cv, save_path:Path):
        # check if Crossfold is already partly trained
        save_path_cv = save_path / ('cv_' + str(cv + 1))
        os.makedirs(save_path_cv, exist_ok=True)
        
        cv_done = False
        for i in os.listdir(save_path_cv):
            if os.path.isfile(os.path.join(save_path_cv,i)) and 'checkpoint_latest' in i:
                cv_done = True
            
        return cv_done, save_path_cv
    
    @staticmethod
    def get_new_lossfunction(class_weights, device, loss_function):
        if loss_function == 'SmoothL1Loss':
            return nn.SmoothL1Loss()
        elif loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss(weight=class_weights.to(device))
        elif loss_function == 'L1Loss':
            return nn.L1Loss()
        else:
            return nn.MSELoss()
    
    @staticmethod
    def train_one_epoch(model, device, loss_function, scaler, optimizer, config):
        start_it_epoch = time.time()
        model.train()
        print('')

        learning_rate_sum = 0
        loss_sum = 0

        for j, (inputs, labels) in enumerate(Dataloaders.trainInd):
            print('Current training batch: ', j + 1, end='\r')
            
            inputs = inputs.to(device)
            labels = labels.squeeze().type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():

                    outputs = model(inputs)

                    if config["auto_encoder"]:
                        
                        # TODO
                        input_scaled = inputs
                        # first_channel = inputs[:,:1,:,:]
                        # input_scaled = F.interpolate(first_channel, size=(32, 32), mode='bilinear', align_corners=False)
                        
                        loss = loss_function(outputs, input_scaled)
                    else:
                        loss = loss_function(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                learning_rate_sum += optimizer.param_groups[0]['lr']
                loss_sum += loss
                
            sys.stdout.write("\033[K")

        if config["enable_wandb"]:
            Wandb.wandb_train_one_epoch(loss / (j + 1), learning_rate_sum / (j + 1), config)
        
        return time.time() - start_it_epoch

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