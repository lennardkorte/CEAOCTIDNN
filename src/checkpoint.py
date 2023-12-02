
import os
from pathlib import Path
import torch
import copy

import torch.optim as optim
import torch.nn as nn

from glob import glob
from torch._C import device
from collections import namedtuple
from torch.nn.parallel.data_parallel import DataParallel

from eval import Eval
from utils_wandb import Wandb
from config import Config

from models.model_resnet_autenc import ResNet18, create_autoenc_resnet18
from models.model_unet1 import UNetClassifier1, load_unet1_with_classifier_weights
from models.model_unet2 import UNetClassifier2, load_unet2_with_classifier_weights
             
class Checkpoint():
    ''' This class represents a checkpoint of the training process, where the current status is stored.
    All training and testing processes use the model in it.
    '''
    
    model_map = {
        'ResNet18': ResNet18,
        'ResNet18AutEnc': create_autoenc_resnet18,
        'UNetClassifier1': UNetClassifier1,
        'load_unet1_with_classifier_weights': load_unet1_with_classifier_weights,
        'UNetClassifier2': UNetClassifier2,
        'load_unet2_with_classifier_weights': load_unet2_with_classifier_weights,
    }
    
    def __init__(self, name:str, save_path_cv:Path, device:device, config:Config):
        ''' Creates the first checkpoint of the training process.
        Stores most important components of the training process, e.g.: model, optimizer, wandb_id, etc.
        Note: DataLoaders are not stored in checkpoints. In deterministic (incl. shuffling).

        Arguments:
            self: The Checkpoint object itself.
            save_path_cv: Location where this CV round is stored.
            device: Hardware to optimize on.
            config: Configuration set by the user.
        Return:
            The class constructor returns a "Checkpoint" object.
        '''
        
        self.model = self.get_new_model(device, config)
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = self.get_new_optimizer(self.model, config)
        self.start_epoch = 1
        if config["enable_wandb"]:
            self.wandb_id = Wandb.get_id()
        self.eval_valid = None
        self.eval_valid_best = None
        
        # Load existing checkpoint
        model_found = False
        for checkpoint_path in glob(str(save_path_cv / '*.pt')):
            if name in checkpoint_path:
                model_found = True
                
                checkpoint = torch.load(checkpoint_path)
                
                self.model.load_state_dict(checkpoint['Model'])
                self.optimizer.load_state_dict(checkpoint['Optimizer'])
                self.scaler.load_state_dict(checkpoint['Scaler'])
                self.start_epoch = checkpoint['Epoch'] + 1
                if config["enable_wandb"]:
                    self.wandb_id = checkpoint['Wandb_ID']
                self.eval_valid = namedtuple("eval_valid", checkpoint['eval_valid'].keys())(*checkpoint['eval_valid'].values())
                self.eval_valid_best = namedtuple("eval_valid_best", checkpoint['eval_valid_best'].keys())(*checkpoint['eval_valid_best'].values())
                break

        if not model_found:
            print(f'No Model with "{name}" found. Train from first epoch.')
    
    def update_eval_valid(self, eval_valid:Eval, config) -> None:
        
        if self.eval_valid_best is None:
            self.eval_valid_best = copy.deepcopy(eval_valid)
            
        if self.eval_valid is not None:
            if config["auto_encoder"]:
                if eval_valid.mean_loss < self.eval_valid_best.mean_loss:
                    self.eval_valid_best = copy.deepcopy(eval_valid)
            else:
                if eval_valid.metrics[5] > self.eval_valid_best.metrics[5]:
                    self.eval_valid_best = copy.deepcopy(eval_valid)
        
        self.eval_valid = eval_valid
    
    def save_checkpoint(self, name:str, epoch:int, save_path_cv:Path, config:Config) -> None:
        ''' Saves the current model under specified name.

        Arguments:
            self: The Checkpoint object.
            name: Name of checkpoint file to save.
            epoch: The epoch number which the checkpoint belongs to.
        Return:
            This Method has nothing to return.
        '''

        state = {
            'Epoch': epoch,
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scaler': self.scaler.state_dict(),
            'eval_valid': vars(self.eval_valid),
            'eval_valid_best': vars(self.eval_valid_best)
        }
        if config["enable_wandb"]:
            state.update({'Wandb_ID': self.wandb_id})

        torch.save(state, save_path_cv / (name + '_at_epoch_' + str(epoch) + '.pt'))
    
    @classmethod
    def get_new_model(cls, device:device, config:Config) -> DataParallel:
        ''' Gets the right model specified in the configurations.

        Arguments:
            self: The Checkpoint class.
            device: The device to fit the model on.
            config: The application configuration.
        Return:
            Returns the right parallelized model fitting to the hardware.
        '''
        
        if config['model_type'] not in cls.model_map:
            raise ValueError('Name of model "{0}" unknown.'.format(config['model_type']))
        else:
            model = cls.model_map[config['model_type']](config)
            
        if len(config['gpus']) > 1:
            model = nn.DataParallel(model)
                
        model.to(device)
        
        return model
    
    @staticmethod    
    def get_new_optimizer(model:DataParallel, config:dict) -> None:
        ''' Gets the right optimizer specified in the configurations.
        
        Arguments:
            model: The model which is created and trained.
            config: Config File defining optimizer settings.
        Return:
            This Method has nothing to return. # TODO: this is wrong
        '''
        
        if config['optimizer'] == 'AdamW':
            return optim.AdamW(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'SGD':
            return optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
        else:
            return optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    @staticmethod
    def delete_checkpoint(name, epoch, save_path:Path) -> None:
        ''' Deletes the specified checkpoint from a given directory.

        Arguments:
            name: Filename beginning of the checkpoint to delete.
            epoch: The epoch the checkpoint belongs to as part of the filename.
            save_path: Location where the checkpoint is stored.
        Return:
            The Method has nothing to return.
        '''
        
        if os.path.isfile(save_path / (name + '_at_epoch_' + str(epoch) + '.pt')):
            os.remove(save_path / (name + '_at_epoch_' + str(epoch) + '.pt'))

