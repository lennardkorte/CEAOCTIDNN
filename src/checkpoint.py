
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

from architectures.builder import architecture_builder
             
class Checkpoint():
    ''' This class represents a checkpoint of the training process, where the current status is stored.
    All training and testing processes use the model in it.
    '''
    
    def __init__(self, name:str, save_path_cv:Path, device:device, config:Config, cv:int):
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
        
        self.model = self.get_new_model(device, config, cv) # TODO: Remove self. for checkpoint (seperation of concerns) -> pipeline should even work when checkpoint loading does not work
        # TODO: Also store best vali epoch no
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = self.get_new_optimizer(self.model, config)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

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
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
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
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'eval_valid': vars(self.eval_valid),
            'eval_valid_best': vars(self.eval_valid_best)
        }
        if config["enable_wandb"]:
            state.update({'Wandb_ID': self.wandb_id})

        torch.save(state, save_path_cv / (name + '_at_epoch_' + str(epoch) + '.pt'))
    
    @classmethod
    def get_new_model(cls, device:device, config:Config, cv:int) -> DataParallel:
        ''' Gets the right model specified in the configurations.

        Arguments:
            self: The Checkpoint class.
            device: The device to fit the model on.
            config: The application configuration.
        Return:
            Returns the right parallelized model fitting to the hardware.
        '''
        
        model = architecture_builder(config['architecture'], config['arch_version'], config['dropout'], config['num_classes'])
        
        # Load the pretrained ResNet model from a ".pt" file
        save_path_ae_cv = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
        for path in glob(str(save_path_ae_cv / '*.pt')):
            if "checkpoint_best" in path:
                checkpoint_path = path

        checkpoint = torch.load(checkpoint_path)
        # TODO only load layers that are the same in encoder
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set the encoder layers to be non-trainable
        for param in encoder.parameters():
            param.requires_grad = False

        # TODO transfer learning image net config['pretrained']
        # TODO: Write why what modules used, i.e. why version 2?
        # TODO: dropout for VGG
        # TODO: add averaging of three output channels
        # TODO: check normalization of GT and generated image images
        # TODO: is there pluspoints for providing code?
        
        '''
        import torch.nn.parallel as parallel

        if args.parallel == 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = parallel.DistributedDataParallel(
                            model.to(args.gpu),
                            device_ids=[args.gpu],
                            output_device=args.gpu
                        )   
        else:
            model = nn.DataParallel(model).cuda()'''

        #if len(config['gpus']) > 1: #TODO
        #    model = nn.DataParallel(model)
                
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

