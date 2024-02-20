
import os
import torch
from glob import glob
from pathlib import Path
from logger import Logger
from config import Config
import torch.optim as optim
from torch._C import device
from architectures.builder import architecture_builder
from torch.nn.parallel.data_parallel import DataParallel
             
class Checkpoint():    
    def __init__(self, name:str, save_path_cv:Path, device:device, config:Config, cv:int):   

        self.model = self.get_new_model(device, config, cv)
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = self.get_new_optimizer(self.model, config)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
        self.epoch = 1
        self.epoch_impr_and_no_overfitting = 1
        self.eval_valid_best_metrics = None
        if config["enable_wandb"]:
            self.wandb_id = Logger.get_id()
        
        for checkpoint_path in glob(str(save_path_cv / '*.pt')):
            if name in checkpoint_path:

                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.epoch = checkpoint['epoch']
                self.epoch_impr_and_no_overfitting = checkpoint['epoch_impr_and_no_overfitting']
                self.eval_valid_best_metrics = checkpoint['eval_valid_best_metrics']
                if config["enable_wandb"]:
                    self.wandb_id = checkpoint['wandb_id']
    
    @staticmethod
    def delete_checkpoint(name, epoch, save_path:Path) -> None:
        if os.path.isfile(save_path / (name + '_at_epoch_' + str(epoch) + '.pt')):
            os.remove(save_path / (name + '_at_epoch_' + str(epoch) + '.pt'))
    
    def save_checkpoint(self, name:str, epoch:int, epoch_impr_and_no_overfitting:int, eval_valid_best_metrics:dict, save_path_cv:Path, config:Config) -> None:
        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'epoch_impr_and_no_overfitting': epoch_impr_and_no_overfitting,
            'eval_valid_best_metrics': eval_valid_best_metrics
        }
        if config["enable_wandb"]:
            state.update({'wandb_id': self.wandb_id})
        torch.save(state, save_path_cv / (name + '_at_epoch_' + str(epoch) + '.pt'))

        self.eval_valid_best_metrics = eval_valid_best_metrics
    
    @classmethod
    def get_new_model(cls, device:device, config:Config, cv:int) -> DataParallel:
        model = architecture_builder(config['architecture'], config['arch_version'], config['dropout'], config['num_classes'], config['autenc_depth'])
        
        # Load the pretrained ResNet model from a ".pt" file
        save_path_ae_cv = Path('./data/train_and_test', config['encoder_group'], config['encoder_name'], ('cv_' + str(cv)))
        for path in glob(str(save_path_ae_cv / '*.pt')):
            if "checkpoint_best" in path:
                checkpoint_path = path
                checkpoint = torch.load(checkpoint_path)
                # TODO only load layers that are the same in encoder
                model.load_state_dict(checkpoint['model_state_dict'])

        '''
        #if len(config['gpus']) > 1: #TODO
        #    model = nn.DataParallel(model)
                
        
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

        
        model.to(device)
        
        return model
    
    @staticmethod    
    def get_new_optimizer(model:DataParallel, config:dict) -> None:      
        if config['optimizer'] == 'AdamW':
            return optim.AdamW(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'SGD':
            return optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
        else:
            return optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])



