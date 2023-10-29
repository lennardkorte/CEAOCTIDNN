
import os
from logger import Logger
import wandb

class Wandb():
    @staticmethod
    def init(cv, id, config):
        if config['enable_wandb']: # TODO: remove these from the file
            #import wandb # Import here because wandb checks version online when imported -> unwanted behavious when wandb is turned off
            if config['wandb'] is not None :
                os.environ['WANDB_API_KEY'] = config['wandb']
            else:
                print('W&B is missing API key argument from this program. See docs for more information.')
                exit()
                
            Logger.print_section_line()
            wandb.init(
                project=config['wb_project'],
                entity='lennardkorte', # TODO: data privacy
                group=config['group'],
                id=id,
                resume="allow",
                name=config['name'] + '_cv_' + str(cv+1),
                reinit=True,
                dir=os.getenv("WANDB_DIR", config.save_path))
    
    @staticmethod   
    def get_id():
        return wandb.util.generate_id()

    @staticmethod   
    def wandb_log(eval_test, class_labels, epoch, optimizer, prefix, config):
        if config['enable_wandb']:
            dict = {
                prefix + ' Loss': eval_test.mean_loss
            }
            if not config['auto_encoder']:
                dict.update({
                    prefix + ' Accuracy': eval_test.metrics[0],
                    prefix + ' Sensitivity': eval_test.metrics[1],
                    prefix + ' Specificity': eval_test.metrics[2],
                    prefix + ' F1': eval_test.metrics[4],
                    prefix + ' BACC': eval_test.metrics[7],
                    prefix + ' MCC': eval_test.metrics[8],
                    prefix + ' PPV': eval_test.metrics[9],
                })
            
            if optimizer is not None:
                dict[prefix + ' Learning Rate'] = optimizer.param_groups[0]['lr']
                
            if epoch != 0: # Do not add when sending sending final testresults of cv
                dict[prefix + ' Epoch'] = epoch
                
            wandb.log(dict)
    
    @staticmethod
    def wandb_train_one_epoch(loss, optimizer, config):
        if config['enable_wandb']:  
            wandb.log({'Loss Training': loss})
            wandb.log({'Learning Rate': optimizer.param_groups[0]['lr']})