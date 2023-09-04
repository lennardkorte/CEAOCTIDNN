
import os
import wandb
# import numpy as np
from logger import Logger

class Wandb():
    @staticmethod
    def init(cv, id, config):
        if config['enable_wandb']:
            if config['wandb'] is not None :
                os.environ['WANDB_API_KEY'] = config['wandb']
            else:
                print('W&B API key is missing as an argument. See docs for more information.')
                exit()
                
            Logger.print_section_line()
            wandb.init(
                project='IDDATDLOCT',
                entity='lennardkorte',
                group=config['group'],
                id=id,
                resume="allow",
                name=config['name'] + '_cv_' + str(cv+1),
                reinit=True,
                dir=os.getenv("WANDB_DIR", './data'))
    
    @staticmethod   
    def wandb_log(mean_loss, metrics, class_labels, epoch, optimizer, prefix, config):
        if config['enable_wandb']:
            dict = {
                prefix + ' Loss': mean_loss,
                prefix + ' Accuracy': metrics[0],
                prefix + ' Sensitivity': metrics[1],
                prefix + ' Specificity': metrics[2],
                prefix + ' F1': metrics[4],
                prefix + ' BACC': metrics[7],
                prefix + ' MCC': metrics[8],
                prefix + ' PPV': metrics[9],
            }
            
            '''
            # TODO: Bug? Only with confusion matrix it takes several seconds to pass the wandb.log
            if class_labels is not None:
                conf_matrix = wandb.plot.confusion_matrix(
                    y_true = eval.targets,
                    preds = np.argmax(eval.predictions, 1),
                    class_names = class_labels,
                    title = prefix + ' Confusion Matrix')
                dict[prefix + ' Confusion Matrix'] = conf_matrix
            '''
            
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