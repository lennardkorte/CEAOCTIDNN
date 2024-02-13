
import os

class Wandb():
    @staticmethod
    def init(cv, id, config):
        import wandb

        if config['wandb'] is not None :
            os.environ['WANDB_API_KEY'] = config['wandb']
        else:
            raise AssertionError("W&B is missing API key argument from this program. See docs for more information.")
            
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
        import wandb
        return wandb.util.generate_id()

    @staticmethod   
    def wandb_log(eval_test, class_labels, epoch, optimizer, prefix, config, loss_matr=True):
        import wandb
        dict = {
            prefix + ' Loss': eval_test.mean_loss
        }
        if not config['auto_encoder']:
            dict.update({
                prefix + ' Accuracy': eval_test.metrics[0],
                prefix + ' Sensitivity': eval_test.metrics[1],
                prefix + ' Specificity': eval_test.metrics[2],
                prefix + ' F1': eval_test.metrics[3],
                prefix + ' BACC': eval_test.metrics[4],
                prefix + ' MCC': eval_test.metrics[5],
                prefix + ' PPV': eval_test.metrics[6],
            })
        if config['auto_encoder'] and loss_matr:
            dict.update({
                'loss_TP': eval_test.mse_loss_conf_matr_mean[1,1],
                'loss_TN': eval_test.mse_loss_conf_matr_mean[0,0],
                'loss_FP': eval_test.mse_loss_conf_matr_mean[1,0],
                'loss_FN': eval_test.mse_loss_conf_matr_mean[0,1]
            })
        
        if optimizer is not None:
            dict[prefix + ' Learning Rate'] = optimizer.param_groups[0]['lr']
            
        if epoch != 0: # Do not add when sending sending final testresults of cv
            dict[prefix + ' Epoch'] = epoch
            
        wandb.log(dict)
    
    @staticmethod
    def wandb_train_one_epoch(loss_avg, learning_rate_avg):
        import wandb
        wandb.log({'Loss Training': loss_avg})
        wandb.log({'Learning Rate': learning_rate_avg})