import os
import csv
import copy
import torch
import argparse
import numpy as np

from eval import Eval
from tqdm import tqdm
from utils import Utils, data_loader_sampling
from config import Config
from logger import Logger
from checkpoint import Checkpoint
from utils_wandb import Wandb
from data_loaders import Dataloaders
from dataset_preparation import DatasetPreparation
from sklearn.utils.class_weight import compute_class_weight

FILE_NAME_TEST_RESULTS = 'test_results.json'
FILE_NAME_TEST_RESULTS_AVERAGE = 'test_results_average.json'

def train_and_eval(config:Config):

    cust_data = DatasetPreparation(config)
    if config['show_samples']:
        data_loader_sampling(cust_data, config.save_path, config['transformations_chosen'], config["dataset_no"], config['sample_no'])

    Dataloaders.setup_data_loader_testset(cust_data, config)
    
    device = Utils.config_torch_and_cuda(config)

    for cv in range(config['num_cv']):

        early_stop = False
        es_counter = 0
        last_best_epoch_current_cv = -1
        if config["auto_encoder"]:
            validation_best_mean_loss_current_cv = 10000000
        else:
            validation_best_metrics_current_cv = [0.0] * 9
        
        save_path_cv = config.save_path / ('cv_' + str(cv + 1))
        os.makedirs(save_path_cv, exist_ok=True)
        cv_done = True if any('test_results' in s for s in os.listdir(save_path_cv)) else False

        valid_ind_for_cv, train_ind_for_cv = cust_data.get_train_valid_ind(cv)
        Dataloaders.setup_data_loaders_training(train_ind_for_cv, train_ind_for_cv[::2], valid_ind_for_cv, cust_data, config)

        cv_labels = cust_data.label_data[train_ind_for_cv]
        class_weights_tensor = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(cv_labels), y=cv_labels), dtype=torch.float)

        if not cv_done:
            checkpoint = Checkpoint('checkpoint_last', save_path_cv, device, config, cv + 1)
            
            if config["enable_wandb"]:
                Wandb.init(cv, checkpoint.wandb_id, config)
            
            early_stop = False
            es_counter = 0
            
            for epoch in tqdm(range(checkpoint.start_epoch, config['epochs']+1), total=config['epochs'], desc=f'Training epochs (CV {cv+1})'):
                if early_stop:
                    break
                
                duration_epoch = Utils.train_one_epoch(checkpoint.model, device, checkpoint.scaler, checkpoint.optimizer, config, class_weights_tensor)
                checkpoint.scheduler.step()
                
                eval_valid = Eval(Dataloaders.validation, device, checkpoint.model, config, save_path_cv, cv + 1, checkpoint_name='checkpoint_valid', class_weights=class_weights_tensor)
                checkpoint.update_eval_valid(eval_valid, config)
                
                if config["enable_wandb"]:
                    Wandb.wandb_log(eval_valid, cust_data.label_classes, epoch, checkpoint.optimizer, 'Validation Set', config)

                if config['calc_train_error']:
                    eval_train = Eval(Dataloaders.training_eval, device, checkpoint.model, config, save_path_cv, cv + 1, class_weights=class_weights_tensor)
                    if config["enable_wandb"]:
                        Wandb.wandb_log(eval_train, None, 0, None, 'Train Set Peak', config, False)

                # Save epoch state and update metrics, es_counter, etc.
                if config["auto_encoder"]:
                    improvement_identified = round(checkpoint.eval_valid.mean_loss, config['early_stop_accuracy']) < round(validation_best_mean_loss_current_cv, config['early_stop_accuracy'])
                else:
                    improvement_identified = round(checkpoint.eval_valid.metrics[5], config['early_stop_accuracy']) > round(validation_best_metrics_current_cv[5], config['early_stop_accuracy'])
                
                no_overfitting = True
                if config['calc_train_error'] and not config["auto_encoder"]:
                    no_overfitting = round(checkpoint.eval_valid.metrics[5], config['early_stop_accuracy']) <= round(eval_train.metrics[5], config['early_stop_accuracy'])                        
                    
                best_epoch = improvement_identified and no_overfitting
                if best_epoch or epoch == 1:
                    checkpoint.save_checkpoint('checkpoint_best', epoch, save_path_cv, config)
                    
                    Checkpoint.delete_checkpoint('checkpoint_best', last_best_epoch_current_cv, save_path_cv)
                    
                    es_counter = 0
                    if config["auto_encoder"]:
                        validation_best_mean_loss_current_cv = copy.deepcopy(checkpoint.eval_valid.mean_loss)
                    else:
                        validation_best_metrics_current_cv = copy.deepcopy(checkpoint.eval_valid.metrics)

                    last_best_epoch_current_cv = epoch
                    
                    if config["enable_wandb"] and config['b_logging_active']:
                        Wandb.wandb_log(checkpoint.eval_valid, cust_data.label_classes, epoch, 'b-Validation', config, False)

                    if config['calc_and_peak_test_error']:
                        eval_test = Eval(Dataloaders.test, device, checkpoint.model, config, save_path_cv, cv + 1, class_weights=class_weights_tensor)
                        if config["enable_wandb"]:
                            Wandb.wandb_log(eval_test, None, 0, None, 'Testset Error', config, False)
                        
                else:
                    es_counter += 1
                    
                if not config["auto_encoder"]:
                    print('no_overfitting:', no_overfitting)

                if config['early_stop_patience']:
                    if es_counter >= config['early_stop_patience']:
                        early_stop = True

                checkpoint.save_checkpoint('checkpoint_last', epoch, save_path_cv, config)
                if epoch != 1:
                    Checkpoint.delete_checkpoint('checkpoint_last', epoch - 1, save_path_cv)

                csv_file_path = f'log_cv_{cv+1}.csv'
                with open(csv_file_path, 'a', newline='') as csvfile:
                    entries_dict = {
                        "Epoch": epoch,
                        "duration": duration_epoch,
                        "overfitting": not no_overfitting
                    }
                    if not config["auto_encoder"]:
                        entries_dict.update(new_entries = {
                            'best_epoch': last_best_epoch_current_cv,
                            'best_mcc': round(validation_best_metrics_current_cv[5], config['early_stop_accuracy']),
                            'improvement_identified': improvement_identified })
                    for key, value in checkpoint.eval_valid.metrics: #"valid_loss": round(checkpoint.eval_valid.mean_loss, config['early_stop_accuracy'])
                        entries_dict[key] = value
                        
                    writer = csv.DictWriter(csvfile, fieldnames=entries_dict.keys())
                    csvfile.seek(0, 2)
                    if csvfile.tell() == 0: writer.writeheader()
                    writer.writerow(entries_dict)
                
                Logger.printer('Validation Metrics (tests on validation set):', config, checkpoint.eval_valid)
                
                if config['peak_train_error']:
                    Logger.printer('Training Metrics (tests on training set mod 2):', config, eval_train)

                if config['calc_and_peak_test_error']:
                    Logger.printer('Testing Metrics (tests on testing set):', config, eval_test)

        for checkpoint_name in ['checkpoint_last', 'checkpoint_best']:
            checkpoint = Checkpoint(checkpoint_name, save_path_cv, device, config, cv + 1)
            if config["enable_wandb"] and 'WANDB_API_KEY' not in os.environ:
                Wandb.init(cv, checkpoint.wandb_id, config)
                
            eval_test = Eval(Dataloaders.test, device, checkpoint.model, config, save_path_cv, cv + 1, checkpoint_name=checkpoint_name, class_weights=class_weights_tensor)
            Logger.printer(checkpoint_name, config, eval_test, if_val_or_test=True)
            if config["enable_wandb"]:
                Wandb.wandb_log(eval_test, cust_data.label_classes, 0, checkpoint.optimizer, checkpoint_name, config)
            Logger.log_test(save_path_cv / FILE_NAME_TEST_RESULTS, checkpoint_name, config, eval_test, if_val_or_test=True)

    for checkpoint_type in ['checkpoint_last', 'checkpoint_best']:
        
        test_mean_loss_sum = 0.0
        test_metrics_sum = [0.0] * 10
        mse_loss_conf_matr_mean_sum = np.array([[0.0, 0.0],[0.0, 0.0]])
            
        for cv in range(config['num_cv']):

            save_path_cv = config.save_path / ('cv_' + str(cv + 1))
            #checkpoint = Checkpoint('checkpoint_best', config.save_path / ('cv_' + str(cv + 1)), device, config) # TODO: remove
            test_mean_loss, test_metrics, mse_loss_conf_matr_mean = Logger.test_read(save_path_cv / FILE_NAME_TEST_RESULTS, checkpoint_type, config, if_val_or_test=True)
            if not config["auto_encoder"]: test_metrics_sum = [sum(x) for x in zip(test_metrics_sum, test_metrics)]
            test_mean_loss_sum += test_mean_loss
            mse_loss_conf_matr_mean_sum += mse_loss_conf_matr_mean
            
        test_metrics_avg = [m / config['num_cv'] for m in test_metrics_sum]
        test_mean_loss_avg = test_mean_loss_sum / config['num_cv']
        mse_loss_conf_matr_mean_avg = mse_loss_conf_matr_mean_sum / config['num_cv']
        
        test_eval_avg = type('obj', (object,), {'mean_loss': test_mean_loss_avg, 'metrics': test_metrics_avg, 'mse_loss_conf_matr_mean': mse_loss_conf_matr_mean_avg})()
        Logger.printer(checkpoint_type + ' Metrics Average:', config, test_eval_avg, if_val_or_test=True)

        Logger.log_test(config.save_path / FILE_NAME_TEST_RESULTS_AVERAGE, checkpoint_type, config, test_eval_avg, if_val_or_test=True)
    
    #if config["enable_wandb"]: # TODO: uncomment and log for last checkpoint as well
    #    Wandb.init(-1, checkpoint.wandb_id, config)
    #    Wandb.wandb_log(test_eval_avg, None, 0, None, 'Average Best', config) # TODO: Called to often (only call once in an epoch) 
        
    print('')
        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='IDDATDLOCT')
    args.add_argument('-cfg', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-gpu', '--gpus', default='0', type=str, help='indices of GPUs to enable (default: all)') # TODO
    args.add_argument('-wb', '--wandb', default=None, type=str, help='Wandb API key (default: None)')
    args.add_argument('-ntt', '--no_trainandtest', dest='trainandtest', action='store_false', help='Deactivation of Training and Testing (default: Activated)')
    args.add_argument('-smp', '--show_samples', dest='show_samples', action='store_true', help='Activate creation of Sample from Data Augmentation (default: Deactivated)')
    args.add_argument('-ycf', '--overwrite_configurations', dest='overwrite_configurations', action='store_true', help='Overwrite Configurations, if config file in this directory already exists. (default: False)')
    args.add_argument('-da', '--data_augmentation', default=None, type=str, help='indices of Data Augmentation techniques to enable (default: None)')
    args.add_argument('-lr', '--learning_rate', default=3e-6, type=float, help='')
    args.add_argument('-wd', '--weight_decay', default=0.001, type=float, help='')
    args.add_argument('-mo', '--momentum', default=0.9, type=float, help='')
    args.add_argument('-bs', '--batch_size', default=128, type=int, help='')
    args.add_argument('-nm', '--name', default='run_0', type=str, help='')
    args.add_argument('-gr', '--group', default='group_0', type=str, help='')
    
    config = Config(args)
    
    train_and_eval(config)
            
        
        
        
        
        
            
        
