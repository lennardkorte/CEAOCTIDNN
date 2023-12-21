import os
import copy
import time
import torch
import argparse
import numpy as np

from eval import Eval
from utils import Utils
from config import Config
from logger import Logger
from checkpoint import Checkpoint
from utils_wandb import Wandb
from data_loaders import Dataloaders
from create_samples import create_samples
from dataset_preparation import DatasetPreparation

FILE_NAME_TEST_RESULTS = 'test_results.json'
FILE_NAME_TRAINING = 'training.log'
FILE_NAME_TEST_RESULTS_AVERAGE = 'test_results_average.json'

def train_and_eval(config:Config):

    cust_data = DatasetPreparation(config)
        
    Logger.print_section_line()
    Dataloaders.setup_data_loader_testset(cust_data, config)
    
    device = Utils.config_torch_and_cuda(config)
    
    loss_function = Utils.get_new_lossfunction(cust_data.class_weights, device, config['loss_function'])

    for cv in range(config['num_cv']):

        early_stop = False
        es_counter = 0
        last_best_epoch_current_cv = -1
        if config["auto_encoder"]:
            validation_best_mean_loss_current_cv = 10000000
        else:
            validation_best_metrics_current_cv = [0.0] * 9
        
        Logger.print_section_line()
        
        save_path_cv = config.save_path / ('cv_' + str(cv + 1))
        os.makedirs(save_path_cv, exist_ok=True)
        cv_done = True if any('test_results' in s for s in os.listdir(save_path_cv)) else False

        if cv_done:
            print('Skipped CV number:', cv+1, '(already trained)')
        else:
            with Logger(save_path_cv / FILE_NAME_TRAINING, 'a'):
                print('Start CV number:', cv+1, '\n')
                
                valid_ind_for_cv, train_ind_for_cv = cust_data.get_train_valid_ind(cv)
                train_eval_ind_for_cv = train_ind_for_cv[::2]
                
                Dataloaders.setup_data_loaders_training(train_ind_for_cv,
                                                        train_eval_ind_for_cv,
                                                        valid_ind_for_cv,
                                                        cust_data,
                                                        config
                                                        )
                print("\nTrain iterations / batches per epoch: ", int(len(Dataloaders.trainInd)))
                
                checkpoint = Checkpoint('checkpoint_last', save_path_cv, device, config, cv + 1)
                
                pytorch_total_params = sum(p.numel() for p in checkpoint.model.parameters() if p.requires_grad)
                print('Number of Training Parameters', pytorch_total_params)
                
                if config["enable_wandb"]:
                    Wandb.init(cv, checkpoint.wandb_id, config)

                start_time = time.time()
                
                Logger.print_section_line()
                print("Training...")
                early_stop = False
                es_counter = 0
                
                for epoch in range(checkpoint.start_epoch, config['epochs']+1):
                    if early_stop:
                        break
                    
                    Logger.print_section_line()
                    duration_epoch = Utils.train_one_epoch(checkpoint.model, device, loss_function, checkpoint.scaler, checkpoint.optimizer, config)
                    checkpoint.scheduler.step()

                    print('Evaluating epoch...')
                    duration_cv = time.time() - start_time
                    
                    eval_valid = Eval(Dataloaders.valInd, device, checkpoint.model, loss_function, config, save_path_cv, cv + 1, if_test=False)
                    checkpoint.update_eval_valid(eval_valid, config)
                    
                    if config["enable_wandb"]:
                        Wandb.wandb_log(eval_valid, cust_data.label_classes, epoch, checkpoint.optimizer, 'Validation Set', config)

                    if config['calc_train_error']:
                        eval_train = Eval(Dataloaders.trainInd_eval, device, checkpoint.model, loss_function, config, save_path_cv, cv + 1, if_test=False)
                        if config["enable_wandb"]:
                            Wandb.wandb_log(eval_train, None, 0, None, 'Train Set Peak', config)

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
                            Wandb.wandb_log(checkpoint.eval_valid, cust_data.label_classes, epoch, 'b-Validation', config)

                        if config['calc_and_peak_test_error']:
                            eval_test = Eval(Dataloaders.testInd, device, checkpoint.model, loss_function, config, save_path_cv, cv + 1, if_test=False)
                            if config["enable_wandb"]:
                                Wandb.wandb_log(eval_test, None, 0, None, 'Testset Error', config)
                            
                    else:
                        es_counter += 1
                        
                    print('improvement_identified:', improvement_identified)
                    if not config["auto_encoder"]:
                        print('no_overfitting:', no_overfitting)

                    if config['early_stop_patience']:
                        if es_counter >= config['early_stop_patience']:
                            early_stop = True

                    checkpoint.save_checkpoint('checkpoint_last', epoch, save_path_cv, config)
                    if epoch != 1:
                        Checkpoint.delete_checkpoint('checkpoint_last', epoch - 1, save_path_cv)
                    
                    
                    # Print epoch results
                    np.set_printoptions(precision=4)
                    print('')
                    print('Name:          ', config['name'])
                    print('Fold:          ', cv+1, '/', config['num_cv'])
                    print('Epoch:         ', epoch, '/', config['epochs'])
                    print('Duration Epoch:', str(round(duration_epoch, 2)) + 's')
                    
                    print('\nCross-fold Validation information:')
                    print('Training time:  %dh %dm %ds' % (int(duration_cv / 3600), int(np.mod(duration_cv, 3600) / 60), int(np.mod(np.mod(duration_cv, 3600), 60))))
                    print(time.strftime('Current Time:   %d.%m. %H:%M:%S', time.localtime()))
                    print('Loss ValInd:   ', round(checkpoint.eval_valid.mean_loss, config['early_stop_accuracy']))
                    if not config["auto_encoder"]:
                        print('Best MCC: ', round(validation_best_metrics_current_cv[5], config['early_stop_accuracy']), 'at Epoch', last_best_epoch_current_cv)
                    
                    Logger.printer('Validation Metrics (tests on validation set):', config, checkpoint.eval_valid)
                    if config['peak_train_error']:
                        Logger.printer('Training Metrics (tests on training set mod 2):', config, eval_train)

                    if config['calc_and_peak_test_error']:
                        Logger.printer('Testing Metrics (tests on testing set):', config, eval_test)
        
        Logger.print_section_line()

        file_log_test_results = save_path_cv / FILE_NAME_TEST_RESULTS
        print('Testing performance with testset on:')
        for checkpoint_name in ['checkpoint_last', 'checkpoint_best']:
            checkpoint = Checkpoint(checkpoint_name, save_path_cv, device, config, cv + 1)
            if config["enable_wandb"] and 'WANDB_API_KEY' not in os.environ:
                Wandb.init(cv, checkpoint.wandb_id, config)
                
            eval_test = Eval(Dataloaders.testInd, device, checkpoint.model, loss_function, config, save_path_cv, cv + 1, if_test=True)
            Logger.printer(checkpoint_name, config, eval_test, if_val_or_test=True)
            if config["enable_wandb"]:
                Wandb.wandb_log(eval_test, cust_data.label_classes, 0, checkpoint.optimizer, checkpoint_name, config)
            Logger.log_test(file_log_test_results, checkpoint_name, config, eval_test, if_val_or_test=True)

    Logger.print_section_line()

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
    
    if config['show_samples']:
        create_samples(config)
    
    train_and_eval(config)
            
        
        
        
        
        
            
        
