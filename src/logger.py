
import sys
import json
import numpy as np

class Logger(object):
    def __enter__(self):    
        self.stdout = sys.stdout
        sys.stdout = self
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout = self.stdout
        self.file.close()
    
    def __init__(self, file_name, mode, terminal=True):
        self.terminal = terminal
        self.file = open(file_name, mode)
        
    def write(self, data):
        if data != '\n':
            self.file.write(data)
        if self.terminal:
            self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        if self.terminal:
            self.stdout.flush()
        
    @staticmethod
    def printer(title, config, eval_test, if_val_or_test = False):
        print('\n' + title)
        print("   Loss:          ", round(eval_test.mean_loss, config['early_stop_accuracy']))
        if not config["auto_encoder"]:
            print("   Accuracy:      ", round(eval_test.metrics[0], config['early_stop_accuracy']))
            print("   Sensitivity:   ", round(eval_test.metrics[1], config['early_stop_accuracy']))
            print("   Specificity:   ", round(eval_test.metrics[2], config['early_stop_accuracy']))
            print("   F1:            ", round(eval_test.metrics[3], config['early_stop_accuracy']))
            print("   BACC:          ", round(eval_test.metrics[4], config['early_stop_accuracy']))
            print("   MCC:           ", round(eval_test.metrics[5], config['early_stop_accuracy']))
            print("   Prec.:         ", round(eval_test.metrics[6], config['early_stop_accuracy']))
        elif if_val_or_test:
            np.set_printoptions(suppress=True, precision=4, floatmode='fixed')
            print('TP:', eval_test.mse_loss_conf_matr_mean[1,1], 'TN:', eval_test.mse_loss_conf_matr_mean[0,0], 'FP:', eval_test.mse_loss_conf_matr_mean[1,0], 'FN:', eval_test.mse_loss_conf_matr_mean[0,1])

    def printer_ae(title, early_stop_accuracy, mean_loss):
        print('\n' + title)
        print("   Loss:        ", round(mean_loss, early_stop_accuracy))

    @staticmethod
    def log_test(file_log_test_results, description, config, eval_test, if_val_or_test=False):
            try:
                # Try to open the JSON file for reading (if it exists)
                with open(file_log_test_results, 'r') as json_file:
                    existing_data = json.load(json_file)
            except (FileNotFoundError, json.JSONDecodeError):
                # If the file doesn't exist or is empty, initialize with an empty dictionary
                existing_data = {}

            data_to_append = {
                description: {
                    'Loss': float(eval_test.mean_loss)
                }
            }
            if not config["auto_encoder"]:
                data_to_append[description].update({
                    'Accuracy': float(eval_test.metrics[0]),
                    'Sensitivity': float(eval_test.metrics[1]),
                    'Specificity': float(eval_test.metrics[2]),
                    'F1': float(eval_test.metrics[3]),
                    'BACC': float(eval_test.metrics[4]),
                    'MCC': float(eval_test.metrics[5]),
                    'Prec.': float(eval_test.metrics[6])
                })
            elif if_val_or_test:
                data_to_append[description].update({
                    'mse_loss_conf_matr_mean': eval_test.mse_loss_conf_matr_mean.tolist(),
                })
            '''
                data_to_append[description].update({
                    'loss_TP': float(eval_test.mse_loss_conf_matr_mean[1,1]),
                    'loss_TN': float(eval_test.mse_loss_conf_matr_mean[0,0]),
                    'loss_FP': float(eval_test.mse_loss_conf_matr_mean[1,0]),
                    'loss_FN': float(eval_test.mse_loss_conf_matr_mean[0,1])
                })
            '''
            

            # Update the existing data with the new data
            existing_data.update(data_to_append)

            # Write the updated data to the JSON file
            with open(file_log_test_results, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
    
    @staticmethod
    def test_read(file_log_test_results, description, config, if_val_or_test=False):
        try:
            # Try to open the JSON file for reading (if it exists)
            with open(file_log_test_results, 'r') as json_file:
                existing_data = json.load(json_file)

            data = existing_data[description]
            
            # Extract the loss value from the data
            mean_loss = data.get('Loss')
            
            # Extract the metrics array from the data
            metrics = []
            mse_loss_conf_matr_mean = np.array([[0,0],[0,0]])
            if not config["auto_encoder"]:
                metrics = [
                    float(data.get('Accuracy')),
                    float(data.get('Sensitivity')),
                    float(data.get('Specificity')),
                    float(data.get('F1')),
                    float(data.get('BACC')),
                    float(data.get('MCC')),
                    float(data.get('Prec.'))
                ]
            elif if_val_or_test:
                mse_loss_conf_matr_mean = np.array(data.get('mse_loss_conf_matr_mean'))
            
            return mean_loss, metrics, mse_loss_conf_matr_mean

        except (FileNotFoundError, json.JSONDecodeError):
            print("JSON file not found or could not be decoded.")

        

    