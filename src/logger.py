
import sys
import json

class Logger(object):
    def __enter__(self):
        pass
    
    def __exit__(self, exception_type, exception_value, traceback):
        pass
    
    def __init__(self, file_name, mode):
        self.file = open(file_name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
        
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
        
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()
        
    @staticmethod
    def print_section_line():
        print('\n--------------------------------------------------------')
        
    @staticmethod
    def printer(title, config, eval_test):
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

    def printer_ae(title, early_stop_accuracy, mean_loss):
        print('\n' + title)
        print("   Loss:        ", round(mean_loss, early_stop_accuracy))

    @staticmethod
    def log_test(file_log_test_results, description, config, eval_test):
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

            # Update the existing data with the new data
            existing_data.update(data_to_append)

            # Write the updated data to the JSON file
            with open(file_log_test_results, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
    
    @staticmethod
    def test_read(file_log_test_results, description, config):
        try:
            # Try to open the JSON file for reading (if it exists)
            with open(file_log_test_results, 'r') as json_file:
                existing_data = json.load(json_file)

            data = existing_data[description]
            
            # Extract the loss value from the data
            mean_loss = data.get('Loss')
            
            # Extract the metrics array from the data
            metrics = []
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
            
            return mean_loss, metrics

        except (FileNotFoundError, json.JSONDecodeError):
            print("JSON file not found or could not be decoded.")

        

    