
import sys

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
        print("   Loss:        ", round(eval_test.mean_loss, config['early_stop_accuracy']))
        if not config["auto_encoder"]:
            print("   Accuracy:    ", round(eval_test.metrics[0], config['early_stop_accuracy']))
            print("   Sensitivity: ", round(eval_test.metrics[1], config['early_stop_accuracy']))
            print("   Specificity: ", round(eval_test.metrics[2], config['early_stop_accuracy']))
            print("   F1:          ", round(eval_test.metrics[4], config['early_stop_accuracy']))
            print("   BACC:        ", round(eval_test.metrics[7], config['early_stop_accuracy']))
            print("   MCC:         ", round(eval_test.metrics[8], config['early_stop_accuracy']))
            print("   Prec.:       ", round(eval_test.metrics[9], config['early_stop_accuracy']))

    def printer_ae(title, early_stop_accuracy, mean_loss):
        print('\n' + title)
        print("   Loss:        ", round(mean_loss, early_stop_accuracy))