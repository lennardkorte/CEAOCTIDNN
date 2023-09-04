
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
    def printer(title, early_stop_accuracy, metrics, mean_loss):
        print('\n' + title)
        print("   Loss:        ", round(mean_loss, early_stop_accuracy))
        print("   Accuracy:    ", round(metrics[0], early_stop_accuracy))
        print("   Sensitivity: ", round(metrics[1], early_stop_accuracy))
        print("   Specificity: ", round(metrics[2], early_stop_accuracy))
        print("   F1:          ", round(metrics[4], early_stop_accuracy))
        print("   BACC:        ", round(metrics[7], early_stop_accuracy))
        print("   MCC:         ", round(metrics[8], early_stop_accuracy))
        print("   Prec.:       ", round(metrics[9], early_stop_accuracy))