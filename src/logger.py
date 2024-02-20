
import os
import csv

class Logger():

    commit = {}
    enable_wandb = False

    @classmethod
    def init(cls, cv, checkpoint, config):
        cls.enable_wandb = config['enable_wandb']
        if cls.enable_wandb:
            import wandb

            if config['wandb'] is not None :
                os.environ['WANDB_API_KEY'] = config['wandb']
            else:
                raise AssertionError("W&B is missing API key argument from this program. See docs for more information.")
                
            wandb.init(
                project=config['wb_project'],
                entity='lennardkorte', # TODO: data privacy
                group=config['group'],
                id=checkpoint.wandb_id,
                resume="allow",
                name=config['name'] + '_cv_' + str(cv),
                reinit=True,
                dir=os.getenv("WANDB_DIR", config.save_path))
    
    @classmethod   
    def get_id(cls):
        import wandb
        return wandb.util.generate_id()

    @classmethod   
    def add(cls, metrics:dict, prefix=None):   
        pf = ''
        if prefix is not None:
            pf = prefix + ' ' 
        for key, value in metrics.items():
            cls.commit[pf + key] = value            
        
    @classmethod
    def push(cls, path):
        with open(path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cls.commit.keys())
            csvfile.seek(0, 2)
            if csvfile.tell() == 0: writer.writeheader()
            writer.writerow(cls.commit)

        if cls.enable_wandb:    
            import wandb
            wandb.log(cls.commit)
        
        cls.commit = {}