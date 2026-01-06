from sklearn.model_selection._search import ParameterGrid
from tools.Trainer import Trainer 
import numpy as np
import time

def write_tuning_results(PG, PG_train_settings, score, save_folder, elapsed_time):
    
    f=open(save_folder + '/tuning_results.txt', 'w')
    p=0
    for params in PG:
        for train_settings in PG_train_settings:
            if p==0:
                f.write('model_index    ')
                for k, v in params.items():
                    f.write(k +'            ')
                for k, v in train_settings.items():
                    f.write(k +'            ')
                f.write('score          elapsed_time        best\n')

            f.write(str(p+1) +'         ')
            for k, v in params.items():
                f.write(str(v) +'              ')
            for k, v in train_settings.items():
                f.write(str(v) +'              ')
            f.write('{:.3e}         '.format(score[p]))
            f.write("{:.2f}s        ".format(elapsed_time[p]))
            if p==np.argmin(score):
                f.write('o\n')
            else:
                f.write('\n')

            p+=1
    f.close()
import torch
def set_seed(seed_value):
    # Set seed for CPU operations
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    # Set seed for GPU operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# Usage example:

class GridSearch():
    def __init__(self, model, config, params_grid, train_settings_grid, train_loader, val_loader, save_folder, device='cuda', seed=42):
        
        set_seed(seed)
        
        self.model=model
        self.config=config
        self.params_grid=params_grid
        self.train_settings_grid=train_settings_grid
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.save_folder=save_folder
        self.device=device

        self.PG=ParameterGrid(params_grid)
        self.PG_train_settings=ParameterGrid(train_settings_grid)

        self.params_str=[]
        self.model_tot=[]
        self.score=[]
        self.elapsed_time=[]
        
    def tune(self):
        p=0
        for params in self.PG:
            self.params_str.append(params)
            
            m=self.model(self.config, params).to(self.device)
            for train_settings in self.PG_train_settings:
                p+=1
                trainer = Trainer(train_settings)
                start = time.time()
                trainer.fit(m, self.train_loader, self.val_loader, model_save=True, save_path=self.save_folder+'/model_{:d}'.format(p))
                end = time.time()
                self.elapsed_time.append(end-start)

                # self.model_tot.append(m)
                self.model_tot.append(trainer.best_loss_val_model)

                self.score.append(min(trainer.log["loss_val"]))
        self.score=np.array(self.score)
        self.model_best=self.model_tot[np.argmin(self.score)]

        write_tuning_results(self.PG, self.PG_train_settings, self.score, self.save_folder, self.elapsed_time)