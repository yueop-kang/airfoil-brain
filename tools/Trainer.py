from xml.etree.ElementTree import TreeBuilder
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchsummary import summary
from copy import deepcopy

def plot_loss_hist(epoch, loss_his, loss_val_his= None):
    
    plt.plot(np.arange(epoch+1), np.array(loss_his),'r-',label='loss')
    to_print = "Loss: {:.3e}".format(loss_his[-1])
    if loss_val_his:
        plt.plot(np.arange(epoch+1), np.array(loss_val_his),'b-',label='loss_val')
        to_print = "Loss: {:.3e}, Loss_val: {:.3e}".format(loss_his[-1],loss_val_his[-1])
    # plt.yscale('log')
    plt.legend()
    plt.show()
    print(to_print)


class Trainer():  #customize to train Gaussian_mlp
    def __init__(self, settings=None):
        
        settings_default={
            'lr_init': 1e-3,
            'epoch': 10000,
            'weight_decay': 1e-4,
            'lr_decay_step_size': 2500,
            'lr_decay_rate': 0.5,
            'verbose': False,
            'earlystopping': False
        }
        
        
        if settings==None:
            self.settings=settings_default
        else:
            self.settings=settings.copy()
            for setting in settings_default:
                if not setting in self.settings:
                    print("No input '{:s}' initialized by default setting".format(setting))
                    self.settings[setting]=settings_default[setting]
            
      
       
        self.log = {'loss':[], 'loss_val': [], 'loss_mse': [], 'loss_val_mse': []}
        self.earlystopping_count=0

    def auto_lr_find(self, model, train_loader, val_loader=None):
        import copy

        model_dummy=copy.deepcopy(model)
        step_find=4
        
        optimizer_lr_find = torch.optim.Adam(model_dummy.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler_lr_find = torch.optim.lr_scheduler.StepLR(optimizer_lr_find, step_size=1, gamma=10)
        
        loss_hist=[]
        loss_val_hist=[]
        lr_hist=[]
        for epoch in range(step_find): # train, val loop
            acc_loss=0
            model_dummy.train()
            for batch_idx, train_batch in enumerate(train_loader):
                loss = model_dummy.training_step(train_batch, batch_idx)
                acc_loss += loss.item()
                loss.backward()
                optimizer_lr_find.step()
                optimizer_lr_find.zero_grad()
            lr_hist.append(scheduler_lr_find.get_last_lr()[0])
            scheduler_lr_find.step()
            acc_loss/=(batch_idx+1)
            loss_hist.append(acc_loss)
            if val_loader:
                with torch.no_grad():
                    acc_loss_val=0
                    model_dummy.eval()
                    for batch_idx, val_batch in enumerate(val_loader):
                        loss_val = model.validation_step(val_batch,batch_idx)
                        acc_loss_val += loss_val.item()
                    acc_loss_val/=(batch_idx+1)
                    loss_val_hist.append(acc_loss_val)
        if self.settings['verbose']:
            plt.plot(lr_hist,loss_hist,'r.-')
            if val_loader:
                plt.plot(lr_hist,loss_val_hist,'b-')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Learning rate')
            plt.ylabel('Loss')
            plt.title('LR range test')
            plt.show()
        
        for i in range(len(lr_hist)-1):
            if loss_hist[i]<loss_hist[i+1]:
                self.lr_init=lr_hist[i]
                break
            if i == len(lr_hist)-2:
                self.lr_init=lr_hist[-1]
   
        print("auto_lr_find: ", self.lr_init)


    def fit(self, model, train_loader, val_loader=None, model_save=False, save_path=None):
        if not save_path==None:
            model_save=True
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.settings['lr_init'], weight_decay=self.settings['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.settings['lr_decay_step_size'], gamma=self.settings['lr_decay_rate'])
            
        best_loss=np.Inf
        best_loss_val=np.Inf
        
        for epoch in tqdm(range(self.settings['epoch'])):
            # Train loop start
            model.train()
            acc_loss=0
            acc_loss_mse=0
            for batch_idx, train_batch in enumerate(train_loader):
                loss, mse = model.training_step(train_batch, batch_idx)
                acc_loss += loss.item()
                acc_loss_mse += mse.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            acc_loss/=(batch_idx+1)
            acc_loss_mse/=(batch_idx+1)
            self.log['loss'].append(acc_loss)
            self.log['loss_mse'].append(acc_loss_mse)
            self.scheduler.step()
            # print(self.scheduler.get_lr())
            
            if best_loss > self.log['loss'][-1]:
                self.best_loss_model=deepcopy(model)
                best_loss = self.log['loss'][-1]

            # Validation loop start
            if val_loader:
                with torch.no_grad():
                    acc_loss_val=0
                    acc_loss_val_mse=0
                    model.eval()
                    for batch_idx, val_batch in enumerate(val_loader):
                        loss_val, loss_val_mse = model.validation_step(val_batch,batch_idx)
                        acc_loss_val += loss_val.item()
                        acc_loss_val_mse += loss_val_mse.item()
                    acc_loss_val/=(batch_idx+1)
                    acc_loss_val_mse/=(batch_idx+1)
                    self.log['loss_val'].append(acc_loss_val)
                    self.log['loss_val_mse'].append(acc_loss_val_mse)
            
        
                    if best_loss_val > self.log['loss_val'][-1]:
                        self.best_loss_val_model=deepcopy(model)
                        best_loss_val = self.log['loss_val'][-1]
                        self.earlystopping_count=0

            # # Early stopping
            if self.settings['earlystopping']:
                self.earlystopping_count+=1
                # print(self.earlystopping_count)
                if self.earlystopping_count > 1000:
                    print('Early stopping!' )
                    plot_loss_hist(epoch=epoch, loss_his=self.log['loss'], loss_val_his=self.log['loss_val'])
                    break
                   
            # training result print
            if self.settings['verbose']:
                if epoch % 100 == 99 or epoch==self.settings['epoch']-1:
                    clear_output(wait=True)
                    plot_loss_hist(epoch=epoch, loss_his=self.log['loss'], loss_val_his=self.log['loss_val'])
        
        if model_save:
            model_state_dict={
                'final_iteration': model.state_dict(),
                'best_loss': self.best_loss_model.state_dict(),
            }
            if val_loader:
                model_state_dict['best_loss_val']=self.best_loss_val_model.state_dict()
                
            
            
            save_dict={
                'model_config_dict': model.config_dict,
                'model_params_dict': model.params_dict,
                'model_state_dict': model_state_dict,
                'trainer_settings': self.settings,
                'trainer_log': self.log
            }
            
            # with open(save_path+"model_save.json", "w") as json_file:
            #     json.dump(save_dict, json_file)
          
            with open(save_path+'.pickle', 'wb') as f:
                pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
            # torch.save(model.state_dict(), save_path + '.torch')
            # torch.save(self.best_loss_model.state_dict(), save_path + '_best_loss.torch')
            # if val_loader: 
            #     torch.save(self.best_loss_val_model.state_dict(), save_path + '_best_loss_val.torch') 