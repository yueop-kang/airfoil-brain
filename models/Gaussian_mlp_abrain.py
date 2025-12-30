import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import pickle
from tools.normalize_new import DataProcessing
from models.Gaussian_layer import Gaussian_layer
# test
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Gaussian_mlp(nn.Module):
    def __init__(self, config_dict=None, params_dict=None):
        super().__init__()

        ### Default params input ###
        
        self.params_default={
            "model_name": self.__class__.__name__,
            "num_hidden_layer": 2,
            "num_hidden_node": 16,
            "activation_f":'relu',
            "batch_norm": False,
        }
        
        if config_dict==None:
            self.config_dict={
            "input_dim": 1,
            "output_dim": 1
            } # dummy
        else:
            self.config_dict=config_dict.copy()
        
        
        
        if params_dict==None:
            self.params_dict=self.params_default
        else:
            self.params_dict=params_dict.copy()
            
                
            if "DP_X_config" in params_dict:
                self.DP_X=DataProcessing(params_dict["DP_X_config"])
            if "DP_Y_config" in params_dict:
                self.DP_X=DataProcessing(params_dict["DP_Y_config"])
                    
        
        self.model_initialize()
        
        
    def model_initialize(self):
        self.input_dim=self.config_dict["input_dim"]
        self.output_dim=self.config_dict["output_dim"]
        for param in self.params_default:
                if not param in self.params_dict:
                    # print("No input '{:s}' initialized by default setting".format(param))
                    self.params_dict[param]=self.params_default[param]
        ### Initialize hyper parameters ###
        
        self.params_dict=self.params_dict.copy()
        self.num_hidden_layer, self.num_hidden_node, self.activation_f, self.batch_norm=\
            self.params_dict["num_hidden_layer"], self.params_dict["num_hidden_node"], self.params_dict["activation_f"],  self.params_dict["batch_norm"]
        
            
        modules = []
        for i in range(self.num_hidden_layer+1):
            # Linear Layer
            if i==0: # input layer
                modules.append(nn.Linear(self.input_dim, self.num_hidden_node))

            else: # hidden layer
                modules.append(nn.Linear(self.num_hidden_node,self.num_hidden_node))

                # Batch Normalization layer    
                if self.batch_norm:
                    modules.append(nn.BatchNorm1d(self.num_hidden_node))
                
                # Activation Layer
                if self.activation_f == None:
                    pass
                elif self.activation_f == 'relu':
                    modules.append(nn.ReLU())
                elif self.activation_f == 'tanh':
                    modules.append(nn.Tanh())
                elif self.activation_f == 'leakyrelu':
                    modules.append(nn.LeakyReLU())
                  
        self.fc_layers=nn.Sequential(*modules)
        self.gaussian_layer=Gaussian_layer(self.num_hidden_node, self.output_dim)
        
    def forward(self, x):
        x=self.fc_layers(x)
        mu, var = self.gaussian_layer(x)

        return mu, var

    def loss_fn(self, mu, var_soft, X):
        SE = (X - mu).pow(2)
        NNL = (torch.log(var_soft) + SE/var_soft).sum()
        NNL/=mu.size(0)*3 # number of qoi is 3
        MSE=SE.sum()/(mu.size(0)*3) # number of qoi is 3
        return NNL, MSE
        # return MSE, MSE


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def dataloader_preprocess(self,x,y):
        return x[0], y[0]
    
    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        x, y = train_batch
        x, y = self.dataloader_preprocess(x, y)
        mu, var_soft = self(x)
        loss, mse = self.loss_fn(mu, var_soft, y)
        return loss, mse

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()
        x, y = val_batch
        x, y = self.dataloader_preprocess(x, y)
        mu, var_soft = self(x)
        loss, mse = self.loss_fn(mu, var_soft, y)

        return loss, mse

    def model_evaluation(self, dataloader, mode='foward'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            try:
                x, y = batch
                x, y = self.dataloader_preprocess(x, y)
            except:
                x = batch

            # if mode =='loss':
            #     temp = self.validation_step(batch, idx)
   
            if mode =='foward':
                mu, var_soft = self(x)
            if idx==0:
                mu_tot=mu
                var_tot=var_soft

            else:
                mu_tot=torch.cat((mu_tot,mu),axis=0)
                var_tot=torch.cat((var_tot,var_soft),axis=0)

        
        return mu_tot, var_tot
    
    def load(self, PATH, mode='best_loss', device='cuda'):
        with open(PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        self.DP_X=DataProcessing(model_data['model_config_dict']['DP_X_config'])
        self.DP_Y=DataProcessing(model_data['model_config_dict']['DP_Y_config'])
        self.config_dict=model_data['model_config_dict']
        self.params_dict=model_data['model_params_dict']
        self.log=model_data['trainer_log']
        self.model_initialize()
        self.load_state_dict(model_data['model_state_dict'][mode])
        self.to(device)
        
    def predict(self, x):
        temp=self.DP_X.transform(x)
        temp=self(temp)
        return self.DP_Y.inv_transform(temp)
        




