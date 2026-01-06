from re import X
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from tools.Trainer import Trainer
from tools.normalize import Center, MeanStd, MeanStd_channel, DataProcessing
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MLP_finetuning(nn.Module):
    def __init__(self, config_dict=None, params_dict=None, decoder=None):
        super().__init__()

        ### Default params input ###
        if not "num_hidden_layer" in params_dict:
            print("No input 'num_hidden_layer', initialized by default setting")
            params_dict["num_hidden_layer"]=2
        if not "num_hidden_node" in params_dict:
            print("No input 'num_hidden_node', initialized by default setting")
            params_dict["num_hidden_node"]=16
        if not "activation_f" in params_dict:
            print("No input 'activation_f', initialized by default setting")
            params_dict["activation_f"]=None
        if not "batch_norm" in params_dict:
            print("No input 'batch_norm', initialized by default setting")
            params_dict["batch_norm"]=False

        
        self.input_dim=config_dict["input_dim"]
        self.output_dim=config_dict["output_dim"]
        self.decoder=config_dict["decoder"]

        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.num_hidden_layer, self.num_hidden_node, self.activation_f, self.batch_norm=\
            params_dict["num_hidden_layer"], params_dict["num_hidden_node"], params_dict["activation_f"],  params_dict["batch_norm"]

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
        
        modules.append(nn.Linear(self.num_hidden_node, self.output_dim)) # output layer
                
        self.fc_layers=nn.Sequential(*modules)
        # self.fc_layers.apply(init_weights)
        
    def forward(self, x):
        x=self.fc_layers(x)
        x=self.decoder(x)
        return x

    def loss_fn(self, x, X):
        MSE = F.mse_loss(x, X, reduction='mean')
        return MSE 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        try:
            x, y = train_batch
        except:
            x = train_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()

        try:
            x, y = val_batch
        except:
            x = val_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        return loss

    def model_evaluation(self, dataloader, mode='foward'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            try:
                x, y = batch
            except:
                x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
   
            if mode =='foward':
                temp = self(x)

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out



class ROM(nn.Module):
    def __init__(self, dr_header=None, dr_config=None, dr_params_dict=None, reg_header=None, reg_config=None, reg_params_dict=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dr_model=dr_header(dr_config, dr_params_dict).to(self.device)
        self.reg_model=reg_header(reg_config, reg_params_dict).to(self.device)


    def fit(self, train_data, val_data, minibatch=None, load_dr=None, load_reg=None):
        

        self.X_train, self.Y_train = train_data
        if minibatch == None:
            minibatch = self.X_train.shape[0]
        self.X_val, self.Y_val = val_data
        self.fit_dr(train_data=self.Y_train, val_data=self.Y_val, minibatch=minibatch, load_dr=load_dr)
        self.fit_reg(train_data=(self.X_train, self.Z_train), val_data=(self.X_val, self.Z_val), minibatch=minibatch, load_reg=load_reg)

    def fit_dr(self, train_data, val_data, minibatch=None, load_dr=None):
        Y = train_data
        Y_val = val_data
        if minibatch == None:
            minibatch = Y.shape[0]       
        # data normalization (normalizer should be customized based on the characteristics of problem)

        self.DP_Y = DataProcessing(MeanStd_channel(Y), self.device)
        
        # data loading
        trainset=self.DP_Y.transform(Y)
        valset=self.DP_Y.transform(Y_val)
        
        dataloader = DataLoader(trainset, batch_size=minibatch, shuffle=False)
        dataloader_shuffle = DataLoader(trainset, batch_size=minibatch, shuffle=True)
        dataloader_val = DataLoader(valset, batch_size=minibatch, shuffle=False)

        if load_dr ==None:
            # train model 
            self.dr_trainer = Trainer()
            self.dr_trainer.fit(self.dr_model, dataloader_shuffle, dataloader_val, epochs=5000)
            self.dr_model=self.dr_trainer.best_loss_val_model
        else:
            # load model
            self.dr_model.load_state_dict(torch.load(load_dr))

        # evaluate latent variables to train regression model
        self.Z_train=self.dr_model.model_evaluation(dataloader, 'encode').detach().cpu().numpy()
        self.Z_val=self.dr_model.model_evaluation(dataloader_val, 'encode').detach().cpu().numpy()

    def fit_reg(self, train_data, val_data, minibatch=None, load_reg=None):

        # dataset preparation
        X, Z = train_data
        X_val, Z_val = val_data
        if minibatch == None:
            minibatch = X.shape[0]
            
        # data normalization (normalizer should be customized based on the characteristics of problem)
        self.DP_X = DataProcessing(MeanStd(X), self.device)
        self.DP_Z = DataProcessing(Center(Z), self.device)

        # data loading
        trainset=TensorDataset(self.DP_X.transform(X), self.DP_Z.transform(Z))
        valset=TensorDataset(self.DP_X.transform(X_val), self.DP_Z.transform(Z_val))
        
        dataloader = DataLoader(trainset, batch_size=minibatch, shuffle=False)
        dataloader_shuffle = DataLoader(trainset, batch_size=minibatch, shuffle=True)
        dataloader_val = DataLoader(valset, batch_size=minibatch, shuffle=False)
        if load_reg ==None:
            # train model
            self.reg_trainer = Trainer()
            self.reg_trainer.fit(self.reg_model, dataloader_shuffle, dataloader_val, epochs=5000)
            self.reg_model=self.reg_trainer.best_loss_val_model
        else:
            self.reg_model.load_state_dict(torch.load(load_reg))

    def predict(self, data):
        data = self.DP_X.transform(data)
        dataloader = DataLoader(data, batch_size=30, shuffle=False)

        out_reg = self.reg_model.model_evaluation(dataloader, mode='foward')
        out_reg =self.DP_Z.inv_transform(out_reg)

        out_reg=torch.Tensor(out_reg).to(self.device)
        dataloader = DataLoader(out_reg, batch_size=30, shuffle=False)

        out_dr = self.dr_model.model_evaluation(dataloader, mode='decode')
        out = self.DP_Y.inv_transform(out_dr)
        return out

    def reconstruct(self, data):
    
        data=self.DP_Y.transform(data)
        dataloader = DataLoader(data, batch_size=30, shuffle=False)

        out_dr = self.dr_model.model_evaluation(dataloader, mode='reconstruct')
        out = self.DP_Y.inv_transform(out_dr)
        return out

