import torch
from torch import nn, optim
import torch.nn.functional as F
from models.blocks import Flatten, UnFlatten
import numpy as np


class AE(nn.Module):
    def __init__(self, config_dict=None,  params_dict=None):
        super().__init__()

        ### Default params input ###
        
        params_default={
            "z_dims": 12,
            "num_hidden_node_list": [256,64],
            "activation_f":'relu',
            "batch_norm": False,
        }

        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]

        self.input_shape=config_dict["input_shape"]
        self.input_dim=np.prod(self.input_shape)

        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.z_dim, self.num_hidden_node, self.activation_f, self.batch_norm=\
            params_dict["z_dims"], params_dict["num_hidden_node_list"], params_dict["activation_f"],  params_dict["batch_norm"]
       
        self.num_hidden_layer=len(self.num_hidden_node)

        ### Initialize modules ###
        encoder_modules=[]
        encoder_modules.append(Flatten())
        for i in range(self.num_hidden_layer):
            # Linear Layer
            if i==0: # input layer
                encoder_modules.append(nn.Linear(self.input_dim, self.num_hidden_node[0]))

            else: # hidden layer
                encoder_modules.append(nn.Linear(self.num_hidden_node[i-1],self.num_hidden_node[i]))

                # Batch Normalization layer    
                if self.batch_norm:
                    encoder_modules.append(nn.BatchNorm1d(self.num_hidden_node[i]))
                
                # Activation Layer
                if self.activation_f == None:
                    pass
                elif self.activation_f == 'relu':
                    encoder_modules.append(nn.ReLU())
                elif self.activation_f == 'tanh':
                    encoder_modules.append(nn.Tanh())
                elif self.activation_f == 'leakyrelu':
                    encoder_modules.append(nn.LeakyReLU())
        
        encoder_modules.append(nn.Linear(self.num_hidden_node[-1], self.z_dim))

        decoder_modules=[]
        for i in range(self.num_hidden_layer):
            
            # Linear Layer
            if i==0: # input layer
                decoder_modules.append(nn.Linear(self.z_dim, self.num_hidden_node[-i-1]))

            else: # hidden layer
                decoder_modules.append(nn.Linear(self.num_hidden_node[-i],self.num_hidden_node[-i-1]))

                # Batch Normalization layer    
                if self.batch_norm:
                    decoder_modules.append(nn.BatchNorm1d(self.num_hidden_node[-i-1]))
                
                # Activation Layer
                if self.activation_f == None:
                    pass
                elif self.activation_f == 'relu':
                    decoder_modules.append(nn.ReLU())
                elif self.activation_f == 'tanh':
                    decoder_modules.append(nn.Tanh())
                elif self.activation_f == 'leakyrelu':
                    decoder_modules.append(nn.LeakyReLU())

        decoder_modules.append(nn.Linear(self.num_hidden_node[0], self.input_dim))
        decoder_modules.append(UnFlatten(self.input_shape))

        self.encoder=nn.Sequential(*encoder_modules)
        self.decoder=nn.Sequential(*decoder_modules)
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z) 

    def forward(self, x):
        z=self.encode(x)
        r=self.decode(z)
        return r, z

    def loss_fn(self, r, x):
        MSE = F.mse_loss(r, x, reduction='mean')
        return MSE 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        x = train_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()
        x = val_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def model_evaluation(self, dataloader, mode='encode'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
            if mode =='encode':
                temp = self.encode(x)
            if mode =='decode':
                temp = self.decode(x)
            if mode =='reconstruct':
                temp, temp_ = self(x)

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out