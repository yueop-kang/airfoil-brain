
import torch
from torch import nn, optim
import torch.nn.functional as F
from models.blocks import Flatten, UnFlatten, ConvBlock1d, ConvBlock2d, Up, Down
from models.parsec_cvae import ParsecCVAE
"""Bezier, a module for creating Bezier curves.
Version 1.1, from < BezierCurveFunction-v1.ipynb > on 2019-05-02
"""
import matplotlib.pyplot as plt
import numpy as np
from models.TC_layer import TC_gen_layer
from tools.xfoil import tanh_spacing

class TC_VAE(nn.Module):
    def __init__(self, config=None, params_dict=None, device='cuda', mean_std=None):
        super().__init__()
        self.device=device
        self.mean_std=mean_std
        self.num_training_step_call=0
        self.num_constraint_violation=0
            
        ### Default params input ###
        params_default={
            "beta":1e-3,
            "h_dims": 24,
            "z_dims": 12,
            "num_hidden_node_list": [256,64],
            "activation_f":'relu',
            "batch_norm": False,
            "bspline_degree": 3,
            "num_cp_t": 7,
            "num_cp_c": 7
        }
        self.t_mode='train'
        
        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]

        self.input_shape=config["input_shape"]
        self.input_dim=np.prod(self.input_shape)

        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.z_dims, self.num_hidden_node, self.activation_f, self.batch_norm, self.h_dims, self.beta, self.bspline_degree,\
            self.num_cp_t, self.num_cp_c=\
            params_dict["z_dims"], params_dict["num_hidden_node_list"], params_dict["activation_f"],  params_dict["batch_norm"], \
                 params_dict["h_dims"], params_dict["beta"], params_dict["bspline_degree"], params_dict["num_cp_t"], params_dict["num_cp_c"]
        
        self.num_hidden_layer=len(self.num_hidden_node)
        self.tc_layer=TC_gen_layer(self.num_cp_t, self.num_cp_c, self.bspline_degree, device=self.device).to(self.device)
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
                elif self.activation_f == 'gelu':
                    encoder_modules.append(nn.GELU())
        
        encoder_modules.append(nn.Linear(self.num_hidden_node[-1], self.h_dims))
        self.encoder=nn.Sequential(*encoder_modules)

        self.fc_t_cp = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.ReLU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.ReLU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_free_var_t),
                                #   nn.Tanh(),
                                #   nn.Sigmoid()
                                  )
        
        self.fc_c_cp = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.ReLU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.ReLU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_free_var_c),
                                #   nn.Tanh(),
                                #   nn.Sigmoid()
                                  )
       
        self.fc1 = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc2 = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc3 = nn.Sequential(nn.Linear(self.z_dims, self.h_dims))

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + torch.mul(std, esp)
        return z
    
    def bottleneck(self, h, mode='train'):
        mu, logvar = self.fc1(h), self.fc2(h)
        if mode=='train':
            z = self.reparameterize(mu, logvar)
        elif mode=='eval':
            z = mu
        return z, mu, logvar
    
    def encode(self, x, mode='train'):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h, mode=mode)
        return z, mu, logvar

    def decode(self, z):
        h = self.fc3(z)
        raw_cp_t=self.fc_t_cp(h)
        raw_cp_c=self.fc_c_cp(h)
        raw_cp_t=torch.abs(raw_cp_t)
        raw_cp_c=torch.abs(raw_cp_c)
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature=self.tc_layer(raw_cp_t, raw_cp_c)
        return r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature

    def forward(self, x):
        if self.training:
            mode='train'
        else:
            mode='eval'
        
        z, mu, logvar= self.encode(x, mode=mode)
    
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature = self.decode(z)

        
        return r, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar

    def loss_fn(self, r, x, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar):
        if not self.mean_std == None:
            x=x*self.mean_std[1]+self.mean_std[0]
        MSE= F.mse_loss(r, x, reduction='mean')
        curvature_threshold=3
        mask_camber= camber_curve[:,0,:] > 0.25
  
        mask_camber_curvature = torch.where(mask_camber, camber_curvature, torch.zeros_like(camber_curvature))
        mask_thickness= thickness_curve[:,0,:] > 0.25

        mask_thickness_curvature = torch.where(mask_thickness, thickness_curvature, torch.zeros_like(thickness_curvature))

        max_curvature_camber=torch.max(mask_camber_curvature, dim=1)[0]
        max_curvature_thickness=torch.max(mask_thickness_curvature, dim=1)[0]
        R1= torch.maximum(torch.zeros_like(max_curvature_camber).to(self.device), max_curvature_camber-torch.ones_like(max_curvature_camber).to(self.device)*curvature_threshold)
        R2= torch.maximum(torch.zeros_like(max_curvature_thickness).to(self.device), max_curvature_thickness-torch.ones_like(max_curvature_thickness).to(self.device)*curvature_threshold)
        R1=torch.mean(R1)
        R2=torch.mean(R2)
        KLD_temp = 1 + logvar - mu.pow(2) - logvar.exp()
        KLD = -0.5 * torch.mean(torch.sum(KLD_temp, axis=1))
        
        if R1.item() > 0 or R2.item() > 0:
            # print('R1: {:.2e}, R2: {:.2e}'.format(R1.item(), R2.item()))
            self.num_constraint_violation+=1
        
        if self.num_training_step_call % 100 == 1:
            print('MSE: {:.2e}, KLD: {:.2e}, R1: {:.2e}, R2: {:.2e}'.format(MSE.item(), KLD.item(), R1.item(), R2.item()))
            print('Number of constraint violation during training interval: {:d}'.format(self.num_constraint_violation))
            self.num_constraint_violation=0
            plt.plot(torch.mean(KLD_temp, axis=0).detach().cpu().numpy())
            plt.show()
        
        return MSE + self.beta * KLD + 1e-3*(R1+R2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.num_training_step_call+=1
        self.train()
        x = train_batch
        r, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar= self(x)

        loss = self.loss_fn(r, x, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        x = val_batch
        r, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar= self(x)
        loss = self.loss_fn(r, x, z, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_curvature, thickness_curvature,  mu, logvar)
        return loss

    def model_evaluation(self, dataloader, mode='encode'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
            if mode =='encode':
                temp = self.encode(x, mode='eval')[0]
            if mode =='mu':
                temp = self.encode(x)[1]
            if mode =='logvar':
                temp = self.encode(x)[2]

            if mode =='decode':
                temp = self.decode(x)[0]
            if mode =='bezier_t':
                z = self.encode(x, mode='eval')
                temp = self.decode(z[0])[1]
            if mode =='bezier_p':
                z = self.encode(x, mode='eval')
                temp = self.decode(z[0])[2]
            if mode =='bezier_w':
                z = self.encode(x, mode='eval')
                temp = self.decode(z[0])[3]
            if mode =='reconstruct':
                z = self.encode(x, mode='eval')
                temp = self.decode(z[0])[0]

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out
    
    




