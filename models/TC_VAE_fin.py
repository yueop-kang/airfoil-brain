
import torch
from torch import nn, optim
import torch.nn.functional as F
from models.blocks import Flatten, UnFlatten, ConvBlock1d, ConvBlock2d, Up, Down
"""Bezier, a module for creating Bezier curves.
Version 1.1, from < BezierCurveFunction-v1.ipynb > on 2019-05-02
"""
import matplotlib.pyplot as plt
import numpy as np
from models.TC_layer_c_te_modi import TC_gen_layer
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
            "beta": 1e-3,
            "h_channels": [16, 32, 64, 128, 256],
            "h_kernel": 3,
            "h_padding": 1,
            "activation_f":'relu',
            "batch_norm": False,
            "reduced_scale": 2,
            "z_dims":12,
            "batch_norm": False,
            "bspline_degree": 3,
            "num_cp_t": 7,
            "num_cp_c": 7
        }

        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]
        

        self.input_shape=config["input_shape"]
        self.input_dim=len(self.input_shape)-1
        
        self.t_mode='train'
        
        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.beta, self.h_channels, self.h_kernel, self.h_padding, self.reduced_scale, self.activation_f, self.batch_norm, self.z_dims, self.bspline_degree, self.num_cp_t, self.num_cp_c =\
            params_dict["beta"], params_dict["h_channels"], params_dict["h_kernel"], params_dict["h_padding"], params_dict["reduced_scale"],\
                 params_dict["activation_f"], params_dict["batch_norm"], params_dict["z_dims"], params_dict["bspline_degree"], params_dict["num_cp_t"], params_dict["num_cp_c"]
        self.t_mode='train'
        
        
        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]

        self.tc_layer=TC_gen_layer(self.num_cp_t, self.num_cp_c, self.bspline_degree, device=self.device).to(self.device)
        
        block_depth=len(self.h_channels)
        
        encoder_modules_c=[]
        if self.input_dim==1:
            self.height_reduced_dims=int(self.input_shape[1]/np.prod(self.reduced_scale))
            
 
            encoder_modules_c.append(nn.Conv1d(self.input_shape[0], self.h_channels[0], kernel_size=1, stride=1, padding=0, bias=False))
            in_channel=self.h_channels[0]
            for i in range(block_depth):
                encoder_modules_c.append(
                    Down(ConvBlock1d(in_channel, self.h_channels[i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale[i], input_dim=self.input_dim)
                )
                in_channel = self.h_channels[i]
            self.h_dims= self.h_channels[-1]*self.height_reduced_dims
            
        encoder_modules_c.append(Flatten())
        self.encoder_c = nn.Sequential(*encoder_modules_c)
        # print(self.encoder_c)
        encoder_modules_t=[]
        if self.input_dim==1:
            self.height_reduced_dims=int(self.input_shape[1]/np.prod(self.reduced_scale))
            encoder_modules_t.append(nn.Conv1d(self.input_shape[0], self.h_channels[0], kernel_size=1, stride=1, padding=0, bias=False))
            in_channel=self.h_channels[0]
            for i in range(block_depth):
                encoder_modules_t.append(
                    Down(ConvBlock1d(in_channel, self.h_channels[i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale[i], input_dim=self.input_dim)
                )
                in_channel = self.h_channels[i]
            self.h_dims= self.h_channels[-1]*self.height_reduced_dims
            
        encoder_modules_t.append(Flatten())
        self.encoder_t = nn.Sequential(*encoder_modules_t)
        ### Initialize modules ###
        
        self.fc_t_cp_x = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_t_cp-2),
                                #   nn.ReLU(),
                                  )
        
        self.fc_t_cp_y = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_t_cp-2),
                                #   nn.ReLU(),
                                  )
        
        self.fc_c_cp_x = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_c_cp-1),
                                #   nn.ReLU(),
                                  )
        
        self.fc_c_cp_y = nn.Sequential(nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.h_dims),
                                  nn.GELU(),
                                  nn.Linear(self.h_dims, self.tc_layer.num_t_cp-2),
                                  )
        
        self.fc1_c = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc2_c = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc3_c = nn.Sequential(nn.Linear(self.z_dims, self.h_dims))
        
        self.fc1_t = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc2_t = nn.Sequential(nn.Linear(self.h_dims, self.z_dims))
        self.fc3_t = nn.Sequential(nn.Linear(self.z_dims, self.h_dims))

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + torch.mul(std, esp)
        return z
    
    def bottleneck(self, h_c, h_t, mode='train'):
        mu_c, logvar_c = self.fc1_c(h_c), self.fc2_c(h_c)
        mu_t, logvar_t = self.fc1_t(h_t), self.fc2_t(h_t)
        if mode=='train':
            z_c = self.reparameterize(mu_c, logvar_c)
            z_t = self.reparameterize(mu_t, logvar_t)
        elif mode=='eval':
            z_c = mu_c
            z_t = mu_t
        return z_c, mu_c, logvar_c, z_t, mu_t, logvar_t
    
    def encode(self, x, mode='train'):
 
        h_c = self.encoder_c(x)
        
        h_t = self.encoder_t(x)
        z_c, mu_c, logvar_c, z_t, mu_t, logvar_t = self.bottleneck(h_c, h_t, mode=mode)
   
        return z_c, mu_c, logvar_c, z_t, mu_t, logvar_t

    def decode(self, z_c, z_t):
        h_c = self.fc3_c(z_c)
        raw_cp_c_x=torch.abs(self.fc_c_cp_x(h_c))
        raw_cp_c_y=self.fc_c_cp_y(h_c)
        
        h_t = self.fc3_t(z_t)
        raw_cp_t_x=torch.abs(self.fc_t_cp_x(h_t))
        raw_cp_t_y=torch.abs(self.fc_t_cp_y(h_t))
    
        raw_cp_c=torch.cat([raw_cp_c_x, raw_cp_c_y], dim=1)
        raw_cp_t=torch.cat([raw_cp_t_x, raw_cp_t_y], dim=1)
        
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature \
            =self.tc_layer(raw_cp_t, raw_cp_c)
        return r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature

    def forward(self, x):
        if self.training:
            mode='train'
        else:
            mode='eval'
        # mode='eval'
        z_c, mu_c, logvar_c, z_t, mu_t, logvar_t= self.encode(x, mode=mode)
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature = self.decode(z_c, z_t)

        return r, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve,camber_slope, thickness_slope, camber_curvature, thickness_curvature,  mu_c, logvar_c,  mu_t, logvar_t

    def loss_fn(self, r, x, y, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,\
        camber_curvature, thickness_curvature,  mu_c, logvar_c,  mu_t, logvar_t):
        x=torch.swapaxes(x, 1,2)
        if not self.mean_std == None:
            x=x*self.mean_std[1]+self.mean_std[0]
        MODE='MSE'
        
        if MODE == 'MSE_norm':
            temp= torch.mul(r-x,r-x)
            temp= torch.mean(temp,dim=-1)
            temp= torch.mean(temp,dim=-1)
            x_lower=torch.flip(x[:,:51,1],dims=[1])
            x_upper=x[:,50:,1]
            thick=x_upper-x_lower
            max_thickness=torch.max(thick, dim=1)[0]
            MSE_thickness_weighted = torch.div(temp, max_thickness)
            MSE=torch.mean(MSE_thickness_weighted)*0.1
        elif MODE == 'MSE':
            MSE= F.mse_loss(r, x, reduction='mean')
            
        curvature_threshold=50
        mask_camber= camber_curve[:,0,:] > 0.25
        mask_camber_curvature = torch.where(mask_camber, camber_curvature, torch.zeros_like(camber_curvature, device=self.device))
        mask_thickness= thickness_curve[:,0,:] > 0.25
        mask_thickness_curvature = torch.where(mask_thickness, thickness_curvature, torch.zeros_like(thickness_curvature, device=self.device))
        
        max_curvature_camber=torch.max(torch.abs(mask_camber_curvature), dim=1)[0]
        max_curvature_thickness=torch.max(torch.abs(mask_thickness_curvature), dim=1)[0]
        
        R1= torch.maximum(torch.zeros_like(max_curvature_camber, device=self.device), max_curvature_camber-torch.ones_like(max_curvature_camber, device=self.device)*curvature_threshold)
        R2= torch.maximum(torch.zeros_like(max_curvature_thickness, device=self.device), max_curvature_thickness-torch.ones_like(max_curvature_thickness, device=self.device)*curvature_threshold)
        R1=torch.mean(R1)
        R2=torch.mean(R2)
        
        KLD_temp_c = 1 + logvar_c - mu_c.pow(2) - logvar_c.exp()
        KLD_c = -0.5 * torch.mean(torch.sum(KLD_temp_c[:,2:], axis=1))
        
        KLD_temp_t = 1 + logvar_t - mu_t.pow(2) - logvar_t.exp()
        KLD_t = -0.5 * torch.mean(torch.sum(KLD_temp_t[:,2:], axis=1))
      
        camber_cp_length=torch.linalg.norm(camber_cp[:,1:]-camber_cp[:,:-1], dim=2)
        thickness_cp_length=torch.linalg.norm(thickness_cp[:,1:]-thickness_cp[:,:-1], dim=2)
        # L1=torch.mean(torch.sum(1/camber_cp_length**2, dim=1))
        # L2=torch.mean(torch.sum(1/thickness_cp_length[:,1:]**2, dim=1))
        
        L1=torch.std(torch.sum(camber_cp_length, dim=1))
        L2=torch.std(torch.sum(thickness_cp_length, dim=1))
        
        def cal_MSE_z(y, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, thickness_curvature, num_training_call=1):

            max_camber, max_camber_idx=torch.max(torch.abs(camber_curve[:,1:,:]), dim=2)
            max_thickness, max_thickness_idx=torch.max(thickness_curve[:,1:,:], dim=2)
            
            max_camber_loc=camber_curve[:,0,:][torch.arange(max_camber_idx.size(0)), max_camber_idx.view(-1)]
            max_thickness_loc=thickness_curve[:,0,:][torch.arange(max_thickness_idx.size(0)), max_thickness_idx.view(-1)]
            
            r_le=-1/thickness_curvature[:,0]
            r_le_log=(torch.log10(r_le)+2.8)/2.5

            c_le_gamma = torch.arctan(camber_cp[:,0,1]-camber_cp[:,1,1]/(camber_cp[:,0,0]-camber_cp[:,1,0]))
            c_te_gamma = torch.arctan(camber_cp[:,-1,1]-camber_cp[:,-2,1]/(camber_cp[:,-1,0]-camber_cp[:,-2,0]))
            t_te_gamma = torch.arctan(thickness_cp[:,-1,1]-thickness_cp[:,-2,1]/(thickness_cp[:,-1,0]-thickness_cp[:,-2,0]))
       
            c_le_gamma=(c_le_gamma*180/torch.pi+90)/180
            c_te_gamma=(c_te_gamma*180/torch.pi+50)/70
            t_te_gamma=-(t_te_gamma*180/torch.pi)/60
            
            MSE_z_c1 = F.mse_loss(z_c[:,0], max_camber[:,0]*6, reduction='mean')
            # MSE_z_c2 = F.mse_loss(z_c[:,1], c_le_gamma, reduction='mean')
            
            MSE_z_c2 = F.mse_loss(z_c[:,1], c_te_gamma, reduction='mean')
            # MSE_z_c3 = F.mse_loss(z_c[:,2], max_camber_loc, reduction='mean')
            
            MSE_z_t1 = F.mse_loss(z_t[:,0], max_thickness[:,0]*5, reduction='mean')
            MSE_z_t2 = F.mse_loss(z_t[:,1], r_le_log, reduction='mean')
            # MSE_z_t3 = F.mse_loss(z_t[:,2], t_te_gamma, reduction='mean')
            # MSE_z_t3 = F.mse_loss(z_t[:,2], max_thickness_loc*2, reduction='mean')
            
            MSE_z_c1_phys = F.mse_loss(y[:,0], max_camber[:,0]*6, reduction='mean')
            # MSE_z_c2_phys = F.mse_loss(y[:,1], c_le_gamma, reduction='mean')
            # MSE_z_c2_phys = F.mse_loss(y[:,1], max_camber_loc, reduction='mean')
            MSE_z_c2_phys = F.mse_loss(y[:,1], c_te_gamma, reduction='mean')
            
            MSE_z_t1_phys = F.mse_loss(y[:,3], max_thickness[:,0]*5, reduction='mean')
            MSE_z_t2_phys = F.mse_loss(y[:,4], r_le_log, reduction='mean')
            # MSE_z_t3_phys = F.mse_loss(y[:,5], t_te_gamma, reduction='mean')
            # MSE_z_t3_phys = F.mse_loss(y[:,5], max_thickness_loc*2, reduction='mean')
           
            MSE_z_c3=0
            MSE_z_t3=0
            MSE_z_c3_phys=0
            MSE_z_t3_phys=0
            
            
            MSE_z = MSE_z_c1 + MSE_z_c2 + MSE_z_c3 +\
                MSE_z_t1 + MSE_z_t2 + MSE_z_t3
                
            MSE_z_phys = MSE_z_c1_phys + MSE_z_c2_phys + MSE_z_c3_phys +\
                MSE_z_t1_phys + MSE_z_t2_phys + MSE_z_t3_phys
            
            if num_training_call  %100 == 1:
                plt.figure(figsize=(15,6))
                plt.subplot(2,2,1)
                plt.scatter(z_c.detach().cpu().numpy()[:,0], max_camber.detach().cpu().numpy()[:,0]*6, c= z_c.detach().cpu().numpy()[:,0],s=4)
                plt.axis('equal')
                plt.grid(True)
                
                # plt.subplot(2,3,2)
                # plt.scatter(z_c.detach().cpu().numpy()[:,2], max_camber_loc.detach().cpu().numpy(), c= z_c.detach().cpu().numpy()[:,0],s=4)
                # # plt.plot(z_c.detach().cpu().numpy()[:,1], c_le_gamma.detach().cpu().numpy(),'k.')
                # plt.axis('equal')
                # plt.grid(True)
                
                plt.subplot(2,2,2)
                plt.plot(z_c.detach().cpu().numpy()[:,1], c_te_gamma.detach().cpu().numpy(),'k.')
                plt.axis('equal')
                plt.grid(True)
                
                plt.subplot(2,2,3)
                plt.plot(z_t.detach().cpu().numpy()[:,0], max_thickness.detach().cpu().numpy()[:,0]*5,'k.')
                plt.axis('equal')
                plt.grid(True)
                
                plt.subplot(2,2,4)
                plt.plot(z_t.detach().cpu().numpy()[:,1], r_le_log.detach().cpu().numpy(),'k.')
                plt.axis('equal')
                plt.grid(True)
                
                # plt.subplot(2,3,6)
                # # plt.plot(z_t.detach().cpu().numpy()[:,2], max_thickness_loc.detach().cpu().numpy()*2,'k.')
                # plt.plot(z_t.detach().cpu().numpy()[:,2], t_te_gamma.detach().cpu().numpy(),'k.')
                # plt.axis('equal')
                # plt.grid(True)
                plt.show()
                    
                
            return MSE_z_phys, MSE_z, MSE_z_c1, MSE_z_c2, MSE_z_c3, MSE_z_t1, MSE_z_t2, MSE_z_t3
         
        def cal_MSE_z2(z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, thickness_curvature):

            max_camber, max_camber_idx=torch.max(torch.abs(camber_curve[:,1:,:]), dim=2)
            max_thickness, max_thickness_idx=torch.max(thickness_curve[:,1:,:], dim=2)
            
            max_camber_loc=camber_curve[:,0,:][torch.arange(max_camber_idx.size(0)), max_camber_idx.view(-1)]
            max_thickness_loc=thickness_curve[:,0,:][torch.arange(max_thickness_idx.size(0)), max_thickness_idx.view(-1)]
            
            r_le=-1/thickness_curvature[:,0]
            r_le_log=(torch.log10(r_le)+2.8)/2.5

            c_le_gamma = torch.arctan(camber_cp[:,0,1]-camber_cp[:,1,1]/(camber_cp[:,0,0]-camber_cp[:,1,0]))
            c_te_gamma = torch.arctan(camber_cp[:,-1,1]-camber_cp[:,-2,1]/(camber_cp[:,-1,0]-camber_cp[:,-2,0]))
            t_te_gamma = torch.arctan(thickness_cp[:,-1,1]-thickness_cp[:,-2,1]/(thickness_cp[:,-1,0]-thickness_cp[:,-2,0]))
       
            c_le_gamma=(c_le_gamma*180/torch.pi+90)/180
            c_te_gamma=(c_te_gamma*180/torch.pi+50)/70
            t_te_gamma=-(t_te_gamma*180/torch.pi)/60
            
            MSE_z_c1 = F.mse_loss(z_c[:,0], max_camber[:,0]*6, reduction='mean')
            # MSE_z_c2 = F.mse_loss(z_c[:,1], c_le_gamma, reduction='mean')
            
            MSE_z_c2 = F.mse_loss(z_c[:,1], c_te_gamma, reduction='mean')
            # MSE_z_c3 = F.mse_loss(z_c[:,2], max_camber_loc, reduction='mean')
            
            MSE_z_t1 = F.mse_loss(z_t[:,0], max_thickness[:,0]*5, reduction='mean')
            MSE_z_t2 = F.mse_loss(z_t[:,1], r_le_log, reduction='mean')
            # MSE_z_t3 = F.mse_loss(z_t[:,2], t_te_gamma, reduction='mean')
            # MSE_z_t3 = F.mse_loss(z_t[:,2], max_thickness_loc*2, reduction='mean')
            
            MSE_z_c3=0
            MSE_z_t3=0
            MSE_z = MSE_z_c1 + MSE_z_c2 + MSE_z_c3 +\
                MSE_z_t1 + MSE_z_t2 + MSE_z_t3
            
            return MSE_z, MSE_z_c1, MSE_z_c2, MSE_z_c3, MSE_z_t1, MSE_z_t2, MSE_z_t3

        MSE_z_phys, MSE_z, MSE_z_c1, MSE_z_c2, MSE_z_c3, MSE_z_t1, MSE_z_t2, MSE_z_t3 =\
            cal_MSE_z(y, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, thickness_curvature, self.num_training_step_call)
        # z_c_min_per_dim = z_c.min(dim=0)[0]
        # z_c_max_per_dim = z_c.max(dim=0)[0]

        # z_t_min_per_dim = z_t.min(dim=0)[0]
        # z_t_max_per_dim = z_t.max(dim=0)[0]

        z_c_min_per_dim = torch.quantile(z_c, 0.003, dim=0)
        z_c_max_per_dim = torch.quantile(z_c, 0.997, dim=0)

        z_t_min_per_dim = torch.quantile(z_t, 0.003, dim=0)
        z_t_max_per_dim = torch.quantile(z_t, 0.997, dim=0)
        # print(z_c_min_per_dim)
        # print(z_c_max_per_dim)
        # Generate random tensors for each dimension
        z_c_random = z_c_min_per_dim + (torch.rand((500, 12), device=self.device) * (z_c_max_per_dim - z_c_min_per_dim))
        z_t_random = z_t_min_per_dim + (torch.rand((500, 12), device=self.device) * (z_t_max_per_dim - z_t_min_per_dim))
        
        r_rand, camber_cp_rand, thickness_cp_rand, camber_curve_rand, thickness_curve_rand,\
            camber_slope_rand, thickness_slope_rand,  camber_curvature_rand, thickness_curvature_rand = self.decode(z_c_random, z_t_random)
        
        MSE_z_rand = cal_MSE_z2(z_c_random, z_t_random, camber_cp_rand, thickness_cp_rand, camber_curve_rand, thickness_curve_rand, thickness_curvature_rand)[0]
        
        
        if R1.item() > 0 or R2.item() > 0:
            # print('R1: {:.2e}, R2: {:.2e}'.format(R1.item(), R2.item()))
            self.num_constraint_violation+=1
        # # print(prof)
        if self.num_training_step_call % 100 == 1:
            print('MSE: {:.2e}, KLD_c: {:.2e}, KLD_t: {:.2e}, R1: {:.2e}, R2: {:.2e}, L1: {:.2e}, L2: {:.2e}'.format(\
                MSE.item(), KLD_c.item(), KLD_t.item(), R1.item(), R2.item(),  L1.item(), L2.item()))
            # print('MSE_zc1: {:.2e}, MSE_zc2: {:.2e}, MSE_zc3: {:.2e},  MSE_zt1: {:.2e}, MSE_zt2: {:.2e}, MSE_zt3: {:.2e}'.format(\
            #     MSE_z_c1.item(), MSE_z_c2.item(), MSE_z_c3.item(), MSE_z_t1.item(), MSE_z_t2.item(), MSE_z_t3.item()))
            print('MSE_zc1: {:.2e}, MSE_zc2: {:.2e}, MSE_zt1: {:.2e}, MSE_zt2: {:.2e},'.format(\
                MSE_z_c1.item(), MSE_z_c2.item(),  MSE_z_t1.item(), MSE_z_t2.item()))
            
            print('MSE_z_phys: {:.2e}, MSE_z: {:.2e}, MSE_z_rand: {:.2e}'.format(MSE_z_phys.item(),MSE_z.item(), MSE_z_rand.item()))
            
            print('Number of constraint violation during training interval: {:d}'.format(self.num_constraint_violation))
            self.num_constraint_violation=0
        
            
            plt.figure(figsize=(15,4))
            plt.subplot(1,2,1)
            plt.plot(x[0,:,0].detach().cpu().numpy(), x[0,:,1].detach().cpu().numpy(),'k-')
            plt.plot(r[0,:,0].detach().cpu().numpy(), r[0,:,1].detach().cpu().numpy(),'r--')
            
            plt.subplot(1,2,2)
            plt.plot(camber_cp[0,:,0].detach().cpu().numpy(),camber_cp[0,:,1].detach().cpu().numpy(),'bo--',alpha=0.5)
            plt.plot(camber_curve[0,0,:].detach().cpu().numpy(),camber_curve[0,1,:].detach().cpu().numpy(),'b')
            # plt.plot(max_camber_loc[0].detach().cpu().numpy(),max_camber[0].detach().cpu().numpy(),'bo')
            # plt.plot(z_c[0,1].detach().cpu().numpy(),z_c[0,0].detach().cpu().numpy()/10,'bx')
            
            plt.plot(thickness_cp[0,:,0].detach().cpu().numpy(),thickness_cp[0,:,1].detach().cpu().numpy(),'ro--',alpha=0.5)
            plt.plot(thickness_curve[0,0,:].detach().cpu().numpy(),thickness_curve[0,1,:].detach().cpu().numpy(),'r')
            # plt.plot(max_thickness_loc[0].detach().cpu().numpy(),max_thickness[0].detach().cpu().numpy(),'ro')
            # plt.plot(z_t[0,1].detach().cpu().numpy(),z_t[0,0].detach().cpu().numpy()/10,'rx')
            
            plt.show()
            z_c_np=z_c.detach().cpu().numpy()
            z_t_np=z_t.detach().cpu().numpy()
            y_np=y.detach().cpu().numpy()
            z_np=np.concatenate((z_c_np, z_t_np), axis=1)
            # plt.figure(figsize=(15,8))
            # p=0
            # for i in range(12):
            #     plt.subplot(2,6,i+1)
            #     plt.hist(z_np[:,i], bins=np.linspace(-5,5,201))
            #     plt.xlim([z_np[:,i].min(), z_np[:,i].max()])
            #     if i in [0,1,2,6,7,8]:
            #         plt.hist(y_np[:,p], bins=np.linspace(-5,5,201), alpha=0.2, edgecolor='k', color='darkblue')
            #         plt.xlim([y_np[:,p].min(), y_np[:,p].max()])
            #         p+=1
                    
                
            #     # plt.xticks(np.arange(11), np.linspace(-5,5,11))
            # plt.show()
 
        # return MSE + self.beta * (KLD_c+KLD_t) + 1e-3*(R1+R2) + 3e-5*(MSE_z_phys+MSE_z+MSE_z_rand) + 1e-10*(L1+L2)
  
        # return MSE + self.beta * (KLD_c+KLD_t) + 3e-5*(MSE_z_phys+MSE_z+MSE_z_rand)
        return MSE + self.beta * (KLD_c+KLD_t) + 1e-5*(MSE_z+MSE_z_rand)
        # return MSE + self.beta * (KLD_c+KLD_t) + 1e-3*(R1+R2) + 3e-5*(MSE_z) + 1e-10*(L1+L2)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.num_training_step_call+=1
        self.train()
        x, y = train_batch
        # stamp0=time.time()
        r, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope,  camber_curvature, thickness_curvature,  mu_c, logvar_c, mu_t, logvar_t= self(x)
        # stamp1=time.time()
        loss = self.loss_fn(r, x, y, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope, camber_curvature, thickness_curvature,  mu_c, logvar_c, mu_t, logvar_t)
        # stamp2=time.time()
        
        # print('foward: {:.2e}'.format(stamp1-stamp0))
        # print('loss: {:.2e}'.format(stamp2-stamp1))
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        x, y = val_batch
        r, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope,  camber_curvature, thickness_curvature,  mu_c, logvar_c, mu_t, logvar_t= self(x)

        loss = self.loss_fn(r, x, y, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope, camber_curvature, thickness_curvature,  mu_c, logvar_c, mu_t, logvar_t)
        
        return loss
    
    
    
    
    
    
    
    def training_step_after(self, z_c, z_t):
        # self.num_training_step_call+=1
        self.train()
        # stamp0=time.time()
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature = self.decode(z_c, z_t)
       
        # stamp1=time.time()
        loss = self.loss_fn_after(r, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope, camber_curvature, thickness_curvature)
        # stamp2=time.time()
        
        # print('foward: {:.2e}'.format(stamp1-stamp0))
        # print('loss: {:.2e}'.format(stamp2-stamp1))
        return loss

    def validation_step_after(self, z_c, z_t):
        self.eval()
        
        r, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature = self.decode(z_c, z_t)

        loss = self.loss_fn_after(r,z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope, camber_curvature, thickness_curvature)
        
        return loss
    
    def loss_fn_after(self, r, z_c, z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, \
            thickness_slope,  camber_curvature, thickness_curvature):
  
  

        camber_cp_length=torch.linalg.norm(camber_cp[:,1:]-camber_cp[:,:-1], dim=2)
        thickness_cp_length=torch.linalg.norm(thickness_cp[:,1:]-thickness_cp[:,:-1], dim=2)
    
        curvature_threshold=50
        mask_camber= camber_curve[:,0,:] > 0.25
        mask_camber_curvature = torch.where(mask_camber, camber_curvature, torch.zeros_like(camber_curvature, device=self.device))
   
        mask_thickness= thickness_curve[:,0,:] > 0.25
        mask_thickness_curvature = torch.where(mask_thickness, thickness_curvature, torch.zeros_like(thickness_curvature, device=self.device))

        max_curvature_camber=torch.max(torch.abs(mask_camber_curvature), dim=1)[0]
        max_curvature_thickness=torch.max(torch.abs(mask_thickness_curvature), dim=1)[0]
        
        max_camber, max_camber_idx=torch.max(torch.abs(camber_curve[:,1:,:]), dim=2)
        max_thickness, max_thickness_idx=torch.max(thickness_curve[:,1:,:], dim=2)
    
        R1= torch.maximum(torch.zeros_like(max_curvature_camber, device=self.device), max_curvature_camber-torch.ones_like(max_curvature_camber, device=self.device)*curvature_threshold)
        R2= torch.maximum(torch.zeros_like(max_curvature_thickness, device=self.device), max_curvature_thickness-torch.ones_like(max_curvature_thickness, device=self.device)*curvature_threshold)
        R1=torch.mean(R1)
        R2=torch.mean(R2)

        r_le=-1/thickness_curvature[:,0]
        r_le_log=(torch.log10(r_le)+2.5)/2
 
        c_le_gamma = torch.arctan(camber_cp[:,0,1]-camber_cp[:,1,1]/(camber_cp[:,0,0]-camber_cp[:,1,0]))
        c_te_gamma = torch.arctan(camber_cp[:,-1,1]-camber_cp[:,-2,1]/(camber_cp[:,-1,0]-camber_cp[:,-2,0]))
        t_te_gamma = torch.arctan(thickness_cp[:,-1,1]-thickness_cp[:,-2,1]/(thickness_cp[:,-1,0]-thickness_cp[:,-2,0]))
       
        
        c_le_gamma=(c_le_gamma*180/torch.pi+10)/60
        c_te_gamma=(c_te_gamma*180/torch.pi+30)/50
        t_te_gamma=(t_te_gamma*180/torch.pi+10)/10
          
        MSE_z_c1 = F.mse_loss(z_c[:,0], max_camber[:,0]*10, reduction='mean')
        MSE_z_c2 = F.mse_loss(z_c[:,1], c_le_gamma, reduction='mean')
        MSE_z_c3 = F.mse_loss(z_c[:,2], c_te_gamma, reduction='mean')
        
        # MSE_z_c2 = F.mse_loss(z_c[:,1], c_te_gamma, reduction='mean')
        # MSE_z_c2 = F.mse_loss(z_c[:,1], max_camber_loc, reduction='mean')
        
        denominator = max_thickness[:,0]*10 + 1e-10
        MSE_z_t1 = F.mse_loss(z_t[:,0], max_thickness[:,0]*5, reduction='mean')
        MSE_z_t2 = F.mse_loss(z_t[:,1], r_le_log/denominator, reduction='mean')
        MSE_z_t3 = F.mse_loss(z_t[:,2], t_te_gamma, reduction='mean')
        

        MSE_z = MSE_z_c1 + MSE_z_c2 + MSE_z_c3 +\
            MSE_z_t1 + MSE_z_t2 + MSE_z_t3
        

        L1=torch.mean(torch.sum(1/camber_cp_length**2, dim=1))
        L2=torch.mean(torch.sum(1/thickness_cp_length[:,1:]**2, dim=1))
        
        if R1.item() > 0 or R2.item() > 0:
            # print('R1: {:.2e}, R2: {:.2e}'.format(R1.item(), R2.item()))
            self.num_constraint_violation+=1
        # # print(prof)
        if self.num_training_step_call % 100 == 1:
            print('After Training')
            print('R1: {:.2e}, R2: {:.2e}, L1: {:.2e}, L2: {:.2e}'.format(\
               R1.item(), R2.item(),  L1.item(), L2.item()))
            print('MSE_zc1: {:.2e}, MSE_zc2: {:.2e}, MSE_zc3: {:.2e},  MSE_zt1: {:.2e}, MSE_zt2: {:.2e}, MSE_zt3: {:.2e}'.format(\
                MSE_z_c1.item(), MSE_z_c2.item(), MSE_z_c3.item(), MSE_z_t1.item(), MSE_z_t2.item(), MSE_z_t3.item()))
            print('Number of constraint violation during training interval: {:d}'.format(self.num_constraint_violation))
            self.num_constraint_violation=0
        
        
        return 1e-3*(R1+R2) + 5e-4*MSE_z + 1e-10*(L1+L2)
    
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
    
    




