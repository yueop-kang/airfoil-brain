import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader, TensorDataset
from tools.normalize import DataProcessing, MeanStd_channel2, MeanStd_channel, Center
from models.cae import CAE
from models.cvae import CVAE
from models.ae import AE
from models.mlp import MLP
from cmd import IDENTCHARS
from re import I
from torch.utils.data import DataLoader
from tools.normalize import DataProcessing, MeanStd_channel2
from models.TC_VAE_fin import TC_VAE
from models.Gaussian_mlp_abrain import Gaussian_mlp
import numpy as np
import matplotlib.pyplot as plt
from tools.dataread import load_airfoil
from tools.ag_modules import find_latent
import scipy
from sklearn.model_selection import LeaveOneOut, KFold


my_seed = 42
def set_seed (my_seed = 42):
  np.random.seed(my_seed)
  torch.manual_seed(my_seed)
  torch.cuda.manual_seed(my_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
set_seed (my_seed)

import copy

# Define class
class Airfoil_Generator():

    def __init__(self, device='cuda'): 
   
        tr_airfoil=np.load('./model_save/ag_model_save/geom_tr.npy')
        train_airfoil=np.load('./model_save/ag_model_save/geom_train.npy')

        config = {"input_shape" : [train_airfoil.shape[1], train_airfoil.shape[2]]}
        params_grid={
            "beta": 2.5e-8, # this value is dummy
            "h_channels": [32,32,32,32,32],
            "h_kernel": 7,
            "h_padding": 3,
            "activation_f":'gelu',
            "reduced_scale": [2,2,2,5,5],
            "z_dims":12,
            "batch_norm": True,
            "bspline_degree": 3,
            "num_cp_t": 12,
            "num_cp_c": 12
        }

        self.device = device

        self.DP=DataProcessing(MeanStd_channel(tr_airfoil), self.device)
        
        mean_std=[torch.tensor(self.DP.normalizer.mean).float().to(self.device), torch.tensor(self.DP.normalizer.std).float().to(self.device)]
        model=TC_VAE(config, params_grid,device=self.device, mean_std=mean_std).to(self.device)
        
        inactive_latent=[2,3,4,5,6,8,10,14,15,16,17,19,22,23]
        # self.std_var='5-sigma'
        self.std_var='uniform'
        self.num_var=model.z_dims*2-len(inactive_latent)
        self.inactive_latent=inactive_latent
        self.active_latent=[]
        for i in range(model.z_dims*2):
            if not i in inactive_latent:
                self.active_latent.append(i)
        model.eval()
        
        # Model load
        model_load_path='./model_save/ag_model_save/ag_dv10.torch'
        self.model=model.to(self.device)
        self.model.load_state_dict(torch.load(model_load_path, map_location=self.device))

        # data pre-processing (initialize normalizer)
        self.train_airfoil = train_airfoil
        self.train_dataset=self.DP.transform(train_airfoil)

        # encode train/test airfoil to obtain latent code
        dataloader=DataLoader(self.train_dataset, batch_size=train_airfoil.shape[0])
        for (idx, batch) in enumerate(dataloader):
            # self.model.eval()

            r, self.train_z_c, self.train_z_t, camber_cp, thickness_cp, camber_curve, thickness_curve, \
                camber_slope, thickness_slope, camber_curvature, thickness_curvature, \
                    mu_c, logvar_c,  mu_t, logvar_t=self.model(batch)
            
    
        self.train_z_c=self.train_z_c.detach().cpu().numpy()
        self.train_z_t=self.train_z_t.detach().cpu().numpy()
        self.train_z=np.concatenate((self.train_z_c, self.train_z_t), axis=1)
       
        # obtain min/max/mean/std value of latent code 
        self.z_min=np.min(self.train_z, axis=0)
        self.z_max=np.max(self.train_z, axis=0)
        self.z_mean=np.mean(self.train_z, axis=0)
        self.z_std=np.std(self.train_z, axis=0)
        
    def append_inactive_latent_code(self, active_latent_code):
        latent_code_append=np.ones((active_latent_code.shape[0],self.model.z_dims*2))*0.5
        p=0
        for i in range(self.model.z_dims*2):
            if not i in self.inactive_latent:
                latent_code_append[:,i]=active_latent_code[:,p].copy()
                p+=1
        
        return latent_code_append
    
    def denormalize_latent_code(self, normalized_latent_code):
  
        z_score=2
        latent_code=2*(normalized_latent_code-0.5)*self.z_std[self.active_latent]*z_score+self.z_mean[self.active_latent]
        latent_code=2*(normalized_latent_code-0.5)*self.z_std[self.active_latent]*z_score+self.z_mean[self.active_latent]

        t_ub=0.25*2.5
        t_lb=0.06*2.5
        c_ub=0.1*6
        c_lb=0.0*6

        latent_code[:,[0]]=normalized_latent_code[:,[0]].copy()*(c_ub-c_lb)+c_lb # max camber range [-0.01, 0.15]
        latent_code[:,[1]]=(normalized_latent_code[:,[1]].copy()-0.5)*2*0.119+(-0.4*latent_code[:,[0]]+0.699) # max camber - te angle correlation
        latent_code[:,[5]]=normalized_latent_code[:,[5]].copy()*(t_ub-t_lb)+t_lb # max thickness range [0.04, 0.4]
        latent_code[:,[6]]=(normalized_latent_code[:,[6]].copy()-0.5)*2*0.34+(0.8*latent_code[:,[5]]+0.240) # max thickness - re radius correlation
                
        return latent_code
    
    def normalize_latent_code(self, latent_code):
        z_score = 2
        normalized_latent_code = (latent_code - self.z_mean[self.active_latent]) / (2 * self.z_std[self.active_latent] * z_score) + 0.5

        t_ub=0.25*2.5
        t_lb=0.06*2.5
        c_ub=0.1*6
        c_lb=0.0*6

        normalized_latent_code[:, [0]] = (latent_code[:, [0]] - c_lb) / (c_ub - c_lb)
        normalized_latent_code[:, [1]] = ((latent_code[:, [1]] - (-0.4 * latent_code[:, [0]] + 0.699)) / (2 * 0.119)) + 0.5
        normalized_latent_code[:, [5]] = (latent_code[:, [5]] - t_lb) / (t_ub - t_lb)
        normalized_latent_code[:, [6]] = ((latent_code[:, [6]] - (0.8 * latent_code[:, [5]] + 0.240)) / (2 * 0.34)) + 0.5

        return normalized_latent_code
    
    def generate_airfoil(self, normalized_latent_code, norm=True, inverse_fit=False):
        # decode
        if norm:
            latent_code_denorm_compact=self.denormalize_latent_code(normalized_latent_code)
            latent_code_norm_compact=normalized_latent_code.copy()
        else:
            latent_code_denorm_compact=normalized_latent_code.copy()
            latent_code_norm_compact=self.normalize_latent_code(normalized_latent_code)
        
        latent_code = self.append_inactive_latent_code(latent_code_denorm_compact)
        for i in self.inactive_latent:
            latent_code[:,i]=np.ones_like(latent_code[:,i])*self.z_mean[i]
            
        latent_code_input=torch.tensor(latent_code).float().to(self.device)
    
        airfoils=self.model.decode(latent_code_input[:,:12], latent_code_input[:,12:])[0]
        airfoils=airfoils.detach().cpu().numpy()
 
        if inverse_fit:
            return airfoils, np.flip(airfoils[:,:101,:], axis=1), airfoils[:,100:]
        else:
            return airfoils, latent_code_norm_compact, latent_code_denorm_compact
    
    def plot_latant_distribution(self, latent_samples=None, norm=True, save_header=None):
        if len(latent_samples)>0:
            latent_code=self.pre_process_latent_code(latent_samples, norm=norm)
        plt.figure(figsize=(20,15))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        for i in range(self.model.z_dims*2):
            plt.subplot(6,4,i+1)
            if i in self.inactive_latent:
                plt.hist(self.train_z[:,i],color='darkred', edgecolor='k',bins=np.linspace(-6,6,51),label='Train z',density=True)
                if len(latent_samples)>0:
                    if len(latent_samples)==1:
                        plt.vlines(latent_code[0,i], 0, 2, linestyles='dashed', color='lightpink')
                    else:
                        plt.hist(latent_code[:,i],color='lightpink', edgecolor='k',bins=np.linspace(-6,6,51),label='Sample z',density=True)
                plt.title('LV {:d} (inactive)'.format(i+1))
            else:
                plt.hist(self.train_z[:,i],color='darkblue', edgecolor='k',bins=np.linspace(-6,6,51),label='Train z',density=True)
                if len(latent_samples)>0:
                    if len(latent_samples)==1:
                        plt.vlines(latent_code[0,i], 0, 0.5, linestyles='dashed', color='deepskyblue')
                    else:
                        plt.hist(latent_code[:,i],color='deepskyblue', edgecolor='k',bins=np.linspace(-6,6,51), label='Sample z',density=True)
                plt.title('LV {:d} (active)'.format(i+1))
            plt.xlim([-6,6])
            plt.legend(frameon=False, loc='upper right')
        if save_header==None:
            plt.show()
        else:
            plt.savefig(save_header+'_latent_histogram.png')
            plt.close()
            

    def find_target_airfoil(self, target_lower, target_upper, plot_results=False, save_header=None):
        
        return find_latent(self, target_lower, target_upper, plot_results=plot_results, save_header=save_header)
    
    
    
class Performance_Predictor():
    def __init__(self, model, model_load_path, device='cuda'):
        self.device = device
        self.model_init=copy.deepcopy(model).to(self.device)
        self.model_init.load(model_load_path, mode='best_loss', device=device)
        self.model_init.eval()
        
        self.model_init.DP_X.config['device']=device
        self.model_init.DP_Y.config['device']=device
        
    def evaluate_performance(self, eval_con):
        num_sample=eval_con.shape[0]
        eval_con=self.model_init.DP_X.transform(eval_con)
        eval_mu, eval_var=self.model_init(eval_con)
        
        return eval_mu, eval_var
 
class Performance_Predictor_Ensemble():
    def __init__(self, M=5, model_path_header='model_save/pp_model_save', device='cuda'):
        self.device = device
        self.M=M
        config_mlp={
            "input_dim": 12,
            "output_dim": 3
        }

        params_mlp={
            "num_hidden_layer": 6,
            "num_hidden_node": 64,
            "activation_f":'tanh',
            "batch_norm": True,
        }
  
        self.model_perform_tot=[]
        for i in range(self.M):
            model = Gaussian_mlp(config_dict=config_mlp, params_dict=params_mlp).to(self.device)
            model_load_path=model_path_header+'/model_{:d}.pickle'.format(i+1)
            self.model_perform_tot.append(Performance_Predictor(model, model_load_path, device=device))
        
        self.scale_factor=np.array([1,1,1])
        
    def evaluate_performance(self, eval_con, norm=True):
        mu_pred_tot=[]
        var_pred_tot=[]
        num_sample=eval_con.shape[0]
        for i in range(self.M):
            mu_pred, var_pred=self.model_perform_tot[i].evaluate_performance(eval_con)
            mu_pred_tot.append(mu_pred)
            var_pred_tot.append(var_pred)
       
        mu_pred_tot=torch.stack(mu_pred_tot)
        var_pred_tot=torch.stack(var_pred_tot)
        mu_pred_avg=torch.mean(mu_pred_tot, dim=0)
        
        var_pred_avg=torch.mean(mu_pred_tot.pow(2), dim=0) + torch.mean(var_pred_tot, dim=0) - mu_pred_avg.pow(2)

        var_pred_avg_ep=torch.mean(mu_pred_tot.pow(2), dim=0) - mu_pred_avg.pow(2)
        var_pred_avg_al=torch.mean(var_pred_tot, dim=0) 
       
        var_pred_avg*=torch.tensor(self.scale_factor).to(self.device)
        var_pred_avg_ep*=torch.tensor(self.scale_factor).to(self.device)
        var_pred_avg_al*=torch.tensor(self.scale_factor).to(self.device)

        if norm:
            mu_pred_avg=mu_pred_avg.detach().cpu().numpy()*self.model_perform_tot[0].model_init.DP_Y.config['scale'] \
                + self.model_perform_tot[0].model_init.DP_Y.config['bias']
            
            var_pred_avg=var_pred_avg.detach().cpu().numpy()*self.model_perform_tot[0].model_init.DP_Y.config['scale']**2
            var_pred_avg_al=var_pred_avg_al.detach().cpu().numpy()*self.model_perform_tot[0].model_init.DP_Y.config['scale']**2
            var_pred_avg_ep=var_pred_avg_ep.detach().cpu().numpy()*self.model_perform_tot[0].model_init.DP_Y.config['scale']**2
        else:
            mu_pred_avg=mu_pred_avg.detach().cpu().numpy()
            var_pred_avg=var_pred_avg.detach().cpu().numpy()
            var_pred_avg_al=var_pred_avg_al.detach().cpu().numpy()
            var_pred_avg_ep=var_pred_avg_ep.detach().cpu().numpy()
                 
        return mu_pred_avg, var_pred_avg**0.5, [np.abs(var_pred_avg_ep)**0.5, var_pred_avg_al**0.5]


class Airfoil_Brain():
    def __init__(self, model_path_header='model_save/pp_model_save', device='cuda'):
        self.device=device
        self.airfoil_generator=Airfoil_Generator(device=self.device)
        self.airfoil_generator_refine=Airfoil_Generator(device=self.device)
        self.geom_save_num_points=201
        self.airfoil_generator_refine.model.tc_layer.num_grid_points=self.geom_save_num_points
        
        self.performance_predictor=Performance_Predictor_Ensemble(model_path_header=model_path_header,device=self.device)
        # self.Ma_grid=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8])
        # self.AoA_grid=np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
        # self.Re_grid=np.array([3e6, 6e6])
        
    def evaluate(self, latent_norm, norm=True):
        geom_ag, latent_code_norm_compact, latent_code_denorm_compact = self.airfoil_generator.generate_airfoil(latent_norm[:,:10], norm=norm)
        latent_pp_inp=np.concatenate([latent_code_norm_compact, latent_norm[:,10:]], axis=1)
        c81_mean, c81_std, c81_std_decompose = self.performance_predictor.evaluate_performance(latent_pp_inp)
        return geom_ag, c81_mean, c81_std, c81_std_decompose

    def geom_save(self, latent_norm, norm=True, save_folder_path='./geom_save'):
        num_geom=latent_norm.shape[0]
        geom_ag, latent_code_norm_compact, latent_code_denorm_compact = self.airfoil_generator_refine.generate_airfoil(latent_norm, norm=norm)
        print(geom_ag.shape)
        geom_lower = np.flip(geom_ag[:,:self.geom_save_num_points], axis=1)
        geom_upper = geom_ag[:,self.geom_save_num_points-1:]
        
        # find aerodynamic center
        ac_tot=[]
        from tools.ag_modules import find_aero_center
        for i in range(num_geom):
            
            ac=find_aero_center(geom_lower[i], geom_upper[i])
            ac_tot.append([ac])
        ac_tot=np.array(ac_tot)
        
        geom_lower_cal=geom_lower-ac_tot
        geom_upper_cal=np.flip(geom_upper-ac_tot, axis=1)
        
        # save geometry in pointwise format
        for i in range(num_geom):
            f=open(save_folder_path+'/geom_{:d}.dat'.format(i+1),mode='w')
            
            f.write('{:d}\n'.format(geom_upper_cal[i].shape[0]))
            for j in range(geom_upper_cal[i].shape[0]):
                f.write('{:12.12f}  {:12.12f}   0.\n'.format(geom_upper_cal[i,j,0], geom_upper_cal[i,j,1]))
            
            f.write('\n')
            
            f.write('{:d}\n'.format(geom_lower_cal[i].shape[0]))
            for j in range(geom_lower_cal[i].shape[0]):
                f.write('{:12.12f}  {:12.12f}   0.\n'.format(geom_lower_cal[i,j,0], geom_lower_cal[i,j,1]))
            f.close()
    
    def find_target_airfoil(self, target_lower, target_upper, plot_results=True):
        fitted_latent, fitted_airfoil, results = \
            self.airfoil_generator.find_target_airfoil(target_lower, target_upper, plot_results=plot_results)
        
        return fitted_latent, fitted_airfoil, results