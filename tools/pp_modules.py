import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def eval_AUCE(fun, data_x, data_y, scale_factor=np.array([1,1,1]), plot=False):
    def is_in_CI(upper_CI, lower_CI, data_y):
        within_upper = data_y <= upper_CI
        within_lower = data_y >= lower_CI
        return np.logical_and(within_upper, within_lower)
    
    mu, var, var2 = fun(data_x, norm=False)
    var*=var
    var2[0]*=var2[0]
    var2[1]*=var2[1]
    
    for i in range(3):
        var2[0][:,i]=scale_factor[i]*var2[0][:,i]
        var2[1][:,i]=scale_factor[i]*var2[1][:,i]
    # for i in range(3,6):
        # var2[1][:,i-3]=scale_factor[i]*var2[1][:,i-3]
    
    var_cal=var2[0]+var2[1]
    
    def data_flatten(data):
        temp=[]
        for i in range(3):
            temp.append(data[:,i].flatten())
        return np.array(temp).T
    
    data_y=data_flatten(data_y)
    mu=data_flatten(mu)
    var=data_flatten(var)
    
    # var_cal=var*scale_factor

    eps=1e-5
    p_values = np.linspace(0.5,1-eps,11)
    f_values = [st.norm.ppf(p) for p in p_values]
    
    score=[]
    score_cal=[]
    score_true=[]
    
    for i in range(p_values.shape[0]):
        upper_CI=mu+f_values[i]*var**0.5
        lower_CI=mu-f_values[i]*var**0.5
        
        upper_CI_cal=mu+f_values[i]*var_cal**0.5
        lower_CI_cal=mu-f_values[i]*var_cal**0.5
        con=is_in_CI(upper_CI, lower_CI, data_y)
        con_cal=is_in_CI(upper_CI_cal, lower_CI_cal, data_y)
        score.append(sum(con)/len(con))
        score_cal.append(sum(con_cal)/len(con_cal))
        score_true.append((p_values[i]-0.5)*2)
    score=np.array(score)
    score_cal=np.array(score_cal)
    
    score_true=np.array([score_true,score_true,score_true]).T
    if plot:
        # plt.plot(score_true, score,'s-',markerfacecolor='none')    
        # plt.plot([0,1],[0,1],'k--',label='Ideal')
        
        for i in range(3):
            plt.figure(figsize=(8,6),dpi=150)
            # plt.subplot(1,3,i+1)
            
            # plt.bar(score_true[:,i], score[:,i], color='darkblue', edgecolor='k', alpha=0.3,width=0.1, label='Uncalibrated')    
            plt.bar(score_true[:,i], score_cal[:,i], color='red', edgecolor='k', alpha=0.7,width=0.1, label='AUCE interval')    
            plt.plot([0,1],[0,1],'k--',label='Ideal')
            
            plt.grid(True)
            plt.legend(frameon=False)
            # plt.axis('equal')
            plt.xlabel('Predicted CI')
            plt.ylabel('Observed CI')
            plt.axis([0,1,0,1])
            plt.title('AUCE: {:.2e}'.format(np.mean(np.abs(score[:,i]-score_true[:,i]))))
            plt.show()
    return np.mean(np.abs(score-score_true),axis=0)

def eval_ENCE(fun, data_x, data_y, scale_factor=np.array([1,1,1]), plot=False):

    mu, var,var2 = fun(data_x, norm=False)
    # var*=var
    var2[0]*=var2[0]
    var2[1]*=var2[1]
    
    for i in range(3):
        var2[0][:,i]=scale_factor[i]*var2[0][:,i]
        var2[1][:,i]=scale_factor[i]*var2[1][:,i]
    # for i in range(3,6):
    #     var2[1][:,i-3]=scale_factor[i]*var2[1][:,i-3]
    
    var=var2[0]+var2[1]
    
    def data_flatten(data):
        temp=[]
        for i in range(3):
            temp.append(data[:,i].flatten())
        return np.array(temp).T
    
    data_y=data_flatten(data_y)
    mu=data_flatten(mu)
    var=data_flatten(var)
   
    sort_idx=[]
    for i in range(3):
        temp=np.argsort(data_y[:,i])
        sort_idx.append(temp)
    sort_idx=np.array(sort_idx).T

    # var*=scale_factor
    n_bins = sort_idx.shape[0]-1
    n_bins = 20
    number_sample_bins=int(sort_idx.shape[0]/(n_bins-1))
    MSE_bin=[]
    var_bin=[]
    
    for i in range(n_bins):
        MSE_temp=[]
        var_temp=[]
        for j in range(3):
            if not i == n_bins-1:
                bin_idx = sort_idx[number_sample_bins*i:number_sample_bins*(i+1), j]
            else:
                bin_idx = sort_idx[number_sample_bins*i:, j]
                
            MSE = np.mean((data_y[bin_idx,j]-mu[bin_idx,j])**2, axis=0)
            MSE_temp.append(MSE**0.5)
            var_temp.append(np.mean(var[bin_idx, j], axis=0)**0.5)
            
        MSE_temp=np.array(MSE_temp)
        var_temp=np.array(var_temp)
  
        MSE_bin.append(MSE_temp)
        var_bin.append(var_temp)
    MSE_bin=np.array(MSE_bin)
    var_bin=np.array(var_bin)
    if plot:
        plt.plot(MSE_bin, var_bin,'s',markerfacecolor='none')
        plt.plot([MSE_bin.min(), MSE_bin.max()],[MSE_bin.min(), MSE_bin.max()],'k--')
    
        plt.grid(True)
        plt.title('ENCE: {:.2e}'.format(np.mean(np.abs(MSE_bin-var_bin)/var_bin)))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    return np.mean(np.abs(MSE_bin-var_bin)/var_bin, axis=0)


class STD_Calibration_Problem():
    def __init__(self, pp, data_x ,data_y):
        self.pp=pp
        self.data_x=data_x
        # self.data_x_norm=(data_x-np.mean(data_x, axis=0))/np.std(data_x, axis=0)
        self.data_y=data_y
        self.data_y_norm=pp.model_perform_tot[0].model_init.DP_Y.transform(data_y).detach().cpu().numpy()
        self.function_call=0
        
    def obj_function(self, scale_factor):
        scale_factor=scale_factor
    
       
        mu_pred_avg, var_pred_avg, var2_pred_avg=self.pp.evaluate_performance3(eval_con=self.data_x, norm=False)
        var2_pred_avg[0]*=var2_pred_avg[0]
        var2_pred_avg[1]*=var2_pred_avg[1]
        for i in range(3):
            var2_pred_avg[0][:,i]=scale_factor[i]*var2_pred_avg[0][:,i]
            var2_pred_avg[1][:,i]=scale_factor[i]*var2_pred_avg[1][:,i]
        # plt.plot(mu_pred_avg[:,0],self.data_y[:,0],'k.')
        # plt.show()
        var_calibrated=var2_pred_avg[0]+var2_pred_avg[1]
        SE=(mu_pred_avg-self.data_y_norm)**2
        NLL = np.mean((np.log(var_pred_avg) + SE/var_pred_avg))
        NLL = np.mean((np.log(var_calibrated) + SE/var_calibrated))
        # print(np.mean(SE))
        # print(NLL)
        # if self.function_call % 100 == 0:
        #     print('F_call: {:d}, NLL: {:.12e}'.format(self.function_call, NLL))
        return NLL
    
    def preprocess_data_x(self, data_x):
        Ma_grid = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8])
        AoA_grid = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])

        # Create a mesh grid, reshape, and normalize it
        mesh = np.array(np.meshgrid(Ma_grid, AoA_grid)).T.reshape(-1, 2)
        mesh_normalized = (mesh - np.mean(mesh, axis=0)) / np.std(mesh, axis=0)

        eval_con = data_x  # Example shape (100, 3), adjust as per your actual use case

        tiles_count = eval_con.shape[0]  # Number of times to repeat the mesh
        tiled_mesh = np.tile(mesh_normalized, (tiles_count, 1))

        eval_con_repeated = np.repeat(eval_con, len(mesh_normalized), axis=0)

        combined = np.concatenate((eval_con_repeated, tiled_mesh), axis=1)

        combined = combined.astype(float)
        return combined
    
def evaluate_pp_nll(pp, data_eval_x, data_eval_y):
    mu_pred_avg, var_pred_avg, var_pred_avg2=pp.evaluate_performance3(eval_con=data_eval_x, norm=False)
    y_norm=pp.model_perform_tot[0].model_init.DP_Y.transform(data_eval_y).detach().cpu().numpy()
    var_pred_avg*=var_pred_avg
    SE=(mu_pred_avg-y_norm)**2
    NLL = np.mean((np.log(var_pred_avg) + SE/var_pred_avg))
    return NLL
    
def evaluate_pp_performance(pp, data_eval_x, data_eval_y, scale_factor=np.array([1,1,1]) ,plot=False):
    mu_pred_avg, var_pred_avg, var_pred_avg2=pp.evaluate_performance3(eval_con=data_eval_x, norm=False)
    mu_pred_avg_invnorm, var_pred_avg_invnorm, var_pred_avg_invnorm2=pp.evaluate_performance3(eval_con=data_eval_x, norm=True)
    y_norm=pp.model_perform_tot[0].model_init.DP_Y.transform(data_eval_y).detach().cpu().numpy()
    var_pred_avg*=var_pred_avg
    var_pred_avg_invnorm*=var_pred_avg_invnorm
    # scale_factor=np.array([0.57,0.57,0.57])
    # var_pred_avg_invnorm*=scale_factor[np.newaxis,:]
    
    AUCE=eval_AUCE(pp.evaluate_performance3, data_eval_x, y_norm, scale_factor=scale_factor, plot=True)
    ENCE=eval_ENCE(pp.evaluate_performance3, data_eval_x, y_norm,  scale_factor=scale_factor, plot=plot)
    SE=(mu_pred_avg-y_norm)**2
    NLL = np.mean((np.log(var_pred_avg) + SE/var_pred_avg))
    
    var_pred_avg_cal_temp0=var_pred_avg2[0]**2*scale_factor[np.newaxis,:3]
    var_pred_avg_cal_temp1=var_pred_avg2[1]**2*scale_factor[np.newaxis,:3]
    var_pred_avg_cal=var_pred_avg_cal_temp0+var_pred_avg_cal_temp1
    NLL_cal = np.mean((np.log(var_pred_avg_cal) + SE/var_pred_avg_cal))
    MSE = np.mean((mu_pred_avg-y_norm)**2,axis=0)
    # NRMSE=np.mean(np.mean(np.mean((mu_pred_avg_invnorm-data_eval_y)**2/(np.abs(data_eval_y)+1e-4),axis=-1),axis=-1),axis=0)
    
    normalize_factor=np.zeros(3)
    for i in range(3):
        normalize_factor[i]=data_eval_y[:,i].max()-data_eval_y[:,i].min()
    
    NRMSE=MSE**0.5/normalize_factor*100
    temp=np.abs(mu_pred_avg_invnorm-data_eval_y)
    
    if plot:
        Ma_grid=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8])
        AoA_grid=np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
        n_show=3
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        plt.subplots_adjust(wspace=0.5)
        for j, ax1 in enumerate(axes):
            ax2 = ax1.twinx()  # Create a twin of ax1 for the secondary y-axis
            
            ax1.bar(AoA_grid, var_pred_avg_invnorm[n_show, j, 5, :]**0.5, color='w', alpha=0.5, hatch='///', edgecolor='k')
            ax1.bar(AoA_grid, temp[n_show, j, 5, :], color='C{:d}'.format(j), alpha=0.5, edgecolor='k')
            ax1.set_ylim([1e-4, 1e-1])  # Set the limits for the secondary y-axis
            # ax2.set_yscale('log')
            ax1.set_ylabel('MAE')
            
            ax2.plot(AoA_grid, data_eval_y[n_show, j, 5, :], '-', c='k'.format(j))
            ax2.fill_between(AoA_grid, mu_pred_avg_invnorm[n_show, j, 5, :]+3*var_pred_avg_invnorm[n_show, j, 5, :]**0.5,\
                mu_pred_avg_invnorm[n_show, j, 5, :]-3*var_pred_avg_invnorm[n_show, j, 5, :]**0.5, color='C{:d}'.format(j),alpha=0.3)
            ax2.plot(AoA_grid, mu_pred_avg_invnorm[n_show, j, 5, :], '--', c='C{:d}'.format(j))
            ax2.grid(True)  # Enable grid
            ax2.set_ylabel('Aero. Coeff.')
        plt.show()
        
        print('M: {:d}, NLL: {:.3e}'.format(pp.M, NLL))
        print('     MSE: {:.3e},   {:.3e},   {:.3e}'.format(MSE[0], MSE[1], MSE[2]))
        print('   NRSME: {:.3e},   {:.3e},   {:.3e}'.format(NRMSE[0], NRMSE[1], NRMSE[2]))
        print('    AUCE: {:.3e},   {:.3e},   {:.3e}'.format(AUCE[0], AUCE[1], AUCE[2]))
        print('    ENCE: {:.3e},   {:.3e},   {:.3e}'.format(ENCE[0], ENCE[1], ENCE[2]))
    
    return MSE, NRMSE, AUCE, ENCE, NLL, NLL_cal, scale_factor
   
   
class Voronoi_cell:
    def __init__(self, train_sample):
        self.train_sample=train_sample
        self.mean=np.mean(train_sample,axis=0)
        self.std=np.std(train_sample,axis=0)
        self.train_sample_norm=(train_sample-self.mean)/self.std

        
    def get_voronoi_idx(self, sample):
        sample_norm=(sample-self.mean)/self.std
        diffs = sample_norm[:, np.newaxis, :] - self.train_sample_norm[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        return np.argmin(dists, axis=1)
    
    def get_nearest_idx(self, sample):
        sample_norm=(sample-self.mean)/self.std
        diffs = sample_norm[:, np.newaxis, :] - self.train_sample_norm[np.newaxis, :, :]
        diffs = np.abs(diffs)
        return np.argmin(diffs,axis=1)