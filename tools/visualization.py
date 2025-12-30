import numpy as np
import matplotlib.pyplot as plt


def plot_c81_results(c81_mean, c81_std, geom, Ma_grid, AoA_grid, levels_list=[np.linspace(-1.5,2.0,21), np.linspace(0,0.6,21), np.linspace(-0.35, 0.35,21)]):
    plt.rcParams['font.family']='times new roman'
    
    plt.figure(figsize=(20,35))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    
    p=0
    for qoi in [0,1,-1]:
        plt.subplot(12,4,12*p+1)
        plt.plot(geom[:,0], geom[:,1],'k-')
        plt.axis('equal')
        
        plt.subplot(12,4,12*p+2)
        plt.plot(geom[:,0], geom[:,1],'k-')
        # plt.axis('equal')
        
        plt.subplot(12,4,12*p+3)
        if qoi==1:
            
            plt.contour(AoA_grid, Ma_grid, np.exp(c81_mean[:,:,qoi]),levels=levels_list[qoi], colors='k',linewidths=1)
            plt.contourf(AoA_grid, Ma_grid, np.exp(c81_mean[:,:,qoi]),levels=levels_list[qoi], cmap='jet')
        else:
            
            plt.contour(AoA_grid, Ma_grid, c81_mean[:,:,qoi],levels=levels_list[qoi], colors='k',linewidths=1)
            plt.contourf(AoA_grid, Ma_grid, c81_mean[:,:,qoi],levels=levels_list[qoi], cmap='jet')
        plt.title('Mean')
        plt.colorbar()
        
        plt.subplot(12,4,12*p+4)
        
        plt.contour(AoA_grid, Ma_grid, c81_std[:,:,qoi], colors='k',linewidths=1)
        plt.contourf(AoA_grid, Ma_grid, c81_std[:,:,qoi], cmap='jet')
        plt.title('Std')
        plt.colorbar()
        
        for mach_idx in range(9):
            plt.subplot(12,5,15*p+6+mach_idx)
            
            if qoi==1:
                plt.plot(AoA_grid, 10**(c81_mean[mach_idx,:,qoi]), '-', c='C'+str(mach_idx), label='Mean')
                plt.fill_between(AoA_grid, 10**(c81_mean[mach_idx,:,qoi]+3*c81_std[mach_idx,:,qoi]),\
                    10**(c81_mean[mach_idx,:,qoi]-3*c81_std[mach_idx,:,qoi]), color='C'+str(mach_idx), alpha=.2, label='$\mu \pm 3\sigma$')
                
            else:
                plt.plot(AoA_grid, c81_mean[mach_idx,:,qoi], '-', c='C'+str(mach_idx), label='Mean')
                plt.fill_between(AoA_grid, c81_mean[mach_idx,:,qoi]+3*c81_std[mach_idx,:,qoi],\
                    c81_mean[mach_idx,:,qoi]-3*c81_std[mach_idx,:,qoi], color='C'+str(mach_idx), alpha=.2, label='$\mu \pm 3\sigma$')
            plt.legend(frameon=False)
            plt.ylim([levels_list[qoi].min(), levels_list[qoi].max()])
            plt.title('Ma: {:.2f}'.format(Ma_grid[mach_idx]))
            plt.grid(True,'both')
        p+=1

    plt.show()
    
    
def plot_c81_results_with_ref(c81_mean, c81_std, geom, c81_ref, Ma_grid, AoA_grid, \
    levels_list=[np.linspace(-1.5,2.0,21), np.linspace(0,0.6,21), np.linspace(-0.35, 0.35,21)], save=False, save_path='./figures/plot_c81.png'):
    plt.rcParams['font.family']='times new roman'
    

    plt.figure(figsize=(20,35))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    p=0
    for qoi in [0,1,-1]:
        plt.subplot(12,4,12*p+1)
        plt.plot(geom[:,0], geom[:,1],'k-')
        plt.axis('equal')
        
        plt.subplot(12,4,12*p+2)
        plt.plot(geom[:,0], geom[:,1],'k-')
        # plt.axis('equal')
        
        plt.subplot(12,4,12*p+3)
        if qoi==1:
            
            plt.contour(AoA_grid, Ma_grid, np.exp(c81_mean[:,:,qoi]),levels=levels_list[qoi], colors='k',linewidths=1)
            plt.contourf(AoA_grid, Ma_grid, np.exp(c81_mean[:,:,qoi]),levels=levels_list[qoi], cmap='jet')
        else:
            
            plt.contour(AoA_grid, Ma_grid, c81_mean[:,:,qoi],levels=levels_list[qoi], colors='k',linewidths=1)
            plt.contourf(AoA_grid, Ma_grid, c81_mean[:,:,qoi],levels=levels_list[qoi], cmap='jet')
        plt.title('Mean')
        plt.colorbar()
        
        plt.subplot(12,4,12*p+4)
        
        plt.contour(AoA_grid, Ma_grid, c81_std[:,:,qoi], colors='k',linewidths=1)
        plt.contourf(AoA_grid, Ma_grid, c81_std[:,:,qoi], cmap='jet')
        plt.title('Std')
        plt.colorbar()
    
        for mach_idx in range(9):
            plt.subplot(12,5,15*p+6+mach_idx)
            if qoi==1:
                plt.plot(AoA_grid, np.exp(c81_ref[mach_idx,:,qoi]), '-', c='k', label='Ref.')
                
                plt.plot(AoA_grid, np.exp(c81_mean[mach_idx,:,qoi]), '-', c='C'+str(mach_idx), label='Mean')
                plt.fill_between(AoA_grid, np.exp(c81_mean[mach_idx,:,qoi]+3*c81_std[mach_idx,:,qoi]),\
                    np.exp(c81_mean[mach_idx,:,qoi]-3*c81_std[mach_idx,:,qoi]), color='C'+str(mach_idx), alpha=.2, label='$\mu \pm 3\sigma$')
                
            else:
                plt.plot(AoA_grid, c81_ref[mach_idx,:,qoi], '-', c='k', label='Ref.')
                
                plt.plot(AoA_grid, c81_mean[mach_idx,:,qoi], '-', c='C'+str(mach_idx), label='Mean')
                plt.fill_between(AoA_grid, c81_mean[mach_idx,:,qoi]+3*c81_std[mach_idx,:,qoi],\
                    c81_mean[mach_idx,:,qoi]-3*c81_std[mach_idx,:,qoi], color='C'+str(mach_idx), alpha=.2, label='$\mu \pm 3\sigma$')
            plt.legend(frameon=False)
            plt.ylim([levels_list[qoi].min(), levels_list[qoi].max()])
            plt.title('Ma: {:.2f}'.format(Ma_grid[mach_idx]))
            plt.grid(True,'both')
        p+=1
    if save:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
def plot_target_airfoil(target_lower, target_upper):
    plt.rcParams['font.family']='times new roman'
    plt.figure(figsize=(15,8))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.subplot(2,2,1)
    plt.plot(target_lower[:,0], target_lower[:,1],'r-',label='Lower surface',alpha=0.7)
    plt.plot(target_upper[:,0], target_upper[:,1],'b-',label='Upper surface',alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(frameon=False)

    plt.subplot(2,2,2)
    plt.plot(target_lower[:,0], target_lower[:,1],'r-',label='Lower surface',alpha=0.7)
    plt.plot(target_upper[:,0], target_upper[:,1],'b-',label='Upper surface',alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

    plt.subplot(2,2,3)
    plt.plot(target_lower[:,0],'r-')
    plt.plot(target_upper[:,0],'b-')
    plt.ylabel('X - coordinate')
    plt.xlabel('Data index')

    plt.title('Note: The x-coordinate of target airfoil data should be ordered as non-decreasing \n (start at 0 and end at 1).')

    plt.subplot(2,2,4)
    plt.plot(target_lower[:,1],'r-')
    plt.plot(target_upper[:,1],'b-')
    plt.ylabel('Y - coordinate')
    plt.xlabel('Data index')
    plt.show()
