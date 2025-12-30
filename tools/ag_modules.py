from re import M
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as ip
from scipy.interpolate import splrep, splev

# airfoil surface point redistribution tools

def cal_panel_length(geom):
    geom_shift=np.roll(geom, shift=-1, axis=0)
    panel_length=np.linalg.norm(geom-geom_shift, axis=1)
    return panel_length

def cal_arc_length(geom):
    panel_length=cal_panel_length(geom)
    arc_length=np.cumsum(panel_length)
    arc_length=np.concatenate((np.array([0]),arc_length[:-1]),axis=0)
    return arc_length

def tanh_spacing(spacing_start, spacing_end, N):
    alpha_start=-(1-spacing_start)
    alpha_end=1-spacing_end
    start=np.arctanh(alpha_start)
    end=np.arctanh(alpha_end)
    temp=np.ones((N+1,1))
    temp=np.array(range(N+1))/N*(end-start)+start
    temp=np.tanh(temp)
    temp=(temp-temp[0])/(temp[-1]-temp[0])
    return temp

def cat_spacing(spacing_start_list, spacing_end_list, break_point_list, N_list):
    dist_list=[]
    for i in range(len(N_list)):
        dist_list.append(tanh_spacing(spacing_start_list[i],spacing_end_list[i],N_list[i]))
    
    dist_cat=dist_list[0]*break_point_list[0]
    for i in range(1,len(dist_list)-1):
        dist_cat=np.concatenate((dist_cat, break_point_list[i-1]+dist_list[i][1:]*(break_point_list[i]-break_point_list[i-1])))
    dist_cat=np.concatenate((dist_cat,dist_list[-1]*(1-break_point_list[-1])+break_point_list[-1]))
    return dist_cat

def cat_spacing2(spacing_list, break_point_list, N_list):
    dist_list=[]
    for i in range(len(N_list)):
        dist_list.append(tanh_spacing(spacing_list[i],spacing_list[i+1],N_list[i]))
    
    dist_cat=dist_list[0]*break_point_list[0]
    for i in range(1,len(dist_list)-1):
        dist_cat=np.concatenate((dist_cat, break_point_list[i-1]+dist_list[i][1:]*(break_point_list[i]-break_point_list[i-1])))
    dist_cat=np.concatenate((dist_cat,dist_list[-1][1:]*(1-break_point_list[-1])+break_point_list[-1]))
    return dist_cat

def spline_airfoil(geom, N_data=1000):
    arc=cal_arc_length(geom)
    spl_x=splrep(arc,geom[:,0])
    spl_y=splrep(arc,geom[:,1])
    arc_sample=np.linspace(arc[0], arc[-1],N_data)
    x=splev(arc_sample,spl_x)
    y=splev(arc_sample,spl_y)
    return np.transpose(np.array([x,y]))

def spline_airfoil_spacing(geom, spacing):
    arc=cal_arc_length(geom)
    spl_x=splrep(arc,geom[:,0])
    spl_y=splrep(arc,geom[:,1])
    arc_sample=spacing*arc[-1]
    x=splev(arc_sample,spl_x)
    y=splev(arc_sample,spl_y)
    return np.transpose(np.array([x,y]))


def redistribute_airfoil(upper, lower, num_points=51):
    xs=np.linspace(0,0.5,num_points)*np.pi
    xs=1-np.cos(xs)
    
    spl_u=splrep(upper[:,0],upper[:,1])
    spl_l=splrep(lower[:,0],lower[:,1])
    
    upper_redist=splev(xs,spl_u)
    lower_redist=splev(xs,spl_l)
    
    upper_redist=np.array([xs, upper_redist]).T
    lower_redist=np.array([xs, lower_redist]).T
    airfoil_redist=np.concatenate((np.flip(lower_redist,axis=0),upper_redist[1:]),axis=0)
    
    return airfoil_redist


class MyProblem(Problem):
    def __init__(self, method, target_lower, target_upper, no_redist):
        super().__init__(n_var=method.num_var, n_obj=1, n_ieq_constr=0, xl=0*np.ones(method.num_var), xu=1*np.ones(method.num_var))
        self.method=method
        self.no_redist=no_redist
        self.target_upper=target_upper.copy()
        self.target_lower=target_lower.copy()
        self.target_redistrtibuted=np.expand_dims(redistribute_airfoil(self.target_lower, self.target_upper), axis=0)
    
      
    def _evaluate(self, x, out, *args, **kwargs):
    
        airfoil_gen, lower, upper= self.method.generate_airfoil(x, inverse_fit=True)
       
        if self.no_redist:
            temp1=airfoil_gen.copy()
        else:
            temp1=[]
            for i in range(airfoil_gen.shape[0]):
                temp1.append(redistribute_airfoil(lower[i], upper[i]))
            temp1=np.array(temp1)
       
        obj=np.mean(np.mean((temp1-self.target_redistrtibuted)**2,axis=-1),axis=-1)
        
        f1 = obj
        out["F"] = [f1]
    
    def plot_opt_results(self, res, save_header=None):
        X = res.X
        F = res.F
        n_evals = np.array([e.evaluator.n_eval for e in res.history])
        opt = np.array([e.opt[0].F for e in res.history])
    
        X=np.expand_dims(X, axis=0)
        airfoil_gen, lower, upper = self.method.generate_airfoil(X, inverse_fit=True)
        airfoil_gen=redistribute_airfoil(lower[0], upper[0])
        airfoil_target=redistribute_airfoil(self.target_lower, self.target_upper)
        plt.figure(figsize=(18,7))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.subplot(2,3,1)
        plt.title("Convergence (Obj: {:.2e})".format(F[0]))
        plt.plot(n_evals, opt, "--")
        plt.yscale("log") 
        plt.xlabel('Number of evaluation')
        plt.ylabel('Objective function')
        
        plt.subplot(2,3,2)
        
        # plt.plot(self.target_lower[:,0],self.target_lower[:,1],'k-',label='Original Airfoil')
        # plt.plot(self.target_upper[:,0],self.target_upper[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_target[:,0], airfoil_target[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_gen[:,0],airfoil_gen[:,1],'r-.', label='Optimized Airfoil')
        plt.axis('equal')
        plt.title('X, Y same scale')
        plt.legend(frameon=False)

        plt.subplot(2,3,3)
        
        # plt.plot(self.target_lower[:,0],self.target_lower[:,1],'k-',label='Original Airfoil')
        # plt.plot(self.target_upper[:,0],self.target_upper[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_target[:,0], airfoil_target[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_gen[:,0],airfoil_gen[:,1],'r-.', label='Optimized Airfoil')
        plt.title('Y scaled')
        
        plt.subplot(2,3,4)
        plt.plot(airfoil_target[:,0], airfoil_target[:,1]-airfoil_gen[:,1],'k-',label='Original Airfoil')
        plt.grid(True, 'both')
        plt.ylim([-2e-3, 2e-3])
   
        plt.subplot(2,3,5)
        
        # plt.plot(self.target_lower[:,0],self.target_lower[:,1],'k-',label='Original Airfoil')
        # plt.plot(self.target_upper[:,0],self.target_upper[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_target[:,0], airfoil_target[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_gen[:,0],airfoil_gen[:,1],'r-.', label='Optimized Airfoil')
        plt.axis([0.03-0.05, 0.03+0.05, -0.05, 0.05])
        plt.title('LE close-up')
        
        plt.subplot(2,3,6)
        
        # plt.plot(self.target_lower[:,0],self.target_lower[:,1],'k-',label='Original Airfoil')
        # plt.plot(self.target_upper[:,0],self.target_upper[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_target[:,0], airfoil_target[:,1],'k-',label='Original Airfoil')
        plt.plot(airfoil_gen[:,0],airfoil_gen[:,1],'r-.', label='Optimized Airfoil')
        plt.axis([1-0.05, 1+0.05, -0.05, 0.05])
        plt.title('TE close-up')
        if not save_header==None:
            plt.savefig(save_header+'.png')
            plt.close()
        else:
            plt.show()
        
        
from pymoo.termination.default import DefaultSingleObjectiveTermination

        

def find_latent(method, target_lower, target_upper, plot_results=False, save_header=None, no_redist=False):
    
    problem=MyProblem(method, target_lower, target_upper, no_redist=no_redist)
    algorithm = GA(
        pop_size=200,
        n_offsprings=200,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20), 
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 200)
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=1e-9,
        period=200,
        n_max_gen=200000,
        n_max_evals=1000000
        )
    results = minimize(problem,
                algorithm,
                termination,
                seed=1,
                save_history=True,
                verbose=True)
    latent_norm=np.expand_dims(results.X,axis=0)
    fitted_airfoil, latent_code_norm_compact, latent_code_denorm_compact=method.generate_airfoil(latent_norm, norm=True)
    
    if plot_results or not save_header==None:
        problem.plot_opt_results(results, save_header)
        
    return latent_code_norm_compact[0], fitted_airfoil[0], results


def find_aero_center(lower, upper):
    def interp_025(point1, point2):
        return (point2[1]-point1[1])/(point2[0]-point1[0])*(0.25-point1[0])+point1[1]
    def find_closest_idx_to_025(x_coord):
        for i in range(x_coord.shape[0]):
            if x_coord[i]>0.25:
                return i-1
            
    idx=find_closest_idx_to_025(upper[:,0])

    yc_minus=interp_025(lower[idx], lower[idx+1])
    yc_plus=interp_025(upper[idx], upper[idx+1])
    ac=np.array([0.25, (yc_minus+yc_plus)/2])
    return ac

