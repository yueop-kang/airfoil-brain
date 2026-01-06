import numpy as np
import os
from tools.airfoil_analysis import airfoil_analizer
import matplotlib.pyplot as plt
import subprocess


def simulate_xfoil_batch(lower, upper, params_dict=None, redist_on=0, verbose=False):
    params_default={
            "Ma": 0.01,
            "visc": 1.8e6,
            "aoa": 0,
            "iter": 50,
            "geom_path": './xfoil_dummy/xfoil_geom',
            "input_path": './',
            "output_header":'./xfoil_dummy/',
        }
    # initalize default setting if parameters are not given
    if params_dict==None:
        params_dict=params_default
    
    for param in params_default:
        if not param in params_dict:
            # print("No input '{:s}' initialized by default setting".format(param))
            params_dict[param]=params_default[param]
    params_dict['batch_size']=lower.shape[0]
    airfoil_tot=[]
    for i in range(params_dict['batch_size']):
        airfoil=np.concatenate((np.flip(lower[i][1:],axis=0), upper[i]), axis=0)
        if redist_on==1:
        # airfoil=redistribute_airfoil(lower, upper, num_points=131)
            airfoil=airfoil_redist(airfoil)
        else:
            airfoil=np.concatenate((np.flip(lower[1:], axis=0), upper))
        airfoil_tot.append(airfoil)


    # delete previous output file
    for i in range(params_dict['batch_size']):
        try:
            os.remove(params_dict["output_header"]+'Save_{:d}.txt'.format(i+1))
            os.remove(params_dict["output_header"]+'Dump_{:d}.txt'.format(i+1))
        except:
            pass
    
    # write airfoil
    for i in range(params_dict['batch_size']):
        write_airfoil(airfoil_tot[i], params_dict["geom_path"]+'_{:d}.txt'.format(i+1))
    
    # write XFOIL input file
    write_input_batch(params_dict)

    subprocess.run("xfoil.exe < XFOIL_inp.txt", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    data_tot=[]
    for i in range(params_dict['batch_size']):
        try:
            data=read_output_batch(params_dict["output_header"]+'Save_{:d}.txt'.format(i+1))
            
        except:
            print('No files to read')
            data=np.array([[-1, -1, 1]])
            
        if any((upper[i,:,1]-lower[i,:,1])<-1e-10):
            
            data=np.array([[-1, -1, 1]])
            if verbose:
                print('Airfoil intertwined')
        
        if data[-1,0] != params_dict['aoa']:
            data=np.array([[-1, -1, 1]])
                
        data_tot.append(data[-1:])
    data_tot=np.array(data_tot)
    return data_tot

def simulate_xfoil(lower, upper, params_dict=None, redist_on=0, verbose=False):
    # default setting
    # print('#############        ############')
    # print(airfoil)
    airfoil=np.concatenate((np.flip(lower[1:],axis=0), upper), axis=0)
    if redist_on==1:
        # airfoil=redistribute_airfoil(lower, upper, num_points=131)
        airfoil=airfoil_redist(airfoil)
    else:
        airfoil=np.concatenate((np.flip(lower[1:], axis=0), upper))
    # plt.plot(lower[:,0], lower[:,1],'k-')
    # plt.plot(lower[:,0], lower[:,1],'k-')
    # plt.plot(airfoil[:,0], airfoil[:,1],'r--')
    # plt.show()
    # print(airfoil)
    # print('##########            #############')
    params_default={
            "Ma": 0.01,
            "visc": 1.8e6,
            "aoa": 0,
            "iter": 50,
            "geom_path": './xfoil_geom.txt',
            "input_path": './',
            "output_header":'./',
        }
    
    # initalize default setting if parameters are not given
    if params_dict==None:
        params_dict=params_default
    
    for param in params_default:
        if not param in params_dict:
            # print("No input '{:s}' initialized by default setting".format(param))
            params_dict[param]=params_default[param]
    
    # delete previous output file

    try:
        os.remove(params_dict["output_header"]+'Save.txt')
        os.remove(params_dict["output_header"]+'Dump.txt')
    except:
        pass
    
    # write airfoil
    write_airfoil(airfoil, params_dict["geom_path"])
    
    # write XFOIL input file
    write_input(params_dict)
    # AZ=airfoil_analizer(airfoil)
    
    # execute XFOIL and read output
    # os.system('xfoilP4.exe < XFOIL_inp.txt')

    CREATE_NO_WINDOW = 0x08000000
    subprocess.run("xfoil.exe < XFOIL_inp.txt", creationflags=CREATE_NO_WINDOW, shell=True)
    # subprocess.run("xfoil.exe < XFOIL_inp.txt", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    try:
        data=read_output(params_dict["output_header"])
    except:
        print('No files to read')
        data=np.array([[-1, -1, 1]])
        
    if any((upper[:,1]-lower[:,1])<-1e-10):
        data=np.array([[-1, -1, 1]])
        if verbose:
            print('Airfoil intertwined')
    
    
    return data



def write_airfoil(airfoil, path):
    f = open(path,'w')
    for i in range(airfoil.shape[0]):
        f.write("{:.12f}, {:.12f}\n".format(airfoil[i,0], airfoil[i,1]))
    f.close()
    
def write_input_batch(params_dict):
    f = open(params_dict["input_path"]+'XFOIL_inp.txt','w')
    for i in range(params_dict['batch_size']):
        f.write('load {:s}\n'.format(params_dict["geom_path"]+'_{:d}.txt'.format(i+1)))
        f.write('geom{:d}\n'.format(i+1))
        f.write('pane\n')
        f.write('oper\n')
        if i==0:
            f.write('iter\n {:d}\n'.format(params_dict["iter"]))
            f.write('visc {:.3f}\n'.format(params_dict["visc"]))
            f.write('Mach {:.3f}\n'.format(params_dict["Ma"]))
        f.write('pacc\n {:s}\n {:s}\n'.format(params_dict["output_header"]+'Save_{:d}.txt'.format(i+1),\
            params_dict["output_header"] +'Dump_{:d}.txt'.format(i+1)))
        f.write('alfa {:.3f}\n'.format(0.0))
        # f.write('aseq 0.0 {:.3f} 0.5\n'.format(params_dict["aoa"]))
        f.write('Pacc\n')
        f.write('pdel 1\n')
        f.write(' \n')
        f.write(' \n')
        f.write(' \n')
        # f.write('quit\n')
    f.write('quit\n')
    f.close()
    
    
def write_input(params_dict):
    f = open(params_dict["input_path"]+'XFOIL_inp.txt','w')
   
    text='{:s}\n{:s}\n{:s}\n{:s}\n{:s}\n{:s}\n{:d}\n{:s} {:.3f}\n{:s} {:.3f}\n{:s}\n{:s}\n{:s}\n{:s}\n{:s} {:f} {:f} {:f}'\
        .format('load', params_dict["geom_path"], 'geom','pane', 'oper', 'iter', params_dict["iter"], \
            'Mach', params_dict["Ma"], 'visc', params_dict["visc"], 'seqp', 'pacc', params_dict["output_header"]+'Save.txt', \
                params_dict["output_header"] +'Dump.txt', 'aseq', 0, params_dict["aoa"], 1)
        
    
    
    # f.write('load {:s}\n'.format(params_dict["geom_path"]))
    # f.write('geom\n')
    # f.write('pane\n')
    # f.write('oper\n')
    # f.write('iter\n {:d}\n'.format(params_dict["iter"]))
    
    # # f.write('Mach {:.3f}\n'.format(params_dict["Ma"]))
    # f.write('Pacc\n {:s}\n {:s}\n'.format(params_dict["output_header"]+'Save.txt',params_dict["output_header"] +'Dump.txt'))
    # f.write('alfa {:.3f}\n'.format(0.0))
    # f.write('visc {:.3f}\n'.format(params_dict["visc"]))
    # f.write('alfa {:.3f}\n'.format(0.0))
    # f.write('{:d}\n'.format(params_dict["iter"]))
    # # f.write('Pacc\n')
    # f.write('quit\n')

    f.write(text)
    f.close()
    
def read_output_batch(filename):
    
    f=open(filename,'r')
    lines=f.readlines()
    p=0
    data=[]
    for line in lines:
        p=p+1
        if p>=13:
            line=line.split()
            try:
                data.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
            except:
                # print(line)
                data.append(np.array([-1, -1, -1]))
    if p==12:
        data.append(np.array([-1, -1, -1]))
    data=np.array(data)
    f.close()
    
    return data
    
def read_output(output_header):
    
    f=open(output_header+'Save.txt','r')
    lines=f.readlines()
    p=0
    data=[]
    for line in lines:
        p=p+1
        if p>=13:
            line=line.split()
            try:
                data.append(np.array([float(line[0]), float(line[1]), float(line[2])]))
            except:
                # print(line)
                data.append(np.array([-1, -1, -1]))
    if p==12:
        data.append(np.array([-1, -1, -1]))
    data=np.array(data)
    f.close()
    
    return data
from scipy.interpolate import splrep, splev
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

def cat_spacing2(spacing_list, break_point_list, N_list):
    dist_list=[]
    for i in range(len(N_list)):
        dist_list.append(tanh_spacing(spacing_list[i],spacing_list[i+1],N_list[i]))
    
    dist_cat=dist_list[0]*break_point_list[0]
    for i in range(1,len(dist_list)-1):
        dist_cat=np.concatenate((dist_cat, break_point_list[i-1]+dist_list[i][1:]*(break_point_list[i]-break_point_list[i-1])))
    dist_cat=np.concatenate((dist_cat,dist_list[-1][1:]*(1-break_point_list[-1])+break_point_list[-1]))
    return dist_cat

def spline_ice_shape(geom, N_data=3000):
    arc=cal_arc_length(geom)
    spl_x=splrep(arc,geom[:,0])
    spl_y=splrep(arc,geom[:,1])
    arc_sample=np.linspace(arc[0], arc[-1],N_data)
    x=splev(arc_sample,spl_x)
    y=splev(arc_sample,spl_y)
    return np.transpose(np.array([x,y]))

def spline_ice_shape_spacing(geom, spacing):
    arc=cal_arc_length(geom)
    spl_x=splrep(arc,geom[:,0])
    spl_y=splrep(arc,geom[:,1])
    arc_sample=spacing*arc[-1]
    x=splev(arc_sample,spl_x)
    y=splev(arc_sample,spl_y)
    return np.transpose(np.array([x,y]))

def cal_panel_length(geom):
    geom_shift=np.roll(geom, shift=-1, axis=0)
    panel_length=np.linalg.norm(geom-geom_shift, axis=1)
    return panel_length

def cal_arc_length(geom):
    panel_length=cal_panel_length(geom)
    arc_length=np.cumsum(panel_length)
    arc_length=np.concatenate((np.array([0]),arc_length[:-1]),axis=0)
    return arc_length


def airfoil_redist(airfoil, N_remesh=101, spacing=0.0005):
    arc=cal_arc_length(airfoil)
    break_point_list=[arc[np.argmin(airfoil[:,0])]/arc[-1]]
    
    spacing_list=[spacing,spacing/2,spacing]
    # soft_factor=1
    # temp_dist=np.array([arc[np.argmin(airfoil[:,0])]/arc[-1], 1-arc[np.argmin(airfoil[:,0])]/arc[-1]])
    # temp_dist_soft=temp_dist**soft_factor/sum(temp_dist**soft_factor)
    # N_list=(N_remesh*temp_dist_soft).astype(int)
    N_list=[int(N_remesh/2),int(N_remesh/2)]
    
    arc_spacing=cat_spacing2(spacing_list, break_point_list, N_list)
    return spline_ice_shape_spacing(airfoil, arc_spacing)
    # return spline_ice_shape(airfoil,N_data=201)
    
def redistribute_airfoil(lower, upper, num_points=101):
    # xs=np.linspace(0,0.5,num_points)*np.pi
    # xs=1-np.cos(xs)
    
    xs=np.linspace(0,1,num_points)
    xs=(1-np.cos(xs*np.pi))/2
    
    spl_u=splrep(upper[:,0], upper[:,1])
    spl_l=splrep(lower[:,0], lower[:,1])
    
    upper_redist=splev(xs,spl_u)
    lower_redist=splev(xs,spl_l)
    
    upper_redist=np.array([xs, upper_redist]).T
    lower_redist=np.array([xs, lower_redist]).T
    airfoil_redist=np.concatenate((np.flip(lower_redist,axis=0),upper_redist[1:]),axis=0)
    
    return airfoil_redist

    

    