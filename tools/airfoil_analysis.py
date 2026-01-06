import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as ip
from scipy.interpolate import splrep, splev


def cal_panel_length(geom):
    geom_shift=np.roll(geom, shift=-1, axis=0)
    panel_length=np.linalg.norm(geom-geom_shift, axis=1)
    return panel_length

def cal_arc_length(geom):
    panel_length=cal_panel_length(geom)
    arc_length=np.cumsum(panel_length)
    arc_length=np.concatenate((np.array([0]),arc_length[:-1]),axis=0)
    return arc_length

def spline_airfoil(geom, N_data=1000, s=None):
    arc=cal_arc_length(geom)
    spl_x=splrep(arc,geom[:,0],s=s)
    spl_y=splrep(arc,geom[:,1],s=s)
    arc_sample=np.linspace(arc[0], arc[-1],N_data)
    x=splev(arc_sample,spl_x)
    y=splev(arc_sample,spl_y)
    return np.transpose(np.array([x,y]))

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

def cal_curvature(data):
    #first derivatives 
    dx= np.gradient(data[:,0])
    dy = np.gradient(data[:,1])

    #second derivatives 
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    #calculation of curvature from the typical formula
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5 
    return curvature  

def split_airfoil(data_refine):
    min_idx=np.argmin(data_refine[:,0], axis=0)
    data_refine_up=np.flip(data_refine[:min_idx+1],axis=0)
    data_refine_lo=data_refine[min_idx:]
    return data_refine_up, data_refine_lo

def align_x_coord(data):
    samples=tanh_spacing(0.01,0.01,100)
    
    idx_tot=[]
    tol=[]
    for i in range(samples.shape[0]):
        temp_idx=np.argmin(abs(data[:,0]-samples[i]))
        tol.append(abs(data[temp_idx,0]-samples[i]))
        idx_tot.append(temp_idx)
    tol=np.array(tol)
    if np.min(tol)>1e-8:
        print('Warning: tolerance={:.3e}'.format(np.min(tol)))
    return data[idx_tot]

def find_x_coord_qc(data):
    samples=np.array([0.25])
    
    idx_tot=[]
    tol=[]
    for i in range(samples.shape[0]):
        temp_idx=np.argmin(abs(data[:,0]-samples[i]))
        tol.append(abs(data[temp_idx,0]-samples[i]))
        idx_tot.append(temp_idx)
    tol=np.array(tol)
    # if np.min(tol)>1e-8:
    #     print('Warning: tolerance={:.3e}'.format(np.min(tol)))
    return data[idx_tot]

def calibrate_airfoil(geom):
    le_idx=np.argmin(geom[:,0])
    return geom-geom[le_idx]

def rotate_airfoil(geom):
    ref_point=(geom[0,:]+geom[-1,:])/2
    ang=np.arctan2(ref_point[1],ref_point[0])
    rot_mat=np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])
    return np.dot(geom,rot_mat)
   
def scale_airfoil(geom):
    ref_point=(geom[0,:]+geom[-1,:])/2
    geom/=ref_point[0]
    return geom

def calibration_sequence(geom):
    geom=calibrate_airfoil(geom)
    geom=rotate_airfoil(geom)
    geom=scale_airfoil(geom)
    # geom=close_trailing_edge(geom)
    return geom



class airfoil_analizer():
    def __init__(self, airfoil_org, resolution=10000):
        self.airfoil_org=airfoil_org
        self.resolution=resolution
        self.process()
        
    def process(self):
        # Processing
        self.airfoil_refine=spline_airfoil(self.airfoil_org, N_data=10000) # airfoil refinement
        self.airfoil_calibrated=calibration_sequence(self.airfoil_refine) # calibration: LE=[0,0], TE=[0,1]
        # self.airfoil_calibrated=calibration_sequence(self.airfoil_org) # calibration: LE=[0,0], TE=[0,1]
        airfoil_up, airfoil_lo = split_airfoil(self.airfoil_calibrated) # split airfoil upper/lower
        airfoil_align_up, airfoil_align_lo = align_x_coord(airfoil_up), align_x_coord(airfoil_lo) # align x coordinate
        airfoil_qc_up, airfoil_qc_lo = find_x_coord_qc(airfoil_align_up), find_x_coord_qc(airfoil_align_lo) # align x coordinate
        self.qc=[0.25,(airfoil_qc_up[0,1]+airfoil_qc_lo[0,1])/2]

        # Calculate thickness and camber
        thickness=np.zeros(airfoil_align_up.shape)
        camber=np.zeros(airfoil_align_up.shape)
        thickness[:,0] = airfoil_align_up[:,0]
        camber[:,0] = airfoil_align_up[:,0]
        thickness[:,1] = (airfoil_align_up[:,1]-airfoil_align_lo[:,1])/2
        camber[:,1] = (airfoil_align_up[:,1]+airfoil_align_lo[:,1])/2

        self.thickness=thickness
        self.camber=camber
        
    def is_interwind(self, threshold=1e-4):
        temp=self.thickness[:,1].copy()
        temp=temp+threshold
        return any(temp<0)
    
    def get_results(self):
        return self.airfoil_calibrated, self.thickness, self.camber, self.qc

    def plot_result(self):

        # Plot
        plt.rcParams['font.family']='times new roman'
        plt.rcParams['font.size']=13

        plt.figure(figsize=(15,10))
        plt.subplot(3,1,1)
        plt.plot(self.airfoil_org[:,0],self.airfoil_org[:,1],'k+', label='Airfoil Raw data', markersize=10)
        plt.plot(self.airfoil_refine[:,0],self.airfoil_refine[:,1],'k-', label='Airfoil Splined')
        plt.plot(self.airfoil_calibrated[:,0],self.airfoil_calibrated[:,1],'b-', label='Airfoil Calibrated')
        plt.plot( self.qc[0], self.qc[1],'ko',label='QC')
        plt.legend(frameon=False)
        plt.title('Airfoil data')       
        plt.grid(True)

        plt.subplot(3,1,2)
        plt.plot(self.thickness[:,0], self.thickness[:,1],'k.-')
        plt.title('Thickness')
        plt.grid(True)

        plt.subplot(3,1,3)
        plt.plot(self.camber[:,0], self.camber[:,1],'k.-')
        plt.title('Camber')
        plt.grid(True)

        plt.show()

        print('Maximum thickness: {:.3f}, QC: (x, y) = ({:.5f}, {:.5f})'.format(np.max(self.thickness[:,1])*2, self.qc[0], self.qc[1]))