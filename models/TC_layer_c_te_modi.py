import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from scipy import interpolate

def B(x, k, i, t, device='cuda'):
    if k == 0:
        if t[i]==t[i+1]:
            return torch.zeros_like(x).float().to(device)
            # return torch.where((x==t[i]), torch.tensor(1.0), torch.tensor(0.0))
        else:
            return torch.where(((t[i] <= x) & (x < t[i+1])), torch.tensor(1.0).float().to(device), torch.tensor(0.0).float().to(device))

    if t[i+k] - t[i]==0:
        c1 = B(x, k-1, i, t, device=device)
    else:
        c1 = (x - t[i]) / (t[i+k] - t[i]) * B(x, k-1, i, t, device=device)
   
    if t[i+k+1] - t[i+1]==0:
        c2 = B(x, k-1, i+1, t, device=device)
    else:
        c2 = (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, device=device)

    return c1 + c2

def B_derivative(x, k, i, t, device='cuda'):
    if k == 0:
        return torch.tensor(0.0).float().to(device)

    if t[i+k] - t[i] == 0:
        c1 = B(x, k-1, i, t, device=device)
    else:
        c1 = k / (t[i+k] - t[i]) * B(x, k-1, i, t, device=device)

    if t[i+k+1] - t[i+1] == 0:
        c2 = B(x, k-1, i+1, t, device=device)
    else:
        c2 = k / (t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, device=device)

    return c1 - c2

def B_second_derivative(x, k, i, t, device='cuda'):
    if k == 0:
        return torch.tensor(0.0).float().to(device)

    if t[i+k] - t[i] == 0:
        c1 = B_derivative(x, k-1, i, t, device)
    else:
        c1 = (k * (k - 1)) / ((t[i+k] - t[i]) ** 2) * B(x, k-2, i, t, device=device)

    if t[i+k+1] - t[i+1] == 0:
        c2 = B_derivative(x, k-1, i+1, t, device)
    else:
        c2 = (k * (k - 1)) / ((t[i+k+1] - t[i+1]) ** 2) * B(x, k-2, i+1, t, device=device)

    return c1 - c2

def bspline(x, t, c, k, device='cuda'):
    
    n = len(t) - k - 1
    # assert (n >= k+1) and (c.size(2) >= n)
    # print(n)
    B_list=[]
    B_der_list=[]
    for i in range(n):
        B_list.append(B(x, k, i, t, device=device).unsqueeze(0))
        B_der_list.append(B_derivative(x, k, i, t, device=device).unsqueeze(0))
    B_list=torch.cat(B_list,dim=0)
    B_der_list=torch.cat(B_der_list,dim=0)
    b_values=torch.matmul(c,B_list)
    b_der_values=torch.matmul(c,B_der_list)
    
    if k >= 3:
  
        B_2der_list=[]
        for i in range(n):
            B_2der_list.append(B_second_derivative(x, k, i, t, device=device).unsqueeze(0))
        B_2der_list=torch.cat(B_2der_list,dim=0)
        b_2der_values=torch.matmul(c,B_2der_list)
        

        return b_values, b_der_values, b_2der_values
    else:
        return b_values, b_der_values

def cal_B_list(x,t,k, device='cuda'):
    n = len(t) - k - 1
    B_list=[]
    B_1st_der_list=[]
    for i in range(n):
        B_list.append(B(x, k, i, t, device=device).unsqueeze(0))
        B_1st_der_list.append(B_derivative(x, k, i, t, device=device).unsqueeze(0))
    B_list=torch.cat(B_list,dim=0)
    B_1st_der_list=torch.cat(B_1st_der_list,dim=0)

    if k >= 3:
        B_2nd_der_list=[]
        for i in range(n):
            B_2nd_der_list.append(B_second_derivative(x, k, i, t, device=device).unsqueeze(0))
        B_2nd_der_list=torch.cat(B_2nd_der_list,dim=0)

        return B_list, B_1st_der_list, B_2nd_der_list
    else:
        return B_list, B_1st_der_list
    
def bspline_lookup(c, B, B_1st_der, B_2nd_der):
    b_values=torch.matmul(c,B)
    b_1nd_der_values=torch.matmul(c,B_1st_der)
    b_2nd_der_values=torch.matmul(c,B_2nd_der)
    return b_values, b_1nd_der_values, b_2nd_der_values

def bspline_derivative(x, t, c, k, device, order=1):
    
    n = len(t) - k - 1
    assert (n >= k+1) and (c.size(2) >= n)

    c_der=k/(t[k+1:-1]-t[1:c.size(2)])*(c[:,:,1:]-c[:,:,:-1])
    if order == 1:
        b_der_values=bspline(x,t[1:-1],c_der,k-1, device=device)[0]
        return b_der_values
    else:
        order-=1
        return bspline_derivative(x,t[1:-1],c_der,k-1,order, device=device)


def linear_interp(x, y, xs):
    batch_size = x.size(0)

    # Find the indices of the nearest neighbors for each value in xs
    indices = torch.searchsorted(x, xs).clamp(min=1)  # Ensure that the indices are at least 1 to prevent negative indexing
    
    # Gather the x and y values at indices and indices-1
    batch_indices = torch.arange(batch_size).unsqueeze(1)
    x0 = torch.gather(x, 1, indices-1)
    x1 = torch.gather(x, 1, indices)
    y0 = torch.gather(y, 1, indices-1)
    y1 = torch.gather(y, 1, indices)
    
    # Compute weights
    w1 = (xs - x0) / (x1 - x0).clamp(min=1e-10)  # Avoid division by zero with a small clamp value
    w0 = 1 - w1

    # Perform linear interpolation
    ys = w0 * y0 + w1 * y1
    inter_curve = torch.stack([xs, ys], dim=-1)
    return inter_curve
##############

import time

class TC_gen_layer(nn.Module):
    def __init__(self, num_t_cp=8, num_c_cp=8, degree_spline=3, device='cuda'):
        super().__init__()
        self.num_grid_points=101
        self.device=device
        self.num_t_cp=num_t_cp
        self.num_c_cp=num_c_cp
        # self.num_free_var_t=(2*num_t_cp-3)
        self.num_free_var_t=(2*num_t_cp-4)
        self.num_free_var_c=(2*num_c_cp-3)
        self.num_free_var=self.num_free_var_t+self.num_free_var_c
        self.degree_spline=degree_spline
        self.B_list_c, self.B_1st_der_list_c, self.B_2nd_der_list_c=self.make_lookup(self.num_c_cp)
        self.B_list_t, self.B_1st_der_list_t, self.B_2nd_der_list_t=self.make_lookup(self.num_t_cp)
    
    def preprocess_cp(self, raw_t_cp, raw_c_cp):
        
    
        # Assuming raw_t_cp and raw_c_cp are already on the desired device
        batch = raw_t_cp.size(0)

        # Pre-allocate zero tensors on the device
        zeros_batch_1 = torch.zeros((batch, 1), device=self.device)
        ones_batch_1 = torch.ones((batch, 1), device=self.device)

        # Process t_cp_x
        t_cp_x = raw_t_cp[:, :self.num_t_cp - 2]
        t_cp_x+=0.5
        t_cp_x = torch.cumsum(t_cp_x, dim=1)
        t_cp_x = t_cp_x / t_cp_x[:, -1:].clamp(min=1e-10)  # Avoid division by zero
        
        t_cp_x = torch.cat((zeros_batch_1, zeros_batch_1, t_cp_x), dim=1)

        # Process t_cp_y
        t_cp_y = raw_t_cp[:, self.num_t_cp - 2:]/5
        t_cp_y = torch.cat((zeros_batch_1, t_cp_y, zeros_batch_1), dim=1)

        thickness_cp = torch.cat((t_cp_x.unsqueeze(-1), t_cp_y.unsqueeze(-1)), dim=-1).float()

        # Process c_cp_x
        
        c_cp_x = torch.abs(raw_c_cp[:, :self.num_c_cp - 2])
     
        c_cp_x[:,1:]+=0.5
        c_cp_x = torch.cumsum(c_cp_x, dim=1)
        
        
        # print(torch.min(c_cp_x[:, -1:]))
        c_cp_x = c_cp_x / c_cp_x[:, -1:].clamp(min=1e-10)*0.95  # Avoid division by zero
        c_cp_x = torch.cat((zeros_batch_1, c_cp_x, ones_batch_1), dim=1)

        # Process c_cp_y
        # c_cp_y = -(raw_c_cp[:, self.num_c_cp - 1:] - 0.5) * 0.3
        c_cp_y = raw_c_cp[:, self.num_c_cp - 1:]/7
        c_cp_y = torch.cat((zeros_batch_1, c_cp_y, zeros_batch_1), dim=1)

        camber_cp = torch.cat((c_cp_x.unsqueeze(-1), c_cp_y.unsqueeze(-1)), dim=-1).float()
  
        return camber_cp, thickness_cp
    
    def make_lookup(self, num_cp):
        c_knots=np.linspace(0,1, num_cp-(self.degree_spline-1), endpoint=True)
        c_knots=np.append(np.zeros(self.degree_spline),c_knots)
        c_knots=np.append(c_knots,np.zeros(self.degree_spline)+1)
        c_knots= torch.tensor(c_knots).float().to(self.device)
        
        t_knots=np.linspace(0,1, num_cp-(self.degree_spline-1), endpoint=True)
        t_knots=np.append(np.zeros(self.degree_spline),t_knots)
        t_knots=np.append(t_knots,np.zeros(self.degree_spline)+1)
        t_knots= torch.tensor(t_knots).float().to(self.device)

        xx=np.linspace(0,1,10000, endpoint=True)
        xx=torch.tensor(xx).float().to(self.device)
        
        B_list, B_1st_der_list, B_2nd_der_list = cal_B_list(xx,t_knots,self.degree_spline, device=self.device)
        B_list[-1,-1]=torch.tensor(1).to(self.device)
        B_1st_der_list[-1,-1]=torch.tensor(1).to(self.device)
        B_2nd_der_list[-1,-1]=torch.tensor(1).to(self.device)
        return B_list, B_1st_der_list, B_2nd_der_list
        
  
    def eval_TC_curve(self, camber_cp, thickness_cp):
        camber_cp=camber_cp.swapaxes(1,2)
        thickness_cp=thickness_cp.swapaxes(1,2)
     
        
        camber_curve, camber_der1, camber_der2 = bspline_lookup(camber_cp, self.B_list_c, self.B_1st_der_list_c, self.B_2nd_der_list_c)
        thickness_curve, thickness_der1, thickness_der2 = bspline_lookup(thickness_cp, self.B_list_t, self.B_1st_der_list_t, self.B_2nd_der_list_t)
        
        camber_slope=camber_der1[:,1]/camber_der1[:,0]
        thickness_slope=thickness_der1[:,1]/thickness_der1[:,0]
        camber_curvature=(camber_der1[:,0]*camber_der2[:,1]-camber_der1[:,1]*camber_der2[:,0])/(camber_der1[:,0]**2+camber_der1[:,1]**2)**(3/2)
        thickness_curvature=(thickness_der1[:,0]*thickness_der2[:,1]-thickness_der1[:,1]*thickness_der2[:,0])/(thickness_der1[:,0]**2+thickness_der1[:,1]**2)**(3/2)
        # print(camber_cp)
        # print(camber_curve)
        return camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature
    
    def eval_airfoil(self, camber_curve, thickness_curve):


        camber=self.inter_curve(camber_curve[:,0,:].contiguous(),camber_curve[:,1,:].contiguous())
        thickness=self.inter_curve(thickness_curve[:,0,:].contiguous(),thickness_curve[:,1,:].contiguous())
        return self.tc2airfoil(camber, thickness)
   
        
    
    def inter_curve(self, x, y):
        batch_size=x.size(0)
        
    
        # if not hasattr(self, 'xs') or self.xs is None:
        if 1 == 1:
            # Using PyTorch directly instead of NumPy
            # with torch.no_grad():
            
            # xs = torch.linspace(0, 0.5, 51, device=self.device) * torch.pi
            # xs = (1 - torch.cos(xs))
            
           
            
            xs = torch.linspace(0, 1, self.num_grid_points, device=self.device) * torch.pi
            xs = (1 - torch.cos(xs))/2
            
            
            def tanh_spacing(spacing_start, spacing_end, N):
                alpha_start=-(1-spacing_start)
                alpha_end=1-spacing_end
                start=torch.arctanh(torch.tensor(alpha_start))
                end=torch.arctanh(torch.tensor(alpha_end))
                
                temp=torch.linspace(0,1,N)*(end-start)+start
                temp=torch.tanh(temp)
                temp=(temp-temp[0])/(temp[-1]-temp[0])
                return temp

            xs=tanh_spacing(1.5e-6, 1.5e-6,self.num_grid_points)
            xs=xs.to(self.device)
            
            # print(xs.size())
            self.xs = xs
            
            # xs = torch.linspace(0, 0.5, 51) * np.pi
            # xs = (1 - torch.cos(xs)).to(self.device)
            # print(xs.size())
            # self.xs = xs

        # Tile xs for the batch
        xs_tiled = self.xs.repeat(batch_size, 1)
        temp=linear_interp(x,y,xs_tiled)
        
        # xs=np.linspace(0,1,self.num_grid_points)*np.pi
        # xs=(1-np.cos(xs))/2
        
        # xs=torch.tensor(xs).float().to(self.device)
        # xs=torch.tile(xs,dims=(batch_size,1))
        
        # temp=linear_interp(x,y,xs)
        # temp=torch.cat((xs.unsqueeze(-1), interp(x,y,xs).unsqueeze(-1)), dim=-1)
  
        return temp
    
    def tc2airfoil(self, camber, thickness):
        
        
        # xu=camber[:,0]
        # yu=camber[:,1]+thickness[:,1]

        # xl=camber[:,0]
        # yl=camber[:,1]-thickness[:,1]
        # airfoil_x=torch.concatenate((torch.flip(xl, dims=[0]), xu[1:]), dim=0)
        # airfoil_y=torch.concatenate((torch.flip(yl, dims=[0]), yu[1:]), dim=0)
        # airfoil=torch.cat((airfoil_x.unsqueeze(-1), airfoil_y.unsqueeze(-1)), dim=-1)
        
        xu=camber[:,:,0]
        yu=camber[:,:,1]+thickness[:,:,1]

        xl=camber[:,:,0]
        yl=camber[:,:,1]-thickness[:,:,1]
        airfoil_x=torch.concatenate((torch.flip(xl, dims=[1]), xu[:,1:]), dim=1)
        airfoil_y=torch.concatenate((torch.flip(yl, dims=[1]), yu[:,1:]), dim=1)
        airfoil=torch.cat((airfoil_x.unsqueeze(-1), airfoil_y.unsqueeze(-1)), dim=-1)
        return airfoil

    def forward(self, raw_t_cp, raw_c_cp):
        # d=self.sigmoid_activation(self.weights)
        # stamp1 = time.time()
        camber_cp, thickness_cp=self.preprocess_cp(raw_t_cp, raw_c_cp)
        # stamp2 = time.time()
        camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature =self.eval_TC_curve(camber_cp, thickness_cp)
        # stamp3 = time.time()
        airfoil=self.eval_airfoil(camber_curve, thickness_curve)
        # stamp4 = time.time()
        
        # print('### Elapsed Time ### \n preprocess_cp: {:.4f}s\n eval_TC_curve: {:.4f}s\n eval_airfoil: {:.4f}s'.\
        #     format(stamp2-stamp1, stamp3-stamp2, stamp4-stamp3))
        return airfoil, camber_cp, thickness_cp, camber_curve, thickness_curve, camber_slope, thickness_slope,  camber_curvature, thickness_curvature

    def plot_results(self, raw_t_cp, raw_c_cp):

  
        camber_cp, thickness_cp=self.preprocess_cp(raw_t_cp, raw_c_cp)
        camber_curve, thickness_curve=self.eval_TC_curve(camber_cp, thickness_cp)
        airfoil=self.eval_airfoil(camber_curve, thickness_curve)
        
        plt.figure(figsize=(10,12))
        plt.subplot(3,1,1)
        plt.plot(camber_cp[0,:,0].detach(), camber_cp[0,:,1].detach(),'ko--')
        plt.plot(camber_curve[0,0,:].detach(), camber_curve[0,1,:].detach(),'r-')
        # print(camber_curve.size())
        plt.title('Camber')
        plt.ylim([-0.1,0.1])
        # plt.axis('equal')
        plt.grid(True)
        
        plt.subplot(3,1,2)
        plt.plot(thickness_cp[0,:,0].detach(), thickness_cp[0,:,1].detach(),'ko--')
        plt.plot(thickness_curve[0,0,:].detach(), thickness_curve[0,1,:].detach(),'r-')
        plt.title('Thickness')
        # plt.axis('equal')
        plt.ylim([-0.3,0.3])
        plt.grid(True)
        
        plt.subplot(3,1,3)
        plt.plot(airfoil[0,:,0].detach(), airfoil[0,:,1].detach(),'k-')
        plt.title('Thickness')
        plt.axis('equal')
        plt.grid(True)
        plt.show()