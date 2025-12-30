import numpy as np
def grid_interp(x,y,z,xi,yi):
    from scipy.interpolate import griddata
    
    values=z.flatten()
    points=np.array([x.flatten(), y.flatten()]).T
    
    zi = griddata(points, values, (xi, yi), method='linear')
    return zi