import numpy as np
import torch

class DataProcessing():
    def __init__(self, normalizer, device):
        self.normalizer = normalizer
        self.device = device
    
    def transform(self, data):
        x = self.normalizer.transform(data)
        x = torch.FloatTensor(x).to(self.device)
        return x

    def inv_transform(self, data):
        x = data.detach().cpu().numpy()
        x = self.normalizer.inv_transform(x)
        return x

class Identical():
    def __init__(self, X):
        self.X=X

    def transform(self,X):
        return X
    
    def inv_transform(self, X):
        return X

class MinMax():
    def __init__(self, X, axis=0):
        self.min_val=np.min(X, axis=axis)
        self.max_val=np.max(X, axis=axis)

    def transform(self, X):
        return (X-self.min_val)/(self.max_val-self.min_val)
    def inv_transform(self, X):
        return X*(self.max_val-self.min_val)+self.min_val

class MeanStd():
    def __init__(self, X, axis=0):
        self.mean=np.mean(X, axis=axis)
        self.std=np.std(X, axis=axis)

    def transform(self, X):
        return (X-self.mean)/self.std
    def inv_transform(self, X):
        return X*self.std+self.mean

class MeanStd_channel():
    def __init__(self, X):
        self.mean=[]
        self.std=[]
        for i in range(X.shape[1]):
            self.mean.append(np.mean(X[:,i,:]))
            self.std.append(np.std(X[:,i,:]))
        self.mean=np.array(self.mean)
        self.std=np.array(self.std)
        
    def transform(self, X):
        X_norm=np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:,i,:]=(X[:,i,:]-self.mean[i])/self.std[i]
        return X_norm
        
    def inv_transform(self, X):
        X_norm=np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:,i,:]=X[:,i,:]*self.std[i]+self.mean[i]
        return X_norm

class MeanStd_channel2():
    def __init__(self, X):
        self.mean=[]
        self.std=[]
        for i in range(X.shape[2]):
            self.mean.append(np.mean(X[:,:,i]))
            self.std.append(np.std(X[:,:,i]))
        self.mean=np.array(self.mean)
        self.std=np.array(self.std)
        
    def transform(self, X):
        X_norm=np.zeros(X.shape)
        for i in range(X.shape[2]):
            X_norm[:,:,i]=(X[:,:,i]-self.mean[i])/self.std[i]
        return X_norm
        
    def inv_transform(self, X):
        X_norm=np.zeros(X.shape)
        for i in range(X.shape[2]):
            X_norm[:,:,i]=X[:,:,i]*self.std[i]+self.mean[i]
        return X_norm
class Center():
    def __init__(self, X, axis=0):
        self.mean=np.mean(X, axis=axis)

    def transform(self, X):
        return X-self.mean
    def inv_transform(self, X):
        return X+self.mean
