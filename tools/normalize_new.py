import numpy as np
import torch

class DataProcessing():
    def __init__(self, config=None):
        self.normalize_mode_list=['Identical', 'MinMax', 'Center', 'MeanStd', 'MeanStd_channel']
        config_default={
            'normalize_mode': 'MeanStd',
            'scale': None,
            'bias': None,
            'device': 'cuda'
        }
        
        if not config['normalize_mode'] in self.normalize_mode_list:
            raise Exception("Input normalize_mode \'{:s}\' is not in the list". format(config['normalize_mode']), self.normalize_mode_list)
        
        self.config=config.copy()
        if self.config==None:
            self.config=config_default
        else:
        
            for key in config_default:
                if not key in self.config:
                    #print("No input '{:s}' initialized by default setting".format(key))
                    self.config[key]=config_default[key]
                    
        
    def fit(self, data):
        self.data=data
        if self.config['normalize_mode']=='Identical':
            self.config['scale'], self.config['bias'] = 1., 0.
        
        if self.config['normalize_mode']=='MinMax':
            self.config['scale'], self.config['bias'] = (np.max(data, axis=0)-np.min(data, axis=0)), np.min(data, axis=0)
            
        if self.config['normalize_mode']=='Center':
            self.config['scale'], self.config['bias'] = 1., np.mean(data, axis=0)
        
        if self.config['normalize_mode']=='MeanStd':
            self.config['scale'], self.config['bias'] = np.std(data, axis=0), np.mean(data, axis=0)
            
        if self.config['normalize_mode']=='MeanStd_channel':
            scale=np.ones_like(data[0])
            bias=np.ones_like(data[0])
            for i in range(data.shape[1]):
                scale[i]*=np.std(data[:,i])
                bias[i]*=np.mean(data[:,i])
        
        
            self.config['scale'], self.config['bias'] = scale, bias
        
        if (np.array([self.config['scale']==0.]).any()):
            raise Exception("At least one zero value is detected in config[\'scale\']. Please check raw data or change normalize mode.")
        
    def load(self, config):
        self.config=config
        
    def get_config(self):
        return self.config
        
    def transform(self, data):
        x = (data-self.config['bias'])/self.config['scale']
        x = torch.FloatTensor(x).to(self.config['device'])
        return x

    def inv_transform(self, data, var_scale=False):
        if self.config['device']=='cuda':
            x = data.detach().cpu().numpy()
        elif self.config['device']=='cpu':
            x = data.detach().numpy()
        if var_scale:
            x = x*self.config['scale']**2
            return x
        else:
            x = x*self.config['scale']+self.config['bias']
            return x
