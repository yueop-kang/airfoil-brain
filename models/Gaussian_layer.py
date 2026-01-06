import torch
from torch import nn, optim
import torch.nn.functional as F
import pickle
from tools.normalize import DataProcessing
from models.blocks import Flatten, UnFlatten


class Gaussian_layer(nn.Module):
    def __init__(self, input_size=None, output_size=None):
        super().__init__()
        
        self.mu_layer=nn.Linear(in_features=input_size, out_features=output_size)
        self.var_layer=nn.Linear(in_features=input_size, out_features=output_size)


    def forward(self, x):

        mu=self.mu_layer(x)
        var=self.var_layer(x)
        var=F.softplus(var)+1e-6

        return mu, var
