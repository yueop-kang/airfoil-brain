import torch
from torch import nn, optim
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape=shape
    def forward(self, input):
        return input.view(input.size(0),*self.shape)

class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, padding = 1, activation_f='relu', batch_norm=None):
        super().__init__()
        
        modules=[]
        modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=False))

        if batch_norm:
            # modules.append(nn.BatchNorm2d(out_channels, track_running_stats=False))
            modules.append(nn.BatchNorm1d(out_channels))

        if activation_f == None:
            pass
        elif activation_f == 'relu':
            modules.append(nn.ReLU())
        elif activation_f == 'tanh':
            modules.append(nn.Tanh())
        elif activation_f == 'leakyrelu':
            modules.append(nn.LeakyReLU())
        elif activation_f == 'gelu':
            modules.append(nn.GELU())

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, padding = 1, activation_f='relu', batch_norm=None):
        super().__init__()
        
        modules=[]
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, bias=False))

        if batch_norm:
            # modules.append(nn.BatchNorm2d(out_channels, track_running_stats=False))
            modules.append(nn.BatchNorm2d(out_channels))

        if activation_f == None:
            pass
        elif activation_f == 'relu':
            modules.append(nn.ReLU())
        elif activation_f == 'tanh':
            modules.append(nn.Tanh())
        elif activation_f == 'leakyrelu':
            modules.append(nn.LeakyReLU())
        elif activation_f == 'gelu':
            modules.append(nn.GELU())

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv_block(x)

class Down(nn.Module):
    def __init__(self, conv_block, scale=2, input_dim=2):
        super().__init__()
        if input_dim==1:
            self.maxpool=nn.MaxPool1d(scale)
        elif input_dim==2:
            self.maxpool=nn.MaxPool2d(scale)

        self.down = nn.Sequential(
            self.maxpool,
            conv_block)

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, conv_block, scale=2, input_dim=2):
        super().__init__()
        if input_dim==1:
            self.upsample=nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)
        elif input_dim==2:
            self.upsample=nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.up = nn.Sequential(
            self.upsample,
            conv_block)

    def forward(self, x1):
        return self.up(x1)
