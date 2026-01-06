import torch
from torch import nn, optim
import torch.nn.functional as F
from models.blocks import Flatten, UnFlatten, ConvBlock1d, ConvBlock2d, Up, Down
class CAE(nn.Module):
    def __init__(self, config, params_dict=None):
        super(CAE, self).__init__()

        ### Default params input ###

        params_default={
            "h_channels": [16, 32, 64, 128, 256],
            "h_kernel": 3,
            "h_padding": 1,
            "activation_f":'relu',
            "batch_norm": False,
            "reduced_scale": 2,
            "z_dims":12
        }

        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]
        
        self.input_shape=config["input_shape"]
        self.input_dim=len(self.input_shape)-1


        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.h_channels, self.h_kernel, self.h_padding, self.reduced_scale, self.activation_f, self.batch_norm, self.z_dims =\
            params_dict["h_channels"], params_dict["h_kernel"], params_dict["h_padding"], params_dict["reduced_scale"],\
                 params_dict["activation_f"], params_dict["batch_norm"], params_dict["z_dims"]

        block_depth=len(self.h_channels)

     
        encoder_modules=[]

        if self.input_dim==1:
            self.height_reduced_dims=int(self.input_shape[1]/self.reduced_scale**block_depth)
            encoder_modules.append(nn.Conv1d(self.input_shape[0], self.h_channels[0], kernel_size=1, stride=1, padding=0, bias=False))
            in_channel=self.h_channels[0]
            for i in range(block_depth):
                encoder_modules.append(
                    Down(ConvBlock1d(in_channel, self.h_channels[i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale, input_dim=self.input_dim)
                )
                in_channel = self.h_channels[i]
            h_dim = self.h_channels[-1]*self.height_reduced_dims

        if self.input_dim==2:
            self.height_reduced_dims=int(self.input_shape[1]/self.reduced_scale**block_depth)
            self.width_reduced_dims=int(self.input_shape[2]/self.reduced_scale**block_depth)
            encoder_modules.append(nn.Conv2d(self.input_shape[0], self.h_channels[0], kernel_size=1, stride=1, padding=0, bias=False))
            in_channel=self.h_channels[0]
            for i in range(block_depth):
                encoder_modules.append(
                    Down(ConvBlock2d(in_channel, self.h_channels[i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale, input_dim=self.input_dim)
                )
                in_channel = self.h_channels[i]
            h_dim = self.h_channels[-1]*self.height_reduced_dims*self.width_reduced_dims


        encoder_modules.append(Flatten())
        self.encoder = nn.Sequential(*encoder_modules)


        self.fc1 = nn.Sequential(nn.Linear(h_dim, self.z_dims), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(self.z_dims, h_dim), nn.Tanh())

        # self.fc1 = nn.Sequential(nn.Linear(h_dim, z_dim))
        # self.fc2 = nn.Sequential(nn.Linear(z_dim, h_dim))

        decoder_modules=[]
        if self.input_dim==1:
            decoder_modules.append(UnFlatten((self.h_channels[-1],self.height_reduced_dims)))
            in_channel=self.h_channels[-1]
            for i in range(block_depth):
                decoder_modules.append(
                    Up(ConvBlock1d(in_channel, self.h_channels[-1-i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale, input_dim=self.input_dim)
                )
                in_channel=self.h_channels[-1-i]
            decoder_modules.append(nn.Conv1d(self.h_channels[0], self.input_shape[0], kernel_size=1, stride=1, padding=0, bias=False))

        if self.input_dim==2:
            decoder_modules.append(UnFlatten((self.h_channels[-1],self.height_reduced_dims,self.width_reduced_dims)))
            in_channel=self.h_channels[-1]
            for i in range(block_depth):
                decoder_modules.append(
                    Up(ConvBlock2d(in_channel, self.h_channels[-1-i], kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
                        scale=self.reduced_scale, input_dim=self.input_dim)
                )
                in_channel=self.h_channels[-1-i]
            decoder_modules.append(nn.Conv2d(self.h_channels[0], self.input_shape[0], kernel_size=1, stride=1, padding=0, bias=False))
        
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc1(h)
        return z

    def decode(self, z):
        h = self.fc2(z)
        r = self.decoder(h)
        return r

    def forward(self, x):
        z = self.encode(x)
        r = self.decode(z)
        return r, z

    def loss_fn(self, Y_pred, Y):
        L1 = F.mse_loss(Y_pred, Y, reduction='mean')
        g1=abs(Y_pred-torch.roll(Y_pred, shifts =1, dims=2))
        g2=abs(Y-torch.roll(Y, shifts =1, dims=2))
        L2 = F.mse_loss(g1,g2, reduction='mean')
        return L1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        x= train_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()
        x = val_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def model_evaluation(self, dataloader, mode='encode'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
            if mode =='encode':
                temp = self.encode(x)
            if mode =='decode':
                temp = self.decode(x)
            if mode =='reconstruct':
                temp, temp_ = self(x)

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out
    
    
    
class CAE_custom(nn.Module):
    def __init__(self, config, params_dict=None):
        super(CAE_custom, self).__init__()

        ### Default params input ###

        params_default={
            "h_channels": 16,
            "h_kernel": 3,
            "h_padding": 1,
            "activation_f":'relu',
            "batch_norm": False,
            "reduced_scale": 2,
            "z_dims":12
        }

        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]
        
        self.input_shape=config["input_shape"]
        self.input_dim=len(self.input_shape)-1


        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.h_channels, self.h_kernel, self.h_padding, self.reduced_scale, self.activation_f, self.batch_norm, self.z_dims =\
            params_dict["h_channels"], params_dict["h_kernel"], params_dict["h_padding"], params_dict["reduced_scale"],\
                 params_dict["activation_f"], params_dict["batch_norm"], params_dict["z_dims"]



        self.preprocessing_layer=nn.Conv2d(self.input_shape[0], self.h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.postprocessing_layer=nn.Conv2d(self.h_channels, self.input_shape[0], kernel_size=1, stride=1, padding=0, bias=False)
        in_channel=self.h_channels
        
        self.encoding_layer1=Down(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=3, input_dim=self.input_dim) # input: (9, 21), output: (3,7)
        # Need Padding layer
        self.encoding_layer2=Down(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=2, input_dim=self.input_dim) # input: (4, 8), output: (2,4)
        self.encoding_layer3=Down(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=2, input_dim=self.input_dim) # input: (2, 4), output: (1,2)
        
        
        self.decoding_layer1=Up(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=2, input_dim=self.input_dim) # input: (1, 2), output: (2,4)
        self.decoding_layer2=Up(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=2, input_dim=self.input_dim) # input: (2, 4), output: (4,8)
        # Need Cropping layer
        self.decoding_layer3=Up(ConvBlock2d(in_channel, self.h_channels, kernel = self.h_kernel, padding = self.h_padding, activation_f=self.activation_f, batch_norm=self.batch_norm),
            scale=3, input_dim=self.input_dim) # input: (3, 7), output: (9,21)

        h_dim=in_channel*2
        
        self.fc1 = nn.Sequential(nn.Linear(h_dim, self.z_dims))
        self.fc2 = nn.Sequential(nn.Linear(self.z_dims, h_dim))
        self.flatten=Flatten()
        self.unflatten=UnFlatten((self.h_channels, 1,2))


    def encode(self, x):
        h = self.preprocessing_layer(x)
        h = self.encoding_layer1(h)
        pad_width=(0,1,0,1)
        h = torch.nn.functional.pad(h, pad_width, mode='reflect')
        h = self.encoding_layer2(h)
        h = self.encoding_layer3(h)
        h=self.flatten(h)
        z = self.fc1(h)
        return z

    def decode(self, z):
        h = self.fc2(z)
        h=self.unflatten(h)
        h = self.decoding_layer1(h)
        h = self.decoding_layer2(h)
        h = h[:,:,:3,:7]
        h = self.decoding_layer3(h)
        r = self.postprocessing_layer(h)
        return r

    def forward(self, x):
        z = self.encode(x)
        r = self.decode(z)
        return r, z

    def loss_fn(self, Y_pred, Y):
        L1 = F.mse_loss(Y_pred, Y, reduction='mean')
        g1=abs(Y_pred-torch.roll(Y_pred, shifts =1, dims=2))
        g2=abs(Y-torch.roll(Y, shifts =1, dims=2))
        L2 = F.mse_loss(g1,g2, reduction='mean')
        return L1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        x= train_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()
        x = val_batch
        r, z = self(x)
        loss = self.loss_fn(r, x)
        return loss

    def model_evaluation(self, dataloader, mode='encode'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
            if mode =='encode':
                temp = self.encode(x)
            if mode =='decode':
                temp = self.decode(x)
            if mode =='reconstruct':
                temp, temp_ = self(x)

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out