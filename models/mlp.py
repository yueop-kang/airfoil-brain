import torch
from torch import nn, optim
import torch.nn.functional as F

# test
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self, config_dict=None, params_dict=None):
        super().__init__()

        ### Default params input ###

        params_default={
            "num_hidden_layer": 2,
            "num_hidden_node": 16,
            "activation_f":'relu',
            "batch_norm": False,
        }

        for param in params_default:
            if not param in params_dict:
                print("No input '{:s}' initialized by default setting".format(param))
                params_dict[param]=params_default[param]

        self.input_dim=config_dict["input_dim"]
        self.output_dim=config_dict["output_dim"]

        ### Initialize hyper parameters ###
        self.params_dict=params_dict
        self.num_hidden_layer, self.num_hidden_node, self.activation_f, self.batch_norm=\
            params_dict["num_hidden_layer"], params_dict["num_hidden_node"], params_dict["activation_f"],  params_dict["batch_norm"]

        modules = []
        for i in range(self.num_hidden_layer+1):
            # Linear Layer
            if i==0: # input layer
                modules.append(nn.Linear(self.input_dim, self.num_hidden_node))

            else: # hidden layer
                modules.append(nn.Linear(self.num_hidden_node,self.num_hidden_node))

                # Batch Normalization layer    
                if self.batch_norm:
                    modules.append(nn.BatchNorm1d(self.num_hidden_node))
                
                # Activation Layer
                if self.activation_f == None:
                    pass
                elif self.activation_f == 'relu':
                    modules.append(nn.ReLU())
                elif self.activation_f == 'tanh':
                    modules.append(nn.Tanh())
                elif self.activation_f == 'leakyrelu':
                    modules.append(nn.LeakyReLU())
        
        modules.append(nn.Linear(self.num_hidden_node, self.output_dim)) # output layer
                
        self.fc_layers=nn.Sequential(*modules)
        # self.fc_layers.apply(init_weights)
        
    def forward(self, x):
        x=self.fc_layers(x)
        return x

    def loss_fn(self, x, X):
        MSE = F.mse_loss(x, X, reduction='mean')
        return MSE 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        if batch_idx==0:
            self.train()
        x, y = train_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if batch_idx==0:
            self.eval()
        x, y = val_batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        return loss

    def model_evaluation(self, dataloader, mode='foward'):
        self.eval()
        
        for idx, batch in enumerate(dataloader):
            try:
                x, y = batch
            except:
                x = batch

            if mode =='loss':
                temp = self.validation_step(batch, idx)
   
            if mode =='foward':
                temp = self(x)

            if idx==0:
                out=temp
            else:
                if mode == 'loss':
                    out+=temp
                else:
                    out=torch.cat((out,temp),axis=0)
        
        return out



