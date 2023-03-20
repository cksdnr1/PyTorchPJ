def p(var,_str='') :
    if _str=='\n' or _str=='cr' :
        _str = '\n'
    else :
        print(f'<<{_str}>>:')
        _str = ''
    if type(var)!=type([]):
        try: 
            print(f'Shape:{var.shape}')
        except : 
            pass
    print(f'Type: {type(var)}')
    print(f'Values: {var}'+_str)

def pst(_x,_name=""):
    print(f'[{_name}] Shape{_x.shape}, {type(_x)}')
def ps(_x,_name=""):
    print(f'[{_name}] Shape{_x.shape}') 
#%%capture
#!pip install pytorch_lightning torchinfo torchmetrics torchviz

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from torchmetrics import functional as FM
from torchinfo import summary

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils.data as data 
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#torch.__version__,pl.__version__,

import torch.nn as nn
import torch.nn.functional as F

print("#############ModuleContainer 사용")

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 10)
        self.linear2 = nn.Linear(10,1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x

model = Model()
summary(model, input_size=(8, 5))

loss_function = nn.CrossEntropyLoss()
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()
    
    def forward(self, x):
        out = self.layers(x)
        return out

    def predict_step(self, x, batch_idx):
        y_pred = self(x) 
        y_pb = nn.Softmax(y_pred)
        return y_pb
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y,task="multiclass",num_classes=10)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y, task="multiclass",num_classes=10)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True
#        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
#        acc = FM.accuracy(y_hat, y)
        acc = FM.accuracy(y_hat, y, task="multiclass",num_classes=10)
        metrics={'test_loss': loss, 'test_acc':acc}
        self.log_dict(metrics)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def dataLoader(batch_size=128):
    train_dataset = MNIST('', transform=transforms.ToTensor(), train=True, download=True)
    test_dataset = MNIST('', transform=transforms.ToTensor(), train=False, download=True)
    trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (trainDataLoader,valDataLoader)

trainDataLoader,valDataLoader = dataLoader()
model = MyModel()
#summary(model, input_size=(8, 1, 28, 28))  

'''
p(model,'model')
p(model.layers,'model.layers')
p(model.layers.linear1,'model.layers.linear1'),print()
p(model.layers.children,'model.layers.children')
'''

epochs=3
#logger = pl.loggers.CSVLogger("logs", name="myModel")
trainer = pl.Trainer(max_epochs=epochs, accelerator='auto')
trainer.fit(model, trainDataLoader)
batch = next(iter(valDataLoader)) # ((128,1,28,28),(128,10))
# forward()실행
model.eval()


   ## disabled gradient calculation.(reduce memory)
with torch.no_grad():             ### w/o grad_fn object ###
    y_predict = model(batch[0])   # model <- image only 

p(y_predict,'\n')                 # (128,10) tensor, w/o grad_fn
p(y_predict[0,:],'\n')            # (10,)    tensor, w/o grad_fn  
p(np.argmax(y_predict[0,:].numpy()),'np') # numpy index      
p(np.argmax(y_predict[0,:]),'tensor') # tensor index
p(np.argmax(model(batch[0]).detach(),axis=1).numpy(),'ndarray') 
p(batch[1],'Tensor')


d = np.zeros((2,3,4)) 
print(d.ndim)  
p(d)

v_num = logger.version  
history = pd.read_csv(f'./logs/train_history_log/version_{v_num}/metrics.csv')

trainer.validate(model,valDataLoader)
trainer.test(model, valDataLoader)

history.groupby('epoch').last().drop('step', axis=1)



