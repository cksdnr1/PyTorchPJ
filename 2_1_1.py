# 변수의 shape, type, value 확인
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

%%capture
!pip install pytorch_lightning torchinfo torchmetrics torchviz

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

torch.__version__,pl.__version__,

loss_function = nn.CrossEntropyLoss()
class FirstModel(pl.LightningModule):
    def __init__(self):
        super(FirstModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 10) )
    def forward(self, x):
        out = self.layers(x)
        return out
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y)
        self.log_dict({'loss':loss, 'acc':acc})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y)
        self.log_dict({'val_loss':loss, 'val_acc':acc})
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def dataLoader(batch_size=128):
    train_dataset = MNIST('', transform=transforms.ToTensor(), train=True, download=True)
    test_dataset = MNIST('', transform=transforms.ToTensor(), train=False, download=True)
    trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (trainDataLoader,valDataLoader)

trainDataLoader,valDataLoader = dataLoader()

model = FirstModel()
summary(model, input_size=(128, 1, 28, 28))

%%time
epochs = 1
logger = pl.loggers.CSVLogger("logs", name="firstModel")
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator="auto")
trainer.fit(model, trainDataLoader, valDataLoader)

test_batch = next(iter(valDataLoader))  #(128,1,28,28),(128,10) <- 
preds = model(test_batch[0])            #(128,10) <- ((128,1,28,28),(128,))
p(preds[0],'preds'),print()                     # Class logits (10,)
p(torch.softmax(preds[0],dim=0),'softmax'),print()# Class Prob.  (10,) 
p(np.argmax(torch.softmax(preds[0],dim=0).detach()),'index') # Class idx 

v_num = logger.version  
history = pd.read_csv(f'./logs/firstModel/version_{v_num}/metrics.csv')
history.groupby('epoch').last().drop('step', axis=1)

history

import torch.nn as nn
import torch.nn.functional as F

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

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()
    
    def forward(self, x):
        out = self.layers(x)
        return out
model = MyModel()
summary(model, input_size=(8, 5))

p(model,'model')
p(model.layers,'model.layers')
p(model.layers.linear1,'model.layers.linear1'),print()
p(model.layers.children,'model.layers.children')

loss_function = nn.CrossEntropyLoss()
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layers =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.Linear(64,10) 
            )  
    def forward(self, x):
        out = self.layers(x)
        return out

model = Model()
summary(model, input_size=(8, 1, 28, 28))

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('linear1', nn.Linear(28*28, 64))
        self.layers.add_module('linear2', nn.Linear(64,10))
    
    def forward(self, x):
        out = self.layers(x)
        return out

model = Model()
summary(model, input_size=(8, 1, 28*28))

from collections import OrderedDict
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(28*28,64)),
            ('linear2', nn.Linear(64,10)),
          ]))
    def forward(self, x):
        out = self.layers(x)
        return out

model = Model()
summary(model, input_size=(8, 1, 28, 28))      

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(5)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)  
        return x
    #0: x = l[0](x)+l[0](x)
    #1: x = l[0](x)+l[1](x)
    #2: x = l[1](x)+l[2](x)
    #3: x = l[1](x)+l[3](x)
    #4: x = l[2](x)+l[4](x)

model = Model()
summary(model, input_size=(8, 1, 10))

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(2)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.layers[choice](x)
        x = self.activations[act](x)
        return x

model = Model()

p(model)

class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten =  nn.Flatten()
        self.linear1 = nn.Linear(28*28, 32)
        self.linear2 = nn.Linear(28*28, 32)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x2 = self.linear2(x)
        x2 = self.relu(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.linear3(x)
        return x

model = Model()
summary(model, input_size=(8, 1, 28, 28))

!pip install onnx 

device ='cuda:0'
torch.onnx.export(model, torch.zeros((8, 1, 28, 28)).to(device), 'model.onnx')

from IPython.core.display import ProgressBar
## History logging (step level)
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = loss_function(y_hat, y)
    self.log("loss", loss, on_step=True, on_epoch=False) 
    return loss

## History logging (epoch level)
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = loss_function(y_hat, y)
    self.log("loss", loss, on_step=False, on_epoch=True) 
    return loss

## Display with PRogressBar 
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = loss_function(y_hat, y)  
    acc = FM.accuracy(y_hat, y)
    mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
    metrics={'loss': loss, 'acc':acc, 'mse': mse}
    self.log_dict(metrics, prog_bar=True) #on_step=True, on_epoch=False
    return loss  

loss_function = nn.CrossEntropyLoss()
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True, on_step=False, on_epoch=True)
#        self.log_dict(metrics,prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
summary(model, input_size=(8, 1, 28, 28))  

from pytorch_lightning.accelerators import accelerator
epochs=3
logger = pl.loggers.CSVLogger("logs", name="train_history_log")
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto')
trainer.fit(model, trainDataLoader, valDataLoader)

v_num = logger.version  

history = pd.read_csv(f'./logs/train_history_log/version_{v_num}/metrics.csv')
history 

history.groupby('epoch').last().drop('step', axis=1) 

loss_function = nn.CrossEntropyLoss()
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True)#on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True
#        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
summary(model, input_size=(8, 1, 28, 28))

epochs=3
logger = pl.loggers.CSVLogger("logs", name="val_history_log")
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto')
trainer.fit(model, trainDataLoader, valDataLoader) 

v_num = logger.version  
history = pd.read_csv(f'./logs/val_history_log/version_{v_num}/metrics.csv')
history  

trainer.validate(model,valDataLoader)

history.groupby('epoch').last().drop('step', axis=1)

def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = loss_function(y_hat, y)
    acc = FM.accuracy(y_hat, y)
    metrics={'test_loss': loss, 'test_acc':acc}
    self.log_dict(metrics) #on_step=False, on_epoch=True 

loss_function = nn.CrossEntropyLoss()
class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()
    
    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True)#on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        metrics={'test_loss': loss, 'test_acc':acc}
        self.log_dict(metrics)  
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
#summary(model, input_size=(8, 1, 28, 28))

epochs=3
trainer = pl.Trainer(max_epochs=epochs, accelerator='auto')
trainer.fit(model, trainDataLoader)

trainer.validate(model, valDataLoader) 
trainer.test(model, valDataLoader) 

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
        acc = FM.accuracy(y_pred, y)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True)#on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
#summary(model, input_size=(8, 1, 28, 28))

epochs=3
trainer = pl.Trainer(max_epochs=epochs, accelerator='auto')
trainer.fit(model, trainDataLoader)

batch = next(iter(valDataLoader)) # ((128,1,28,28),(128,1))
y_predict = model(batch[0])       # model <- image only 
p(y_predict,'\n')                 # (128,10) 

batch = next(iter(valDataLoader)) # ((128,1,28,28),(128,10))
# forward()실행
y_predict = model(batch[0])       # model <- image only 
p(y_predict[0],'logit')           # 첫image에 대한 출력(logit) 
# predict_step()실행 
y_predict = trainer.predict(model, batch[0]) # model <- image only 
p(y_predict[0],'prob.')           #  첫image에 대한 출력(prob.)

batch = next(iter(valDataLoader)) # ((128,1,28,28),(128,10))
## Sets the module in evaluation mode.(Dropout,BN,..)
model.eval() 
y_predict = model(batch[0])       # model <- image only 
p(y_predict,'\n')                 # (128,10) 
p(y_predict[0,:],'\n')            # (10,) 
p(y_predict[0,:].detach(),'\n')   # (10,) w/o grad_fn
p(np.argmax(y_predict[0,:].detach()).numpy(),'np') # index      
p(np.argmax(y_predict[0,:].detach()),'tensor') # index tensor     
#p(np.argmax(y_predict[0,:])) ## Error: Can't call numpy() on Tensor that requires grad.

batch = next(iter(valDataLoader)) # ((128,1,28,28),(128,10))
## Sets the module in evaluation mode.(Dropout,BN,..)
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

## rank, shape, axis 
d = np.zeros((2,3,4)) 
print(d.ndim)  
p(d)
# #of_rank = 3
# shape=(2,3,4), 
# axis:(0,1,2) or (0,1,-1) 
#     (z, y, x)


