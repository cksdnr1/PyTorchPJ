# 변수의 shape, type, value 확인
def p(var,_str='') :
    if _str=='\n' or _str=='cr' :
        _str = '\n'
    else :
        print(f'[{_str}]:')
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

#torch.__version__,pl.__version__

## GPU check
#torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.current_device(),torch.cuda.device(0),torch.cuda.get_device_name(0)

#device ='cuda:0'

mnist_transform = transforms.Compose([
    transforms.ToTensor(), # 255로 나누어주고 tensor로 변환
])

_batch_size = 128
download_root = ''
train_dataset = MNIST(download_root, train=True, download=True, transform=mnist_transform)
test_dataset = MNIST(download_root, train=False, download=True, transform=mnist_transform)
trainDataLoader = DataLoader(train_dataset, _batch_size, True)
valDataLoader = DataLoader(test_dataset, _batch_size, False)

pst(trainDataLoader.dataset.data)
pst(trainDataLoader.dataset.targets)
plt.imshow(trainDataLoader.dataset.data[0])

batch_n = len(trainDataLoader.dataset.data) // _batch_size
batch_n

_epochs = 10
'''
# MLP model
class MLP(pl.LightningModule):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(  # Container
            nn.Flatten(),
            nn.Linear(784, 384),
            nn.ReLU(),
            nn.Linear(384, 10))
            
    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        if ((batch_idx+1)%batch_n)==0: print(f'Bachs:{batch_idx:04d}, Loss:{loss:0.4f}, Acc:{acc:0.4f}')
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

mlp = MLP()
summary(mlp, input_size=(_batch_size, 1, 28, 28))

#%%time
mlp = MLP()

name="mlp"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=_epochs, logger=logger, accelerator="auto",
                     limit_train_batches=0.4, limit_val_batches=0.2)
trainer.fit(mlp, trainDataLoader, valDataLoader)

v_num = logger.version
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df = history.groupby('epoch').last().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:[',df['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="val_acc")
plt.plot(df['val_loss'], linestyle='-', label="val_loss")

#plt.ylim(0.2,0.95)
plt.legend()
plt.grid()
plt.show()

'''

'''
nn.Conv2d(1, 16, (3, 3), padding=1),
nn.LeakyReLu(0.1),
nn.MaxPool2d(2,2)
nn.Conv2d(16, 32, (3, 3), padding=1),
nn.LeakyReLu(0.1),
nn.MaxPool2d(2,2)
nn.Flatten(),
nn.Linear(32*7*7, 10)
'''


# Convolutional model
loss_f = nn.CrossEntropyLoss()
class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), padding=1),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(512*7*7, 10)
        )
        

        
    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        if ((batch_idx+1)%batch_n)==0:
          print(f'Bachs:{batch_idx:04d}, Loss:{loss:0.4f}, Acc:{acc:0.4f}')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

cnn = CNN()
summary(cnn, input_size=(_batch_size, 1, 28, 28))

#%%time
cnn = CNN()

name="cnn"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=_epochs, logger=logger, accelerator="auto",
                     limit_train_batches=0.4,limit_val_batches=0.2)
trainer.fit(cnn, trainDataLoader, val_dataloaders=valDataLoader)

v_num = logger.version ## cnn.get_progress_bar_dict()['v_num']
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df = history.groupby('epoch').last().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:[',df['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="val_acc")
plt.plot(df['val_loss'], linestyle='-', label="val_loss")

#plt.ylim(0.2,0.95)
plt.legend()
plt.grid()
plt.show()

'''


# LeNet like model
class LeNet_like(pl.LightningModule):
    def __init__(self):
        super(LeNet_like, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
##            nn.Conv2d(8, 8, (3, 3), padding=1),
##
            nn.Flatten(),
##            nn.Linear(64*3*3, 10)
        )
    def forward(self, x):
        out = self.layers(x)
        return out
       
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        if ((batch_idx+1)%batch_n)==0:
          print(f'Bachs:{batch_idx:04d}, Loss:{loss:0.4f}, Acc:{acc:0.4f}')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

LeNet = LeNet_like()
summary(LeNet, input_size=(_batch_size, 1, 28, 28))

LeNet = LeNet_like()

name="lenet"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=_epochs, logger=logger, accelerator="auto",
                     limit_train_batches=0.4,limit_val_batches=0.2)
trainer.fit(LeNet, trainDataLoader, valDataLoader)

v_num = logger.version ## LeNet.get_progress_bar_dict()['v_num']
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df = history.groupby('epoch').last().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:[',df['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="val_acc")
plt.plot(df['val_loss'], linestyle='-', label="val_loss")
# plt.plot(df2['val_acc'], linestyle='--', label="_val_acc")
# plt.plot(df2['val_loss'], linestyle='--', label="_val_loss")

#plt.ylim(0.2,0.95)
plt.legend()
plt.grid()
plt.show()


'''
