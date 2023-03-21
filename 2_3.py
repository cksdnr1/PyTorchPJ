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

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
])

epochs=3
batch_size=1024

download_root = './MNIST'
train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

download_root = './F-MNIST'
fashion_train = FashionMNIST(download_root, transform=mnist_transform, train=True, download=True)
fashion_test = FashionMNIST(download_root, transform=mnist_transform, train=True, download=True)
fTrainDataLoader = DataLoader(fashion_train, batch_size=batch_size, shuffle=True)
fValDataLoader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False)

download_root = './CIFAR10'
cifar10_train = CIFAR10(download_root, transform=mnist_transform, train=True, download=True)
cifar10_test = CIFAR10(download_root, transform=mnist_transform, train=True, download=True)
c10TrainDataLoader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
c10ValDataLoader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

download_root = './CIFAR100'
cifar100_train = CIFAR100(download_root, transform=mnist_transform, train=True, download=True)
cifar100_test = CIFAR100(download_root, transform=mnist_transform, train=True, download=True)
c100TrainDataLoader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
c100ValDataLoader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

## target lable의 image 확인하기 
def img_plot(data, lable, target):
    plt.figure(figsize=(12, 3))
    idx = 0
    data_ = data.permute(0, 2, 3, 1)
    for i in range(36):
        while lable[idx] != target :
            idx += 1
            if idx >=1024 : idx=0
        plt.subplot(3, 12, i+1)
        if data_.shape[-1] == 1:
            data_ = data_[..., 0]
        plt.imshow(data_[idx])
        plt.axis("off")
        idx += 1
    plt.show()

x_train, y_train = next(iter(trainDataLoader))
img_plot(x_train, y_train, 9 )
pst(x_train)

x_train, y_train = next(iter(fTrainDataLoader))
img_plot(x_train, y_train, 6)
pst(x_train)

x_train, y_train = next(iter(c10TrainDataLoader))
img_plot(x_train, y_train, 6)
pst(x_train)

x_train, y_train = next(iter(c100TrainDataLoader))
img_plot(x_train, y_train, 25)
pst(x_train)

### input dataset을 CIFA-10으로 바꿔봅니다. 

import matplotlib.pylab as plt

# 데이터 준비 
x_train, y_train = next(iter(c10TrainDataLoader))
x_test, y_test = next(iter(c10ValDataLoader))
pst(x_train)

loss_function = nn.CrossEntropyLoss()
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 128), 
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.01),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.01),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.01),
            nn.Linear(256,10),
        )

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y,task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y,task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = Model()
summary(model, input_size=(8, 3, 32, 32))

#%%time
model = Model()

name = 'Model'
logger = pl.loggers.CSVLogger("logs", name=name) 
trainer = pl.Trainer(max_epochs=15, logger=logger, accelerator='auto') 
trainer.fit(model, c10TrainDataLoader, c10ValDataLoader) 

v_num = logger.version ## model.get_progress_bar_dict()['v_num'] 
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv') 
df = history.groupby('epoch').mean().drop('step', axis=1) 

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

















loss_function = nn.CrossEntropyLoss()
class ModelTest(pl.LightningModule):
    def __init__(self):
        super(ModelTest, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 10),
            nn.BatchNorm1d(10))

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
model = ModelTest()
summary(model, input_size=(8, 3, 32, 32))

loss_function = nn.CrossEntropyLoss()
class modelMLP(pl.LightningModule):
    def __init__(self):
        super(modelMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4),
            nn.Linear(128, 10))

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
        
model = modelMLP()
summary(model, input_size=(8, 3, 32, 32))

loss_function = nn.CrossEntropyLoss()
class modelMLP2(pl.LightningModule):
    def __init__(self):
        super(modelMLP2, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10))

    def forward(self, x):
        out = self.layers(x)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics={'loss':loss, 'acc':acc}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
        self.log_dict(metrics, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = modelMLP2()
summary(model, input_size=(8, 3, 32, 32))
