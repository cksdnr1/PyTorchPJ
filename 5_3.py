#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/cksdnr1/PyTorchPJ/blob/main/5_3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


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

# In[6]:


#%%capture
#!pip install pytorch_lightning torchinfo torchmetrics pandas torchviz

# ## 3.3 RNN Model Design (Time series)

# ### 실습 3-1 : Stock prediction
# 

# #### **Import Module**

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import functional as FM
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#torch.__version__, pl.__version__

# #### **DataSet**

# ##### Load

# In[8]:


#%%capture
# https://github.com/FinanceData/FinanceDataReader
#!pip install finance-datareader    

# In[9]:


import FinanceDataReader as fdr

# KOSPI
#pd_data = fdr.DataReader('KS11', '2000-01-01') 

# Samsung(005930), 1992-01-01 ~ 2018-10-31
pd_data = fdr.DataReader('005930', '2000-01-01')

pd_data.head(),pd_data.tail()

# #### Preprocessing

# In[10]:


# 최대,최소값으로 정규화 하기
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = (np.max(data, 0) - np.min(data, 0)) + 1e-7
    return numerator/denominator  

# In[11]:


xy = np.array(pd_data) #np <- pandas data format 

## 필요시 reordering 
#xy = xy[::-1]           # 과거부터로 정렬 (chronically ordered)
xy = MinMaxScaler(xy)   # minmax 정규화
x = xy                  # input data(Open,High,Low,Close,Volume,change)
y = xy[:, [3]]         # 종가만 slicing -> target data

# data type 확인
print ("SHAPE OF X IS %s" % (x.shape,))
print ("SHAPE OF Y IS %s" % (y.shape,))  

# In[12]:


# sequence generator
## x:7일치 데이터, y:8일차 종가
timesteps = seq_length = 30
data_dim  = x.shape[1]
batch_size = 32

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length] # 7일치 데이터
    _y = y[i + seq_length]   # 다음날 종가
    dataX.append(_x)
    dataY.append(_y)

# train data : test data = 9 : 1
train_size = int(len(dataY) * 0.9)
test_size  = len(dataY) - train_size
x_train = np.array(dataX[0:train_size])
y_train = np.array(dataY[0:train_size])
x_test  = np.array(dataX[train_size:len(dataX)])
y_test  = np.array(dataY[train_size:len(dataY)])  

# In[13]:


ps(x_train,'x_train')
ps(y_train,'y_train')
ps(x_test,'x_test')
ps(y_test,'y_test')

# In[14]:


class CustomDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x[idx]),torch.FloatTensor(self.y[idx]) 

# In[15]:


trainDataset = CustomDataset(x_train, y_train)
testDataset = CustomDataset(x_test, y_test)
trainDataLoader = DataLoader(trainDataset, shuffle=True, drop_last=False, 
                             batch_size=batch_size)
testDataLoader = DataLoader(testDataset, drop_last=False, batch_size=batch_size)    

# #### **Model**

# ##### Define

# In[ ]:


class BasicModel(pl.LightningModule):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_dim, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())  

# In[ ]:


hidden_dim = 10
model_B = BasicModel(data_dim, hidden_dim) 
summary(model_B, input_size=(batch_size, timesteps, data_dim))  


class AdvancedModel(pl.LightningModule):
    def __init__(self, input_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_features, hidden_dim,
                             batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        p(x)
        x = self.linear(x[:, -1, :])
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.mse_loss(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())  

# In[ ]:


model_S = AdvancedModel(data_dim, hidden_dim)
summary(model_S, input_size=(batch_size, timesteps, data_dim))

# In[ ]:


#%%time
trainer = Trainer(max_epochs=50,accelerator="auto")
trainer.fit(model_B, trainDataLoader, testDataLoader)  
v_num_B = model_B.logger.version  

# Epoch 99: 100%  
# 158/158 [00:01<00:00, 99.00it/s, loss=1.54e-05, v_num=0]
# CPU times: user 1min 32s, sys: 2.1 s, total: 1min 34s  
# Wall time: 1min 37s

# In[ ]:


#%%time
trainer2 = Trainer(max_epochs=50,accelerator="auto")
trainer2.fit(model_S, trainDataLoader, testDataLoader)  
v_num_S = model_S.logger.version   

# Epoch 99: 100%    
# 158/158 [00:01<00:00, 110.79it/s, loss=9.54e-06, v_num=1]
# CPU times: user 2min 20s, sys: 1.97 s, total: 2min 22s    
# Wall time: 2min 29s

# #### **Analysis**

# **Trainer log지정이 없으면 default로 "./lightning_logs"에 저장됨**  
# * TensorBoard를 사용하여 처리 가능

# **tensorboard.backend.event_processing.event_accumulator**   
# http://www.legendu.net/en/blog/read-tensorboard-logs/  
# https://www.programcreek.com/python/example/114903/tensorboard.backend.event_processing.event_accumulator.EventAccumulator  

# In[ ]:


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
log = {} 
# model_B train_logdir 확인
log_dir = f'lightning_logs/version_{v_num_B}'
# load log-directory 
event_accumulator = EventAccumulator(log_dir) 
event_accumulator.Reload()

event_accumulator.Tags()  

# In[ ]:


p(event_accumulator.Scalars("epoch")[0],'class')
p(event_accumulator.Scalars("epoch")[0].value,'value')  

# In[ ]:


pd.DataFrame(event_accumulator.Scalars("epoch"))

# In[ ]:


pd.DataFrame(event_accumulator.Scalars("val_loss"))

# In[ ]:


val_loss = [x.value for x in event_accumulator.Scalars('val_loss')]
train_loss = [x.value for x in event_accumulator.Scalars('loss')]
log['model_B'] = [train_loss, val_loss]   

# In[ ]:


# model_S train_logdir 확인
log_dir = f'lightning_logs/version_{v_num_S}' 

event_accumulator = EventAccumulator(log_dir)
event_accumulator.Reload()

val_loss = [x.value for x in event_accumulator.Scalars('val_loss')]
train_loss = [x.value for x in event_accumulator.Scalars('loss')]
log['model_S'] = [train_loss, val_loss]  

# In[ ]:


# train mse compare
plt.plot(log['model_B'][0], label='model_B')
plt.plot(log['model_S'][0], label='model_S')
plt.title("Training mse")
plt.semilogy()  # y축에 log scale적용
plt.grid(True)
plt.legend(loc='best')
plt.show()  