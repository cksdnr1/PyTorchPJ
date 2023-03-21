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
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler 

'''
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

# model = Model()
# summary(model, input_size=(8, 1, 28, 28))

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
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        mse = FM.mean_squared_error(torch.argmax(y_pred, dim=1), y)
        metrics={'loss': loss, 'acc':acc, 'mse': mse}
        self.log_dict(metrics,prog_bar=True)#on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y, task="multiclass",num_classes=10)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True
        return 

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = loss_function(y_hat, y)
        acc = FM.accuracy(y_hat, y, task="multiclass",num_classes=10)
        mse = FM.mean_squared_error(torch.argmax(y_hat, dim=1), y)
        metrics={'val_loss': loss, 'val_acc':acc, 'val_mse': mse}
        self.log_dict(metrics) #on_step=False, on_epoch=True
        return 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
#summary(model, input_size=(8, 1, 28, 28))


'''

class MNISTDataModule(pl.LightningDataModule):
  def __init__(self, data_dir: str = '', batch_size: int = 32):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    
  def setup(self, stage):
    # transforms for images 
    transform=transforms.Compose([transforms.ToTensor(), # 1/255,tensor로 변환
                                  transforms.Normalize((0.1307,), (0.3081,))])
    self.mnist_test = MNIST(self.data_dir, train=False, transform=transform, download=True)
    mnist_full = MNIST(self.data_dir, train=True, transform=transform, download=True)
    self.mnist_train, self.mnist_val = data.random_split(mnist_full, [55000, 5000])

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=self.batch_size)

  def val_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.mnist_test, batch_size=self.batch_size)

data_module = MNISTDataModule(batch_size=256)    

'''

model = MyModel()
trainer = pl.Trainer(max_epochs=3, accelerator='auto')
trainer.fit(model, data_module) 

trainer.validate(model, data_module) 

trainer.test(model, data_module)

##################################################################!!!!!!!!!!!!!!!


class Onehot(object):        
    def __call__(self, sample):
        sample = sample
        target = np.eye(10)[sample]
        return torch.FloatTensor(target)
        
# MSE 계산을 위해 target을 one-hot encoding 
target_transform = transforms.Compose([Onehot()])             # target label  
mnist_transform = transforms.Compose([transforms.ToTensor()]) # input image 

train_dataset = MNIST('', transform=mnist_transform, target_transform=target_transform, train=True)
test_dataset = MNIST('', transform=mnist_transform, target_transform=target_transform, train=False)

batch_size=128
trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


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

# loss 함수 만들기
def custom_mean_squared_error(pred, target):
    error = torch.mean(torch.square(pred - target))
    return error

# metric 함수 만들기
def custom_mean_error(y_pred, y_true):
    error = torch.mean(y_true - y_pred)
    return error

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  Model()

    def forward(self, x):
        out = self.layers(x)
        out = torch.softmax(out, dim=-1) ##
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = custom_mean_squared_error(y_pred, y)
        error = custom_mean_error(y_pred, y)
        metrics={'loss':loss, 'error':error}
        self.log_dict(metrics)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = custom_mean_squared_error(y_pred, y)
        error = custom_mean_error(y_pred, y)
        metrics = {'val_loss':loss, 'val_error':error}
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = MyModel()
summary(model, input_size=(8, 1, 28, 28))

epochs=3
name="custom_losses"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto')
trainer.fit(model, trainDataLoader, valDataLoader) 

v_num = logger.version  
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
history 


history.groupby('epoch').last().drop('step', axis=1)



# no learning rate scheduler
def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-3)


### 2개의 모델이 분리 되어 있을떄
# multiple optimizer case (e.g.: GAN)
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    return gen_opt, dis_opt

### 스케쥴러가 분리되어 있을때
# multi-optimizer has its own scheduler(step decay)
def configure_optimizers(self):
    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
    dis_opt = Adam(self.model_dis.parameters(), lr=0.02)
    gen_sch = {
        'scheduler': lr_scheduler.ExponentialLR(gen_opt, 0.99),
        'interval': 'step'  # called after each training step
    }
    dis_sch = lr_scheduler.ExponentialLR(dis_opt, 0.99) # called every epoch
    return [gen_opt, dis_opt], [gen_sch, dis_sch]

def configure_optimizers(self):
    optimizer = Adam(self.model_gen.parameters(), lr=0.05)

## StepLR : step_size단위로 lr <- lr*gamma
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...


## MultiStepLR : milestones list로 step down할 epoch 지정 
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80

## LambdaLR-1 : lr_strat에 곱해질 함수를 지정 
    lambda1 = lambda epoch: 0.95 ** epoch 
    lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
# lr = lr_start * 0.95**epoch 

## LambdaLR-2
    def func(epoch):
        return 0.5 ** (epoch//10)
        lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = func)
# lr = lr_start * 0.5**(epoch//10) 

## LambdaLR-3 : parameter group별 lr-scheduling 적용 
# Assuming optimizer has two groups.
        lambda1 = lambda epoch: 0.5 ** (epoch // 10)
        lambda2 = lambda epoch: 0.95 ** epoch
        lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

## ExponentialLR 
        gamma=0.98
        lr_scheduler=lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=- 1, verbose=False)

        return [optimizer], [lr_scheduler]  


'''

def lr_polt(lr_s,epochs=100,lr_start=0.01):
  lr=[]
  for step in range(epochs):
      lr.append(lr_s(step)*lr_start)
  plt.plot(lr, linestyle='--', label="delayed_Exp_decay")
  plt.legend()
  plt.grid()
  plt.show()

'''

lr_start = 0.01
decay_steps = 20
gamma = 0.5
delay = 20

d_scheduler = lambda epoch: gamma**(
                max(0,epoch-delay) / decay_steps) 

lr_polt(d_scheduler,epochs=100,lr_start=lr_start) 

decay_steps = 20  
gamma=0.5

exponential_decay = lambda epoch: gamma**(epoch / decay_steps) 
exponential_decay_step = lambda epoch: gamma**(epoch // decay_steps) 

lr_start = 0.01
lr_polt(exponential_decay,epochs=100,lr_start=lr_start)
lr_polt(exponential_decay_step,epochs=100,lr_start=lr_start)

## lr_scheduler.ExponentialLR() : decay_steps = 1
decay_steps = 1  
gamma=0.95

exponential_decay = lambda epoch: gamma**(epoch / decay_steps) 

lr_polt(exponential_decay,epochs=100,lr_start=0.01)

lr_start = 0.01
end_lrate = 0.001
decay_steps = epochs = 100
power=0.5

PolynomialDecay = lambda epoch: (((lr_start - end_lrate) 
              * ((1 - epoch / decay_steps) ** (power)) )
            + end_lrate)/lr_start 

lr_polt(PolynomialDecay,epochs=epochs,lr_start=lr_start)

decay_steps = 10
decay_rate = 0.5

InverseTimeDecay = lambda epoch: 1 / (1 + 
              decay_rate * epoch / decay_steps) 

lr_polt(InverseTimeDecay,epochs=100,lr_start=lr_start)

'''
'''
from torch.optim import Adam
lr = 0.005
loss_function = nn.CrossEntropyLoss()

class MyModel(pl.LightningModule):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 64),
            nn.Linear(64,10) 
            )  
    def forward(self, x):
        out = self.layers(x)
        return out
 
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'loss':loss, 'acc':acc}
#        self.log_dict(metrics)        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x) 
        loss = loss_function(y_pred, y)
        acc = FM.accuracy(y_pred, y, task="multiclass",num_classes=10)
        metrics = {'val_loss':loss, 'val_acc':acc}
#        self.log_dict(metrics,prog_bar=True,on_step=True,on_epoch=False)        
        self.log_dict(metrics,prog_bar=True,on_step=False,on_epoch=True)        

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer

model = MyModel()
summary(model, input_size=(8, 1, 28, 28))

data_module = MNISTDataModule(batch_size=256)

model = MyModel()
epoch=30
name="model_defaults"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=epoch, logger=logger, accelerator='auto',overfit_batches=0.1)
trainer.fit(model, data_module)

v_num = logger.version   
meric_file = f'./logs/{name}/version_{v_num}/metrics.csv'
history1 = pd.read_csv(meric_file)
print(meric_file)
history1

max = history1['val_acc'].max()
print(f'Max Acc:{max}')
history1_plot = history1.drop('step', axis=1)#.groupby('epoch').mean()
plt.plot(history1_plot['val_acc'], linestyle='--', label="val_acc")
plt.plot(history1_plot['val_loss'], linestyle='--', label="val_loss")
plt.title("No Schedule")
# plt.ylim(0.1, 1)
plt.legend()
plt.grid()

decay_steps = 5
gamma=0.25
exponential_decay_step = lambda epoch: gamma**(epoch // decay_steps) 

lr_polt(exponential_decay_step,epochs=epoch,lr_start=lr)

import torch.optim.lr_scheduler as lr_scheduler
# decay_steps = 10
# gamma=0.1
# exponential_decay_step = lambda epoch: gamma**(epoch // decay_steps) 

class Model_expDeacy(MyModel):
    def __init__(self):
        super(Model_expDeacy, self).__init__()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        lr_s = lr_scheduler.LambdaLR(optimizer, 
                                     lr_lambda = exponential_decay_step, verbose=True)
        return [optimizer], [lr_s]   

model_exp = Model_expDeacy()
name2="model_exp_decay"
logger = pl.loggers.CSVLogger("logs", name=name2)
trainer = pl.Trainer(max_epochs=epoch, logger=logger, accelerator='auto',overfit_batches=0.1)
trainer.fit(model_exp, data_module)

v_num2 = logger.version  
meric_file = f'./logs/{name2}/version_{v_num2}/metrics.csv'
print(meric_file)

history2 = pd.read_csv(meric_file)
history2

max2 = history2['val_acc'].max()
print(f'Max Acc:{max2}')
history2 = pd.read_csv(meric_file)
history2_plot = history2.drop('step', axis=1)#.groupby('epoch').mean()
plt.plot(history2_plot['val_acc'], linestyle='--', label="val_acc")
plt.plot(history2_plot['val_loss'], linestyle='--', label="val_loss")
plt.title("Exp_decay Schedule")
# plt.ylim(0.1, 1)
plt.legend()
plt.grid()

print(f'Normal Model Max Acc:{max}\nlr_scheduling Model Max Acc:{max2}')
plt.plot(history1_plot['val_acc'], linestyle='--', label="normal_val_acc")
plt.plot(history2_plot['val_acc'], linestyle='-', label="lr_schedule_val_acc")
plt.plot(history1_plot['val_loss'], linestyle='--', label="normal_val_loss")
plt.plot(history2_plot['val_loss'], linestyle='-', label="lr_schedule_val_loss")
plt.title("Exp_decay Schedule")
# plt.ylim(0.1, 1)
plt.legend()
plt.grid()
plt.show() 
'''
#!printenv

#!env CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8

## 사용하기 위해서는 cuda 변경사항 적용이 필요함 
## 환경변수에 `CUBLAS_WORKSPACE_CONFIG=:4096:8`추가 해야 함 
# pl.seed_everything(42, workers=True)
# trainer = pl.Trainer(deterministic=True) 

#%load_ext tensorboard  
### tensorboard 실행
### %tensorboard --logdir ./tb_logs  


loss_function = nn.CrossEntropyLoss()
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten =  nn.Flatten()
        self.linear1 = nn.Linear(28*28, 32)
        self.linear2 = nn.Linear(28*28, 32)
        self.linear3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x0 = self.flatten(x)
        x1 = self.linear1(x0)
        x1 = self.relu(x1)
        x2 = self.linear2(x0)
        x2 = self.relu(x2)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.linear3(x3)
        return x4
    
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

model = Model()
'''

#torch.use_deterministic_algorithms(True)
logger = pl.loggers.TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(max_epochs=20, logger=logger, accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

######얼리 스탑핑
logger = pl.loggers.CSVLogger("logs", name="model_custom_losses")
trainer = pl.Trainer(max_epochs=100, logger=logger, 
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
                     accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

# colab(linux)
# windows
# code/logs code/aicamp 폴더 삭제해 주세요


checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_acc',
    dirpath='./aicamp/',
    filename='{epoch:02d}-{val_acc:.4f}',
    save_top_k=2)
logger = pl.loggers.TensorBoardLogger("tb_logs", name="sample-mnist")
trainer = pl.Trainer(max_epochs=10, logger=logger, 
                     callbacks=[checkpoint_callback],accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

# file name 수정!! 
checkpoint_path = "aicamp/epoch=08-val_acc=0.8835.ckpt"  
# 모델 읽어오기
new_model = model.load_from_checkpoint(checkpoint_path)
trainer.validate(new_model, data_module)

'''
class MyPrintingCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'Epoch : {trainer.current_epoch}')

    def on_train_batch_end(self, *args, **kwargs):
        loss = trainer.callback_metrics['loss']
        self.losses.append(loss)
        print(f'Batch:{trainer.global_step}, loss:{loss}')

model = Model()

logger = pl.loggers.TensorBoardLogger("tb_logs", name="sample-mnist")
trainer = pl.Trainer(max_epochs=3, logger=logger, accelerator='auto', 
                     callbacks=[MyPrintingCallback()])
trainer.fit(model, data_module)

trainer.callbacks[0].losses[:]  
#p(trainer.callbacks[0].losses[:])

losses = [i.cpu() for i in trainer.callbacks[0].losses]
#print("sssssssssss")
#p(losses)

losses = np.array(losses)
p(losses)
plt.plot(losses, linestyle='-', label="batch_loss")
#plt.semilogy()
plt.legend()
plt.grid()
plt.show() 

############profiler를 주면 어디부분에 로그가 많이 걸리는지 보여주는 것
'''

model = Model()

trainer = pl.Trainer(max_epochs=3, accelerator='auto', profiler="simple", overfit_batches=0.3) 
trainer.fit(model, data_module) 

model = Model()

trainer = pl.Trainer(max_epochs=1, accelerator='auto', profiler="advanced",overfit_batches=0.3) 
trainer.fit(model, data_module) 

trainer = pl.Trainer(max_epochs=1, accelerator='auto') 

'''

# torch.save로 모델 전체 저장
torch.save(model, './aicamp/model_save.pt')
# torch.load로 모델 전체 불러오기 
new_model2 = torch.load('./aicamp/model_save.pt')
trainer.validate(new_model2, data_module)

# torch.save(model.state_dict())로 모델 파라메터 저장 모델구조는 내버려두고 글만 저장
torch.save(model.state_dict(), './aicamp/model_dict_save.pt')
new_model2 = Model() 
# model.load_state_dict()로 파라메터 불러오기 
new_model2.load_state_dict(torch.load('./aicamp/model_dict_save.pt'))
trainer.validate(new_model2, data_module) 

for param_tensor in model.state_dict():
    print(param_tensor) 
    print(model.state_dict()[param_tensor].size(),'\n')  

## state_dict 내부 구조 확인 
p(model.state_dict())  



