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


'''


# 생성하기 
x = torch.Tensor([[1, 2], [3, 4]])
p(x,'\n')
x = torch.Tensor([2, 3, 4])
p(x,'\n') 
x = torch.Tensor(2, 3, 4) #(2,3,4)
p(x,'(2,3,4)') 
x = torch.rand(2, 3, 4)
p(x,'rand(2,3,4)')
print(x.shape)
print(x.size()) 

# 변환하기 ndarray <-> tensor
np_arr = np.array([[1, 2], [3, 4]])
p(np_arr,'np_arr') 

tensor = torch.from_numpy(np_arr)
p(tensor,'tensor') 

np_arr = tensor.numpy()
p(np_arr,'np_arr') 

# add 
x1 = torch.Tensor([[1, 2], [3, 4]])
x2 = torch.Tensor([[1, 2], [3, 4]])
y = x1 + x2
print("X1", x1)
print("X2", x2)
print("Y", y)

x2.add_(x1) # ??_() in-place operation.
print("X2", x2)

# reshape, permute 
x = torch.arange(6)
print("X", x)
x = x.view(2, 3)     # reshape 
print("X", x)
x = x.permute(1, 0)  # Swapping dimension 0 and 1
print("X", x)

# matrix multiplications
x = torch.arange(6).view(2, 3)
print("X", x)
W = torch.arange(6).view(3,2)  
print("W", W)
h = torch.matmul(x, W) # or .mm or .bmm(batch mm, for 3D)
print("h", h)

# indexing 
x = torch.arange(12).view(3, 4)
print(x)
print(x[:, 1])  # Second column
print(x[0])     # First row
print(x[:2, -1])# First two rows, last column
print(x[1:3, :])# Middle two rows 

# with gradients # Only float tensors can have gradients
x = torch.arange(3, dtype=torch.float32)
p(x,'x')

x.requires_grad_(True) # _() in-place oper.
p(x,'x_grad')

x = torch.arange(3, dtype=torch.float32, requires_grad=True)  
p(x,'x_grad')

def func(x):
  y = (x-2)**2 + 10
  return y

yl=[]
x = np.linspace(-5,5,100) 
for i, x_ in enumerate(x):
  y_ = func(x_)
  yl.append(y_)
y = np.array(yl)
p(y,'y')

x = torch.linspace(-5,5,100,requires_grad=True) 
p(x,'x')

def func(x):
  y = (x-2)**2 + 10
  return y

yl=[]
x = torch.linspace(-5,5,100,requires_grad=True) 
for i, x_ in enumerate(x):
  y_ = func(x_)
  y_.backward()
  yl.append(y_.detach().numpy())
y = np.array(yl)

print('\ny:',y)
print('\nx.grad:',x.grad)

plt.plot(x.detach().numpy(),y,label='y')
plt.plot(x.detach().numpy(),x.grad,label='gradient')
plt.grid()
plt.legend()
plt.show() 

# Dynamic Computation Graph and Backpropagation
x = torch.arange(3, dtype=torch.float32, requires_grad=True)  
print("X", x)

def func(x):
  a = x + 2
  print("a",a)
  b = a ** 2
  print("b",b)
  c = b + 3
  print("c",c)
  return c.mean()

y = func(x)
print("Y", y)

print('x.grad:',x.grad)
y.backward()
print('x.grad:',x.grad)

p(x,'x')
p(x.grad,'x.grad')
#p(x.grad_fn,'x.grad_fn') #None

p(y,'y')
#p(y.grad,'y.grad') #None
p(y.grad_fn,'y.grad_fn') 

'''
torch.manual_seed(42)  # Setting the seed
layer = nn.Sequential(
    nn.Linear(10, 2), 
    nn.ReLU())
inputs = torch.rand((3, 10))
outputs = layer(inputs)

p(inputs,'inputs')

p(layer,'layer')      # Sequential class
p(layer[0],'layer[0]')# Linear class  

p(layer.state_dict(),'layer')    # all layers weight,bias     
p(layer[0].state_dict(),'layer[0]') #(2 node):weight,bias      
p(layer[1].state_dict(),'layer[1]') # Null Dict.     

p(layer[0].state_dict()['weight']) #(2,10)
p(layer[0].state_dict()['weight'][0,0:3])

p(layer.named_parameters(),'\n') # generator 
m = next(layer.named_parameters()) 
p(m,'named_para.')  #1st layer:(name, parameter)

p(m[0],'name')
p(m[1],'parameter')

p(outputs)

'''
class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden) #(2,4)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs) #(4,1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x
model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
print(model)  

p(model.named_parameters(),'\n') # generator 
m = next(model.named_parameters())
p(m,'m')    # 1st layer : (name, parameter)
p(m[0],'name')
p(m[1],'parameter')  

for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")  

class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label

dataset = XORDataset(size=200)
print("Size of dataset:", dataset.__len__())
print("Data point 0:", dataset[0]) 

def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

visualize_samples(dataset.data, dataset.label)
plt.show()

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
data_inputs, data_labels = next(iter(data_loader))

print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("Data labels", data_labels.shape, "\n", data_labels)

loss_module = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

from tqdm import tqdm  ## tqdm means "progress" in Arabic

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    model.train() # Set model to train mode
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            # Step 1: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # [Batch size] <- [Batch size, 1]
            # Step 2: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            # Step 3: Perform backpropagation
            optimizer.zero_grad()
            # Perform backpropagation
            loss.backward()
            # Step 4: Update the parameters
            optimizer.step()  
    print('\n',loss)  



train_dataset = XORDataset(size=1000)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
train_model(model, optimizer, train_data_loader, loss_module)

def eval_model(model, data_loader):
  model.eval()  # Set model to eval mode
  true_preds, num_preds = 0.0, 0.0
  with torch.no_grad():  # Deactivate gradients for the following code
    for data_inputs, data_labels in data_loader:
      preds = model(data_inputs)
      preds = preds.squeeze(dim=1)
      preds = torch.sigmoid(preds)  # Sigmoid to map predictions between 0 and 1
      pred_labels = (preds >= 0.5).long()  # Binarize predictions to 0 and 1
      # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
      true_preds += (pred_labels == data_labels).sum()
      num_preds += data_labels.shape[0]
  acc = true_preds / num_preds
  print(f"Accuracy of the model: {100.0*acc:4.2f}%")

test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
eval_model(model, test_data_loader)  

from matplotlib.colors import to_rgba

@torch.no_grad() # Decorator: torch.no_grad(), over the whole function.
def visualize_classification(model, data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    # Let's make use of a lot of operations we have learned above
    c0 = torch.Tensor(to_rgba("C0")) #[4]
    c1 = torch.Tensor(to_rgba("C1")) 
    x1 = torch.arange(-0.5, 1.5, step=0.01) #[200]
    x2 = torch.arange(-0.5, 1.5, step=0.01) 
    xx1, xx2 = torch.meshgrid(x1, x2) #[200,200],[200,200]
    model_inputs = torch.stack([xx1, xx2], dim=-1) #[200,200,2]

    preds = model(model_inputs) #[200,200,1]
    preds = torch.sigmoid(preds)

    # Specifying "None" in a dimension creates a new one
    # [200,200,4] <- [200,200,1] * ([1,1,4] <- [4])
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]
    output_image = output_image.numpy()

    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)

visualize_classification(model, dataset.data, dataset.label)
plt.show()  

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader

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

loss_function = nn.CrossEntropyLoss()
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
        
    def forward(self, x):
        x = self.layers(x)
        #nn.CrossEntropyLoss()사용시 softmax와 onehot 필요없음
        return x
        
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
        
model = Model()
summary(model, input_size=(8, 1, 28, 28))

%%time
epochs = 50
name="ReLU_Model"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

v_num = logger.version  
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df = history.groupby('epoch').mean().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df['val_acc'].max())
plt.plot(df['val_acc'], linestyle='--', label="ReLU_val_acc")
plt.plot(df['val_loss'], linestyle='--', label="ReLU_val_loss")

plt.ylim(0.1,1.1)
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5, 5, 0.01)
 
# Sigmoid 유형
def sigmiod_func(x): # = sigmiod, Logistic 
    return 1 / (1 + np.exp(-x))
plt.plot(x, sigmiod_func(x), linestyle='--', label="Sigmoid (= Logistic)")
 
def tanh_func(x): # TanH 
    return np.tanh(x)
plt.plot(x, tanh_func(x), linestyle='--', label="TanH")
  
def softsign_func(x): # Softsign 
    return x / ( 1+ np.abs(x) )
plt.plot(x, softsign_func(x), linestyle='--', label="Softsign")
 
plt.ylim(-1.5, 1.5)
plt.legend()
plt.grid()
plt.show() 

x = np.arange(-6, 3, 0.01)

# ReLU 변형
def relu_func(x): # ReLU(Rectified Linear Unit)
    return (x>0)*x  
plt.plot(x, relu_func(x), linestyle='--', label="ReLU")
 
def leakyrelu_func(x,a=0.01): # Leaky ReLU
    return (x>=0)*x + (x<0)*a*x  
plt.plot(x, leakyrelu_func(x,a=0.1), linestyle='--', label="Leaky ReLU")
 
def trelu_func(x,th=1): # Thresholded ReLU
    return (x>th)*x  
plt.plot(x, trelu_func(x), linestyle='--', label="Thresholded ReLU")
 
def elu_func(x,a=1): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*a*(np.exp(x)-1)  
plt.plot(x, elu_func(x,a=1), linestyle='--', label="ELU")
  
plt.ylim(-2, 3)
plt.legend()
plt.grid()
plt.show() 

def elu_func(x,a=1): # ELU(Exponential linear unit)
    return (x>=0)*x + (x<0)*a*(np.exp(x)-1) # 0.9 조정
plt.plot(x, elu_func(x,a=1), linestyle='--', label="ELU")
 
def selu_func(x): # SELU 
    a = 1.6732632423
    scale = 1.0507009873
    return scale*((x>=0)*x + (x<0)*a*(np.exp(x)-1)) 
plt.plot(x, selu_func(x), linestyle='--', label="SELU")

def celu_func(x,a=1): # CELU 
    return (x>=0)*x + (x<0)*a*(np.exp(x/a)-1)
plt.plot(x, celu_func(x,a=1), linestyle='--', label="CELU")

plt.ylim(-2, 3)
plt.legend()
plt.grid()
plt.show() 

plt.plot(x, elu_func(x,a=0.5), linestyle='--', label="ELU,a=0.5")
plt.plot(x, selu_func(x), linestyle='--', label="SELU")
plt.plot(x, celu_func(x,a=0.5), linestyle='--', label="CELU,a=0.5")

plt.ylim(-2, 3)
plt.legend()
plt.grid()
plt.show() 

x = np.arange(-5, 5, 0.01)

# 기타
def softplus_func(x): 
    return np.log(np.exp(x)+1) 
plt.plot(x, softplus_func(x), linestyle='--', label="softplus_func")

def SiLU_func(x):  
    return x*(sigmiod_func(x))
plt.plot(x, SiLU_func(x), linestyle='--', label="SiLU/Swish")

def Mish_func(x): 
    return x*np.tanh(softplus_func(x))
plt.plot(x, Mish_func(x), linestyle='--', label="Mish")
 
plt.ylim(-2, 4)
plt.legend()
plt.grid()
plt.show()

loss_function = nn.CrossEntropyLoss()
class Model2(pl.LightningModule):
    def __init__(self):
        super(Model2, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
#            nn.SELU(),
            nn.PReLU(num_parameters=128),   
#            nn.Mish(),
#            nn.ReLU(),
            nn.Linear(128, 10))
        
    def forward(self, x):
        x = self.layers(x)
        return x
        
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

model = Model2()
summary(model, input_size=(8, 1, 28, 28))

%%time
epochs = 50
name="PReLU_Model"
#name="Mish_Model"
#name="SELU_Model"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

v_num = logger.version  
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df2 = history.groupby('epoch').mean().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df['val_acc'].max(),'] PReLU.[',df2['val_acc'].max(),']')
#print('MaxAcc:ReLU[',df['val_acc'].max(),'] SELU.[',df4['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="ReLU_val_acc")
plt.plot(df['val_loss'], linestyle='-', label="ReLU_val_loss")
plt.plot(df2['val_acc'], linestyle='--', label="PReLU_val_acc")
plt.plot(df2['val_loss'], linestyle='--', label="PReLU_val_loss")

# plt.plot(df4['val_acc'], linestyle='--', label="SELU_val_acc")
# plt.plot(df4['val_loss'], linestyle='--', label="SELU_val_loss")
# plt.plot(df2['val_acc'], linestyle='--', label="Mish_val_acc")
# plt.plot(df2['val_loss'], linestyle='--', label="Mish_val_loss")

#plt.ylim(0.1,0.6)
plt.legend()
plt.grid()
plt.show()

loss_function = nn.CrossEntropyLoss()
l2_regularization = 0.001
class Model3(Model):
    def __init__(self):
        super(Model3, self).__init__()
        self.layers = Model()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=l2_regularization)

model = Model3()
summary(model, input_size=(8, 1, 28, 28))

%%time
epochs = 50 
name="L2_Reg._Model"
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

v_num = logger.version ## model.get_progress_bar_dict()['v_num']
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df2 = history.groupby('epoch').mean().drop('step', axis=1) 

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df['val_acc'].max(),'] L2Reg.[',df2['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="ReLU_val_acc")
plt.plot(df['val_loss'], linestyle='-', label="ReLU_val_loss")
plt.plot(df2['val_acc'], linestyle='--', label="L2reg._val_acc")
plt.plot(df2['val_loss'], linestyle='--', label="L2reg._val_loss")

#plt.ylim(0.1, 0.6)
plt.legend()
plt.grid()
plt.show() 

model.layers 

model.layers.state_dict()

p(model.layers.state_dict()['layers.3.weight'])

p(model.layers.state_dict()['layers.3.bias'])

weight = model.layers.state_dict()['layers.3.weight']
p(weight,'cr')
p(weight.cpu().detach().numpy())

p(np.array([x.cpu().detach().numpy() for x in model.layers.parameters()])[2])

for p in model.layers.parameters():
  ps(p.cpu().detach().numpy())

from pytorch_lightning.callbacks import Callback
layer_n = 2
class WeightHistory(Callback):
  def on_train_start(self, trainer, pl_module):
    self.k_weight = []
  def on_train_batch_end(self, *args, **kwargs):
    w = model.layers.state_dict()['layers.3.weight'].cpu().detach().numpy()
    self.k_weight.append(w)

model = Model3()

name = 'L2_weight'
logger = pl.loggers.CSVLogger("logs", name=name)
trainer = pl.Trainer(max_epochs=20, logger=logger, callbacks=[WeightHistory()], accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

ps(np.array(trainer.callbacks[0].k_weight))

w = np.array(trainer.callbacks[0].k_weight)
plt.plot(w[:,:,0])   #(batch, i/10, 0/128)

v_num = logger.version  
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv')
df2 = history.groupby('epoch').mean().drop('step', axis=1)

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df2['val_acc'].max()) 

plt.plot(df2['val_acc'], linestyle='-', label="ReLU_val_acc")
plt.plot(df2['val_loss'], linestyle='-', label="ReLU_val_loss")

#plt.ylim(0.1, 0.6)
plt.legend()
plt.grid()
plt.show() 

loss_function = nn.CrossEntropyLoss()
class Model4(pl.LightningModule):
    def __init__(self):
        super(Model4, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10))
        
    def forward(self, x):
        x = self.layers(x)
        return x
            
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

model = Model4()
summary(model, input_size=(8, 1, 28, 28))

model = Model4()

name = 'Dropout'
logger = pl.loggers.CSVLogger("logs", name=name) 
trainer = pl.Trainer(max_epochs=100, logger=logger, accelerator='auto',overfit_batches=0.3)
trainer.fit(model, data_module)

v_num = logger.version ## model.get_progress_bar_dict()['v_num'] 
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv') 
df2 = history.groupby('epoch').mean().drop('step', axis=1) 

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df['val_acc'].max(),'] Dropout.[',df2['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="ReLU_val_acc")
plt.plot(df['val_loss'], linestyle='-', label="ReLU_val_loss")
plt.plot(df2['val_acc'], linestyle='--', label="Dropout_val_acc")
plt.plot(df2['val_loss'], linestyle='--', label="Dropout_val_loss")

#plt.ylim(0.2,0.95)
plt.legend()
plt.grid()
plt.show() 

loss_function = nn.CrossEntropyLoss()
class Model5(pl.LightningModule):
    def __init__(self):
        super(Model5, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10) )    
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
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

model = Model5()
summary(model, input_size=(8, 1, 28, 28))

model = Model5()

name = 'BatchNorm'
logger = pl.loggers.CSVLogger("logs", name=name) 
trainer = pl.Trainer(max_epochs=50, logger=logger, accelerator='auto',overfit_batches=0.1)
trainer.fit(model, data_module)

v_num = logger.version ## model.get_progress_bar_dict()['v_num'] 
history = pd.read_csv(f'./logs/{name}/version_{v_num}/metrics.csv') 
df2 = history.groupby('epoch').mean().drop('step', axis=1) 

import matplotlib.pylab as plt

print('MaxAcc:ReLU[',df['val_acc'].max(),'] BatchNorm.[',df2['val_acc'].max(),']')

plt.plot(df['val_acc'], linestyle='-', label="ReLU_val_acc")
plt.plot(df['val_loss'], linestyle='-', label="ReLU_val_loss")
plt.plot(df2['val_acc'], linestyle='--', label="BatchNorm_val_acc")
plt.plot(df2['val_loss'], linestyle='--', label="BatchNorm_val_loss")

#plt.ylim(0.2,0.95)
plt.legend()
plt.grid()
plt.show() 

model

model.layers.state_dict()  

p = model.layers.state_dict()
p_k = list(p.keys())
#print(p_k)
for k in p_k: 
  print(k,':',model.layers.state_dict()[str(k)].size())  

for p in model.layers.parameters():
  print(p.size()) 

'''
