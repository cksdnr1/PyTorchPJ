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
#!pip install pytorch_lightning torchinfo torchmetrics pandas torchviz

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import functional as FM
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import CSVLogger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from IPython import display

device ='cuda:0'
torch.__version__


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((48, 48)), ]) 
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# num_workers : how many subprocesses to use for data loading.
# 0 means that the data will be loaded in the main process.   

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
batch = next(iter(trainloader))
print(batch[0].shape, batch[1].shape)
print(batch[0][0].min(), batch[0][1].max())

plt.imshow(batch[0][0].permute(1, 2, 0)) 

plt.figure(figsize=(20, 6))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(batch[0][i].permute(1, 2, 0))
    plt.title(f'label: {batch[1][i]}')
plt.show()  

vgg16 = torchvision.models.vgg16(pretrained=True)

p(vgg16,'cr')
p(vgg16.features,'cr')  
vgg16_f = vgg16.features  

summary(vgg16, input_size=(16, 3, 48, 48))

summary(vgg16_f, input_size=(16, 3, 48, 48))  

vgg16_f = vgg16.features
for i, child in enumerate(vgg16_f.children()):
  if i < len(vgg16_f) - 7:
    # Freeze the layers except the last 7 layers
    for param in child.parameters():
      param.requires_grad = False
  else : 
    print(child)  

for child in vgg16_f.children():
    for param in child.parameters():
        print(child, param.requires_grad)
        break  

summary(vgg16_f, input_size=(16, 3, 48, 48))

loss_function = nn.CrossEntropyLoss()

class VGGFineTune(pl.LightningModule):
    def __init__(self, vggnet, num_classes=10):
        super(VGGFineTune, self).__init__()
        self.features = vggnet
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
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
        self.log_dict(metrics,prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

vggFineTune = VGGFineTune(vgg16_f, 10)
summary(vggFineTune, input_size=(batch_size, 3, 48, 48))  

vggFineTune = VGGFineTune(vgg16_f, 10)

epochs=10
logger = CSVLogger("logs", name="VGGNetFineTune")
trainer = Trainer(max_epochs=epochs, logger=logger, accelerator='auto')
trainer.fit(vggFineTune, trainloader, val_dataloaders=testloader)

v_num = logger.version ## vggFineTune.get_progress_bar_dict()['v_num']

history = pd.read_csv(f'./logs/VGGNetFineTune/version_{v_num}/metrics.csv')
history_plot = history.drop('step', axis=1).groupby('epoch').mean()

np.max(history_plot['val_acc'])

def plot_history1(history):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.title('Accuray')
    plt.plot( history['acc'], 'b--', label='acc')
    plt.plot( history['val_acc'], 'r', label='val_acc')
    plt.grid(True)
#    plt.ylim([0.6, 1])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Loss')
    plt.plot( history['loss'], 'b--', label='loss')
    plt.plot( history['val_loss'], 'r', label='val_loss')
    plt.grid(True)
    #plt.ylim([0.0, 0.6])
    plt.legend()
    plt.show()

np.max(history_plot['val_acc'])
plot_history1(history_plot)

# testset[i] : (image, label)  
testset[10][0].unsqueeze(dim=0).shape 

n_index = 0
n_plot = 10
plt.figure(figsize=(2.3*n_plot, n_plot))
vggFineTune.eval() # Trainable = False
for i in range(n_plot):
    img_idx = n_index+i
    predict = vggFineTune(testset[img_idx][0].unsqueeze(dim=0))\
                                         .cpu().detach().numpy()
    img = testset[img_idx][0].permute(1, 2, 0)
    plt.subplot(1,n_plot,i+1)
    plt.imshow(img, cmap='gray') 
    plt.title('pre:{}/true:{}'.format(np.argmax(predict),testset[img_idx][1]))

plt.show()
