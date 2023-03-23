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

#!pip install torchinfo 

# Neural Networks
import torch
import torch.nn as nn
from torchinfo import summary
torch.__version__

x_dim = data_dim = features = 25
timesteps = sequence_length = 7
units = cells = 50  

batch_size = 1
inputs = torch.randn(batch_size, timesteps, features) #(1,7,25)
lstm = nn.LSTM(input_size=25,hidden_size=50, batch_first=True)
whole_seq_output, (h_T, c_T) = lstm(inputs) 

ps(whole_seq_output) #(1,7,50)
ps(h_T)              #(1,1,50)
ps(c_T)              #(1,1,50) 

ps(lstm.weight_ih_l0) #(4*units,features)
ps(lstm.weight_hh_l0) #(4*units,units)  

lstm = nn.LSTM(input_size=features,hidden_size=units, proj_size=4)
ps(lstm.weight_ih_l0) #(4*units,features)
ps(lstm.weight_hh_l0) #(4*units,proj_size) 

batch_size = 1
inputs = torch.randn(batch_size, timesteps, features) #(1,7,25)
lstm = nn.LSTM(input_size=features,hidden_size=units, proj_size=4, batch_first=True)
whole_seq_output, (h_T, c_T) = lstm(inputs) 
ps(whole_seq_output) #(1,7,4)
ps(h_T)              #(1,1,4)
ps(c_T)              #(1,1,50) 

inputs = torch.randn(batch_size, 1, features) #(1,1,25)
output, (_,_) = nn.LSTM(25,50, batch_first=True)(inputs)
ps(output) #(1,1,50)  

inputs = torch.randn(batch_size, features) #(1,25)
output = nn.Linear(25, 50)(inputs)
ps(output) #(1,50)  

########## one to one model ##########
# time step이 1인 경우
batch_size=1
input_1 = torch.randn(batch_size, 1, features)
class OneToOneLSTM(nn.Module):
    def __init__(self, features, units, batch_first=False):
        super(OneToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=features, 
                            hidden_size=units, 
                            batch_first=batch_first)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return out
    
model = OneToOneLSTM(features, units, batch_first=True)
output = model(input_1)
ps(output) #(1,1,50)   

summary(model, input_size=(1, batch_size, features))

"""


## 모델구조(data flow)를 볼 수 있는 라이브러리 설치  
# !pip install netron 
# !netron model.onnx

import torch.onnx

torch.onnx.export(model, input_1, 'OneToOneLSTM_batch_first.onnx') 

########## one to one model ##########
# time step이 1인 경우
batch_size=2
batch_first=False #(timesteps, batch_size, features)

input_1 = torch.randn(1, batch_size, features)
class OneToOneLSTM(nn.Module):
    def __init__(self, features, units, batch_first=False):
        super(OneToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=features, 
                            hidden_size=units, 
                            batch_first=batch_first)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return out
    
model = OneToOneLSTM(features, units) #(ts, bs, features) 
output = model(input_1)
ps(output) #(1,2,50)  

summary(model, input_size=(1, batch_size, features))

torch.onnx.export(model, input_1, 'OneToOneLSTM.onnx') 

########## many to many ##########
batch_size=2
inputs = torch.randn(batch_size, timesteps, features)
class ManyToManyLSTM(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(ManyToManyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=features, 
                            hidden_size=units, 
                            batch_first=batch_first)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return out

model = ManyToManyLSTM(features, units)
output = model(inputs)
output.shape #(2,7,50)  

summary(model, input_size=(batch_size, timesteps, features))

torch.onnx.export(model, inputs, 'ManyToManyLSTM.onnx')

########## many to one ##########
inputs = torch.randn(batch_size, timesteps, features)
class ManyToOneLSTM(nn.Module):
    def __init__(self, features, units):
        super(ManyToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=units)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x) #(2,7,50)
        return out[:, -1, :] #(2,50)

model = ManyToOneLSTM(features, units)
output = model(inputs)
ps(output) #(2,50) 

summary(model, input_size=(batch_size, timesteps, features))

torch.onnx.export(model, inputs, 'ManyToOneLSTM.onnx')

## contiguous vs. non-contiguous tensor 
import numpy as np
a = torch.tensor(np.arange(12).reshape(3,4)) 
av = a.view(4,3) 
at = a.T   
att = at.T 
ar = at.reshape(3,4)
arr = ar.reshape(3,4)
ap = at.permute(1,0)
app = ap.permute(1,0)

avv = av.view(3,4)
acv = at.contiguous().view(3,4)
## error!! non-contiguous tensor  
av = at.view(3,4) 

#print(a)
#print(av)
#print(at)
#print(att)
#print(as_)
#print(ass_)
#print(ap)
#print(app)
#print(avv) 
#print(acv)
#print(av) 

########## many to many : TimeDistributed-1 ##########
inputs = torch.randn(batch_size, timesteps, features)
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x) 
        y = self.module(x[:,0,:]).reshape(x.size(0),1,-1) #(2,1,1)
        for i in range(x.size()[1]-1): #(2,7,1) <- cat((2,1,1),dim=1)
          y = torch.cat((y,self.module(x[:,i+1,:]).\
                         reshape(x.size(0),1,-1)),dim=1)
        if not  self.batch_first:
            y = y.transpose(0,1).contiguous() #(7,2,1)  
        return y

batch_size = 2
class ManyToMany(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(ManyToMany, self).__init__()
        self.lstm = nn.LSTM(features, units, batch_first=batch_first)
        self.timeDistributed = TimeDistributed(nn.Linear(units, 1), 
                                               batch_first=batch_first)

    def forward(self, x): 
        output, (h_T, c_T) = self.lstm(x) #(2,7,50)
        y = self.timeDistributed(output)  #(2,7,1)
        return y
    
model = ManyToMany(features, units, batch_first=True)    
whole_seq_output = model(inputs) #(2,7,1)<-(2,7,25)
summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'graph_many2one.onnx')  

########## many to many : TimeDistributed-2 ##########
inputs = torch.randn(batch_size, timesteps, features)
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x) 
        y = self.module(x[:,0,:]).reshape(x.size(0),1,-1)
        for i in range(x.size()[1]-1): 
          y = torch.cat((y,self.module(x[:,i+1,:]).\
                         reshape(x.size(0),1,-1)),dim=1)
        if not  self.batch_first:
            y = y.transpose(0,1).contiguous()
        return y

batch_size = 2
class ManyToMany(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(ManyToMany, self).__init__()
        self.timeDistributed = TimeDistributed(nn.Linear(features, features), 
                                               batch_first=batch_first)
        self.lstm = nn.LSTM(features, units, batch_first=batch_first)
        self.linear = nn.Linear(units,1)
    def forward(self, x): 
        output = self.timeDistributed(x)  #(2,7,25)<-(2,7,25)
        output, (h_T, c_T) = self.lstm(output) #(2,7,50)
        output = self.linear(output) #(2,7,1)
        return output
    
model = ManyToMany(features, units, batch_first=True)    
whole_seq_output = model(inputs) #(2,7,1)<-(2,7,25)
ps(whole_seq_output,'whole_seq_output')  

summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'graph_many2one2.onnx')  

########### LSTM AutoEncoder ############
latent_dim = 4
inputs = torch.randn(batch_size, timesteps, features) 
class Many2ManyRepeated(nn.Module):
    def __init__(self, features, latent_dim, batch_first=True):
        super(Many2ManyRepeated, self).__init__()
        self.encoder = nn.LSTM(features, latent_dim, 
                               batch_first=batch_first)
        self.decoder = nn.LSTM(latent_dim, features, 
                               batch_first=batch_first)
    def forward(self, x):
        encoded, _ = self.encoder(x) #(2,7,4)<-(2,7,25)
        decoder_input = encoded[:, -1:].repeat(1, timesteps, 1)
        decoded, _ = self.decoder(decoder_input) #(2,7,25)<-(2,7,4)
        return decoded 
    
model = Many2ManyRepeated(features, latent_dim)
output = model(inputs)
ps(output) #(2,7,25)
summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'Many2ManyRepeated.onnx')

features = 20
L_units = 6
inputs = torch.randn(batch_size, timesteps, features)

class ParallelLSTM(nn.Module):
    def __init__(self, features, units, l_units, batch_first=True):
        super(ParallelLSTM, self).__init__()
        self.lstm1 = nn.LSTM(int(features/2), units, 
                             batch_first=batch_first)
        self.lstm2 = nn.LSTM(int(features/2), units, 
                             batch_first=batch_first)
        self.linear = nn.Linear(units*2, l_units) #()
        
    def forward(self, x):
        input1 = x[:, :, :int(features/2)] #(2,7,10)
        input2 = x[:, :, int(features/2):] #(2,7,10)
        
        output1, _ = self.lstm1(input1)    #(2,7,50)
        output2, _ = self.lstm2(input2)    #(2,7,50)
        concat = torch.cat([output1, output2], dim=-1)#(2,7,100) 
        
        output = self.linear(concat) #(2,7,6)<-(2,7,100)
        output = torch.tanh(output)
        return output
    
model = ParallelLSTM(features, units, L_units)
output = model(inputs)
ps(output)
summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'ParallelLSTM.onnx')

##### Stacked LSTM-1 #####
inputs = torch.randn(batch_size,timesteps,features)
class StackedLSTM(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(StackedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(features, units, 
                             batch_first=batch_first)
        self.lstm2 = nn.LSTM(units, units, 
                             batch_first=batch_first)
        self.linear = nn.Linear(units, 1)
    def forward(self, x):        
        x, _ = self.lstm1(x)
        x = torch.relu(x)
        x, _ = self.lstm2(x)
        x = torch.relu(x)
        output = self.linear(x)
        return output

model = StackedLSTM(features, units)
output = model(inputs)
summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'StackedLSTM.onnx')

##### Stacked LSTM-2 #####
inputs = torch.randn(batch_size,timesteps,features)
class StackedLSTM(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(StackedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(features, units,
                             num_layers=2, 
                             batch_first=batch_first)
        self.linear = nn.Linear(units, 1)
    def forward(self, x):        
        x, _ = self.lstm1(x)
        x = torch.relu(x)
        output = self.linear(x)
        return output

model = StackedLSTM(features, units)
output = model(inputs)
summary(model, input_size=(batch_size, timesteps, features))  

torch.onnx.export(model, inputs, 'StackedLSTM2.onnx')

##### Bidirectional LSTM #####
class BidirectionalLSTMManyToMany(nn.Module):
    def __init__(self, features, units, batch_first=True):
        super(BidirectionalLSTMManyToMany, self).__init__()
        self.biLstm = nn.LSTM(features, units, 
                              bidirectional=True,
                              batch_first=batch_first)
        self.linear = nn.Linear(units*2, 1)
    def forward(self, x):        
        out, (hn, cn) = self.biLstm(x)
        x = self.linear(out) # (2,7,1) 
        # x = self.linear(out[:, -1, :]) #(2,1)
        return x
    
model = BidirectionalLSTMManyToMany(features, units)
output = model(inputs)
output.shape  

summary(model, input_size=(batch_size, timesteps, features))

torch.onnx.export(model, inputs, 'BidirectionalLSTM.onnx')





"""
