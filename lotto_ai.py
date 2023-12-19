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

import datetime
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

lotto_data = pd.read_csv("./lotto-1052.csv")

print(lotto_data.head())
print(lotto_data.tail())

def numbers2ohbin(numbers):
    ohbin = np.zeros(45,dtype=np.float64) #45개의 빈 칸을 만듬
    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):
    numbers = []
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    return numbers# 최대,최소값으로 정규화 하기




#print("1:"+str(numbers2ohbin(lotto_data.drop(["date"],axis=1).loc[2,:].values.flatten().tolist())))

with open("output.txt", "a") as file:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write("Current Time: " + current_time + "\n")

count = 0
while count < 10:
    count +=1
    row_count = len(lotto_data)
    print("row count: " + str(len(lotto_data)))

    ohbins = lotto_data.iloc[:,1:7].apply(numbers2ohbin,axis=1) 
    x_samples = np.array([list(ohbin) for ohbin in ohbins[0:-1]], dtype=np.float64)
    y_samples = np.array([list(ohbin) for ohbin in ohbins[1:]], dtype=np.float64)

    xy = np.array(ohbins) #np <- pandas data format 

    timesteps = seq_length = row_count
    data_dim  =45
    batch_size = 32
    timesteps = seq_length = 30

    dataX = []
    dataY = []
    for i in range(0, row_count-1-seq_length):
        _x = x_samples[i:i+seq_length]
        _y = y_samples[i+seq_length] 
        dataX.append(_x)
        dataY.append(_y)

    # train data : test data = 9 : 1
    train_size = int(len(dataY) * 0.9)
    test_size  = len(dataY) - train_size
    x_train = np.array(dataX[0:train_size])
    y_train = np.array(dataY[0:train_size])
    x_test  = np.array(dataX[train_size:len(dataX)])
    y_test  = np.array(dataY[train_size:len(dataY)] )

    ps(x_train,'x_train')
    ps(y_train,'y_train')
    ps(x_test,'x_test')
    ps(y_test,'y_test')

    class CustomDataset(Dataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return torch.FloatTensor(self.x[idx]),torch.FloatTensor(self.y[idx]) 

    trainDataset = CustomDataset(x_train, y_train)
    testDataset = CustomDataset(x_test, y_test)
    trainDataLoader = DataLoader(trainDataset, shuffle=True, drop_last=False, 
                                batch_size=batch_size)
    testDataLoader = DataLoader(testDataset, drop_last=False, batch_size=batch_size)    

    hidden_dim = 32
    threshold = 0.7
    learning_rate=0.01

    class LottoModel(pl.LightningModule):
        def __init__(self, input_features, hidden_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_features, hidden_dim,
                                batch_first=True)
            self.linear = nn.Linear(hidden_dim, 45)
            self.relu=nn.ReLU()
        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.relu(x)
            x = self.linear(x[:, -1,:])
            return x
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred_origin = self(x)
            y_pred = (y_pred_origin > threshold).float()
            loss = loss_function(y_pred_origin, y)
            accuracy = torch.mean((y_pred == y).float())
            metrics={'loss':loss, 'acc':accuracy}
            self.log_dict(metrics)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_pred_origin = self(x)
            y_pred = (y_pred_origin > threshold).float()
            loss = loss_function(y_pred_origin, y)
            accuracy = torch.mean((y_pred == y).float())
            metrics = {'val_loss':loss, 'val_acc':accuracy}
            self.log_dict(metrics)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=learning_rate)  

    model_S = LottoModel(data_dim, hidden_dim)
    summary(model_S, input_size=(batch_size, timesteps, data_dim))

    loss_function = nn.BCEWithLogitsLoss()
    trainer2 = Trainer(max_epochs=1300,accelerator="auto")
    trainer2.fit(model_S, trainDataLoader, testDataLoader)  
    v_num_S = model_S.logger.version   

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    log = {} 

    log_dir = f'lightning_logs/version_{v_num_S}' 

    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    val_loss = [x.value for x in event_accumulator.Scalars('val_loss')]
    train_loss = [x.value for x in event_accumulator.Scalars('loss')]
    val_accuracy = [x.value for x in event_accumulator.Scalars('val_acc')]
    train_accuracy = [x.value for x in event_accumulator.Scalars('acc')]
    log['model_S'] = [train_loss, val_loss, val_accuracy, train_accuracy ]  



    model_S.eval()  # 모델을 평가 모드로 변경
    predictions = []
    with torch.no_grad():
        for x, _ in testDataLoader:
            y_pred = model_S(x)
            predictions.append(y_pred.detach().numpy())
    predictions = np.concatenate(predictions, axis=0)

    print(predictions)

    with torch.no_grad():
        inputs = torch.FloatTensor(x_test[-30:])
        predictions = model_S(inputs)
        predictions = torch.sigmoid(predictions)
        print("this is weekly lotto")
        print(predictions[-1:])
        predictions_list = predictions[-1:].tolist()
        sorted_indices = sorted(range(len(predictions_list[0])), key=lambda i: predictions_list[0][i], reverse=True)

        # 1부터 45까지의 인덱스 출력

        with open("output.txt", "a") as file:
            i = 1
            out_str = "행운의 숫자 : "
            for index in sorted_indices[:6]:
                out_str += str(index + 1) 
                out_str += " " 
                i += 1
            out_str += "\n"
            file.write(out_str)

# # train mse compare
# plt.plot(log['model_S'][0], label='model_S_train_loss')
# plt.plot(log['model_S'][1], label='model_S_val_loss')
# plt.plot(log['model_S'][2], label='model_S_val_acc')
# plt.plot(log['model_S'][3], label='model_S_train_acc')
# plt.title("Training mse")
# #plt.semilogy()  # y축에 log scale적용
# plt.grid(True)
# plt.legend(loc='best')
# plt.show()  