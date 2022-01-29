import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split


#CHANGE IT TO SELECT CLASS
nik = 1

#CHANGE IT
df = pd.read_csv('train_0{}.csv'.format(nik))

def feature_engineering(ds,c):
    t = TfidfVectorizer(ngram_range=(1, 3),max_features = 3000, dtype=np.float32)
    tres = t.fit_transform(ds[c])
    print(tres.shape)
    tres = np.array(tres.todense(), dtype=np.float32)
    for j in range(tres.shape[1]):
        ds['{}_tfidf_{}'.format(c,j)] = tres[:, j]
    ds.drop(c, axis = 1, inplace = True)
    return (ds,t)

#CHANGE IT
tr = pd.read_csv('xtrain.csv')
tst = pd.read_csv('xtest.csv')

tr = tr.set_index('Unnamed: 0')
tst = tst.set_index('Unnamed: 0')

train = pd.merge(tr,df['target'], left_index=True, right_index=True)
test = pd.merge(tst,df['target'], left_index=True, right_index=True)

del tr
del tst
del df

X_train = train.iloc[:,0:-1]
y_train = train.iloc[:,-1]
X_test = test.iloc[:,0:-1]
y_test = test.iloc[:,-1]

del train
del test

EPOCHS = 75
BATCH_SIZE = 2048*2
LEARNING_RATE = 0.00001

class LoadData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = LoadData(torch.FloatTensor(np.array(X_train,dtype=np.float16)), 
                       torch.FloatTensor(np.array(y_train)))


class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = LoadData(torch.FloatTensor(np.array(X_test,dtype=np.float16)),torch.FloatTensor(np.array(y_test)))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class NClassifierLP(nn.Module):
    def __init__(self):
        super(NClassifierLP, self).__init__()
        self.layer_1 = nn.Linear(6006, 12012) 
        self.layer_2 = nn.Linear(12012, 6006)
        self.layer_3 = nn.Linear(6006, 9)
        self.layer_out = nn.Linear(9, 1) 
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.005)
        self.batchnorm = nn.BatchNorm1d(12012)
        self.sig = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm(x)
        x = self.tanh(self.layer_2(x))
        x = self.tanh(self.layer_3(x))
        x = self.dropout(x)
        x = self.sig(self.layer_out(x))
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = NClassifierLP()
model.to(device)
print(model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

from sklearn.metrics import f1_score

def score(y_p, y_t):
    y_pred_tag = torch.round(y_p)
    return f1_score(y_t.cpu().detach().numpy(),y_pred_tag.cpu().detach().numpy())

for e in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_loss_valid = 0
    epoch_acc_valid = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = score(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    model.eval()
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        y_pred = model(X_batch)
        
        loss_valid = criterion(y_pred, y_batch.unsqueeze(1))
        acc_valid = score(y_pred, y_batch.unsqueeze(1))
        
        epoch_loss_valid += loss.item()
        epoch_acc_valid += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss_train: {epoch_loss/len(train_loader):.5f} | F1_train: {epoch_acc/len(train_loader):.3f}| Loss_valid: {epoch_loss_valid/len(test_loader):.5f} | F1_valid: {epoch_acc_valid/len(test_loader):.3f}')
    
