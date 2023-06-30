# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
#from torchvision import datasets
# %%
Path = "D:\\CriptocurrencyAlgorithm\\Data\\"
DATA = pd.read_csv(Path+"doge_3monthsago_1min_data.csv")
# %%
HP = DATA['h']
High = DATA['h']
Low = DATA['l']
Open = DATA['o']
Close = DATA['c']
n_moving_L = 9
p1 = .9
n_moving_H = 9
p2 = .9
L = pd.Series(Low).rolling(n_moving_L).quantile(p1, interpolation='midpoint')
H = pd.Series(High).rolling(n_moving_H).quantile(p2, interpolation='midpoint')
d = {  'X1' : (Open - Close) / Open,'X2' : (Close - Low) / Close, 'X3' : (Low - L) / Low}
dataL = pd.DataFrame(data=d)
#[int(x >= 0) for x in L[0:(len(L)-1)] - Low[1:(len(L))]]
dataL['Y'] = [int(x >= 0) for x in L[0:(len(L)-1)] - Low[1:(len(L))]]
dataL = dataL.dropna()

i = 7000
data = dataL[1+5000:i+5000]
sequence_length = 2
features = ['X1', 'X2', 'X3']
target = 'Y'
# %%
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
# %%
batch_size = 1
dataset = SequenceDataset(
    data,
    target=target,
    features=features,
    sequence_length=sequence_length
)
k = []
for i in range(len(dataset)):
    if (i%6 ==0):
        k.append(i)
k2 = list(set(range(len(dataset)))-set(k))
train_dataset = torch.utils.data.Subset(dataset, k2)
test_dataset = torch.utils.data.Subset(dataset, k)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
A = torch.utils.data.Subset(dataset, [(len(dataset)-1)])
S = DataLoader(A, batch_size=batch_size, shuffle=False)
print(S)
X_Target, y_Target = next(iter(S))
X_Target
# %%
class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
# %%
print("datatype of the 1st training sample: ", X_Target.type())
print("size of the 1st training sample: ", y_Target.size())
# %%
n_inputs = 3*2 # makes a 1D vector of 784
n_outputs = 1
log_regr = LogisticRegression(n_inputs, n_outputs)
# %%
for X, y in train_loader:
    for i in range(len(X)):
        print(X[i].reshape([sequence_length*3]))
# %%
# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.001)
# defining Cross-Entropy loss
criterion = torch.nn.MSELoss()
# %%
torch.round(torch.tensor([1.7]))
# %%
Loss = []
acc = []
epochs = 30
for epoch in range(epochs):
    for X, labels in train_loader:
        for i in range(len(X)):
            optimizer.zero_grad()
            #print(X[i].reshape([sequence_length*3]))
            #print(labels[i])
            outputs = log_regr(X[i].reshape([sequence_length*3]))
            loss = criterion(outputs, labels[i])
            #print(loss)
           # print(outputs)
            #print(labels)

            # Loss.append(loss.item())
            loss.backward()
            optimizer.step()
            #break
    Loss.append(loss)
    #break
    correct = 0
    #break
    for X, labels in test_loader:
        for i in range(len(X)):
            predicted = log_regr(X[i].reshape([sequence_length*3]))
            #print(outputs)
            #predicted = torch.max(outputs, torch.tensor([1]))
            #print(predicted)
            #print(torch.round(predicted))
            #print(labels)
            correct += (torch.round(predicted) == labels[i]).sum()
    accuracy = 100 * (correct) / len(test_dataset)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss, accuracy))
# %%
#if batch_size = 1
Loss = []
acc = []
epochs = 430
for epoch in range(epochs):
    for X, labels in train_loader:
        optimizer.zero_grad()
        outputs = log_regr(X.reshape([sequence_length*3]))
        loss = criterion(outputs, labels)
        #print(loss)
       # print(outputs)
        #print(labels)
        
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
    Loss.append(loss)
    correct = 0
    #break
    for X, labels in test_loader:
        outputs = log_regr(X.reshape([sequence_length*3]))
        #print(outputs)
        #predicted = torch.max(outputs, torch.tensor([1]))
        #print(predicted)
        #print(torch.round(predicted))
        #print(labels)
        correct += (torch.round(predicted) == labels).sum()
    accuracy = 100 * (correct) / len(test_dataset)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss, accuracy))
    #break
# %%
