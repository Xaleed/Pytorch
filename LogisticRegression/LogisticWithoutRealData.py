#!/usr/bin/env python
# coding: utf-8

# In[177]:


import torch
#import torchvision.transforms as transforms
#from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

#



# In[178]:


class LogisticRegression(torch.nn.Module):    
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

log_regr = LogisticRegression(5, 1)
# %%
X = torch.normal(0, 1, (10000, 5))
print(X)
y = log_regr(X)
a = torch.empty(10000, 1).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
Y = torch.max(y.round().detach() , torch.bernoulli(a))
print(y)
print((y.round().detach()))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
# %%
sum(Y)/len(Y)
# %%
# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.0001)
# defining Cross-Entropy loss
criterion = torch.nn.BCELoss()


# In[253]:


for epoch in range(100):
    y_pred = log_regr(X_train)
    loss = criterion(y_pred, y_train) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')



# In[254]:
with torch.no_grad():
    y_predicted = log_regr(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
# %%
X_test
# %%
y_predicted
# %%
y_predicted_cls

# In[255]:
y_predicted_cls.eq(y_test).sum()
# %%
y_test
# %%
#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted_cls))


# In[256]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predicted_cls)
print(confusion_matrix)


# In[257]:


from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, X,y):
        self.y = y
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# In[258]:
2**132
# %%

batch_size = len(y_train)
train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# In[259]:


n_inputs = 5
n_outputs = 1
log_regr = LogisticRegression(n_inputs, n_outputs)

# defining the optimizer
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.0001)
# defining Cross-Entropy loss
criterion = torch.nn.CrossEntropyLoss()


# In[260]:


epochs = 20
Loss = []
acc = []
for epoch in range(epochs):
    for images, labels in train_loader:
        #optimizer.zero_grad()
        outputs = log_regr(images)
        loss = criterion(outputs, labels)
        # Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
    correct = 0
    for images, labels in test_loader:
        outputs = log_regr(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
    accuracy = 100 * (correct.item()) / len(test_dataset)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))

plt.plot(Loss)
plt.xlabel("no. of epochs")
plt.ylabel("total loss")
plt.title("Loss")
plt.show()

plt.plot(acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.title("Accuracy")
plt.show()


# In[261]:


with torch.no_grad():
    y_predicted = log_regr(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')


# In[262]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted_cls))


# In[263]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_predicted_cls)
print(confusion_matrix)


# In[ ]:




