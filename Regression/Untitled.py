#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Import library
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import pandas as pd


# # Create a Dataset

# In[14]:


X1 = np.random.normal(0, 1, size=(100, 1)) 
X2 = np.random.normal(0, 1, size=(100, 1)) 
X3 = np.random.normal(0,1 , size=(100, 1)) 
Y = 1*X1+2*X2+4*X3+3 + np.random.normal(0,1 , size=(100, 1))
X = np.hstack((X1,X2, X3))
X_Train = Variable(torch.Tensor(X))
Y_Train = Variable(torch.Tensor(Y))


# In[15]:


data = {"X1" : X1[:,0],"X2" : X2[:,0],"X3" : X3[:,0], "Y":Y[:,0]}
df = pd.DataFrame(data)  
df


# In[16]:


X = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))/1000000
############
Y = y.reshape((-1, 1))
X_Train = X
Y_Train = Y
data = {"X1" : X.numpy()[:,0],"X2" : X.numpy()[:,1],"X3" : X.numpy()[:,2], "Y":y}
df = pd.DataFrame(data)  
df


# In[ ]:





# In[17]:


X_Train  = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))/100000
############
Y_Train = y.reshape((-1, 1))


# # Define the model

# In[18]:


InputDim = 3
OutputDim = 1
class LinearRegression(torch.nn.Module):
    def __init__(self): 
        super(LinearRegression, self).__init__() 
        self.linear = torch.nn.Linear(InputDim, OutputDim)  
    def forward(self, x): 
        y_hat = self.linear(x) 
        return y_hat 
linear_model = LinearRegression()


# # Define the Loss Function and an Optimization Algorithm

# In[19]:


#define_criterion = torch.nn.MSELoss(size_average=False) replace with:
criterion = torch.nn.MSELoss( reduction='sum')
Optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.000001)


# In[20]:


for epoch in range(5000): 
    yhat = linear_model(X_Train)
    loss = criterion(yhat, Y_Train) 
    Optimizer.zero_grad() 
    loss.backward() 
    Optimizer.step() 
    print('epoch {}, loss function {}'.format(epoch, loss.item()))


# In[21]:


test_variable = torch.randn(1, 3, requires_grad=True)

predict_y = linear_model(test_variable)
predict_y


# In[11]:


X_Test  = torch.normal(0, 1, (1, 3))
y1 = torch.matmul(X_Test, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0,torch.Size([1]))
Y_Test = y1.reshape((-1, 1))
yhat = linear_model(X_Test)
criterion(yhat, Y_Test)


# In[631]:


Y_Test


# # Now I want to go deeper into this topic
# * https://www.docker.com/blog/how-to-train-and-deploy-a-linear-regression-model-using-pytorch-part-1/

# In[110]:


from torch.utils import data


# In[573]:


num_examples = 1000
true_m = torch.tensor([2, -3.4, 8])
true_c = 4.2
X = torch.normal(0, 1, (num_examples, len(true_m)))
y = torch.matmul(X, true_m) + true_c
y += torch.normal(0, 0.01, y.shape)


# In[552]:


y


# In[547]:





# In[553]:


labels = y.reshape((-1, 1))
labels


# In[114]:


batch_size = 10


# In[115]:


data_arrays = (X, labels)


# In[116]:


dataset = data.TensorDataset(*data_arrays)
dataset


# In[117]:


data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
data_iter


# In[123]:


for X, y in data_iter:
    print(X,y)


# In[124]:


for X, y in data_iter:
    print(net(X),y)


# In[126]:


#Step4: Define model &amp;amp;amp; initialization
 
# Create a single layer feed-forward network with 2 inputs and 1 outputs.
 
net = nn.Linear(3, 1)
 

 
#Initialize model params
 
net.weight.data.normal_(0, 0.01)
 
net.bias.data.fill_(0)


# In[120]:


loss = nn.MSELoss()


# In[121]:


#Step 6: Define optimization algorithm
# implements a stochastic gradient descent optimization method
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# In[128]:


num_epochs = 5
 
for epoch in range(num_epochs):
 
    for X, y in data_iter:
 
        l = loss(net(X) ,y)
 
        trainer.zero_grad() #sets gradients to zero
 
    l.backward() # back propagation
 
    trainer.step() # parameter update
 
    l = loss(net(features), labels)
 
print(f'epoch {epoch + 1}, loss {l:f}')


# # batch 
# * https://deeplizard.com/learn/video/U4WB9p6ODjM
# * https://www.analyticsvidhya.com/blog/2021/03/introduction-to-batch-normalization/
# * https://www.baeldung.com/cs/epoch-vs-batch-vs-mini-batch
# * https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
# * https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/
# * https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/
# * https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3

# In[ ]:




