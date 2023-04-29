#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import library
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import pandas as pd


# # Create a Dataset

# In[2]:


X = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))/1000000
############
X = X.numpy()
Y = y.numpy()
X = np.hstack([np.ones((X.shape[0], 1), X.dtype),X])


# # Gradient Descent

# In[3]:


par = np.zeros((X.shape[1], 1))
Y = Y.reshape((X.shape[0], 1))
epochs = 1000
lr = 0.0001
for epoch in range(epochs):
    e =  X.dot(par) - Y
    grad = (2/X.shape[0])*(X.T).dot(e)
    par = par - lr*grad


# In[4]:


par


# In[ ]:




