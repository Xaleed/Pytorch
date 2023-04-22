# PyTorch
PyTorch is a library for Python programs that facilitates building deep learning projects. It emphasizes flexibility and allows deep learning models to be expressed in idiomatic Python. In this post, I’ll write about how to implement a simple linear regression model using PyTorch.
## Table of contents
* [Motivation](#Motivation)
* [import libraries](#import-libraries)
* [Data](#Data)
* [Linear Regression Model](#Linear-Regression-Model)
  * [Gradient Descent Algorithm](#Gradient-Descent-Algorithm)
* [Logistic Regression](#Logistic-Regression)

## Motivation
Hi, it's been two hours since I woke up and had breakfast, and the good news is that we have a four-day holiday in Iran. Well, I want to enjoy this vacation so I decided to prepare some content about PyTorch and shared it on LinkedIn. Honestly, this is a motivation for me to write about PyTorch.
## import libraries
```
#Import library
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import pandas as pd
```
## Data
At first, I create a dataset with three independent and one dependent variables to fit a regression model using Pytorch. I do this in two ways:
```
X1 = np.random.normal(0, 1, size=(100, 1)) 
X2 = np.random.normal(0, 1, size=(100, 1)) 
X3 = np.random.normal(0,1 , size=(100, 1)) 
Y = 1*X1+2*X2+4*X3+3 + np.random.normal(0,1 , size=(100, 1))/1000000
X = np.hstack((X1,X2, X3))
X_Train = Variable(torch.Tensor(X))
Y_Train = Variable(torch.Tensor(Y))
```
You can see  data as dataframe by running the following script:
```
data = {"X1" : X1[:,0], "X2" : X2[:,0], "X3" : X3[:,0], "Y" : Y[:,0]}
df = pd.DataFrame(data)  
df
```
Also, we can create similar data set by using torch:
```

X_Train  = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))
############
Y_Train = y.reshape((-1, 1))

data = {"X1" : X.numpy()[:,0], "X2" : X.numpy()[:,1], "X3" : X.numpy()[:,2], "Y" : y}
df = pd.DataFrame(data)  
df
```
## Linear Regression Model
Regression model:
```
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
```
Mean squared error is considered as a loss function and for optimization, the SGD method is implemented:
```
criterion = torch.nn.MSELoss(reduction='sum')
Optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.0001)
```
The definded model is trained with the following script:
```
for epoch in range(500): 
    yhat = linear_model(X_Train)
    loss = criterion(yhat, Y_Train) 
    Optimizer.zero_grad() 
    loss.backward() 
    Optimizer.step() 
    print('epoch {}, loss function {}'.format(epoch, loss.item()))
```
Now, we can test the trained model:
```
X_Test  = torch.normal(0, 1, (1, 3))
y1 = torch.matmul(X_Test, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0,torch.Size([1]))
Y_Test = y1.reshape((-1, 1))
yhat = linear_model(X_Test)
criterion(yhat, Y_Test)
```
Well, I think many questions come for beginners after going through these steps, for example, what is SGD, lr, epoch and ... ? For answered some of these questions, I start with the GD algorithm. GD stands for Gradient Descent algorithm that is the common optimization algorithm in deep learning.
 ### Gradient Descent Algorithm
 Consider the followng optimization problem:
 ```math
 \underset{\boldsymbol{\theta}}{min}\frac{1}{n}\sum_{i=1}^{n}f_{i}(\boldsymbol{\theta})
 ```
 As $$ \nabla \sum_{i=1}^{n}f_{i}(\boldsymbol{\theta}) = \sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}),$$  gradient descent would repeat:
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t_{k}\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)}), \,\,\,\, k = 1,2,3,...
```
step sizes $t_k$ chosen to be fixed and small. For solving this optimization problem, we can implement the gradient descent algorithm as follows:

 **Algorithm**:
*  input: initial guess $\boldsymbol{\theta}^{(0)}$, step size $t$ (let $t_k$ be constant for all $k$);
* for $k =  1, 2, · · · $ do
$ \,\,\,\,\,\,\,\,\,\,\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)})$
end for
return $\boldsymbol{\theta}^{(k)}$ ;


Now, with this optimization problem, think about the relation between ```k``` and ```t```  with ```epoch``` and ```lr```  in the aforementioned Python code.

Ok, it seems that we need to apply the GD algorithm to the above regression problem. As we all know, we need to find $\hat{y}=\hat{\theta_0}+\hat{\theta_1}x_1+\hat{\theta_2}x_2+\hat{\theta_3}x_3 $ such that:
```math
L(\hat{\theta_0},\hat{\theta_1},\hat{\theta_2},\hat{\theta_3}) =\frac{1}{n} \sum_{i=1}^n(\hat{y}_i-y_i)^2=\min_{\boldsymbol{\theta}}L(\theta_0,\theta_1,\theta_2,\theta_3)
``` 
 It is easy to see that
```math
\frac{\partial L}{\partial\theta_{0}}=\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i}),\\
\frac{\partial L}{\partial\theta_{k}}=\frac{2}{n}\sum_{i=1}^{n}x_{ki}(\hat{y}_{i}-y_{i})=\frac{2}{n}X_k^T\times(\hat{y}-y)\qquad k=1,2,3
```
or in an equivalent formula
```math
\frac{\partial L}{\partial\boldsymbol{\theta}}=\frac{2}{n}X^T\times(\hat{y}-y)
```
Then, continue the following recursive algorithm until convergence:
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-lr \frac{\partial L}{\partial\boldsymbol{\theta}^{(k-1)}},\,\,\,\,\,\,\,\,k=1,2,3,... \,\,\,\,\,\,\,and \,\,\,\,\,\,\,\boldsymbol{\theta}^{(0)} = c
```
where c is orbitrary constant.
Let's convert GD algorithm into code. At first we need create a data set once again:


```
X = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))/1000000
############
X = X.numpy()
Y = y.numpy()
X = np.hstack([np.ones((X.shape[0], 1), X.dtype),X])
```
GD algorithm:
```
par = np.zeros((X.shape[1], 1))
Y = Y.reshape((X.shape[0], 1))
epochs = 1000
lr = 0.0001
for epoch in range(epochs):
    e =  X.dot(par) - Y
    grad = (2/X.shape[0])*(X.T).dot(e)
    par = par - lr*grad
```
Now let's go back to the part where the model is executed with PyTorch. I think the content has become clearer.
For this short article, I studied and used the following works. I tried to write about only some simple concepts. You can find many useful and important concepts in the following list.
* [Stochastic Gradient Descent](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/stochastic-gd.pdf)
* [Mathematical Foundations of Machine Learning](https://skim.math.msstate.edu/LectureNotes/Machine_Learning_Lecture.pdf) (chapter 4)
* LeCun Y, Bengio Y, Hinton G. Deep learning. nature. 2015 May 28;521(7553):436-44. (section 5.9)
* [Gradient Descent For Linear Regression In Python](https://matgomes.com/gradient-descent-for-linear-regression-in-python/)
* [Gradient Descent Algorithm and Its Variants](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
* [How Does the Gradient Descent Algorithm Work in Machine Learning?](https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/)
* [Building a Regression Model in PyTorch](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/)
* [Differences Between Epoch, Batch, and Mini-batch](https://www.baeldung.com/cs/epoch-vs-batch-vs-mini-batch)
* [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
