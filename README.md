# PyTorch
PyTorch is an open-source machine learning framework. It is a Python-based scientific computing package that adopts the power of graphics processing units (GPUs) and deep learning techniques to proffer maximum flexibility and speed. [You can find more information about PyTorch on their official website](https://pytorch.org/). In this post, I’ll write about how to implement a simple linear regression model using PyTorch.
Admittedly, in order to find answers to all the questions that arise when implementing a model with some packages in a programming language we need to know a little bit about the theory behind what that package does. With this perspective, let’s talk about performing a linear regression model with PyTorch and answer some questions about it.
## Table of contents
* [Motivation](#Motivation)
* [import libraries](#import-libraries)
* [Data](#Data)
* [Linear Regression Model](#Linear-Regression-Model)
  * [Gradient Descent Algorithm](#Gradient-Descent-Algorithm)
* [Logistic Regression](#Logistic-Regression)
  * [With a created data set](#With-a-created-data-set)
  * [With a real data set](#With-a-real-data-set)

## Motivation
I enjoy writing about statistics and math topics! It’s so  riveting to learn about how these concepts  are appled to the real world and how they can be used to solve problems. When I write about these topics, it gives me the motivation to keep learning and exploring new ideas.
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
At first, I create a dataset with three independent variables and one dependent variable to fit a regression model using Pytorch. I do this in two ways:
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

X = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))/1000000
############
Y = y.reshape((-1, 1))
X_Train = X
Y_Train = Y

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
Mean squared error is considered as the loss function and the GD algorithm is implemented for optimization
```
criterion = torch.nn.MSELoss(reduction='sum')
Optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.0001)
```
The following script is used to train the defined model:
```
for epoch in range(500): 
    yhat = linear_model(X_Train)
    loss = criterion(yhat, Y_Train) 
    Optimizer.zero_grad() 
    loss.backward() 
    Optimizer.step() 
    print('epoch {}, loss function {}'.format(epoch, loss.item()))
```
Now, we can test the trained model as follows:
```
X_Test  = torch.normal(0, 1, (1, 3))
y1 = torch.matmul(X_Test, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0,torch.Size([1]))
Y_Test = y1.reshape((-1, 1))
yhat = linear_model(X_Test)
criterion(yhat, Y_Test)
```
Well, it is possible that some questions or ambiguities might be posed here, especially for beginners after going through these steps. For starters, what is SGD, lr, epoch and ... ? For answering some of these questions, I start with the GD algorithm. GD stands for Gradient Descent algorithm that is the common optimization algorithm in deep learning.
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
* for $k =  1, 2, · · · $ do: 
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)})
```

* return $\boldsymbol{\theta}^{(k)}$ ;

By considering this optimization problem, think about the relation between ```k``` and ```t``` with ```epoch``` and ```lr``` in the Python code mentioned earlier. Did you find any relation?

All right, it seems that we need to apply the GD algorithm to the above regression problem. As we all know, we need to find $\hat{y}=\hat{\theta_0}+\hat{\theta_1}x_1+\hat{\theta_2}x_2+\hat{\theta_3}x_3 $ such that:
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
Now, with these explanations in mind, we can convert GD algorithm into code. At first, we need to create a data set once again:


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
Let’s go back to the part where we ran the model with PyTorch and read it once more. I believe that the content has become clearer now.

For this short article, I studied and used the following sources. I tried to write about only some simple concepts. You can find many useful and important concepts in the following list. In addition, I have created a [repository on GitHub](https://github.com/Xaleed/Pytorch) for more complex cases such as logistic regression, time series, LSTM, and etc. I would be more than glad if you could add something to it.

* [Stochastic Gradient Descent](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/stochastic-gd.pdf)
* [Mathematical Foundations of Machine Learning](https://skim.math.msstate.edu/LectureNotes/Machine_Learning_Lecture.pdf) (chapter 4)
* LeCun Y, Bengio Y, Hinton G. Deep learning. nature. 2015 May 28;521(7553):436-44. (section 5.9)
* [Gradient Descent For Linear Regression In Python](https://matgomes.com/gradient-descent-for-linear-regression-in-python/)
* [Gradient Descent Algorithm and Its Variants](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)
* [How Does the Gradient Descent Algorithm Work in Machine Learning?](https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/)
* [Building a Regression Model in PyTorch](https://machinelearningmastery.com/building-a-regression-model-in-pytorch/)
* [Differences Between Epoch, Batch, and Mini-batch](https://www.baeldung.com/cs/epoch-vs-batch-vs-mini-batch)
* [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
 ## Logistic Regression
  ### With a created data set
  ### With a real data set
  
