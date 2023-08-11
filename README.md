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
     * [Stochastic Gradient Descent](#Stochastic-Gradient-Descent)
     * [mini-batch stochastic gradient descent](#mini-batch-stochastic-gradient-descent)
  * [With a real data set](#With-a-real-data-set)
* [Learning PyTorch with Examples](#Learning-PyTorch-with-Examples)
* [Examples from linkedin](#Examples-from-linkedin)

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

For this short article, I studied and used the following sources. I tried to write about only some simple concepts. You can find many useful and important concepts in the following list.

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
```
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
```

define logistic model:
```
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
```
creating a data set:
```
X = torch.normal(0, 1, (10000, 5))
y = log_regr(X)
a = torch.empty(10000, 1).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
Y = torch.max(y.round().detach() , torch.bernoulli(a))
print(Y)
print((y.round().detach()))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
```
defining the optimizer and loss function:
```
optimizer = torch.optim.SGD(log_regr.parameters(), lr=0.0001)
criterion = torch.nn.BCELoss()
```
GD algorithm:
```
for epoch in range(100):
    y_pred = lr(X_train)
    loss = criterion(y_pred, y_train) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
```
CALCULATING ACCURACY:
```
with torch.no_grad():
    y_predicted = log_regr(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
```
#### Stochastic Gradient Descent
 Consider the followng optimization problem:
 ```math
 \underset{\boldsymbol{\theta}}{min}\frac{1}{n}\sum_{i=1}^{n}f_{i}(\boldsymbol{\theta})
 ```
 As 
 $$ \nabla \sum_{i=1}^{n}f_{i}(\boldsymbol{\theta}) = \sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}),$$ 
stochastic gradient descent repeats:
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t_{k}.\nabla f_{i_k}(\boldsymbol{\theta}^{(k-1)}), \,\,\,\, k = 1,2,3,...
```
where $i_k \in \{1,...,m\}$ is some chosen index at iteration $k$:
* Randomized rule:choose $i_k \in \{1,2, ..., m\}$ uniformly  at random.
* cyclic rule: choose $i_k = 1,2, ..., m, 1,2,..., m, ...$
#### mini-batch stochastic gradient descent
we should repeat:
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t_{k}\frac{1}{b}\sum_{i\in I_k}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)}), \,\,\,\, k = 1,2,3,...
```
where $I_k$ is chosen randomly.
### With a real data set
## Learning PyTorch with Examples
This section is exactly what is in this [link](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).
Create random input and output data
```
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)
```
Consider the following prediction
```math
y_{pred} = \theta_0 + \theta_1 X + \theta_2 X^2 + \theta_3 X^3
```
and the foollowing loss function
```math
L=\sum (y-y_{pred})^2
```
Now, we apply the following algorithm to the data
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-lr \frac{\partial L}{\partial\boldsymbol{\theta}^{(k-1)}},\,\,\,\,\,\,\,\,k=1,2,3,... \,\,\,\,\,\,\,and \,\,\,\,\,\,\,\boldsymbol{\theta}^{(0)} = c
```
```
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
learning_rate = 1e-6
for t in range(20000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

## Examples from linkedin
You can see the source of this example from this [link](https://www.linkedin.com/feed/update/urn:li:activity:7080475464179277824/).

  Consider per-minute data on dogecoin transactions for three months. In this section, we applied the logistic regression model based on Stochastic gradient descent using the PyTorch library.
