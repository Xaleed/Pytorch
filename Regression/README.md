# PyTorch
## Table of contents
* [Motivation](#Motivation)
* [Data](#Data)
* [Linear Regression Model](#Linear-Regression-Model)
  * [Gradient Descent Algorithm](#Gradient-Descent-Algorithm)
* [Logistic Regression](#Logistic-Regression)

## Motivation
Hi, it's been two hours since I woke up and had breakfast, and the good news is that we have a four-day holiday in Iran. Well, I want to enjoy this vacation so I decided to prepare some content about PyTorch and shared it on LinkedIn. Honestly, this is a motivation for me to write about PyTorch.
	
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

X_Train  = torch.normal(0, 1, (1000, 3))
y = torch.matmul(X, torch.tensor([1.0, 2, 4])) + 3 + torch.normal(0, 1.0, torch.Size([1000]))
############
Y_Train = y.reshape((-1, 1))

data = {"X1" : X.numpy()[:,0], "X2" : X.numpy()[:,1], "X3" : X.numpy()[:,2], "Y" : y}
df = pd.DataFrame(data)  
df
```
## Linear Regression Model
I define a simple linear regression model:
```
InputDim = 3
OutputDim = 1
class LinearRegression(torch.nn.Module):
    def __init__(self): 
        super(LinearRegression, self).__init__() 
        self.linear = torch.nn.Linear(InputDim, OutputDim)  
    def forward(self, x): 
        predict_y = self.linear(x) 
        return predict_y 
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
Well, I think many questions come for beginners after going through these steps, for example, what is SGD, lr, epoch and ... ? For answered some of these questions, I start with SGD. SGD stands for Stochastic Gradient Descent that is the common optimization algorithm in deep learning.
 ### Gradient Descent Algorithm
 Consider the followng optimization problem:
 ```math
 \underset{\boldsymbol{\theta}}{min}\frac{1}{n}\sum_{i=1}^{n}f_{i}(\boldsymbol{\theta})
 ```
 As $ \nabla \sum_{i=1}^{n}f_{i}(\boldsymbol{\theta}) = \sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta})$,  gradient descent would repeat:
```math
\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t_{k}\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)}), \,\,\,\, k = 1,2,3,...
```
For solving this optimization problem, we can implement the gradient descent algorithm as follows:

> **Algorithm**:
>> input: initial guess $\boldsymbol{\theta}^{(0)}$, step size $t$ (let $t_k$ be constant for all $k$);
>>for $k =  1, 2, · · · $ do
$ \,\,\,\,\,\,\,\,\,\,\boldsymbol{\theta}^{(k)}=\boldsymbol{\theta}^{(k-1)}-t\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(\boldsymbol{\theta}^{(k-1)})$
end for
return $\boldsymbol{\theta}^{(k)}$ ;

Now with this optimization problem, think about ```epoch``` and ```lr``` in the mentioned Python code.
Now, with this optimization problem, think about the relation between ```k``` and ```t```  with ```epoch``` and ```lr```  in the aforementioned Python code.

Now, I want applied the GD algorithm to the above regression example. As we all know, we need to find $\hat{y}=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3 $ such that $L(\theta_0,\theta_1,\theta_2,\theta_3) =\frac{1}{n} \sum_{i=1}^n(\hat{y}_i-y_i)^2$ is minimized. It is easy to see that
```math
\frac{\partial L}{\partial\theta_{0}}=\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})\\
\frac{\partial L}{\partial\theta_{k}}=\frac{2}{n}\sum_{i=1}^{n}x_{ki}(\hat{y}_{i}-y_{i})=\frac{2}{n}X_k^T\times(\hat{y}-y)\qquad k=1,2,3
```
or in an equivalent formula
```math
\frac{\partial L}{\partial\boldsymbol{\theta}}=\frac{2}{n}X^T\times(\hat{y}-y)
```
Now let's convert GD algorithm into code. At first we need create a data set Once again:


useful links: [(1)](https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931), [(2)](https://matgomes.com/gradient-descent-for-linear-regression-in-python/)
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