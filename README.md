# PyTorch
## Table of contents
* [Motivation](#Motivation)
* [Data](#Data)
* [Linear Regression Model](#Linear-Regression-Model)
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
Well, I think many questions come for beginners after going through these steps, for example, what is SGD, lr, epoch and ... ? For answered some of these questions, I start with SGD.