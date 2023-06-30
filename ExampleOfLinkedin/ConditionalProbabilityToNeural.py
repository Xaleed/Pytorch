# In[1]
import torch 
import torch.nn as nn
# %%
class MyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3_mu = nn.Linear(64, output_dim)
        self.fc3_sigma = nn.Linear(64, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        sigma = torch.exp(self.fc3_sigma(x))
        return mu, sigma
def gaussian_log_likelihood(y, mu, sigma):
    return -0.5*torch.log(2*torch.pi*sigma**2)-(y-mu)**2/(2*sigma)
# %%
net = MyNetwork(input_dim = 10, output_dim=1)
optimizer = torch.optim.Adam(net.parameters())


# %%
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        input, targets = data

        # Zero the parameter gradients
        optimiser.zero_grad()
        # Forwar pass 
        mu, sigma = net(inputs)
        loss = -gaussian_log_likelihood(targets, mu, sigma).mean()

        # backward pass and optimization
        loss.backward()
        optimiser.step()
        running_loss +=loss.item()
    print(f"Epoch {epoch +1}, Loss: {running_loss / i}")
# %%
