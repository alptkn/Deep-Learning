import torch 
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt 



class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    
    
torch.manual_seed(1)
x = torch.unsqueeze(torch.linspace(-1, 1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

net = Net(n_features=1, n_hidden=30, n_out=1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

plt.ion()

for t in range(200):
    prediction = net.forward(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 10 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r--', lw=6)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.show()
        plt.pause(0.1)

plt.ioff()



