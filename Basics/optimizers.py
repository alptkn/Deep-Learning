import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt 



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.out = nn.Linear(10,1)
    
    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = self.out(X)
        return X

LR = 0.01

torch.manual_seed(1)
x = torch.unsqueeze(torch.linspace(-1,1,1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(x.size()))

torch_dataset = Data.TensorDataset(x,y)
loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

net_SGD         = Net()
net_Momentum    = Net()
net_RMSprop     = Net()
net_Adam        = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = nn.MSELoss()
losses_his = [[], [], [], []]

for epoch in range(12):
    print('Epoch: ', epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)
        
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            prediction = net(b_x)
            loss = loss_func(prediction, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss)

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()



