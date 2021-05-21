import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import matplotlib.pyplot as plt 

torch.manual_seed(1)

class Net(nn.Module):
    
    def __init__(self, n_features, n_hidden, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x





n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)     
y0 = torch.zeros(100)              
x1 = torch.normal(-2*n_data, 1)    
y1 = torch.ones(100)               
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  
y = torch.cat((y0, y1), ).type(torch.LongTensor)   


x, y = Variable(x), Variable(y)



net = Net(n_features=2, n_hidden=10, n_out=2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 10 == 0 or t in [3, 6]:
        # plot and show learning process
        plt.cla()
        _, prediction = torch.max(F.softmax(prediction), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.show()
        plt.pause(0.1)
plt.ioff()