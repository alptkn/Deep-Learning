import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader

net = nn.Sequential(
    nn.Linear(5,1),
    nn.ReLU(),
    nn.Linear(1, 1))

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = DataLoader(dataset=torch_dataset,batch_size=5, num_workers=2)

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        prediction = net(batch_x)
        loss = loss_func(prediction, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())