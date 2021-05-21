import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,5,padding=2), nn.ReLU(), nn.MaxPool2d(2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,5,padding=2),nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32*7*7,10)
    
    def forward(self,X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(X.size(0),-1)
        output = self.out(X)
        return output

def graph():
    plt.plot(losses_his, label='loss')
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 1))
    plt.show()

train_data = datasets.MNIST(root='../Data', train=True, download=False,transform=transforms.ToTensor() )
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
test_data = datasets.MNIST(root='../Data', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()

losses_his = []

cnn = CNN()
cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoc in range(1):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        b_y = Variable(y).cuda()
        output = cnn(b_x)
        loss = loss_func(output,b_y)
        losses_his.append(loss.data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', 1, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)


graph()

#-----Testing------#
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze() # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

