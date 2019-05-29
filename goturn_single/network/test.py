import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,groups=2,padding=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,padding=1,groups=2)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,padding=1,groups=2)
        self.fc6_new = nn.Linear(in_features=6*6*256*2, out_features=4096)
        self.fc7_new = nn.Linear(in_features=4096, out_features=4096)
        self.fc7_newb = nn.Linear(in_features=4096, out_features=4096)
        self.fc8_shapes = nn.Linear(in_features=4096, out_features=4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.lrn = nn.LocalResponseNorm(5)

    def conv(self,x):
        x = self.lrn(self.pool(F.relu(self.conv1(x))))
        x = self.lrn(self.pool(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(1,-1,x.shape[2],x.shape[3])
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 6*6*2*256)
        x = F.relu(self.fc6_new(x))
        x = F.relu(self.fc7_new(x))
        x = F.relu(self.fc7_newb(x))
        x = self.fc8_shapes(x)
        x = x.numpy()
        print(x)
        print(x.shape)

net = AlexNet()
alex_dict = net.state_dict()
#print(net.state_dict().keys())
pre = torch.load('../../nets/tracker.pt')
new = {k.replace('-','_'):v for k, v in pre.items() if '_p' not in k}
#print(new.keys())
alex_dict.update(new)
net.load_state_dict(alex_dict)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
net.to(device)
net.double()
imgs = np.random.rand(2,3,227,227)
print(imgs.shape)
imgs = torch.from_numpy(imgs)
#imgs = imgs.type('torch.DoubleTensor')
imgs.to(device)
net.eval()
with torch.no_grad():
    net(imgs)
