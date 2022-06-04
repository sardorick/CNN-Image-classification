import torch.nn as nn
import torch.nn.functional as F 
import torch 

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d()
        self.batch2 = nn.BatchNorm1d()
        self.batch3 = nn.BatchNorm1d()
        # self.batch2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128*28*28, 200)
        self.fc2 = nn.Linear(200, 64)
        self.fc3 = nn.Linear(64, 6)

        
    def forward(self, x):
        
        out = F.relu(F.max_pool2d(self.batch1(self.conv1(x),2)))
        out = F.relu(F.max_pool2d(self.batch2(self.conv2(out),2)))
        out = F.relu(F.max_pool2d(self.batch3(self.conv3(out),2)))
        out = out.view(-1, 128*28*28)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

