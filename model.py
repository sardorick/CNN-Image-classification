import torch.nn as nn
import torch.nn.functional as F 
import torch 
from data_loader import load_batch



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm1d(num_features=256)
        self.batch2 = nn.BatchNorm1d(num_features=128)
        self.batch3 = nn.BatchNorm1d(num_features=6)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 6)

        
    def forward(self, x):
        
        out = F.relu(F.max_pool2d(self.conv1(x),2))
        out = F.relu(F.max_pool2d(self.conv2(out),2))
        out = F.relu(F.max_pool2d(self.conv3(out),2))
        out = F.relu(F.max_pool2d(self.conv4(out),2))
        out = out.view(-1, 64*8*8)
        out = F.relu(self.batch1(self.fc1(out)))
        out = F.relu(self.batch2(self.fc2(out)))
        out = self.batch3(self.fc3(out))
        return out

# train_loader, test_loader, class_list, class_idx=load_batch('/Users/szokirov/Documents/Datasets/Intel')
# img, label = next(iter(train_loader))
# model = CNN()
# print(model.forward(img))

