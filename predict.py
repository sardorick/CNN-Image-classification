
import torch.nn as nn
import torch 
from model import CNN
import matplotlib.pyplot as plt
from data_loader import load_batch
from torch import optim
from train import fit, test



model = CNN()
train_loader, test_loader, class_list, class_idx=load_batch('/Users/szokirov/Documents/Datasets/Intel')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

trained_model = fit(model, device, num_epoch, optimizer, criterion, train_loader)

test_model = test(trained_model, device, criterion, test_loader)

