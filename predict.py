
import torch.nn as nn
import torch.nn.functional as F 
import torch 
from model import CNN
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_batch
from torch import optim
from sklearn.metrics import accuracy_score
from train import fit, test



model = CNN()
train_loader, test_loader, class_list, class_idx=load_batch('/Users/szokirov/Documents/Datasets/Intel')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epoch = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

trained_model = fit(model, device, num_epoch, optimizer, criterion, train_loader)

test_model = test(trained_model, device, criterion, test_loader)

