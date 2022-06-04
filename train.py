import torch.nn as nn
import torch.nn.functional as F 
import torch 
from model import CNN
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_batch
from torch import optim



model = CNN()
train_loader, test_loader, class_list, class_idx=load_batch('/Users/szokirov/Documents/Datasets/Intel')

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epoch = 1
train_losses_list = []
test_losses_list = []
test_acc = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)

for epoch in range(num_epoch):
    train_loss = []
    test_loss = []
    benchmark = 0.90
    for idx, (image, label) in enumerate(iter(train_loader)):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model.forward(image)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print(f"Epoch: {epoch+1}/{num_epoch}, train losses: {loss.item()}")
    avg_loss = sum(train_loss)/len(train_loss)
    train_losses_list.append(avg_loss)

    model.eval()
    acc_list=[]
    with torch.no_grad():
        for idx, (image, label) in enumerate(iter(test_loader)):
            image, label = image.to(device), label.to(device)
            probability = F.softmax(model(image), dim=1)
            test_loss = criterion(probability, label)
            pred_label = torch.argmax(probability, dim=1)
            accuracy=sum(pred_label==label)/pred_label.shape[0]
            acc_list.append(accuracy)
            mean_accur_batch=sum(acc_list)/len(acc_list)
        print(f"Mean accuracy is: {accuracy}")
        test_acc.append(mean_accur_batch)
        





"""
        if benchmark < mean_acc:
            torch.save(model.state_dict(),'model_trained.pth')
            state_dict = torch.load('model_trained.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = mean_acc
        model.train()
        """



    
