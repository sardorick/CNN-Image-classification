import torch.nn as nn
import torch.nn.functional as F 
import torch 
from model import CNN
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_batch
from torch import optim
from sklearn.metrics import accuracy_score




def fit(model, device, num_epoch, optimizer, criterion, train_loader):
    train_losses_list = []
    mean_acc = []
    for epoch in range(num_epoch):

        train_loss = []
        test_loss = []
        benchmark = 0.90
        acc_list = []
        for idx, (image, label) in enumerate(iter(train_loader)):

            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            predictions = model.forward(image)
            train_label = predictions.argmax(dim=1)
            loss = criterion(predictions, label)
            loss.backward()

            optimizer.step()
        
            train_loss.append(loss.item())
            accuracy = accuracy_score(label, train_label)
            acc_list.append(accuracy)
            # print(f"Accuracy for each epoch is: {accuracy}, test losses: {loss.item()}")

            print(f"Epoch: {epoch+1}/{num_epoch}, train losses: {loss.item()}")
        mean_accur_batch=sum(acc_list)/len(acc_list)
        print(f"Train acc {mean_accur_batch}")
        avg_loss = sum(train_loss)/len(train_loss)
        train_losses_list.append(avg_loss)
        mean_acc.append(mean_accur_batch)
    # for i in mean_acc:
    #     if benchmark < i:
    #         torch.save(model.state_dict(),'model_trained_test.pth')
    #         state_dict = torch.load('model_trained_test.pth')
    #         print(state_dict.keys())
    #         print(model.load_state_dict(state_dict))
    #         benchmark = i
    return model

def test(model, device, criterion, test_loader):
    test_acc = []
    model.eval()
    acc_list=[]
    with torch.no_grad():
        for idx, (image, label) in enumerate(iter(test_loader)):
            image, label = image.to(device), label.to(device)
            probability = F.softmax(model(image), dim=1)
            test_loss = criterion(probability, label)
            pred_label = probability.argmax(dim=1)
            accuracy = accuracy_score(label, pred_label)
            acc_list.append(accuracy)
            mean_accur_batch=sum(acc_list)/len(acc_list)
            print(f"Accuracy for each epoch is: {accuracy}, test losses: {test_loss.item()}")
        print(f"Test acc {mean_accur_batch}")
        test_acc.append(mean_accur_batch)

    model.train()
    return test_acc


        





"""
        if benchmark < mean_acc:
            torch.save(model.state_dict(),'model_trained.pth')
            state_dict = torch.load('model_trained.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = mean_acc
        model.train()
        """



    
