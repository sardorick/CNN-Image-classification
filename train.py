import torch.nn as nn
import torch.nn.functional as F 
import torch 
from model import CNN
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_batch
from torch import optim
from sklearn.metrics import accuracy_score

model = CNN()
train_loader, test_loader, class_list, class_idx=load_batch('/Users/szokirov/Documents/Datasets/Intel')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


train_losses_list = []
mean_acc = []
test_losses_list = []
test_acc = []

for epoch in range(num_epoch):
    test_accuracy, train_accuracy = 0, 0

    train_loss = []
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
        train_accuracy += accuracy_score(label, train_label)

    mean_accur_batch_train=train_accuracy/len(train_loader)
    print(f"Train acc {mean_accur_batch_train}")
    avg_loss = sum(train_loss)/len(train_loss)
    train_losses_list.append(avg_loss)
    mean_acc.append(mean_accur_batch_train)
    for i in mean_acc:
        if benchmark < i:
            torch.save(model.state_dict(),'model_trained_2.pth')
            state_dict = torch.load('model_trained_2.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = i

    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(iter(test_loader)):
            image, label = image.to(device), label.to(device)
            probability = F.softmax(model(image), dim=1)
            test_loss = criterion(probability, label)
            pred_label = probability.argmax(dim=1)
            test_accuracy += accuracy_score(label, pred_label)
            acc_list.append(test_accuracy)
        mean_accur_batch_test=test_accuracy/len(test_loader)

        print(f"Test acc {mean_accur_batch_test}")
        test_acc.append(mean_accur_batch_test)
        # avg_loss_test = sum(test_loss)/len(test_loss)
        # test_losses_list.append(avg_loss_test)


    for i in test_acc:
        if benchmark < i:
            torch.save(model.state_dict(),'model_tested_2.pth')
            state_dict = torch.load('model_tested_2.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = i

    model.train()
    print(f'{epoch+1:3}/{num_epoch} :  | Accuracy : {mean_accur_batch_test:.4f}')


torch.save({"model_state": model.state_dict()}, 'saved_final_model')

plt.plot(train_losses_list, label='Training Losses')
plt.plot(test_losses_list,  label='Testing Lossess')
plt.plot(acc_list, label='Accuracy')
plt.show()


        





"""
        if benchmark < mean_acc:
            torch.save(model.state_dict(),'model_trained.pth')
            state_dict = torch.load('model_trained.pth')
            print(state_dict.keys())
            print(model.load_state_dict(state_dict))
            benchmark = mean_acc
        model.train()
        """



    
