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
num_epoch = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


train_losses_list = []
mean_acc = []
test_losses_list = []
test_acc = []

for epoch in range(num_epoch):
    test_accuracy, train_accuracy, running_loss_test = 0, 0, 0

    train_loss = []
    benchmark = 0.90
    acc_list = []
    for idx, (image, label) in enumerate(iter(train_loader)):

        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model.forward(image)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    avg_loss = sum(train_loss)/len(train_loss)
    print(f"Train loss after epoch {epoch+1}: {avg_loss:.4f}")
    train_losses_list.append(avg_loss)

    model.eval()
    with torch.no_grad():
        for idx, (image, label) in enumerate(iter(test_loader)):
            image, label = image.to(device), label.to(device)
            probability = model.forward(image)
            test_loss = criterion(probability, label)
            running_loss_test += test_loss.item()
            pred_label = F.softmax(probability, dim=1).argmax(dim=1)
            test_accuracy += accuracy_score(label, pred_label)
        mean_accur_epoch_test=test_accuracy/len(test_loader) 
        acc_list.append(mean_accur_epoch_test)

            
        print(f"Test acc {mean_accur_epoch_test}, Test losses: {test_loss:.4f}")
        test_acc.append(mean_accur_epoch_test)
        avg_loss_test = running_loss_test/len(test_loader)
        test_losses_list.append(avg_loss_test)
        if benchmark < mean_accur_epoch_test:
            torch.save(model.state_dict(),'model_tested_2.pth')
            benchmark = mean_accur_epoch_test



    model.train()
    print(f'{epoch+1:3}/{num_epoch} :  | Accuracy : {mean_accur_epoch_test:.4f}')


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



    
