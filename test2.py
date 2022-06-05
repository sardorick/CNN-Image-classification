from ast import arg
from tkinter import font
from PIL import Image
from torchvision import transforms 
import torch  
from model import CNN
import matplotlib.pyplot as plt 


train_model = torch.load('model_tested_2.pth')

model_state = train_model

model = CNN()
model.load_state_dict(model_state)

def pred(image_path, model):
     preprocessor = transforms.Compose([transforms.Resize((150, 150)),
                                        transforms.CenterCrop(140),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])


     classes_dict = {
         0: 'buildings',
         1: 'forest',
         2: 'glacier',
         3: 'mountain',
         4: 'sea',
         5: 'street'
     }

     image = Image.open(image_path)
     processed_image = preprocessor(image)
     processed_image = processed_image.view(1, *processed_image.shape)

     model.eval()
     with torch.no_grad():
          prediction = model.forward(processed_image)
          _, prediction_class = torch.max(prediction, dim=1)

     plt.imshow(image)
     plt.title(f'Prediction class:  {prediction_class.item()}    ({classes_dict[prediction_class.item()]})', fontsize=15)
     plt.show()

     return
    
image_path = '/Users/szokirov/Documents/Datasets/pred_intel/seg_pred/seg_pred/117.jpg'
pred(image_path, model)