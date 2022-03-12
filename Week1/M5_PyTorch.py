
import pandas as pd
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd
import shutil
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import *
from torch.optim import *
from tqdm import tqdm
import copy
import torch.nn.functional as F
from torchsummary import summary
import wandb


train_data_dir='datasets/MIT_small_train_3/test'
val_data_dir='datasets/MIT_small_train_1/test'
test_data_dir='datasets/MIT_small_train_2/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
wandb.init(project="pytorch-intro")

img_width=256
img_height=256
batch_size=64
number_of_epoch=100
num_filters=32

train_transform = transforms.Compose([transforms.Resize(img_width),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])

val_transform = transforms.Compose([transforms.Resize(img_width),
                                 transforms.ToTensor()])

training_data = ImageFolder(train_data_dir, transform=train_transform)
val_data = ImageFolder(val_data_dir, val_transform)
test_data = ImageFolder(test_data_dir, val_transform)

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

class G6_Model(Module):   
    def __init__(self):
        super(G6_Model, self).__init__()

        self.G6_Model = Sequential(
        self.ConvBlock(3, num_filters, kernel_size=3, stride=2),
        self.DepthwiseSeperableConvBlock(num_filters, num_filters, stride=1),

        self.ConvBlock(num_filters, num_filters*2, kernel_size=1, stride=1),
        self.DepthwiseSeperableConvBlock(num_filters*2, num_filters*2, stride=2),

        self.ConvBlock(num_filters*2, num_filters*4, kernel_size=1, stride=1),
        self.DepthwiseSeperableConvBlock(num_filters*4, num_filters*4, stride=1),

        self.ConvBlock(num_filters*4, num_filters*8, kernel_size=1, stride=1),
        self.DepthwiseSeperableConvBlock(num_filters*8, num_filters*8, stride=2),

        AvgPool2d(8),
        Flatten(),
        Linear(num_filters*8, num_filters*8),
        ReLU(),
        Linear(num_filters*8, 8)
        )

    def ConvBlock(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):

      return Sequential(
          Conv2d(input_channels, output_channels, kernel_size, stride, padding),
          ReLU(inplace=True),
          BatchNorm2d(output_channels)
        )

    def DepthwiseSeperableConvBlock(self, input_channels, output_channels, stride=1, padding=1):

      return Sequential(
          Conv2d(input_channels, input_channels, kernel_size=3, stride=stride, padding=padding, groups=input_channels),
          ReLU(inplace=True),
          BatchNorm2d(input_channels),
          Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=padding),
          ReLU(inplace=True),
          BatchNorm2d(output_channels)
        )

    def forward(self, x):

        x = self.G6_Model(x)
        #x = Linear(input_channels, output_channels)(x)
        #x =  ReLU(inplace=True)(x)

        return x

mdl = G6_Model()
mdl.to(device)

criterion = CrossEntropyLoss()

optimizer = Adam(mdl.parameters(), lr=0.0008)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

wandb.watch(mdl)


best_model_wts = copy.deepcopy(mdl.state_dict())

def train_or_evaluate(phase, mdl, optimizer, criterion):
        
      if phase == 'train':
          mdl.train()  
          loader = train_loader
      else:
          mdl.eval()   
          loader = val_loader

      running_loss = 0.0
      running_corrects = 0


      for inputs, labels in loader:
          inputs = inputs.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == 'train'):

              outputs = mdl(inputs)
              loss = criterion(outputs, labels)
              
              _, preds = torch.max(outputs, 1)

              if phase == 'train':
                  loss.backward()
                  optimizer.step()           

          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss/len(loader.dataset)
      epoch_acc = running_corrects.double()/len(loader.dataset)
      


      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
      return epoch_loss, epoch_acc
  

val_acc_history = []
train_acc_history = []
best_acc = 0.0

for epoch in range(1, number_of_epoch+1):

    print("Epoch: {}".format(epoch))
    
    train_loss, train_acc = train_or_evaluate("train", mdl, optimizer, criterion)
    val_loss, val_acc = train_or_evaluate("val", mdl, optimizer, criterion)
    
    wandb.log({
              "Epoch": epoch,
              "Train_Loss": train_loss,
              "Train_Acc": train_acc,
              "Val_Loss": val_loss,
              "Val_Acc": val_acc
    })
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(mdl.state_dict())

    val_acc_history.append(val_acc.item())
    train_acc_history.append(train_acc.item())    
	
	scheduler.step()
    
 
df_res = pd.DataFrame()
df_res["val_acc"] = val_acc_history
df_res["train_acc"] = train_acc_history

df_res.to_csv("torch_results.csv", index=False)
 