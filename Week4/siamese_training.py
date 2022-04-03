import os, datetime
from utils import *

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import *
from torch.optim import *
from tqdm import tqdm
import copy
import torch.nn.functional as F
import numpy as np

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import *
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder,FaissKNN , CustomKNN
import faiss
import itertools

import pickle


train_data_dir="/export/home/mcv/datasets/MIT_split/train"
val_data_dir="/export/home/mcv/datasets/MIT_split/test"

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

img_width=256
img_height=256
batch_size=64
number_of_epoch=20

transform = transforms.Compose([transforms.Resize(img_width),
                                 transforms.ToTensor()])

training_data = ImageFolder(train_data_dir, transform)
val_data = ImageFolder(val_data_dir, transform)

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

class_map = {v: k for k, v in training_data.class_to_idx.items()}

img_paths = [img[0] for img in training_data.imgs]


labels_to_indices = c_f.get_labels_to_indices(training_data.targets)

distance_names = {
          "dot_product": DotProductSimilarity(),
          "lp": LpDistance(),
          "SNRDistance": SNRDistance()
}

loss_names = {
          "Contrastive": losses.ContrastiveLoss,
          "NCALoss": losses.NCALoss,
}


combs = itertools.product(list(distance_names.keys()), list(loss_names.keys()))

res_dict = {}
for comb in combs:

    print(comb)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    
    model.fc = c_f.Identity()
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=0.00008)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    
    distance = distance_names[comb[0]]
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = loss_names[comb[1]](distance=distance, reducer=reducer)
    mining_func = miners.PairMarginMiner(distance=distance, )

    
    for epoch in range(1, number_of_epoch + 1):
        train(model, loss_func, mining_func, device, train_loader, val_loader, optimizer, epoch)
        scheduler.step()
        if epoch%5 == 0:
            print("Train Prec.")
            test(training_data, training_data, model, device, labels_to_indices)
            print("Val Prec.")
            test(training_data, val_data, model, device, labels_to_indices)

    comb_res = test(training_data, val_data, model, device, labels_to_indices)
    
    res_dict[comb] = comb_res
    
    with open("siam_fine_res.pkl", "wb") as f:
        pickle.dump(res_dict, f)