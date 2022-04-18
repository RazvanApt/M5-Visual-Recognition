import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import *
from torch.optim import *

from dataset import FlickrDataset
from tqdm import tqdm
import copy
import torch.nn.functional as F
import numpy as np
import itertools
import pickle
from utils import train, test
from model import Model, ContrastiveLoss
          
agg = "sum" 
img_feats = "vgg"
text_feats = "bert"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
number_of_epoch = 50

model = Model(agg, device, img_feats, text_feats)

pth = "/export/home/mcv/datasets/Flickr30k/"

train_data = FlickrDataset(pth, "train", agg, img_feats, text_feats)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

val_data = FlickrDataset(pth, "val", agg, img_feats, text_feats)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)    


params = list(model.text_embedder.parameters())
params += list(model.img_embedder.parameters())

optimizer = Adam(params, lr=0.0002)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    
#distance = LpDistance()
#reducer = reducers.ThresholdReducer(low=0)
#loss_func = losses.TripletMarginLoss(distance=distance, reducer=reducer)
#mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets=comb[2])

precs_train = {"i2t_prec_1": [],
          "i2t_prec_5": [],
          "t2i_prec_1": [],
          "t2i_prec_5": []}
          
precs_val = {"i2t_prec_1": [],
          "i2t_prec_5": [],
          "t2i_prec_1": [],
          "t2i_prec_5": []}
          
          
loss_func = ContrastiveLoss(margin=0.1, max_violation=True)
for epoch in range(1, number_of_epoch + 1):
    train(model, loss_func, device, train_loader, val_loader, optimizer, epoch)
    scheduler.step()
     
    if epoch%5 == 0:
    
        print("Train Results")
        res_train = test(model, train_loader, device)
        precs_train["i2t_prec_1"].append(res_train[0])
        precs_train["i2t_prec_5"].append(res_train[1])
        precs_train["t2i_prec_1"].append(res_train[2])
        precs_train["t2i_prec_5"].append(res_train[3])
        print(precs_train)
        
        print("Val Results")
        res_val = test(model, val_loader, device)
        precs_val["i2t_prec_1"].append(res_val[0])
        precs_val["i2t_prec_5"].append(res_val[1])
        precs_val["t2i_prec_1"].append(res_val[2])
        precs_val["t2i_prec_5"].append(res_val[3])
        
        print(precs_val)
        if len(precs_val["i2t_prec_5"]) > 1:
            if res_val[1] > max(precs_val["i2t_prec_5"][:-1]):
                print("Saving best weights")
                torch.save(model.text_embedder.state_dict(), "best_text_embedder.pth")
                
        with open("train_res.pkl", "wb") as f:
            pickle.dump(precs_train, f)
            
        with open("val_res.pkl", "wb") as f:
            pickle.dump(precs_val, f)