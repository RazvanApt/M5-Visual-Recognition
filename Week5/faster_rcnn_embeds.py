from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils import visualizer as vis
import numpy as np
import cv2
from PIL import Image
import random
import shutil
import os
from torch import nn
import torch
import json
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
 
 

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

model.fc = nn.Identity()
model.to(device)
model.eval()


pth = "/export/home/mcv/datasets/Flickr30k/"

modes = ["train", "val", "test"]

for mode in modes:

    feat_list = []
    with open(os.path.join(pth, mode + ".json")) as f:
        labels = json.load(f)
        
    for i, inst in enumerate(labels):

        img = read_image(os.path.join("flickr30k-images", inst["filename"])).to(device).float()
        res = model(img[None, ...])

        feat_list.append(res.to("cpu").detach().numpy().flatten())

            
    print("Done {}".format(mode))
    print(np.stack(feat_list).shape)
    
    with open(mode + "_faster_feats.npy", "wb") as f:
        np.save(f, np.stack(feat_list))
    