import pandas as pd
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd
import shutil
import cv2
import numpy as np
from utils import *
import pickle

import faiss
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import *
from torch.optim import *
from tqdm import tqdm
import copy
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import *
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pytorch_metric_learning.utils.inference import CustomKNN

    
    
train_data_dir="/export/home/mcv/datasets/MIT_split/train"
val_data_dir="/export/home/mcv/datasets/MIT_split/test"

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

img_width=256
img_height=256
batch_size=64

    
transform = transforms.Compose([transforms.Resize(img_width),
                                 transforms.ToTensor()])

training_data = ImageFolder(train_data_dir, transform)
val_data = ImageFolder(val_data_dir, transform)

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

class_map = {v: k for k, v in training_data.class_to_idx.items()}

img_paths = [img[0] for img in training_data.imgs]

pre_resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

labels_to_indices = c_f.get_labels_to_indices(training_data.targets)
pre_resnet.fc = c_f.Identity()
pre_resnet.to(device);


d = 512

tester = testers.BaseTester(data_device=device)

train_embeddings, train_labels = tester.get_all_embeddings(training_data, pre_resnet)
test_embeddings, test_labels = tester.get_all_embeddings(val_data, pre_resnet)
train_labels = train_labels.squeeze(1).to("cpu").numpy()
test_labels = test_labels.squeeze(1).to("cpu").numpy()

train_embeddings = train_embeddings.to("cpu").numpy()
test_embeddings = test_embeddings.to("cpu").numpy()

#faiss.normalize_L2(train_embeddings)
#faiss.normalize_L2(test_embeddings)


flat_index = faiss.IndexFlatL2(d)   # build the index
#gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
#print(index.is_trained)
flat_index.add(train_embeddings)                  # add vectors to the index
#print(index.ntotal)

D, I = flat_index.search(test_embeddings, 10)

print(D[0], I[0])
#print(training_data.targets, I)
metric_dict = {}

all_preds = []
for val in I:
    pred_classes = []
    for pred in val:
        for key in labels_to_indices.keys():
            if pred.item() in labels_to_indices[key]:
              pred_classes.append(key)
    all_preds.append(pred_classes)
print(all_preds[0])

for k in [1, 5, 10]:

    metrics = compute_metrics(test_labels, np.array(all_preds), k)
    metric_dict[str(k)] = metrics
    print(k, np.mean([v for k,v in metrics.items()]))
    
res = (D, I), metric_dict

with open("pre_res.pkl", "wb") as f:
    pickle.dump(res, f)
