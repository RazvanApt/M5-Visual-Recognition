import pandas as pd
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd
import shutil
import cv2
import numpy as np
import time
import umap
from utils import *

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
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


out_path = "trip_inf"
os.makedirs(out_path, exist_ok=True)


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


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

model.fc = c_f.Identity()
model.to(device)

optimizer = Adam(model.parameters(), lr=0.00008)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


distance = DotProductSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")


for epoch in range(1, number_of_epoch + 1):
    train(model, loss_func, mining_func, device, train_loader, val_loader, optimizer, epoch)
    scheduler.step()
    if epoch%5 == 0:
        print("Train Prec.")
        test(training_data, training_data, model, device, labels_to_indices)
        print("Val Prec.")
        test(training_data, val_data, model, device, labels_to_indices)

comb_res = test(training_data, val_data, model, device, labels_to_indices)


with open(os.path.join(out_path, "trip_inf.pkl"), "wb") as f:
    pickle.dump(comb_res, f)
    
    

d = 512

tester = testers.BaseTester(data_device=device)

train_embeddings, train_labels = tester.get_all_embeddings(training_data, model)
test_embeddings, test_labels = tester.get_all_embeddings(val_data, model)
train_labels = train_labels.squeeze(1).to("cpu").numpy()
test_labels = test_labels.squeeze(1).to("cpu").numpy()

train_embeddings = train_embeddings.to("cpu").numpy()
test_embeddings = test_embeddings.to("cpu").numpy()


reducer = umap.UMAP()
embedding = reducer.fit_transform(train_embeddings)


df = pd.DataFrame()
df["y"] = [class_map[k] for k in train_labels]
df['umap-one'] = embedding[:,0]
df['umap-two'] = embedding[:,1] 

plt.figure(figsize=(16,10))
rndperm = np.random.permutation(df.shape[0])
sns.scatterplot(
    x="umap-one", y="umap-two",
    hue="y",
    palette=sns.color_palette("hls", 8),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.7
)
plt.title("UMAP", fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.legend(title="", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join(out_path, "UMAP.png"))

    
pca = PCA(n_components=3)
pca_result = pca.fit_transform(train_embeddings)

df = pd.DataFrame()
df["y"] = [class_map[k] for k in train_labels]
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]

plt.figure(figsize=(16,10))
rndperm = np.random.permutation(df.shape[0])
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 8),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.7
)
plt.title("PCA", fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.legend(title="", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.savefig(os.path.join(out_path, "PCA.png"))



tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(train_embeddings)

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 8),
    data=df,
    legend="full",
    alpha=0.7)

plt.title("t-SNE", fontsize=20)
plt.xlabel("")
plt.ylabel("")
plt.legend(title="", fontsize=14)
plt.xticks([])
plt.yticks([])

plt.savefig(os.path.join(out_path, "t-SNE.png"))