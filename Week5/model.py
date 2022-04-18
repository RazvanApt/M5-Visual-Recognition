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
           
           

def l2norm(X):

    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def cosine_sim(im, s):

    return im.mm(s.t())


def order_sim(im, s):

    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(Module):


    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s, device):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        #print(scores[0])
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        mask = mask.to(device)
        #I = torch.tensor(mask)
        
        #if torch.cuda.is_available():
        #    I = I.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class TextEmbed(Module):

    def __init__(self, agg, img_feats, text_feats):
        super(TextEmbed, self).__init__()
        
        self.agg = agg
        self.img_feats = img_feats
        self.text_feats = text_feats
        
        if self.img_feats == "vgg":
            out_shape = 4096
        else:
            out_shape = 2048
            
        if self.text_feats == "bert":
            in_shape = 67584
        else:
            if self.agg == "concat":
                in_shape = 117000
            else:
                in_shape = int(117000/5)
                
        self.fc1 = Linear(in_shape, 4096)    
        self.fc2 = Linear(4096, out_shape)
        
        
        self.drop = Dropout(0.5)
        #self.relu = ReLU()
        #self.fc3 = Linear(1024, 512)
        
        #self.leakyrelu = LeakyReLU()
        
        
    
    def forward(self, x):
    
        x = l2norm(x) 
        
        x = self.fc1(x)
        x = self.drop(x)
        #x = self.relu(x) 
        #x = self.maxpool(x)
        #x = self.bn(x)
        x = self.fc2(x)
        #x = self.drop(x)
        x = l2norm(x)
        #x = self.relu(x)       
        #x = self.fc3(x)
        #x = self.relu(x)    

        return x
        
        
class ImgEmbed(Module):

    def __init__(self, img_feats):
        super(ImgEmbed, self).__init__()
        
        self.img_feats = img_feats
        
        if self.img_feats == "vgg":
            in_shape = 4096
        else:
            in_shape = 2048
        
        #self.fc1 = Linear(in_shape, 2048)
        #self.fc2 = Linear(2048, 512)
        #self.relu = ReLU()

    def forward(self, x):

        x = l2norm(x)
        #x = self.fc1(x)
        #x = self.relu(x)    
        #x = self.fc2(x)
        #x = self.relu(x)         
        #x = l2norm(x) 
        
        return x
        
        
class Model():

    def __init__(self, agg, device, img_feats="vgg", text_feats="fasttext"):
    
        self.agg = agg
        self.device = device
        self.img_feats = img_feats
        self.text_feats = text_feats
        
        self.img_embedder = ImgEmbed(self.img_feats)
        self.img_embedder.to(self.device)
        
        print(self.img_embedder)
        
        self.text_embedder = TextEmbed(self.agg, self.img_feats, self.text_feats)
        self.text_embedder.to(self.device)
        
        print(self.text_embedder)
        
    def train_mode(self):
    
        self.img_embedder.train()
        self.text_embedder.train()


    def eval_mode(self):
    
        self.img_embedder.eval()
        self.text_embedder.eval()
        
    def load_state_dict(self, state_dict_path):
        
        self.text_embedder.load_state_dict(torch.load(state_dict_path))
        
    
    def forward_emb(self, img_embed, text_embed):
    
        return self.img_embedder(img_embed), self.text_embedder(text_embed)