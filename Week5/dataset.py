import os
import numpy as np
import scipy.io
import json 
import cv2
import random
from torch import nn
import torch

from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
from torch.utils.data.dataset import Dataset



class FlickrDataset(Dataset):

    def __init__(self, path, mode, agg, img_pre="vgg", text_pre="fasttext"):
    
      self.pth = path
      self.agg = agg
      self.mode = mode
      self.img_pre = img_pre
      self.text_pre = text_pre
      
      with open(os.path.join(self.pth, self.mode + ".json")) as f:
          self.labels = json.load(f)
          
      img_ids = [lbl["imgid"] for lbl in self.labels]
      
      
      if img_pre == "vgg":

          img_embeds = scipy.io.loadmat(os.path.join(self.pth, "vgg_feats.mat"))["feats"]
          self.img_embeds = torch.tensor(np.swapaxes(img_embeds, 0, 1), dtype=torch.float32)[img_ids]
          
      else:
      
          self.img_embeds = np.load(self.mode + "_faster_feats.npy", allow_pickle=True)
          self.img_embeds = torch.tensor(self.img_embeds, dtype=torch.float32)
          
      if self.text_pre == "fasttext":
          text_embeds = np.load(os.path.join(self.pth, "fasttext_feats.npy"), allow_pickle=True)
          
          max_lengths = {}
          for level in range(0, 5):
            max_s = 0
            for elem in text_embeds:
                if elem[level].shape[0] > max_s:
                    max_s = elem[level].shape[0]
                
            max_lengths[level] = max_s
            
          for level in range(0, 5):
              max_length = max(max_lengths.values())
              for elem in text_embeds:
                  if elem[level].shape[0] < max_length:
                      elem[level] = np.concatenate((elem[level], np.zeros((max_length - elem[level].shape[0], 300))))
            
          fin_arr = []
          for i in range(text_embeds.shape[0]):
              if self.agg == "concat": 
                  fin_arr.append(np.stack(text_embeds[i]))
              elif self.agg == "random":
                  rnd_idx = random.randint(0, 4)
                  fin_arr.append(text_embeds[i][rnd_idx])
              elif self.agg == "avg":
                  fin_arr.append(np.mean(np.stack(text_embeds[i]), axis=0))
              elif self.agg == "sum":
                  fin_arr.append(np.sum(np.stack(text_embeds[i]), axis=0))
              else:
                  print("Aggregation Not Known!")
                        
          text_embeds = np.reshape(np.stack(fin_arr), (31014, -1))[img_ids]
          self.text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
            
      else:
  
          max_len = 0
          for emd in os.listdir("bert_embeds"):
              embedding = np.load("bert_embeds/" + emd, allow_pickle=True)
              if embedding.shape[1] > max_len:
                  max_len = embedding.shape[1]
          
          text_embeds = np.load("bert_embeds/" + self.mode + "_sum_bert_embeds.npy", allow_pickle=True)
          text_embeds = np.concatenate((text_embeds, np.zeros((text_embeds.shape[0], max_len-text_embeds.shape[1]))), axis=1)
          
          self.text_embeds = torch.tensor(text_embeds, dtype=torch.float32)
    
      
    def __getitem__(self, index):
        sents = []
        for sent in self.labels[index]["sentences"]:
            sents.append(sent["raw"])

        return self.img_embeds[index], self.text_embeds[index], self.labels[index]["imgid"], sents, \
               self.labels[index]["filename"] 

    def __len__(self):
    
        return len(self.labels)
    