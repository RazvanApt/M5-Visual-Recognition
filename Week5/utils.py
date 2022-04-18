from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import *
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
import torch
import numpy as np
import faiss


def train(model, loss_func, device, train_loader, val_loader, optimizer, epoch):

    model.train_mode()
    running_loss = 0

    for batch_idx, (img_embeds, text_embeds, _, _, _) in enumerate(train_loader):
    
        img_embeddings, text_embeddings = img_embeds.to(device), text_embeds.to(device)
        optimizer.zero_grad()
        
        img_embeddings, text_embeddings = model.forward_emb(img_embeddings, text_embeddings)
        anc, pos, neg = ([] for _ in range(3))
        
        for i in range(len(img_embeds)):
            for j in range(len(img_embeds)):
                if i != j:
                  anc.append(i)
                  pos.append(i)
                  neg.append(j)
                 
        indices_tuple = (torch.tensor(anc), torch.tensor(pos), torch.tensor(neg))
        loss = loss_func(img_embeddings, text_embeddings, device)

        #loss = loss_func(img_embeddings, torch.zeros(img_embeddings.shape[0]), indices_tuple, text_embeddings, torch.zeros(img_embeddings.shape[0]))

        running_loss += loss
        loss.backward()
        optimizer.step()

    print("Epoch {}: Training Loss = {}".format(
                epoch, running_loss))
    
    model.eval_mode()
    running_loss = 0
    with torch.no_grad():
    
        for batch_idx, (img_embeds, text_embeds, _, _, _)  in enumerate(val_loader):
            
            img_embeddings, text_embeddings = img_embeds.to(device), text_embeds.to(device)
            
            img_embeddings, text_embeddings = model.forward_emb(img_embeddings, text_embeddings)
            anc, pos, neg = ([] for _ in range(3))
            
            for i in range(len(img_embeds)):
                for j in range(len(img_embeds)):
                    if i != j:
                      anc.append(i)
                      pos.append(i)
                      neg.append(j)
                     
            indices_tuple = (torch.tensor(anc), torch.tensor(pos), torch.tensor(neg))
            loss = loss_func(img_embeddings, text_embeddings, device)
            
            #loss = loss_func(img_embeddings, torch.zeros(img_embeddings.shape[0]), indices_tuple, text_embeddings, torch.zeros(img_embeddings.shape[0]))
            running_loss += loss
    
    
        print("Epoch {}: Val Loss = {}".format(
                        epoch, running_loss))

    
def test(model, loader, device):

    img_data = []
    text_data = []
    
    
    model.eval_mode()
    with torch.no_grad():
    
        for (img_embeds, text_embeds, _, _, _) in loader:
        
            img_embeddings, text_embeddings = img_embeds.to(device), text_embeds.to(device)
            img_embeddings, text_embeddings = model.forward_emb(img_embeddings, text_embeddings)
            
            if isinstance(img_data, list):
                img_data = img_embeddings
            else:
                img_data = torch.cat((img_data, img_embeddings))
                
            if isinstance(text_data, list):
                text_data = text_embeddings
            else:
                text_data = torch.cat((text_data, text_embeddings))
                
                
    img_dists = img_data.mm(text_data.t())
    text_dists = text_data.mm(img_data.t())
    
    print("Image-to-Text Retrieval")
    prec_1 = 0
    prec_5 = 0
    
    for i, k in enumerate(img_dists):

        dist, indices = torch.sort(k, 0, descending=True)
        if i == indices[0]:
            prec_1 += 1
        if i in indices[:5]:
            prec_5 += 1
        
    i2t_prec_1 = round(prec_1/text_data.shape[0], 3)
    i2t_prec_5 = round(prec_5/text_data.shape[0], 3)
    print("Precision@1: {}, Precision@5: {}".format(i2t_prec_1, i2t_prec_5))

    
    print("Text-to-Image Retrieval")
    prec_1 = 0
    prec_5 = 0
    
    for i, k in enumerate(text_dists):

        dist, indices = torch.sort(k, 0, descending=True)
        if i == indices[0]:
            prec_1 += 1
        if i in indices[:5]:
            prec_5 += 1
     
    t2i_prec_1 = round(prec_1/text_data.shape[0], 3)
    t2i_prec_5 = round(prec_5/text_data.shape[0], 3)   
    print("Precision@1: {}, Precision@5: {}".format(t2i_prec_1, t2i_prec_5))
    
    return i2t_prec_1, i2t_prec_5, t2i_prec_1, t2i_prec_5
    
    #print(dists[0])
    
    """
    img_data = img_data.to("cpu")
    text_data = text_data.to("cpu")
    
    flat_index = faiss.IndexFlatL2(512)
    flat_index.add(img_data)         

    D, I = flat_index.search(text_data, 5)
    
    #print(img_data[:10])
    #print(text_data[:10])
    print(I[:10])
    print(D[:10])

    acc = 0
    for i, val in enumerate(I):
        if val[0] == i:
            acc += 1
    print("Accuracy: ", round(acc/text_data.shape[0], 3))
    """ 


def get_preds(res):

    preds = []
    for pred in res[1][0].to("cpu"):
        for key in labels_to_indices.keys():
            if pred.item() in labels_to_indices[key]:
              preds.append((img_paths[pred.item()], class_map[key]))

    return preds


def get_pred_classes(res):

    pred_classes = []
    for pred in res[1][0].to("cpu"):
        for key in labels_to_indices.keys():
            if pred.item() in labels_to_indices[key]:
              pred_classes.append(key)

    return pred_classes


def compute_metrics(labels, all_preds, k=1):

    metrics_per_class = {}

    for cls in set(labels):

      precs = []
      total_inst = len(labels[labels == cls])
      start_in = np.where(labels==cls)[0].min()
      stop_in = np.where(labels==cls)[0].max()

      for i in range(start_in, stop_in):
          preds_at_k = all_preds[i][:k]
          
          pos_preds = len([pred for pred in preds_at_k if pred == labels[i]])
          precs.append(pos_preds/k)

      metrics_per_class[cls] = np.mean(precs)

    return metrics_per_class
    


def get_pred_classes(res, labels_to_indices):

    pred_classes = []
    for pred in res[1][0].to("cpu"):
        for key in labels_to_indices.keys():
            if pred.item() in labels_to_indices[key]:
              pred_classes.append(key)

    return pred_classes
    
    
    
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)
    
