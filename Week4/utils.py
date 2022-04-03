from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import *
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
import torch
import numpy as np
import faiss


def train(model, loss_func, mining_func, device, train_loader, val_loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        running_loss += loss
        loss.backward()
        optimizer.step()

    print("Epoch {}: Training Loss = {}".format(
                epoch, running_loss
            )
        )
            
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(val_loader):
            data, labels = data.to(device), labels.to(device)
    
            embeddings = model(data)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)
            running_loss += loss
    
    
        print("Epoch {}: Val Loss = {}".format(
                        epoch, running_loss
                    )
                )


def test(train_set, test_set, model, device, labels_to_indices):

    tester = testers.BaseTester(data_device=device)
    
    train_embeddings, train_labels = tester.get_all_embeddings(train_set, model)
    test_embeddings, test_labels = tester.get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1).to("cpu").numpy()
    test_labels = test_labels.squeeze(1).to("cpu").numpy()
    
    train_embeddings = train_embeddings.to("cpu").numpy()
    test_embeddings = test_embeddings.to("cpu").numpy()
    
    flat_index = faiss.IndexFlatL2(512)

    flat_index.add(train_embeddings)         

    
    D, I = flat_index.search(test_embeddings, 10)

    all_preds = []
    for val in I:
        pred_classes = []
        for pred in val:
            for key in labels_to_indices.keys():
                if pred.item() in labels_to_indices[key]:
                  pred_classes.append(key)
        all_preds.append(pred_classes)
    
    metric_dict= {}
    for k in [1, 5, 10]:
    
        metrics = compute_metrics(test_labels, np.array(all_preds), k)
        print(k, np.mean([v for k,v in metrics.items()]))
        metric_dict[k] = metrics

    return (D, I), metric_dict


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
    
