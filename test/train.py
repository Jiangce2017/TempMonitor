#### train the model
import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch_geometric.nn import radius, TAGConv, global_max_pool as gmp, knn
from TempMonitor import GraphDataset

#from point_cloud_models import BallConvNet, DynamicEdge, MixConv
#from models import ARMAConvNet

def train():
    model.train()
    train_metrics = {"loss": [], "acc": []}
    for batch_i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        predictions = model(data)
        loss = F.nll_loss(predictions, data.y)
        loss.backward()
        optimizer.step()
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        train_metrics["loss"].append(loss.item())
        train_metrics["acc"].append(acc)
    return np.mean(train_metrics["acc"]), np.mean(train_metrics["loss"])
        
def test():
    model.eval()
    test_metrics = {"acc": []}
    correct = 0
    for batch_i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            predictions = model(data)
        acc = 100 * (predictions.detach().argmax(1) == data.y).cpu().numpy().mean()
        test_metrics["acc"].append(acc)
    return np.mean(test_metrics["acc"])
    

if __name__ == '__main__':
    data_path = osp.join('dataset')
    #pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    dataset = GraphDataset(data_path, '10', True)
    


