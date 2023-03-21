### test TempMonitor
import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from TempMonitor import GraphDataset

if __name__ == '__main__':
    data_path = osp.join('dataset')
    dataset = GraphDataset(data_path, '10', True)