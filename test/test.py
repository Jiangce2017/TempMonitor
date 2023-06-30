### test TempMonitor
import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from TempMonitor import GraphDataset

if __name__ == '__main__':
    data_path = osp.join('dataset')
    geo_name = 'townhouse_7'
    train_dataset = GraphDataset(root=osp.join(data_path,geo_name), name=geo_name,train=True, \
                                     bottom_included=True,sliced_2d=False,bjorn=True)
    it = iter(train_dataset)
    data0 = next(it)
    print(data0["x_input"][:,:3])
    points = data0["x_input"][:,:3].numpy()
    dx = np.min(np.max(np.abs(points[:-1]-points[1:]),axis=1))
    print(dx)
    figure_path = "figures/" + geo_name +"graph_example.png"
    train_dataset.plot_graph(data0['x_input'][:,:3], data0['edge_index'],data0['boundary_mask'],figure_path)
    

    # simu_wall_dataset = GraphDataset(root=osp.join(data_path,'wall'), name='wall',train=True, \
    #                                  bottom_included=False,sliced_2d=True,bjorn=False)
    # it = iter(simu_wall_dataset)
    # data0 = next(it)
    # print(data0["x_input"][:,:3])
    # points = data0["x_input"][:,:3].numpy()
    # dx = np.min(np.max(np.abs(points[:-1]-points[1:]),axis=1))
    # print(dx)
    # figure_path = "figures/" + "simu_wall"+"graph_example.png"
    # simu_wall_dataset.plot_graph(data0['x_input'][:,:3], data0['edge_index'],data0['boundary_mask'],figure_path)
    
    # exp_wall_dataset = GraphDataset(root=osp.join(data_path,'exp_wall'),name= 'exp_wall',train=True, \
    #                                 bottom_included=False,sliced_2d=True,bjorn=False)
    # it = iter(exp_wall_dataset)
    # data0 = next(it)
    # figure_path = "figures/" + "exp_wall"+"graph_example.png"
    # exp_wall_dataset.plot_graph(data0['x_input'][:,:3], data0['edge_index'],data0['boundary_mask'],figure_path)
    