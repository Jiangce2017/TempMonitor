### test TempMonitor
import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from TempMonitor import GraphDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

if __name__ == '__main__':
    data_path = osp.join('dataset')
    geo_name = 'hollow_1'
    train_dataset = GraphDataset(root=osp.join(data_path,geo_name), name=geo_name,train=True, \
                                     bottom_included=True,sliced_2d=False,bjorn=True)
    it = iter(train_dataset)
    data0 = train_dataset[-1]
    print(data0["x_input"][:,:3])
    points = data0["x_input"][:,:3].numpy()
    dx = np.min(np.max(np.abs(points[:-1]-points[1:]),axis=1))
    print(dx)
    #figure_path = "figures/" + geo_name +"graph_example.png"
    #train_dataset.plot_graph(data0['x_input'][:,:3], data0['edge_index'],data0['boundary_mask'],figure_path)
    
    points = data0['x_input'][:,:3]
    edge_index = data0['edge_index']
    boundary_mask = data0['boundary_mask']
    points = points.numpy()
    edge_index = edge_index.numpy()


    for azim in range(0,360,10):
        fig = plt.figure(figsize=(32,32))
        ax = fig.add_subplot(projection='3d')

        p1s = points[edge_index[0, :]]
        ax.plot(p1s[:, 0], p1s[:, 1], p1s[:, 2], '.r', markersize=1)

        p2s = points[edge_index[1, :]]
        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.2, colors='b')       
        ax.add_collection(lc)                              
        # inside_points = points[~boundary_mask]
        # ax.plot(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], '.m', markersize=1)
        # bound_points = points[boundary_mask]
        # ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], '.c', markersize=1)
        ax.set_xlim(-0.04,0.04)
        ax.set_ylim(-0.04,0.04)
        ax.set_zlim(-0.02,0.02)
        ax.view_init(elev=30, azim=azim, roll=0)
        #ax.axis('scaled')
        plt.axis('off')
        figure_path = "figures/gifs/graph/" + str(azim) +"_graph_example.png"
        plt.savefig(figure_path)
        plt.close()

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
    