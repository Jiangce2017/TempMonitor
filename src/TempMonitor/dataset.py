### dataset
import os
import os.path as osp
import shutil
import glob
import meshio
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utilities3 import *
import time

class GraphDataset(InMemoryDataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, name='10', train=True,bottom_included=True,sliced_2d = False,bjorn=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.train = train
        self.bottom_included = bottom_included
        self.sliced_2d = sliced_2d
        self.bjorn = bjorn
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['vtu']  # define later
        pass

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']
        
    def process(self):
        if self.name == 'exp_wall':
            torch.save(self.process_exp_set('train'), self.processed_paths[0])
        else:
            torch.save(self.process_set('train'), self.processed_paths[0])
            torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        dx = 0.002
        base_depth = 50e-3
        data_list = []
        paths = glob.glob(osp.join(self.raw_dir,'vtk', f'*.vtu'))
        fem_file = pickle.load( open( osp.join(self.root, self.name+".p"), "rb" ) ) 
        print(np.abs(np.min(fem_file["vertices"][:,2])))
        ratio = np.abs(np.min(fem_file["vertices"][:,2]))/base_depth
        print(len(paths))
        for i_path, path in enumerate(paths):
            if ((dataset == 'train') and (i_path % 10 != 0)) or\
                ((dataset == 'test') and (i_path % 10 == 0)):

                print("Processing file: {}".format(path))    
                node_features, edge_index = self.cell2graph(path, dx, ratio, self.bottom_included, self.sliced_2d,self.bjorn)
                boundary_mask = self.find_boundary_cells(node_features,edge_index,self.bottom_included,self.sliced_2d)
                y_output = node_features[:,3]
                x_input = node_features.detach().clone()
                x_input[~boundary_mask,3] = 0
                num_nodes = node_features.shape[0]
                print("num_nodes: {}".format(num_nodes))
                print("y_output: {}".format(y_output.shape))
                print("num boud nodes: {}".format(node_features[boundary_mask].shape[0]))
                edge_index = self.indirect_edges(edge_index)
                data = Data(x_input=x_input, edge_index=edge_index, y_output=y_output, boundary_mask= boundary_mask,num_nodes=num_nodes)
                data_list.append(data)
                # figure_path = "figures/" + self.name+"graph_example.png"
                # self.plot_graph(data['x_input'][:,:3], data['edge_index'],data['boundary_mask'],figure_path)
                

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        #if self.normalize is not None:
        #    pass

        return self.collate(data_list)
    
    def process_exp_set(self, dataset):
        dx = 0.001
        ## create the whole coord graph
        ## Node 0 (-61.0, 0.0)
        ## Node 3334 (53.0, 28.0)

        ### load exp data
        misha_data_dir = os.path.join(Path.home(), 'data','misha')
        misha_data_path = os.path.join(misha_data_dir,'Wall 2', 'ProcessAndNodeData.tsv')
        df = pd.read_table(misha_data_path)
        headers = list(df.columns.values)
        #begin_ind = self.headers.index('Node 0 (-61.0, 0.0)')
        begin_ind = 26
        print(headers[begin_ind-1])
        print(headers[begin_ind])
        first_node = headers[begin_ind]
        ind_s = first_node.find('(')
        ind_m = first_node.find(',')
        ind_e = first_node.find(')')
        x_min = float(first_node[ind_s+1:ind_m])
        z_min =  float(first_node[ind_m+1:ind_e])
        
        last_node = headers[-1]
        print(last_node)
        ind_s = last_node.find('(')
        ind_m = last_node.find(',')
        ind_e = last_node.find(')')
        x_max = float(last_node[ind_s+1:ind_m])
        z_max =  float(last_node[ind_m+1:ind_e])
        
        nxel = int(x_max-x_min+1)
        nyel = int(z_max-z_min+1)
        
        temperatures = df.iloc[:,begin_ind:]
        nt = temperatures.shape[0]
        nel = temperatures.shape[1]
        
            
        temps = temperatures.to_numpy().astype(np.float32)

        
        temps = torch.from_numpy(temps)

        x_start = -61
        x_end = 53
        nodes = torch.zeros((nel,3))
        i_cur = 0
        for z_coord in range(nyel):
            nodes[i_cur:i_cur+nxel,0] = torch.arange(x_start,x_end+1)
            nodes[i_cur:i_cur+nxel,2] = z_coord
            i_cur += nxel
        ## create graphs from the nodes --> edges
        nodes /= 1e3
        edges_total = self.nodes2graph(nodes,dx)

        nodes[:,2]+=0.02

        # figure_path = "figures/" + "exp_wall"+"graph_example.png"
        # self.plot_graph(nodes, edges_total,np.zeros((nodes.shape[0]),dtype=bool),figure_path)


        current_list = df['Current (A)']
        laser_on = current_list > 0
        laser_on_frames = torch.arange(nt)[laser_on]

        #indexes = torch.randperm(laser_on_frames.shape[0])
        #laser_on_frames = laser_on_frames[indexes]
        data_list = []

        figure_path = "figures/" + "exp_wall"+"temp_frame_example.png"
        self.plot_frame(temps[laser_on_frames[1000]],nxel,nyel,figure_path)
        toolpath_file = "dataset/" + "toolpath.csv"
        toolpath = np.loadtxt(toolpath_file,delimiter=',')
        
        toolpath_idx = np.arange(toolpath.shape[0],dtype=int)
        end_i_frame = np.max(toolpath_idx[toolpath[:,3]<0.002*3])
        print("end i frame: {}".format(end_i_frame))
        for i_frame, frame in enumerate(laser_on_frames[:end_i_frame]):
            start = time.time()

            tool_z = toolpath[i_frame,3]+0.02
            act_mask = (temps[frame] > 50) * (nodes[:,2]<tool_z*1)*(temps[frame] < 1500)

            edg0_mask = act_mask[edges_total[0,:]]
            edg1_mask = act_mask[edges_total[1,:]]
            act_edge_mask = edg0_mask*edg1_mask

            edge_index = edges_total[:,act_edge_mask]

            num_nodes = nodes.shape[0]
            num_rest_nodes = nodes[act_mask].shape[0]
            reindex_nodes = torch.zeros(num_nodes,dtype=int)
            reindex_nodes[act_mask] = torch.arange(num_rest_nodes,dtype=int)

            edge_index= reindex_nodes[edge_index]

            node_features = torch.cat((nodes[act_mask], torch.unsqueeze( temps[frame][act_mask],dim=1)), dim=1)
            boundary_mask = self.find_boundary_cells(node_features,edge_index,self.bottom_included,self.sliced_2d)
            #boundary_mask = self.find_wall_boundary(node_features)
            y_output = node_features[:,3]
            x_input = node_features.detach().clone()
            x_input[~boundary_mask,3] = 0
            num_nodes = node_features.shape[0]
            #print("num_nodes: {}".format(num_nodes))
            #print("y_output: {}".format(y_output.shape))
            #print("num boud nodes: {}".format(node_features[boundary_mask].shape[0]))
            edge_index = self.indirect_edges(edge_index)
            data = Data(x_input=x_input, edge_index=edge_index, y_output=y_output, boundary_mask= boundary_mask,num_nodes=num_nodes)
            data_list.append(data)

            end = time.time()
            #print("time cost for one frame: {}".format(end-start))

            if i_frame % 10 == 0:     
                figure_path = "figures/" + "exp_wall"+"graph_example.png"
                self.plot_graph(data['x_input'][:,:3], data['edge_index'],data['boundary_mask'],figure_path)
                    


        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
    
    def plot_frame(self,temp,nxel,nyel,figure_path):
        temp = temp.numpy()
        temp = np.reshape(temp,(nyel,nxel))
        vmax = np.max(temp)
        vmin = np.min(temp)
        plt.imshow(temp[::-1,:],vmin=vmin,vmax=vmax, cmap='hot', interpolation='none')
        plt.colorbar()
        #plt.show()
        plt.savefig(figure_path)
    
    def nodes2graph(self, nodes, dx):
        # convert centeroids to inds (float coordinates to integer coordinates)
        inds = self._coord2inds(nodes, dx)
        max_inds = torch.max(inds, dim=0).values
        inds_hash = self._grid_hash(inds, max_inds)
        lookup_table = self._create_lookup_table(inds_hash, max_inds)

        origins = torch.arange(inds.shape[0], dtype=torch.int32)
        moves = torch.tensor([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]], dtype=torch.int32)
        edges = []
        for i_move in range(3):
            moved_inds = inds + moves[i_move]
            mask_in = (moved_inds[:, 0] >= 0) & (moved_inds[:, 0] <= max_inds[0]) & (moved_inds[:, 1] >= 0) & (
                        moved_inds[:, 1] <= max_inds[1]) & \
                      (moved_inds[:, 2] >= 0) & (moved_inds[:, 2] <= max_inds[2])
            moved_inds_hash = self._grid_hash(moved_inds[mask_in], max_inds)
            end0 = origins[mask_in]
            end1 = lookup_table[moved_inds_hash]
            mask_act = end1 >= 0
            edg = torch.stack([end0[mask_act], end1[mask_act]], dim=0)
            # print(edg.shape)
            edges.append(edg)
        edges_total = torch.cat(edges, dim=1)

        return edges_total
    
    def find_wall_boundary(self,node_features):
        pos = node_features[:,:3]
        pos_min = torch.min(node_features,dim=0).values
        pos_max = torch.max(node_features,dim=0).values
        left_bound = pos[:,0] == pos_min[0]
        right_bound = pos[:,0] == pos_max[0]
        upper_bound = pos[:,2] == pos_max[2]
        lower_bound = pos[:,2] == pos_max[2]
        bound_mask = left_bound+right_bound+upper_bound+lower_bound
        return bound_mask

    def cell2graph(self, path, dx, ratio, bottom_included, sliced_2d=False,bjorn=True):
        # load file
        mesh = meshio.read(path)
        points = mesh.points ## points coordinates
        

        
        if bjorn:
            cells = mesh.cells_dict['hexahedron'].astype(int)
            sol_center = torch.tensor(mesh.cell_data['T'][0].astype(np.float32))
            sol_center = torch.unsqueeze(sol_center, dim=1)
            points /= 1e3
        else:
            cells = mesh.cells_dict['hexahedron']
            sol = mesh.point_data['sol']
            sol_center = torch.tensor(np.mean(sol[cells],axis=1))

        # the cells with non-zero temperature are activated cells
        dx = np.min(np.max(np.abs(points[:-1]-points[1:]),axis=1))
        print(dx)
        print(ratio)
        points = torch.from_numpy(points.astype(np.float32))
        points[points[:,2]<0,2] = points[points[:,2]<0,2]*ratio

        centeroids = torch.mean(points[cells], dim=1)
        if bottom_included:
            active_mask = (sol_center[:, 0] > 0.1) #& (centeroids[:, 2] > 0)  # remove base     
        else:
            if self.name == 'wall':
                z_bottom = 0.02
            else:
                z_bottom = 0
            active_mask = (sol_center[:, 0] > 0.1) & (centeroids[:, 2] >= z_bottom)  # remove base

        if sliced_2d:
            y_coord = 0
            active_mask *= centeroids[:,1] == y_coord

        sol_center = sol_center[active_mask]
        cells = cells[active_mask]
        centeroids = centeroids[active_mask]
        


        # convert centeroids to inds (float coordinates to integer coordinates)
        inds = self._coord2inds(centeroids, dx)
        if self.name == 'wall':
            coord_mask = inds % 2 == 0
            point_mask = coord_mask[:,0]*coord_mask[:,1]*coord_mask[:,2]
            centeroids = centeroids[point_mask]
            sol_center = sol_center[point_mask]
            inds = inds[point_mask]
            cells = cells[point_mask]

        print(inds.shape)
        max_inds = torch.max(inds, dim=0).values
        inds_hash = self._grid_hash(inds, max_inds)
        lookup_table = self._create_lookup_table(inds_hash, max_inds)

        origins = torch.arange(inds.shape[0], dtype=torch.int32)
        
        if self.name == 'wall':
            step_size = 2
        else:
            step_size = 1
        moves = torch.tensor([[step_size, 0, 0],
                              [0, step_size, 0],
                              [0, 0, step_size]], dtype=torch.int32)
        edges = []
        for i_move in range(3):
            moved_inds = inds + moves[i_move]
            mask_in = (moved_inds[:, 0] >= 0) & (moved_inds[:, 0] <= max_inds[0]) & (moved_inds[:, 1] >= 0) & (
                        moved_inds[:, 1] <= max_inds[1]) & \
                      (moved_inds[:, 2] >= 0) & (moved_inds[:, 2] <= max_inds[2])
            moved_inds_hash = self._grid_hash(moved_inds[mask_in], max_inds)
            end0 = origins[mask_in]
            end1 = lookup_table[moved_inds_hash]
            mask_act = end1 >= 0
            edg = torch.stack([end0[mask_act], end1[mask_act]], dim=0)
            # print(edg.shape)
            edges.append(edg)
        edges_total = torch.cat(edges, dim=1)

        node_features = torch.cat((centeroids, sol_center), dim=1)
        return node_features, edges_total

    def indirect_edges(self,edges):
        edges_swap = edges.detach().clone()
        edges_swap[0,:] = edges[1,:]
        edges_swap[1,:] = edges[0,:]
        edges_total = torch.cat((edges,edges_swap),dim=1)
        return edges_total

    def find_boundary_cells(self, nodes, edges,bottom_included=True,sliced_2d=False):
        """
        Return a boundary mask (True for boundary cells)
        :param data: pytorch_geometric
        :return: boundary_mask
        """
        z_min = torch.min(nodes[:,2])
        if sliced_2d:
            key_number = 4
        else:
            key_number = 6
        count = torch.tensor([(edges == i).sum().item() for i in range(len(nodes))])
        #print("max count: {}".format( torch.max(count)))
       #print("min count: {}".format(torch.min(count)))
        if bottom_included:
            boundary_mask = (count != key_number)
        else:
            boundary_mask = (count != key_number) * (nodes[:,2] >= z_min)
        return boundary_mask
    
      

    def _coord2inds(self, points, d):
        coord_min = torch.min(points, dim=0, keepdim=True).values
        inds = torch.round((points - coord_min) / d).int()
        return inds

    def _grid_hash(self, arr, max_inds):
        int_hash = arr[:, 0] + arr[:, 1] * (max_inds[0] + 1) + arr[:, 2] * (max_inds[0] + 1) * (max_inds[1] + 1)
        return int_hash.long()

    def _create_lookup_table(self, arr_hash, max_inds):
        lookup_table = -1 * torch.ones(((max_inds[0] + 1) * (max_inds[1] + 1) * (max_inds[2] + 1)), dtype=torch.int64)
        lookup_table[arr_hash] = torch.arange(arr_hash.shape[0])
        return lookup_table

    def plot_graph(self, points, edge_index,boundary_mask,figure_path):
        points = points.numpy()
        edge_index = edge_index.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p1s = points[edge_index[0, :]]
        #ax.plot(p1s[:, 0], p1s[:, 1], p1s[:, 2], '.r', markersize=1)

        p2s = points[edge_index[1, :]]
        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors='b')       
        #ax.add_collection(lc)
        inside_points = points[~boundary_mask]
        ax.plot(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2], '.r', markersize=1)
        bound_points = points[boundary_mask]
        ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], '.g', markersize=1)
        plt.savefig(figure_path)
        #plt.show()
