### dataset
import os
import os.path as osp
import shutil
import glob
import meshio
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from utilities3 import *


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

    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.name = name
        self.train = train
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
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        dx = 0.002
        base_depth = 50e-3

        data_list = []
        paths = glob.glob(osp.join(self.raw_dir, 'vtu', f'*.vtu'))
        node_features_list = []
        boundary_mask_list = []
        edge_index_list = []
        for i_path, path in enumerate(paths):
            if dataset == 'train':
                if i_path % 10 != 0:
                    node_features, edge_index = self.cell2graph(path, dx, base_depth)
                    # node_features_list.append(node_features)
                    edge_index_list.append(edge_index)
                    print("Processed file: {}".format(path))
                    boundary_mask = self.find_boundary_cells(node_features, edge_index)
                    # boundary_mask_list.append(boundary_mask)
                    # self.plot_graph(node_features[:, :3], edge_index,boundary_mask)
                    # y_output = node_features[~boundary_mask,3]
                    y_output = node_features[:, 3]
                    x_input = node_features.detach().clone()
                    # mean_temp = torch.mean(x_input[boundary_mask,3])
                    x_input[~boundary_mask, 3] = 0
                    num_nodes = node_features.shape[0]
                    print("num_nodes: {}".format(num_nodes))
                    print("y_output: {}".format(y_output.shape))
                    edge_index = self.indirect_edges(edge_index)
                    data = Data(x_input=x_input, edge_index=edge_index, y_output=y_output, boundary_mask=boundary_mask,
                                num_nodes=num_nodes)
                    data_list.append(data)
            else:
                if i_path % 10 == 0:
                    node_features, edge_index = self.cell2graph(path, dx, base_depth)
                    # node_features_list.append(node_features)
                    edge_index_list.append(edge_index)
                    print("Processed file: {}".format(path))
                    boundary_mask = self.find_boundary_cells(node_features, edge_index)
                    # boundary_mask_list.append(boundary_mask)
                    # self.plot_graph(node_features[:, :3], edge_index,boundary_mask)
                    # y_output = node_features[~boundary_mask,3]
                    y_output = node_features[:, 3]
                    x_input = node_features.detach().clone()
                    # mean_temp = torch.mean(x_input[boundary_mask,3])
                    x_input[~boundary_mask, 3] = 0
                    num_nodes = node_features.shape[0]
                    print("num_nodes: {}".format(num_nodes))
                    print("y_output: {}".format(y_output.shape))
                    edge_index = self.indirect_edges(edge_index)
                    data = Data(x_input=x_input, edge_index=edge_index, y_output=y_output, boundary_mask=boundary_mask,
                                num_nodes=num_nodes)
                    data_list.append(data)

        # node_features_arr = torch.cat((node_features_list),dim=0)
        # if self.train:
        # self.normalizer = UnitGaussianNormalizer(node_features_arr)
        # for i_data in range(len(node_features_list)):
        # node_features = node_features_list[i_data]
        # edge_index = edge_index_list[i_data]
        # #node_features = self.normalizer.encode(node_features)
        # boundary_mask = boundary_mask_list[i_data]
        # y_output = node_features[~boundary_mask,3]
        # x_input = node_features.detach().clone()
        # mean_temp = torch.mean(x_input[boundary_mask,3])
        # x_input[~boundary_mask,3] = mean_temp
        # num_nodes = node_features.shape[0]
        # print("num_nodes: {}".format(num_nodes))
        # print("y_output: {}".format(y_output.shape))
        # data = Data(x_input=x_input, edge_index=edge_index, y_output=y_output, boundary_mask= boundary_mask,num_nodes=num_nodes)
        # data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # if self.normalize is not None:
        #    pass

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

    def cell2graph(self, path, dx, base_depth):
        # load file
        mesh = meshio.read(path)

        # extract points and cells
        points = mesh.points
        points /= 1e3
        points = torch.from_numpy(points.astype(np.float32))

        sol_center = torch.tensor(mesh.cell_data['T'][0].astype(np.float32))
        sol_center = torch.unsqueeze(sol_center, dim=1)

        cells = mesh.cells_dict['hexahedron'].astype(int)

        # the cells with non-zero temperature are activated cells
        centeroids = torch.mean(points[cells], dim=1)
        active_mask = (sol_center[:, 0] > 0.1) & (centeroids[:, 2] > 0)  # remove base
        sol_center = sol_center[active_mask]
        cells = cells[active_mask]
        centeroids = centeroids[active_mask]

        # convert centeroids to inds (float coordinates to integer coordinates)
        inds = self._coord2inds(centeroids, dx)
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

        node_features = torch.cat((centeroids, sol_center), dim=1)
        return node_features, edges_total

    def indirect_edges(self, edges):
        edges_swap = edges.detach().clone()
        edges_swap[0, :] = edges[1, :]
        edges_swap[1, :] = edges[0, :]
        edges_total = torch.cat((edges, edges_swap), dim=1)
        return edges_total

    def find_boundary_cells(self, nodes, edges):
        """
        Return a boundary mask (True for boundary cells)
        :return: boundary_mask
        """
        z_min = torch.min(nodes[:, 2])
        count = torch.tensor([(edges == i).sum().item() for i in range(len(nodes))])
        boundary_mask = (count != 6) * (nodes[:, 2] > z_min)
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

    def plot_graph(self, points, edge_index, boundary_mask):
        points = points.numpy()
        edge_index = edge_index.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        p1s = points[edge_index[0, :]]
        ax.plot(p1s[:, 0], p1s[:, 1], p1s[:, 2], '.r', markersize=1)
        p2s = points[edge_index[1, :]]
        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=0.5, colors='b')
        # ax.add_collection(lc)
        bound_points = points[boundary_mask]
        ax.plot(bound_points[:, 0], bound_points[:, 1], bound_points[:, 2], '.g', markersize=5)

        plt.show()
