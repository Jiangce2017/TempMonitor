### dataset
import os
import os.path as osp
import shutil
import glob
import meshio

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.data import Data


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
        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['vtu']

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def process(self):

        paths = glob.glob(osp.join(self.raw_dir, f'*.vtu'))
        data_list = []
        
        for path in paths:
            node_features, edge_index = self.mesh2graph(path)
            print("Processed file: {}".format(path))
            data = Data(x=node_features, edge_index=edge_index)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))

    def face2edge(self, faces):
        """
        Create edge index tensor from hexahedron faces
        :param faces: a tensor of hexahedron faces (N*8)
        :return: a tensor of edges (N*2)
        """
        edges = torch.cat([torch.stack(
            [torch.tensor([face[0], face[1]]), torch.tensor([face[0], face[3]]), torch.tensor([face[0], face[4]]),
             torch.tensor([face[1], face[0]]), torch.tensor([face[1], face[2]]), torch.tensor([face[1], face[5]]),
             torch.tensor([face[2], face[1]]), torch.tensor([face[2], face[3]]), torch.tensor([face[2], face[6]]),
             torch.tensor([face[3], face[0]]), torch.tensor([face[3], face[2]]), torch.tensor([face[3], face[7]]),
             torch.tensor([face[4], face[0]]), torch.tensor([face[4], face[5]]), torch.tensor([face[4], face[7]]),
             torch.tensor([face[5], face[1]]), torch.tensor([face[5], face[4]]), torch.tensor([face[5], face[6]]),
             torch.tensor([face[6], face[2]]), torch.tensor([face[6], face[5]]), torch.tensor([face[6], face[7]]),
             torch.tensor([face[7], face[3]]), torch.tensor([face[7], face[4]]), torch.tensor([face[7], face[6]])])
            for face in faces])     # 24 edges for a hexahedron cell
        # an edge may belong to several cells
        edges = torch.unique(edges, sorted=False, dim=0)
        return edges

    def mesh2graph(self, path):
        """
        Convert the .vtu mesh file to a torch graph instance
        :param path: mesh file '*.vtu'
        :return: node_features, edge_index
        """
        # load file
        mesh = meshio.read(path)

        # Extract vertices and faces from mesh
        vertices = mesh.points
        vertices /= 1e3
        faces = mesh.cells_dict['hexahedron'].astype(int)

        # Create edge index tensor from faces
        faces = torch.tensor(faces, dtype=torch.int32).contiguous()
        edges = self.face2edge(faces)
        edge_index = edges.transpose(0, 1).contiguous()

        node_features = torch.tensor(vertices, dtype=torch.float)
        # edge_index = edge_index.long()
        return node_features, edge_index

    def find_boundary_points(self, data):
        ####
        pass
