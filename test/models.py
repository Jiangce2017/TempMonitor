import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch.nn import Dropout, Sigmoid
from torch_geometric.nn import DynamicEdgeConv, SplineConv, TAGConv, ARMAConv, GINConv, SGConv
from torch_geometric.nn import global_mean_pool, global_max_pool as gmp, global_add_pool
from torch_cluster import knn


def MLP(channels, batch_norm=True):
    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), BN(channels[i]), nn.ReLU(),)
            for i in range(1, len(channels))
        ])
    else:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), )
            for i in range(1, len(channels))
        ])


class MixConv(torch.nn.Module):
    def __init__(self, k=32, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = MLP([128 + 64, 512])

        # self.mlp = Seq(
        # MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        # Lin(256, 10))

        self.conv_gcnn1 = TAGConv(4, 64, 3)
        self.conv_gcnn2 = TAGConv(64, 128, 3)
        self.lin_gcnn1 = MLP([128 + 64, 512])

        self.mix_output = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, 1))

        # self.mlp_gcnn = Seq(
        # MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
        # Lin(256, 10))

        # self.mix_output = Seq(
        # Lin(20,10),
        # ReLU()
        # )

    def forward(self, data):
        pos, batch = data.x_input[:, :3], data.batch
        edge_index, node_features = data.edge_index, data.x_input
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out1 = self.lin1(torch.cat([x1, x2], dim=1))
        # out1 = global_max_pool(out1, batch)
        # out1 = self.mlp(out1)

        w1 = F.relu(self.conv_gcnn1(node_features[:4], edge_index))
        w2 = F.relu(self.conv_gcnn2(w1, edge_index))
        out2 = self.lin_gcnn1(torch.cat([w1, w2], dim=1))
        # out2 = global_max_pool(out2, batch)
        # out2 = self.mlp_gcnn(out2)

        out = self.mix_output(torch.cat([out1, out2], dim=1))

        return out


class TAGConvNet(nn.Module):
    def __init__(self, node_features, K=3, num_filters=128, dropout=0.5):
        super(TAGConvNet, self).__init__()

        self.lin0 = Seq(
            nn.Linear(node_features, num_filters),
            nn.ReLU(),
            # BN(512),
        )

        self.conv1 = TAGConv(num_filters, num_filters, K)
        self.conv2 = TAGConv(num_filters, num_filters, K)

        self.lin1 = Seq(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            # nn.Dropout(dropout),
            # BN(512),
        )

        self.lin2 = Seq(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            # nn.Dropout(dropout)
            # BN(512),
        )

        self.lin3 = Seq(
            nn.Linear(num_filters, 1),
            nn.ReLU(),
            # BN(512),
        )

    def forward(self, data):
        x, edge_index, batch = data.x_input, data.edge_index, data.batch
        x = self.lin0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x


class ARMAConvNet(nn.Module):
    def __init__(self, node_features, num_filters=128, K=3, dropout=.5):
        super(ARMAConvNet, self).__init__()

        # self.lin0 = nn.Sequential(
        #     nn.Linear(node_features, num_filters),
        #     nn.ReLU(),
        # )

        self.conv1 = ARMAConv(node_features, num_filters, 3, K)
        # self.bn1 = BN(128)
        self.conv2 = ARMAConv(num_filters, num_filters, 3, K)
        # self.bn2 = BN(128)

        self.lin1 = nn.Sequential(
            torch.nn.Linear(num_filters, num_filters),
            torch.nn.ReLU(),
            # BN(512),
        )

        self.output = nn.Sequential(
            torch.nn.Linear(num_filters, 1)
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x_input.float(), data.edge_index, data.batch, data.edge_attr
        # x = self.lin0(x)
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x = self.lin1(x2)
        x = self.output(x)
        return F.relu(x)


class GINConvNet(nn.Module):
    def __init__(self, node_features, hidden_dim=128, eps=0., train_eps=False, dropout=.5):
        super(GINConvNet, self).__init__()
        self.conv1 = GINConv(
            MLP([node_features, hidden_dim, hidden_dim], batch_norm=True), eps, train_eps
        )
        self.conv2 = GINConv(
            MLP([hidden_dim, hidden_dim, hidden_dim], batch_norm=True), eps, train_eps
        )

        self.conv3 = GINConv(
            MLP([hidden_dim, hidden_dim, hidden_dim], batch_norm=True), eps, train_eps
        )

        self.lin1 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x_input, data.edge_index, data.batch
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        # x1 = global_add_pool(x1, batch)
        # x2 = global_add_pool(x2, batch)
        # x3 = global_add_pool(x3, batch)

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        return F.relu(self.output(x))


class SGConvNet(nn.Module):
    def __init__(self, node_features, K=3, hidden_dim=128, dropout=0.5):
        super(SGConvNet, self).__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
        )

        self.conv1 = SGConv(hidden_dim, hidden_dim, K)
        self.conv2 = SGConv(hidden_dim, hidden_dim, K)

        self.lin1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # BN(hidden_dim),
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # BN(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )

        self.lin3 = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x_input, data.edge_index, data.batch
        x = self.lin0(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return F.relu(x)
