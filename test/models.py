import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch.nn import Dropout, Sigmoid
from torch_geometric.nn import DynamicEdgeConv, fps, radius, SplineConv, TAGConv, TopKPooling, ChebConv, ARMAConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_cluster import knn


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
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


class TAGConvNet(torch.nn.Module):
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


class SplineConvNet(torch.nn.Module):
    def __init__(self, node_features, classes_num, K=3):
        super(SplineConvNet, self).__init__()
        self.conv1 = SplineConv(node_features, 128, 3, K)
        self.bn1 = BN(128)
        self.conv2 = SplineConv(128, 128, 3, K)
        self.bn2 = BN(128)
        self.lin1 = MLP([256, 256])
        self.final = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, classes_num),
        )

    def forward(self, data):
        x, edge_index, pseudo, batch = data.x.float(), data.edge_index, data.pseudo, data.batch
        x1 = F.relu(self.conv1(x, edge_index, pseudo))
        x1 = self.bn1(x1)
        x2 = F.relu(self.conv2(x1, edge_index, pseudo))
        x2 = self.bn2(x2)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = gmp(out, batch)
        out = self.final(out)
        return F.log_softmax(out, dim=-1)


class ARMAConvNet(torch.nn.Module):
    def __init__(self, node_features, classes_num, K=3):
        super(ARMAConvNet, self).__init__()
        self.conv1 = ARMAConv(node_features, 256, 3, K)
        # self.bn1 = BN(128)
        self.conv2 = ARMAConv(256, 256, 3, K)
        # self.bn2 = BN(128)

        self.lin1 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            # BN(512),
        )
        self.lin2 = torch.nn.Sequential(
            BN(256),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            BN(256),
            torch.nn.Dropout(0.5)
        )
        self.lin3 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            BN(256),
            torch.nn.Dropout(0.5),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(256, classes_num)
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x.float(), data.edge_index, data.batch, data.edge_attr.float()
        x1 = F.relu(self.conv1(x[:, :4], edge_index, edge_attr))

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))

        x = self.lin1(x2)

        x = gmp(x, batch)
        x = self.lin2(x)

        x = self.lin3(x)

        x = self.output(x)
        return F.log_softmax(x, dim=-1)
