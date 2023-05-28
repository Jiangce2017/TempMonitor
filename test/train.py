### train the model
import os
import os.path as osp
import torch
import numpy as np
from numpy import linalg as LA
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch_geometric.nn import radius, TAGConv, global_max_pool as gmp, knn
from src.TempMonitor import GraphDataset
import csv
from torch.utils.tensorboard import SummaryWriter

# from point_cloud_models import BallConvNet, DynamicEdge, MixConv
from models import TAGConvNet, ARMAConvNet
import warnings

warnings.filterwarnings("ignore")


def r2loss(pred, y):
    # print(pred.shape,y.shape)
    SS_res = torch.sum((pred - y) ** 2)
    y_mean = torch.mean(y)
    SS_tot = torch.sum((y - y_mean) ** 2)
    r2 = 1 - SS_res / SS_tot
    return r2


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def train(optimizer=None, lr_scheduler=None):
    model.train()
    train_metrics = {"mse_loss": [], "r2_loss": []}
    for batch_i, data in enumerate(train_loader):
        if optimizer:
            optimizer.zero_grad()
        data = data.to(device)
        predictions = model(data)
        predictions = predictions[~data.boundary_mask].view(-1)
        true_temp = data.y_output[~data.boundary_mask]
        mse_loss = F.mse_loss(predictions, true_temp, reduction='mean')
        mse_loss.backward()
        if optimizer:
            optimizer.step()
        r2_loss = r2loss(predictions, true_temp)
        # print(mse_loss.item())
        train_metrics["mse_loss"].append(mse_loss.item())
        train_metrics["r2_loss"].append(r2_loss.item())

    if lr_scheduler:
        lr_scheduler.step()
    return np.mean(train_metrics["mse_loss"]), np.mean(train_metrics["r2_loss"])


def test():
    model.eval()
    test_metrics = {"mse_loss": [], "r2_loss": []}
    # correct = 0
    for batch_i, data in enumerate(test_loader):
        data = data.to(device)
        with torch.no_grad():
            predictions = model(data)
            predictions = predictions[~data.boundary_mask].view(-1)
            true_temp = data.y_output[~data.boundary_mask]
            mse_loss = F.mse_loss(predictions, true_temp, reduction='mean')
            r2_loss = r2loss(predictions, true_temp)
        test_metrics["mse_loss"].append(mse_loss.item())
        test_metrics["r2_loss"].append(r2_loss.item())
    return np.mean(test_metrics["mse_loss"]), np.mean(test_metrics["r2_loss"])


if __name__ == '__main__':

    # hyper parameters
    num_hops = 10
    dropout = .2
    bz = 32  # training batchsize
    lr = 5e-3   # learning rate for Adam
    # scheduler
    step_size = 100
    gamma = .5      # Step LR
    t_max = 500     # Cos LR

    dataset_name = 'Temp_Monitor'
    data_path = osp.join('dataset')
    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = GraphDataset(data_path, '10', True)
    # normalizer = train_dataset.normalizer
    test_dataset = GraphDataset(data_path, '10', False)

    print(len(train_dataset))
    print(len(test_dataset))
    print('Dataset loaded.')

    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True, drop_last=True,
                              num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=True, drop_last=True,
                             num_workers=2)

    # model = TAGConvNet(4, K=num_hops, dropout=dropout)
    model = ARMAConvNet(4, dropout=dropout)
    model_name = 'Deep_ARMAConvNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-4)

    exp_name = dataset_name + '_' + model_name
    print(exp_name)
    result_path = osp.join('.', 'results')

    # train_logger = Logger(
    #     osp.join(result_path, exp_name + '_train.log'),
    #     ['ep', 'train_mse', 'train_r2', 'test_mse', 'test_r2']
    # )

    # training logs
    if not osp.exists(result_path):
        os.mkdir(result_path)
    last_train = max([eval(s.split("-")[-1]) for s in os.listdir(result_path)] + [0])
    current_train = last_train + 1
    save_dir = osp.join(result_path, exp_name + "-{}".format(current_train))
    os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)

    cur_ep = 0
    path_model = osp.join(save_dir, exp_name + '_checkpoint.pth')
    # if osp.isfile(path_model):
    #     checkpoint = torch.load(path_model)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     cur_ep = checkpoint['epoch']

    num_epochs = 200
    best_test_r2 = 0
    for epoch in range(cur_ep, cur_ep + num_epochs):
        # train_mse_loss, train_r2_loss = train(optimizer=optimizer)
        train_mse_loss, train_r2_loss = train(optimizer=optimizer, lr_scheduler=scheduler)
        test_mse_loss, test_r2_loss = test()
        is_best = test_r2_loss > best_test_r2
        best_test_r2 = max(best_test_r2, test_r2_loss)
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_test_r2': best_test_r2
        }
        if is_best:
            print('Saving...')
            torch.save(state, '%s/%s_checkpoint.pth' % (save_dir, exp_name))

        log = 'Epoch: {:03d}, Train_MSE_Loss: {:.4f}, Train_R2: {:.4f}, Test_MSE_Loss: {:.4f}, Test_R2: {:.4f}'
        print(log.format(epoch, train_mse_loss, train_r2_loss, test_mse_loss, test_r2_loss))
        # train_logger.log({
        #     'ep': epoch,
        #     'train_mse': train_mse_loss,
        #     'train_r2': train_r2_loss,
        #     'test_mse': test_mse_loss,
        #     'test_r2': test_r2_loss,
        # })
        writer.add_scalars(
            "loss",
            {"train": train_mse_loss.item()},
            global_step=epoch * len(train_loader),
        )
        writer.add_scalars(
            "r2",
            {"train": train_r2_loss.item()},
            global_step=epoch * len(train_loader),
        )
        writer.add_scalars(
            "loss",
            {"test": test_mse_loss.item()},
            global_step=epoch * len(train_loader)
        )
        writer.add_scalars(
            "r2",
            {"test": test_r2_loss.item()},
            global_step=epoch * len(train_loader),
        )
