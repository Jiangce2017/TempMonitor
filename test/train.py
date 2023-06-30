#### train the model
import torch
import numpy as np
from numpy import linalg as LA
import os.path as osp
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Softmax
from torch_geometric.nn import radius, TAGConv, global_max_pool as gmp, knn
from TempMonitor import GraphDataset
import csv

#from point_cloud_models import BallConvNet, DynamicEdge, MixConv
from models import TAGConvNet,MixConv

def r2loss(pred, y):
    #print(pred.shape,y.shape)
    SS_res = torch.sum((pred-y)**2)
    y_mean = torch.mean(y)
    SS_tot = torch.sum((y-y_mean)**2)
    r2 = 1 - SS_res/SS_tot
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
  
def train():
    model.train()
    train_metrics = {"mse_loss": [],"r2_loss":[]}
    for batch_i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        predictions = model(data)
        predictions = predictions[~data.boundary_mask].view(-1)
        true_temp = data.y_output[~data.boundary_mask]
        mse_loss = F.mse_loss(predictions, true_temp, reduction='mean')
        mse_loss.backward()
        optimizer.step()
        r2_loss = r2loss(predictions, true_temp)
        #print(mse_loss.item())
        train_metrics["mse_loss"].append(mse_loss.item())
        train_metrics["r2_loss"].append(r2_loss.item())
        
    return np.mean(train_metrics["mse_loss"]), np.mean(train_metrics["r2_loss"])
        
def test():
    model.eval()
    test_metrics = {"mse_loss": [],"r2_loss":[]}
    correct = 0
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
    return np.mean(test_metrics["mse_loss"]),np.mean(test_metrics["r2_loss"])

    

if __name__ == '__main__':
    data_path = osp.join('dataset')
    geo_models = ['hollow_1','hollow_2','hollow_3','hollow_4','hollow_5','townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7']
    dataset_name = 'eval_wall_general_data_'
    train_datasets = []
    test_datasets = []
    botton_included = True
    for geo_name in geo_models:
        
        data_root = osp.join(data_path,geo_name)
        train_dataset = GraphDataset(root=data_root, name=geo_name,train=True, \
                                     bottom_included=True,sliced_2d=False,bjorn=True)
        it = iter(train_dataset)
        data0 = next(it)
        figure_path = "figures/" + geo_name+"graph_example.png"
        train_dataset.plot_graph(data0['x_input'][:,:3], data0['edge_index'],data0['boundary_mask'],figure_path)
        test_dataset = GraphDataset(root=data_root, name=geo_name,train=False, \
                                     bottom_included=True,sliced_2d=False,bjorn=True)
        
        print(len(train_dataset))
        print(len(test_dataset))
        print('Dataset loaded.')
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    simu_wall_dataset = GraphDataset(root=osp.join(data_path,'wall'), name='wall',train=True, \
                                     bottom_included=False,sliced_2d=True,bjorn=False)
    
    simu_wall_dataset_test = GraphDataset(root=osp.join(data_path,'wall'), name='wall',train=False, \
                                     bottom_included=False,sliced_2d=True,bjorn=False)
    exp_wall_dataset = GraphDataset(root=osp.join(data_path,'exp_wall'),name= 'exp_wall',train=True, \
                                    bottom_included=False,sliced_2d=True,bjorn=False)
    #train_datasets.append(simu_wall_dataset)
    concat_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
  
    bz = 8
    train_loader = DataLoader(concat_train_dataset, batch_size=bz, shuffle=True, drop_last=True,
                                num_workers=2)
    test_loader = DataLoader(exp_wall_dataset, batch_size=bz, shuffle=True, drop_last=True,
                                num_workers=2)
    
        
    model = TAGConvNet(4,10)
    model_name = 'Deep_TAGConvNet'
    device = torch.device('cuda:1')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    exp_name = dataset_name+model_name+'with_platform'
    num_epochs = 50
    print(exp_name)
    result_path = osp.join('.', 'results')
    
    train_logger = Logger(
        osp.join(result_path, exp_name+'_train.log'),
        ['ep', 'train_mse','train_r2','test_mse','test_r2']
    )
    
    cur_ep = 0
    path_model = osp.join(result_path,exp_name+'_checkpoint.pth')
    if osp.isfile(path_model):
        checkpoint = torch.load(path_model)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cur_ep = checkpoint['epoch']
    
    best_test_r2 = 0
    for epoch in range(cur_ep,cur_ep+num_epochs): 
        train_mse_loss, train_r2_loss = train()
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
            torch.save(state, '%s/%s_checkpoint.pth' % (result_path, exp_name))

        print(exp_name)
        #log = 'Epoch: {:03d}, Train_Loss: {:.4f}, Train_Acc: {:.4f}, Test_Acc: {:.4f}'
        log = 'Epoch: {:03d}, Train_MSE_Loss: {:.4f}, Train_R2_loss: {:.4f}, Test_MSE_Loss: {:.4f}, Test_R2: {:.4f}'
        print(log.format(epoch, train_mse_loss,train_r2_loss, test_mse_loss, test_r2_loss))
        train_logger.log({
            'ep': epoch,             
            'train_mse': train_mse_loss,
            'train_r2': train_r2_loss,
            'test_mse': test_mse_loss,
            'test_r2': test_r2_loss,
        })
