### evaluate model

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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from hammer.utilities.plotting import plot_data
from scipy import stats
import matplotlib.cm as cm

#from point_cloud_models import BallConvNet, DynamicEdge, MixConv
from models import TAGConvNet,MixConv

def r2loss(pred, y):
    #print(pred.shape,y.shape)
    SS_res = torch.sum((pred-y)**2)
    y_mean = torch.mean(y)
    SS_tot = torch.sum((y-y_mean)**2)
    r2 = 1 - SS_res/SS_tot
    return r2
    
def coord2inds(float_coords, dx):
        float_min = np.min(float_coords,axis=0,keepdims=True)
        int_coords = np.round((float_coords-float_min)/dx).astype(int)
        return int_coords
    
def evaluate():
    model.eval()
    test_metrics = {"mse_loss": [],"r2_loss":[]}

    pred_u = []
    true_u = []
    ele_number = []
    max_temp = []

    pred_u_vec = []
    true_u_vec = []

    z_pos = []
    ele_number_for_all = []
      
    world_res = 25
    ni,mi,li = np.indices((world_res,world_res,world_res))
    
    for batch_i, data in enumerate(test_loader):
        data = data.to(device)
        batch = data.batch
        idx = batch==0
        with torch.no_grad():
            pos = data.x_input[:,:3]
            pos = pos[~data.boundary_mask*idx]
            pos = pos.to('cpu').detach().numpy()
            if pos.shape[0] > 0:
                predictions = model(data)
                predictions = predictions[~data.boundary_mask*idx].view(-1)
                true_temp = data.y_output[~data.boundary_mask*idx]
                mse_loss = F.mse_loss(predictions, true_temp, reduction='mean')
                r2_loss = r2loss(predictions, true_temp)
                
                
                global_pred_u = np.zeros(ni.shape,dtype=np.float32)
                global_true_u = np.zeros(ni.shape,dtype=np.float32)
                
                active_mask = pos[:,2] > 0
                pos = pos[active_mask]
                
                dx = 0.002
                int_coords = coord2inds(pos,dx)
                predictions = predictions[active_mask].to('cpu').detach().numpy()
                true_temp = true_temp[active_mask].to('cpu').detach().numpy()
                #print(int_coords[:,2])
                global_pred_u[int_coords[:,0],int_coords[:,1],int_coords[:,2]] = predictions
                global_true_u[int_coords[:,0],int_coords[:,1],int_coords[:,2]] = true_temp
                
                pred_u.append([global_pred_u])
                true_u.append([global_true_u])

                ele_number.append(true_temp.shape[0])
                max_temp.append(np.max(true_temp))
                
                if true_temp.shape[0] > 0:
                    pred_u_vec.append(predictions)
                    true_u_vec.append(true_temp)
                    z_pos.append(data.x_input[~data.boundary_mask*idx,2].to('cpu').detach().numpy())
                    ele_number_for_all.append(np.ones(predictions.shape[0])*predictions.shape[0] )
            
        test_metrics["mse_loss"].append(mse_loss.item())
        test_metrics["r2_loss"].append(r2_loss.item())
    pred_u = np.concatenate(pred_u,axis=0)
    true_u = np.concatenate(true_u,axis=0)

    pred_u_vec = np.concatenate(pred_u_vec,axis=0)
    true_u_vec = np.concatenate(true_u_vec,axis=0)
    z_pos = np.concatenate(z_pos,axis=0)
    ele_number_for_all = np.concatenate(ele_number_for_all,axis=0)


        
    return np.mean(test_metrics["mse_loss"]),np.mean(test_metrics["r2_loss"]),pred_u,true_u,ele_number,test_metrics["mse_loss"],max_temp,pred_u_vec,true_u_vec,z_pos,ele_number_for_all

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,s=0.2)

    # now determine nice limits by hand:
    binwidth = 10
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth


    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histx.set_ylim([0,20000])
    density_x = stats.gaussian_kde(x)
    x_pos = np.arange(0,xymax)
    ax_histx_twin = ax_histx.twinx()
    ax_histx_twin.plot(density_x(x_pos),'black')
    ax_histx_twin.set_ylim([0,0.02])

    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histy.set_xlim([0,20000])
    density_y = stats.gaussian_kde(y)
    y_pos = np.arange(0,xymax)
    ax_histy_twin = ax_histy.twiny()
    ax_histy_twin.plot(density_y(y_pos),y_pos,'black')
    ax_histy_twin.set_xlim([0,0.02])

if __name__ == '__main__':

    dataset_name = 'hollow1_bjorn'
    data_path = osp.join('dataset')
    #pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = GraphDataset(data_path, '10', True)
    #normalizer = train_dataset.normalizer
    test_dataset = GraphDataset(data_path, '10', False)
    
    print(len(train_dataset))
    print(len(test_dataset))
    print('Dataset loaded.')
    
    bz = 2
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=False, drop_last=True,
                                  num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=False, drop_last=True,
                                 num_workers=1)
          
    model = TAGConvNet(4,10)
    model_name = 'Deep_TAGConvNet'
    device = torch.device('cuda:1')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    exp_name = dataset_name+model_name+'with_platform'
    #num_epochs = 2000
    print(exp_name)
    result_path = osp.join('.', 'results')
    
    
    path_model = osp.join(result_path,exp_name+'_checkpoint.pth')
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    cur_ep = checkpoint['epoch']
    

    mse_loss, r2_loss, pred_u,true_u, ele_number, mse_arr,max_temp,pred_u_vec,true_u_vec,z_pos,ele_number_for_all = evaluate()
    
    print("mse_loss:{}, r2_loss:{}".format(mse_loss, r2_loss))
    print(pred_u.shape)
    print(true_u.shape)

    non_zero = true_u > 0
    print(np.sum(non_zero[-1]))

    number_samples = 10
    columns = 4

    random_samples = sorted(np.random.randint(0,len(true_u), number_samples))  # Generates 10 ordered samples
    #random_samples = np.arange(len(true_u)-10,len(true_u))
    #random_samples = np.arange(0,10)

    fontsize = number_samples * 7
    file_name_valid = "test_error"

    window_numbers = np.arange(len(true_u))
    time_steps = np.arange(len(true_u))
    fig3 = fig2 = plt.figure(figsize=(16*columns, number_samples*10))
    plot_data(true_u, pred_u, window_numbers, time_steps, random_samples, file_name_valid, fig3, fontsize, worst_error=True)

    # fig,ax = plt.subplots()
    # ax.plot(ele_number,mse_arr,'o')
    # ax.set_xlabel('element number')
    # ax.set_ylabel('mse')
    # plt.savefig("figures/" + "ele_num_correlation.png")

    # fig,ax = plt.subplots()
    # ax.plot(max_temp,mse_arr,'o')
    # ax.set_xlabel('max temperature')
    # ax.set_ylabel('mse')
    # plt.savefig("figures/" + "max_temp_correlation.png")

    # fig, ax = plt.subplots()
    # ax.scatter(true_u_vec, pred_u_vec,s=0.2)
    # ax.plot([pred_u_vec.min(), pred_u_vec.max()], [pred_u_vec.min(), pred_u_vec.max()], 'r--', lw=4)
    # ax.set_xlabel('Ground Truth Temperature (Kelvin)')
    # ax.set_ylabel('Predicted Temperature (Kelvin)')
    # #regression line
    # true_u_vec, pred_u_vec = true_u_vec.reshape(-1,1), pred_u_vec.reshape(-1,1)
    # ax.plot(true_u_vec, LinearRegression().fit(true_u_vec, pred_u_vec).predict(true_u_vec))
    # ax.annotate("r-squared = {:.3f}".format(r2_score(true_u_vec, pred_u_vec)),(200,1000))
    # plt.savefig("figures/" + "scatter_point.png")
    # print(r2_score(true_u_vec, pred_u_vec))



    # # Start with a square Figure.
    # fig = plt.figure(figsize=(8, 8))
    # # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # # the size of the marginal axes and the main axes in both directions.
    # # Also adjust the subplot parameters for a square plot.
    # gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
    #                     left=0.1, right=0.9, bottom=0.1, top=0.9,
    #                     wspace=0.05, hspace=0.05)
    # # Create the Axes.
    # ax = fig.add_subplot(gs[1, 0])
    # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    # # Draw the scatter plot and marginals.
    # scatter_hist(true_u_vec, pred_u_vec, ax, ax_histx, ax_histy)
    # ax.set_xlabel('Ground Truth Temperature (Celsius)')
    # ax.set_ylabel('Predicted Temperature (Celsius)')
    # #regression line
    # true_u_vec, pred_u_vec = true_u_vec.reshape(-1,1), pred_u_vec.reshape(-1,1)
    # #ax.plot(true_u_vec, LinearRegression().fit(true_u_vec, pred_u_vec).predict(true_u_vec))
    # ax.plot([pred_u_vec.min(), pred_u_vec.max()], [pred_u_vec.min(), pred_u_vec.max()], 'r--', lw=1)
    # ax.annotate("r-squared = {:.3f}".format(r2_score(true_u_vec, pred_u_vec)),(200,1000))
    # plt.savefig("figures/" + "scatter_histogram.png")



    # fig,ax = plt.subplots()
    # ax.scatter(z_pos, (pred_u_vec-true_u_vec)/true_u_vec,s=0.2 )
    # plt.savefig("figures/" + "z_pos_error.png")

    # fig,ax = plt.subplots()
    # color = (pred_u_vec-true_u_vec)/true_u_vec
    # c_min = np.min(color)
    # c_max = np.max(color)
    # color = (color-c_min)/(c_max-c_min)
    # #colors = cm.rainbow(color)
    # for i_data in range(color.shape[0]):
    #     ax.scatter(z_pos[i_data], ele_number_for_all[i_data],c=color[i_data],s=0.2 )
    # plt.savefig("figures/" + "z_pos_error.png")
