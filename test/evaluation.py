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

#from hammer.utilities.plotting import plot_data
from plotting import plot_data
from scipy import stats
import matplotlib.cm as cm
import scipy.stats as st
import pickle

#from point_cloud_models import BallConvNet, DynamicEdge, MixConv
from models import TAGConvNet,MixConv

from train import my_dataset_loader

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import matplotlib
font = {'size'   : 10}
matplotlib.rc('font', **font)

csfont = {'fontname':'Times New Roman'}

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
    
def evaluate(test_loader,geo_type):
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

    pos_list = []
    pred_list = []
    true_list = []
    boundary_mask_list = []
    
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

                pred_list.append(predictions[idx].to('cpu').detach().numpy())
                true_list.append(data.y_output[idx].to('cpu').detach().numpy())
                pos_list.append(data.x_input[idx,:3].to('cpu').detach().numpy())
                
                local_mask = data.boundary_mask[idx]                
                boundary_mask_list.append(local_mask.to('cpu').detach().numpy())

                predictions = predictions[~data.boundary_mask*idx].view(-1)
                true_temp = data.y_output[~data.boundary_mask*idx]
                mse_loss = F.mse_loss(predictions, true_temp, reduction='mean')
                r2_loss = r2loss(predictions, true_temp)
                
                
                
                dx = np.min(np.max(np.abs(pos[:-1]-pos[1:]),axis=1))
                active_mask = pos[:,2] > 0
                pos = pos[active_mask]
                if pos.shape[0] > 0:
                    #dx = 0.002
                    
                    int_coords = coord2inds(pos,dx)
                    predictions = predictions[active_mask].to('cpu').detach().numpy()
                    true_temp = true_temp[active_mask].to('cpu').detach().numpy()
                    #print(np.max(int_coords,axis=0))
                    if geo_type == 'wall':
                        ni,mi,li = np.indices((120,1,10))
                    else:
                        world_res = 25
                        ni,mi,li = np.indices((world_res,world_res,world_res))
                    global_pred_u = np.zeros(ni.shape,dtype=np.float32)
                    global_true_u = np.zeros(ni.shape,dtype=np.float32)

                    global_pred_u[int_coords[:,0],int_coords[:,1],int_coords[:,2]] = predictions
                    global_true_u[int_coords[:,0],int_coords[:,1],int_coords[:,2]] = true_temp
                    
                    pred_u.append([global_pred_u])
                    true_u.append([global_true_u])

                    ele_number.append(true_temp.shape[0])
                    max_temp.append(np.max(true_temp))
                    
                    if true_temp.shape[0] > 0:
                        pred_u_vec.append(predictions)
                        true_u_vec.append(true_temp)

                        
                        #z_pos.append(data.x_input[~data.boundary_mask*idx,2].to('cpu').detach().numpy())
                        z_pos.append(pos[:,2])
                        ele_number_for_all.append(np.ones(predictions.shape[0])*predictions.shape[0] )
            
        test_metrics["mse_loss"].append(mse_loss.item())
        test_metrics["r2_loss"].append(r2_loss.item())
    pred_u = np.concatenate(pred_u,axis=0)
    true_u = np.concatenate(true_u,axis=0)

    pred_u_vec = np.concatenate(pred_u_vec,axis=0)
    true_u_vec = np.concatenate(true_u_vec,axis=0)
    z_pos = np.concatenate(z_pos,axis=0)
    ele_number_for_all = np.concatenate(ele_number_for_all,axis=0)


        
    return np.mean(test_metrics["mse_loss"]),np.mean(test_metrics["r2_loss"]),\
        pred_u,true_u,ele_number,test_metrics["mse_loss"],max_temp,\
            pred_u_vec,true_u_vec,z_pos,ele_number_for_all,\
            pred_list,true_list,pos_list,boundary_mask_list

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
    ax_histx.set_ylim([0,10000])
    density_x = stats.gaussian_kde(x)
    x_pos = np.arange(0,xymax)
    ax_histx_twin = ax_histx.twinx()
    ax_histx_twin.plot(density_x(x_pos),'black')
    ax_histx_twin.set_ylim([0,0.01])

    ax_histy.hist(y, bins=bins, orientation='horizontal')
    ax_histy.set_xlim([0,10000])
    density_y = stats.gaussian_kde(y)
    y_pos = np.arange(0,xymax)
    ax_histy_twin = ax_histy.twiny()
    ax_histy_twin.plot(density_y(y_pos),y_pos,'black')
    ax_histy_twin.set_xlim([0,0.01])

if __name__ == '__main__':

    data_path = osp.join('dataset')
    geo_models = ['hollow_1','hollow_2','hollow_3','hollow_4','hollow_5','townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7']
    train_datasets = []
    test_datasets = []
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
        
        #print(len(train_dataset))
        #print(len(test_dataset))
        #print('Dataset loaded.')
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    task_name = 'cross_validation'
    #task_name = 'valid_wall_with_geo_data'
    #task_name = 'valid_wall_with_all_data'
    #task_name = 'valid_wall_with_simu_wall'
    
    geo_type = 'not_wall'
    train_loader_list, test_loader_list,dataset_name_list = my_dataset_loader(task_name,geo_models,train_datasets,test_datasets)

    std_mean_array = np.zeros((len(train_loader_list),2))

    # fig, ax = plt.subplots(2,5)


    for i_model in range(len(train_loader_list)):
        dataset_name = dataset_name_list[i_model]
        test_loader = test_loader_list[i_model]

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
        

        mse_loss, r2_loss, pred_u,true_u, ele_number, \
            mse_arr,max_temp,pred_u_vec,true_u_vec,\
            z_pos,ele_number_for_all,\
            pred_list,true_list,pos_list,boundary_mask_list = evaluate(test_loader,geo_type)
        

        print("mse_loss:{}, r2_loss:{}".format(mse_loss, r2_loss))

                ### calculate the error distribution
        # now determine nice limits by hand:
        
        errors = pred_u-true_u
        true_u_1d = np.reshape(true_u,(-1))
        non_zero_mask = true_u_1d > 0
        errors = np.reshape(errors,(-1))

        errors = errors[non_zero_mask]

        print(dataset_name)
        print(np.std(errors))
        print(np.mean(errors))
        std_mean_array[i_model,0] = np.std(errors)
        std_mean_array[i_model,1] = np.mean(errors)

        # with open('results/wall_2d.pkl', 'wb') as f:
        #     list = [pred_list,true_list,pos_list,boundary_mask_list]
        #     pickle.dump(list, f)

        # ###
        # pos = pos_list[0]
        # boundary_mask = boundary_mask_list[0]
        # print(pos.shape)
        # print(boundary_mask.shape)
        # fig, ax = plt.subplots(1,1)
        # ax.plot(pos[~boundary_mask, 0], pos[~boundary_mask, 2], '.b', markersize=1)
        # ax.plot(pos[boundary_mask, 0], pos[boundary_mask, 2], '.r', markersize=1)
        # plt.savefig("figures/" + "wall_2d_plot.png")


        # ##### draw 2D density
        # true_u_vec, pred_u_vec = true_u_vec.reshape(-1), pred_u_vec.reshape(-1)
        # xmin,xmax = np.min(true_u_vec), np.max(true_u_vec)
        # ymin,ymax = np.min(pred_u_vec), np.max(pred_u_vec)
        # xx, yy = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
        # positions = np.vstack([xx.ravel(), yy.ravel()])
        # values = np.vstack([true_u_vec, pred_u_vec])
        # kernel = st.gaussian_kde(values)
        # f = np.reshape(kernel(positions).T, xx.shape)
        # fig = plt.figure()
        # ax = fig.gca()
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(ymin, ymax)
        # # Contourf plot
        # #cfset = ax.contourf(xx, yy, f, cmap='Blues')
        # ## Or kernel density estimate plot instead of the contourf plot
        # shw = ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
        # plt.colorbar(shw)
        # # Contour plot
        # #cset = ax.contour(xx, yy, f, colors='k')
        # # Label plot
        # #ax.clabel(cset, inline=0.1, fontsize=5)
        # ax.set_xlabel('Ground Truth Temperature (Celsius)')
        # ax.set_ylabel('Predicted Temperature (Celsius)')
        # ax.plot([pred_u_vec.min(), pred_u_vec.max()], [pred_u_vec.min(), pred_u_vec.max()], 'r--', lw=1)
        # plt.savefig("figures/" + 'valid'+dataset_name+"kernel_density_estimate.png")




        # binwidth = 1
        # error_max = np.max(np.abs(errors))
        # lim = (int(error_max/binwidth) + 1) * binwidth
        # bins = np.arange(-lim, lim + binwidth, binwidth)

        # i_x = i_model % 5
        # i_y = i_model // 5
        # ax[i_y,i_x].hist(errors, bins=bins,color='black')
        # ax[i_y,i_x].set_ylim([0,20000])
        # ax[i_y,i_x].set_xlim([-60,60])
        # ax[i_y,i_x].set_title(str(i_model+1))
        # ax[i_y,i_x].set_xticks([])
        # ax[i_y,i_x].set_yticks([])
        #plt.savefig("figures/" + 'valid'+dataset_name+"error_distribution.png")


        number_samples = 1
        columns = 4
        random_samples = sorted(np.random.randint(0,len(true_u), number_samples))  # Generates 10 ordered samples
        #random_samples = np.arange(len(true_u)-10,len(true_u))
        #random_samples = np.arange(0,10)

        fontsize = 60
        file_name_valid = 'valid'+dataset_name+'test_error_worst'

        window_numbers = np.arange(len(true_u))
        time_steps = np.arange(len(true_u))
        fig3 = fig2 = plt.figure(figsize=(16*columns, number_samples*10))
        plot_data(true_u, pred_u, window_numbers, time_steps, random_samples, file_name_valid, fig3, fontsize, worst_error=True)





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
        # plt.savefig("figures/" + 'valid'+dataset_name+"scatter_histogram.png")



        # fig,ax = plt.subplots()
        # ax.scatter(z_pos, (pred_u_vec-true_u_vec)/true_u_vec,s=0.2 )
        # plt.savefig("figures/" + 'valid'+dataset_name+"z_pos_error.png")

    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.3,
    #                 hspace=0.4)
    # plt.savefig("figures/" + 'valid_all'+"error_distribution.png")   
    # np.savetxt('results/cross_validation_std_mean2.csv',std_mean_array,delimiter=',')
