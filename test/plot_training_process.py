#### plot training process
import numpy as np
import trimesh
import os.path as osp
import os
from pathlib import Path
import argparse
import json
import glob
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

import matplotlib
font = {'size'   : 10}
matplotlib.rc('font', **font)

csfont = {'fontname':'Times New Roman'}

import pickle

### wall 2D plot
# with open('results/wall_2d.pkl', 'rb') as f:
#     list = pickle.load(f)
#     pred_list,true_list,pos_list,boundary_mask_list = list[0],list[1],list[2],list[3]
# print(len(pred_list))

# plt.figure(figsize=(32, 24))
# fig, ax = plt.subplots(3,1)

# #idx_list = [100,5,6,13,14]
# idx_list = [14]
# idx = 5
# for idx, i_sample in enumerate(idx_list):
#     pos = pos_list[i_sample]
#     boundary_mask = boundary_mask_list[i_sample]


#     interior_mask = [not elem for elem in boundary_mask]
#     pred = np.reshape(pred_list[i_sample],(-1))
#     true = np.reshape(true_list[i_sample],(-1))

#     vmax = np.max(true)
#     vmin = np.min(true)
#     shw = ax[0].scatter(pos[:, 0], pos[:, 2],  s=1.5, c=true,cmap='bwr',vmin=vmin,vmax=vmax)
#     plt.colorbar(shw,shrink=1)
#     #ax[0].axis('scaled')
#     ax[0].set_aspect(1.5)
#     ax[0].axes.yaxis.set_ticklabels([])

#     shw = ax[1].scatter(pos[:, 0], pos[:, 2],  s=1.5, c=pred,cmap='bwr',vmin=vmin,vmax=vmax)
#     plt.colorbar(shw,shrink=1)
#     #ax[1].axis('scaled')
#     ax[1].set_aspect(1.5)
#     ax[1].axes.yaxis.set_ticklabels([])


#     vmax = np.max(np.abs(pred-true))
#     shw = ax[2].scatter(pos[:, 0], pos[:, 2],  s=1, c=pred-true,cmap='bwr',vmin=-vmax,vmax=vmax)
#     plt.colorbar(shw,shrink=1)
#     ax[2].set_aspect(1.5)
#     #ax[2].axis('scaled')
#     ax[2].axes.yaxis.set_ticklabels([])

# # plt.subplots_adjust(left=0.1,
# #                 bottom=0,
# #                 right=0.9,
# #                 top=1.0,
# #                 wspace=0.05,
# #                 hspace=0.01)

# plt.savefig("figures/" + "wall_2d_plot_5.png")


traing_log_dir = osp.join('results')
total_model_names = ['hollow_1','hollow_2','hollow_3','hollow_4','hollow_5','townhouse_2','townhouse_3','townhouse_5','townhouse_6','townhouse_7']
n_ep = 50
#plt.figure(figsize=(16, 12))
# fig, axes = plt.subplots(10,4)
# fig.figsize = (32,24)
# fig.tight_layout(pad=10.0)
# col_titles = ['Train MSE','Train R2','Test MSE','Test R2']
# row_titles = ['No. {}'.format(i_row) for i_row in range(1, 11)]
# for ax, col in zip(axes[0], col_titles):
#     ax.set_title(col)

# for ax, row in zip(axes[:,0], row_titles):
#     ax.set_ylabel(row, rotation=90,  fontsize=10)

# for i,valid_model_name in enumerate(total_model_names):
#     model_name = 'Deep_TAGConvNet'
#     exp_name = 'eval_'+ valid_model_name+'_'+model_name+'with_platform'
#     file_path = osp.join(traing_log_dir,exp_name+'_train.log')
#     df = pd.read_table(file_path)
#     axes[i,0].plot(df['ep'][:n_ep],df['train_mse'][:n_ep])
#     axes[i,1].plot(df['ep'][:n_ep],df['train_r2'][:n_ep])
#     axes[i,2].plot(df['ep'][:n_ep],df['test_mse'][:n_ep])
#     axes[i,3].plot(df['ep'][:n_ep],df['test_r2'][:n_ep])
# #plt.subplot_tool()
# plt.savefig('./figures/cross_valid_training_process1.png')

# fig, axes = plt.subplots(2,2)
# fig.figsize = (32,24)
# col_titles2 = ['MSE', 'R2']
# row_titles2 = ['Train','Valid']
# for ax, col in zip(axes[0], col_titles2):
#     ax.set_title(col)
# for ax, row in zip(axes[:,0], row_titles2):
#     ax.set_ylabel(row, rotation=0,  size='large')
    
# for i,valid_model_name in enumerate(total_model_names):
#     model_name = 'Deep_TAGConvNet'
#     exp_name = 'eval_'+ valid_model_name+'_'+model_name+'with_platform'
#     file_path = osp.join(traing_log_dir,exp_name+'_train.log')
#     df = pd.read_table(file_path)
#     axes[0,0].plot(df['ep'][:n_ep],df['train_mse'][:n_ep])
#     axes[0,1].plot(df['ep'][:n_ep],df['train_r2'][:n_ep])
#     axes[1,0].plot(df['ep'][:n_ep],df['test_mse'][:n_ep])
#     axes[1,1].plot(df['ep'][:n_ep],df['test_r2'][:n_ep])

# for i_row in range(2):
#     for i_col in range(2):
#         axes[i_row,i_col].legend(row_titles)    
  
# #plt.subplot_tool()
# plt.savefig('./figures/cross_valid_training_process2.png')

task_names = ['valid_wall_with_geo_data','valid_wall_with_simu_wall','valid_wall_with_all_data']
fig, axes = plt.subplots(2,2)
fig.figsize = (32,24)
col_titles2 = ['MSE', 'R2']
row_titles2 = ['Training','Validation']
for ax, col in zip(axes[0], col_titles2):
    ax.set_title(col)
    #ax.set_xlabel('Epoch', size='large')
for ax, row in zip(axes[:,0], row_titles2):
    ax.set_ylabel(row, rotation=90,  size='large')
    #ax.set_xlabel('Epoch', size='large')

for ax in axes[1,:]:
    ax.set_xlabel('Epoch', size='large')

line = [':', '--','-']    
for i,valid_model_name in enumerate(task_names):
    model_name = 'Deep_TAGConvNet'
    exp_name = valid_model_name+'_'+model_name+'with_platform'
    file_path = osp.join(traing_log_dir,exp_name+'_train.log')
    df = pd.read_table(file_path)
    axes[0,0].plot(df['ep'][:n_ep],df['train_mse'][:n_ep],line[i],color='black')
    axes[0,1].plot(df['ep'][:n_ep],df['train_r2'][:n_ep],line[i],color='black')
    axes[1,0].plot(df['ep'][:n_ep],df['test_mse'][:n_ep],line[i],color='black')
    axes[1,1].plot(df['ep'][:n_ep],df['test_r2'][:n_ep],line[i],color='black')

# for i_row in range(2):
#     for i_col in range(2):
#         axes[i_row,i_col].legend(['Normal data','Wall simulation','Augmented data'])    
  
#plt.subplot_tool()
plt.savefig('./figures/wall_training_process2.png')
