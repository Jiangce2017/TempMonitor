import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import imageio
import numpy as np
import os.path as osp
import trimesh
import pickle

#import plotly.graph_objects as go
#from plotly_voxel_display.VoxelData import VoxelData
from matplotlib import cm
import matplotlib

def matplot_voxels(arr, color_bar_label, title=None, minimum=None, maximum=None, cmap='bwr',subplot=None, fig=None, fontsize=12, colorbar=True):
    if subplot == None:
        # Create a new figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot(*subplot, projection='3d')
    #

    # Plot the voxels
    cmap = plt.get_cmap(cmap)

    if minimum == None:
        minimum = np.amin(arr)
    if maximum == None:
        maximum = np.amax(arr)

    norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])

    if colorbar:
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cbar = plt.colorbar(m, cax=cax, aspect=0.5)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label(color_bar_label, fontsize=fontsize)
    plt.ticklabel_format(style="plain")
    ax.voxels(arr, edgecolor="None", facecolors=cmap(norm(arr)), alpha=0.5)
    # ax.invert_zaxis()

    # Display the plot
    if title != None:
        ax.set_title(title, fontsize=fontsize)

    # ax.tick_params(axis='x', labelsize=fontsize)
    # ax.tick_params(axis='y', labelsize=fontsize)
    # ax.tick_params(axis='z', labelsize=fontsize)
    ax.set_axis_off()

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return ax, fig

########################################################################################################################
def plot_data(true, prediction, windows, time_steps, sample_indices, file_name, fig, fontsize, worst_error=False):
    count = 0
    columns = 4
    number_samples = len(sample_indices)
    if worst_error:
        num_samples = true.shape[0]
        true_u = true.reshape(num_samples, -1)
        pred_u = prediction.reshape(num_samples, -1)
        mse = np.mean((pred_u - true_u) ** 2, axis=1)

        worst_sample_indices = np.argsort(mse)[-number_samples:]       
        sample_indices = np.flip(worst_sample_indices)



    for row, index in enumerate(sample_indices):

        # activation_sample = activation[index]
        # a_sample = a[index]
        u_pred_sample = prediction[index]
        u_true_sample = true[index]
        window_sample = windows[index]
        time_step_sample = time_steps[index]

        # List titles
        if row == 0:  # Only plot the titles for the first row
            title1 = "True"
            title2 = "Predicted"
            title3 = "Difference"
            title4 = "Percent Error"
        else:
            title1 = title2 = title3 = title4 = None
        # Plot the True Sample
        count += 1
        ax1, fig1 = matplot_voxels(u_true_sample, "Temperature",title=title1, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize,colorbar=True)
        # ax1.text(-18, -18, 4,
        #          "Sample: " + str(index) + "\nTime step: " + str(time_step_sample) + "\nWindow: " + str(window_sample),
        #          verticalalignment='center', fontsize=fontsize)
        
        ax1.text(-30, -18, 4,
                 "Sample: " + str(index), verticalalignment='center', fontsize=fontsize)

        # Plot the predicted sample
        count += 1
        ax2, fig2 = matplot_voxels(u_pred_sample, "Temperature",title=title2, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize,colorbar=True)

        # Plot Difference
        #error = u_true_sample - u_pred_sample
        error = u_pred_sample - u_true_sample
        count += 1
        max_err = np.max(np.abs(error))
        ax3, fig3 = matplot_voxels(error, "Temperature", title=title3,minimum=-max_err, maximum=max_err,cmap='PuOr',\
                                    subplot=(number_samples, columns, count), fig=fig, fontsize=fontsize,colorbar=True)
        
        
        #c = plt.colorbar()
        #plt.clim(-max_err, max_err)
        
        # Plot Percent Error
        # Calculate the percentage errors, handling the case where actual is 0
        u_error = np.where(u_true_sample == 0, 0, np.divide(abs(error), abs(u_true_sample)) * 100)
        count += 1
        ax4, fig4 = matplot_voxels(u_error, "% Error", title=title4, subplot=(number_samples, columns, count), fig=fig,
                                   fontsize=fontsize, cmap='Reds')

    plt.savefig("figures/" + file_name)