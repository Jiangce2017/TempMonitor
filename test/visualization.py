import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.TempMonitor import GraphDataset
from torch_geometric.loader import DataLoader

data_path = osp.join('dataset')
test_dataset = GraphDataset(data_path, '10', False)
plot_graph = test_dataset[0]

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111, projection='3d')
plot = ax.scatter3D(plot_graph['x_input'][:, 0],
                    plot_graph['x_input'][:, 1],
                    plot_graph['x_input'][:, 2],
                    c=plot_graph['y_output'])
ax.view_init(elev=35, azim=225)
cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
cbar.set_ticks([200, 400, 600, 800, 1000, 1200, 1400])
cbar.set_ticklabels(['200', '400', '600', '800', '1000', '1200', '1400℃'])
plt.show()
