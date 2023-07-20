import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from models import ARMAConvNet

from src.TempMonitor import GraphDataset

data_path = osp.join('dataset')
test_dataset = GraphDataset(data_path, '10', False)
plot_graph = test_dataset[0]

path_model = osp.join('.', 'results/Temp_Monitor_Deep_ARMAConvNet-15/Temp_Monitor_Deep_ARMAConvNet_checkpoint.pth')
model = ARMAConvNet(4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

checkpoint = torch.load(path_model)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']

model.eval()
predictions = model(plot_graph)
predictions = predictions[~plot_graph.boundary_mask].view(-1)
# print(predictions.tolist())
error = predictions - plot_graph['y_output'][~plot_graph.boundary_mask]
print(max(error).item(), min(error).item())
error_percent = error / plot_graph['y_output'][~plot_graph.boundary_mask]
print(max(error_percent).item(), min(error_percent).item())

fig = plt.figure(figsize=(12, 16))

ax1 = plt.subplot(221, projection='3d')
plot1 = ax1.scatter3D(plot_graph['x_input'][~plot_graph.boundary_mask][:, 0],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 1],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 2],
                      c=plot_graph['y_output'][~plot_graph.boundary_mask])
ax1.view_init(elev=35, azim=225)
ax1.set_title('Actual Inner Temperature')
cbar1 = fig.colorbar(plot1, ax=ax1, shrink=0.6)
cbar1.set_ticks([200, 400, 600])
cbar1.set_ticklabels(['200', '400', '600℃'])

ax2 = plt.subplot(222, projection='3d')
plot2 = ax2.scatter3D(plot_graph['x_input'][~plot_graph.boundary_mask][:, 0],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 1],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 2],
                      c=predictions.detach().numpy())
ax2.view_init(elev=35, azim=225)
ax2.set_title('Predicted Inner Temperature')
cbar2 = fig.colorbar(plot2, ax=ax2, shrink=0.6)
cbar2.set_ticks([200, 400, 600])
cbar2.set_ticklabels(['200', '400', '600℃'])

ax3 = plt.subplot(223, projection='3d')
plot3 = ax3.scatter3D(plot_graph['x_input'][~plot_graph.boundary_mask][:, 0],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 1],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 2],
                      c=error.detach().numpy())
ax3.view_init(elev=35, azim=225)
ax3.set_title('Absolute Error')
cbar3 = fig.colorbar(plot3, ax=ax3, shrink=0.6)
cbar3.set_ticks([-20, -10, 0, 10])
cbar3.set_ticklabels(['-20', '-10', '0', '10℃'])

ax4 = plt.subplot(224, projection='3d')
plot4 = ax4.scatter3D(plot_graph['x_input'][~plot_graph.boundary_mask][:, 0],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 1],
                      plot_graph['x_input'][~plot_graph.boundary_mask][:, 2],
                      c=error_percent.detach().numpy())
ax4.view_init(elev=35, azim=225)
ax4.set_title('Relative Error')
cbar4 = fig.colorbar(plot4, ax=ax4, shrink=0.6)
cbar4.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
cbar4.set_ticklabels(['-10', '-5', '0', '5', '10%'])

plt.subplots_adjust(wspace=.0, hspace=.3)
plt.show()
