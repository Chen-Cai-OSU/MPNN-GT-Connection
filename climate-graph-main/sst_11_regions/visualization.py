import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

mlp = np.load('pytorch/MLP.predict.npy')
tfnet = np.load('pytorch/TF-Net.predict.npy')
transformer = np.load('pytorch/Transformer.predict.npy')
mpnn = np.load('pyg/MPNN.predict.npy')
mpnn_vn = np.load('pyg/MPNN_VN.predict.npy')
target = np.load('pyg/MPNN_VN.target.npy')

day_name = []
day_name.append('Feb 12, 2020')
day_name.append('Feb 13, 2020')
day_name.append('Feb 14, 2020')
day_name.append('Feb 15, 2020')
day_name.append('Feb 16, 2020')
day_name.append('Feb 17, 2020')
day_name.append('Feb 18, 2020')

print(mlp.shape)
print(tfnet.shape)
print(transformer.shape)
print(mpnn.shape)
print(mpnn_vn.shape)

num_days = mlp.shape[2]

fig, axs = plt.subplots(6, num_days, figsize = (10, 10), sharey = True, sharex = True)

def draw_figures(model_idx, model_name, model_data, day_name = None):
    for day in range(num_days):
        matrix = model_data[:, :, day]
        axs[model_idx, day].imshow(matrix)
        if day == 0:
            axs[model_idx, day].set_ylabel(model_name)
        if day_name is not None:
            axs[model_idx, day].set_xlabel(day_name[day])

num_digits = 4

mse_mlp = round(mean_squared_error(mlp.flatten(), target.flatten()), num_digits)
mse_tfnet = round(mean_squared_error(tfnet.flatten(), target.flatten()), num_digits)
mse_transformer = round(mean_squared_error(transformer.flatten(), target.flatten()), num_digits)
mse_mpnn = round(mean_squared_error(mpnn.flatten(), target.flatten()), num_digits)
mse_mpnn_vn = round(mean_squared_error(mpnn_vn.flatten(), target.flatten()), num_digits)

draw_figures(0, 'MLP\n MSE = ' + str(mse_mlp), mlp)
draw_figures(1, 'TF-Net\n MSE = ' + str(mse_tfnet), tfnet)
draw_figures(2, 'Transformer + LapPE\n MSE = ' + str(mse_transformer), transformer)
draw_figures(3, 'MPNN\n MSE = ' + str(mse_mpnn), mpnn)
draw_figures(4, 'MPNN + VN\n MSE = ' + str(mse_mpnn_vn), mpnn_vn)
draw_figures(5, 'Ground-truth', target, day_name)
plt.savefig('figure.png')

print('Done')
