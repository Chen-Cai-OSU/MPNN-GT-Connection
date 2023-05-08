from PIL import Image
from matplotlib import cm
import numpy as np

def create_gif(input_fn, output_fn):
    data = np.load(input_fn)
    num_frames = data.shape[0]
    all_images = []
    for frame in range(num_frames):
        arr = data[frame, :, :]
        img = Image.fromarray(np.uint8(cm.gist_earth(arr) * 255))
        all_images.append(img)
    all_images = iter(all_images)
    img = next(all_images)  # extract first image from iterator
    img.save(fp = output_fn, format = 'GIF', append_images = all_images, save_all = True, duration = 200, loop = 0)

fn_1 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.1.num_shortcuts.100.test_predicts.npy'
fn_2 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.1.num_shortcuts.100.test_targets.npy'
create_gif(fn_1, 'predict.window.1.gif')
create_gif(fn_2, 'target.window.1.gif')

fn_1 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.7.num_shortcuts.100.test_predicts.npy'
fn_2 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.7.num_shortcuts.100.test_targets.npy'
create_gif(fn_1, 'predict.window.7.gif')
create_gif(fn_2, 'target.window.7.gif')

fn_1 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.28.num_shortcuts.100.test_predicts.npy'
fn_2 = 'train_mpnn_shortcuts/train_mpnn_shortcuts.num_epoch.10.batch_size.20.learning_rate.0.001.seed.123456789.n_layers.6.hidden_dim.256.z_dim.256.window.28.num_shortcuts.100.test_targets.npy'
create_gif(fn_1, 'predict.window.28.gif')
create_gif(fn_2, 'target.window.28.gif')

print('Done')
