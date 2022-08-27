import torch
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
from src.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
checkpoint_dir = cfg['generation']['checkpoint_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

input_type = cfg['data']['input_type'] # pointcloud
vis_n_outputs = cfg['generation']['vis_n_outputs'] # 2
if vis_n_outputs is None:
    vis_n_outputs = -1

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
network = config.get_network(cfg, device=device, dataset=dataset) # 这个model 就是个ConvolutionalOccupancyNetwork
# Generate
network.eval() # 将该网络改成非训练模式

checkpoint_io = CheckpointIO(checkpoint_dir, model=network)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(network, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud'] # 我们暂时不需要生成点云，所以把它置为false了

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Statistics
time_dicts = []

# Count how many models already created
model_counter = defaultdict(int)

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis')

    # Get index etc.
    idx = data['idx'].item()

    try:
        data_dict = dataset.get_model_dict(idx)
    except AttributeError:
        data_dict = {'model': str(idx), 'category': 'n/a'}
    
    data_name = data_dict['model']
    category_id = data_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'

    if category_id != 'n/a':
        mesh_dir = os.path.join(mesh_dir, str(category_id))
        pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
        in_dir = os.path.join(in_dir, str(category_id))

        folder_name = str(category_id)
        if category_name != 'n/a':
            folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

    # Create directories if necessary
    if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
        os.makedirs(generation_vis_dir)

    if generate_mesh and not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    if generate_pointcloud and not os.path.exists(pointcloud_dir):
        os.makedirs(pointcloud_dir)

    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'data_name': data_name,
    }
    time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    if generate_mesh:
        t0 = time.time()

        out = generator.generate_mesh(data) # 开始啦
        time_dict['mesh'] = time.time() - t0

        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        time_dict.update(stats_dict)

        # Write output
        mesh_out_file = os.path.join(mesh_dir, '%s.off' % data_name)
        mesh.export(mesh_out_file)
        out_file_dict['mesh'] = mesh_out_file

    # Copy to visualization directory for first vis_n_output samples
    c_it = model_counter[category_id]
    if c_it < vis_n_outputs:
        # Save output files
        img_name = '%02d.off' % c_it
        for k, filepath in out_file_dict.items():
            ext = os.path.splitext(filepath)[1]
            out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                    % (c_it, k, ext))
            shutil.copyfile(filepath, out_file)

    model_counter[category_id] += 1

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
