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
from src.conv_onet import models
import numpy as np
import cutils
from src.data.utils import write_obj_triangle

parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')


if not os.path.exists(cfg['model']['checkpoint_dir']):
    os.makedirs(cfg['model']['checkpoint_dir'])
    
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

net_bool = False
net_float = False
# if FLAGS.train_bool or FLAGS.test_bool:
if cfg['training']['bool'] or cfg['test']['bool']:
    net_bool = True
if cfg['training']['float'] or cfg['test']['float']:
    net_float = True

if cfg['test']['input'] != None:
    quick_testing = True
    net_bool = True
    net_float = True


# Dataset
test_dataset = config.init_dataset(cfg, train=True, out_bool=net_bool, out_float=net_float) # 获取训练数据

# Model
if net_bool:
    # network_bool = model(out_bool=True, out_float=False)
    network_bool = config.init_network(cfg, device=device)
    network_bool.to(device)
    network_bool.eval() # 将该网络改成eval模式
    optimizer = torch.optim.Adam(network_bool.parameters(), lr=1e-4)
    
if net_float:
    # network_float = model(out_bool=False, out_float=True)
    network_float = config.init_network(cfg, device=device)    
    network_float.to(device)
    network_float.eval() # 将该网络改成eval模式
    optimizer = torch.optim.Adam(network_float.parameters(), lr=1e-4)

print('loading net...')
input_type = cfg['data']['input_type']
if net_bool:
    network_bool.load_state_dict(torch.load(cfg['model']['checkpoint_dir']+"/weights_"+input_type+"_bool.pth"))    
    print('network_bool weights loaded')
if net_float:
    network_float.load_state_dict(torch.load(cfg['model']['checkpoint_dir']+"/weights_"+input_type+"_float.pth"))
    print('network_float weights loaded')
print('loading net... complete')


# Generate


# checkpoint_io = CheckpointIO(checkpoint_dir, model=network)
# checkpoint_io.load(cfg['test']['model_file'])

# Generator
# generator = config.init_generator(network, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud'] # 我们暂时不需要生成点云，所以把它置为false了

# Loader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=0, shuffle=False)




# Statistics
time_dicts = []

# Count how many objects already created
model_counter = defaultdict(int)


grid_size = 32

for it, data in enumerate(tqdm(test_loader)):
    # Output folders
    mesh_dir = os.path.join(generation_dir, 'meshes')
    pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    in_dir = os.path.join(generation_dir, 'input')
    generation_vis_dir = os.path.join(generation_dir, 'vis')

    # Get index etc.
    # idx = data['idx'].item()

    # try:
    #     data_dict = dataset.get_model_dict(idx)
    # except AttributeError:
    #     data_dict = {'model': str(idx), 'category': 'n/a'}
    
    # data_name = data_dict['model']
    # category_id = data_dict.get('category', 'n/a')

    # try:
    #     category_name = dataset.metadata[category_id].get('name', 'n/a')
    # except AttributeError:
    #     category_name = 'n/a'

    # if category_id != 'n/a':
    #     mesh_dir = os.path.join(mesh_dir, str(category_id))
    #     pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))
    #     in_dir = os.path.join(in_dir, str(category_id))

    #     folder_name = str(category_id)
    #     if category_name != 'n/a':
    #         folder_name = str(folder_name) + '_' + category_name.split(',')[0]

        # generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

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
    # time_dict = {
    #     'idx': idx,
    #     'class id': category_id,
    #     'class name': category_name,
    #     'data_name': data_name,
    # }
    # time_dicts.append(time_dict)

    # Generate outputs
    out_file_dict = {}

    if generate_mesh:
        t0 = time.time()

        # out = generator.generate_mesh(data) # 开始啦

        input_points = data['input_points'].to(device)
        gt_output_bool_ = data['gt_output_bool'].to(device)
        gt_output_float_ = data['gt_output_float'].to(device)
        with torch.no_grad():
            if net_bool:
                pred_output_bool = network_bool(input_points)
            if net_float:
                pred_output_float = net_float(input_points)
                
            if not net_bool:
                pred_output_bool = gt_output_bool_[0].to(device)
            if not net_float:
                pred_output_float = gt_output_float_[0].to(device)
                
            pred_output_bool_grid = torch.zeros([grid_size+1,grid_size+1,grid_size+1,3], dtype=torch.int32, device=device)
            pred_output_float_grid = torch.full([grid_size+1,grid_size+1,grid_size+1,3], 0.5, device=device)
            pred_output_bool_grid[:-1, :-1, :-1] = (pred_output_bool>0.5).int()
            pred_output_float_grid[:-1, :-1, :-1] = pred_output_float
            if cfg['test']['postprocessing']:
                pred_output_bool_grid = models.postprocessing(pred_output_bool_grid)
            
            pred_output_bool_numpy = pred_output_bool_grid.detach().cpu().numpy()
            pred_output_float_numpy = pred_output_float_grid.detach().cpu().numpy()
                
    pred_output_float_numpy = np.clip(pred_output_float_numpy,0,1)
    
    #vertices, triangles = utils.dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
    vertices, triangles = cutils.dual_contouring_undc(np.ascontiguousarray(pred_output_bool_numpy, np.int32), np.ascontiguousarray(pred_output_float_numpy, np.float32))
    write_obj_triangle(cfg['data']['sample_dir']+"/test_"+str(it)+".obj", vertices, triangles)

        # time_dict['mesh'] = time.time() - t0

        # # Get statistics
        # try:
        #     mesh, stats_dict = out
        # except TypeError:
        #     mesh, stats_dict = out, {}
        # # time_dict.update(stats_dict)

        # # Write output
        # mesh_out_file = os.path.join(mesh_dir, '%s.off' % data_name)
        # mesh.export(mesh_out_file)
        # out_file_dict['mesh'] = mesh_out_file

    # Copy to visualization directory for first vis_n_output samples
    # c_it = model_counter[category_id]
    # if c_it < vis_n_outputs:
    #     # Save output files
    #     img_name = '%02d.off' % c_it
    #     for k, filepath in out_file_dict.items():
    #         ext = os.path.splitext(filepath)[1]
    #         out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
    #                                 % (c_it, k, ext))
    #         shutil.copyfile(filepath, out_file)

    # model_counter[category_id] += 1

# Create pandas dataframe and save
# time_df = pd.DataFrame(time_dicts)
# time_df.set_index(['idx'], inplace=True)
# time_df.to_pickle(out_time_file)

# # Create pickle files  with main statistics
# time_df_class = time_df.groupby(by=['class name']).mean()
# time_df_class.to_pickle(out_time_file_class)

# # Print results
# time_df_class.loc['mean'] = time_df_class.mean()
# print('Timings [s]:')
# print(time_df_class)
