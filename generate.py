import torch
import os
# import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
# import pandas as pd
from src import config
from src.checkpoints import CheckpointIO
# from src.utils.io import export_pointcloud
# from src.utils.visualize import visualize_data
# from src.utils.voxels import VoxelGrid
from src.conv_onet import models
import numpy as np
import cutils
from src.data.utils import write_obj_triangle, write_ply_point

cuda_visible_devices = "0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_visible_devices # 将2, 3号显卡设置为可见卡



parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')

 
# 配置 device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
        
# Set t0
t0 = time.time()

out_dir = cfg['training']['out_dir']
checkpoint_dir = cfg['model']['checkpoint_dir']
# generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
# out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
# out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

input_type = cfg['data']['input_type'] # pointcloud

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
test_dataset = config.get_init_dataset(cfg, train=False, out_bool=True, out_float=True) # 获取训练数据

# Model
if net_bool:
    # network_bool = model(out_bool=True, out_float=False)
    network_bool = config.get_init_network(cfg, device=device)
    network_bool.to(device)
    network_bool.eval() # 将该网络改成eval模式
    optimizer = torch.optim.Adam(network_bool.parameters(), lr=1e-4)
    model = network_bool

    
print('loading net...')

checkpoint_io = CheckpointIO(checkpoint_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])

print('loading net... complete')

# Loader
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, num_workers=1, shuffle=False)

# Statistics
time_dicts = []

# Count how many objects already created
model_counter = defaultdict(int)


grid_size = 32
num = 0 
for it, data in enumerate(tqdm(test_loader)):

    if num > 100:
        break
    else:
        num += 1
    if 1:
        input_points = data['input_points'].to(device)
        gt_output_bool_ = data['gt_output_bool'].to(device)
        gt_output_float_ = data['gt_output_float'].to(device)
        input_name = data.get('input_name')[0]
        
        with torch.no_grad():
            if net_bool:
                pred_output_bool = network_bool(input_points).probs

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
    write_obj_triangle(cfg['data']['sample_dir']+"/test_"+input_name+'_'+str(it)+".obj", vertices, triangles)
    write_ply_point("samples/test_"+input_name+'_'+str(it)+".ply", input_points[0].detach().cpu().numpy())
