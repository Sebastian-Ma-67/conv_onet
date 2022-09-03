import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil

'''
注：为了分析训练的过程，暂时把可视化部分的代码注释掉
'''

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3" # 将0, 1, 2, 3号显卡设置为可见卡

my_device_ids=[0, 1, 2, 3]

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')



net_bool = False
net_float = False
# if FLAGS.train_bool or FLAGS.test_bool:
if cfg['training']['bool'] or cfg['test']['bool']:
    net_bool = True
if cfg['training']['float'] or cfg['test']['float']:
    net_float = True
if cfg['test']['bool'] and cfg['training']['float']:
    net_bool = True
    net_float = True

if cfg['test']['input'] != None:
    quick_testing = True
    net_bool = True
    net_float = True


receptive_padding = 3 #for grid input


is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml')) # 将参数文件同样保存在输出文件夹中

# Dataset
train_dataset = config.init_dataset(cfg, train=True, out_bool=net_bool, out_float=net_float) # 获取训练数据
val_dataset = config.init_dataset(cfg,  train=False, out_bool=True, out_float=True, return_idx=True) # 获取验证数据

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn) # 构建训练数据加载器
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn) # 构建验证数据加载器

# Model
# model = config.get_network(cfg, device=device, dataset=train_dataset)
# model = torch.nn.DataParallel(model, device_ids=my_device_ids) 
if net_bool:
    # network_bool = model(out_bool=True, out_float=False)
    network_bool = config.get_network(cfg, device=device, dataset=train_dataset)
    network_bool.to(device)
    optimizer = torch.optim.Adam(network_bool.parameters(), lr=1e-4)
    
if net_float:
    # network_float = model(out_bool=False, out_float=True)
    network_float = config.get_network(cfg, device=device, dataset=train_dataset)    
    network_float.to(device)
    optimizer = torch.optim.Adam(network_float.parameters(), lr=1e-4)

# Intialize training
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# optimizer = torch.nn.DataParallel(optimizer, device_ids=my_device_ids)
    

if net_bool:
    trainer = config.get_trainer(network_bool, optimizer, cfg, device=device) # 将模型，优化器，配置参数攒在一起，就构成了训练器

if net_float:
    trainer = config.get_trainer(network_float, optimizer, cfg, device=device) # 将模型，优化器，配置参数攒在一起，就构成了训练器

if net_bool:
    model = network_bool
elif net_float:
    model = net_float
    

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters) # 1068545 ≈ 1M

print('output path: ', cfg['training']['out_dir'])


if net_bool:
    optimizer = torch.optim.Adam(network_bool.parameters())
if net_float:
    optimizer = torch.optim.Adam(network_float.parameters())
        
# while True:
epoch_it += 1

for epoch in range(cfg['training']['epoch']):
    
    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch) # 开始训练
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                    % (epoch, it, loss, time.time() - t0, t.hour, t.minute))

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch, it=it,
                            loss_val_best=metric_val_best)
        
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader, it)


        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch, it=it,
                            loss_val_best=metric_val_best)
            exit(3)
