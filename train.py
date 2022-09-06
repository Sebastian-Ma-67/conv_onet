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

cuda_visible_devices = "2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_visible_devices # 将2, 3号显卡设置为可见卡
# my_device_ids=[0,1]P

print("CUDA_VISIBLE_DEVICES",": ", cuda_visible_devices )
print('the new GPU device list is: ', list(range(torch.cuda.device_count())))

# Command Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
args = parser.parse_args()

# yaml file load
cfg = config.load_config(args.config, 'configs/default.yaml')


# 创建mesh输出文件夹以及 check_point 存储文件夹
if not os.path.exists(cfg['data']['sample_dir']):
    os.makedirs(cfg[data]['sample_dir'])
if not os.path.exists(cfg['model']['checkpoint_dir']):
    os.makedirs(cfg['model']['checkpoint_dir'])

# 配置 device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')




# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
input_type = cfg['data']['input_type']
model_selection_metric = cfg['training']['model_selection_metric']
checkpoint_dir = cfg['model']['checkpoint_dir']
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']

net_bool = False
net_float = False
receptive_padding = 3 #for grid input
saving_checkpoint_name = "weights_" + input_type + "_bool.pth"

exit_after = args.exit_after


if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# if FLAGS.train_bool or FLAGS.test_bool:
if cfg['training']['bool'] or cfg['test']['bool']:
    net_bool = True
if cfg['training']['float'] or cfg['test']['float']:
    net_float = True
if cfg['test']['input'] != None:
    quick_testing = True
    net_bool = True
    net_float = True


# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml')) # 将参数文件同样保存在输出文件夹中

# Dataset
train_dataset = config.init_dataset(cfg, train=True, out_bool=net_bool, out_float=net_float) # 获取训练数据
val_dataset = config.init_dataset(cfg,  train=False, out_bool=True, out_float=True, return_idx=True) # 获取验证数据

# 构建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn) 
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn) 


# 初始化网络，优化器，训练器，并确定训练的模型类型
if net_bool:
    network_bool = config.init_network(cfg, device=device)
    optimizer = torch.optim.Adam(network_bool.parameters(), lr=1e-4)
    # if torch.cuda.device_count() > 1:
        # print(my_device_ids)
        # network_bool = torch.nn.DataParallel(network_bool, device_ids=my_device_ids)
        # optimizer = torch.nn.DataParallel(optimizer, device_ids=my_device_ids)
        
    network_bool.to(device)  
    trainer = config.get_trainer(network_bool, optimizer, cfg, device=device) # 将模型，优化器，配置参数攒在一起，就构成了训练器
      
    model = network_bool
if net_float:
    network_float = config.init_network(cfg, device=device)    
    # if torch.cuda.device_count() > 1:
        # network_float = torch.nn.DataParallel(network_float)

    network_float.to(device)
    optimizer = torch.optim.Adam(network_float.parameters(), lr=1e-4)
    trainer = config.get_trainer(network_float, optimizer, cfg, device=device) # 将模型，优化器，配置参数攒在一起，就构成了训练器
    model = network_float
    
# 初始化 checkpoint_io 类，并加载已经训练好的模型（如果有的话，没有的话，就初始化为空）
checkpoint_io = CheckpointIO(checkpoint_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load(saving_checkpoint_name)
except FileExistsError:
    load_dict = dict()
# 得到 epoch 以及 it(eration) 的次数
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)

# 得到 val_metric
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Print model's size
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters) # 1068545 ≈ 1M

# print('output path: ', cfg['training']['out_dir'])

        
while True:
    epoch_it += 1
# for epoch in range(cfg['training']['epoch']):
    
    avg_loss = 0
    avg_acc_bool_all = 0
    avg_acc_float_all = 0
    avg_loss_count = 0
    avg_acc_bool_count = 0
    avg_acc_float_count = 0
    
    for batch in train_loader:
        it += 1
        loss, avg_acc_bool, avg_acc_float = trainer.train_step(batch) # 开始训练
        logger.add_scalar('train/loss', loss, it)

        if net_bool:
            avg_acc_bool_all += avg_acc_bool
            avg_acc_bool_count += 1

        if net_float:
            avg_acc_float_all += avg_acc_float
            avg_acc_float_count += 1
            
        avg_loss += loss
        avg_loss_count += 1

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d' % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):                
            print('Saving checkpoint')
            checkpoint_io.save(saving_checkpoint_name, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            print('saving net... complete')
        
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            # eval_dict = trainer.evaluate(val_loader, it)
            eval_dict = trainer.evaluate(val_loader, it)
            # metric_val = eval_dict[model_selection_metric]
            # print('Validation metric (%s): %.4f'
            #       % (model_selection_metric, metric_val))
            
            # for k, v in eval_dict.items():
            #     logger.add_scalar('val/%s' % k, v, it)
                
            # if model_selection_sign * (metric_val - metric_val_best) > 0:
            #     metric_val_best = metric_val
            #     print('New best model (loss %.4f)' % metric_val_best)
            #     checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save(saving_checkpoint_name, epoch_it=epoch_it, it=it,
                            loss_val_best=metric_val_best)
            exit(3)

            
    # print('[%d/%d] time: %.0f  avg_loss: %.8f  avg_acc_loss_bool_mean: %.8f  avg_acc_loss_float_mean: %.8f' % (
    #     epoch_it, 
    #     cfg['training']['epoch'], 
    #     time.time() - t0, 
    #     avg_loss / max(avg_loss_count, 1), 
    #     avg_acc_bool_all / max(avg_acc_bool_count, 1), 
    #     avg_acc_float_all / max(avg_acc_float_count, 1)))
