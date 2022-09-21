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
import shutil
import wandb

os.environ["WANDB_DISABLE_CODE"]= 'true'
os.environ['WANDB_MODE']='online' # offline 表示暂停使用wandb服务

cuda_visible_devices = "3" # 将 3 号显卡设置为可见卡
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_visible_devices 

wandb.init(
    # Set the project where this run will be logged
    project="convonet_dc",# project name
    entity="qixuema",# account name
    # Track hyperparameters and run metadata
   )

# Command Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
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
learning_rate = cfg['training']['lr']
train_epochs = cfg['training']['epochs']

net_bool = False
net_float = False
receptive_padding = 3 #for grid input
saving_checkpoint_name = "weights_" + input_type + "_bool.pth"


wandb.config={
    "learning_rate": learning_rate,
    "epochs": train_epochs,
    "batch_size": batch_size,
    }


if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# if FLAGS.train_bool or FLAGS.test_bool:
if cfg['training']['bool']:
    net_bool = True

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml')) # 将参数文件同样保存在输出文件夹中

# Dataset
train_dataset = config.get_init_dataset(cfg, train=True, out_bool=net_bool, out_float=net_float) # 获取训练数据
val_dataset = config.get_init_dataset(cfg,  train=False, out_bool=True, out_float=True, return_idx=True) # 获取验证数据

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
network_bool = config.get_init_network(cfg, device=device)
optimizer = torch.optim.Adam(network_bool.parameters(), lr=1e-4)
    
network_bool.to(device)  
trainer = config.get_init_trainer(network_bool, optimizer, cfg, device=device) # 将模型，优化器，配置参数攒在一起，就构成了训练器
model = network_bool
    
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
nparameters = sum(p.numel() for p in network_bool.parameters())
print('Total number of parameters: %d' % nparameters) # 1068545 ≈ 1M
        
for epoch in range(train_epochs):
    epoch_it += 1
    
    avg_loss = 0
    avg_acc_bool_all = 0
    avg_acc_float_all = 0
    avg_loss_count = 0
    avg_acc_bool_count = 0
    avg_acc_float_count = 0
    
    if epoch_it % cfg['training']['lr_half_life'] == 0: # 调整学习率
        for g in optimizer.param_groups:
            lr = cfg['training']['lr']/(2**(epoch_it//cfg['training']['lr_half_life']))
            print("Setting learning rate to", lr)
            g['lr'] = lr
    
    for batch in train_loader:
        it += 1
        loss, avg_acc_bool, avg_acc_float = trainer.train_step(batch) # 开始训练
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            t = datetime.datetime.now()
            print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d' % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))
            wandb.log({"Train loss": loss})

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0): 
            print('Saving checkpoint')
            checkpoint_io.save(saving_checkpoint_name, epoch_it=epoch_it, it=it, loss_val_best=metric_val_best)
            print('saving net... complete')
        
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            print('val')
    
wandb.finish()
