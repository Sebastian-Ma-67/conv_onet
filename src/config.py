import yaml
from torchvision import transforms
from src import data
from src import conv_onet
# import sys
# sys.path.append('.')
# from data import datasetpc #可以考虑把datasetpc 放在某个包下面，比如和core放在一起

method_dict = {
    'conv_onet': conv_onet
} # 目前好像就这一个方法


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_network(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = conv_onet.config.get_network(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = conv_onet.config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = conv_onet.config.get_init_generator(model, cfg, device)
    return generator


# Datasets
def init_dataset(mode, cfg, train=False, out_bool=False, out_float=False, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method'] # convonet
    dataset_type = cfg['data']['dataset'] # Shapes3D
    dataset_folder = cfg['data']['path'] # data/demo/synthetic_room_dataset
    categories = cfg['data']['classes'] # 
    data_dir = cfg['data']['data_dir']


    point_num = cfg['data']['point_num']
    grid_size = cfg['data']['grid_size']
    pooling_radius = 2 #for pointcloud input
    input_type = cfg['data']['input_type']


    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode] # mdoe = 'test'

    # Create dataset 似乎现在只有一类dataset type: shapes3d

    # Dataset fields
    # Method specific fields (usually correspond to output)
    points_fields = conv_onet.config.init_points_fields(mode, cfg) # points.npz
    
    # Input fields
    point_cloud_field = init_point_cloud_field(mode, cfg) # pointcloud.npz
    if point_cloud_field is not None:
        points_fields['normal_points'] = point_cloud_field             

    if return_idx:
        points_fields['idx'] = data.IndexField() # 这个类似乎还没有开发完

    # shapes_3d_dataset = data.Shapes3dDataset(
    #     dataset_folder,
    #     points_fields,
    #     split=split,
    #     categories=categories,
    #     cfg = cfg
    # )
    
    shapes_3d_dataset = data.ABC_pointcloud_hdf5(
        data_dir,
        point_num,
        grid_size,
        pooling_radius,
        input_type,
        train,
        out_bool=out_bool,
        out_float=out_float        
    )
 
    return shapes_3d_dataset


def init_point_cloud_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''

    pointcloud_subsample = data.PointcloudSubsample(cfg['data']['pointcloud_n'])
    pointcloud_noise_transform = data.PointcloudNoiseTransform(cfg['data']['pointcloud_noise'])
    total_transforms = transforms.Compose([pointcloud_subsample, pointcloud_noise_transform]) # 这里只是初始化
    
    point_cloud_field = data.PointCloudField(
        cfg['data']['pointcloud_file'], 
        total_transforms,
        multi_files= cfg['data']['multi_files']
    ) # 这里只是初始化

    return point_cloud_field