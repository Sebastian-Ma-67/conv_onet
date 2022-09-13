import yaml
# from src import data
from src.data.datasetpc import ABC_pointcloud_hdf5
from src import conv_onet

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
def get_init_network(cfg, device=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    # method = cfg['method']
    model = conv_onet.config.get_init_network(
        cfg, device=device)
    return model


# Trainer
def get_init_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    trainer = conv_onet.config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def init_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''

    generator = conv_onet.config.get_init_generator(model, cfg, device)
    return generator


# Datasets
def get_init_dataset(cfg, train=False, out_bool=False, out_float=False, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    
    data_dir = cfg['data']['data_dir']
    point_num = cfg['data']['point_num']
    grid_size = cfg['data']['grid_size']
    pooling_radius = 2 #for pointcloud input
    input_type = cfg['data']['input_type']
    input_points_only = cfg['data']['input_points_only']
    
    
    shapes_3d_dataset = ABC_pointcloud_hdf5(
        data_dir,
        point_num,
        grid_size,
        pooling_radius,
        input_type,
        train,
        out_bool=out_bool,
        out_float=out_float,       
        input_points_only=input_points_only 
    )
 
    return shapes_3d_dataset
