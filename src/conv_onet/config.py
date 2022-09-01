import torch
import torch.distributions as dist
from torch import nn
import os
from src.encoder import pointnet
from src.conv_onet import models, training
from src.conv_onet import generation
from src import data
from src import config
from src.common import decide_total_volume_range, update_reso
from torchvision import transforms
import numpy as np

'''
get something:
get model, trainer, generator, data_fields


'''
def get_network(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']
    out_bool = cfg['training']['bool']
    out_float = cfg['training']['float']
    
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg['model'].keys():
        encoder_kwargs['local_coord'] = cfg['model']['local_coord']
        decoder_kwargs['local_coord'] = cfg['model']['local_coord']
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']
    

    # decoder = models.decoder_dict[decoder](
    decoder = models.decoder.LocalDecoder(    
        out_bool=out_bool,
        out_float=out_float,
        dim=dim,
        c_dim=c_dim,
        padding=padding,
        **decoder_kwargs
    )

    if encoder is not None:
        # encoder = encoder_dict[encoder](
        encoder = pointnet.LocalPoolPointnet(
            dim=dim,
            c_dim=c_dim,
            padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    model = models.ConvolutionalOccupancyNetwork(
        decoder, 
        encoder, 
        device=device
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer,
        device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=cfg['training']['eval_sample'],
    )

    return trainer


def get_init_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    
    vol_bound = None
    vol_info = None

    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        input_type = cfg['data']['input_type'],
        padding=cfg['data']['padding'],
        vol_info = vol_info,
        vol_bound = vol_bound,
    )
    return generator


def init_points_fields(mode, cfg):
    ''' Returns the data fields. 这里只是返回一个字典，字典里面是初始化好的累的对象，这里并没有真正地获取到 fields 数据

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_subsample = data.PointsSubsample(cfg['data']['points_subsample']) # 这里只是初始化一个类

    points_fields = {}

    points_fields['occ_points'] = data.PointsField(
        cfg['data']['points_file'], 
        points_subsample,
        unpackbits=cfg['data']['points_unpackbits'],
        multi_files=cfg['data']['multi_files']
    ) # 这里也是初始化一个类
            
    if mode in ('val', 'test'):
        points_fields['iou_occ_points'] = data.PointsField(
            cfg['data']['points_iou_file'], # points.npz
            unpackbits=cfg['data']['points_unpackbits'],
            multi_files=cfg['data']['multi_files']
        )

    return points_fields            