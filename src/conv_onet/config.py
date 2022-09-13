import os
from src.encoder import pointnet
from src.encoder import encoder_dict
from src.decoder import decoder_dict
from src.conv_onet.models import decoder
from src.conv_onet import models, training
from src.conv_onet import generation

'''
get something:
get model, trainer, generator, data_fields


'''
def get_init_network(cfg, device=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    inside_decoder = cfg['model']['decoder']
    inside_encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']
    out_bool = cfg['training']['bool'] or cfg['test']['bool']
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
    
    # network_decoder = decoder.LocalDecoder(    
    #     out_bool=out_bool,
    #     out_float=out_float,
    #     dim=dim,
    #     c_dim=c_dim,
    #     padding=padding,
    #     **decoder_kwargs
    # )

    # network_encoder = pointnet.LocalPoolPointnet(
    #     dim=dim,
    #     c_dim=c_dim,
    #     padding=padding,
    #     **encoder_kwargs
    # )
        
    network_decoder = decoder_dict[inside_decoder](
        out_bool=out_bool, out_float=out_float,
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )
    
    network_encoder = encoder_dict[inside_encoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **encoder_kwargs
    )
        
    model = models.ConvolutionalOccupancyNetwork(
        network_decoder, 
        network_encoder, 
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
        model, 
        optimizer,
        device=device, 
        input_type=input_type,
        vis_dir=vis_dir, 
        threshold=threshold,
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