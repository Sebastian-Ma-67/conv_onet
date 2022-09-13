import torch
import torch.nn as nn
from torch import distributions as dist
from src.conv_onet.models import decoder
# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    # 'simple_local_crop': decoder.PatchLocalDecoder,
    'with_probe_decoder': decoder.WithProbeDecoder
}

class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        
        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''

        encoded_features = self.encode_inputs(inputs)
        pred_logits = self.decode(encoded_features, **kwargs)
        return pred_logits

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        features = self.encoder(inputs)

        return features

    def decode(self, encoded_features, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        pred_logits = self.decoder(encoded_features, **kwargs)
        pred_Bernoulli = dist.Bernoulli(logits=pred_logits)
        return pred_Bernoulli

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


# 后处理，用于弥补小洞
def postprocessing(pred_output_bool):
    for t in range(2):

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        #create a quad if meet 3 open edges
        pred_output_bool = torch.max( pred_output_bool, (tmp_int>=3).int() )

        #delete a quad if meet 3 open edges
        pred_output_bool = torch.min( pred_output_bool, (tmp_int<3).int() )


    for t in range(1): #radical

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        #create a quad if meet 2 open edges, only if it helps close a hole, see below code
        pred_output_bool_backup = pred_output_bool
        pred_output_bool = torch.max( pred_output_bool, (tmp_int>=2).int() )

        #open edges
        gridedge_x_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 0]
        gridedge_x_outedge_y_1 = pred_output_bool[:-1, :,   1: , 0]
        gridedge_x_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 0]
        gridedge_x_outedge_z_1 = pred_output_bool[:-1, 1:,  :  , 0]
        gridedge_y_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 1]
        gridedge_y_outedge_x_1 = pred_output_bool[:,   :-1, 1: , 1]
        gridedge_y_outedge_z_0 = pred_output_bool[:-1, :-1, :  , 1]
        gridedge_y_outedge_z_1 = pred_output_bool[1:,  :-1, :  , 1]
        gridedge_z_outedge_x_0 = pred_output_bool[:,   :-1, :-1, 2]
        gridedge_z_outedge_x_1 = pred_output_bool[:,   1:,  :-1, 2]
        gridedge_z_outedge_y_0 = pred_output_bool[:-1, :,   :-1, 2]
        gridedge_z_outedge_y_1 = pred_output_bool[1:,  :,   :-1, 2]
        outedge_x = gridedge_y_outedge_x_0+gridedge_y_outedge_x_1+gridedge_z_outedge_x_0+gridedge_z_outedge_x_1
        outedge_y = gridedge_x_outedge_y_0+gridedge_x_outedge_y_1+gridedge_z_outedge_y_0+gridedge_z_outedge_y_1
        outedge_z = gridedge_x_outedge_z_0+gridedge_x_outedge_z_1+gridedge_y_outedge_z_0+gridedge_y_outedge_z_1
        boundary_x_flag = (outedge_x==1).int()
        boundary_y_flag = (outedge_y==1).int()
        boundary_z_flag = (outedge_z==1).int()

        tmp_int = torch.zeros(pred_output_bool.size(), dtype=torch.int32, device=pred_output_bool.device)
        tmp_int[:,    :-1, :-1, 1] += boundary_x_flag
        tmp_int[:,    :-1, 1: , 1] += boundary_x_flag
        tmp_int[:,    :-1, :-1, 2] += boundary_x_flag
        tmp_int[:,    1:,  :-1, 2] += boundary_x_flag
        tmp_int[:-1, :,    :-1, 0] += boundary_y_flag
        tmp_int[:-1, :,    1: , 0] += boundary_y_flag
        tmp_int[:-1, :,    :-1, 2] += boundary_y_flag
        tmp_int[1:,  :,    :-1, 2] += boundary_y_flag
        tmp_int[:-1, :-1, :   , 0] += boundary_z_flag
        tmp_int[:-1, 1:,  :   , 0] += boundary_z_flag
        tmp_int[:-1, :-1, :   , 1] += boundary_z_flag
        tmp_int[1:,  :-1, :   , 1] += boundary_z_flag

        pred_output_bool = torch.min( pred_output_bool, (tmp_int<2).int() )
        pred_output_bool = torch.max( pred_output_bool, pred_output_bool_backup )

    return pred_output_bool

