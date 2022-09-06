import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, out_bool, out_float, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_bool = out_bool
        self.out_float = out_float

        if c_dim != 0:
            self.fc = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        # self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])


        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        # self.fc_out = nn.Linear(hidden_size, 1)
        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(32, 3)
        if self.out_float:
            self.pc_conv_out_float = nn.Linear(32, 3)        


        
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, encoded_features, **kwargs):
        encoded_features = encoded_features.permute(0, 2, 3, 4, 1)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                if  i == 0:
                    net = self.fc[0](encoded_features)
                else:
                    net = net + self.fc[i](encoded_features) # [1, 35937, 32]

            net = self.blocks[i](net)
        

        if self.out_bool and self.out_float:
            out_bool = self.pc_conv_out_bool(net)
            out_float = self.pc_conv_out_float(net)
            
            out_bool = out_bool.squeeze(-1)  # [1, 35937]
            out_float = out_float.squeeze(-1)  # [1, 35937]
            
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.pc_conv_out_bool(net)
            out_bool = torch.sigmoid(out_bool)
 
            return out_bool
        elif self.out_float:        
            out_float = self.pc_conv_out_float(net)            

            return out_float

        # return out

