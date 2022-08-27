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

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, query_points, encoded_features, **kwargs):
        if self.c_dim != 0:
            grid_sampled_features = self.sample_grid_feature(query_points, encoded_features['grid'])

            grid_sampled_features = grid_sampled_features.transpose(1, 2) # [1, 35937, 3]

        query_points = query_points.float() # [1, 35937, 3]
        net = self.fc_p(query_points) # [1, 35937, 32] / 3->32

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](grid_sampled_features) # [1, 35937, 32]

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net)) # [1, 35937, 1]
        out = out.squeeze(-1)  # [1, 35937]

        return out

