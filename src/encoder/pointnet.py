import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from src.encoder.unet import UNet
from src.encoder.unet3d import UNet3D


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn_relu = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_grid_features(self, points_raw, features):
        p_normalize = normalize_3d_coordinate(points_raw.clone(), padding=self.padding)
        index = coordinate2index(p_normalize, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = features.new_zeros(points_raw.size(0), self.c_dim, self.reso_grid**3)
        features = features.permute(0, 2, 1)
        fea_grid = scatter_mean(features, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(points_raw.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 32 x reso x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, pos, index, features):
        bs, feature_dim = features.size(0), features.size(2)
        keys = pos.keys()

        features_out = 0
        for key in keys:
            # scatter plane features from points???
            if key == 'grid':
                features_scatter_out, features_scatter_index = self.scatter(features.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3) # 根据点集的features，得到64*64*64个网格的features； [1,32,64^3]
                
            if self.scatter == scatter_max:
                features_scatter = features_scatter_out
            # gather feature back to points
            index_tmp = index[key].expand(-1, feature_dim, -1)
            features_scatter_gather = features_scatter.gather(dim=2, index=index_tmp) # 然后根据点原始的索引，得到原始点索引位置的features [1,32,10000]
            features_out += features_scatter_gather
        return features_out.permute(0, 2, 1)


    def forward(self, points_raw):
        batch_size, T, D = points_raw.size()

        # acquire the index for each point
        coord = {}
        index = {}
        
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(points_raw.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        
        net = self.fc_pos(points_raw)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        features = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(points_raw, features)

        return fea