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

        self.actvn = nn.ReLU()
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


    # def generate_plane_features(self, p, c, plane='xz'):
    #     # acquire indices of features in plane
    #     xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
    #     index = coordinate2index(xy, self.reso_plane)

    #     # scatter plane features from points
    #     fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
    #     c = c.permute(0, 2, 1) # B x 512 x T
    #     fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
    #     fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

    #     # process the plane features with UNet
    #     if self.unet is not None:
    #         fea_plane = self.unet(fea_plane)

    #     return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        # if 'xz' in self.plane_type:
        #     fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        # if 'xy' in self.plane_type:
        #     fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        # if 'yz' in self.plane_type:
        #     fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

# class PatchLocalPoolPointnet(nn.Module):
#     ''' PointNet-based encoder network with ResNet blocks.
#         First transform input points to local system based on the given voxel size. / 首先基于给定的 voxel size 将输入的点云转换到local system
#         Support non-fixed number of point cloud, but need to precompute the index / 支持非固定数量的点云，但是得提前计算索引
    
#     Args:
#         c_dim (int): dimension of latent code c / 这个是啥？
#         dim (int): input points dimension
#         hidden_dim (int): hidden dimension of the network
#         scatter_type (str): feature aggregation when doing local pooling
#         unet (bool): weather to use U-Net / 这肯定得用啊
#         unet_kwargs (str): U-Net parameters
#         unet3d (bool): weather to use 3D U-Net / 那必须是3D
#         unet3d_kwargs (str): 3D U-Net parameters
#         plane_resolution (int): defined resolution for plane feature / 暂时不管，我们先考虑 Volume Encoder
#         grid_resolution (int): defined resolution for grid feature / 似乎默认是32？
#         plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume / 那必须是3D ['grid']
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#         n_blocks (int): number of blocks ResNetBlockFC layers
#         local_coord (bool): whether to use local coordinate
#         pos_encoding (str): method for the positional encoding, linear|sin_cos
#         unit_size (float): defined voxel unit size for local system
#     '''

#     def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
#                  unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
#                  plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, 
#                  local_coord=False, pos_encoding='linear', unit_size=0.1):
#         super().__init__()
#         self.c_dim = c_dim

#         self.res_fc_blocks = nn.ModuleList([
#             ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks) # 
#         ])
#         self.fc_c = nn.Linear(hidden_dim, c_dim)

#         self.actvn = nn.ReLU()
#         self.hidden_dim = hidden_dim
#         self.reso_plane = plane_resolution
#         self.reso_grid = grid_resolution
#         self.plane_type = plane_type
#         self.padding = padding

#         if unet:
#             self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
#         else:
#             self.unet = None

#         self.unet3d = UNet3D(**unet3d_kwargs)

#         self.scatter = scatter_max


#         if local_coord:
#             self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
#         else:
#             self.map2local = None
        
#         if pos_encoding == 'sin_cos':
#             self.fc_pos = nn.Linear(60, 2*hidden_dim)
#         else:
#             self.fc_pos = nn.Linear(dim, 2*hidden_dim)


#     def generate_grid_features(self, index, c):
#         # scatter grid features from points        
#         c = c.permute(0, 2, 1)
#         if index.max() < self.reso_grid**3:
#             fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
#             fea_grid = scatter_mean(c, index, out=fea_grid) # B x c_dim x reso^3
#         else:
#             fea_grid = scatter_mean(c, index) # B x c_dim x reso^3
#             if fea_grid.shape[-1] > self.reso_grid**3: # deal with outliers
#                 fea_grid = fea_grid[:, :, :-1]
#         fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

#         if self.unet3d is not None:
#             fea_grid = self.unet3d(fea_grid)

#         return fea_grid

#     def pool_local(self, index, c):
#         fea_dim = c.size(2)
#         # scatter plane features from points
        
#         fea_out, fea_argmax = self.scatter(c.permute(0, 2, 1), index['grid']) # 将每个小cell中的每个特征值的最大值
#         if self.scatter == scatter_max:
#             fea = fea_out # 这个只取{0}是因为{0}是输出的值，{1}是argmax值，这里我们只需要输出的值
#         # gather feature back to points
#         idx = index['grid'].expand(-1, fea_dim, -1)
#         fea = fea.gather(dim=2, index=idx)
#         c_out = fea
#         return c_out.permute(0, 2, 1)


#     def forward(self, inputs):
#         point = inputs['points']
#         index = inputs['index']
    
#         batch_size, T, D = point.size()

#         if self.map2local:
#             pp = self.map2local(point)
#             net = self.fc_pos(pp)
#         else:
#             net = self.fc_pos(point)

#         net = self.res_fc_blocks[0](net)
#         for block in self.res_fc_blocks[1:]:
#             pooled = self.pool_local(index, net)
#             net = torch.cat([net, pooled], dim=2)
#             net = block(net)

#         c = self.fc_c(net)

#         fea = {}
        
#         fea['grid'] = self.generate_grid_features(index['grid'], c)

#         return fea
