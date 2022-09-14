# import torch.nn as nn
# import torch.nn.functional as F
# from src.layers import ResnetBlockFC
# from src.common import normalize_coordinate, normalize_3d_coordinate, map2local


# class LocalDecoder(nn.Module):
#     ''' Decoder.
#         Instead of conditioning on global features, on plane/volume local features.

#     Args:
#         dim (int): input dimension
#         c_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         n_blocks (int): number of blocks ResNetBlockFC layers
#         leaky (bool): whether to use leaky ReLUs
#         sample_mode (str): sampling feature strategy, bilinear|nearest
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#     '''

#     def __init__(self, out_bool, out_float, dim=3, c_dim=128,
#                  hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
#         super().__init__()
#         self.c_dim = c_dim
#         self.n_blocks = n_blocks
#         self.out_bool = out_bool
#         self.out_float = out_float

#         if c_dim != 0:
#             self.fc = nn.ModuleList([
#                 nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
#             ])

#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for i in range(n_blocks)
#         ])


#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)

#         if self.out_bool:
#             self.pc_conv_out_bool = nn.Linear(hidden_size, 3)      


#     def forward(self, encoded_features, **kwargs):
#         encoded_features = encoded_features.permute(0, 2, 3, 4, 1)
        
#         net = self.fc[0](encoded_features)
#         net = F.leaky_relu(net, negative_slope=0.01, inplace=True)
#         net = self.blocks[0](net)
#         for i in range(1, self.n_blocks):
#             net = net + F.leaky_relu(self.fc[i](encoded_features), negative_slope=0.01, inplace=True)        
#             net = self.blocks[i](net)
            
#         if self.out_bool:
#             net = self.actvn(net)
#             out_bool = self.pc_conv_out_bool(net)

#             return out_bool



# class WithProbeDecoder(nn.Module):
#     ''' Decoder.
#         Instead of conditioning on global features, on plane/volume local features.

#     Args:
#         dim (int): input dimension
#         c_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         n_blocks (int): number of blocks ResNetBlockFC layers
#         leaky (bool): whether to use leaky ReLUs
#         sample_mode (str): sampling feature strategy, bilinear|nearest
#         padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
#     '''

#     def __init__(self, out_bool, out_float, dim=3, c_dim=128,
#                  hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
#         super().__init__()
#         self.c_dim = c_dim
#         self.n_blocks = n_blocks
#         self.out_bool = out_bool
#         self.out_float = out_float

#         if c_dim != 0:
#             self.fc = nn.ModuleList([
#                 nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
#             ])

#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for i in range(n_blocks)
#         ])


#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)

#         if self.out_bool:
#             self.pc_conv_out_bool = nn.Linear(hidden_size, 3)      


#     def forward(self, encoded_features, **kwargs):
#         encoded_features = encoded_features.permute(0, 2, 3, 4, 1)
        
#         net = self.fc[0](encoded_features)
#         net = F.leaky_relu(net, negative_slope=0.01, inplace=True)
#         net = self.blocks[0](net)
#         for i in range(1, self.n_blocks):
#             net = net + F.leaky_relu(self.fc[i](encoded_features), negative_slope=0.01, inplace=True)        
#             net = self.blocks[i](net)
            
#         if self.out_bool:
#             net = self.actvn(net)
#             out_bool = self.pc_conv_out_bool(net)

#             return out_bool