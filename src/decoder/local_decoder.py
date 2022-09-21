import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import ResnetBlockFC
from src.common import normalize_coordinate, normalize_3d_coordinate, map2local


class pc_resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        output = self.linear_1(input)
        output = F.leaky_relu(output, negative_slope=0.01)
        output = self.linear_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01)
        return output

class pc_conv(nn.Module):
    def __init__(self, ef_dim):
        super(pc_conv, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim + 3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        output = input
        
        output = self.linear_1(output)
        #[newpointnum*KNN_num,ef_dim]
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        #[newpointnum*KNN_num,ef_dim]
        output = self.linear_2(output)
        #[newpointnum*KNN_num,ef_dim]
        
        output = output.view(-1, self.ef_dim) # 这里要修改一下，但还不知道要修改成什么样子
        #[newpointnum, KNN_num, ef_dim]
        output = torch.max(output, 1)[0] # 相当于pointnet++里面的pooling操作
        #[newpointnum, ef_dim]
        return 

class resnet_block_rec3(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block_rec3, self).__init__()
        self.ef_dim = ef_dim
        self.pc_conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.pc_conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.pc_conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01)
        output = self.pc_conv_2(output)
        output = output + input
        output = F.leaky_relu(output, negative_slope=0.01)
        return output



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
        
        self.c_dim = 128
        self.n_blocks = n_blocks
        self.out_bool = out_bool
        self.out_float = out_float


        if self.c_dim != 0:
            self.fc = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)])


        self.pc_res_1 = pc_resnet_block(self.c_dim)
        self.pc_res_2 = pc_resnet_block(self.c_dim)
        self.pc_res_3 = pc_resnet_block(self.c_dim)
        self.pc_res_4 = pc_resnet_block(self.c_dim)
        self.pc_res_5 = pc_resnet_block(self.c_dim)
        self.pc_res_6 = pc_resnet_block(self.c_dim)
        self.pc_res_7 = pc_resnet_block(self.c_dim)

        self.conv_1 = nn.Conv3d(self.c_dim, self.c_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.c_dim, self.c_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.c_dim, self.c_dim, 3, stride=1, padding=1, bias=True)

        self.conv_4 = nn.Linear(self.c_dim, self.c_dim)
        self.conv_5 = nn.Linear(self.c_dim, self.c_dim)


        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(self.c_dim, 3)


    def forward(self, encoded_features, **kwargs):
        
        encoded_features = encoded_features.permute(0, 2, 3, 4, 1)

        out = self.pc_res_1(encoded_features)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_2(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_3(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_4(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_5(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_6(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_7(out)
        # out = F.leaky_relu(out, negative_slope=0.01)
        
        out = out.permute(0, 4, 1, 2, 3)
        
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = out.permute(0, 2, 3, 4, 1)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01)
        
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01)
        
            
        if self.out_bool:
            out_bool = self.pc_conv_out_bool(out)

            return out_bool
        
class LocalDecoderLarger(nn.Module):
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
        
        self.c_dim = 128
        self.n_blocks = n_blocks
        self.out_bool = out_bool
        self.out_float = out_float


        if self.c_dim != 0:
            self.fc = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.pc_conv_0 = pc_conv(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)



        self.res_1 = resnet_block_rec3(self.ef_dim)
        self.res_2 = resnet_block_rec3(self.ef_dim)
        self.res_3 = resnet_block_rec3(self.ef_dim)
        self.res_4 = resnet_block_rec3(self.ef_dim)
        self.res_5 = resnet_block_rec3(self.ef_dim)
        self.res_6 = resnet_block_rec3(self.ef_dim)
        self.res_7 = resnet_block_rec3(self.ef_dim)
        self.res_8 = resnet_block_rec3(self.ef_dim)


        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(self.c_dim, 3)


    def forward(self, encoded_features, **kwargs):
        
        encoded_features = encoded_features.permute(0, 2, 3, 4, 1)
        
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)
        
        
        
        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        
        out = out.permute(0, 4, 1, 2, 3)
        
        
        out = self.linear_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.linear_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        
            
        if self.out_bool:
            out_bool = self.pc_conv_out_bool(out)

            return out_bool



class WithProbeDecoder(nn.Module):
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
        self.c_dim = 128
        self.n_blocks = n_blocks
        self.out_bool = out_bool
        self.out_float = out_float

        if c_dim != 0:
            self.fc_feature = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_probe = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])


        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if self.out_bool:
            self.pc_conv_out_bool = nn.Linear(hidden_size, 3)      


    def forward(self, input_probes, encoded_features, **kwargs):
        input_probes = input_probes.float()
        encoded_features = encoded_features.permute(0, 2, 3, 4, 1)
        
        net = self.fc_probe(input_probes)
                
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + F.leaky_relu(self.fc_feature[i](encoded_features), negative_slope=0.01)

            net = self.blocks[i](net)        
        
        # net = self.fc_feature[0](encoded_features)
        # net = F.leaky_relu(net, negative_slope=0.01)
        # net = self.blocks[0](net)
        # for i in range(1, self.n_blocks):
        #     net = net + F.leaky_relu(self.fc_feature[i](encoded_features), negative_slope=0.01)        
        #     net = self.blocks[i](net)
            
        if self.out_bool:
            net = self.actvn(net)
            out_bool = self.pc_conv_out_bool(net)

            return out_bool