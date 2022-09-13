from src.encoder import pointnet

encoder_dict = {
    'local_pool_pointnet': pointnet.LocalPoolPointnet
    # 'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    # 'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    # 'voxel_simple_local': voxels.LocalVoxelEncoder,
}
