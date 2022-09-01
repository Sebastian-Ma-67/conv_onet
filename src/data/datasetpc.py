import os
import numpy as np
import time
import h5py

import torch

# from sklearn.neighbors import KDTree
import trimesh
from src.data.utils import read_data,read_and_augment_data_undc,read_data_input_only, write_ply_point


class ABC_pointcloud_hdf5(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_point_num, output_grid_size, pooling_radius, input_type, train, out_bool, out_float, input_only=False):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        # self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.train = train
        self.input_type = input_type
        self.out_bool = out_bool
        self.out_float = out_float
        self.input_only = input_only

        if self.out_bool and self.out_float and self.train:
            print("ERROR: out_bool and out_float cannot both be activated in training")
            exit(-1)

        #self.hdf5_names = os.listdir(self.data_dir)
        #self.hdf5_names = [name[:-5] for name in self.hdf5_names if name[-5:]==".hdf5"]
        #self.hdf5_names = sorted(self.hdf5_names)

        fin = open("abc_obj_list.txt", 'r')
        self.hdf5_names = [name.strip() for name in fin.readlines()] # name.strip() 剔除字符串首尾的空白字符
        fin.close()

        # 初始化 self.hdf5_names 和 self.hdf5_gridsizes
        if self.input_type=="pointcloud":
            if self.train:
                self.hdf5_names = self.hdf5_names[:int(len(self.hdf5_names)*0.8)] # 前80%数据用来训练
                print("Total#", "train", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)
                #separate 32 and 64
                temp_hdf5_names = []
                temp_hdf5_gridsizes = []
                for name in self.hdf5_names:
                    for grid_size in [32,64]: # 这里为什么要有32 和 64 种分辨率呢？
                        temp_hdf5_names.append(name)
                        temp_hdf5_gridsizes.append(grid_size)
                self.hdf5_names = temp_hdf5_names
                self.hdf5_gridsizes = temp_hdf5_gridsizes
            else:
                self.hdf5_names = self.hdf5_names[int(len(self.hdf5_names)*0.8):] # 后20%数据用来测试
                self.hdf5_gridsizes = [self.output_grid_size]*len(self.hdf5_names) # 这里就全部是64的grids了
                print("Total#", "test", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)

            print("Non-trivial Total#", len(self.hdf5_names), ", input type: ", self.input_type, ", bool: ", self.out_bool, ", float: ", self.out_float)


    def __len__(self):
        return len(self.hdf5_names)

    def __getitem__(self, index):
        hdf5_dir = self.data_dir + "/" + self.hdf5_names[index] + ".hdf5"
        if self.input_type=="pointcloud": 
            grid_size = self.hdf5_gridsizes[index]
        elif self.input_type=="noisypc": #augmented data
            grid_size = self.output_grid_size
            shape_scale = self.hdf5_shape_scale[index]


        if self.train:
            gt_output_bool_, gt_output_float_, gt_input_ = read_and_augment_data_undc(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,aug_permutation=True,aug_reversal=True,aug_inversion=False)
        else:
            if self.input_only:
                gt_output_bool_,gt_output_float_,gt_input_ = read_data_input_only(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,is_undc=True)
            else:
                gt_output_bool_,gt_output_float_,gt_input_ = read_data(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,is_undc=True)


        if self.train:
            if self.input_type=="pointcloud": 
                #augment input point number depending on the grid size
                #grid   ideal?  range
                #32     1024    512-2048
                #64     4096    2048-8192
                np.random.shuffle(gt_input_)
                if grid_size==32:
                    count = np.random.randint(512,2048)
                elif grid_size==64:
                    count = np.random.randint(2048,8192)
                gt_input_ = gt_input_[:count]
            elif self.input_type=="noisypc": #augmented data
                #augment input point number depending on the shape scale
                #grid   ideal?  range
                #64     16384    8192-32768
                np.random.shuffle(gt_input_)
                rand_int_s = int(8192*(shape_scale/10.0)**2)
                rand_int_t = int(32768*(shape_scale/10.0)**2)
                count = np.random.randint(rand_int_s,rand_int_t)
                gt_input_ = gt_input_[:count]
        else:
            gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)

        #add Gaussian noise
        if self.input_type=="noisypc": #augmented data
            if not self.train:
                np.random.seed(0)
            gt_input_ = gt_input_ + np.random.randn(gt_input_.shape[0],gt_input_.shape[1]).astype(np.float32)*0.5



        # 接下来这部分，我着重说一下啊，我们不需要kdtree，也不需要KNN，我们只需要input points [Np, 3], 一个 gt_output_bool 的值，这个值应该是 [32, 32, 32], 有可能是33 
        # 然后，我们拿着这部分数据，把他交给convonet，让他训练，然后用 ndc 的方法计算 loss
        pointcloud_data = {}
        pointcloud_data['gt_output_bool'] = gt_output_bool_
        pointcloud_data['input_points'] = gt_input_
        pointcloud_data['gt_output_bool_query_points'] = 
        
        return pointcloud_data





        #point cloud convolution, with KNN
        #basic building block:
        #-for each point
        #-find its K nearest neighbors
        #-and then use their relative positions to perform convolution
        #last layer (pooling):
        #-for each grid cell
        #-if it is within range to the point cloud
        #-find K nearest neighbors to the cell center
        #-and do convolution

        # pc_xyz = gt_input_
        # kd_tree = KDTree(pc_xyz, leaf_size=8)
        # pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)
        # pc_KNN_idx = np.reshape(pc_KNN_idx,[-1])
        # pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        # pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz),self.KNN_num,3]) - np.reshape(pc_xyz,[len(pc_xyz),1,3])
        # pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz)*self.KNN_num,3])
        # #this will be used to group point features

        # #consider all grid cells within range to the point cloud
        # pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        # pc_xyz_int = np.clip(pc_xyz_int,0,grid_size)
        # tmp_grid = np.zeros([grid_size+1,grid_size+1,grid_size+1], np.uint8)
        # tmp_grid[pc_xyz_int[:,0],pc_xyz_int[:,1],pc_xyz_int[:,2]] = 1
        # for ite in range(self.pooling_radius):
        #     tmp_mask = np.copy(tmp_grid[1:-1,1:-1,1:-1])
        #     for i in range(3):
        #         for j in range(3):
        #             for k in range(3):
        #                 tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k] = tmp_mask | tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k]
        # voxel_x,voxel_y,voxel_z = np.nonzero(tmp_grid)
        # voxel_xyz = np.concatenate([np.reshape(voxel_x,[-1,1]),np.reshape(voxel_y,[-1,1]),np.reshape(voxel_z,[-1,1])],1)
        # voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        # voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)

        # voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        # voxel_KNN_idx = np.reshape(voxel_KNN_idx,[-1])
        # voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        # voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz),self.KNN_num,3]) - np.reshape(voxel_xyz,[len(voxel_xyz),1,3])
        # voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz)*self.KNN_num,3])


        # if self.out_bool:
        #     gt_output_bool = gt_output_bool_[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] # 在这里相当于对空间中的 gt 以原始点及其附近点为索引，进行了采样
        #     gt_output_bool = np.ascontiguousarray(gt_output_bool, np.float32)


        # if self.out_float:
        #     gt_output_float = gt_output_float_[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]]
        #     gt_output_float = np.ascontiguousarray(gt_output_float, np.float32)
        #     gt_output_float_mask = (gt_output_float>=0).astype(np.float32)


        # if self.out_bool and self.out_float:
        #     return pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz, gt_output_bool, gt_output_float, gt_output_float_mask
        # elif self.out_bool:
        #     return pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz, gt_output_bool
        # elif self.out_float:
        #     return pc_KNN_idx, pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz, gt_output_float, gt_output_float_mask

        
        
        
        # category = self.datasets[idx]['category'] # '02691156'
        # category_idx = self.metadata[category]['idx'] # 类别索引
        # single_object = self.datasets[idx]['model'] # '80da27a121142718e15a23e1c3d8f46d'


        # single_sample_path = os.path.join(self.dataset_folder, category, single_object)
        # pointcloud_data = {}

        # info = category_idx
        
        # for field_name, field in self.fields.items(): # 这里面的item 有两个，一个是points ，好像是从3d shape 所占用的空间中采样得到的，另外一个是input，好像是从mesh上采样的                
        #     if field_name == 'occ_points':
        #         occ_points_fielder = field # 此时，这个field为PointField
        #         occ_points_field = occ_points_fielder.load(single_sample_path, idx, info)
                
        #         if isinstance(occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'occ'
        #             for k, v in occ_points_field.items():
        #                 pointcloud_data['%s.%s' % (field_name, k)] = v
                
        #     if field_name == 'normal_points':
        #         iou_occ_points_fielder = field # 此时，这个field为PointCloudField
        #         iou_occ_points_field = iou_occ_points_fielder.load(single_sample_path, idx, info)
                
        #         if isinstance(iou_occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'normals'
        #             for k, v in iou_occ_points_field.items():
        #                 pointcloud_data['%s.%s' % (field_name, k)] = v
                        
        #     if field_name == 'iou_occ_points':
        #         iou_occ_points_fielder = field # 此时，这个field为PointCloudField
        #         iou_occ_points_field = iou_occ_points_fielder.load(single_sample_path, idx, info)
                
        #         if isinstance(iou_occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'normals'
        #             for k, v in iou_occ_points_field.items():
        #                 pointcloud_data['%s.%s' % (field_name, k)] = v





#only for testing
class single_shape_pointcloud(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_point_num, output_grid_size, KNN_num, pooling_radius, normalize):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.normalize = normalize

    def __len__(self):
        return 1

    def __getitem__(self, index):
        grid_size = self.output_grid_size

        if self.data_dir.split(".")[-1]=="ply":
            LOD_input = trimesh.load(self.data_dir)
            LOD_input = LOD_input.vertices.astype(np.float32)
        elif self.data_dir.split(".")[-1]=="hdf5":
            hdf5_file = h5py.File(self.data_dir, 'r')
            LOD_input = hdf5_file["pointcloud"][:].astype(np.float32)
            hdf5_file.close()
        else:
            print("ERROR: invalid input type - only support ply or hdf5")
            exit(-1)

        #normalize
        if self.normalize:
            LOD_input_min = np.min(LOD_input,0)
            LOD_input_max = np.max(LOD_input,0)
            LOD_input_mean = (LOD_input_min+LOD_input_max)/2
            LOD_input_scale = np.sum((LOD_input_max-LOD_input_min)**2)**0.5
            LOD_input = LOD_input-np.reshape(LOD_input_mean, [1,3])
            LOD_input = LOD_input/LOD_input_scale

        gt_input_ = (LOD_input+0.5)*grid_size #denormalize

        if len(gt_input_)<self.input_point_num:
            print("WARNING: you specified",str(self.input_point_num),"points as input but the given file only has",str(len(gt_input_)),"points")
        np.random.shuffle(gt_input_)
        gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)

        #write_ply_point(str(index)+".ply", gt_input_)

        pc_xyz = gt_input_
        kd_tree = KDTree(pc_xyz, leaf_size=8)
        pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)
        pc_KNN_idx = np.reshape(pc_KNN_idx,[-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz),self.KNN_num,3]) - np.reshape(pc_xyz,[len(pc_xyz),1,3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz)*self.KNN_num,3])
        #this will be used to group point features

        #consider all grid cells within range to the point cloud
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int,0,grid_size)
        tmp_grid = np.zeros([grid_size+1,grid_size+1,grid_size+1], np.uint8)
        tmp_grid[pc_xyz_int[:,0],pc_xyz_int[:,1],pc_xyz_int[:,2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1,1:-1,1:-1])
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k] = tmp_mask | tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k]
        voxel_x,voxel_y,voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate([np.reshape(voxel_x,[-1,1]),np.reshape(voxel_y,[-1,1]),np.reshape(voxel_z,[-1,1])],1)
        voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)

        voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx,[-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz),self.KNN_num,3]) - np.reshape(voxel_xyz,[len(voxel_xyz),1,3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz)*self.KNN_num,3])

        return pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz




#only for testing
class scene_crop_pointcloud(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_point_num, output_grid_size, KNN_num, pooling_radius, block_num_per_dim, block_padding):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.block_num_per_dim = block_num_per_dim
        self.block_padding = block_padding

        if self.data_dir.split(".")[-1]=="ply":
            LOD_input = trimesh.load(self.data_dir)
            LOD_input = LOD_input.vertices.astype(np.float32)
        elif self.data_dir.split(".")[-1]=="hdf5":
            hdf5_file = h5py.File(self.data_dir, 'r')
            LOD_input = hdf5_file["pointcloud"][:].astype(np.float32)
            hdf5_file.close()
        else:
            print("ERROR: invalid input type - only support ply or hdf5")
            exit(-1)

        #normalize to unit cube for each crop
        LOD_input_min = np.min(LOD_input,0)
        LOD_input_max = np.max(LOD_input,0)
        LOD_input_scale = np.max(LOD_input_max-LOD_input_min)
        LOD_input = LOD_input-np.reshape(LOD_input_min, [1,3])
        LOD_input = LOD_input/(LOD_input_scale/self.block_num_per_dim)
        self.full_scene = LOD_input
        self.full_scene_size = np.ceil(np.max(self.full_scene,0)).astype(np.int32)
        print("Crops:", self.full_scene_size)
        self.full_scene = self.full_scene*self.output_grid_size


    def __len__(self):
        return self.full_scene_size[0]*self.full_scene_size[1]*self.full_scene_size[2]

    def __getitem__(self, index):
        grid_size = self.output_grid_size+self.block_padding*2

        idx_x = index//(self.full_scene_size[1]*self.full_scene_size[2])
        idx_yz = index%(self.full_scene_size[1]*self.full_scene_size[2])
        idx_y = idx_yz//self.full_scene_size[2]
        idx_z = idx_yz%self.full_scene_size[2]

        gt_input_mask_ = (self.full_scene[:,0]>idx_x*self.output_grid_size-self.block_padding) & (self.full_scene[:,0]<(idx_x+1)*self.output_grid_size+self.block_padding) & (self.full_scene[:,1]>idx_y*self.output_grid_size-self.block_padding) & (self.full_scene[:,1]<(idx_y+1)*self.output_grid_size+self.block_padding) & (self.full_scene[:,2]>idx_z*self.output_grid_size-self.block_padding) & (self.full_scene[:,2]<(idx_z+1)*self.output_grid_size+self.block_padding)

        if np.sum(gt_input_mask_)<100:
            return np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32),np.zeros([1],np.float32)
        
        gt_input_ = self.full_scene[gt_input_mask_] - np.array([[idx_x*self.output_grid_size-self.block_padding,idx_y*self.output_grid_size-self.block_padding,idx_z*self.output_grid_size-self.block_padding]], np.float32)

        np.random.shuffle(gt_input_)
        gt_input_ = gt_input_[:self.input_point_num]
        gt_input_ = np.ascontiguousarray(gt_input_)

        #write_ply_point(str(index)+".ply", gt_input_)

        pc_xyz = gt_input_
        kd_tree = KDTree(pc_xyz, leaf_size=8)
        pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num, return_distance=False)
        pc_KNN_idx = np.reshape(pc_KNN_idx,[-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz),self.KNN_num,3]) - np.reshape(pc_xyz,[len(pc_xyz),1,3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz,[len(pc_xyz)*self.KNN_num,3])
        #this will be used to group point features
        
        #consider all grid cells within range to the point cloud
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int,0,grid_size)
        tmp_grid = np.zeros([grid_size+1,grid_size+1,grid_size+1], np.uint8)
        tmp_grid[pc_xyz_int[:,0],pc_xyz_int[:,1],pc_xyz_int[:,2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1,1:-1,1:-1])
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k] = tmp_mask | tmp_grid[i:grid_size-1+i,j:grid_size-1+j,k:grid_size-1+k]
        voxel_x,voxel_y,voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate([np.reshape(voxel_x,[-1,1]),np.reshape(voxel_y,[-1,1]),np.reshape(voxel_z,[-1,1])],1)
        voxel_xyz = voxel_xyz.astype(np.float32)+0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)
            
        voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num, return_distance=False)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx,[-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz),self.KNN_num,3]) - np.reshape(voxel_xyz,[len(voxel_xyz),1,3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz,[len(voxel_xyz)*self.KNN_num,3])

        return pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz


