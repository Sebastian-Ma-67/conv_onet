import numpy as np
import torch

from src.data.utils import read_data,read_and_augment_data_undc,read_data_input_only


class ABC_pointcloud_hdf5(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_point_num, output_grid_size, pooling_radius, input_type, train, out_bool, out_float, input_points_only=False):
        self.data_dir = data_dir
        self.input_point_num = input_point_num
        self.output_grid_size = output_grid_size
        # self.KNN_num = KNN_num
        # self.pooling_radius = pooling_radius
        self.train = train
        
        self.input_type = input_type
        self.out_bool = out_bool
        self.out_float = out_float
        self.input_only = input_points_only

        if self.out_bool and self.out_float and self.train:
            print("ERROR: out_bool and out_float cannot both be activated in training")
            exit(-1)

        #self.hdf5_names = os.listdir(self.data_dir)
        #self.hdf5_names = [name[:-5] for name in self.hdf5_names if name[-5:]==".hdf5"]
        #self.hdf5_names = sorted(self.hdf5_names)

        fin = open("abc_obj_list.txt", 'r')
        # fin = open("test_lists.txt", 'r')
        
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
                    # for grid_size in [32,64]: # 这里为什么要有32 和 64 种分辨率呢？
                    for grid_size in [self.output_grid_size]: # 这里为什么要有32 和 64 种分辨率呢？ 为了实验，我们先把他改成32                   
                        temp_hdf5_names.append(name)
                        temp_hdf5_gridsizes.append(grid_size)
                self.hdf5_names = temp_hdf5_names
                self.hdf5_gridsizes = temp_hdf5_gridsizes
            else:
                # self.hdf5_names = self.hdf5_names[int(len(self.hdf5_names)*0.8):] # 后20%数据用来测试
                self.hdf5_names = self.hdf5_names[:int(len(self.hdf5_names)*0.8)] # 前80%数据用来训练
                # self.hdf5_names = self.hdf5_names[int(len(self.hdf5_names)*0.01):int(len(self.hdf5_names)*0.0125)] # 后20%数据用来测试
                
                self.hdf5_gridsizes = [self.output_grid_size]*len(self.hdf5_names) # 这里就全部是64的grids了
                print("Total#", "test", len(self.hdf5_names), self.input_type, self.out_bool, self.out_float)

            print("Non-trivial Total#", len(self.hdf5_names), ", input type: ", self.input_type, ", bool: ", self.out_bool, ", float: ", self.out_float)


    def __len__(self):
        return len(self.hdf5_names)

    def __getitem__(self, index):
        pointcloud_data = {}
    
        # hdf5_dir = self.data_dir + "/" + self.hdf5_names[index] + ".hdf5"
        hdf5_dir = self.data_dir + "/" + self.hdf5_names[index%100] + ".hdf5"
        if self.input_type=="pointcloud": 
            # grid_size = self.hdf5_gridsizes[index]
            grid_size = self.hdf5_gridsizes[index%100]

        if self.train:
            gt_output_bool_, gt_output_float_, gt_input_ = read_and_augment_data_undc(hdf5_dir,
                grid_size, self.input_type, 
                self.out_bool, self.out_float,
                # True, True, 
                aug_permutation=False, aug_reversal=False, aug_inversion=False)
        else:
            if self.input_only: # gt_output_bool_ = np.zeros, gt_output_float_ = np.zeros 
                gt_output_bool_, gt_output_float_, gt_input_ = read_data_input_only(hdf5_dir,grid_size,self.input_type,self.out_bool,self.out_float,is_undc=True)
            else:
                gt_output_bool_, gt_output_float_, gt_input_ = read_data(hdf5_dir,grid_size,self.input_type,
                                                                        #  self.out_bool,self.out_float,
                                                                        True, True,
                                                                         is_undc=True)
                pointcloud_data['input_name'] = self.hdf5_names[index]


        if self.train:
            #augment input point number depending on the grid size
            #grid   ideal?  range
            #32     1024    512-2048
            #64     4096    2048-8192
            np.random.shuffle(gt_input_)
            if grid_size==32:
                count = self.input_point_num
                
            gt_input_ = gt_input_[:count]
            
        else:
            gt_input_ = gt_input_[:self.input_point_num]
            
        gt_input_ = np.ascontiguousarray(gt_input_)

        # 接下来这部分，我着重说一下啊，我们不需要kdtree，也不需要KNN，我们只需要input points [Np, 3], 一个 gt_output_bool 的值，这个值应该是 [32, 32, 32], 有可能是33 
        # 然后，我们拿着这部分数据，把他交给convonet，让他训练，然后用 ndc 的方法计算 loss

        

        pointcloud_data['input_points'] = gt_input_
        pointcloud_data['input_probes']  = input_probes???
        if self.out_bool:
            pointcloud_data['gt_output_bool'] = gt_output_bool_[:-1, :-1, :-1, :] # 强行舍弃掉边缘的值，使得其变成32x32x32x3
        if not self.train:
            pointcloud_data['gt_output_float'] = gt_output_float_[:-1, :-1, :-1, :]
                
        return pointcloud_data


