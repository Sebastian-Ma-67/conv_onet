import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from src.data.core import Field
from src.utils import binvox_rw
from src.common import coord2index, normalize_coord


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

# 3D Fields

class PointsField(Field): # 这些点时 query 点吗？
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape. 

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        points_with_occ = {
            'points': points,
            'occ': occupancies,
        }

        if self.transform is not None:
            points_with_occ = self.transform(points_with_occ)

        return points_with_occ

class PointCloudField(Field): # 这些是从 mesh 上采样得到的点
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform_methods (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform_methods=None, multi_files=None):
        self.file_name = file_name
        self.points_transform = transform_methods
        self.multi_files = multi_files

    def load(self, single_object_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            pointcloud_file_path = os.path.join(single_object_path, self.file_name)
        else:
            np.random.seed(0) # 这里我们先让种子固定，方便测试
            num = np.random.randint(self.multi_files)
            pointcloud_file_path = os.path.join(single_object_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(pointcloud_file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)
        
        points_with_normals = {
            'points': points, # 这里为什么起名为none，也没解释清楚，我觉的不好,我还是把他改成‘points’吧
            'normals': normals,
        }

        if self.points_transform is not None:
            points_with_normals = self.points_transform(points_with_normals) # 其实这里我们只是简单地进行另一个随机采样，以及添加了高斯噪声（暂时sigma=0,训练shapenet的时候使用的是0.005）

        return points_with_normals

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
