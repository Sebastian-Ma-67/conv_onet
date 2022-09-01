import os
import logging
from torch.utils import data
import numpy as np
import yaml
from src.common import decide_total_volume_range, update_reso


logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None, cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
                
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories # 这些参数都不知道有啥用？？？
            } 
        
        # Set index
        for categories_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = categories_idx

        # Get all models
        self.datasets = []
        for categories_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is None:
                self.datasets += [
                    {'category': c, 'model': m} for m in [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '') ]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
                
                if '' in models_c:
                    models_c.remove('')

                self.datasets += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]

            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.datasets)

    def __getitem__(self, idx): # 
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.datasets[idx]['category'] # '02691156'
        category_idx = self.metadata[category]['idx'] # 类别索引
        single_object = self.datasets[idx]['model'] # '80da27a121142718e15a23e1c3d8f46d'


        single_sample_path = os.path.join(self.dataset_folder, category, single_object)
        pointcloud_data = {}

        info = category_idx
        
        for field_name, field in self.fields.items(): # 这里面的item 有两个，一个是points ，好像是从3d shape 所占用的空间中采样得到的，另外一个是input，好像是从mesh上采样的                
            if field_name == 'occ_points':
                occ_points_fielder = field # 此时，这个field为PointField
                occ_points_field = occ_points_fielder.load(single_sample_path, idx, info)
                
                if isinstance(occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'occ'
                    for k, v in occ_points_field.items():
                        pointcloud_data['%s.%s' % (field_name, k)] = v
                
            if field_name == 'normal_points':
                iou_occ_points_fielder = field # 此时，这个field为PointCloudField
                iou_occ_points_field = iou_occ_points_fielder.load(single_sample_path, idx, info)
                
                if isinstance(iou_occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'normals'
                    for k, v in iou_occ_points_field.items():
                        pointcloud_data['%s.%s' % (field_name, k)] = v
                        
            if field_name == 'iou_occ_points':
                iou_occ_points_fielder = field # 此时，这个field为PointCloudField
                iou_occ_points_field = iou_occ_points_fielder.load(single_sample_path, idx, info)
                
                if isinstance(iou_occ_points_field, dict): # 判断是不是字典类型，正常情况下，里面应该是有一个 'points' 和一个'normals'
                    for k, v in iou_occ_points_field.items():
                        pointcloud_data['%s.%s' % (field_name, k)] = v
            if field_name == 'idx':
                idx_field = field.load(single_sample_path, idx, info)
                pointcloud_data[field_name] = idx_field

        if self.transform is not None:
            pointcloud_data = self.transform(pointcloud_data)

        return pointcloud_data
    
    def get_vol_info(self, model_path):
        ''' Get crop information

        Args:
            model_path (str): path to the current data
        '''
        query_vol_size = self.cfg['data']['query_vol_size']
        unit_size = self.cfg['data']['unit_size']
        field_name = self.cfg['data']['pointcloud_file']
        plane_type = self.cfg['model']['encoder_kwargs']['plane_type']
        recep_field = 2**(self.cfg['model']['encoder_kwargs']['unet3d_kwargs']['num_levels'] + 2)

        if self.cfg['data']['multi_files'] is None:
            file_path = os.path.join(model_path, field_name)
        else:
            num = np.random.randint(self.cfg['data']['multi_files'])
            file_path = os.path.join(model_path, field_name, '%s_%02d.npz' % (field_name, num))
        
        points_dict = np.load(file_path)
        p = points_dict['points']
        if self.split == 'train':
            # randomly sample a point as the center of input/query volume
            p_c = [np.random.uniform(p[:,i].min(), p[:,i].max()) for i in range(3)]
            # p_c = [np.random.uniform(-0.55, 0.55) for i in range(3)]
            p_c = np.array(p_c).astype(np.float32)
            
            reso = query_vol_size + recep_field - 1
            # make sure the defined reso can be properly processed by UNet
            reso = update_reso(reso, self.depth)
            input_vol_metric = reso * unit_size
            query_vol_metric = query_vol_size * unit_size

            # bound for the volumes
            lb_input_vol, ub_input_vol = p_c - input_vol_metric/2, p_c + input_vol_metric/2
            lb_query_vol, ub_query_vol = p_c - query_vol_metric/2, p_c + query_vol_metric/2

            input_vol = [lb_input_vol, ub_input_vol]
            query_vol = [lb_query_vol, ub_query_vol]
        else:
            reso = self.total_reso
            input_vol = self.total_input_vol
            query_vol = self.total_query_vol

        vol_info = {'plane_type': plane_type,
                    'reso'      : reso,
                    'input_vol' : input_vol,
                    'query_vol' : query_vol}
        return vol_info
    
    def get_model_dict(self, idx):
        return self.datasets[idx]

    def test_model_complete(self, category, data):
        ''' Tests if model is complete.

        Args:
            data (str): data_name
        '''
        data_path = os.path.join(self.dataset_folder, category, data)
        files = os.listdir(data_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, data_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    def set_num_threads(nt):
        try: 
            import mkl; mkl.set_num_threads(nt)
        except: 
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE']='1'
            for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
