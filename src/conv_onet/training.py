import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist, empty
from src.common import (
    compute_iou, make_3d_grid, add_key,
)
from src.utils import visualize as vis
from src.training import BaseTrainer
import numpy as np
from src.data.utils import dual_contouring_undc_test, write_obj_triangle, write_ply_point

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train() # 将网络设置为训练模式
        self.optimizer.zero_grad() # 将梯度置为0，也就是把loss关于weight的导数置为0；注意，这里对于每个batch都做了这样的操作
        loss = self.compute_loss(data) # 将input_data, 输入到网络中，并计算loss
        loss.backward() # 反向传播
        self.optimizer.step() # 更新所有参数

        return loss.item()
    
    def eval_step(self, data, i):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        gt_output_bool_ = data.get('gt_output_bool').to(device)
        gt_output_float_ = data.get('gt_output_float').to(device)
        input_points = data.get('input_points').to(device)

        # normal_points_points = data.get('normal_points.points', torch.empty(occ_points_points.size(0), 0)).to(device)
        # voxels_occ = data.get('voxels') # 咦，哪里来的 voxel

        # iou_occ_points_points = data.get('iou_occ_points.points').to(device)
        # iou_occ = data.get('iou_occ_points.occ').to(device)
        
        # batch_size = occ_points_points.size(0)

        kwargs = {}
        
        # add pre-computed index
        # normal_points_points = add_key(normal_points_points, data.get('inputs.ind'), 'points', 'index', device=device)
        # add pre-computed normalized coordinates
        # occ_points_points = add_key(occ_points_points, data.get('points.normalized'), 'p', 'p_n', device=device)
        # iou_occ_points_points = add_key(iou_occ_points_points, data.get('points_iou.normalized'), 'p', 'p_n', device=device)

        # Compute iou
        with torch.no_grad():
            pred_output_bool = self.model(input_points, **kwargs)

        pred_output_bool_ = pred_output_bool[0].detach().cpu().numpy()
        pred_output_float_ = gt_output_float_[0].detach().cpu().numpy()
                
        pred_output_bool_numpy = pred_output_bool_
        pred_output_float_numpy = pred_output_float_
        
        pred_output_bool_numpy = (pred_output_bool_numpy>0.5).astype(np.int32)
        pred_output_float_numpy = np.clip(pred_output_float_numpy, 0, 1)
        
        vertices, triangles = dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
        if vertices.shape[0] != 0:         
            write_obj_triangle("samples"+"/test_"+str(i)+".obj", vertices, triangles)
            write_ply_point("samples/test_"+str(i)+".ply", input_points[0].detach().cpu().numpy())
        
        
        # occ_iou_np = (iou_occ >= 0.5).cpu().numpy()
        # occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()

        # iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        # eval_dict['iou'] = iou

        # # Estimate voxel iou
        # if voxels_occ is not None:
        #     voxels_occ = voxels_occ.to(device)
        #     points_voxels = make_3d_grid(
        #         (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, voxels_occ.shape[1:])
        #     points_voxels = points_voxels.expand(
        #         batch_size, *points_voxels.size())
        #     points_voxels = points_voxels.to(device)
        #     with torch.no_grad():
        #         p_out = self.model(points_voxels, normal_points_points,
        #                            sample=self.eval_sample, **kwargs)

        #     voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
        #     occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
        #     iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

        #     eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def compute_loss(self, total_points):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        if self.model.encoder.out_bool:
            # 获取bool的真值
            gt_output_bool = total_points.get('gt_output_bool').to(device)
            # 获取真值查询点
            # gt_output_bool_query_points = total_points.get('gt_output_bool_query_points').to(device)
            
            # get input_points，如果没有，则用torch.empty() 代替
            input_points = total_points.get('input_points').to(device) # 好气哦，他这里好像没有用到法向量
        
        # 将数据输入到网络中，对数据进行编码，得到 encoded features
        encoded_features = self.model.encode_inputs(input_points)

        
        # kwargs = {}
        # General points 根据encoded features ，输入到 decoder中之后，得到预测的值
        
        # logits = self.model.decode(occ_points_points, encoded_features, **kwargs).logits
        # loss_i = F.binary_cross_entropy_with_logits(
        #     logits, occ, reduction='none')
        # loss = loss_i.sum(-1).mean()


        # pred_output = self.model.decode(gt_output_bool_query_points, encoded_features, **kwargs)                
        
        # 加载 gt_bool 或者 gt_float # 计算 loss 
        if self.model.encoder.out_bool:
            gt_output_bool = total_points.get('gt_output_bool').to(device)
            pred_output_bool = encoded_features
            loss_bool = - torch.mean( gt_output_bool*torch.log(torch.clamp(pred_output_bool, min=1e-10)) + (1-gt_output_bool)*torch.log(torch.clamp(1-pred_output_bool, min=1e-10)) )
            acc_bool = torch.mean( gt_output_bool*(pred_output_bool>0.5).float() + (1-gt_output_bool)*(pred_output_bool<=0.5).float() )

            loss = loss_bool
        return loss
