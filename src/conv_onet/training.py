import os
from tqdm import trange
import torch
from torch.nn import functional as F
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
        loss, avg_acc_bool, avg_acc_float = self.compute_loss(data) # 将input_data, 输入到网络中，并计算loss
        loss.backward() # 反向传播
        self.optimizer.step() # 更新所有参数

        return loss.item(), avg_acc_bool, avg_acc_float  # 为了防止名字为loss 的tensor 无限制地叠加导致的显存爆炸，我们要是用 loss.item
    
    

    def compute_loss(self, total_points):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        # get input_points
        input_points = total_points.get('input_points').to(device) # 好气哦，他这里好像没有用到法向量
        # 将数据输入到网络中，对数据进行编码，得到 encoded features
        encoded_features = self.model.encode_inputs(input_points)
        pred_output = self.model.decode(encoded_features)
        pred_output_logits = pred_output.logits

        avg_acc_float = 0
        avg_acc_bool = 0
        if self.model.encoder.out_bool:
            # 获取bool的真值
            gt_output_bool = total_points.get('gt_output_bool').to(device)
            loss_i = F.binary_cross_entropy_with_logits(pred_output_logits, gt_output_bool.float(), reduction='none')
            loss = loss_i.mean()
            
            pred_output_bool_probs = pred_output.probs
            acc_bool = torch.mean( gt_output_bool*(pred_output_bool_probs>0.5).float() + (1-gt_output_bool)*(pred_output_bool_probs<=0.5).float() )
            avg_acc_bool = acc_bool.item()
            
        # if self.model.encoder.out_float:
        #     # 获取 float 的真值
        #     gt_output_float = total_points.get('gt_output_float').to(device)
        #     gt_output_float_mask = total_points.get('gt_output_float_mask').to(device)
        #     pred_output_float = pred_output_logits
        
        #     loss_float = torch.sum(( (pred_output_float-gt_output_float)**2 )*gt_output_float_mask )/torch.clamp(torch.sum(gt_output_float_mask),min=1)
        #     loss = loss_float
        #     avg_acc_float = loss_float.item()
            # avg_acc_float_count += 1    
            
        return loss, avg_acc_bool, avg_acc_float
    
    
    def eval_step(self, data, it):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        # threshold = self.threshold
        eval_dict = {}

        input_points = data.get('input_points').to(device)
        gt_output_bool_ = data.get('gt_output_bool').to(device)
        gt_output_float_ =  data.get('gt_output_float').to(device)
        input_name = data.get('input_name')[0]
             
        kwargs = {}   
        # Compute iou
        with torch.no_grad():
            pred_output = self.model(input_points, **kwargs)


        # 先计算一下 val_loss        
        if self.model.encoder.out_bool:
            pred_output_bool_logits = pred_output
            loss_i = F.binary_cross_entropy_with_logits(pred_output_bool_logits.logits, gt_output_bool_.float(), reduction='none')
            # loss_bool = loss_i.sum(dim=(1,2,3,4))
            # loss_bool = loss_bool.mean()
            loss_bool = loss_i.mean()
            
            eval_dict['eval_loss'] = loss_bool.item()
        
        # 然后我们根据预测的output_bool 来生成一下mesh
        pred_output_probs = pred_output.probs
        
        if self.model.encoder.out_bool:
            pred_output_bool_ = pred_output_probs                    
            pred_output_float_ = gt_output_float_
        
        pred_output_bool_numpy = pred_output_bool_[0].detach().cpu().numpy()
        pred_output_float_numpy = pred_output_float_[0].detach().cpu().numpy()
                
        pred_output_bool_numpy = (pred_output_bool_numpy > 0.5).astype(np.int32)
        pred_output_float_numpy = np.clip(pred_output_float_numpy, 0, 1)  
   
        vertices, triangles = dual_contouring_undc_test(pred_output_bool_numpy, pred_output_float_numpy)
        if vertices.shape[0] != 0:         
            print('start write '+input_name)
            write_obj_triangle("samples"+"/test_"+str(it)+'_'+input_name+".obj", vertices, triangles)
            write_ply_point("samples/test_"+str(it)+'_'+input_name+".ply", input_points[0].detach().cpu().numpy())
        else:
            print('validate '+input_name+' failed...')

        return eval_dict 
