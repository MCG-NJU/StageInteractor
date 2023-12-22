import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply, bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead
from mmdet.core import bbox_overlaps
import math
import numpy as np

import os

DEBUG = 'DEBUG' in os.environ


class SubqueryFeatureSampler(nn.Module):
    IND = 0

    def __init__(self,
                 content_dim,
                 in_points,
                 G_sub_q=4,
                 num_queries=100,
                 featmap_dim=None,
                 subbox_poolsize=9,
                 zero_init=False,
                 progress_filter=False,
                 use_holistic_sampling_points=False,
                 stage_type=False,
                 inhibit_cls=False,
                 stage_idx=None,
                 ):
        super(SubqueryFeatureSampler, self).__init__()
        
        self.featmap_dim = content_dim if featmap_dim is None else featmap_dim
        
        progress_filter = False 
        
        self.G = G_sub_q
        self.in_points = in_points
        self.content_dim = content_dim
        self.subbox_poolsize = subbox_poolsize
        self.zero_init = zero_init
        self.progress_filter = progress_filter
        self.use_holistic_sampling_points = use_holistic_sampling_points
        self.stage_type = stage_type
        self.inhibit_cls = inhibit_cls
        self.stage_idx = stage_idx
        
        if self.use_holistic_sampling_points:
            self.gen_one_d = None
            self.gen_double_d = nn.Sequential(
                nn.Linear(content_dim, 3 * G_sub_q * in_points * subbox_poolsize),
            )
        else:
            self.gen_one_d = nn.Sequential(
                nn.Linear(content_dim, 3 * G_sub_q * in_points),
            )
            self.gen_double_d = nn.Sequential(
                nn.Linear(content_dim, 3 * subbox_poolsize),
            )

        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        if self.gen_one_d is not None:
            nn.init.zeros_(self.gen_one_d[-1].weight)
            nn.init.zeros_(self.gen_one_d[-1].bias)
            bias = self.gen_one_d[-1].bias.data.view(
                self.G, -1, 3)
            bias.mul_(0.0)
            

            bandwidth = 0.5 * 1.0
            
            if self.zero_init:
                nn.init.zeros_(bias[:, :, :-1])
            else:
                nn.init.uniform_(bias[:, :, :-1], -bandwidth, bandwidth)
        
            nn.init.zeros_(self.gen_double_d[-1].weight)
            nn.init.zeros_(self.gen_double_d[-1].bias)
            bias = self.gen_double_d[-1].bias.data.view(
                -1, self.subbox_poolsize, 3)
            bias.mul_(0.0)
            
            bandwidth = 0.5 * 1.0 / math.sqrt(2.)
            
            if self.zero_init:
                nn.init.zeros_(bias[:, :, :-1])
            else:
                nn.init.uniform_(bias[:, 1:, :-1], -bandwidth, bandwidth)
        else:
            nn.init.zeros_(self.gen_double_d[-1].weight)
            nn.init.zeros_(self.gen_double_d[-1].bias)
            bias = self.gen_double_d[-1].bias.data.view(
                self.G, -1, self.subbox_poolsize, 3)
            with torch.no_grad():
                dxdydz00 = bias[:, :, [0], :].clone().detach() # G, P, 1, 4
                bandwidth = 0.5 * 1.0
                nn.init.uniform_(dxdydz00[:, :, :, :-1], -bandwidth, bandwidth)
                dx0 = dxdydz00[..., 0:1]
                dy0 = dxdydz00[..., 1:2]
                dz0 = dxdydz00[..., 2:3]
                
                
                dxdydz01 = bias[:, :, :, :].clone().detach() # G, 1, 9, 4
                bandwidth = 0.5 * 1.0 / math.sqrt(2.)
                nn.init.uniform_(dxdydz01[:, :, 1:, :-1], -bandwidth, bandwidth)
                dx1 = dxdydz01[..., 0:1]
                dy1 = dxdydz01[..., 1:2]
                dz1 = dxdydz01[..., 2:3]
                
                newx = dx0 + 2**dz0 * dx1
                newy = dy0 + 2**dz0 * dy1
                newz = dz0 + dz1
                xyz = torch.cat([newx, newy, newz], -1)
                
                bias.mul_(0.0).add_(xyz)
            
    
    
    def get_subbox_feat_position(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            subbox_feat_xy=None,
            subbox_feat_z=None,
            last_gen_one=None,
        ):
        
        C_map = featmap_list[0].shape[1]
        num_levels = len(featmap_list)
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        if self.use_holistic_sampling_points:
            ori_xyzr = query_box.view(B, N, 1, 1, 4)
            orix = ori_xyzr[..., 0:1]
            oriy = ori_xyzr[..., 1:2]
            oriz = ori_xyzr[..., 2:3]
            orir = ori_xyzr[..., 3:4]
            
            dxdydz = self.gen_double_d(query_content)
            dxdydz = dxdydz.view(B, N, -1, self.subbox_poolsize, 3)
            dx1 = dxdydz[..., 0:1]
            dy1 = dxdydz[..., 1:2]
            dz1 = dxdydz[..., 2:3]
            newx = orix + 2**(oriz - 0.5*orir) * dx1
            newy = oriy + 2**(oriz + 0.5*orir) * dy1
            newz = oriz + dz1
        else:
            ori_xyzr = query_box.view(B, N, 1, 1, 4)
            orix = ori_xyzr[..., 0:1]
            oriy = ori_xyzr[..., 1:2]
            oriz = ori_xyzr[..., 2:3]
            orir = ori_xyzr[..., 3:4]
            
            dxdydz00 = self.gen_one_d(query_content)
            dxdydz00 = dxdydz00.view(B, N, G*P, -1, 3) # B, N, G*P, 1, 4
            dx0 = dxdydz00[..., 0:1]
            dy0 = dxdydz00[..., 1:2]
            dz0 = dxdydz00[..., 2:3]
            
            dxdydz01 = self.gen_double_d(query_content)
            dxdydz01 = dxdydz01.view(B, N, -1, self.subbox_poolsize, 3) # B, N, G*P, 9, 4
            dx1 = dxdydz01[..., 0:1]
            dy1 = dxdydz01[..., 1:2]
            dz1 = dxdydz01[..., 2:3]
            
            newx = orix + 2**(oriz - 0.5*orir) * (dx0 + 2**dz0 * (dx1))
            newy = oriy + 2**(oriz + 0.5*orir) * (dy0 + 2**dz0 * (dy1))
            newz = oriz + dz0 + dz1
        
        grid = torch.cat([newx, newy], -1)
        grid = grid.view(B, N, G, -1, 2)# B, N, G, P*9, 2
        grid = grid.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P*9, 2
        grid = grid.view(B*G, -1, self.subbox_poolsize, 2) # B*G, N*P, 9, 2
        
        
        newz = newz.view(B, N, G, -1, 1)
        newz = newz.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P*9, 2
        newz = newz.view(B*G, -1, self.subbox_poolsize, 1) # B*G, N*P, 9, 1
        
        return grid, newz
    
    def get_subbox_feat(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            grid,
            newz,
        ):

        C_map = featmap_list[0].shape[1]
        num_levels = len(featmap_list)
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        
        featmap_list = [i.reshape(B*G, -1, i.shape[-2], i.shape[-1]) for i in featmap_list]
        # B*G, 64, H, W

        weight_z = self.regress_z(
            newz[..., 0], 
            stride_size=len(featmap_strides), 
            tau=2.0, 
            mask_size=None,
        )
        
        sample_points_lvl_weight_list = weight_z.unbind(-1)
        
        
        if DEBUG:
            torch.save(grid, 
                'demo/grid_{}.pth'.format(self.stage_idx))
            torch.save(weight_z, 
                'demo/weight_z_{}.pth'.format(self.stage_idx))
        
        sample_feature = weight_z.new_zeros(B, G, C_map//G, N, P*self.subbox_poolsize)
        for i in range(num_levels):
            Hk, Wk = featmap_list[i].shape[-2:]
            
            featmap = featmap_list[i] # B*G, 64, H, W
            lvl_weights = sample_points_lvl_weight_list[i]  # B*G, N*P*9
            
            stride = featmap_strides[i]
            mapping_size = featmap.new_tensor(
                [featmap.size(3), featmap.size(2)]) * stride
            mapping_size = mapping_size.view(1, 1, -1)
            
            
            normalized_xyxy = grid / mapping_size 
            
            normalized_xyxy = normalized_xyxy.view(B*G, N, -1, 2)
            normalized_xyxy = normalized_xyxy*2.0-1.0
            sample_feats = F.grid_sample(
                featmap, normalized_xyxy,
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False,
            ) # B*G, C//G, N, P

            sample_feats = sample_feats.view(B, G, C_map//G, N, -1, self.subbox_poolsize) # B, G, C//G, N, P ,9
            lvl_weights = lvl_weights.reshape(B, G, 1, N, sample_feats.shape[4], -1)  # B, G, 1, N, P, 9
            sample_feats *= lvl_weights
            
            sample_feats = sample_feats.view(B, G, C_map//G, N, -1)
            
            
            sample_feature += sample_feats
        
        # B, G, C_map//G, N, P
        sample_feature = sample_feature.permute(0, 3, 1, 4, 2).contiguous()
        #B, N, G, P, C_map//G
        
        grid = grid.view(B, G, N, -1, 2)
        newz = newz.view(B, G, N, -1, 1)
        
        return sample_feature, grid, newz, query_content
    
    def forward(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            subbox_feat_xyz=None,
        ):
        '''
            query_content: B, N, (C = G * Cs)
            query_box: B, N, 4 : x1y1x2y2
            sample_feats: B, N, G, P, C_map//G
        '''

        
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        subbox_feat, subbox_feat_xy, subbox_feat_z = None, None, None
        if subbox_feat_xyz is not None:
            subbox_feat_xy, subbox_feat_z, _, last_gen_one = subbox_feat_xyz
        
        if (not self.progress_filter) or (self.stage_type <= 1):
            last_gen_one = None
        
        grid, newz = \
            self.get_subbox_feat_position(
                featmap_list,
                featmap_strides,
                query_content,
                query_box,
                subbox_feat_xy,
                subbox_feat_z,
                last_gen_one=last_gen_one,
            )
        
        subbox_feat, subbox_feat_xy, \
        subbox_feat_z, query_content = \
            self.get_subbox_feat(
                featmap_list,
                featmap_strides,
                query_content,
                query_box,
                grid, 
                newz,
            )

        if DEBUG:
            torch.save(
                subbox_feat, 'demo/subbox_feat_{}.pth'.format(self.stage_idx))
        
        SubqueryFeatureSampler.IND += 1
        return subbox_feat, query_content
    
    
    def regress_z(self, z, stride_size=4, tau=2.0, mask_size=None):
        def translate_to_linear_weight(ref, stride_size=4, tau=2.0, mask_size=None):
            grid = torch.arange(stride_size, device=ref.device, \
                dtype=ref.dtype).view(*[len(ref.shape)*[1, ]+[-1, ]])

            ref = ref.unsqueeze(-1).clone()
            l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
            if mask_size is not None:
                c = torch.argmax(l2, dim=-1, keepdim=False)

                r = c + mask_size // 2
                r = torch.clamp(r, min=0, max=stride_size-1)
                l = c - mask_size // 2
                l = torch.clamp(l, min=0, max=stride_size-1)
                
                mask_l = F.one_hot(l, num_classes=stride_size)
                mask_r = F.one_hot(r, num_classes=stride_size)
                cumsum_mask_l = torch.cumsum(mask_l, dim=-1)
                cumsum_mask_r = 1 - torch.cumsum(mask_r, dim=-1) + mask_r
                mask = -(cumsum_mask_l + cumsum_mask_r - 2)
                
                l2 = l2.masked_fill(mask.bool(), -np.inf)
                

            weight = torch.softmax(l2, dim=-1)
            return weight
        
        
        sample_points_lvl = z.clone()
        sample_points_lvl_mapped = sample_points_lvl - 3.
        sample_points_lvl_weight = \
            translate_to_linear_weight(
                sample_points_lvl_mapped,
                stride_size=stride_size,
                tau=tau,
                mask_size=mask_size,
            )
        
        return sample_points_lvl_weight