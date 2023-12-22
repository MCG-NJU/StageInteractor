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

from .cascade_dynamic_mixing import CascadeDynamicMixing
from .sub_query_feature_sampler import SubqueryFeatureSampler

DEBUG = 'DEBUG' in os.environ



def relative_pe_interact_area(boxq, boxk, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    '''
        boxq: (N, B, 4), xyxy
        boxk: (M, B, 4), xyxy
    '''
    N, B = boxq.shape[:2]
    M = boxq.shape[0]
    boxq = boxq.view(N, 1, B, 4)
    boxk = boxk.view(1, M, B, 4)
    ans = interact_area(boxq, boxk)
    ans = ans.view(N*M, B, 4)
    ans = position_embedding_query(ans, num_feats, \
        max_box_size=max_box_size, temperature=temperature, ver=ver)
    ans = ans.view(N, M, B, 4, -1)
    return ans

def interact_area(box1, box2):
    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])

    wh = rb - lt
    cxcy = (rb + lt) / 2
    ans = torch.cat([cxcy, wh], -1)
    return ans
    

def distance_interact_area(A, B):
    '''
        F = AxB(A>0,B>0);
            AxB(A<0,B>0); 
            AxB(A>0,B<0); 
            -AxB(A<0,B<0); 
    '''
    relu = nn.ReLU(inplace=False)
    ans = -relu(-A) * relu(-B) + \
        relu(A) * relu(B) + \
        -relu(-A) * relu(B) + \
        -relu(A) * relu(-B)
    return ans
    

def calc_distri_distance(x, temper=2., neg_inf=-7.):
    # x: B, N, C
    B, N = x.shape[:2]
    pz = torch.exp(-0.5*(x*x).sum(-1)) / math.sqrt(2 * math.pi)
    pz = -torch.clamp(pz, min=1e-7).log().mean(-1)
    return pz
    
    

def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)


def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

def position_embedding_key(box, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    token_xyzr = box
    term = token_xyzr.new_tensor(
        [max_box_size, max_box_size, max_box_size, max_box_size]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].cos(), -pos_x[..., 1::2].sin()),
        dim=4).flatten(3)
    return pos_x

def position_embedding_query(box, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    token_xyzr = box
    term = token_xyzr.new_tensor(
        [max_box_size, max_box_size, max_box_size, max_box_size]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=4).flatten(3)
    return pos_x

def position_embedding(box, num_feats, temperature=10000, ver='xyzr'):
    if box.size(-1) == 4 and ver == 'xyzr':
        token_xyzr = box
        term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
        token_xyzr = token_xyzr / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=token_xyzr.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = token_xyzr[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    elif box.size(-1) == 3:
        token_xyzr = box
        term = token_xyzr.new_tensor([1000, 1000, 1]).view(1, 1, -1)
        token_xyzr = token_xyzr / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=token_xyzr.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = token_xyzr[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    elif box.size(-1) == 2:
        term = box.new_tensor([1000, ] * box.size(-1)).view(1, 1, -1)
        box = box / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=box.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = box[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    else:
        term = box.new_tensor([1, ] * box.size(-1)).view(1, 1, -1)
        box = box / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=box.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = box[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
        # B, N, 4, num_feats
    return pos_x


@HEADS.register_module()
class StageInteractorHead(BBoxHead):
    _DEBUG = -1
    
    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 in_points=32,
                 out_points=128,
                 n_heads=4,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 N_scale=4,
                 anchor_point_num=8,
                 anchor_channel=64,
                 roi_inpoint_h = 7,
                 roi_inpoint_w = 7,
                 in_point_sampling_rate = 1,
                 subbox_poolsize = 3,
                 stage_type = 0,
                 stage_idx = None,
                 use_bg_idx_classifier = False, ###!!!
                 ori_focal_loss=True, 
                 use_soft_iou=False, 
                 box_delta_loss=None, 
                 iou_eval_eta=0.5, 
                 use_static_128=False,
                 use_static_S=False,
                 targets_candi_ids=[0, -1, 1],
                 progress_filter=False,
                 use_topk_labels=False,
                 use_thres_filter=False,
                 use_iof=False,
                 use_hard_label=False,
                 soft2hard_label=False,
                 soft2ambigu=False,
                 use_last_anchor=True,
                 gate_res=True,
                 softlabel_cls_score=False,
                 inhibit_cls=False,
                 use_twice_xyzr_iou=False,
                 no_local_S=False,
                 use_last_cls_logits_perstage=0,
                 use_from_gt_perspective=False,
                 use_gaussian_deter=False,
                 last_in_point_num=None,
                 use_last_box_target=False,
                 use_stage0_static_kernel=False,
                 prefix_M_len=None,
                 lim_outpoints_times=None,
                 use_stage0_M=True,
                 use_axis_atten=False,
                 no_previous_filters=False,
                 use_holistic_sampling_points=False,
                 decompose_S=False,
                 abla_use_static_spatial_mix=False,
                 abla_use_static_channel_mix=False,
                 use_pre_feats=False,
                 reuse_st_id=2,
                 share_W_dygen=False, 
                 shrinkS1=False,
                 delete_reg=False,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(StageInteractorHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False
        
        self.use_soft_iou = use_soft_iou
        self.ori_focal_loss = ori_focal_loss
        self.box_delta_loss = build_loss(box_delta_loss) if (box_delta_loss is not None) else None
        self.iou_eval_eta = iou_eval_eta
        self.use_static_128 = use_static_128
        self.use_static_S = use_static_S
        self.targets_candi_ids = targets_candi_ids
        self.progress_filter = progress_filter
        self.use_topk_labels = use_topk_labels
        self.use_thres_filter = use_thres_filter
        self.use_iof = use_iof
        self.use_hard_label = use_hard_label
        self.soft2hard_label = soft2hard_label
        self.soft2ambigu = soft2ambigu
        self.use_last_anchor = use_last_anchor
        self.softlabel_cls_score = softlabel_cls_score
        self.use_twice_xyzr_iou = use_twice_xyzr_iou
        self.use_last_cls_logits_perstage = use_last_cls_logits_perstage
        self.use_from_gt_perspective = use_from_gt_perspective
        self.use_gaussian_deter = use_gaussian_deter
        self.use_last_box_target = use_last_box_target
        self.delete_reg = delete_reg
        
        
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        #if self.loss_cls.use_sigmoid:
        if not use_bg_idx_classifier:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)
        
        self.use_bg_idx_classifier = use_bg_idx_classifier


        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 4)

        self.in_points = in_points
        self.n_heads = n_heads
        self.out_points = out_points
        
        self.stage_type = stage_type
        self.stage_idx = stage_idx
        
        self.exist_multilabel_q_id = False

        
        self.dynamic_conv = CascadeDynamicMixing(
            in_dim=feat_channels, 
            in_points=in_points, 
            p_groups=n_heads, 
            num_queries=num_queries,
            dim_feedforward=feedforward_channels,
            query_dim=content_dim,
            out_points=out_points, 
            out_dim=feat_channels, 
            sampling_rate=in_point_sampling_rate, 
            dropout=dropout,
            subbox_poolsize=subbox_poolsize,
            use_static_128=use_static_128,
            use_static_S=use_static_S,
            progress_filter=progress_filter,
            stage_type=stage_type,
            gate_res=gate_res,
            inhibit_cls=inhibit_cls,
            no_local_S=no_local_S,
            last_in_point_num=last_in_point_num,
            use_stage0_static_kernel=use_stage0_static_kernel,
            stage_idx=stage_idx,
            prefix_M_len=prefix_M_len,
            lim_outpoints_times=lim_outpoints_times,
            use_stage0_M=use_stage0_M,
            use_axis_atten=use_axis_atten,
            no_previous_filters=no_previous_filters,
            decompose_S=decompose_S,
            abla_use_static_spatial_mix=abla_use_static_spatial_mix,
            abla_use_static_channel_mix=abla_use_static_channel_mix,
            use_pre_feats=use_pre_feats,
            reuse_st_id=reuse_st_id,
            share_W_dygen=share_W_dygen,
            shrinkS1=shrinkS1,
        )
            
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        self.iof_tau = nn.Parameter(torch.ones(num_heads, ))
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        
        zero_init = False  
        self.feat_extractor = \
            SubqueryFeatureSampler(
                content_dim,
                in_points,
                G_sub_q=n_heads,
                num_queries=num_queries,
                subbox_poolsize=subbox_poolsize,
                zero_init=zero_init,
                progress_filter=progress_filter,
                use_holistic_sampling_points=use_holistic_sampling_points,
                stage_type=stage_type,
                inhibit_cls=inhibit_cls,
                stage_idx=stage_idx,
            )
        
        self.act = nn.ReLU(inplace=True)

        self.momentum = 5e-4
        self.max_iter_num = math.log(1e-7) / math.log(1 - self.momentum)
        
        self.iou_snyc_statistics = nn.SyncBatchNorm(1, eps=1e-05, momentum=0.9, affine=False)
        nn.init.constant_(self.iou_snyc_statistics.running_mean, iou_eval_eta)
        nn.init.constant_(self.iou_snyc_statistics.running_var, 0)

    @torch.no_grad()
    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(StageInteractorHead, self).init_weights()
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
            if self.use_bg_idx_classifier:
                nn.init.constant_(self.fc_cls.bias[-1], 0.)
            
            if self.use_last_cls_logits_perstage > 0:
                nn.init.constant_(self.fc_cls.bias, 0.)
        

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)

        self.feat_extractor.init_weights()
        self.dynamic_conv.init_weights()
        self.attention.init_weights()

    @auto_fp16()
    def forward(self,
                x,
                xyzr,
                query_content,
                featmap_strides,
                imgs_whwh, 
                feats=None,
                dyconv1_feats=None, 
                bbox_feats=None,
                roialign_func=None,
                ):
        '''
            imgs_whwh: (bs, 4)
            bbox_feats: (B*N, C, Pools, Pools): 200, 256, 7, 7
        '''

        StageInteractorHead._DEBUG += 1

        P = self.in_points
        G = self.n_heads
        B, N = query_content.shape[:2]
        xyzr = xyzr.reshape(B, N, 4)
        
        cur_last_query = query_content
        
        with torch.no_grad():
            pe = position_embedding(xyzr, query_content.size(-1) // 4)
            rois = decode_box(xyzr)
            roi_box_batched = rois.view(B, N, 4)
            roi_wh_batched = roi_box_batched[..., 2:] - roi_box_batched[..., :2]
            roi_whwh_batched = torch.cat([roi_wh_batched, roi_wh_batched], -1)
            
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]
        
        
        x_iof = iof
        iof = (iof + 1e-7).log()
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)) # B, N_head=8, N, N
        
        query_content_attn = query_content + pe
        query_content = query_content_attn
        
        query_content_attn = query_content_attn.permute(1, 0, 2)
        query_content = self.attention(
            query_content_attn,
            key=query_content_attn,
            value=query_content_attn,
            attn_mask=attn_bias.flatten(0, 1),
            identity=query_content_attn,
        )
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)
    
        
        last_feats, subbox_feat_xyz, last_dyconv_gen_vec, last_query = None, None, None, None
        
        dyconv_gen_vec = query_content
        
        ''' adaptive 3D sampling and mixing '''
        feats, dyconv_gen_vec = \
            self.feat_extractor(
                x,
                featmap_strides,
                dyconv_gen_vec,
                xyzr,
                None,
            )
        
        query_content, query_content_cls, dyconv1_feats = \
            self.dynamic_conv(
                feats, 
                query_content, 
                dyconv_pack=dyconv1_feats,
                last_feats=last_feats,
                last_dyconv_gen_vec=last_dyconv_gen_vec,
                last_query=last_query,
            )

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content))
        
        query_content = query_content.view(B, N, -1)
        
        
        cls_feat = query_content
        reg_feat = query_content
        
        
        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fc_cls(cls_feat).view(B, cls_feat.shape[1], -1)
        
        
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        box_delta = self.fc_reg(reg_feat).view(B, reg_feat.shape[1], -1)

        return cls_score, box_delta, query_content.view(B, N, -1), dyconv1_feats

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        r = xyzr[..., 3:4]
        new_x = xyzr[..., 0:1] + xyzr_delta[..., 0:1] * (2 ** (z - 0.5*r))
        new_y = xyzr[..., 1:2] + xyzr_delta[..., 1:2] * (2 ** (z + 0.5*r))
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_x, new_y, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr
    

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             imgs_whwh=None,
             reduction_override=None, 
             box_delta=None, 
             bbox_targets_candidates=None,
             detach_new_xyzr=None,
             stage=None,
             gt2predid_in_all_stage_list=None,
             pred2gtid_in_fg_stage_list=None,
             all_stage_ret_costs_list=None,
             iou_snyc_statistics=None,
             **kwargs):
        
        ori_imgs_whwh = imgs_whwh
        
        losses = dict()
        bg_class_ind = self.num_classes 
        
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = ((labels >= 0) & (labels < bg_class_ind)).sum().float() 
        avg_factor = (1 * pos_inds).sum().clamp(min=1).float()
        
        thres_eta = None
        if self.use_from_gt_perspective and bbox_targets_candidates is not None and cls_score.numel() > 0:
            labels_new, label_weights, \
            iou_target, iou_bbox_weights, \
            iou_target_imgs_whwh, pos_bbox_pred_imgs_whwh, \
            avg_factor_cls, avg_factor_iou, \
            pos_bbox_pred, imgs_whwh, thres_eta = \
                self.target_list_from_gt_perspective_pre(
                    cls_score, 
                    bbox_pred,
                    labels,
                    label_weights,
                    bbox_targets,
                    bbox_weights,
                    bbox_targets_candidates,
                    imgs_whwh,
                    box_delta,
                    detach_new_xyzr,
                    avg_factor=avg_factor,
                    eval_eta=self.iou_eval_eta,
                    stage=stage,
                    gt2predid_in_all_stage_list=gt2predid_in_all_stage_list,
                    pred2gtid_in_fg_stage_list=pred2gtid_in_fg_stage_list,
                    all_stage_ret_costs_list=all_stage_ret_costs_list,
                    iou_snyc_statistics=iou_snyc_statistics,
                )


        if cls_score is not None:
            if cls_score.numel() > 0:
                
                if bbox_targets_candidates is None:
                    if self.ori_focal_loss:
                        labels_new = labels
                        avg_factor_cls = avg_factor
                
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels_new,
                    label_weights,
                    avg_factor=avg_factor_cls,
                    reduction_override=reduction_override)

                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if bbox_targets_candidates is None:
                    pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                4)[pos_inds.type(torch.bool)]
                    imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                            4)[pos_inds.type(torch.bool)]
                    
                    iou_target = bbox_targets[pos_inds.type(torch.bool)]
                    iou_bbox_weights = bbox_weights[pos_inds.type(torch.bool)]
                    avg_factor_iou = avg_factor
                    
                    pos_bbox_pred_imgs_whwh = pos_bbox_pred / imgs_whwh
                    iou_target_imgs_whwh = iou_target / imgs_whwh
                
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred_imgs_whwh,
                    iou_target_imgs_whwh,
                    iou_bbox_weights,
                    avg_factor=avg_factor_iou)
                
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    iou_target,
                    iou_bbox_weights,
                    avg_factor=avg_factor_iou)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        
        return losses, thres_eta
    
    def get_pad_statistics_per_stage(self, 
            all_stage_ret_costs_list, 
            gt2predid_in_all_stage_list, 
            last_stage, 
            cls_score, 
            B, 
            N, 
            stage_num, 
            cost_type='cls',
            min_cost_val=-100,
        ):
        
        s_query = 0
        stage_fg_min_val = cls_score.new_zeros(B*N)
        stage_bg_max_val = cls_score.new_zeros(B*N)
        lst_cost_batchlist = all_stage_ret_costs_list[last_stage] # cost: last id, last q ###!!!!
        lst_pred_inds = gt2predid_in_all_stage_list[last_stage]
        for b, lst_cost in enumerate(lst_cost_batchlist):
            
            cls_lst_id_lst_q_cost, all_cls_lst_id_lst_q_cost, lst_pred_id = self.select_cost_type(
                lst_cost, lst_pred_inds, s_query, N,
                cost_type=cost_type,
            )
            
            if cls_lst_id_lst_q_cost.size(-1) > 0: 
                stage_fg_min_val[s_query:s_query+N] = cls_lst_id_lst_q_cost.min()
                
                nofg_pred_cls_score = all_cls_lst_id_lst_q_cost.clone().detach()
                nofg_pred_cls_score[lst_pred_id] = min_cost_val
                stage_bg_max_val[s_query:s_query+N] = nofg_pred_cls_score.max()
            
            s_query += N
        
        return stage_fg_min_val, stage_bg_max_val
    
    def get_pad_cost(self, lst_id_cur_q_cost_list, lst_pred_inds, pad_len, min_cost_val):
        lst_id_cur_q_cost_list = torch.cat(lst_id_cur_q_cost_list, 0)
        cost_pad_zeros = lst_id_cur_q_cost_list.new_zeros(pad_len).view(-1) + min_cost_val
        pad_match_cost = cost_pad_zeros.clone().detach()
        pad_match_cost[lst_pred_inds] = lst_id_cur_q_cost_list
        return pad_match_cost
    
    def select_cost_type(self, 
            lst_cost,
            lst_pred_inds,
            s_query,
            N,
            cost_type='iou',
            targets_cands=None,
            idx=None,
            pred_boxes=None,
            last_stage=None,
            cur_stage=None,
            bbox_targets=None,
        ):
        
        match_cost, cls_cost, reg_cost, iou_cost = lst_cost
        lst_pred_id = lst_pred_inds[(lst_pred_inds>=s_query)&(lst_pred_inds<s_query+N)] - s_query
        gt_idx = torch.arange(lst_pred_id.shape[0], device=lst_pred_id.device)
        
        if cost_type == 'iou':
            if last_stage != cur_stage:
                last_labels, last_label_weights, last_bbox_targets, last_bbox_weights = targets_cands[idx]
                t_box = last_bbox_targets[s_query:s_query+N]
            else:
                t_box = bbox_targets[s_query:s_query+N]
            
            p_box = pred_boxes[s_query:s_query+N]
            iou_cost = bbox_overlaps(t_box, p_box, mode='iou', is_aligned=True).view(-1)
            match_cost = -iou_cost
        else:
            if cost_type == 'loc':
                match_cost = match_cost - 2*cls_cost
            
            elif cost_type == 'cls':
                match_cost = cls_cost
            
            elif cost_type == 'giou':
                match_cost = iou_cost
            
            elif cost_type == 'reg':
                match_cost = reg_cost

            elif cost_type == 'match':
                match_cost = match_cost
                
            else:
                assert False, 'No such cost type'

            lst_pred_id = (lst_pred_id, gt_idx)
        
        match_cost = -match_cost
        ret_cost = match_cost[lst_pred_id]
        return ret_cost, match_cost, lst_pred_id
    
    def get_stagewise_costs(self, 
             cls_score,
             pred_boxes,
             all_stage_ret_costs_list, 
             gt2predid_in_all_stage_list, 
             pred2gtid_in_fg_stage_list,
             targets_stage_list, 
             targets_cands,
             min_cost_val,
             cur_stage,
             labels,
             bbox_targets,
             last_id=True,
             last_cost=True,
             cost_type='iou',
             ret_cost=True,
            ):
        
        bg_class_ind = self.num_classes
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        avg_factor_cls = (1 * pos_inds).sum().float()
        avg_factor_reg = (1 * pos_inds).sum().float()
        
        cost_type_loc = cost_type 
        
        stage_lst_id_cur_q_cost_list = [] 
        stage_lst_id_cur_q_cls_cost_list = []

        future_idx_list = []

        for idx, last_stage in enumerate(targets_stage_list):
            
            if last_stage > cur_stage:
                future_idx_list.append(idx)
            
            cur_cost_batchlist = all_stage_ret_costs_list[cur_stage] # cost: last id, last q ###!!!!
            cur_pred_inds = gt2predid_in_all_stage_list[cur_stage]
            
            lst_cost_batchlist = all_stage_ret_costs_list[last_stage] # cost: last id, last q ###!!!!
            lst_pred_inds = gt2predid_in_all_stage_list[last_stage]
            
            avg_factor_cls = min(avg_factor_cls, lst_pred_inds.sum().float())
            avg_factor_reg = min(avg_factor_reg, lst_pred_inds.sum().float())
            
            
            lst_id_cur_q_cost_list = []
            cls_lst_id_cur_q_cost_list = []
            s_query = 0
            N = pred_boxes.shape[0] // len(lst_cost_batchlist)
            for b, (cur_cost, lst_cost) in enumerate(zip(cur_cost_batchlist, lst_cost_batchlist)):
                
                lst_id_cur_q_cost, _, _ = self.select_cost_type(
                    cur_cost, lst_pred_inds, s_query, N,
                    cost_type=cost_type_loc,
                    targets_cands=targets_cands,
                    idx=idx,
                    pred_boxes=pred_boxes,
                    last_stage=last_stage,
                    cur_stage=cur_stage,
                    bbox_targets=bbox_targets,
                )
                cls_lst_id_cur_q_cost, _, _ = self.select_cost_type(
                    cur_cost, lst_pred_inds, s_query, N,
                    cost_type='cls',
                )
                

                lst_id_cur_q_cost_list.append(lst_id_cur_q_cost)
                cls_lst_id_cur_q_cost_list.append(cls_lst_id_cur_q_cost)
                s_query += N
            

            lst_id_cur_q_cost_list = self.get_pad_cost(
                lst_id_cur_q_cost_list, lst_pred_inds, labels.shape[0], min_cost_val)
            stage_lst_id_cur_q_cost_list.append(lst_id_cur_q_cost_list)
            
            cls_lst_id_cur_q_cost_list = self.get_pad_cost(
                cls_lst_id_cur_q_cost_list, lst_pred_inds, labels.shape[0], min_cost_val)
            stage_lst_id_cur_q_cls_cost_list.append(cls_lst_id_cur_q_cost_list)
        
        return stage_lst_id_cur_q_cost_list, stage_lst_id_cur_q_cls_cost_list, \
            avg_factor_cls, avg_factor_reg
        
    
    def target_list_from_gt_perspective_pre(self, 
             cls_score, 
             pred_boxes,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             bbox_targets_candidates,
             imgs_whwh,
             box_delta,
             detach_new_xyzr,
             avg_factor=None,
             eval_eta=0.5,
             stage=None,
             gt2predid_in_all_stage_list=None,
             pred2gtid_in_fg_stage_list=None,
             all_stage_ret_costs_list=None,
             iou_snyc_statistics=None,
            ):
        
        assert bbox_targets_candidates is not None
        box_delta_list, targets_cands, predbox_cands, cls_cands, targets_stage_list = bbox_targets_candidates
        
        with torch.no_grad():
            pred_cls_score = torch.sigmoid(cls_score.clone().detach())
        
        thres_eta = None
        bg_class_ind = self.num_classes
        iou_mode = 'iof' if self.use_iof else 'iou'
        eval_eta = eval_eta if (self.use_thres_filter or self.use_from_gt_perspective) else 1e-7
        min_cost_val = min(0, eval_eta)
        if all_stage_ret_costs_list is not None:
            min_cost_val = min(min_cost_val, -100)
            all_stage_ret_costs_list, stage_statistics_list = all_stage_ret_costs_list
            
            with torch.no_grad():
                
                stage_lst_id_cur_q_cost_list, stage_lst_id_cur_q_cls_cost_list, \
                avg_factor_cls, avg_factor_reg = \
                    self.get_stagewise_costs(
                        cls_score,
                        pred_boxes,
                        all_stage_ret_costs_list, 
                        gt2predid_in_all_stage_list, 
                        pred2gtid_in_fg_stage_list,
                        targets_stage_list + [stage, ], 
                        targets_cands,
                        min_cost_val,
                        stage,
                        labels,
                        bbox_targets,
                    )
                cur_fg_min_val, cur_bg_max_val = stage_statistics_list[stage]

        
        with torch.no_grad():
            new_label = labels.clone().detach()
            new_label_weights = label_weights.clone().detach()
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            soft_label = pred_boxes.new_zeros(labels.shape[0], self.num_classes) 
            soft_label[pos_inds, labels[pos_inds]] = 1. 
            total_pos_inds = pos_inds

            ori_soft_label = soft_label.clone().detach()
            
            if all_stage_ret_costs_list is not None:
                soft_label = pred_boxes.new_zeros(labels.shape[0], self.num_classes) + min_cost_val
                iou = stage_lst_id_cur_q_cost_list[-1].clone().detach()
            else:
                iou = bbox_overlaps(bbox_targets, pred_boxes, mode=iou_mode, is_aligned=True)
            
            iou = iou.view(-1)
            soft_label[pos_inds, labels[pos_inds]] = iou[pos_inds]
            
            ori_cost = iou.clone().detach()
            
            new_bbox_targets = bbox_targets.clone().detach()
            new_bbox_weights = bbox_weights.clone().detach()
            soft_reg_label = torch.zeros_like(labels).type_as(pred_boxes)
            soft_reg_label[pos_inds] = 1.
            ori_reg_label = soft_reg_label.clone().detach()
            if self.softlabel_cls_score:
                soft_reg_label[pos_inds] = pred_cls_score[pos_inds, labels[pos_inds]]
            
            replaced_gt_mask = (torch.zeros_like(gt2predid_in_all_stage_list[stage]) > 0)
            future_soft_reg_label = torch.zeros_like(soft_reg_label).type_as(pred_boxes)
            

        if all_stage_ret_costs_list is not None:
            record_query_chain = gt2predid_in_all_stage_list[stage].clone().detach() #[5, 8, 1, 101, 109, 120]
            max_cost_cur_id_cur_q = stage_lst_id_cur_q_cost_list[-1].clone().detach()
            cls_cost_cur_id_cur_q = stage_lst_id_cur_q_cls_cost_list[-1].clone().detach()
        
        gt2predid_in_all = gt2predid_in_all_stage_list[stage].clone().detach()
        
        with torch.no_grad():
            if self.use_gaussian_deter:
                
                pos_iou_counts_label = soft_label.view(-1).clone().detach()
                select_pos_iou_counts_label = pos_iou_counts_label[pos_iou_counts_label > min_cost_val]
                
                pos_iou_counts_label = pos_iou_counts_label[0] if len(select_pos_iou_counts_label) == 0 else select_pos_iou_counts_label
                
                pos_iou_counts_label = pos_iou_counts_label.view(-1, 1)
                m_pos_iou_counts_label = self.iou_snyc_statistics(pos_iou_counts_label)
                
                pos_iou_mu = self.iou_snyc_statistics.running_mean
                pos_iou_std = self.iou_snyc_statistics.running_var.clamp(min=1e-05) ** 0.5
                
                thres_eta = pos_iou_mu
                thres_eta = 0.5 
            
            flg = True
            
            for idx, (targets, lst_predboxes, lst_clslogits) in enumerate(zip(targets_cands, predbox_cands, cls_cands)):
                last_labels, last_label_weights, last_bbox_targets, last_bbox_weights = targets
                last_predbox = lst_predboxes.view(-1, 4)
                last_cls = lst_clslogits.view(-1, lst_clslogits.size(-1))
                last_pred_cls_score = torch.sigmoid(last_cls.clone().detach()).view_as(pred_cls_score)
                
                last_stage = targets_stage_list[idx]
                last_gt2predid_in_all = gt2predid_in_all_stage_list[last_stage]
                last_pred2gtid_in_all = pred2gtid_in_fg_stage_list[last_stage]
                
                last_thres = iou_snyc_statistics[last_stage]
                
                
                last_pos_inds = (last_labels >= 0) & (last_labels < bg_class_ind)
                label_change_inds = last_pos_inds & (~pos_inds)
                
                
                last_iou_with_curbox = bbox_overlaps(last_bbox_targets, pred_boxes, mode=iou_mode, is_aligned=True)
                last_iou_with_curbox = last_iou_with_curbox.view(-1)  
                
                if all_stage_ret_costs_list is not None: 
                    cost_last_id_cur_q = stage_lst_id_cur_q_cost_list[idx].clone().detach() 
                    cls_cost_last_id_cur_q = stage_lst_id_cur_q_cls_cost_list[idx].clone().detach() 
                    
                    score_select = cost_last_id_cur_q 
            
                if all_stage_ret_costs_list is None:
                    unpermute_query_id = gt2predid_in_all[gt2predid_in_all == last_gt2predid_in_all]
                    assert (unpermute_query_id.shape[0]==0) or \
                        (labels[gt2predid_in_all] == last_labels[last_gt2predid_in_all]).all(), \
                        '{}, {}, {}'.format(unpermute_query_id, \
                        labels[gt2predid_in_all], last_labels[last_gt2predid_in_all])
                    last_permute_query_id = last_gt2predid_in_all[gt2predid_in_all != last_gt2predid_in_all]
                    permute_query_id = gt2predid_in_all[gt2predid_in_all != last_gt2predid_in_all]
                
                if stage < last_stage:
                    if all_stage_ret_costs_list is not None:
                        
                        last_better_id_offset = \
                          (cost_last_id_cur_q[last_gt2predid_in_all] >= thres_eta)
                        last_better_id_relax = last_gt2predid_in_all[last_better_id_offset]
                        
                        last_better_id_offset = \
                          (cost_last_id_cur_q[last_gt2predid_in_all] >= thres_eta) \
                          & (cls_cost_last_id_cur_q[last_gt2predid_in_all] >= cur_fg_min_val[last_gt2predid_in_all])
                        last_better_id = last_gt2predid_in_all[last_better_id_offset]
                        
                        last_better_id_relax = last_better_id 
                        
                        
                        update_last_better_offset = (score_select[last_better_id] > \
                            soft_label[last_better_id, last_labels[last_better_id]])
                        update_last_better_id = last_better_id[update_last_better_offset]
                        soft_reg_label[update_last_better_id] = 1.
                        future_soft_reg_label[update_last_better_id] = 1.
                        new_bbox_targets[update_last_better_id] = last_bbox_targets[update_last_better_id]
                        new_bbox_weights[update_last_better_id] = last_bbox_weights[update_last_better_id]
                        
                        replaced_gt_mask[last_pred2gtid_in_all[update_last_better_id]] = True

                        update_last_better_offset = (score_select[last_better_id_relax] > \
                            soft_label[last_better_id_relax, last_labels[last_better_id_relax]])
                        update_last_better_id = last_better_id_relax[update_last_better_offset]
                        soft_label[update_last_better_id, last_labels[update_last_better_id]] = \
                            score_select[update_last_better_id]
                    else:
                        
                        if self.use_last_box_target:
                            if last_permute_query_id.shape[0] > 0:
                                changed_query_mask = last_iou_with_curbox[last_permute_query_id] > \
                                    (soft_label[last_permute_query_id, :].max(-1)[0] + 1e-7)
                                changed_query_mask = changed_query_mask & (ori_reg_label[last_permute_query_id] < 1)
                                changed_query_id = last_permute_query_id[changed_query_mask]
                                new_bbox_targets[changed_query_id] = last_bbox_targets[changed_query_id]
                                new_bbox_weights[changed_query_id] = last_bbox_weights[changed_query_id]
                                soft_reg_label[changed_query_id] = last_iou_with_curbox[changed_query_id]
                        
                        
                        soft_label[unpermute_query_id, labels[unpermute_query_id]] = \
                            torch.maximum(last_iou_with_curbox[unpermute_query_id], \
                                soft_label[unpermute_query_id, labels[unpermute_query_id]])
                        soft_label[last_permute_query_id, last_labels[last_permute_query_id]] = \
                            torch.maximum(last_iou_with_curbox[last_permute_query_id], \
                                soft_label[last_permute_query_id, last_labels[last_permute_query_id]])
            
                elif stage > last_stage:
                    
                    if all_stage_ret_costs_list is not None:
                        
                        last_better_id_offset = \
                          (score_select[last_gt2predid_in_all] >= thres_eta)
                        last_better_id_relax = last_gt2predid_in_all[last_better_id_offset]
                        
                        last_better_id_offset = \
                          (score_select[last_gt2predid_in_all] >= thres_eta) \
                          & (cls_cost_last_id_cur_q[last_gt2predid_in_all] >= cur_fg_min_val[last_gt2predid_in_all]) 
                        last_better_id = last_gt2predid_in_all[last_better_id_offset]
                        
                        last_better_id_relax = last_better_id
                        

                        update_last_better_offset = (score_select[last_better_id_relax] >= \
                            soft_label[last_better_id_relax, last_labels[last_better_id_relax]])
                        update_last_better_id = last_better_id_relax[update_last_better_offset]
                        soft_label[update_last_better_id, last_labels[update_last_better_id]] = \
                            score_select[update_last_better_id]

                        if last_stage == stage-1:
                            cur_worse_id_offset = \
                              (max_cost_cur_id_cur_q[gt2predid_in_all] <  last_thres)
                            
                            cur_worse_id = gt2predid_in_all[cur_worse_id_offset]
                            soft_label[cur_worse_id, labels[cur_worse_id]] = min_cost_val
                        
                    else:
                        
                        soft_label[unpermute_query_id, labels[unpermute_query_id]] = \
                            torch.maximum(last_iou_with_curbox[unpermute_query_id], \
                                soft_label[unpermute_query_id, labels[unpermute_query_id]])
                        
                        soft_label[last_permute_query_id, last_labels[last_permute_query_id]] = \
                            torch.maximum(last_iou_with_curbox[last_permute_query_id], \
                                soft_label[last_permute_query_id, last_labels[last_permute_query_id]])
                else:
                    if stage > 0 and all_stage_ret_costs_list is not None:
                        last_stage = stage-1
                        last_thres = iou_snyc_statistics[last_stage]
                        
                        cur_worse_id_offset = torch.where(
                          (max_cost_cur_id_cur_q[gt2predid_in_all] <  last_thres)
                        )[0]
                        cur_worse_id = gt2predid_in_all[cur_worse_id_offset]
                        soft_label[cur_worse_id, labels[cur_worse_id]] = min_cost_val
                
                new_label[label_change_inds] = last_labels[label_change_inds]
                new_label_weights[label_change_inds] = last_label_weights[label_change_inds]
                
                total_pos_inds = total_pos_inds | label_change_inds
        
        topk_soft_label = None
        if self.use_topk_labels:
            soft_label_topk_val, soft_label_topk_id = soft_label.view(-1).topk(pos_inds.sum())
            topk_soft_label = soft_label.new_zeros(soft_label.view(-1).shape)
            topk_soft_label[soft_label_topk_id] = soft_label_topk_val
            topk_soft_label = topk_soft_label.view(ori_soft_label.shape)
        
        if self.delete_reg:
            future_soft_reg_label[gt2predid_in_all[~replaced_gt_mask]] = 1 
            soft_reg_label = future_soft_reg_label
        
        thres_soft_label = None
        if self.use_thres_filter:
            thres_soft_label = 1. * (soft_label >= eval_eta)
            if self.use_last_box_target:
                soft_reg_label = 1. * (soft_reg_label >= eval_eta)
        
        
        eval_eta = min_cost_val
        reg_eval_eta = 0
        
        use_gaussian_deter_label = None
        if self.use_gaussian_deter: 
            
            bg_mask = soft_label <= min_cost_val 
            fg_mask = ~bg_mask
            
            soft_label[bg_mask] = eval_eta
            soft_label[fg_mask] = max(eval_eta + 1, 1)
        
        if (topk_soft_label is not None) or (thres_soft_label is not None):
            soft_label = 0 * soft_label
        
        if topk_soft_label is not None:
            soft_label = torch.maximum(soft_label, topk_soft_label)
            
        if thres_soft_label is not None:
            soft_label = torch.maximum(soft_label, thres_soft_label)
        
        
        soft_label = 1. * (soft_label > eval_eta)
        soft_reg_label = 1. * (soft_reg_label > reg_eval_eta)



        total_pos_inds = (soft_reg_label > 0)
        new_bbox_targets = new_bbox_targets[total_pos_inds]
        new_bbox_weights = new_bbox_weights[total_pos_inds]
        soft_reg_label = soft_reg_label[total_pos_inds]
        
        pos_bbox_pred = pred_boxes.reshape(pred_boxes.size(0), 4)[total_pos_inds]
        imgs_whwh = imgs_whwh.reshape(pred_boxes.size(0), 4)[total_pos_inds]
        
        new_l1_target = (new_bbox_targets/imgs_whwh, )
        new_iou_target = (new_bbox_targets, )
        
        pos_bbox_pred_imgs_whwh = pos_bbox_pred / imgs_whwh
        
        if self.use_hard_label:
            avg_factor_cls = (1 * pos_inds).sum().float()
            avg_factor_cls = (soft_label * (soft_label>0)).sum().clamp(min=avg_factor_cls).float() 
            avg_factor_cls = reduce_mean(avg_factor_cls)
        else:
            avg_factor_cls = (soft_label * (soft_label>0)).sum().float()
            avg_factor_cls = reduce_mean(avg_factor_cls)
        
        avg_factor_reg = (1 * pos_inds).sum().float()
        avg_factor_reg = (soft_reg_label * (soft_reg_label>0)).sum().clamp(min=avg_factor_reg).float()
        avg_factor_reg = reduce_mean(avg_factor_reg)
        
        return (new_label, soft_label), new_label_weights, \
            new_iou_target, new_bbox_weights, \
            new_l1_target, pos_bbox_pred_imgs_whwh, \
            avg_factor_cls, avg_factor_reg, pos_bbox_pred, imgs_whwh, thres_eta


    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1
        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights
