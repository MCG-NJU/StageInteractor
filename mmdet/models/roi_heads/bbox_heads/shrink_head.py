import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead
from mmdet.core import bbox_overlaps
import math

import os

DEBUG = 'DEBUG' in os.environ

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
    else:
        term = box.new_tensor([1000, ] * box.size(-1)).view(1, 1, -1)
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
class ShrinkHead(BBoxHead):
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
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(ShrinkHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        
        self.ffn_cls = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm_cls = build_norm_layer(dict(type='LN'), content_dim)[1]
        

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)


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

        self.anchor_point_num = anchor_point_num
        self.anchor_channel = anchor_channel
        self.feat_extractor = \
            SubqueryFeatureExtractor(
                content_dim,
                in_points,
                C_sub_q=feat_channels//n_heads,
                G_sub_q=n_heads,
                N_scale=N_scale,
                dim_feedforward=feedforward_channels,
                anchor_point_num=anchor_point_num,
                anchor_channel=anchor_channel,
                num_heads=num_heads,
                dropout=dropout,
                num_queries=num_queries,
            )
        
        self.dynamic_conv = DynamicConv(
            in_dim=feat_channels, 
            in_points=in_points, 
            p_groups=n_heads, 
            num_queries=num_queries,
            num_interact_heads=num_interact_heads,
            num_interact_channel_groups=num_interact_channel_groups,
            dim_feedforward=feedforward_channels,
            query_dim=content_dim,
            out_points=out_points, 
            out_dim=feat_channels, 
        )
        '''
        self.dynamic_conv = DynamicConv(
            in_dim=feat_channels, 
            in_points=in_points//anchor_point_num, 
            p_groups=n_heads, 
            num_queries=num_queries,
            num_interact_heads=num_interact_heads,
            num_interact_channel_groups=num_interact_channel_groups,
            dim_feedforward=feedforward_channels,
            query_dim=anchor_channel,
            out_points=out_points//anchor_point_num, 
            out_dim=anchor_channel, 
        )
        self.sub2main = \
            nn.Linear(anchor_point_num * anchor_channel, content_dim)
        #self.dynamic_conv_norm = nn.Sequential(
        #    nn.LayerNorm(content_dim),
        #)
        '''

    @torch.no_grad()
    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(ShrinkHead, self).init_weights()
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

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)


        self.feat_extractor.init_weights()
        self.dynamic_conv.init_weights()

    @auto_fp16()
    def forward(self,
                x,
                xyzr,
                query_content,
                featmap_strides,
                sub_query_xy,
                sub_query_z,
                sub_query_vec,
                imgs_whwh,
                ):
        '''
            imgs_whwh: (bs, 4)
        '''

        ShrinkHead._DEBUG += 1


        P = self.in_points
        G = self.n_heads
        AN = self.anchor_point_num
        B, N = query_content.shape[:2]
        xyzr = xyzr.reshape(B, N, 4)

        ''' adaptive 3D sampling and mixing '''
        feats, sub_xy, sub_z, \
        sub_query_vec, query_content, \
        sample_points_xy, offset, sample_points_z, scale_logit = \
            self.feat_extractor(
                x,
                featmap_strides,
                query_content,
                xyzr,
                sub_query_xy,
                sub_query_z,
                sub_query_vec,
                imgs_whwh,
            )
        
        query_content, query_content_cls = self.dynamic_conv(feats, query_content, 
            sample_points_xy, offset, sample_points_z, scale_logit, xyzr)
        '''
        # B, N, AN, G*P//AN, C
        feats = feats.view(B, N*AN, G, P//AN, -1)
        # B, N', G, P, C
        
        # B, N, AN, -1
        sub_query_vec = sub_query_vec.view(B, N*AN, -1)
        
        sub_query_vec = self.dynamic_conv(feats, sub_query_vec)
        sub_query_vec = sub_query_vec.reshape(B, N, -1)
        query_content = query_content + self.sub2main(sub_query_vec)
        #query_content = self.dynamic_conv_norm(query_content)
        '''
        

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content))
        
        query_content_cls = self.ffn_norm_cls(self.ffn_cls(query_content_cls))
        
        cls_feat = query_content_cls
        #cls_feat = query_content
        
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(B, N, -1)
        box_delta = self.fc_reg(reg_feat).view(B, N, -1)

        return cls_score, box_delta, query_content.view(B, N, -1), \
            sub_xy, sub_z, sub_query_vec

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr
    
    @force_fp32(apply_to=('deltas', ))
    def refine_xyxy(self, boxes, deltas):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = 2.0, 2.0, 1.0, 1.0
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(100000.0 / 16))
        dh = torch.clamp(dh, max=math.log(100000.0 / 16))

        pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
        pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

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
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        return losses

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




class DynamicConv(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 gfeat_groups_c = None,
                 gfeat_groups_p = None,
                 ):
        '''
            in_dim, out_dim: dim of featmap
        '''
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        gfeat_groups_c = gfeat_groups_c if gfeat_groups_c is not None else p_groups
        gfeat_groups_p = gfeat_groups_p if gfeat_groups_p is not None else p_groups
        
        gfeat_groups_c = 4
        gfeat_groups_p = 8
        
        # out_points = in_points
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        self.gfeat_groups_c = gfeat_groups_c
        self.gfeat_groups_p = gfeat_groups_p
        
        
        #self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups) * (out_points // in_points)
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        
        
        self.s_parameters = (in_points // sampling_rate) * out_points 
        #self.s_parameters = (out_points // sampling_rate) * out_points 
        
        
        
        self.total_parameters = self.m_parameters + self.s_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
 

        self.m_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters * p_groups),
        )
        self.s_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.s_parameters * p_groups),
        )
        
        self.act = nn.ReLU(inplace=True)
        
        
        
        self.temper = (in_dim // p_groups) ** 0.5
        self.v_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * (out_dim // p_groups) * p_groups, 1, groups=1),
            nn.Linear(query_dim, (in_dim // p_groups) * (out_dim // p_groups) * p_groups),
        )
        self.k_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * (in_dim // p_groups) * p_groups, 1, groups=1),
            ##nn.Linear(query_dim, in_points * (in_dim // p_groups) * p_groups),
            #nn.Linear(query_dim, in_points * in_points * p_groups),
            nn.Conv2d(in_dim, in_points * p_groups, 1, groups=p_groups),
        )
        self.q_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * out_points * p_groups, 1, groups=1),
            nn.Linear(query_dim, in_points * out_points * p_groups),
        )
        #in_points * out_points * p_groups
        
        self.Wv_layer_norm2 = nn.LayerNorm(out_dim)
        self.Wv2 = nn.Linear(out_points*out_dim, out_dim, bias=True)
        
        # self.xy_embed = nn.Sequential(
        #     nn.Linear(2, out_points), #out_points #in_dim // p_groups
        # )
        # self.z_embed = nn.Sequential(
        #     nn.Linear(1, out_points), #out_points
        # )
        
        #self.box_embed = nn.Sequential(
        #    nn.Linear(4, out_points * (in_dim // p_groups) * p_groups), 
        #    #nn.Linear(4, out_points * (in_points // sampling_rate) * p_groups),
        #)
        
        # G = p_groups
        # Gc = gfeat_groups_c 
        # Gp = gfeat_groups_p
        # mix_dim_pre_group = in_dim//G//Gc * out_points//Gp
        # self.mix_dim_pre_group = mix_dim_pre_group
        # self.mix_sc = nn.Sequential(
        #     nn.Linear(query_dim, mix_dim_pre_group * mix_dim_pre_group),
        # )
        
        # self.highway_pw = nn.Sequential(
        #     nn.Linear(query_dim, out_dim * in_points),
        # )
        # self.highway_dw = nn.Sequential(
        #     nn.Linear(query_dim, out_dim * out_points),
        # )
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.m_filter_generator[-1].weight)
        nn.init.zeros_(self.s_filter_generator[-1].weight)
        
        #nn.init.zeros_(self.xy_embed[-1].weight)
        #nn.init.zeros_(self.z_embed[-1].weight)
        #nn.init.zeros_(self.box_embed[-1].weight)
        
        
        #nn.init.zeros_(self.q_generator[-1].bias)
        #nn.init.zeros_(self.k_generator[-1].bias)
        
        #nn.init.zeros_(self.Wv.bias)
        #nn.init.zeros_(self.Wv_cls.bias)
        
        #nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.zeros_(self.v_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].weight)
        #nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        
        #nn.init.zeros_(self.mix_sc[-1].weight)
        #nn.init.zeros_(self.highway_pw[-1].weight)
        #nn.init.zeros_(self.highway_dw[-1].weight)
        

    def forward(self, feats, query_vec, sample_points_xy, offset, sample_points_z, scale_logit, query_box):
        '''
            offset (B, N, P, G, 2)
            scale_logit (B, N, P, G)
            query_box (B, N, 4)
        '''
        sc = None
        B, N, G, P, _ = feats.shape
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats
        
        # sample_points_xy = sample_points_xy.detach()
        # offset = offset.detach()
        # sample_points_z = sample_points_z.detach()
        # scale_logit = scale_logit.detach()
        # sample_points_xy = sample_points_xy.permute(0, 1, 3, 2, 4).contiguous()
        # sample_points_xy = sample_points_xy.view(B*N, G*P, 2) 
        # pe = position_embedding(sample_points_xy, self.in_dim // G // 2)
        # pe = pe.reshape(B*N, G, P, -1)
        
        #Hd = self.num_interact_heads
        #feats_x = feats_x.permute(0, 1, 3, 2, 4).contiguous()
        ##feats_x = feats_x.view(B*N, G, P, -1)
        # feats_x = feats_x + pe
        
        #group_query_vec = query_vec

        
        
        ###################################
        
        #highway_M = self.highway_pw(query_vec)
        #highway_M = highway_M.view(B*N, G, P, -1)
        
        M = self.m_filter_generator(query_vec)
        M = M.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats_M = torch.matmul(feats, M) # B*N, G, P, outdim//G
        #feats_M = feats_M + highway_M + feats
        feats_M = F.layer_norm(feats_M, [feats_M.size(-2), feats_M.size(-1)])
        feats_M = self.act(feats_M)
        
        #feats_M = feats_M.reshape(B*N, G, -1, self.out_dim//G)
        
        #highway_S = self.highway_dw(query_vec)
        #highway_S = highway_S.view(B*N, G, self.out_points, -1)
        
        S = self.s_filter_generator(query_vec)
        S = S.reshape(
            B*N, G, self.out_points, -1)
        
        feats_MS = torch.matmul(S, feats_M) # B*N, G, outP, outdim//G
        #feats_MS = feats_MS + highway_S
        feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        feats_MS = self.act(feats_MS)
        # B*N, G, out_points, out_dim // G
        
        
        
        
        ###################################
        
        q = query_vec #.clone().detach() #.reshape(B*N, -1, 1, 1)
        q = self.q_generator(q)
        q = q.reshape(B*N, G, self.out_points, -1)
        
        feats_cls = feats
        # feats_cls = torch.matmul(q, feats)
        # feats_cls = F.layer_norm(feats_cls, [feats_cls.size(-2), feats_cls.size(-1)])
        # feats_cls = self.act(feats_cls)
        
        v = query_vec #.reshape(B*N, -1, 1, 1)
        v = self.v_generator(v)
        v = v.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        v = torch.matmul(feats_cls, v) # B*N, G, P, outdim//G
        v = F.layer_norm(v, [v.size(-2), v.size(-1)])
        v = self.act(v)
        
        
        # k = query_vec #.reshape(B*N, -1, 1, 1)
        # k = self.k_generator(k)
        # k = k.reshape(B*N, G, self.in_dim // G, -1)
        # k = torch.matmul(feats, k) # B*N, G, P, -1
        # k = k.permute(0, 1, 3, 2).contiguous()
        #k = feats.clone().detach()
        k = v
        k = k.permute(0, 2, 1, 3).contiguous() #B*N, P, G, -1
        k = k.reshape(B*N*P, -1, 1, 1)
        k = self.k_generator(k)
        k = k.view(B*N, P, G, -1)
        k = k.permute(0, 2, 3, 1).contiguous()
        
        #k = query_vec 
        #k = self.k_generator(k)
        #k = k.reshape(B*N, G, -1, P)
        
        
        sc = torch.matmul(q, k)
        # sc = sc.view(B*N, G, -1)
        sc = F.softmax(sc / self.temper, -1)
        # sc = sc.view(B*N, G, self.out_points, -1)
        
        feats_cls = torch.matmul(sc, v)
        feats_cls = F.layer_norm(feats_cls, [feats_cls.size(-2), feats_cls.size(-1)])
        feats_cls = self.act(feats_cls)
        
        
        
        ###################################
        
        # Gc = self.gfeat_groups_c
        # Gp = self.gfeat_groups_p
        # 
        # mix_sc_q = self.mix_sc(query_vec)
        # mix_sc_q = mix_sc_q.reshape(B*N, self.mix_dim_pre_group, self.mix_dim_pre_group)
        # 
        # # B*N, G, out_points, out_dim // G
        # feats_MS = feats_MS.reshape(B*N, G, Gp, self.out_points // Gp, Gc, -1)
        # feats_MS = feats_MS.permute(0, 1, 2, 4, 3, 5).contiguous()
        # # B*N, G, Gp, Gc, self.out_points // Gp, out_dim // G // Gc
        # feats_MS = feats_MS.reshape(B*N, G*Gp*Gc, -1)
        # feats_MS = torch.matmul(feats_MS, mix_sc_q)
        # feats_MS = feats_MS.reshape(B*N, G, Gp, Gc, self.out_points // Gp, -1)
        # feats_MS = feats_MS.permute(0, 1, 2, 4, 3, 5).contiguous()
        # feats_MS = feats_MS.view(B*N, G, self.out_points, -1)
        # feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        # feats_MS = self.act(feats_MS)
        # feats_cls = feats_MS
        
        ###################################
        
        feats_MS_flat = feats_MS.reshape(B, N, -1)
        feats_MS_flat = self.Wv(feats_MS_flat)
        
        feats_cls = feats_cls.reshape(B, N, -1)
        feats_cls = self.Wv2(feats_cls)
        
        feats_MS_q = self.Wv_layer_norm(query_vec + feats_MS_flat)
        #feats_cls_q = self.Wv_layer_norm2(query_vec + feats_MS_flat + feats_cls)
        feats_cls_q = self.Wv_layer_norm2(query_vec + feats_cls)
        
        ###################################
        
        #feats_reg = feats_cls_q
        feats_reg = feats_MS_q
        feats_cls = feats_cls_q
        
        
        # feats_MS_flat = feats_MS.reshape(B, N, -1)
        # feats_MS_flat = self.Wv(feats_MS_flat)
        # feats_cls = self.Wv_layer_norm(query_vec + feats_MS_flat)
        # feats_reg = feats_cls
        
        if DEBUG:
            torch.save(S, './demo/S_{}.pth'.format(DynamicConv.IND))
            if sc is not None:
                torch.save(sc, './demo/sc_{}.pth'.format(DynamicConv.IND))
            
        
        DynamicConv.IND += 1
        return feats_reg, feats_cls




class SubqueryFeatureExtractor(nn.Module):
    IND = 0

    def __init__(self,
                 content_dim,
                 in_points,
                 C_sub_q=64,
                 G_sub_q=4,
                 N_scale=5,
                 dim_feedforward=1024,
                 anchor_point_num=8,
                 anchor_channel=64,
                 num_heads=8,
                 dropout=0.,
                 num_queries=100,
                 featmap_dim=None,
                 ):
        super(SubqueryFeatureExtractor, self).__init__()
        
        self.featmap_dim = content_dim if featmap_dim is None else featmap_dim
        
        self.G = G_sub_q
        self.Cq = C_sub_q
        self.in_points = in_points
        self.content_dim = content_dim
        self.num_channel_heads = num_heads
        
        
        
        self.offset_generator = nn.Sequential(
            nn.Linear(content_dim, 2 * G_sub_q * in_points),
        )
        self.scale_generator = nn.Sequential(
            nn.Linear(content_dim, 1 * G_sub_q * in_points),
        )
        

        #'''
        self.attention = MultiheadAttention(content_dim, self.num_channel_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        #'''
        self.iof_tau = nn.Parameter(torch.ones(num_heads, ))
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        
        #self.dypw_attendw = \
        #    DyPWAttenDW(
        #        query_dim=content_dim, 
        #        p_groups=num_heads, 
        #        num_queries=num_queries,
        #        dim_feedforward=dim_feedforward,
        #    )
        
        self.anchor_num = anchor_point_num
        self.anchor_channel = anchor_channel #content_dim // G_sub_q
        self.anchor_offset_generator = nn.Sequential(
            nn.Linear(content_dim, 4 * self.anchor_num),
        )
        self.anchor_feat_generator = nn.Sequential(
            nn.Linear(content_dim, 
                self.anchor_channel * self.anchor_num),
        )
        '''
        self.anchor_feat2offset = nn.Sequential(
            nn.Linear(self.anchor_channel, 
                2 * G_sub_q * in_points // self.anchor_num),
        )
        '''
        
        '''
        #self.featmap_shrink = nn.Sequential(
        #    nn.Linear(content_dim, self.anchor_channel),
        #)
        local_dim_fnn = dim_feedforward * self.anchor_channel // content_dim
        self.local_attention = MultiheadAttention(
            self.anchor_channel, self.num_channel_heads, dropout)
        self.local_attention_norm = nn.LayerNorm([self.anchor_num, self.anchor_channel])
        self.local_ffn = nn.Sequential(
            nn.Linear(self.anchor_channel, local_dim_fnn),
            nn.ReLU(inplace=True),
            nn.Linear(local_dim_fnn, self.anchor_channel),
        )
        self.local_ffn_norm = nn.LayerNorm([self.anchor_num, self.anchor_channel])
        '''
        
        #self.stage_subquery_bboxes = \
        #    nn.Embedding(self.anchor_num, 4)
        '''
        self.stage_query_specific_bboxes = \
            nn.Embedding(num_queries * G_sub_q * in_points, 2)
        self.stage_query_specific_scale = \
            nn.Embedding(num_queries * G_sub_q * in_points, 1)
        '''
        
        '''
        self.current_feat_q = nn.Sequential(
            nn.Linear(self.featmap_dim // self.G, content_dim),
        )
        self.last_subvec_k = nn.Sequential(
            nn.Linear(content_dim // self.G, content_dim),
        )
        self.current_xy_q = nn.Sequential(
            nn.Linear(2, content_dim, bias=False),
        )
        self.last_xy_k = nn.Sequential(
            nn.Linear(2, content_dim, bias=False),
        )
        self.current_z_q = nn.Sequential(
            nn.Linear(1, content_dim, bias=False),
        )
        self.last_z_k = nn.Sequential(
            nn.Linear(1, content_dim, bias=False),
        )
        '''
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.scale_generator[-1].weight)
        nn.init.zeros_(self.scale_generator[-1].bias)
        bias = self.scale_generator[-1].bias.data
        bias.mul_(0.0)
        
        nn.init.zeros_(self.offset_generator[-1].weight)
        nn.init.zeros_(self.offset_generator[-1].bias)
        bias = self.offset_generator[-1].bias.data.view(
            self.G, self.in_points, -1)
        bias.mul_(0.0)
        if int(self.in_points ** 0.5) ** 2 == self.in_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        
        '''
        nn.init.zeros_(self.anchor_feat2offset[-1].weight)
        nn.init.zeros_(self.anchor_feat2offset[-1].bias)
        bias = self.anchor_feat2offset[-1].bias.data
        bandwidth = 0.5 * 1.0
        nn.init.uniform_(bias, -bandwidth, bandwidth)
        '''
        #nn.init.constant_(self.stage_subquery_bboxes.weight, 0.0)
        #nn.init.normal_(self.stage_subquery_bboxes.weight[:, :2], 0.0, bandwidth)
        '''
        nn.init.constant_(self.stage_query_specific_bboxes.weight, 0.0)
        nn.init.constant_(self.stage_query_specific_scale.weight, 0.0)
        '''
    
    def forward(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            sub_query_xy,
            sub_query_z,
            sub_query_vec,
            imgs_whwh,
        ):
        '''
            query_content: B, N, (C = G * Cs)
            query_box: B, N, 4 : x1y1x2y2
            sub_query_xy: B, N, P, G, 2
            sub_query_z: B, N, P, G, S
            sub_query_vec: B, N, P, G, Cs
            wh_image: B, N, 2
            
            sample_feats: B, N, G, P, C_map//G
            sample_points_xy: B, N, P, G, 2
            sample_points_z: B, N, P, G, num_levels
        '''

        wh_image = imgs_whwh[..., :2]
        
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        box_cxcy, box_wh, logscale = \
            self.get_cxcy_wh_logscale(query_box, box_ver='xyzr')
        logscale = logscale.view(B, N, 1, 1)
        
        
        ############################
        
        with torch.no_grad():
            rois = decode_box(query_box)
            roi_box_batched = rois.view(B, N, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(query_box, query_content.size(-1) // 4)
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1))
        query_content_attn = query_content + pe

        #'''
        query_content_attn = query_content_attn.permute(1, 0, 2)
        query_content = self.attention(
            query_content_attn,
            attn_mask=attn_bias.flatten(0, 1),
        )
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)
        #'''
        #query_content = self.dypw_attendw(query_content_attn, attn_mask=attn_bias,)
        
            
        # Every sub-query shares a same filter, 
        # only its vector makes a difference 

        
        #pe = self.get_spatial_info(query_content, query_box)
        
        
        offset, sub_query_xy = self.anchor_xy(
            query_content, sub_query_vec, sub_query_xy)
        offset = offset.reshape(B, N, P, G, 2)
        
        
        scale_logit = self.scale_generator(query_content)
        scale_logit = scale_logit.reshape(B, N, P, G, -1)
        '''
        query_scale = self.stage_query_specific_scale.weight
        query_scale = query_scale[None].expand(B, *query_scale.size())
        query_scale = query_scale.reshape(B, N, P, G, -1)
        scale_logit = scale_logit + query_scale
        '''

        
        box_cxcy = box_cxcy.view(B, N, 1, 1, 2)
        sample_points_xy, delta_sample_points_xy = \
            self.make_sample_points_xy(
                offset, box_cxcy,
                box_wh,
            )
        # sample_points_z, delta_sample_points_z = \
        #     self.make_sample_points_z(
        #         scale_logit,
        #         box_wh, wh_image, featmap_strides,
        #     )
        scale_logit = scale_logit.reshape(B, N, P, G)
        sample_points_z = scale_logit + logscale
        sample_points_z_w = self.regress_z(sample_points_z)
        delta_sample_points_z=None
        
        
        
        sample_feats = self.feat_extract(sample_points_xy, \
            sample_points_z_w, featmap_list, featmap_strides)
        
        
        sub_query_vec, sample_feats = \
            self.intercept_local_feat(sample_feats, sub_query_vec, offset, scale_logit)
        '''
        sample_feats = sub_query_vec
        '''
        #return sample_feats, delta_sample_points_xy, delta_sample_points_z, sub_query_vec ,\
        return sample_feats, sub_query_xy, sub_query_z, sub_query_vec ,\
            query_content, sample_points_xy, offset, sample_points_z, scale_logit
    
    def anchor_xy(self, query_content, sub_query_vec, sub_query_xy):
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        '''
        sub_query_xy = sub_query_xy[..., :2]
        sub_query_xy = sub_query_xy.view(B, N, self.anchor_num, -1, 2)
        
        #anchor_offset = self.anchor_offset_generator(query_content)
        #anchor_offset = sub_query_xy.detach()
        anchor_offset = sub_query_xy
        
        #sub_xy = self.stage_subquery_bboxes.weight.clone()
        #sub_xy = sub_xy[None, None].expand(B, N, *sub_xy.size())
        #sub_xy = sub_xy.reshape(B, N, self.anchor_num, 1, 4)
        #anchor_offset = sub_xy
        
        anchor_offset = anchor_offset.reshape(B, N, self.anchor_num, -1, 2) #dx dy dz dr: B, N, self.anchor_num, 2
        beta = anchor_offset[...,:2]
        #gamma = torch.sigmoid(anchor_offset[...,2:])
        
        anchor_feat = self.anchor_feat_generator(query_content)
        sub_query_vec = sub_query_vec + anchor_feat.view(B, N, self.anchor_num, -1)
        
        anchorbase_offset = self.anchor_feat2offset(sub_query_vec)
        anchorbase_offset = anchorbase_offset.reshape(B, N, self.anchor_num, -1, 2)
        # B, N, AN(=8), G*P/AN, 2 
        
        offset = anchorbase_offset
        #offset = offset * gamma + beta
        ####offset = offset + beta ##
        
        #sub_query_xy = offset[:, :, :, 0, :]
        sub_query_xy = offset[:, :, :, :, :].detach()
        '''
        
        
        
        ###########################################################
        offset = self.offset_generator(query_content)
        offset = offset.reshape(B, N, P, G, 2)
        '''
        query_offset = self.stage_query_specific_bboxes.weight
        query_offset = query_offset[None].expand(B, *query_offset.size())
        query_offset = query_offset.reshape(B, N, P, G, 2)
        
        offset = offset + query_offset
        '''
        return offset, sub_query_xy
    
    def make_sample_points_z(self, 
            predicted_deltas_z,  
            box_wh, 
            wh_image, 
            featmap_strides,
            tau=0.9,
        ):
        
        B, N, P, G, _ = predicted_deltas_z.shape
        
        original_area = box_wh[..., 0] * box_wh[..., 1]
        image_area = wh_image[..., 0] * wh_image[..., 1]
        
        original_z = torch.sqrt(
            torch.clamp(image_area / original_area, min=0, max=1)
        )
        original_z = torch.log2(original_z / featmap_strides[0])
        original_z = len(featmap_strides)-1 - \
            torch.clamp(original_z, min=0, max=len(featmap_strides)-1).long()

        original_z = F.one_hot(original_z, \
            len(featmap_strides)).type_as(predicted_deltas_z)
        original_z = original_z.view(B, N, 1, 1, -1)
        
        delta_z = predicted_deltas_z
        sample_z = (1 - tau) * F.softmax(delta_z, -1) + tau * original_z

        return sample_z, delta_z
    
    def intercept_local_feat(self, sample_feats, sub_query_vec, offset, scale_logit):
        '''
            offset: B, N, P, G, 2
            scale_logit: B, N, P, G
        '''
        B, N, G, P, C = sample_feats.shape
        AN = self.anchor_num
        '''
        offset = offset.permute(0, 1, 3, 2, 4).contiguous()
        scale_logit = scale_logit.permute(0, 1, 3, 2).contiguous()
        scale_logit = scale_logit.reshape(B, N, G, P, 1)
        
        sub_query_vec = sub_query_vec.reshape(B, N, G, P, -1)
        
        q = self.current_feat_q(sample_feats) + \
            self.current_xy_q(offset) + \
            self.current_z_q(scale_logit)
        k = self.last_subvec_k(sub_query_vec)
        q = q.view(B, N, G, P, 1, -1)
        k = k.view(B, N, G, P, -1, 1)
        l = torch.matmul(q, k).view(B, N, G, P, 1)
        
        s = torch.sigmoid(l)
        sub_query_vec = s * sub_query_vec + sample_feats
        '''
        
        '''
        sample_feats = sample_feats\
            .permute(0, 1, 3, 2, 4).contiguous()
        # B, N, P, G, -1
        
        #sample_feats = sample_feats.view(B, N, P*G, -1)
        sample_feats = sample_feats.reshape(B, N, AN, -1, C)
        # B, N, AN(=8), G*P/AN, 2 
        
        sample_feats = sample_feats.view(B*N*AN, -1, C)
        
        #sample_feats = self.featmap_shrink(sample_feats)
        sub_query_vec = sub_query_vec.view(B*N*AN, 1, -1)
        
        #B, N, P, G, 2
        #offset = offset.reshape(B, N, AN, -1, 2)
        offset = offset.reshape(B*N*AN, -1, 2)
        sample_xy_pe = position_embedding(
            offset, sample_feats.size(-1) // 2
        )
        
        #sample_feats = sample_feats + sample_xy_pe
        #sample_feats = sample_feats.reshape(B, N, AN, -1, C)
        
        
        #sample_feats = sample_feats.reshape(B, N, P, G, C)
        #sample_feats = sample_feats.permute(0, 1, 3, 2, 4).contiguous()
        
        sub_query_vec = sub_query_vec.reshape(B, N, AN, -1)
        
        # sample_xy_pe = sample_xy_pe.permute(1, 0, 2)
        # sample_feats = sample_feats.permute(1, 0, 2)
        # sub_query_vec = sub_query_vec.permute(1, 0, 2)
        # sub_query_vec = self.local_attention(
        #     sub_query_vec, 
        #     key=sample_feats + sample_xy_pe, 
        #     value=sample_feats,
        # )
        # sub_query_vec = sub_query_vec.permute(1, 0, 2)
        # sub_query_vec = sub_query_vec.reshape(B, N, AN, -1)
        # sub_query_vec = self.local_attention_norm(sub_query_vec)
        # 
        # sub_query_vec = self.local_ffn_norm(
        #     sub_query_vec + self.local_ffn(sub_query_vec)
        # )
        '''
        return sub_query_vec, sample_feats
        
    
    def get_cxcy_wh_logscale(self, query_box, box_ver='xyzr'):
        if box_ver == 'xyzr':

            box_cxcy = query_box[..., :2]
            scale = 2.00 ** query_box[..., 2:3]
            ratio = 2.00 ** torch.cat(
                [query_box[..., 3:4] * -0.5, 
                    query_box[..., 3:4] * 0.5], dim=-1)
            box_wh = scale * ratio
            
            logscale = query_box[..., 2:3]
            
        elif box_ver == 'xyxy':
        
            box_cxcy = (query_box[..., 0::2] + query_box[..., 1::2]) * 0.5
            box_wh = query_box[..., 2:] - query_box[..., :2]
            
            logscale = 0.5 * torch.log2(box_wh[..., 0] * box_wh[..., 1])
            
        return box_cxcy, box_wh, logscale
    
    def make_sample_points_xy(self, 
        predicted_deltas_xy, cxcy, wh):
        
        B, N, P, G, _ = predicted_deltas_xy.shape

        offset_xy = predicted_deltas_xy
        offset_xy = offset_xy * wh[:, :, None, None, :] #.view(B, N, 1, 1, 2)
        
        sample_xy = offset_xy + cxcy
        delta_xy = sample_xy - cxcy
        return sample_xy, delta_xy
    
    def regress_z(self, z):
        
        def translate_to_linear_weight(ref, tau=2.0):
            grid = torch.arange(4, device=ref.device, \
                dtype=ref.dtype).view(*[len(ref.shape)*[1, ]+[-1, ]])

            ref = ref.unsqueeze(-1).clone()
            l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
            weight = torch.softmax(l2, dim=-1)

            return weight
        
        sample_points_lvl = z.clone()
        sample_points_lvl_mapped = sample_points_lvl - 3.
        sample_points_lvl_weight = \
            translate_to_linear_weight(
                sample_points_lvl_mapped)
        
        return sample_points_lvl_weight
    
    
    def feat_extract(self, 
            sample_points_xy,
            sample_points_z,
            featmap_list,
            featmap_strides,
        ):
        
        B, N, P, G, num_levels = sample_points_z.shape
        B, C_map, H0, W0 = featmap_list[0].shape
        
        
        sample_points_lvl_weight_list = \
            sample_points_z.unbind(-1)
        
        sample_feature = \
            sample_points_z.new_zeros(B, G, C_map//G, N, P)
        
        for i in range(num_levels):
            featmap = featmap_list[i]
            lvl_weights = sample_points_lvl_weight_list[i]
            
            stride = featmap_strides[i]
            mapping_size = featmap.new_tensor(
                [featmap.size(3), featmap.size(2)]) * stride
            mapping_size = mapping_size.view(1, 1, 1, 1, -1)
            normalized_xy = sample_points_xy / mapping_size
            
            sample_feature += self.point_feat_sample(\
                normalized_xy, featmap, weight=lvl_weights)
        
        sample_feature = sample_feature\
            .permute(0, 3, 1, 4, 2).contiguous()
        return sample_feature #B, N, G, P, C_map//G
    
    def point_feat_sample(self, 
            sample_points, featmap, weight=None):
        
        B, N, P, G = sample_points.shape[:-1]
        B, Ck, Hk, Wk = featmap.shape
        
        sample_points = sample_points\
            .permute(0, 3, 1, 2, 4).contiguous()
        # B, n_heads, Hq, Wq, n_points, 2
        # B, n_heads, N, n_points, 2 
        # = (B, G, N, P, 2)
        
        sample_points = sample_points.view(B*G, N, P, 2)
        sample_points = sample_points*2.0-1.0
        featmap = featmap.view(B*G, -1, Hk, Wk)
        ans = F.grid_sample(
            featmap, sample_points,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False,
        )
        
        if weight is not None:
            weight = weight.permute(0, 3, 1, 2).contiguous()
            weight = weight.view(B*G, 1, N, P)
            ans *= weight
        
        ans = ans.view(B, G, -1, N, P)
        return ans
    
    def get_spatial_info(self, query_content, xyzr):
        with torch.no_grad():
            pe = position_embedding(xyzr, query_content.size(-1) // 4)
        return pe
    
    





class DyPWAttenDW(nn.Module):
    def __init__(self,
                 query_dim=256, 
                 p_groups=8, 
                 num_queries=100,
                 dim_feedforward=2048,
                 out_points_rate=8, 
                 ):
        super(DyPWAttenDW, self).__init__()
        
        self.temper = (query_dim // p_groups) ** 0.5
        
        
        self.query_dim = query_dim
        self.p_groups = p_groups
        self.out_points_rate = out_points_rate
        
        
        self.m_parameters = query_dim * (query_dim // out_points_rate)
        self.filter_generator_channel = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters, bias=False),
        )
        self.filter_generator_group = nn.Sequential(
            nn.Linear(num_queries, out_points_rate, bias=False),
        )
        self.filter_bias = nn.Embedding(query_dim, query_dim)
        
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.v_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        #M = num_queries * out_points_rate
        #self.k_generator_group = nn.Sequential(
        #    nn.Linear(num_queries, M),
        #)
        
        self.Wv_layer_norm = nn.LayerNorm(query_dim)
        self.Wv = nn.Linear(query_dim, query_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.filter_generator_channel[-1].weight)
        nn.init.zeros_(self.filter_generator_group[-1].weight)
        nn.init.xavier_normal_(self.filter_bias.weight)
        
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.xavier_uniform_(self.v_generator[-1].weight)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        nn.init.zeros_(self.v_generator[-1].bias)
        
        nn.init.zeros_(self.Wv.bias)
        
        
    
    def forward(self,
            query_content,
            attn_mask=0.,
        ):
        B, N = query_content.shape[:2]
        G = self.p_groups
        '''
        normed_query_content = F.softmax(query_content, -1)
        normed_query_content = F.softmax(normed_query_content, -2)
        
        params = self.filter_generator_channel(normed_query_content) # B,N, C*C//G
        params = params.permute(0, 2, 1).contiguous() # B,C*C//G,N
        
        params = self.filter_generator_group(params) # B,C*C//G,G
        params = params.reshape(B, 1, self.query_dim, self.query_dim) # B,1,C,C
        
        params = params + self.filter_bias.weight
        
        v = query_content.view(B, N, 1, -1)
        v = torch.matmul(v, params)  # B,N, 1,C
        v = v.view(B, N, G, -1) # B,N,G,C//G
        v = v.permute(0, 2, 1, 3).contiguous() # B, G, N, C//G
        
        v = F.layer_norm(v, [v.size(-2), v.size(-1)])
        v = self.act(v) # B, G, N, C//G
        '''
        v = self.v_generator(query_content)
        v = v.view(B, N, G, -1)
        v = v.permute(0, 2, 1, 3).contiguous()
        
        
        #x = v
        #x = x.permute(0, 2, 1, 3).contiguous()
        #x = x.view(B, N, -1)
        
        q = self.q_generator(query_content)
        q = q.view(B, N, G, -1)
        q = q.permute(0, 2, 1, 3).contiguous() # B, G, N, C//G
        
        
        k = self.k_generator(query_content)
        k = k.view(B, N, G, -1)
        k = k.permute(0, 2, 3, 1).contiguous() # B, G, C//G, M
        #k = self.k_generator_group(k) # B, G, C//G, M
        
        
        s = torch.matmul(q, k) # B, G, N, M
        #'''
        s = F.softmax(s / self.temper + attn_mask, -1)
        v = torch.matmul(s, v) # B, G, N, C//G
        #'''
        
        '''
        v = s
        '''
        ###v = F.layer_norm(v, [v.size(-2), v.size(-1)])
        ###v = self.act(v)
        '''
        v = v.permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, -1) # B, N, G*M
        '''
        v = v.permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, -1)
        
        v = self.Wv(v)
        query_content = self.Wv_layer_norm(query_content + v)
        return query_content
        


# class DynamicConv(nn.Module):
#     def __init__(self, 
#                  in_dim=256, 
#                  in_points=32, 
#                  p_groups=4, 
#                  num_queries=100,
#                  num_interact_heads=4,
#                  num_interact_channel_groups=4,
#                  dim_feedforward=2048,
#                  query_dim=None,
#                  out_dim=None, 
#                  out_points=None, 
#                  sampling_rate=None,
#                  ):
#         '''
#             in_dim, out_dim: dim of featmap
#         '''
#         super(DynamicConv, self).__init__()
#         out_dim = out_dim if out_dim is not None else in_dim
#         out_points = out_points if out_points is not None else in_points
#         query_dim = query_dim if query_dim is not None else in_dim
#         sampling_rate = sampling_rate if sampling_rate is not None else 1
#         
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.p_groups = p_groups
#         self.out_points = out_points
#         
#         self.split_g = 1 #######
#         
#         self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
#         self.s_parameters = (in_points // sampling_rate) * out_points // self.split_g
#         self.total_parameters = self.m_parameters + self.s_parameters
#         '''
#         self.dynamic_conv_norm = torch.nn.ModuleList()
#         self.dynamic_conv_norm.append(
#             nn.Sequential(
#                 nn.LayerNorm([in_points, out_dim // p_groups]),
#                 #nn.LayerNorm(out_dim // p_groups),
#                 nn.ReLU(inplace=True),
#             )
#         )
#         self.dynamic_conv_norm.append(
#             nn.Sequential(
#                 nn.LayerNorm([out_points, out_dim // p_groups]),
#                 #nn.LayerNorm(out_dim // p_groups),
#                 nn.ReLU(inplace=True),
#             )
#         )
#         
#         self.pointwise_conv = nn.Sequential(
#             nn.Conv1d(out_dim, dim_feedforward, out_points, \
#                 stride=1, padding=0, groups=out_dim, bias=True),
#             nn.SiLU(inplace=True),
#         )
#         self.depthwise_Wv = nn.Sequential(
#             nn.Linear(dim_feedforward, out_dim, bias=True),
#             nn.LayerNorm(out_dim),
#             nn.ReLU(inplace=True),
#         )
#         '''
#         
#         self.Wv_layer_norm = nn.LayerNorm(out_dim)
#         
#         #'''
#         self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
#         #'''
#         
# 
#         self.filter_generator = nn.Sequential(
#             nn.Linear(query_dim, self.total_parameters * self.p_groups),
#         )
#         #InteractDynamicFilterGenerator(
#         #    num_queries = num_queries,
#         #    num_heads = num_interact_heads,
#         #    num_groups = num_interact_channel_groups,
#         #    c_in = query_dim,
#         #    c_neck = query_dim, 
#         #    c_d = self.total_parameters * self.p_groups,
#         #)
#         ## c_d = self.total_parameters // num_interact_heads * self.p_groups,
#         
#         self.act = nn.ReLU(inplace=True)
#         
#     
#     @torch.no_grad()
#     def init_weights(self):
#         nn.init.zeros_(self.filter_generator[-1].weight)
# 
#     def forward(self, feats, query_vec):
#         B, N, G, P, _ = feats.shape
#         feats = feats.view(B*N, G, P, -1)
#         
#         params = self.filter_generator(query_vec)
#         params = params.reshape(B*N, G, -1)
# 
#         M, S = params.split(
#             [self.m_parameters, self.s_parameters], 2)
#         M = M.reshape(
#             B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G*self.split_g
#         S = S.reshape(
#             B*N, G, self.out_points, -1) # B*N, G, outP, P
#         
#         feats = torch.matmul(feats, M) # B*N, G, P, outdim//G
#         #feats = self.dynamic_conv_norm[0](feats)
#         feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
#         feats = self.act(feats)
#         
#         feats = feats.reshape(B*N, G, self.split_g, P//self.split_g, -1)
#         S = S.reshape(B*N, G, 
#             self.split_g, self.out_points//self.split_g, P//self.split_g)
#         
#         feats = torch.matmul(S, feats) # B*N, G, K, outP, outdim//G
#         #feats = self.dynamic_conv_norm[1](feats)
#         #feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
#         feats = F.layer_norm(feats, [feats.size(-3), feats.size(-2), feats.size(-1)])
#         feats = self.act(feats)
#         # B*N, G, out_points, out_dim // G
#         
#         '''
#         feats = feats.permute(0, 2, 1, 3).contiguous()
#         feats = feats.reshape(B * N, self.out_points, -1)
#         # B * N, out_points, out_dim
#         
#         feats = feats.permute(0, 2, 1).contiguous()
#         feats = self.pointwise_conv(feats)
#         feats = feats.view(B * N, -1)
#         feats = self.depthwise_Wv(feats)
#         '''
#         feats = feats.reshape(B, N, -1)
#         #'''
#         feats = self.Wv(feats)
#         #'''
#         feats = self.Wv_layer_norm(query_vec + feats)
#         return feats

'''

class DynamicConv(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 ):
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        # out_points = in_points
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        
        
        self.s_parameters = (in_points // sampling_rate) * out_points 
        #self.s_parameters = (out_points // sampling_rate) * out_points 
        
        
        
        self.total_parameters = self.m_parameters + self.s_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
 

        self.m_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters * p_groups),
        )
        self.s_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.s_parameters * p_groups),
        )
        
        self.act = nn.ReLU(inplace=True)
        
        
        
        self.temper = (in_dim // p_groups) ** 0.5
        self.v_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * (out_dim // p_groups) * p_groups, 1, groups=1),
            nn.Linear(query_dim, (in_dim // p_groups) * (out_dim // p_groups) * p_groups),
        )
        self.k_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * (in_dim // p_groups) * p_groups, 1, groups=1),
            #nn.Linear(query_dim, (in_points) * (in_dim // p_groups) * p_groups),
            nn.Linear(query_dim, in_points * in_points * p_groups),
            ##nn.Conv2d(in_dim, in_points * p_groups, 1, groups=p_groups),
        )
        self.q_generator = nn.Sequential(
            #nn.Conv2d(query_dim, (in_dim // p_groups) * out_points * p_groups, 1, groups=1),
            nn.Linear(query_dim, in_points * out_points * p_groups),
        )
        #in_points * out_points * p_groups
        
        self.Wv_layer_norm2 = nn.LayerNorm(out_dim)
        self.Wv2 = nn.Linear(out_points*out_dim, out_dim, bias=True)
        
        # self.xy_embed = nn.Sequential(
        #     nn.Linear(2, out_points), #out_points #in_dim // p_groups
        # )
        # self.z_embed = nn.Sequential(
        #     nn.Linear(1, out_points), #out_points
        # )
        
        #self.box_embed = nn.Sequential(
        #    nn.Linear(4, out_points * (in_dim // p_groups) * p_groups), 
        #    #nn.Linear(4, out_points * (in_points // sampling_rate) * p_groups),
        #)
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.m_filter_generator[-1].weight)
        nn.init.zeros_(self.s_filter_generator[-1].weight)
        
        #nn.init.zeros_(self.xy_embed[-1].weight)
        #nn.init.zeros_(self.z_embed[-1].weight)
        #nn.init.zeros_(self.box_embed[-1].weight)
        
        
        #nn.init.zeros_(self.q_generator[-1].bias)
        #nn.init.zeros_(self.k_generator[-1].bias)
        
        #nn.init.zeros_(self.Wv.bias)
        #nn.init.zeros_(self.Wv_cls.bias)
        
        #nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.zeros_(self.v_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].weight)
        #nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        

    def forward(self, feats, query_vec, sample_points_xy, offset, sample_points_z, scale_logit, query_box):
        sc = None
        B, N, G, P, _ = feats.shape
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats
        
        # sample_points_xy = sample_points_xy.detach()
        # offset = offset.detach()
        # sample_points_z = sample_points_z.detach()
        # scale_logit = scale_logit.detach()
        # sample_points_xy = sample_points_xy.permute(0, 1, 3, 2, 4).contiguous()
        # sample_points_xy = sample_points_xy.view(B*N, G*P, 2) 
        # pe = position_embedding(sample_points_xy, self.in_dim // G // 2)
        # pe = pe.reshape(B*N, G, P, -1)
        
        #Hd = self.num_interact_heads
        #feats_x = feats_x.permute(0, 1, 3, 2, 4).contiguous()
        ##feats_x = feats_x.view(B*N, G, P, -1)
        # feats_x = feats_x + pe
        
        #group_query_vec = query_vec

        
        
        ###################################
        
        M = self.m_filter_generator(query_vec)
        M = M.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats_M = torch.matmul(feats, M) # B*N, G, P, outdim//G
        feats_M = F.layer_norm(feats_M, [feats_M.size(-2), feats_M.size(-1)])
        feats_M = self.act(feats_M)
        
        S = self.s_filter_generator(query_vec)
        S = S.reshape(
            B*N, G, self.out_points, -1)
        
        feats_MS = torch.matmul(S, feats_M) # B*N, G, outP, outdim//G
        feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        feats_MS = self.act(feats_MS)
        # B*N, G, out_points, out_dim // G
        
        
        
        
        ###################################
        
        v = query_vec #.reshape(B*N, -1, 1, 1)
        v = self.v_generator(v)
        v = v.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        v = torch.matmul(feats, v) # B*N, G, P, outdim//G
        v = F.layer_norm(v, [v.size(-2), v.size(-1)])
        v = self.act(v)
        
        q = query_vec #.clone().detach() #.reshape(B*N, -1, 1, 1)
        q = self.q_generator(q)
        q = q.reshape(B*N, G, self.out_points, -1)
        
        ## k = query_vec #.reshape(B*N, -1, 1, 1)
        ## k = self.k_generator(k)
        ## k = k.reshape(B*N, G, self.in_dim // G, -1)
        ## k = torch.matmul(feats, k) # B*N, G, P, -1
        ## k = k.permute(0, 1, 3, 2).contiguous()
        # k = feats.clone().detach()
        # k = k.permute(0, 2, 1, 3).contiguous() #B*N, P, G, -1
        # k = k.reshape(B*N*P, -1, 1, 1)
        # k = self.k_generator(k)
        # k = k.view(B*N, P, G, -1)
        # k = k.permute(0, 2, 3, 1).contiguous()
        k = query_vec 
        k = self.k_generator(k)
        k = k.reshape(B*N, G, -1, P)
        
        
        sc = torch.matmul(q, k)
        # sc = sc.view(B*N, G, -1)
        # sc = F.softmax(sc / self.temper, -1)
        # sc = sc.view(B*N, G, self.out_points, -1)
        
        feats_cls = torch.matmul(sc, v)
        feats_cls = F.layer_norm(feats_cls, [feats_cls.size(-2), feats_cls.size(-1)])
        feats_cls = self.act(feats_cls)
        
        
        
        ###################################
        
        feats_MS_flat = feats_MS.reshape(B, N, -1)
        feats_MS_flat = self.Wv(feats_MS_flat)
        
        feats_cls = feats_cls.reshape(B, N, -1)
        feats_cls = self.Wv2(feats_cls)
        
        feats_MS_q = self.Wv_layer_norm(query_vec + feats_MS_flat)
        #feats_cls_q = self.Wv_layer_norm2(query_vec + feats_MS_flat + feats_cls)
        feats_cls_q = self.Wv_layer_norm2(query_vec + feats_cls)
        
        ###################################

        #feats_reg = feats_cls_q
        feats_reg = feats_MS_q
        feats_cls = feats_cls_q
        
        
        # feats_MS_flat = feats_M.reshape(B, N, -1)
        # feats_MS_flat = self.Wv(feats_MS_flat)
        # feats_cls = self.Wv_layer_norm(query_vec + feats_MS_flat)
        # feats_reg = feats_cls
        
        if DEBUG:
            torch.save(S, './demo/S_{}.pth'.format(DynamicConv.IND))
            if sc is not None:
                torch.save(sc, './demo/sc_{}.pth'.format(DynamicConv.IND))
            assert False
        
        DynamicConv.IND += 1
        return feats_reg, feats_cls




class DynamicConv(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 ):
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        
        #self.s_parameters = (in_points // sampling_rate) + out_points
        #self.s_parameters = out_points // sampling_rate * (in_dim // p_groups)
        self.s_parameters = (in_points // sampling_rate) * out_points 
        
        
        self.total_parameters = self.m_parameters + self.s_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
 

        self.m_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters * p_groups),
        )
        self.s_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.s_parameters * p_groups),
        )
        
        self.act = nn.ReLU(inplace=True)
        
        
        
        self.temper = (in_dim // p_groups) ** 0.5
        self.v_generator = nn.Sequential(
            nn.Linear(query_dim, (in_dim // p_groups) * (out_dim // p_groups) * p_groups),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, (in_dim // p_groups) * (in_dim // p_groups) * p_groups),
        )
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, in_dim // p_groups * out_points * p_groups),
        ) #in_points * out_points * p_groups
        
        #self.Wv_layer_norm_cls = nn.LayerNorm(out_dim)
        #self.Wv_cls = nn.Linear(out_points*out_dim, out_dim, bias=True)
        
        # self.xy_embed = nn.Sequential(
        #     nn.Linear(2, out_points), #out_points #in_dim // p_groups
        # )
        # self.z_embed = nn.Sequential(
        #     nn.Linear(1, out_points), #out_points
        # )
        
        #self.box_embed = nn.Sequential(
        #    nn.Linear(4, out_points * (in_dim // p_groups) * p_groups), 
        #    #nn.Linear(4, out_points * (in_points // sampling_rate) * p_groups),
        #)
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.m_filter_generator[-1].weight)
        nn.init.zeros_(self.s_filter_generator[-1].weight)
        
        #nn.init.zeros_(self.xy_embed[-1].weight)
        #nn.init.zeros_(self.z_embed[-1].weight)
        #nn.init.zeros_(self.box_embed[-1].weight)
        
        #nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.zeros_(self.v_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].weight)
        #nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        
        #nn.init.zeros_(self.q_generator[-1].bias)
        #nn.init.zeros_(self.k_generator[-1].bias)
        
        #nn.init.zeros_(self.Wv.bias)
        #nn.init.zeros_(self.Wv_cls.bias)
        

    def forward(self, feats, query_vec, sample_points_xy, offset, sample_points_z, scale_logit, query_box):
        sc = None
        B, N, G, P, _ = feats.shape
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats
        
        #group_query_vec = query_vec
        
        
        M = self.m_filter_generator(query_vec)
        S = self.s_filter_generator(query_vec)
        
        M = M.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats_M = torch.matmul(feats, M) # B*N, G, P, outdim//G
        feats_M = F.layer_norm(feats_M, [feats_M.size(-2), feats_M.size(-1)])
        feats_M = self.act(feats_M)
        
        S = S.reshape(
            B*N, G, self.out_points, -1)
        
        feats_MS = torch.matmul(S, feats_M) # B*N, G, outP, outdim//G
        feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        feats_MS = self.act(feats_MS)
        # B*N, G, out_points, out_dim // G
        
        feats_MS = feats_MS.reshape(B, N, -1)
        feats_MS = self.Wv(feats_MS)
        feats_MS = self.Wv_layer_norm(query_vec + feats_MS)
        
        ###################################
        
        # sample_points_xy = sample_points_xy.detach()
        # offset = offset.detach()
        # sample_points_z = sample_points_z.detach()
        # scale_logit = scale_logit.detach()
        # sample_points_xy = sample_points_xy.permute(0, 1, 3, 2, 4).contiguous()
        # sample_points_xy = sample_points_xy.view(B*N, G*P, 2) 
        # pe = position_embedding(sample_points_xy, self.in_dim // G // 2)
        # pe = pe.reshape(B*N, G, P, -1)

        
        #Hd = self.num_interact_heads
        #feats_x = feats_x.permute(0, 1, 3, 2, 4).contiguous()
        ##feats_x = feats_x.view(B*N, G, P, -1)
        # feats_x = feats_x + pe
        
        
        
        q = self.q_generator(query_vec)
        q = q.reshape(B*N, G, self.out_points, -1)
        sc = q
        
        k = self.k_generator(query_vec)
        k = k.reshape(B*N, G, self.in_dim // G, -1)
        k = torch.matmul(feats, k) # B*N, G, P, -1
        k = k.permute(0, 1, 3, 2).contiguous()
        sc = torch.matmul(q, k)
        sc = sc.view(B*N, G, -1)
        sc = F.softmax(sc / self.temper, -1)
        sc = sc.view(B*N, G, self.out_points, -1)
        
        v = self.v_generator(query_vec)
        v = v.reshape(B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        v = torch.matmul(feats, v)
        
        feats_cls = torch.matmul(sc, v)
        #feats_cls = torch.matmul(sc, feats_M)
        feats_cls = F.layer_norm(feats_cls, [feats_cls.size(-2), feats_cls.size(-1)])
        feats_cls = self.act(feats_cls)
        
        feats_cls = feats_cls.reshape(B, N, -1)
        feats_cls = self.Wv(feats_cls)
        feats_cls = self.Wv_layer_norm(query_vec + feats_cls)
        
        #feats_MS = feats_cls ####
        
        ###################################
        
        
        
        feats_reg = feats_MS
        feats_cls = feats_MS
        
        if DEBUG:
            torch.save(S, './demo/S_{}.pth'.format(DynamicConv.IND))
            if sc is not None:
                torch.save(sc, './demo/sc_{}.pth'.format(DynamicConv.IND))
            assert False
        
        DynamicConv.IND += 1
        return feats_reg, feats_cls



class DynamicConv(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 ):
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        
        #self.s_parameters = (in_points // sampling_rate) + out_points
        #self.s_parameters = out_points // sampling_rate * (in_dim // p_groups)
        self.s_parameters = (in_points // sampling_rate) * out_points 
        
        
        self.total_parameters = self.m_parameters + self.s_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
        #self.Wv = nn.Linear(out_dim, out_dim, bias=True)
        

        self.m_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters * p_groups),
        )
        self.s_filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.s_parameters * p_groups),
        )
        
        self.act = nn.ReLU(inplace=True)
        
        
        
        self.temper = (in_dim // p_groups) ** 0.5
        self.k_generator = nn.Sequential(
            nn.Linear(in_dim // p_groups, self.in_points, bias=True),
        )
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, in_points * out_points * p_groups, bias=True),
        )
        
        #self.Wv_layer_norm_cls = nn.LayerNorm(out_dim)
        #self.Wv_cls = nn.Linear(out_points*out_dim, out_dim, bias=True)
        
        # self.xy_embed = nn.Sequential(
        #     nn.Linear(2, out_points), #out_points #in_dim // p_groups
        # )
        # self.z_embed = nn.Sequential(
        #     nn.Linear(1, out_points), #out_points
        # )
        
        #self.box_embed = nn.Sequential(
        #    nn.Linear(4, out_points * (in_dim // p_groups) * p_groups), 
        #    #nn.Linear(4, out_points * (in_points // sampling_rate) * p_groups),
        #)
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.m_filter_generator[-1].weight)
        nn.init.zeros_(self.s_filter_generator[-1].weight)
        
        #nn.init.zeros_(self.xy_embed[-1].weight)
        #nn.init.zeros_(self.z_embed[-1].weight)
        #nn.init.zeros_(self.box_embed[-1].weight)
        
        #nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].weight)
        #nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        
        #nn.init.zeros_(self.q_generator[-1].bias)
        #nn.init.zeros_(self.k_generator[-1].bias)
        
        #nn.init.zeros_(self.Wv.bias)
        #nn.init.zeros_(self.Wv_cls.bias)
        

    def forward(self, feats, query_vec, sample_points_xy, offset, sample_points_z, scale_logit, query_box):
        sc = None
        B, N, G, P, _ = feats.shape
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats
        
        #group_query_vec = query_vec
        
        
        M = self.m_filter_generator(query_vec)
        S = self.s_filter_generator(query_vec)
        
        M = M.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats_M = torch.matmul(feats, M) # B*N, G, P, outdim//G
        feats_M = F.layer_norm(feats_M, [feats_M.size(-2), feats_M.size(-1)])
        feats_M = self.act(feats_M)
        
        S = S.reshape(
            B*N, G, self.out_points, -1)

        feats_MS = torch.matmul(S, feats_M) # B*N, G, outP, outdim//G
        feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        feats_MS = self.act(feats_MS)
        # B*N, G, out_points, out_dim // G
        
        feats_MS = feats_MS.reshape(B, N, -1)
        feats_MS = self.Wv(feats_MS)
        feats_MS = self.Wv_layer_norm(query_vec + feats_MS)
        
        ##############
        
        # sample_points_xy = sample_points_xy.detach()
        # offset = offset.detach()
        # sample_points_z = sample_points_z.detach()
        # scale_logit = scale_logit.detach()
        # sample_points_xy = sample_points_xy.permute(0, 1, 3, 2, 4).contiguous()
        # sample_points_xy = sample_points_xy.view(B*N, G*P, 2) 
        # pe = position_embedding(sample_points_xy, self.in_dim // G // 2)
        # pe = pe.reshape(B*N, G, P, -1)

        
        #Hd = self.num_interact_heads
        #feats_x = feats_x.permute(0, 1, 3, 2, 4).contiguous()
        ##feats_x = feats_x.view(B*N, G, P, -1)
        # feats_x = feats_x + pe
        
        
        ##sc = torch.matmul(S, k)
        
        q = self.q_generator(query_vec)
        q = q.reshape(B*N, G, self.out_points, -1)
        sc = q
        
        #k = feats_x.clone().detach()
        #k = self.k_generator(k)
        #k = k.permute(0, 1, 3, 2).contiguous() #B*N, G, -1, P
        #sc = torch.matmul(sc, k)
        #sc = sc.view(B*N, G, -1)
        #sc = F.softmax(sc / self.temper, -1)
        #sc = sc.view(B*N, G, self.out_points, -1)
        
        
        feats_cls = torch.matmul(sc, feats_M)
        feats_cls = F.layer_norm(feats_cls, [feats_cls.size(-2), feats_cls.size(-1)])
        feats_cls = self.act(feats_cls)
        
        feats_cls = feats_cls.reshape(B, N, -1)
        feats_cls = self.Wv(feats_cls)
        feats_cls = self.Wv_layer_norm(query_vec + feats_cls)
        
        if DEBUG:
            torch.save(S, './demo/S_{}.pth'.format(DynamicConv.IND))
            if sc is not None:
                torch.save(sc, './demo/sc_{}.pth'.format(DynamicConv.IND))
            assert False
        
        DynamicConv.IND += 1
        return feats_MS, feats_cls


class DynamicConv(nn.Module):
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 ):
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        
        #self.s_parameters = (in_points // sampling_rate) + out_points
        #self.s_parameters = out_points // sampling_rate * (in_dim // p_groups)
        self.s_parameters = (in_points // sampling_rate) * out_points 
        
        
        self.total_parameters = self.m_parameters + self.s_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)
        

        self.filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.total_parameters * p_groups),
        )
        
        self.act = nn.ReLU(inplace=True)
        self.temper = (in_dim // p_groups) ** 0.5
        
        #self.feat_k_generator = nn.Sequential(
        #    nn.Linear(in_dim // p_groups, in_dim // p_groups),
        #)
        #self.q_generator = nn.Sequential(
        #    nn.Linear(query_dim, (out_points * in_dim // p_groups) * p_groups),
        #)
        
        # self.xy_embed = nn.Sequential(
        #     nn.Linear(2, out_points), #out_points #in_dim // p_groups
        # )
        # self.z_embed = nn.Sequential(
        #     nn.Linear(1, out_points), #out_points
        # )
        
        #self.box_embed = nn.Sequential(
        #    nn.Linear(4, out_points * (in_dim // p_groups) * p_groups), 
        #    #nn.Linear(4, out_points * (in_points // sampling_rate) * p_groups),
        #)
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.filter_generator[-1].weight)
        #nn.init.zeros_(self.feat_k_generator[-1].weight)
        #nn.init.zeros_(self.q_generator[-1].weight)
        
        #nn.init.zeros_(self.xy_embed[-1].weight)
        #nn.init.zeros_(self.z_embed[-1].weight)
        #nn.init.zeros_(self.box_embed[-1].weight)

    def forward(self, feats, query_vec, sample_points_xy, offset, sample_points_z, scale_logit, query_box):
        B, N, G, P, _ = feats.shape
        x = query_vec
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats
        
        params = self.filter_generator(x)
        params = params.reshape(B*N, G, -1)

        #M = params

        M, S = params.split(
            [self.m_parameters, self.s_parameters], 2)

        M = M.reshape(
            B*N, G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats = torch.matmul(feats, M) # B*N, G, P, outdim//G
        feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
        feats = self.act(feats)
        
        sample_points_xy = sample_points_xy.detach()
        offset = offset.detach()
        sample_points_z = sample_points_z.detach()
        scale_logit = scale_logit.detach()

        sample_points_xy = sample_points_xy.view(B, -1, 2)
        xy_embed = position_embedding(sample_points_xy, self.in_dim // 2)
        xy_embed = xy_embed.reshape(B, N, P, G, -1)
        xy_embed = xy_embed.permute(0, 1, 3, 4, 2).contiguous()
        xy_embed = xy_embed.view(B*N, G, -1, P) # B*N, G, in_dim, P
        query_box_embed = position_embedding(query_box, self.in_dim // 4)
        query_box_embed = query_box_embed.reshape(B*N, 1, 1, -1)
        query_box_embed = query_box_embed.expand(B*N, G, self.out_points, self.in_dim)
        
        offset_xyz_embed = torch.matmul(query_box_embed, xy_embed)
        
        # offset_xy_embed = self.xy_embed(offset) # B, N, P, G, (in_dim // G)
        # offset_xy_embed = offset_xy_embed.permute(0, 1, 3, 4, 2).contiguous()
        # offset_xy_embed = offset_xy_embed.view(B*N, G, -1, P) # B*N, G, self.in_dim//G, P
        # scale_logit = scale_logit.view(B, N, P, G, 1)
        # offset_z_embed = self.z_embed(scale_logit) # B, N, P, G, (in_dim // G)
        # offset_z_embed = offset_z_embed.permute(0, 1, 3, 4, 2).contiguous()
        # offset_z_embed = offset_z_embed.view(B*N, G, -1, P) # B*N, G, self.in_dim//G, P
        # offset_xyz_embed = offset_xy_embed + offset_z_embed
        #query_box_embed = self.box_embed(query_box)
        #query_box_embed = query_box_embed.reshape(B*N, G, self.out_points, -1)
        
        #sc = offset_xyz_embed + query_box_embed

        S = S.reshape(
            B*N, G, self.out_points, -1)

        #feats_k = self.feat_k_generator(feats_x)
        #feats_k = feats_k.permute(0, 1, 3, 2).contiguous() # B*N, G, self.in_dim//G, P
        #feats_k = feats_k + offset_xyz_embed
        #feats_k = offset_xyz_embed
        
        #query_q = self.q_generator(x)
        #query_q = query_q.reshape(B*N, G, self.out_points, -1)
        #query_q = query_q + query_box_embed
        
        #sc = torch.matmul(query_q, feats_k)
        #sc = F.softmax(sc / self.temper, -1)
        #sc = F.normalize(sc, p=1.0, dim=-2)
        #S = sc
        S = S + offset_xyz_embed
        
        #S1, S2 = S.split(
        #    [self.in_points // self.sampling_rate, self.out_points], 2)
        #S1 = S1.reshape(
        #    B*N, G, -1, self.in_points // self.sampling_rate) # B*N, G, T, P
        #S2 = S2.reshape(
        #    B*N, G, self.out_points, -1) # B*N, G, outP, T
        #S = torch.matmul(S2, S1)
        
        
        feats = torch.matmul(S, feats) # B*N, G, outP, outdim//G
        feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
        feats = self.act(feats)
        # B*N, G, out_points, out_dim // G
        
        feats = feats.reshape(B, N, -1)
        feats = self.Wv(feats)
        feats = self.Wv_layer_norm(x + feats)
        return feats







class DynamicAtten(nn.Module):
    def __init__(self, 
                 query_dim=256, 
                 in_points=32, 
                 num_heads=8, 
                 num_queries=100,
                 dim_feedforward=2048,
                 out_points_sampling_rate=4, 
                 out_dim=None, 
                 ):
        super(DynamicAtten, self).__init__()
        out_dim = out_dim if out_dim is not None else query_dim

        
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.out_p_sr = out_points_sampling_rate
        
        self.m_parameters = (query_dim // num_heads) * (out_dim // num_heads)
        self.s1_parameters = num_queries * num_queries // self.out_p_sr
        self.total_parameters = self.m_parameters + self.s1_parameters
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        
        
        self.Wv = nn.Linear(num_queries*out_dim, out_dim, bias=True)
        
        
        self.filter_generator = nn.Sequential(
            nn.Linear(query_dim, self.total_parameters * self.num_heads),
        )
        
        self.act = nn.ReLU(inplace=True)
        
    
    @torch.no_grad()
    def init_weights(self):
        self.filter_generator.init_weights()

    def forward(self, query_vec):
        B, N, _ = query_vec.shape
        G = self.num_heads
        
        x = query_vec
        
        params = self.filter_generator(query_vec)
        params = params.reshape(B, G, -1)

        M, S1, S2 = params.split(
            [self.m_parameters, self.s1_parameters, self.s2_parameters], 2)
        M = M.reshape(
            B, G, self.query_dim // G, self.out_dim // G)
        S1 = S1.reshape(
            B, G, self.num_queries, -1) # 
        
        query_vec = query_vec.reshape(B, N, G, -1)
        query_vec = query_vec.permute(0, 2, 1, 3).contiguous() # B,G,N,C//G
        
        query_vec = torch.matmul(query_vec, M) # B, G, N, outdim//G
        query_vec = F.layer_norm(query_vec, [query_vec.size(-2), query_vec.size(-1)])
        query_vec = self.act(query_vec)
        
        query_vec = torch.matmul(S1, query_vec) # B, G, outP, outdim//G
        query_vec = F.layer_norm(query_vec, [query_vec.size(-2), query_vec.size(-1)])
        query_vec = self.act(query_vec)
        
        
        
        
        feats = feats.view(B*N, G, P, -1)
        
        
        
        feats = torch.matmul(feats, M) # B*N, G, P, outdim//G
        feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
        feats = self.act(feats)
        
        feats = torch.matmul(S, feats) # B*N, G, outP, outdim//G
        feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
        feats = self.act(feats)
        # B*N, G, num_queries, out_dim // G
        
        
        feats = feats.permute(0, 2, 1, 3).contiguous()
        feats = feats.reshape(B, N, self.num_queries, -1)
        # B * N, num_queries, out_dim
        
        feats = feats.permute(0, 2, 1).contiguous()
        feats = self.pointwise_conv(feats)
        feats = feats.view(B * N, -1)
        feats = self.depthwise_Wv(feats)
        
        feats = feats.reshape(B, N, -1)
        
        feats = self.Wv(feats)
        
        feats = self.Wv_layer_norm(query_vec + feats)
        return feats




class InteractDynamicFilterGenerator(nn.Module):
    IND = 0

    def __init__(self,
                 num_queries=100,
                 num_heads=8,
                 num_groups=4,
                 c_in=256,
                 c_neck=256, 
                 c_d=None,
                 ):
        super(InteractDynamicFilterGenerator, self).__init__()
        self.num_heads = num_heads #k1
        self.num_groups = num_groups #k2
        self.c_neck = c_neck
        self.c_in = c_in
        
        if c_d is None: c_d = 64 * 256 // num_heads
        
        
        self.S_generator = nn.Sequential(
            nn.Linear(c_in, num_queries * num_heads),
        )
        self.M_generator = nn.Sequential(
            nn.Linear(c_in, c_in * c_d // num_heads),
        )
        
        self.dynamic_conv_norm = torch.nn.ModuleList()
        self.dynamic_conv_norm.append(
            nn.Sequential(
                nn.LayerNorm([1, c_in // num_heads]),
                nn.ReLU(inplace=True),
            )
        )
        
    
    @torch.no_grad()
    def init_weights(self):
        pass
    
    def forward(self,
            query_content,
        ):
        B, N = query_content.shape[:2]
        Hd = self.num_heads
        
        x = query_content
        
        query_content = query_content.view(B, N, Hd, -1)
        query_content = query_content.permute(0, 2, 1, 3).contiguous() #B,H,N,C//H
        feats = query_content.view(B, Hd, 1, N, -1)
        feats = feats.expand(B, Hd, N, N, feats.shape[-1])
        
        S = self.S_generator(x)
        S = S.reshape(B, N, Hd, 1, N)
        S = S.permute(0, 2, 1, 3, 4).contiguous() #B,H,N, 1,N
        query_vector = torch.matmul(S, feats) # B,H,N, 1,C//H
        
        query_vector = self.dynamic_conv_norm[0](query_vector)
        query_vector = query_vector + query_content.view(B, Hd, N, 1, -1)

        M = self.M_generator(x)  # B,N, C*D//H
        M = M.view(B, N, H, self.c_in//Hd, -1) 
        M = M.permute(0, 2, 1, 3, 4).contiguous() #B,H,N, C//H*D//H
        M = M.reshape(B, Hd, N, self.c_in//Hd, -1) #B,H,N,C//H, D//H
        query_vector = torch.matmul(query_vector, M) # B,H,N, 1,D//H
        #print(query_vector.shape)
        query_vector = query_vector.view(B, Hd, N, -1)
        query_vector = query_vector.permute(0, 2, 1, 3).contiguous()
        query_vector = query_vector.view(B, N, -1) # B,N,D
        #print(query_vector.shape)
        return query_vector
'''


'''
class InteractDynamicFilterGenerator(nn.Module):
    IND = 0

    def __init__(self,
                 num_queries=100,
                 num_heads=8,
                 num_groups=8,
                 c_in=256,
                 c_neck=256, 
                 c_d=None,
                 ):
        super(InteractDynamicFilterGenerator, self).__init__()
        self.num_heads = num_heads #k1
        self.num_groups = num_groups #k2
        self.c_neck = c_neck
        
        if c_d is None: c_d = 64 * 256 // num_heads
        
        self.key_w = nn.Linear(c_in, c_neck)
        self.query_w = nn.Linear(c_in, c_neck)
        self.channel_interact = nn.Sequential(
            nn.Linear(c_in // num_groups, c_in // num_groups),
            nn.LayerNorm(c_in // num_groups),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // num_groups, c_in // num_groups),
        )
        self.spatial_channel_interact_q_w = nn.Sequential(
            nn.Linear(num_queries * num_groups, c_in),
            nn.LayerNorm(c_in),
            nn.ReLU(inplace=True),
        )
        self.filter_generator = nn.Linear(c_in, c_d * num_groups)
        self.c_d = c_d
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.filter_generator.weight)
    
    def forward(self,
            query_content,
        ):
        
        c_d = self.c_d
        c_neck = self.c_neck
        num_heads = self.num_heads
        num_groups = self.num_groups
        B, n_query = query_content.shape[:2]
        
        key_vector = self.key_w(query_content)
        key_vector = key_vector.view(B, n_query, num_heads, -1)
        key_vector = key_vector.permute(0, 2, 1, 3).contiguous() # B, Nh, N, c_neck/Nh
        
        
        query_vector = query_content.view(B, n_query * num_groups, -1).contiguous()
        query_vector = query_vector + self.channel_interact(query_vector)
        query_vector = query_vector.permute(0, 2, 1) # B, c/k2, N * k2
        query_vector = self.spatial_channel_interact_q_w(query_vector)
        query_vector = self.filter_generator(query_vector) # B, c/k2, D * k2
        query_vector = query_vector.view(B, num_heads, -1, c_d)
        query_vector = torch.matmul(key_vector, query_vector) # B,Nh, N, D
        query_vector = query_vector.permute(0, 2, 1, 3)
        query_vector = query_vector.reshape(B, n_query, -1) # B, N, Nh * D
        
        # query_vector = self.query_w(query_content)
        # query_vector = query_vector.view(B, n_query, num_heads, -1)
        # query_vector = query_vector.permute(0, 2, 1, 3).contiguous() # B, Nh, N, c_neck/Nh
        # 
        # query_vector = query_vector.view(B, num_heads, n_query * num_groups, -1) # B,Nh, N*k2, c_neck/(Nh*k2)
        # query_vector = query_vector.permute(0, 1, 3, 2).contiguous() # B,Nh, c_neck/(Nh*k2), N*k2
        # 
        # query_vector = self.spatial_channel_interact_q_w(query_vector) # B,Nh, c_neck/(Nh*k2), D * k2
        # query_vector = query_vector.permute(0, 1, 3, 2).contiguous() # B,Nh, D*k2 , c_neck/(Nh*k2)
        # query_vector = query_vector.view(B, num_heads, -1, c_neck//num_heads) # B,Nh, D, c_neck/Nh
        # query_vector = query_vector.permute(0, 1, 3, 2).contiguous() # B,Nh, c_neck/Nh, D
        # 
        # query_vector = torch.matmul(key_vector, query_vector) # B,Nh, N, D
        # query_vector = query_vector.permute(0, 2, 1, 3).contiguous() # B,N, Nh, D
        # query_vector = query_vector.view(B, n_query, -1)
        
        return query_vector
'''





'''
class ShrinkHead(nn.Module):
    IND = 0

    def __init__(self,
                 num_queries=100,
                 num_heads=8,
                 num_groups=8,
                 hidden_dim=64,
                 c_in=256,
                 c_out=256,
                 c_neck=256, 
                 ):
        super(ShrinkHead, self).__init__()
        self.num_heads = num_heads #k1
        self.num_groups = num_groups #k2
        self.c_out = c_out
        self.hidden_dim = hidden_dim
        
        self.channel_extension = nn.Sequential(
            nn.Linear(c_in // num_groups, c_neck),
            nn.LayerNorm(c_neck),
            nn.ReLU(inplace=True),
            nn.Linear(c_neck, c_neck),
        )
        self.spatial_interact = nn.Sequential(
            nn.Linear(num_groups * num_queries, c_neck),
            nn.LayerNorm(c_neck),
            nn.ReLU(inplace=True),
            nn.Linear(c_neck, num_groups * num_queries),
        )
        self.group_filter_generator = nn.Sequential(
            nn.Conv2d(num_groups * num_queries, 2 * c_out * hidden_dim, groups=num_groups),
        )
        self.key_w = nn.Linear(c_in, c_neck)
    
    @auto_fp16()
    def involved_query(self, 
                    query_content,
                    ):
        c_out = self.c_out
        hidden_dim = self.hidden_dim
        num_heads = self.num_heads
        num_groups = self.num_groups
        B, n_query = query_content.shape[:2]
        
        key_vector = self.key_w(query_content)
        key_vector = key_vector.view(B, n_query, num_heads, -1)
        key_vector = key_vector.permute(0, 2, 1, 3).contiguous() # B, num_head, N, c_neck/num_head
        
        query_filter = query_filter.view(B, n_query * num_groups, -1).contiguous()
        query_filter = self.channel_extension(query_filter)
        query_filter = query_filter.view(B, n_query * num_groups, num_heads, -1)
        query_filter = query_content.permute(0, 2, 3, 1).contiguous() # B, num_head, c_neck/num_head, N*num_groups
        query_filter = query_filter + self.spatial_interact(query_filter)
        query_filter = torch.matmul(key_vector, query_filter) # B, num_head, N, N*num_groups
        
        query_filter = query_filter.permute(0, 3, 1, 2).contiguous() # B, N*num_groups, num_head, N
        query_filter = self.group_filter_generator(query_filter) # B, 2*c*h, num_head, N
        query_filter = query_filter.permute(0, 2, 3, 1).contiguous() # B, num_head, N, 2*c*h
        #query_filter = self.filter_generator(query_filter) # B, num_head, N, 2*c*h
        
        
        query_filter = query_filter.view(B, num_head, n_query, 2, c_out, hidden_dim).contiguous()
        return query_filter
    
    @auto_fp16()
    def forward(self,
                x,
                xyzr,
                query_content,
                featmap_strides):
        B, n_query = query_content.shape[:2]
        
'''