import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from mmcv.runner import BaseModule

from ...core import bbox_cxcywh_to_xyxy

from .detr_head import DETRHead

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply, bbox2result, bbox2roi, bbox_xyxy_to_cxcywh

from mmdet.core import build_assigner, build_sampler, reduce_mean

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from mmdet.core import bbox_overlaps
import math
import numpy as np

import os

DEBUG = 'DEBUG' in os.environ

def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

def refine_xyzr(xyzr, xyzr_delta, return_bbox=True):
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


@HEADS.register_module()
class QueryBackbone_SubInitialQueryGenerator(BaseModule):
    """
    This module produces initial content vector $\mathbf{q}$ and positional vector $(x, y, z, r)$.
    Note that the initial positional vector is **not** learnable.
    """

    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 num_query=100,
                 content_dim=256,
                 scale_num=4,
                 per_group_point_num=32,
                 point_group_num=4,
                 anchor_point_num=8,
                 anchor_channel=64,
                 subbox_poolsize=9,
                 featmap_strides=[4, 8, 16, 32],
                 rpn_num_query=500,
                 num_heads=8,
                 dropout=0.,
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfgs=dict(
                        assigner=dict(
                            type='HungarianAssigner',
                            cls_cost=dict(type='FocalLossCost', weight=2.0),
                            reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                            iou_cost=dict(type='IoUCost', iou_mode='giou',
                                          weight=2.0)),
                        sampler=dict(type='PseudoSampler'),
                        pos_weight=1),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(QueryBackbone_SubInitialQueryGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.num_query = num_query
        self.in_channels = in_channels
        
        self.train_cfg = train_cfgs
        
        
        self.assigner = build_assigner(self.train_cfg['assigner'])
        self.sampler = build_sampler(self.train_cfg['sampler'], context=self)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        
        self.content_dim = content_dim
        self.scale_num = scale_num
        self.P = per_group_point_num
        self.G = point_group_num
        
        self.anchor_num = anchor_point_num
        self.anchor_channel = anchor_channel
        
        
        self.rpn_num_query = rpn_num_query
        
        
        self.featmap_strides = featmap_strides
        self.rpn_subquery_sampler = \
            RPN_SubqueryFeatureExtractor(
                content_dim,
                per_group_point_num,
                G_sub_q=point_group_num,
                num_queries=rpn_num_query,
                featmap_dim=in_channels,
                subbox_poolsize=subbox_poolsize,
            )
        
        self.direct_decoder = DirectDecoder(
            in_dim=in_channels,
            query_dim=content_dim,
            in_points=per_group_point_num,
            subbox_poolsize=subbox_poolsize,
            num_classes=num_classes,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self._init_layers()

    def _init_layers(self):

        self.init_subquery_bboxes = \
            nn.Embedding(self.rpn_num_query, 4)
        self.init_subquery_z = \
            nn.Embedding(self.rpn_num_query, self.scale_num)
        self.init_subquery_vec = \
            nn.Embedding(self.rpn_num_query, self.content_dim)
        
        self.init_proposal_bboxes = nn.Embedding(self.num_query, 4)
        self.init_content_features = nn.Embedding(
            self.num_query, self.content_dim)
        
        self.rpn_subquery_sampler.init_weights()
        self.direct_decoder.init_weights()

    def init_weights(self):
        super(QueryBackbone_SubInitialQueryGenerator, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)
        nn.init.normal_(self.init_content_features.weight, 0.0, 1.0)
        
        nn.init.normal_(self.init_subquery_vec.weight, 0.0, 1.0)
        nn.init.constant_(self.init_subquery_z.weight, 0.0)
        
        nn.init.constant_(self.init_subquery_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_subquery_bboxes.weight[:, 2:], 1)
        #nn.init.uniform_(self.init_subquery_bboxes.weight[:, :2], 0, 1)
        #nn.init.uniform_(self.init_subquery_bboxes.weight[:, 2:], 0, 1)

    def rpn_detect(self, imgs, featmap_strides, query_content, xyzr):
        feats = self.get_subbox_feat(
            imgs, featmap_strides, query_content, xyzr)
        cls_score, bbox_pred = self.direct_decoder(feats, query_content, xyzr)
        
        xyzr, proposal_bboxes = refine_xyzr(xyzr, bbox_pred)
        
        proposal_list = [bboxes for bboxes in proposal_bboxes]
        
        num_imgs = xyzr.shape[0]
        
        bbox_results = dict(
            cls_score=cls_score,
            xyzr=xyzr,
            decode_bbox_pred=proposal_bboxes,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list],
            proposal_list=proposal_list,
        )
        
        return bbox_results

    def get_subbox_feat(self, imgs, featmap_strides, query_content, query_box):
        feats = self.rpn_subquery_sampler(
            imgs, featmap_strides, 
            query_content, query_box,
        )
        return feats

    def _decode_init_proposals(self, imgs, img_metas):
        """
        Hacks based on 'sparse_roi_head.py'.
        For the positional vector, we first compute (x, y, z, r) that fully covers an image. 
        """
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        proposals = proposals * imgs_whwh
        
        #####################
        xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
        wh = proposals[..., 2:4] - proposals[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()
        proposals = torch.cat([xy, z, r], dim=-1)
        
        
        
        sub_xy = self.init_subquery_bboxes.weight.clone()
        sub_xy = sub_xy[None].expand(num_imgs, *sub_xy.size())
        sub_xy = sub_xy.reshape(num_imgs, self.rpn_num_query, 4)
        xy = 0.5 * (sub_xy[..., 0:2] + sub_xy[..., 2:4])
        wh = sub_xy[..., 2:4] - sub_xy[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()
        sub_xy = torch.cat([xy, z, r], dim=-1)
        
        sub_z = self.init_subquery_z.weight.clone()
        sub_z = sub_z[None].expand(num_imgs, *sub_z.size())
        sub_z = sub_z.reshape(num_imgs, self.rpn_num_query, -1)
        subquery_vec = self.init_subquery_vec.weight.clone()
        subquery_vec = subquery_vec[None].expand(num_imgs, *subquery_vec.size())
        subquery_vec = torch.layer_norm(
            subquery_vec, normalized_shape=[subquery_vec.size(-1)])
        subquery_vec = subquery_vec.reshape(num_imgs, self.rpn_num_query, -1)

        init_content_features = self.init_content_features.weight.clone()
        init_content_features = init_content_features[None].expand(
            num_imgs, *init_content_features.size())

        init_content_features = torch.layer_norm(
            init_content_features, normalized_shape=[init_content_features.size(-1)])
        
        proposals = proposals.detach()
        sub_xy = sub_xy.detach()
        sub_z = sub_z.detach()

        return proposals, init_content_features, imgs_whwh, \
            sub_xy, sub_z, subquery_vec

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)

    def forward_train(self, 
                      img, 
                      img_metas, 
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                    ):
        """Forward function in training stage."""
        
        proposals, init_content_features, imgs_whwh, \
        sub_xy, sub_z, subquery_vec = \
            self._decode_init_proposals(img, img_metas)
        
        query_content = subquery_vec
        xyzr = sub_xy
        bbox_results = self.rpn_detect(
            img, self.featmap_strides, query_content, xyzr)
        
        
        
        losses = self.get_loss(
            xyzr.shape[1], bbox_results, img_metas, imgs_whwh, gt_bboxes, gt_labels)
        
        return proposals, init_content_features, imgs_whwh, \
            sub_xy, sub_z, subquery_vec, losses

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    
    def get_loss(self,
             num_proposals,
             bbox_results,
             img_metas,
             imgs_whwh,
             gt_bboxes,
             gt_labels,
             **kwargs):
        
        num_imgs = len(img_metas)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        
        sampling_results = []
        cls_pred_list = bbox_results['detach_cls_score_list']
        proposal_list = bbox_results['detach_proposal_list']

        xyzr = bbox_results['xyzr'].detach()
        
        all_stage_loss = {}

        for i in range(num_imgs):
            normalize_bbox_ccwh = \
                bbox_xyxy_to_cxcywh(proposal_list[i] / imgs_whwh[i])
            
            assign_result = self.assigner.assign(
                normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                gt_labels[i], img_metas[i])
            sampling_result = self.sampler.sample(
                assign_result, proposal_list[i], gt_bboxes[i])
            sampling_results.append(sampling_result)
        bbox_targets = self.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg,
            True)

        cls_score = bbox_results['cls_score']
        decode_bbox_pred = bbox_results['decode_bbox_pred']

        single_stage_loss = self.loss(
            cls_score.view(-1, cls_score.size(-1)),
            decode_bbox_pred.view(-1, 4),
            *bbox_targets,
            imgs_whwh=imgs_whwh)
        for key, value in single_stage_loss.items():
            all_stage_loss[f'rpn_{key}'] = value * 1
        return all_stage_loss
    
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
                losses['rpn_loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['rpn_pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['rpn_loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['rpn_loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['rpn_loss_bbox'] = bbox_pred.sum() * 0
                losses['rpn_loss_iou'] = bbox_pred.sum() * 0
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
            
            #if not self.reg_decoded_bbox:
            #    pos_bbox_targets = self.bbox_coder.encode(
            #        pos_bboxes, pos_gt_bboxes)
            #else:
            #    pos_bbox_targets = pos_gt_bboxes
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














class mhsa(nn.Module):
    
    def __init__(self,
                 content_dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        super(mhsa, self).__init__()
        
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        self.iof_tau = nn.Parameter(torch.ones(num_heads, ))
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
    
    def forward(self, query_content, xyzr):
        
        B, N = query_content.shape[:2]
        xyzr = xyzr.reshape(B, N, 4)
        
        with torch.no_grad():
            rois = self.decode_box(xyzr)
            roi_box_batched = rois.view(B, N, 4)
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]
            iof = (iof + 1e-7).log()
            pe = self.position_embedding(xyzr, query_content.size(-1) // 4)
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1))
        query_content_attn = query_content + pe

        query_content_attn = query_content_attn.permute(1, 0, 2) # N, B, C
        
        
        query_content_attn = self.attention(
            query_content_attn,
            attn_mask=attn_bias.flatten(0, 1),
        ) # identity=0,
        query_content = query_content_attn
        query_content = self.attention_norm(query_content)
        
        
        query_content = query_content.permute(1, 0, 2)
        return query_content
    
    def decode_box(self, xyzr):
        scale = 2.00 ** xyzr[..., 2:3]
        ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                                  xyzr[..., 3:4] * 0.5], dim=-1)
        wh = scale * ratio
        xy = xyzr[..., 0:2]
        roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
        return roi
    
    def position_embedding(self, box, num_feats, temperature=10000, ver='xyzr'):
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

class DirectDecoder(nn.Module):
    def __init__(self,
                 in_dim=256,
                 query_dim=256,
                 in_points=32,
                 subbox_poolsize=9,
                 num_classes=80,
                 num_heads=8,
                 dropout=0.,
                 ):
        super(DirectDecoder, self).__init__()
        self.in_points = in_points
        self.subbox_poolsize = subbox_poolsize
        self.direct_proj = nn.Sequential(
            nn.Linear(in_dim*in_points, query_dim, bias=True),
        )
        self.direct_cls = nn.Linear(query_dim, num_classes, bias=True)
        self.direct_reg = nn.Linear(query_dim, 4, bias=True)
        
        self.act = nn.ReLU(inplace=True)
        
        self.mhsa = mhsa(
            in_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.direct_cls.bias, bias_init)

        nn.init.zeros_(self.direct_reg.weight)
        nn.init.zeros_(self.direct_reg.bias)
    
    def forward(self,
            sample_feats, 
            query_content, 
            query_box, 
        ):
        P = self.in_points
        B, N, G = sample_feats.shape[:3] # B, N, G, P * self.subbox_poolsize, C_map//G
        sample_feats = sample_feats.view(B, N, G, P, self.subbox_poolsize, -1)
        sample_feats = sample_feats.mean(-2)

        sample_feats = sample_feats.reshape(B, N, -1)
        
        out_feats = self.direct_proj(sample_feats)
        
        out_feats = self.mhsa(out_feats, query_box)
        
        out_feats = self.act(out_feats)
        out_cls = self.direct_cls(out_feats)
        out_reg = self.direct_reg(out_feats)

        return out_cls, out_reg


class RPN_SubqueryFeatureExtractor(nn.Module):
    IND = 0

    def __init__(self,
                 content_dim,
                 in_points,
                 G_sub_q=4,
                 num_queries=100,
                 featmap_dim=None,
                 subbox_poolsize=9,
                 ):
        super(RPN_SubqueryFeatureExtractor, self).__init__()
        
        self.featmap_dim = content_dim if featmap_dim is None else featmap_dim
        
        self.G = G_sub_q
        self.content_dim = content_dim
        self.subbox_poolsize = subbox_poolsize
        self.num_center_in_points = in_points
        self.in_points = in_points

        
        self.subbox_generator = nn.Sequential(
            nn.Linear(content_dim, 4 * G_sub_q * in_points),
        )
        self.points_in_subbox_generator = nn.Sequential(
            nn.Linear(content_dim, 2 * G_sub_q * in_points * subbox_poolsize),
        )
        
        kernel_indices = self.create_box_element_indices(subbox_poolsize)
        self.register_buffer('kernel_indices', kernel_indices)

        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):

        nn.init.zeros_(self.subbox_generator[-1].weight)
        nn.init.zeros_(self.subbox_generator[-1].bias)
        bias = self.subbox_generator[-1].bias.data.view(
            self.G, -1, 4)
        bias.mul_(0.0)
        if int((self.in_points * self.G) ** 0.5) ** 2 == self.in_points * self.G:
            h = int((self.in_points * self.G) ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1).view(self.G, -1, 2)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias[:, :, :2], -bandwidth, bandwidth)
        
        nn.init.zeros_(self.points_in_subbox_generator[-1].weight)
        nn.init.zeros_(self.points_in_subbox_generator[-1].bias)
        bias = self.points_in_subbox_generator[-1].bias.data.view(
            self.G, self.in_points, -1, 2)
        bias.mul_(0.0)
        bandwidth = 0.5 * 1.0
        nn.init.uniform_(bias[:, :, 1:, :2], -bandwidth, bandwidth)


    def create_box_element_indices(self, kernel_size):
        if int(kernel_size**0.5)**2 == kernel_size:
            kernel_size = int(kernel_size**0.5)
            if kernel_size % 2 == 0:
                start_idx = -kernel_size // 2
                end_idx = kernel_size // 2

                indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
                
            else:
                start_idx = -(kernel_size - 1) // 2
                end_idx = (kernel_size - 1) // 2

                indices = torch.linspace(start_idx, end_idx, kernel_size)
                
            tmp = indices[0].clone()
            indices[0] = indices[len(indices)//2].clone()
            indices[len(indices)//2] = tmp
            
            i, j = torch.meshgrid(indices, indices)
            kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / kernel_size
        else:
            delta_theta = 2 * math.pi / max(kernel_size - 1, 1)
            ids = dim_t = torch.arange(
                kernel_size, dtype=torch.float32)
            ids = torch.clamp(ids - 1, min=0.)
            delta_thetas = delta_theta * ids
            i = delta_thetas.clone()
            j = delta_thetas.clone()
            i = 0.5 * i.cos()
            j = 0.5 * j.sin()
            i[0] = 0.
            j[0] = 0.
            kernel_indices = torch.stack([i, j], dim=-1).view(-1, 2)
        
        return kernel_indices
    
    
    def get_subbox_feat(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            use_adaptive_kernel_coordinates=True,
            use_constrain_r=True,
        ):
        
        
        C_map = featmap_list[0].shape[1]
        num_levels = len(featmap_list)
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        ori_box = query_box.view(B, N, 1, 4)
        
        xyzr_delta = self.subbox_generator(query_content)
        xyzr_delta = xyzr_delta.view(B, N, -1, 4) # B, N, G*P//9=4*36/9=16, 4
        if use_constrain_r:
            xyzr_delta[:, :, :, -1] = 1
        sub_query_box_xyzr = self.subbox_refine_xyzr(ori_box, xyzr_delta, return_bbox=False)
        
        sub_query_box_xyzr = sub_query_box_xyzr.view(B, N, G, -1, 4) # B, N, G, P//9, 4
        sub_query_box_xyzr = sub_query_box_xyzr.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P//9, 4
        sub_query_box_xyzr = sub_query_box_xyzr.view(B*G, -1, 4) # B*G, N*P//9, 4
        featmap_list = [i.reshape(B*G, -1, i.shape[-2], i.shape[-1]) for i in featmap_list]
        # B*G, 64, H, W

        weight_z = self.regress_z(
            sub_query_box_xyzr[:, :, 2], 
            stride_size=len(featmap_strides), 
            tau=2.0, 
            mask_size=None, #self.subbox_poolsize
        )
        
        sample_points_lvl_weight_list = weight_z.unbind(-1)
        
        sub_query_box_xyxy = decode_box(sub_query_box_xyzr) # B*G, N*P//9, 4
        sub_query_box_xyxy = sub_query_box_xyxy.view(B*G, -1, 1, 4)
        cxcy = 0.5 * (sub_query_box_xyxy[..., :2] + sub_query_box_xyxy[..., 2:])
        wh = sub_query_box_xyxy[..., 2:] - sub_query_box_xyxy[..., :2]
        if use_adaptive_kernel_coordinates:
            kernel_indices = self.points_in_subbox_generator(query_content)
            kernel_indices = kernel_indices.view(B, N, G, P, -1, 2)
            kernel_indices = kernel_indices.permute(0, 2, 1, 3, 4, 5).contiguous()
            kernel_indices = kernel_indices.view(B*G, N*P, -1, 2)
        else:
            kernel_indices = self.kernel_indices.view(1, 1, -1, 2)
        grid = cxcy + kernel_indices * wh # B*G, N*P//9, 9, 2
        
        if DEBUG:
            torch.save(sub_query_box_xyxy, 
                'demo/rpn_sub_query_box_xyxy_{}.pth'.format(RPN_SubqueryFeatureExtractor.IND))
            torch.save(weight_z, 
                'demo/rpn_weight_z_{}.pth'.format(RPN_SubqueryFeatureExtractor.IND))
        
        sample_feature = weight_z.new_zeros(B, G, C_map//G, N, P*self.subbox_poolsize) ###
        for i in range(num_levels):
            Hk, Wk = featmap_list[i].shape[-2:]
            
            featmap = featmap_list[i] # B*G, 64, H, W
            lvl_weights = sample_points_lvl_weight_list[i]  # B*G, N*P//9
            
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

            sample_feats = sample_feats.view(B, G, C_map//G, N, -1, self.subbox_poolsize) # B, G, C//G, N, P//9 ,9
            lvl_weights = lvl_weights.reshape(B, G, 1, N, -1, 1)  # B, G, 1, N, P//9
            sample_feats *= lvl_weights
            
            sample_feats = sample_feats.view(B, G, C_map//G, N, -1)
            
            
            sample_feature += sample_feats
        
        # B, G, C_map//G, N, P
        sample_feature = sample_feature.permute(0, 3, 1, 4, 2).contiguous()
        return sample_feature #B, N, G, P, C_map//G
    
    def forward(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
        ):
        '''
            query_content: B, N, (C = G * Cs)
            query_box: B, N, 4 : x1y1x2y2
            wh_image: B, N, 2
            
            sample_feats: B, N, G, P, C_map//G
            sample_points_xy: B, N, P, G, 2
            sample_points_z: B, N, P, G, num_levels
        '''


        
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        subbox_feat = \
            self.get_subbox_feat(
                featmap_list,
                featmap_strides,
                query_content,
                query_box,
            )
        
        if DEBUG:
            torch.save(
                subbox_feat, 'demo/rpn_subbox_feat_{}.pth'.format(RPN_SubqueryFeatureExtractor.IND))
        
        RPN_SubqueryFeatureExtractor.IND += 1
        
        return subbox_feat

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
    

    def subbox_refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        
        ori_x = xyzr[..., 0:1]
        ori_y = xyzr[..., 1:2]
        z = xyzr[..., 2:3]
        r = xyzr[..., 3:4]
        zr = xyzr[..., 2:4]
        ori_w = (2 ** (z - 0.5*r))
        ori_h = (2 ** (z + 0.5*r))
        
        
        new_x = ori_x + xyzr_delta[..., 0:1] * ori_w
        new_y = ori_y + xyzr_delta[..., 1:2] * ori_h
        
        new_zr = zr + xyzr_delta[..., 2:4]
        
        xyzr = torch.cat([new_x, new_y, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr
    