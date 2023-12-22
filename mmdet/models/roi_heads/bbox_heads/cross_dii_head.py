import torch
import torch.nn as nn
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
import numpy as np

@HEADS.register_module()
class CrossDIIHead(BBoxHead):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        num_classes (int): Number of class in dataset.
            Defaults to 80.
        num_ffn_fcs (int): The number of fully-connected
            layers in FFNs. Defaults to 2.
        num_heads (int): The hidden dimension of FFNs.
            Defaults to 8.
        num_cls_fcs (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        num_reg_fcs (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
        feedforward_channels (int): The hidden dimension
            of FFNs. Defaults to 2048
        in_channels (int): Hidden_channels of MultiheadAttention.
            Defaults to 256.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        ffn_act_cfg (dict): The activation config for FFNs.
        dynamic_conv_cfg (dict): The convolution config
            for DynamicConv.
        loss_iou (dict): The config for iou or giou loss.

    """

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 stage_idx=0,
                 feat_channels=64,
                 out_channels=256,
                 input_feat_shape=7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 targets_candi_ids=None,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(CrossDIIHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]
        self.targets_candi_ids = targets_candi_ids

        #self.instance_interactive_conv = build_transformer(dynamic_conv_cfg)
        self.instance_interactive_conv = CrossDynamicConv(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            input_feat_shape=input_feat_shape,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            stage_idx=stage_idx,
        )
        
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), in_channels)[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(in_channels, 4)
        
        self.iou_snyc_statistics = nn.SyncBatchNorm(1, eps=1e-05, momentum=1e-04)
        nn.init.constant_(self.iou_snyc_statistics.running_mean, 0.5)

        assert self.reg_class_agnostic, 'CrossDIIHead only ' \
            'suppport `reg_class_agnostic=True` '
        assert self.reg_decoded_bbox, 'CrossDIIHead only ' \
            'suppport `reg_decoded_bbox=True`'

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(CrossDIIHead, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        
        #nn.init.zeros_(self.fc_reg.weight)
        #nn.init.zeros_(self.fc_reg.bias)
        self.instance_interactive_conv.init_weights()

    @auto_fp16()
    def forward(self, roi_feat, proposal_feat, dy_filter_list):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

          Returns:
                tuple[Tensor]: Usually a tuple of classification scores
                and bbox prediction and a intermediate feature.

                    - cls_scores (Tensor): Classification scores for
                      all proposals, has shape
                      (batch_size, num_proposals, num_classes).
                    - bbox_preds (Tensor): Box energies / deltas for
                      all proposals, has shape
                      (batch_size, num_proposals, 4).
                    - obj_feat (Tensor): Object feature before classification
                      and regression subnet, has shape
                      (batch_size, num_proposal, feature_dimensions).
        """
        N, num_proposals = proposal_feat.shape[:2]

        # Self attention
        proposal_feat = proposal_feat.permute(1, 0, 2)
        proposal_feat = self.attention_norm(self.attention(proposal_feat))

        # instance interactive
        proposal_feat = proposal_feat.permute(1, 0,
                                              2).reshape(-1, self.in_channels)
        proposal_feat_iic, dy_filter_list = self.instance_interactive_conv(
            proposal_feat, roi_feat, dy_filter_list)
        proposal_feat = proposal_feat + self.instance_interactive_conv_dropout(
            proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)

        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        reg_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        bbox_delta = self.fc_reg(reg_feat).view(N, num_proposals, -1)

        return cls_score, bbox_delta, obj_feat.view(N, num_proposals, -1), dy_filter_list

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
             cls_length=None,
             box2qid=None,
             obj_feat_pair=None,
             feats_pair=None,
             dyconv1_feats_pair=None,
             lst_cls_score=None,
             lst_bbox_pred=None,
             lst_labels=None,
             lst_bbox_targets=None,
             lst_bbox_weights=None,
             box_delta=None,
             lst_box_delta=None,
             gt_permute_query_id=None,
             perclsbox_pred=None,
             bbox_targets_candidates=None,
             detach_new_xyzr=None,
             addi_info_res=None,
             addi_bbox_targets_list=None,
             stage=None,
             gt2predid_in_all_stage_list=None,
             pred2gtid_in_fg_stage_list=None,
             **kwargs):
        """"Loss function of CrossDIIHead, get loss of all images.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            labels (Tensor): Label of each proposals, has shape
                (batch_size * num_proposals_single_image
            label_weights (Tensor): Classification loss
                weight of each proposals, has shape
                (batch_size * num_proposals_single_image
            bbox_targets (Tensor): Regression targets of each
                proposals, has shape
                (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents
                [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression loss weight of each
                proposals's coordinate, has shape
                (batch_size * num_proposals_single_image, 4),
            imgs_whwh (Tensor): imgs_whwh (Tensor): Tensor with\
                shape (batch_size, num_proposals, 4), the last
                dimension means
                [img_width,img_height, img_width, img_height].
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

            Returns:
                dict[str, Tensor]: Dictionary of loss components
        """
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        
        
        labels_new, label_weights, \
        iou_target, iou_bbox_weights, \
        iou_target_imgs_whwh, pos_bbox_pred_imgs_whwh, \
        avg_factor_cls, avg_factor_iou, \
        pos_bbox_pred, imgs_whwh = \
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
                addi_info_res=addi_info_res,
                addi_bbox_targets_list=addi_bbox_targets_list,
                stage=stage,
                gt2predid_in_all_stage_list=gt2predid_in_all_stage_list,
                pred2gtid_in_fg_stage_list=pred2gtid_in_fg_stage_list,
            )
        
        if cls_score is not None:
            if cls_score.numel() > 0:
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
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Almost the same as the implementation in `bbox_head`,
        we add pos_inds and neg_inds to select positive and
        negative samples instead of selecting the first num_pos
        as positive samples.

        Args:
            pos_inds (Tensor): The length is equal to the
                positive sample numbers contain all index
                of the positive sample in the origin proposal set.
            neg_inds (Tensor): The length is equal to the
                negative sample numbers contain all index
                of the negative sample in the origin proposal set.
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains all the gt_boxes,
                has shape (num_gt, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains all the gt_labels,
                has shape (num_gt).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all proposals, has
                  shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all proposals, has
                  shape (num_proposals, 4), the last dimension 4
                  represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all proposals,
                  has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
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
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise just
                  a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals,) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list has
                  shape (num_proposals, 4) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals, 4),
                  the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
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
             addi_info_res=None,
             addi_bbox_targets_list=None,
             stage=None,
             gt2predid_in_all_stage_list=None,
             pred2gtid_in_fg_stage_list=None,
            ):
        
        assert bbox_targets_candidates is not None
        targets_cands, predbox_cands, cls_cands, targets_stage_list = bbox_targets_candidates
        
        iou_mode = 'iou'
        
        avg_factor_cls = avg_factor
        avg_factor_reg = avg_factor
        bg_class_ind = self.num_classes
        with torch.no_grad():
            pred_cls_score = torch.sigmoid(cls_score.clone().detach())
            
            new_label = labels.clone().detach()
            new_label_weights = label_weights.clone().detach()
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            soft_label = pred_boxes.new_zeros(labels.shape[0], self.num_classes) 
            soft_label[pos_inds, labels[pos_inds]] = 1. 
            total_pos_inds = pos_inds
            ori_soft_label = soft_label.clone().detach()
            
            iou = bbox_overlaps(bbox_targets, pred_boxes, mode=iou_mode, is_aligned=True)
            iou = iou.view(-1)
            soft_label[pos_inds, labels[pos_inds]] = iou[pos_inds]
            
            ori_soft_soft_label = soft_label.clone().detach()
            
            new_bbox_targets = bbox_targets.clone().detach()
            new_bbox_weights = bbox_weights.clone().detach()
            soft_reg_label = torch.zeros_like(labels).type_as(pred_boxes)
            soft_reg_label[pos_inds] = 1.
            ori_reg_label = soft_reg_label.clone().detach() 
        
        for idx, (targets, lst_predboxes, lst_clslogits) in enumerate(zip(targets_cands, predbox_cands, cls_cands)):
            
            last_labels, last_label_weights, last_bbox_targets, last_bbox_weights = targets
            last_predbox = lst_predboxes.view(-1, 4)
            last_cls = lst_clslogits.view(-1, lst_clslogits.size(-1))
            last_pred_cls_score = torch.sigmoid(last_cls.clone().detach()).view_as(pred_cls_score)
            
            
            last_stage = targets_stage_list[idx]
            gt2predid_in_all = gt2predid_in_all_stage_list[stage]
            last_gt2predid_in_all = gt2predid_in_all_stage_list[last_stage]
            
            with torch.no_grad():
                last_pos_inds = (last_labels >= 0) & (last_labels < bg_class_ind)
                label_change_inds = last_pos_inds & (~pos_inds)
                
                last_iou_with_curbox = bbox_overlaps(last_bbox_targets, pred_boxes, mode=iou_mode, is_aligned=True)
                last_iou_with_curbox = last_iou_with_curbox.view(-1)
                
                unpermute_query_id = gt2predid_in_all[gt2predid_in_all == last_gt2predid_in_all]
                assert (unpermute_query_id.shape[0]==0) or \
                    (labels[gt2predid_in_all] == last_labels[last_gt2predid_in_all]).all(), \
                    '{}, {}, {}'.format(unpermute_query_id, \
                    labels[gt2predid_in_all], last_labels[last_gt2predid_in_all])
                
                last_permute_query_id = last_gt2predid_in_all[gt2predid_in_all != last_gt2predid_in_all]
                permute_query_id = gt2predid_in_all[gt2predid_in_all != last_gt2predid_in_all]
                
                if stage < last_stage:
                    
                    soft_label[unpermute_query_id, labels[unpermute_query_id]] = \
                        torch.maximum(last_iou_with_curbox[unpermute_query_id], \
                            soft_label[unpermute_query_id, labels[unpermute_query_id]])
        
                    soft_label[last_permute_query_id, last_labels[last_permute_query_id]] = \
                        torch.maximum(last_iou_with_curbox[last_permute_query_id], \
                            soft_label[last_permute_query_id, last_labels[last_permute_query_id]])
        
                elif stage > last_stage:

                    soft_label[unpermute_query_id, labels[unpermute_query_id]] = \
                        torch.maximum(last_iou_with_curbox[unpermute_query_id], \
                            soft_label[unpermute_query_id, labels[unpermute_query_id]])

                    soft_label[last_permute_query_id, last_labels[last_permute_query_id]] = \
                        torch.maximum(last_iou_with_curbox[last_permute_query_id], \
                            soft_label[last_permute_query_id, last_labels[last_permute_query_id]])

                new_label[label_change_inds] = last_labels[label_change_inds]
                new_label_weights[label_change_inds] = last_label_weights[label_change_inds]
                
                total_pos_inds = total_pos_inds | label_change_inds
        
        topk_soft_label = None
        
        thres_soft_label = None
        #thres_soft_label = 1. * (soft_label >= eval_eta)
        
        
        pos_iou_counts_label = soft_label.view(-1).clone().detach()
        select_pos_iou_counts_label = pos_iou_counts_label[pos_iou_counts_label > 0]
        if len(select_pos_iou_counts_label) == 0:
            pos_iou_counts_label = pos_iou_counts_label[0]
        else:
            pos_iou_counts_label = select_pos_iou_counts_label
        pos_iou_counts_label = pos_iou_counts_label.view(-1, 1)
        m_pos_iou_counts_label = self.iou_snyc_statistics(pos_iou_counts_label)
        pos_iou_mu = self.iou_snyc_statistics.running_mean
        pos_iou_std = self.iou_snyc_statistics.running_var.clamp(min=1e-05) ** 0.5
        soft_label[soft_label < pos_iou_mu] = 0
        
        
        
        use_gaussian_deter_label = None
        
        if (topk_soft_label is not None) or (thres_soft_label is not None):
            soft_label = 0 * soft_label
        
        if topk_soft_label is not None:
            soft_label = torch.maximum(soft_label, topk_soft_label)
            
        if thres_soft_label is not None:
            soft_label = torch.maximum(soft_label, thres_soft_label)
        
        
        eval_eta = 0 #1e-7
        soft_label = 1. * (soft_label > eval_eta)
        soft_reg_label = 1. * (soft_reg_label > eval_eta)

        total_pos_inds = (soft_reg_label > eval_eta)
        new_bbox_targets = new_bbox_targets[total_pos_inds]
        new_bbox_weights = new_bbox_weights[total_pos_inds]
        soft_reg_label = soft_reg_label[total_pos_inds]
        
        pos_bbox_pred = pred_boxes.reshape(pred_boxes.size(0), 4)[total_pos_inds]
        imgs_whwh = imgs_whwh.reshape(pred_boxes.size(0), 4)[total_pos_inds]
        
        new_l1_target = (new_bbox_targets/imgs_whwh, )
        new_iou_target = (new_bbox_targets, )
        
        pos_bbox_pred_imgs_whwh = pos_bbox_pred / imgs_whwh
        
        
        avg_factor_cls = (1 * pos_inds).sum().float()
        avg_factor_cls = (soft_label * (soft_label>0)).sum().clamp(min=avg_factor_cls).float()
        #avg_factor_cls = (soft_label * (soft_label>0)).sum().clamp(min=1).float()
        
        avg_factor_cls = reduce_mean(avg_factor_cls)
        avg_factor_reg = (soft_reg_label * (soft_reg_label>0)).sum().float()
        avg_factor_reg = reduce_mean(avg_factor_reg)
        
        return (new_label, soft_label), new_label_weights, \
            new_iou_target, new_bbox_weights, \
            new_l1_target, pos_bbox_pred_imgs_whwh, \
            avg_factor_cls, avg_factor_reg, pos_bbox_pred, imgs_whwh


class CrossDynamicConv(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 stage_idx=0,
                ):
        super(CrossDynamicConv, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels
        self.stage_idx = stage_idx

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        self.fc_layer = nn.Linear(num_output, self.out_channels)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]
        
        M_score_list_len = 4
        self.M_score_list_len = M_score_list_len
        self.M_score_list = nn.Linear(in_channels, self.feat_channels * M_score_list_len, bias=True)
        self.M_score_list2 = nn.Linear(in_channels, self.in_channels * M_score_list_len, bias=True)
        
        self.S_score_list = nn.Linear(in_channels, self.feat_channels * M_score_list_len, bias=True)
        self.S_score_list2 = nn.Linear(in_channels, self.out_channels * M_score_list_len, bias=True)
        
        self.subbox_h = nn.Linear(input_feat_shape * out_channels, input_feat_shape * out_channels, bias=True)
        self.subbox_w = nn.Linear(input_feat_shape * out_channels, input_feat_shape * out_channels, bias=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.M_score_list.weight)
        nn.init.zeros_(self.M_score_list.bias)
        nn.init.zeros_(self.M_score_list2.weight)
        nn.init.zeros_(self.M_score_list2.bias)
        nn.init.zeros_(self.S_score_list.weight)
        nn.init.zeros_(self.S_score_list.bias)
        nn.init.zeros_(self.S_score_list2.weight)
        nn.init.zeros_(self.S_score_list2.bias)
        
        nn.init.zeros_(self.subbox_h.weight)
        nn.init.zeros_(self.subbox_w.weight)


    def forward(self, param_feature, input_feature, dy_filter_list):
        num_proposals = param_feature.size(0)
        input_feature = input_feature.view(num_proposals, self.in_channels,
                                           -1).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        
        num_all_proposals, HW = input_feature.shape[:2]
        H, W = self.input_feat_shape, self.input_feat_shape
        
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        ori_feat = input_feature
        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)
        
        M1w = self.M_score_list(param_feature).sigmoid().view(num_all_proposals, 1, self.feat_channels, self.M_score_list_len).split(self.M_score_list_len*[1,], -1)
        M2w = self.M_score_list2(param_feature).sigmoid().view(num_all_proposals, self.in_channels, 1, self.M_score_list_len).split(self.M_score_list_len*[1,], -1)
        S1w = self.S_score_list(param_feature).sigmoid().view(num_all_proposals, self.feat_channels, 1, self.M_score_list_len).split(self.M_score_list_len*[1,], -1)
        S2w = self.S_score_list2(param_feature).sigmoid().view(num_all_proposals, 1, self.out_channels, self.M_score_list_len).split(self.M_score_list_len*[1,], -1)
        
        for idx, (pre_param_in, pre_param_out) in enumerate(dy_filter_list):
            updated_param_in = M1w[idx].squeeze(-1) * pre_param_in + (1 - M1w[idx].squeeze(-1)) * param_in
            updated_param_in = M2w[idx].squeeze(-1) * updated_param_in + (1 - M2w[idx].squeeze(-1)) * param_in
            updated_param_out = S1w[idx].squeeze(-1) * pre_param_out + (1 - S1w[idx].squeeze(-1)) * param_out
            updated_param_out = S2w[idx].squeeze(-1) * updated_param_out + (1 - S2w[idx].squeeze(-1)) * param_out
            
            features = features.reshape(num_all_proposals, H, W, self.out_channels)
            feat_w = features.view(num_all_proposals, H, W*self.out_channels)
            feat_h = features.permute(0, 2, 1, 3).reshape(num_all_proposals, W, H*self.out_channels)
            feat_w = self.subbox_w(feat_w).view(num_all_proposals, H*W, self.out_channels)
            feat_h = self.subbox_h(feat_h).view(num_all_proposals, W, H, self.out_channels)
            feat_h = feat_h.permute(0, 2, 1, 3).reshape(num_all_proposals, H*W, self.out_channels)
            features = feat_w + feat_h + ori_feat
            ori_feat = features
            
            features = torch.bmm(input_feature, updated_param_in)
            features = self.norm_in(features)
            features = self.activation(features)
            features = torch.bmm(features, updated_param_out)
            features = self.norm_out(features)
            features = self.activation(features)

        features = features.flatten(1)
        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)
        
        if self.stage_idx == 1:
            dy_filter_list = []
        
        dy_filter = (param_in, param_out)
        dy_filter_list.append(dy_filter)
        
        #dy_filter_list = []
        
        return features, dy_filter_list