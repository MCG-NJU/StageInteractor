import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh, multiclass_nms
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead

from mmcv.runner import ModuleList
from ..builder import HEADS, build_head, build_roi_extractor


from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)

import os

def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

@HEADS.register_module()
class StageInteractorDecoder(CascadeRoIHead):

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 align_head_weight_init=False,
                 use_undetach_boxloss=False,
                 use_undetach_boxloss_list=[0, 0, 1, 1, 1, 1],
                 share_modules=3,
                 nms_thres=-1.,
                 use_iou=False,
                 init_noise_feat=-1.,
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        
        self.use_undetach_boxloss = use_undetach_boxloss 
        self.nms_thres = nms_thres
        
        self.align_head_weight_init = align_head_weight_init
        self.use_undetach_boxloss_list = use_undetach_boxloss_list 
        self.share_modules = share_modules 
        self.use_iou = use_iou
        
        
        self.iter_counts = 0
        self.cum_expect_instable = 0

        super(StageInteractorDecoder, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does mean that our method need RoIAlign. We put this as a placeholder to satisfy the argument for the parent class.
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=featmap_strides), 
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)
        
        self.init_noise_feat = init_noise_feat
        if init_noise_feat > 0:
            self.drop_out_feat = nn.ModuleList()
            for _ in range(num_stages):
                self.drop_out_feat.append(
                    nn.Dropout(p=init_noise_feat, inplace=False))

    def get_aligned_head_weight_init(self, anchor_modules, modif_modules):
        for a_p, p in zip(anchor_modules.parameters(), modif_modules.parameters()):
            p.data.copy_(a_p.data.detach())
        

    def _bbox_forward(self, 
        stage, x, xyzr, object_feats, img_metas,  
        imgs_whwh, cls_logit, xyzr_undetach, 
        feats, dyconv1_feats,  
    ):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            x (List[Tensor]): List of FPN features
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(img_metas) 
        bbox_head = self.bbox_head[stage]
        
        
        
        ori_proposal_bboxes = decode_box(xyzr)
        
        if self.init_noise_feat > 0 and self.training:
            object_feats = self.drop_out_feat[stage](object_feats)

        cls_score, bbox_pred, object_feats, dyconv1_feats = bbox_head(
            x, xyzr, object_feats, self.featmap_strides, \
            imgs_whwh, feats=feats, dyconv1_feats=dyconv1_feats,  
        )

        if self.use_undetach_boxloss and (self.use_undetach_boxloss_list[stage] > 0):
            xyzr, proposal_bboxes = self.bbox_head[stage].refine_xyzr(
                xyzr_undetach, bbox_pred)
        else:
            xyzr, proposal_bboxes = self.bbox_head[stage].refine_xyzr(
                xyzr, bbox_pred)
        
        proposal_list = [bboxes for bboxes in proposal_bboxes]
        ori_proposal_list = [bboxes for bboxes in ori_proposal_bboxes]
        
        feats = None

        bbox_results = dict(
            cls_score=cls_score,
            xyzr=xyzr,
            decode_bbox_pred=proposal_bboxes,
            object_feats=object_feats,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list],
            ori_detach_proposal_list=[item.detach() for item in ori_proposal_list],
            proposal_list=proposal_list,
            dyconv1_feats = dyconv1_feats,
            feats = feats,
            box_delta = bbox_pred,
        )
        
        return bbox_results

    def forward_train(self,
                      x,
                      proposal_boxes,
                      proposal_features,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None, 
                    ):

        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        
        xyzr = proposal_boxes
        object_feats = proposal_features
        
        
        all_stage_loss = {}
        
        cls_logit = 0
        
        xyzr_undetach = proposal_boxes
        
        feats, dyconv1_feats = None, None
        
        ori_gt_bboxes, ori_gt_labels = gt_bboxes, gt_labels
        
        lst_ori_assign_results = None
        
        bbox_targets_list = []
        gt_permute_query_id_list = []
        
        gt2predid_in_notall_stage_list = []
        gt2predid_in_all_stage_list = []
        pred2gtid_in_fg_stage_list = []
        
        
        last_x0_reg_base = xyzr 
        additional_q_stage_list = [0 for i in range(self.num_stages+1)]
        all_stage_bbox_results = []
        all_stage_ret_costs_list = []
        iou_snyc_statistics = []

        for stage in range(self.num_stages):

            bbox_results = \
                self._bbox_forward(
                    stage=stage, x=x, xyzr=xyzr, object_feats=object_feats, 
                    img_metas=img_metas,  
                    imgs_whwh=imgs_whwh,
                    cls_logit=cls_logit, xyzr_undetach=xyzr_undetach, \
                    feats=feats, dyconv1_feats=dyconv1_feats, 
                ) 
            
            all_stage_bbox_results.append(bbox_results)
            iou_snyc_statistics.append(self.bbox_head[stage].iou_snyc_statistics.running_mean)
            
            
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']


            xyzr = bbox_results['xyzr'].detach()

            object_feats = bbox_results['object_feats']
            xyzr_undetach = bbox_results['xyzr']
            feats = bbox_results['feats']
            dyconv1_feats = bbox_results['dyconv1_feats']
            cls_logit = bbox_results['cls_score'].clone().detach()
            
            

        for cur_stage in range(self.num_stages):
            
            stage = cur_stage
            
            bbox_results = all_stage_bbox_results[stage]

            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            
            last_proposal_list = bbox_results['ori_detach_proposal_list']
            
            xyzr = bbox_results['xyzr'].detach()
            object_feats = bbox_results['object_feats']
            
            xyzr_undetach = bbox_results['xyzr']
            
            feats = bbox_results['feats']
            dyconv1_feats = bbox_results['dyconv1_feats']
            
            box_delta = bbox_results['box_delta'] 


            if self.stage_loss_weights[stage] <= 0:
                continue
            
            
            ori_assign_results = []
            sampling_results = []
            ret_costs_list = []
            
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            
            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] / imgs_whwh[i])
                last_normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(last_proposal_list[i] / imgs_whwh[i])
                boxes_for_label_assign = None 
                
                try: 
                    assign_result = self.bbox_assigner[stage].assign(
                        normalize_bbox_ccwh, 
                        cls_pred_list[i], 
                        gt_bboxes[i],  
                        gt_labels[i],  
                        img_metas[i], 
                        boxes_for_label_assign=boxes_for_label_assign,
                        ret_costs=True, priors=last_normalize_bbox_ccwh,
                    )
                    ret_costs = assign_result[1:]
                    assign_result = assign_result[0]
                    ori_assign_result = assign_result
                    ori_assign_results.append(ori_assign_result)
                    
                    sampling_result = self.bbox_sampler[stage].sample(
                        assign_result, proposal_list[i], gt_bboxes[i])
                    sampling_results.append(sampling_result)
                    ret_costs_list.append(ret_costs)
                except:
                    assert False
            
            all_stage_ret_costs_list.append(ret_costs_list)
            

            gt2predid_in_all, pred2gtid_in_fg, gt2predid_in_notall = \
                self.get_query2gt_onehot_mat(ori_assign_results)
            
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)


            gt2predid_in_notall_stage_list.append(gt2predid_in_notall)
            gt2predid_in_all_stage_list.append(gt2predid_in_all)
            pred2gtid_in_fg_stage_list.append(pred2gtid_in_fg)
            bbox_targets_list.append(bbox_targets) 
            
            lst_ori_assign_results = ori_assign_results
        

        stage_statistics_list = []
        if self.bbox_head[0].targets_candi_ids is not None:
            for stage in range(self.num_stages):
                bbox_results = all_stage_bbox_results[stage]
                stage_statistics = \
                    self.bbox_head[stage].get_pad_statistics_per_stage(
                        all_stage_ret_costs_list, 
                        gt2predid_in_all_stage_list, 
                        stage, 
                        bbox_results['cls_score'], 
                        num_imgs, 
                        bbox_results['cls_score'].shape[1], 
                        self.num_stages, 
                        cost_type='cls',
                    )
                stage_statistics_list.append(stage_statistics)
        
        all_stage_ret_costs_list = (all_stage_ret_costs_list, stage_statistics_list)
        if self.use_iou:
            all_stage_ret_costs_list = None

        for stage in range(self.num_stages):
            bbox_results = all_stage_bbox_results[stage]
            
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            
            xyzr = bbox_results['xyzr'].detach()
            object_feats = bbox_results['object_feats']
            
            xyzr_undetach = bbox_results['xyzr']
            
            feats = bbox_results['feats']
            dyconv1_feats = bbox_results['dyconv1_feats']
            
            box_delta = bbox_results['box_delta']
            
            bbox_targets = bbox_targets_list[stage]
            
            bbox_targets_candidates = None
            if self.bbox_head[stage].targets_candi_ids is not None:
                box_delta_list = []
                bbox_targets_stage_list = []
                bbox_targets_candidates = []
                pred_bbox_candidates = []
                cls_logits_candidates = []
                for i in self.bbox_head[stage].targets_candi_ids:
                    if (i+stage)>=0 and (i+stage)<self.num_stages: 
                        box_delta_list.append(box_delta)
                        bbox_targets_stage_list.append(i+stage)
                        bbox_targets_candidates.append(bbox_targets_list[i+stage])
                        pred_bbox_candidates.append(all_stage_bbox_results[i+stage]['decode_bbox_pred'])
                        cls_logits_candidates.append(all_stage_bbox_results[i+stage]['cls_score'])
                bbox_targets_candidates = (box_delta_list, bbox_targets_candidates, \
                    pred_bbox_candidates, cls_logits_candidates, bbox_targets_stage_list)
            

            cls_logit = bbox_results['cls_score'].clone()  
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            
            

            single_stage_loss, thres_eta = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh,
                box_delta=box_delta, 
                bbox_targets_candidates=bbox_targets_candidates,
                detach_new_xyzr=xyzr,
                stage=stage,
                gt2predid_in_all_stage_list=gt2predid_in_all_stage_list,
                pred2gtid_in_fg_stage_list=pred2gtid_in_fg_stage_list,
                all_stage_ret_costs_list=all_stage_ret_costs_list,  
                iou_snyc_statistics=iou_snyc_statistics,
            )
            
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                    self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x,
                    proposal_boxes,
                    proposal_features,
                    img_metas,
                    imgs_whwh,
                    rescale=False, 
                    ):

        assert self.with_bbox, 'Bbox head must be implemented.'
        
        # Decode initial proposals
        num_imgs = len(img_metas)
        object_feats = proposal_features
        xyzr = proposal_boxes 
        
        cls_logit = 0
        
        xyzr_undetach = proposal_boxes
        
        feats, dyconv1_feats = None, None
        last_x0_reg_base = xyzr
        
        all_stage_bbox_results = []

        for stage in range(self.num_stages):

            bbox_results = \
                self._bbox_forward(
                    stage=stage, x=x, xyzr=xyzr, object_feats=object_feats, 
                    img_metas=img_metas,  
                    imgs_whwh=imgs_whwh,
                    cls_logit=cls_logit, xyzr_undetach=xyzr_undetach, \
                    feats=feats, dyconv1_feats=dyconv1_feats, 
                )
                
            
            all_stage_bbox_results.append(bbox_results)
            
            object_feats = bbox_results['object_feats']
            cls_logit = bbox_results['cls_score'].clone()  
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list'] 

            xyzr = bbox_results['xyzr'] 
            
            xyzr_undetach = bbox_results['xyzr']
            
            feats = bbox_results['feats']
            dyconv1_feats = bbox_results['dyconv1_feats'] 

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]
        

        for img_id in range(num_imgs):
            
            if self.nms_thres > 0:
                detboxes = proposal_list[img_id]
                detscores = torch.cat([cls_score[img_id], torch.zeros_like(cls_score[img_id][:, [-1, ]])], -1)
                N, C = detscores.shape
                detboxes = detboxes.view(N, 1, 4).expand(N, C, 4)
                detscores = detscores.view(N, 1, C)
                detscores = detscores * torch.eye(C, dtype=detboxes.dtype, device=detboxes.device).view(1, C, C)
                detscores = detscores.reshape(-1, C)
                detboxes = detboxes.reshape(-1, 4)
                
                bbox_pred_per_img, labels_per_img = multiclass_nms(
                    detboxes,
                    detscores,
                    0.,
                    dict(type='nms', iou_threshold=self.nms_thres),
                    max_num=self.test_cfg.max_per_img)
                
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor'] 
                    if bbox_pred_per_img.shape[-1] == 5:
                        bbox_pred_per_img[..., :-1] /= bbox_pred_per_img.new_tensor(scale_factor)
                    else:
                        bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                
                det_bboxes.append(bbox_pred_per_img)
            else:
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg.max_per_img, sorted=False)
                labels_per_img = topk_indices % num_classes 
                bbox_pred_per_img = proposal_list[img_id][topk_indices // num_classes]
            
                if rescale:
                    scale_factor = img_metas[img_id]['scale_factor']
                    bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
                det_bboxes.append(
                    torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, x, proposal_boxes,
                      proposal_features, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []

        num_imgs = len(img_metas)
        object_feats = proposal_features
        xyzr = proposal_boxes
        
        xyzr_undetach = proposal_boxes
        
        feats, dyconv1_feats = None, None
        last_x0_reg_base = xyzr
        
        if self.with_bbox:
            cls_logit = 0
            for stage in range(self.num_stages):
                
                bbox_results = \
                    self._bbox_forward(
                        stage=stage, x=x, xyzr=xyzr, object_feats=object_feats, 
                        img_metas=img_metas,  
                        imgs_whwh=None,
                        cls_logit=cls_logit, xyzr_undetach=xyzr_undetach, \
                        feats=feats, dyconv1_feats=dyconv1_feats, 
                    )
                    
                
                all_stage_bbox_results.append(bbox_results)
                
                object_feats = bbox_results['object_feats']
                cls_logit = bbox_results['cls_score'].clone() 
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list'] 

                xyzr = bbox_results['xyzr'] 
                
                xyzr_undetach = bbox_results['xyzr']
                
                feats = bbox_results['feats']
                dyconv1_feats = bbox_results['dyconv1_feats']
                


        return all_stage_bbox_results
    
    
    def get_query2gt_onehot_mat(self, assign_result_list):
        
        gt2predid_in_notall_list = []
        gt2predid_in_all_list = []
        pred2gtid_in_fg_list = []
        s_query = 0
        s_gt = 0
        for assign_result in assign_result_list:
            gt_inds = assign_result.gt_inds 
            ori_gt_inds = gt_inds
            q_id = torch.where(gt_inds > 0)[0]
            gt_inds = gt_inds[q_id]
            gt_inds_seq_id = torch.argsort(gt_inds)
            q_id = q_id[gt_inds_seq_id]
            
            gt2predid_in_notall = q_id
            gt2predid_in_all = q_id + s_query

            pred2gtid_in_fg = ori_gt_inds - 1 
            pred2gtid_in_fg[ori_gt_inds==0] = -1000
            pred2gtid_in_fg = pred2gtid_in_fg + s_gt
            
            gt2predid_in_notall_list.append(gt2predid_in_notall)
            gt2predid_in_all_list.append(gt2predid_in_all)
            pred2gtid_in_fg_list.append(pred2gtid_in_fg) 
            
            s_query += len(assign_result.gt_inds) 
            s_gt += len(gt_inds)
        
        gt2predid_in_all_list = torch.cat(gt2predid_in_all_list, 0)
        pred2gtid_in_fg_list = torch.cat(pred2gtid_in_fg_list, 0)
        
        gt2predid_in_notall = gt2predid_in_notall_list
        gt2predid_in_all = gt2predid_in_all_list
        pred2gtid_in_fg = pred2gtid_in_fg_list 
        return gt2predid_in_all, pred2gtid_in_fg, gt2predid_in_notall
    