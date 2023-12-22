import torch

from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead

from mmcv.runner import ModuleList
from ..builder import HEADS, build_head, build_roi_extractor

import os
DEBUG = 'DEBUG' in os.environ


@HEADS.register_module()
class ShrinkDecoder(CascadeRoIHead):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 share_head_weight=False,
                 share_head_list=[0, 1, 1, 2, 2, 2],
                 twice_hung=False,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        
        self.share_head_weight = share_head_weight
        self.share_head_list = share_head_list
        self.twice_hung = twice_hung
        
        super(ShrinkDecoder, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does mean that our method need RoIAlign. We put this as a placeholder to satisfy the argument for the parent class.
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = ModuleList()
        self.bbox_head = ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        
        if self.share_head_weight:
            shared_head = []
            cnt = self.share_head_list[0] - 1
            for i, (roi_extractor, shared_stage_id) in enumerate(zip(bbox_roi_extractor, self.share_head_list)):
                self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
                if shared_stage_id > cnt:
                    cnt = shared_stage_id
                    head = build_head(bbox_head[i])
                
                self.bbox_head.append(head)
        else:
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
                self.bbox_head.append(build_head(head))

    def _bbox_forward(self, stage, x, xyzr, object_feats, img_metas, \
      sub_q_xy, sub_q_z, sub_q_vec, imgs_whwh, cls_logit, xyzr_undetach):
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
        # bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]

        cls_score, bbox_pred, object_feats, \
        sample_points_xy, sample_points_z, \
        sub_query_vec, cls_length, box2qid = bbox_head(
            x, xyzr, object_feats, self.featmap_strides, \
            sub_q_xy, sub_q_z, sub_q_vec, imgs_whwh, \
            cls_logit, xyzr_undetach, 
        )

        ##################
        xyzr, proposal_bboxes = self.bbox_head[stage].refine_xyzr(
            xyzr, bbox_pred)
        #xyzr, proposal_bboxes = self.bbox_head[stage].refine_xyzr(
        #    xyzr_undetach, bbox_pred)
        proposal_list = [bboxes for bboxes in proposal_bboxes]

        # proposal_list = self.bbox_head[stage].refine_bboxes(
        #     xyxy,
        #     xyxy.new_zeros(len(xyxy)),  # dummy arg
        #     bbox_pred.view(-1, bbox_pred.size(-1)),
        #     [xyxy.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
        #     img_metas)
        # proposal_bboxes = torch.cat(proposal_list)
        
        ###proposal_bboxes = self.bbox_head[stage].refine_xyxy(
        ###    xyxy, bbox_pred)
        ###proposal_list = [bboxes for bboxes in proposal_bboxes]
        ###xyxy = proposal_bboxes
        

        bbox_results = dict(
            cls_score=cls_score,
            xyzr=xyzr,
            decode_bbox_pred=proposal_bboxes,
            object_feats=object_feats,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_proposal_list=[item.detach() for item in proposal_list],
            proposal_list=proposal_list,
            sub_q_xy = sample_points_xy,
            sub_q_z = sample_points_z,
            sub_q_vec = sub_query_vec,
            cls_length = cls_length,
            box2qid = box2qid,
        )
        ShrinkDecoder._DEBUG += 1
        if DEBUG:
            with torch.no_grad():
                torch.save(
                    bbox_results, 'demo/bbox_results_{}.pth'.format(ShrinkDecoder._DEBUG))
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
                      sub_xy=None,
                      sub_z=None,
                      subquery_vec=None,
                    ):
        """Forward function in training stage.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, content_dim)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """

        num_imgs = len(img_metas)
        num_proposals = proposal_boxes.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_proposals, 1)
        all_stage_bbox_results = []
        xyzr = proposal_boxes
        object_feats = proposal_features
        
        sub_q_xy = sub_xy
        sub_q_z = sub_z
        sub_q_vec = subquery_vec
        
        all_stage_loss = {}
        
        cls_logit = subquery_vec
        
        xyzr_undetach = proposal_boxes

        for stage in range(self.num_stages):
            bbox_results = \
                self._bbox_forward(stage, x, xyzr, object_feats, \
                    img_metas, sub_q_xy, sub_q_z, sub_q_vec, imgs_whwh, cls_logit, xyzr_undetach)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            proposal_list = bbox_results['detach_proposal_list']
            
            xyzr = bbox_results['xyzr'].detach()
            #if stage < 2:
            #    object_feats = bbox_results['object_feats'].detach()
            #else:
            #    object_feats = bbox_results['object_feats']
            object_feats = bbox_results['object_feats']
            
            
            sub_q_xy = bbox_results['sub_q_xy']
            sub_q_z = bbox_results['sub_q_z']
            sub_q_vec = bbox_results['sub_q_vec']
            
            cls_length = bbox_results['cls_length']
            box2qid = bbox_results['box2qid']
            
            xyzr_undetach = bbox_results['xyzr']


            if self.stage_loss_weights[stage] <= 0:
                continue
            
            if self.twice_hung:
                sampling_results_topk = []

            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(proposal_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, proposal_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
                
                if self.twice_hung:
                    #print(gt_bboxes[i].shape)
                    #assert False
                    assign_result_topk = self.bbox_assigner[stage].assign(
                        normalize_bbox_ccwh, 
                        cls_pred_list[i], 
                        gt_bboxes[i], gt_labels[i], img_metas[i],
                        topk_hung=True)
                    sampling_result_topk = self.bbox_sampler[stage].sample(
                        assign_result_topk, proposal_list[i], gt_bboxes[i])
                    sampling_results_topk.append(sampling_result_topk)
            
            
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)
            
            if self.twice_hung:
                bbox_targets_topk = self.bbox_head[stage].get_targets(
                    sampling_results_topk, gt_bboxes, gt_labels, 
                    self.train_cfg[stage], True)

            cls_logit = bbox_results['cls_score'].clone() #.detach()
            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']
            
            if cls_length is not None:
                cls_length = cls_length.view(-1, cls_length.size(-1))
            
            if box2qid is not None:
                box2qid = box2qid.view(num_imgs, -1, box2qid.size(-1))

            
            
            if self.twice_hung:
                single_stage_loss = self.bbox_head[stage].loss(
                    None,
                    decode_bbox_pred.view(-1, 4),
                    *bbox_targets,
                    imgs_whwh=imgs_whwh,
                    cls_length=cls_length,
                    box2qid=box2qid,
                )
                single_stage_loss_topk = self.bbox_head[stage].loss(
                    cls_score.view(-1, cls_score.size(-1)),
                    decode_bbox_pred.view(-1, 4),
                    *bbox_targets_topk,
                    imgs_whwh=imgs_whwh,
                )
                for key, value in single_stage_loss_topk.items():
                    single_stage_loss[f'topk_{key}'] = value
            else:
                single_stage_loss = self.bbox_head[stage].loss(
                    cls_score.view(-1, cls_score.size(-1)),
                    decode_bbox_pred.view(-1, 4),
                    *bbox_targets,
                    imgs_whwh=imgs_whwh,
                    cls_length=cls_length,
                    box2qid=box2qid,
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
                    sub_xy=None,
                    sub_z=None,
                    subquery_vec=None,
                    ):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_boxes (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 4)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, content_dim)
            img_metas (dict): meta information of images.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            rescale (bool): If True, return boxes in original image
                space. Defaults to False.

        Returns:
            bbox_results (list[tuple[np.ndarray]]): \
                [[cls1_det, cls2_det, ...], ...]. \
                The outer list indicates images, and the inner \
                list indicates per-class detected bboxes. The \
                np.ndarray has shape (num_det, 5) and the last \
                dimension 5 represents (x1, y1, x2, y2, score).
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        if DEBUG:
            torch.save(img_metas, 'demo/img_metas.pth')

        # Decode initial proposals
        num_imgs = len(img_metas)
        object_feats = proposal_features
        xyzr = proposal_boxes
        
        sub_q_xy = sub_xy
        sub_q_z = sub_z
        sub_q_vec = subquery_vec
        
        cls_logit = subquery_vec
        
        xyzr_undetach = proposal_boxes

        for stage in range(self.num_stages):
            bbox_results = \
                self._bbox_forward(stage, x, xyzr, object_feats,
                    img_metas, sub_q_xy, sub_q_z, sub_q_vec, imgs_whwh, cls_logit, xyzr_undetach)
            object_feats = bbox_results['object_feats']
            cls_logit = bbox_results['cls_score'].clone() #.detach()
            cls_score = bbox_results['cls_score']
            proposal_list = bbox_results['detach_proposal_list']
            xyzr = bbox_results['xyzr']
            
            sub_q_xy = bbox_results['sub_q_xy']
            sub_q_z = bbox_results['sub_q_z']
            sub_q_vec = bbox_results['sub_q_vec']
            
            xyzr_undetach = bbox_results['xyzr']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = proposal_list[img_id][topk_indices //
                                                      num_classes]
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
        
        if self.with_bbox:
            cls_logit = sub_q_vec
            for stage in range(self.num_stages):
                bbox_results = \
                    self._bbox_forward(stage, x, xyzr, object_feats,
                        img_metas, sub_q_xy, sub_q_z, sub_q_vec, None, cls_logit, xyzr_undetach)
                object_feats = bbox_results['object_feats']
                cls_logit = bbox_results['cls_score'].clone() #.detach()
                cls_score = bbox_results['cls_score']
                proposal_list = bbox_results['detach_proposal_list']
                xyzr = bbox_results['xyzr']
                
                sub_q_xy = bbox_results['sub_q_xy']
                sub_q_z = bbox_results['sub_q_z']
                sub_q_vec = bbox_results['sub_q_vec']
                
                xyzr_undetach = bbox_results['xyzr']
                
        return all_stage_bbox_results
