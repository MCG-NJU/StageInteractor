from ..builder import DETECTORS
from .two_stage import TwoStageDetector

from mmcv.runner import load_checkpoint
from .. import build_detector
from ..builder import DETECTORS

from mmcv.runner import ModuleList

from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.fp16_utils import cast_tensor_type

from torch.cuda.amp import autocast, GradScaler

import torch

import os
#DEBUG = 'DEBUG' in os.environ
DEBUG = int(os.environ['DEBUG']) if 'DEBUG' in os.environ else -1

@DETECTORS.register_module()
class StableDiffusionDet(TwoStageDetector):

    def __init__(self, self_ckpt = None, autocast_fp16=False, eval_teacher=False, ema_rate=0.9999, *args, **kwargs):
        
        self.eval_teacher = eval_teacher
        self.ema_rate = ema_rate
        self.autocast_fp16 = autocast_fp16
        if self_ckpt is not None:
            self._is_init = True
        
        super(StableDiffusionDet, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'StableDiffusionDet do not support external proposals'

        is_distill_stage_list = self.roi_head.is_distill_stage_list
        num_stages = self.roi_head.num_stages
        basic_inference_indices = self.roi_head.basic_inference_indices
        chain_length_idx = self.roi_head.chain_length_idx
        link_forward_star = self.roi_head.link_forward_star
        if self_ckpt is not None:
            self._is_init = True
            
            load_checkpoint(self, self_ckpt, map_location='cpu')
            
            link_forward_star = self.roi_head.link_forward_star
            if link_forward_star is not None:
                basic_inference_indices = self.roi_head.basic_inference_indices
                num_stages = self.roi_head.num_stages # 12
                
                chain_length_idx = self.roi_head.chain_length_idx
                chain_length_idx = torch.tensor(chain_length_idx).long()
                
                #for stage in range(len(basic_inference_indices), num_stages):
                for stage in range(0, num_stages):
                    last_stage = torch.where(
                        chain_length_idx == chain_length_idx[stage])[0][0]
                    #print(chain_length_idx, chain_length_idx[stage], last_stage)
                    
                    if last_stage >= len(basic_inference_indices):
                        last_stage = stage - 1
                    
                    is_valid = False
                    while True:
                        is_valid = self.roi_head.get_aligned_head_weight_init(
                            self.roi_head.bbox_head[last_stage], self.roi_head.bbox_head[stage], stage)
                        last_stage = last_stage - 1
                        if is_valid:
                            break
                        elif last_stage < 0:
                            break
            else:
                pre_distill_num = self.roi_head.pre_distill_num #1
                num_teacher_stage = self.roi_head.num_teacher_stage #4
                num_post_additional_heads = self.roi_head.num_post_additional_heads #3
                num_stages = self.roi_head.num_stages # 12
                
                for i in range(pre_distill_num + num_teacher_stage + num_post_additional_heads, num_stages): 
                    self.roi_head.get_aligned_head_weight_init(
                        self.roi_head.bbox_head[i-1], self.roi_head.bbox_head[i], stage)
        
        self.teacher_stage_list = []
        if link_forward_star is not None:
            for i in range(num_stages):
                if i not in basic_inference_indices:
                    self.teacher_stage_list.append(i)
        
        # self.teacher_model = ModuleList()
        # for i in basic_inference_indices:
        #     self.teacher_model.append(self.roi_head.bbox_head[i])
        
        self.num_stages = num_stages
        self.basic_inference_indices = basic_inference_indices
        self.chain_length_idx = chain_length_idx
        self.is_distill_stage_list = is_distill_stage_list
        
        self.stage_correspond_basic = [0 for i in range(num_stages)]
        if link_forward_star is not None:
            for i in range(num_stages):
                for j in basic_inference_indices:
                    if (self.chain_length_idx[i] == self.chain_length_idx[j]):
                        self.stage_correspond_basic[i] = j
        
        assert ((not eval_teacher) or (link_forward_star is not None))

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      caption=None,
                      **kwargs):

        assert proposals is None, 'StableDiffusionDet does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'StableDiffusionDet does not instance segmentation'

        if DEBUG >= 0:
            for n,p in self.named_parameters():
                p.requires_grad = False

        x = self.extract_feat(img)
        
        #proposal_boxes, proposal_features, \
        #imgs_whwh, sub_xy, sub_z, subquery_vec = \
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)
        #print(caption)
        #assert False
        roi_losses = self.roi_head.forward_train(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks,
            imgs_whwh=imgs_whwh,
            sub_xy=None,
            sub_z=None,
            subquery_vec=None,
        )
        
        # for n,p in self.named_parameters():
        #     if p.grad is None and p.requires_grad:
        #         print(n)
        
        # for n,p in self.named_parameters():
        #     print(n, p.mean().item(), p.var().item())
        # 
        # assert False
        
        if self.eval_teacher:
            vis_dict = set()
            for stage in self.teacher_stage_list:
                last_stage = self.stage_correspond_basic[stage]
                # if self.is_distill_stage_list[last_stage] == 3:
                #     if self.chain_length_idx[stage] <= 2:
                #         last_stage = -1
                #     else:
                #         last_stage = -1
                # print(stage, last_stage)
                vis_dict = self._momentum_update(
                    self.roi_head.bbox_head[last_stage], 
                    self.roi_head.bbox_head[stage], 
                    momentum=self.ema_rate,
                    vis_dict=vis_dict,
                )
        
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        #proposal_boxes, proposal_features, \
        #imgs_whwh, sub_xy, sub_z, subquery_vec = \
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale,
            sub_xy=None,
            sub_z=None,
            subquery_vec=None,
        )
        return bbox_results

    def forward_dummy(self, img):
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        #proposal_boxes, proposal_features, \
        #imgs_whwh, sub_xy, sub_z, subquery_vec = \
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = \
            self.roi_head.forward_dummy(
                x, proposal_boxes,
                proposal_features,
                dummy_img_metas,
            )
        return roi_outs

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            for stage in self.teacher_stage_list:
                self.roi_head.bbox_head[stage].train(False)
        else:
            for stage in self.teacher_stage_list:
                self.roi_head.bbox_head[stage].train(mode)
        
        super().train(mode)
    
    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
    
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
    
    #@auto_fp16(apply_to=('img'), out_fp32=True)
    #def extract_feat(self, img):
    #    """Directly extract features from the backbone+neck."""
    #    x = self.backbone(img)
    #    if self.with_neck:
    #        x = self.neck(x)
    #    return x
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if self.autocast_fp16:
            with autocast():
                x = self.backbone(img)
                if self.with_neck:
                    x = self.neck(x)
                
                x = [ x_i.float() for x_i in x]
        else:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            
        return x
    
    @torch.no_grad() 
    def _momentum_update(self, update_modules, momentum_modules, momentum=0.9999, vis_dict=None):
        
        for (p_name, param_m) in momentum_modules.named_parameters():
            if (p_name in vis_dict):
                continue
            
            for (a_name, param) in update_modules.named_parameters():
                if (p_name == a_name):
                    param_m.data = param_m.data * momentum + param.data * (1. - momentum)
                    vis_dict.add(p_name)
        
        return vis_dict