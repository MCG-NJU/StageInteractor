import torch

import torch.nn as nn

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from .hungarian_assigner import HungarianAssigner

from scipy.optimize import linear_sum_assignment
#try:
#    from scipy.optimize import linear_sum_assignment
#except ImportError:
#    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class FullMatchAssigner(BaseAssigner):

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               boxes_for_label_assign=None,
               ret_costs=True,):

        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        matched_col_inds = torch.arange(len(assigned_gt_inds)).to(bbox_pred.device)
        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[:] = matched_col_inds + 1
        assigned_labels[:] = gt_labels[matched_col_inds]
        
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), None

