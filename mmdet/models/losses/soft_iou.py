import math

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def soft_giou_loss(pred, target, alpha=0.25, beta=2., eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    
    if len(target) == 2:
        label, score = target
        score = score.view(-1)
    elif len(target) == 1:
        label = target[0]
    else:
        assert False
    
    gious = bbox_overlaps(pred, label, mode='giou', is_aligned=True, eps=eps)
    #ious = bbox_overlaps(pred, label, mode='iou', is_aligned=True, eps=eps)
    #d = ious - gious
    
    if len(target) == 1:
        loss = 1-gious
    else:
        loss = (1-gious) * score
    
        #loss = (-ious + score).abs().pow(1) + d
        #loss = (-ious + score * (ious < score) + 1 * (ious >= score)).abs().pow(1) + d
    '''
    scale_factor = score - ious
    hard_fg_label = score.new_zeros(score.shape) + 1
    hard_scale_factor = hard_fg_label - ious
    
    r_sig_ious = -(1 / ious.clamp(min=eps) - 1).clamp(min=eps).log()
    
    # loss = d + \
    #     F.binary_cross_entropy_with_logits(
    #         r_sig_ious, 
    #         score,
    #         reduction='none'
    #     ) * scale_factor.abs().pow(beta) + \
    #     F.binary_cross_entropy_with_logits(
    #         r_sig_ious, 
    #         hard_fg_label,
    #         reduction='none'
    #     ) * hard_scale_factor.abs().pow(beta) 
    #     # * (ious < score)
    #     #* (ious >= score)
    
    ious_clamp = ious.clamp(min=eps)
    loss[pos, pos_label] = -1 * (
        ious_clamp[pos, pos_label].log() * score[pos] * alpha + \
        (1 - ious_clamp[pos, pos_label]).log() * (1 - score[pos]) * (1 - alpha)
    ) * scale_factor.abs().pow(beta)
    '''
    return loss


@LOSSES.register_module()
class SoftIou(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, alpha=0.25, beta=2.):
        super(SoftIou, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * soft_giou_loss(
            pred,
            target,
            weight,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
