import mmcv
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

from .gfocal_loss import QualityFocalLoss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def quality_focal_loss(pred, target, beta=2.0, alpha=0.25, alpha_bgfg2fg=0.5):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """

    if len(target) == 2:
        label, score = target
    elif len(target) == 4:
        label, score, label_transform, score_transform = target
    else:
        assert False
    
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(beta) * (1 - alpha)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    
    if len(target) == 4:
        pos = ((label_transform >= 0) & (label_transform < bg_class_ind)).nonzero().squeeze(1)
        pos_label = label_transform[pos].long()
        scale_factor = score_transform[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score_transform[pos],
            reduction='none') * scale_factor.abs().pow(beta) * alpha_bgfg2fg #* \
            #(pred_sigmoid[pos, pos_label] <= score_transform[pos])
        #loss[pos, pos_label] = -1 * (
        #    pred_sigmoid[pos, pos_label].log() * score_transform[pos] * alpha_bgfg2fg + \
        #    (1 - pred_sigmoid[pos, pos_label]).log() * (1 - score_transform[pos]) * (1 - alpha_bgfg2fg)
        #) * scale_factor.abs().pow(beta)
        
    
    pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos],
        reduction='none') * scale_factor.abs().pow(beta) * alpha #* \
        #(pred_sigmoid[pos, pos_label] <= score[pos])

    loss = loss.sum(dim=1, keepdim=False)
    return loss



@LOSSES.register_module()
class AlphaQualityFocalLoss(QualityFocalLoss):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 alpha_bgfg2fg=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(AlphaQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_bgfg2fg = alpha_bgfg2fg
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.gamma,
                alpha=self.alpha,
                alpha_bgfg2fg=self.alpha_bgfg2fg,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

