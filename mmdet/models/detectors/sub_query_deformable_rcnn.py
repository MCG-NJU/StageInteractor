from ..builder import DETECTORS
from .two_stage import TwoStageDetector

import os
#DEBUG = 'DEBUG' in os.environ
DEBUG = int(os.environ['DEBUG']) if 'DEBUG' in os.environ else -1

@DETECTORS.register_module()
class SubQueryDeformableRCNN(TwoStageDetector):

    def __init__(self, *args, **kwargs):
        super(SubQueryDeformableRCNN, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'SubQueryDeformableRCNN do not support external proposals'

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

        assert proposals is None, 'SubQueryDeformableRCNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'SubQueryDeformableRCNN does not instance segmentation'

        if DEBUG >= 0:
            for n,p in self.named_parameters():
                p.requires_grad = False

        x = self.extract_feat(img)
        
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(x, img_metas)
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
            # sub_xy=None,
            # sub_z=None,
            # subquery_vec=None,
        )
        
        
        return roi_losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale,
            # sub_xy=None,
            # sub_z=None,
            # subquery_vec=None,
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
