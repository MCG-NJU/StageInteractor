from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class RPNSubQueryDeformableRCNN(TwoStageDetector):

    def __init__(self, *args, **kwargs):
        super(RPNSubQueryDeformableRCNN, self).__init__(*args, **kwargs)
        assert self.with_rpn, 'RPNSubQueryDeformableRCNN do not support external proposals'

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        assert proposals is None, 'RPNSubQueryDeformableRCNN does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'RPNSubQueryDeformableRCNN does not instance segmentation'

        x = self.extract_feat(img)
        
        losses = dict()
        
        proposal_boxes, proposal_features, \
        imgs_whwh, sub_xy, sub_z, subquery_vec, \
        rpn_losses = \
            self.rpn_head.forward_train(
                x, img_metas, 
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                gt_masks=gt_masks,
            )
        
        losses.update(rpn_losses)
        
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
            sub_xy=sub_xy,
            sub_z=sub_z,
            subquery_vec=subquery_vec,
        )
        losses.update(roi_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        proposal_boxes, proposal_features, \
        imgs_whwh, sub_xy, sub_z, subquery_vec = \
            self.rpn_head.simple_test_rpn(x, img_metas)
        
        bbox_results = self.roi_head.simple_test(
            x,
            proposal_boxes,
            proposal_features,
            img_metas,
            imgs_whwh=imgs_whwh,
            rescale=rescale,
            sub_xy=sub_xy,
            sub_z=sub_z,
            subquery_vec=subquery_vec,
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
        proposal_boxes, proposal_features, \
        imgs_whwh, sub_xy, sub_z, subquery_vec = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        # roi_head
        roi_outs = \
            self.roi_head.forward_dummy(
                x, proposal_boxes,
                proposal_features,
                dummy_img_metas,
                sub_xy=sub_xy,
                sub_z=sub_z,
                subquery_vec=subquery_vec,
            )
        return roi_outs
