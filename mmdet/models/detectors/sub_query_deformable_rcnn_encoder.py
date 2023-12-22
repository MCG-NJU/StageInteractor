import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..utils.transformer import DeformableDetrTransformer
from mmcv import ConfigDict
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence


def get_valid_ratio(mask):
    """Get the valid radios of feature maps of all  level."""
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


@DETECTORS.register_module()
class SubQueryDeformableRCNNEnc(TwoStageDetector):

    def __init__(self, *args, **kwargs):
        super(SubQueryDeformableRCNNEnc, self).__init__(*args, **kwargs)

        self.positional_encoding = build_positional_encoding(ConfigDict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5))
        self.encoder = build_transformer_layer_sequence(ConfigDict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=ConfigDict(
                type='BaseTransformerLayer',
                attn_cfgs=ConfigDict(
                    type='MultiScaleDeformableAttention', embed_dims=256),
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))))
        self.level_embeds = nn.Parameter(torch.Tensor(4, 256))

        assert self.with_rpn, 'SubQueryDeformableRCNNEnc do not support external proposals'

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

        assert proposals is None, 'SubQueryDeformableRCNNEnc does not support' \
                                  ' external proposals'
        assert gt_masks is None, 'SubQueryDeformableRCNNEnc does not instance segmentation'

        mlvl_feats = self.extract_feat(img)

        batch_size = mlvl_feats[0].size(0)

        input_img_h, input_img_w = tuple(img[0].size()[-2:])
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_pos_embeds.append(self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = DeformableDetrTransformer.get_reference_points(
            spatial_shapes,
            valid_ratios,
            device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        level_start_index_cpu = level_start_index.cpu().numpy()
        spatial_shapes_cpu = spatial_shapes.cpu().numpy()

        mlvl_feats_w_encoder = []
        for lvl in range(len(mlvl_feats)):
            start = level_start_index_cpu[lvl]
            end = level_start_index_cpu[lvl + 1] if lvl < 3 else memory.shape[0]
            feat_h, feat_w = spatial_shapes_cpu[lvl]

            feat = memory[start:end]
            feat = feat.reshape(feat_h, feat_w, batch_size, 256)
            feat = feat.permute(2, 3, 0, 1)
            assert feat.shape == mlvl_feats[lvl].shape  # make sure the feat has the same shape with before

            mlvl_feats_w_encoder.append(feat)

        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.forward_train(mlvl_feats_w_encoder, img_metas)
        roi_losses = self.roi_head.forward_train(
            mlvl_feats_w_encoder,
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
        proposal_boxes, proposal_features, imgs_whwh = \
            self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        roi_outs = \
            self.roi_head.forward_dummy(
                x, proposal_boxes,
                proposal_features,
                dummy_img_metas,
            )
        return roi_outs
