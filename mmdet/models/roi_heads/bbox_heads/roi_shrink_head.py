import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.runner import auto_fp16, force_fp32

from mmdet.core import multi_apply, bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_transformer
from .bbox_head import BBoxHead
from mmdet.core import bbox_overlaps
import math
import numpy as np

import os

DEBUG = 'DEBUG' in os.environ



def relative_pe_interact_area(boxq, boxk, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    '''
        boxq: (N, B, 4), xyxy
        boxk: (M, B, 4), xyxy
    '''
    N, B = boxq.shape[:2]
    M = boxq.shape[0]
    boxq = boxq.view(N, 1, B, 4)
    boxk = boxk.view(1, M, B, 4)
    ans = interact_area(boxq, boxk)
    ans = ans.view(N*M, B, 4)
    ans = position_embedding_query(ans, num_feats, \
        max_box_size=max_box_size, temperature=temperature, ver=ver)
    ans = ans.view(N, M, B, 4, -1)
    return ans

def interact_area(box1, box2):
    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])

    wh = rb - lt
    cxcy = (rb + lt) / 2
    ans = torch.cat([cxcy, wh], -1)
    return ans
    

def distance_interact_area(A, B):
    '''
        F = AxB(A>0,B>0);
            AxB(A<0,B>0); 
            AxB(A>0,B<0); 
            -AxB(A<0,B<0); 
    '''
    relu = nn.ReLU(inplace=False)
    ans = -relu(-A) * relu(-B) + \
        relu(A) * relu(B) + \
        -relu(-A) * relu(B) + \
        -relu(A) * relu(-B)
    return ans
    

def calc_distri_distance(x, temper=2., neg_inf=-7.):
    # x: B, N, C
    B, N = x.shape[:2]
    pz = torch.exp(-0.5*(x*x).sum(-1)) / math.sqrt(2 * math.pi)
    pz = -torch.clamp(pz, min=1e-7).log().mean(-1)
    return pz
    
    

def dprint(*args, **kwargs):
    import os
    if 'DEBUG' in os.environ:
        print(*args, **kwargs)


def decode_box(xyzr):
    scale = 2.00 ** xyzr[..., 2:3]
    ratio = 2.00 ** torch.cat([xyzr[..., 3:4] * -0.5,
                              xyzr[..., 3:4] * 0.5], dim=-1)
    wh = scale * ratio
    xy = xyzr[..., 0:2]
    roi = torch.cat([xy - wh * 0.5, xy + wh * 0.5], dim=-1)
    return roi

def position_embedding_key(box, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    token_xyzr = box
    term = token_xyzr.new_tensor(
        [max_box_size, max_box_size, max_box_size, max_box_size]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].cos(), -pos_x[..., 1::2].sin()),
        dim=4).flatten(3)
    return pos_x

def position_embedding_query(box, num_feats, max_box_size=1, temperature=10000, ver='xyxy'):
    token_xyzr = box
    term = token_xyzr.new_tensor(
        [max_box_size, max_box_size, max_box_size, max_box_size]).view(1, 1, -1)
    token_xyzr = token_xyzr / term
    dim_t = torch.arange(
        num_feats, dtype=torch.float32, device=token_xyzr.device)
    dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
    pos_x = token_xyzr[..., None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
        dim=4).flatten(3)
    return pos_x

def position_embedding(box, num_feats, temperature=10000, ver='xyzr'):
    if box.size(-1) == 4 and ver == 'xyzr':
        token_xyzr = box
        term = token_xyzr.new_tensor([1000, 1000, 1, 1]).view(1, 1, -1)
        token_xyzr = token_xyzr / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=token_xyzr.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = token_xyzr[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    elif box.size(-1) == 3:
        token_xyzr = box
        term = token_xyzr.new_tensor([1000, 1000, 1]).view(1, 1, -1)
        token_xyzr = token_xyzr / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=token_xyzr.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = token_xyzr[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    elif box.size(-1) == 2:
        term = box.new_tensor([1000, ] * box.size(-1)).view(1, 1, -1)
        box = box / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=box.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = box[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
    else:
        term = box.new_tensor([1, ] * box.size(-1)).view(1, 1, -1)
        box = box / term
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=box.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = box[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
        # B, N, 4, num_feats
    return pos_x

class GroupLinear(nn.Linear):
    def __init__(self,
            in_features: int, 
            out_features: int, 
            bias: bool = True,
            groups = 1,) -> None:
        
        
        super(GroupLinear, self).__init__(
            in_features=in_features,
            out_features=out_features // groups,
            bias=bias,
        )
        self.groups = groups
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        
    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight.view(self.out_features // self.groups, self.groups, -1)
        weight = weight.permute(1, 2, 0) # G, C//G, Co//G
        x = input.view(-1, self.groups, input.shape[-1] // self.groups) # B*N*M, G, C//G
        x = x.permute(1, 0, 2) # G, B*N*M, C//G

        x = torch.bmm(x, weight) # G, B*N*M, Co//G
        x = x.permute(1, 0, 2) #B*N*M, G, Co//G
        x = x.reshape(*input.shape[:-1], -1) #B,N,M, Co
        x = x + self.bias
        return x

@HEADS.register_module()
class RoiShrinkHead(BBoxHead):
    _DEBUG = -1
    
    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=1,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 in_points=32,
                 out_points=128,
                 n_heads=4,
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_length_token=None,
                 loss_box2qid=None,
                 init_cfg=None,
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 N_scale=4,
                 anchor_point_num=8,
                 anchor_channel=64,
                 roi_inpoint_h = 7,
                 roi_inpoint_w = 7,
                 in_point_sampling_rate = 1,
                 subbox_poolsize = 3,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(RoiShrinkHead, self).__init__(
            num_classes=num_classes,
            reg_decoded_bbox=True,
            reg_class_agnostic=True,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_iou = build_loss(loss_iou)
        self.content_dim = content_dim
        self.fp16_enabled = False

        self.ffn = FFN(
            content_dim,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout)
        self.ffn_norm = build_norm_layer(dict(type='LN'), content_dim)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.cls_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))

        # over load the self.fc_cls in BBoxHead
        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(content_dim, self.num_classes)
        else:
            self.fc_cls = nn.Linear(content_dim, self.num_classes + 1)


        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(content_dim, content_dim, bias=True))
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), content_dim)[1])
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True)))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 4)
        #self.fc_reg = nn.Linear(content_dim, 4 * num_queries)

        self.in_points = in_points
        self.n_heads = n_heads
        self.out_points = out_points

        self.anchor_point_num = anchor_point_num
        self.anchor_channel = anchor_channel
        self.feat_extractor = \
            SubqueryFeatureExtractor(
                content_dim,
                in_points,
                C_sub_q=feat_channels//n_heads,
                G_sub_q=n_heads,
                N_scale=N_scale,
                dim_feedforward=feedforward_channels,
                anchor_point_num=anchor_point_num,
                anchor_channel=anchor_channel,
                num_heads=num_heads,
                dropout=dropout,
                num_queries=num_queries,
                subbox_poolsize=subbox_poolsize,
            )
        
        self.dynamic_conv = DynamicConv(
            in_dim=feat_channels, 
            in_points=in_points, 
            p_groups=n_heads, 
            num_queries=num_queries,
            num_interact_heads=num_interact_heads,
            num_interact_channel_groups=num_interact_channel_groups,
            dim_feedforward=feedforward_channels,
            query_dim=content_dim,
            out_points=out_points, 
            out_dim=feat_channels, 
            sampling_rate=in_point_sampling_rate, 
            roi_inpoint_h=roi_inpoint_h,
            roi_inpoint_w=roi_inpoint_w,
            num_heads=num_heads,
            dropout=dropout,
            feedforward_channels=feedforward_channels,
            num_ffn_fcs=num_ffn_fcs,
            ffn_act_cfg=ffn_act_cfg,
            subbox_poolsize=subbox_poolsize,
        )
        
        #self.fc_last_cls_reverse = nn.Linear(self.num_classes, content_dim)
        #self.attention = DyPWAttenDW(content_dim, num_heads, dropout)
        #num_heads = 32
        #self.attention = LocalSemanticAtt(content_dim, num_heads, dropout, num_queries, self.num_classes)
        #self.attention = ReLUAtten(content_dim, num_heads, dropout, num_queries, self.num_classes)
        #self.attention = SpatialGroupAtt(content_dim, num_heads, dropout, num_queries, self.num_classes)
        #self.attention = SpatialConsistencyAtt(content_dim, num_heads, dropout, num_queries, self.num_classes)
        self.attention = MultiheadAttention(content_dim, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        self.iof_tau = nn.Parameter(torch.ones(num_heads, ))
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)
        
        self.iof_gamma = nn.Parameter(torch.ones(num_heads, ))
        nn.init.constant_(self.iof_gamma, -7)
        
        
        self.l1_tau = nn.Parameter(torch.ones(num_heads, ))
        nn.init.uniform_(self.l1_tau, 0.0, 4.0)
        
        self.cls_tau = nn.Parameter(torch.ones(num_heads, ))
        nn.init.uniform_(self.cls_tau, 0.0, 4.0)
        
        #self.local_fusion = LocalSemanticAttFusion(content_dim, num_heads, dropout, num_queries)
        #self.local_fusion = LocalSemanticAtt(content_dim, num_heads, dropout, num_queries)
        #self.local_fusion = MultiheadAttention(content_dim, num_heads, dropout)
        #self.local_fusion_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        # self.local_fusion = Ada_projs(content_dim, n_heads, num_queries) #n_heads
        # #self.local_fusion = MultiheadAttention(content_dim, num_heads, dropout)
        # #self.local_fusion_norm = build_norm_layer(dict(type='LN'), content_dim)[1]
        
        # self.attention2 = MultiheadAttention(content_dim, num_heads, dropout)
        # self.att2_tau = nn.Parameter(torch.ones(num_queries, ))
        # nn.init.constant_(self.att2_tau, 0.0)
        
        #self.ranking_maker = RankingMaker(content_dim, num_queries)
        
        if loss_length_token is not None:
            self.loss_length_token = build_loss(loss_length_token)
            self.fc_cls_length = nn.Linear(content_dim, num_queries)
        else:
            self.loss_length_token = None
        
        if loss_box2qid is not None:
            self.loss_box2qid = build_loss(loss_box2qid)
            self.fc_box2qid = nn.Linear(content_dim, num_queries)
            self.box2qid_fcs = nn.ModuleList()
            for _ in range(num_reg_fcs):
                self.box2qid_fcs.append(
                    nn.Linear(content_dim, content_dim, bias=True))
                self.box2qid_fcs.append(
                    build_norm_layer(dict(type='LN'), content_dim)[1])
                self.box2qid_fcs.append(
                    build_activation_layer(dict(type='ReLU', inplace=True)))
        else:
            self.loss_box2qid = None
        
        #query_id_permute = np.random.choice(num_queries, num_queries)
        query_id_permute = np.arange(num_queries)
        query_id_permute = torch.from_numpy(query_id_permute).long()
        self.register_buffer('query_id_permute', query_id_permute)
        
        self.act = nn.ReLU(inplace=True)
        

    @torch.no_grad()
    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        super(RoiShrinkHead, self).init_weights()
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        
        if self.loss_box2qid is not None:
            nn.init.zeros_(self.fc_box2qid.weight)
            nn.init.zeros_(self.fc_box2qid.bias)
        
        #nn.init.zeros_(self.fc_last_cls_reverse.weight)
        #nn.init.zeros_(self.fc_last_cls_reverse.bias)


        self.feat_extractor.init_weights()
        self.dynamic_conv.init_weights()
        
        #self.local_fusion.init_weights()
        self.attention.init_weights()
        #self.ranking_maker.init_weights()

    @auto_fp16()
    def forward(self,
                x,
                xyzr,
                query_content,
                featmap_strides,
                sub_query_xy,
                sub_query_z,
                sub_query_vec,
                imgs_whwh,
                cls_logit = None,
                bbox_feats=None,
                roialign_func=None,
                xyzr_undetach=None,
                nll=None,
                nll_cls=None,
                ):
        '''
            imgs_whwh: (bs, 4)
            bbox_feats: (B*N, C, Pools, Pools): 200, 256, 7, 7
        '''

        RoiShrinkHead._DEBUG += 1


        P = self.in_points
        G = self.n_heads
        AN = self.anchor_point_num
        B, N = query_content.shape[:2]
        xyzr = xyzr.reshape(B, N, 4)
        
        cls_logit = cls_logit.view(B, N, -1)
        
        #query_content, xyzr = self.ranking_maker(query_content, xyzr)
        
        with torch.no_grad():
            pe = position_embedding(xyzr, query_content.size(-1) // 4)
            rois = decode_box(xyzr)
            roi_box_batched = rois.view(B, N, 4)
            roi_wh_batched = roi_box_batched[..., 2:] - roi_box_batched[..., :2]
            roi_whwh_batched = torch.cat([roi_wh_batched, roi_wh_batched], -1)
            
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[
                :, None, :, :]

            l1 = -torch.norm(
                (roi_box_batched.view(B, N, 1, 4) - 
                roi_box_batched.view(B, 1, N, 4)) / 
                roi_whwh_batched.view(B, N, 1, 4), 
                p = 1, dim = -1) / 4
            l1 = l1.view(B, 1, N, N)
            #l1 = (l1 + 1e-7).log()
            
            roi_box_batched = roi_box_batched / imgs_whwh #B,N,4
            
            # tril_mask = 0
            # tril_mask = torch.ones_like(iof[0, 0, :, :])
            # tril_mask = torch.tril(tril_mask).view(1, 1, N, N)
            # #iof = iof * tril_mask
            # tril_mask = (tril_mask + 1e-7).log()
            if self.loss_length_token is not None:
                iof[:, :, :-1, -1] = 0.
        
        if DEBUG:
            torch.save(iof, './demo/iof_{}.pth'.format(RoiShrinkHead._DEBUG))
            torch.save(self.iof_tau, './demo/iof_tau_{}.pth'.format(RoiShrinkHead._DEBUG))
        
        x_iof = iof
        
        #iof = (iof + torch.sigmoid(self.iof_gamma).view(1, -1, 1, 1)).log() #####
        #iof = ((iof>0)*iof + 1e-7).log()
        #iof = ((iof>0.5)*iof + 1e-7).log()
        #iof = iof
        iof = (iof + 1e-7).log()
        
        # l2_cls_logit = torch.norm(cls_logit, p=2, dim=-1)
        # cls_bias = torch.bmm(cls_logit, cls_logit.permute(0, 2, 1))
        # cls_bias = cls_bias / l2_cls_logit.view(B, N, 1) / l2_cls_logit.view(B, 1, N)
        # cls_bias = cls_bias.view(B, 1, N, N)
        
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)) # B, N_head=8, N, N
        # attn_bias = attn_bias + tril_mask
        
        # iof_tau = self.iof_tau
        # iof_tau = iof_tau + 1
        # attn_bias = (iof ** iof_tau.view(1, -1, 1, 1)) # B, N_head=8, N, N
        
        
        #print(iof, l1, cls_bias)
        #attn_bias = attn_bias + (l1 * self.l1_tau.view(1, -1, 1, 1))
        # attn_bias = attn_bias + (cls_bias * self.cls_tau.view(1, -1, 1, 1)) ###
        
        #query_content_attn_qk = pe + self.fc_last_cls_reverse(cls_logit)
        #query_content_attn_qk = query_content_attn_qk.permute(1, 0, 2)
        
        #query_content_sp = query_content.permute(2, 0, 1)
        #query_content_sp = self.attention_sp(
        #    query_content_sp,
        #    key=query_content_sp,
        #    value=query_content_sp,
        #    identity=query_content_sp,
        #)
        #query_content_sp = self.attention_norm_sp(query_content_sp)
        #query_content_sp = query_content_sp.permute(1, 2, 0)
        
        
        
        #query_content_attn = query_content
        query_content_attn = query_content + pe #+ self.fc_last_cls_reverse(cls_logit)
        query_content = query_content_attn
        #'''
        
        ##cls_logit_detach = cls_logit.detach()
        #rank_embed = self.ranking_maker(cls_logit)
        #query_content_attn = query_content_attn + rank_embed
        
        #query_content_attn_qk = query_content_attn
        # query_content_attn_qk = self.local_fusion(query_content_attn, x_iof)
        # query_content_attn_qk = query_content_attn_qk.permute(1, 0, 2)
        # #query_content_attn_qk = self.local_fusion(query_content_attn_qk, attn_mask=attn_bias.flatten(0, 1),)
        # #query_content_attn_qk = self.local_fusion_norm(query_content_attn_qk)
        query_content_attn = query_content_attn.permute(1, 0, 2)
        #query_content = query_content.permute(1, 0, 2)
        #query_content_attn = self.local_fusion(query_content_attn)
        query_content = self.attention(
            query_content_attn,
            key=query_content_attn,
            value=query_content_attn,
            attn_mask=attn_bias.flatten(0, 1),
            identity=query_content_attn,
        ) #iou=x_iof,
        # xyxy_q=roi_box_batched.permute(1,0,2),
        # xyxy_k=roi_box_batched.permute(1,0,2),
        #cls_logit=cls_logit.permute(1, 0, 2),
        
        #att2_s = torch.sigmoid(self.att2_tau).view(N, 1, 1)
        #query_content = self.attention_norm((1-att2_s) * query_content + att2_s * query_content2)
        query_content = self.attention_norm(query_content)
        
        query_content = query_content.permute(1, 0, 2)
        
        #'''
        
        
        # query_content = query_content.permute(1, 0, 2)
        # query_content = self.local_fusion(
        #     query_content,
        #     key=query_content,
        #     value=query_content,
        #     attn_mask=attn_bias.flatten(0, 1),
        #     identity=query_content,
        # )
        # # query_content = self.local_fusion_norm(query_content)
        # query_content = query_content.permute(1, 0, 2)
        
        
        #query_content = query_content + query_content_sp
        
        #query_content = self.ffn_norm2(self.ffn2(query_content))


        ''' adaptive 3D sampling and mixing '''
        feats, sub_xy, sub_z, \
        sub_query_vec, query_content, \
        sample_points_xy, offset, sample_points_z, scale_logit, \
        subbox_feat, subbox_feat_xy = \
            self.feat_extractor(
                x,
                featmap_strides,
                query_content,
                xyzr,
                sub_query_xy,
                sub_query_z,
                sub_query_vec,
                imgs_whwh,
                roialign_func=roialign_func,
            )
        
        query_content, query_content_cls = \
            self.dynamic_conv(
                feats, 
                query_content, 
                sample_points_xy, 
                offset, 
                sample_points_z, 
                scale_logit, 
                xyzr,
                subbox_feat_xy,
                imgs_whwh,
                bbox_feats=subbox_feat,
            )

        # FFN
        if nll is not None:
            query_content = nll(query_content)
        else:
            query_content = self.ffn_norm(self.ffn(query_content))
        
        query_content = query_content.view(B, N, -1)
        
        #query_content_cls = self.ffn_norm_cls(self.ffn_cls(query_content_cls))
        #cls_feat = query_content_cls
        
        cls_feat = query_content
        reg_feat = query_content
        
        if nll_cls is not None:
            for cls_layer in nll_cls[:-1]:
                cls_feat = cls_layer(cls_feat)
            
            cls_score = nll_cls[-1](cls_feat).view(B, cls_feat.shape[1], -1)
        else:
            for cls_layer in self.cls_fcs:
                cls_feat = cls_layer(cls_feat)
            
            cls_score = self.fc_cls(cls_feat).view(B, cls_feat.shape[1], -1)
        
        if self.loss_length_token is not None:
            cls_length = cls_feat[:, -1, :]
            cls_length = self.fc_cls_length(cls_length).view(B, -1)
        else:
            cls_length = None
        
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)
        box_delta = self.fc_reg(reg_feat).view(B, reg_feat.shape[1], -1)
        #box_delta = self.fc_reg(reg_feat).view(B, reg_feat.shape[1], N, -1)
        #ids_delta = torch.arange(reg_feat.shape[1]).to(box_delta.device)
        #box_delta = box_delta[:, ids_delta, ids_delta, :]
        
        if self.loss_length_token is not None:
            box_delta[:, -1, :] = 0.
            
        if self.loss_box2qid is not None:
            #box2qid = query_content.clone().detach().view(B, N, -1)
            box2qid = query_content.clone().view(B, N, -1)
            box2qid = box2qid + pe
            for box2qid_fcs_layer in self.box2qid_fcs:
                box2qid = box2qid_fcs_layer(box2qid)
            
            box2qid = self.fc_box2qid(box2qid)
        else:
            box2qid = None

        return cls_score, box_delta, query_content.view(B, N, -1), \
            sub_xy, sub_z, sub_query_vec, cls_length, box2qid

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        # z = xyzr[..., 2:3]
        # new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        # new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        # xyzr = torch.cat([new_xy, new_zr], dim=-1)
        z = xyzr[..., 2:3]
        r = xyzr[..., 3:4]
        new_x = xyzr[..., 0:1] + xyzr_delta[..., 0:1] * (2 ** (z - 0.5*r))
        new_y = xyzr[..., 1:2] + xyzr_delta[..., 1:2] * (2 ** (z + 0.5*r))
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_x, new_y, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr
    
    @force_fp32(apply_to=('deltas', ))
    def refine_xyxy(self, boxes, deltas):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = 2.0, 2.0, 1.0, 1.0
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(100000.0 / 16))
        dh = torch.clamp(dh, max=math.log(100000.0 / 16))

        pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
        pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes

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
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos)
        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(cls_score[pos_inds],
                                             labels[pos_inds])
        if bbox_pred is not None:
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                pos_bbox_pred = bbox_pred.reshape(bbox_pred.size(0),
                                                  4)[pos_inds.type(torch.bool)]
                imgs_whwh = imgs_whwh.reshape(bbox_pred.size(0),
                                              4)[pos_inds.type(torch.bool)]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred / imgs_whwh,
                    bbox_targets[pos_inds.type(torch.bool)] / imgs_whwh,
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
                losses['loss_iou'] = self.loss_iou(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=avg_factor)
            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
                losses['loss_iou'] = bbox_pred.sum() * 0
        
        if cls_length is not None:
            B = cls_length.shape[0]
            length_labels = pos_inds.view(B, -1).sum(-1)
            length_labels = length_labels.clamp(max=cls_length.shape[-1]-1).long()
            #length_labels = F.one_hot(length_labels, num_classes=cls_length.shape[-1])
            
            losses['loss_length_token'] = self.loss_length_token(
                cls_length, length_labels,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            
            losses['acc_cls_length'] = accuracy(cls_length, length_labels)
        
        if box2qid is not None:
            B, N = box2qid.shape[:2]
            pos_query_id = pos_inds.view(B, -1)
            batch_id, pos_query_id = torch.where(pos_query_id > 0)
            #pos_query_id = pos_query_id.view(-1)
            #box2qid = box2qid.view(-1, box2qid.size(-1))
            sampled_box2qid = box2qid[batch_id, pos_query_id]
            label_pos_query_id = self.query_id_permute[pos_query_id]
            #print(pos_query_id, label_pos_query_id)
            losses['loss_box2qid'] = self.loss_box2qid(
                sampled_box2qid, label_pos_query_id,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc_box2qid'] = accuracy(sampled_box2qid, label_pos_query_id)
        
        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_bboxes, neg_bboxes,
                           pos_gt_bboxes, pos_gt_labels, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples,),
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




class DynamicConv(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 num_interact_heads=4,
                 num_interact_channel_groups=4,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 gfeat_groups_c = None,
                 gfeat_groups_p = None,
                 roi_inpoint_h = 7,
                 roi_inpoint_w = 7,
                 num_heads=8,
                 dropout=0.,
                 feedforward_channels=2048,
                 num_ffn_fcs=2,
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 subbox_poolsize=9,
                 final_inpoint=32,
                 ):
        '''
            in_dim, out_dim: dim of featmap
        '''
        super(DynamicConv, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        
        gfeat_groups_c = gfeat_groups_c if gfeat_groups_c is not None else p_groups
        gfeat_groups_p = gfeat_groups_p if gfeat_groups_p is not None else p_groups
        
        gfeat_groups_c = 4
        gfeat_groups_p = 8
        
        # out_points = in_points
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        self.gfeat_groups_c = gfeat_groups_c
        self.gfeat_groups_p = gfeat_groups_p
        
        self.subbox_poolsize = subbox_poolsize
        #self.subbox_s2c = num_heads # 8 # 4
        self.subbox_s2c = subbox_poolsize # 8 # 4
        
        #self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups) * (out_points // in_points)
        #self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups) * self.subbox_s2c ###!!!!!
        self.m_parameters = (in_dim // p_groups) * (out_dim // p_groups)
        #self.m_parameters = 0
        
        
        #self.s_parameters = (in_dim // p_groups) * out_points 
        #self.s_parameters = (final_inpoint // sampling_rate) * out_points
        #self.s_parameters = (in_points // sampling_rate) * out_points 
        self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * out_points
        #self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * out_points //self.subbox_poolsize
        
        
        #self.s_parameters_shrink = 0
        #self.s_parameters_shrink = subbox_poolsize * final_inpoint // in_points * in_points
        
        #self.m_parameters_res = 0
        #self.m_parameters_res = (in_dim // p_groups) * (out_dim // p_groups)
        
        
        self.total_parameters = \
            self.m_parameters + self.s_parameters #+ \
            #self.m_parameters_res #+ self.s_parameters_shrink
        
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        self.Wv = nn.Linear(out_points*out_dim, out_dim, bias=True)

        # self.m_filter_generator = nn.Sequential(
        #     nn.Linear(query_dim, self.m_parameters * p_groups),
        # )
        # self.s_filter_generator = nn.Sequential(
        #     nn.Linear(query_dim, self.s_parameters * p_groups),
        # )
        
        self.act = nn.ReLU(inplace=True)
        
        
        self.out_points_cls = out_points
        self.temper = (in_dim // p_groups) ** 0.5
        
        
        self.subbox_cross_box_param = 0
        #self.subbox_cross_box_param = self.subbox_s2c * in_points * in_points
        
        #self.subbox_dw_param = subbox_poolsize * self.subbox_s2c
        self.subbox_dw_param = in_points * subbox_poolsize * self.subbox_s2c
        
        self.subbox_pw_param = (in_dim // p_groups) * (in_dim // p_groups)
        self.total_subbox_parameters = self.subbox_dw_param + self.subbox_pw_param #+ self.subbox_cross_box_param
        self.parameter_generator = nn.Sequential(
            nn.Linear(query_dim, p_groups * (self.total_subbox_parameters + self.total_parameters)),
            #nn.Linear(query_dim, p_groups * self.total_subbox_parameters),
        )
        
        #self.parameter_generator2 = nn.Sequential(
        #    nn.Linear(query_dim, p_groups * self.total_parameters),
        #)
        
        
        self.subbox_local = nn.Sequential(
            GroupLinear(in_dim * self.subbox_s2c, in_dim * self.subbox_poolsize, groups=p_groups)
        )
        

        self.subbox_score = nn.Sequential(
            nn.Linear(query_dim, p_groups * in_points * subbox_poolsize),
        )
        
        
        #self.q2p_att = QueryAllPointAttention(in_dim, p_groups)
        
        
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        #nn.init.zeros_(self.parameter_generator2[-1].weight)
        
        nn.init.zeros_(self.parameter_generator[-1].weight)
        nn.init.zeros_(self.subbox_local[-1].weight)
        
        nn.init.zeros_(self.subbox_score[-1].weight)
        nn.init.zeros_(self.subbox_score[-1].bias)

    def forward(self, feats, query_vec, \
      sample_points_xy, offset, sample_points_z, \
      scale_logit, query_box, subbox_feat_xy, imgs_whwh, bbox_feats=None):
        '''
            feats (B, N, G, P, C//G)
            offset (B, N, P, G, 2)
            scale_logit (B, N, P, G)
            query_box (B, N, 4)
            origin, bbox_feats: (B*N, C, Pools, Pools): 200, 256, 7, 7
            subbox, bbox_feats: B, N, G, (P(//9)) *9, C_map//G
        '''
        sc = None
        B, N, G, P, C_map = feats.shape
        C_map *= G
        feats = feats.view(B*N, G, P, -1)
        feats_x = feats

        feats_reg = feats
        feats_cls = feats
        if bbox_feats is not None:
            feats_cls = bbox_feats.view(B*N, G, -1, C_map//G)
            feats_reg = feats_cls ################
            feats_cls = feats_cls.view(
                B*N, G, P, self.subbox_poolsize, C_map//G
            )
            feats_ori = feats_cls
            feats = feats_cls[:, :, :, 0, :]
            feats_align = feats_cls.mean(-2)

        
        ###################################
        #query_vec_relu = self.act(query_vec)
        #params = self.parameter_generator(query_vec_relu)
        params = self.parameter_generator(query_vec)
        params = params.reshape(B*N, G, -1)
        subbox_pwconv, subbox_dwconv, M, S = params.split(
            [self.subbox_pw_param, self.subbox_dw_param, \
                self.m_parameters, self.s_parameters, \
            ], 2
        )
        #subbox_pwconv, subbox_dwconv = params.split(
        #    [self.subbox_pw_param, self.subbox_dw_param, \
        #    ], 2
        #)

        ###################################
        
        subbox_score = self.subbox_score(query_vec)
        subbox_score = subbox_score.view(B*N, G, P, self.subbox_poolsize, -1)
        subbox_score = F.softmax(subbox_score, -1)
        feat_res = feats_cls * subbox_score
        
        
        ###################################
        
        subbox_pwconv = subbox_pwconv.view(B*N*G, self.in_dim // G, -1)
        
        feats_subbox = feats_reg.view(B*N*G, -1, C_map//G) #B*N, G, P*9, C_map//G
        
        feats_pw = torch.bmm(feats_subbox, subbox_pwconv) #B*N,G,P*9,(in_dim//G)  ### B*N,G,P,9,(in_dim//G)//K
        
        feats_pw = feats_pw.view(B*N*G, P, -1, self.in_dim // G)
        feats_pw = feats_pw.view(B*N*G, -1, self.in_dim // G)
        
        feats_pw = F.layer_norm(
            feats_pw, 
            [feats_pw.size(-2), feats_pw.size(-1)]
        )
        feats_pw = self.act(feats_pw)
        

        subbox_dwconv = subbox_dwconv.reshape(B*N*G*P, self.subbox_poolsize, -1)
        feats_pw = feats_pw.view(
            B*N*G*P, self.subbox_poolsize, -1
        )
        feats_pw = feats_pw.permute(0, 2, 1).contiguous() # B*N*G*P, (in_dim // G), 9
        feats_pwdw = torch.bmm(feats_pw, subbox_dwconv) # B*N*G*P, (in_dim // G), K
        feats_pwdw = F.layer_norm(
            feats_pwdw, 
            [feats_pwdw.size(-2), feats_pwdw.size(-1)]
        )
        feats_pwdw = self.act(feats_pwdw)
        feats_pwdw = feats_pwdw.view(B*N, G, P, -1) # B*N,G,P, K*(in_dim // G)
        
        
        
        feats_pwdw = feats_pwdw.permute(0, 2, 1, 3).contiguous().view(B*N, P, -1)
        feats_pwdw = self.subbox_local(feats_pwdw)
        feats_pwdw = feats_pwdw.view(B*N, P, G, -1)
        feats_pwdw = feats_pwdw.permute(0, 2, 1, 3).contiguous()
        
        feats_pwdw = feats_pwdw.view(B*N, G, P, self.subbox_poolsize, -1)
        
        feats_reg = feats_pwdw + feat_res
        
        feats = feats_reg
        
        
        
        #query_vec = self.q2p_att(query_vec, feats, subbox_feat_xy, query_box, imgs_whwh)
        #params = self.parameter_generator2(query_vec)
        #params = params.reshape(B*N, G, -1)
        #M, S = params.split(
        #    [self.m_parameters, self.s_parameters, \
        #    ], 2
        #)
        
        
        ###################################
        
        feats = feats.view(B*N*G, -1, C_map//G)
        
        
        
        M = M.reshape(
            B*N*G, self.in_dim // G, -1) #B*N, G, self.in_dim//G, self.out_dim//G
        feats_M = torch.bmm(feats, M) # B*N, G, P, outdim//G
        
        feats_M = F.layer_norm(feats_M, [feats_M.size(-2), feats_M.size(-1)])
        feats_M = self.act(feats_M)
        
        
        
        S = S.reshape(
            B*N*G, self.out_points, -1)
        


        feats_MS = torch.bmm(S, feats_M) # B*N, G, outP, outdim//G
        feats_MS = feats_MS.view(B*N*G, -1, self.in_dim // G)
        feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
        feats_MS = self.act(feats_MS)
        # B*N, G, out_points, out_dim // G
        
        
        
        
        feats_MS_flat = feats_MS.reshape(B, N, -1)
        feats_MS_flat = self.Wv(feats_MS_flat)


        feats_MS_q = self.Wv_layer_norm(query_vec + feats_MS_flat)

        ###################################
        
        feats_reg = feats_MS_q
        feats_cls = feats_MS_q
        
        
        if DEBUG:
            torch.save(S, './demo/S_{}.pth'.format(DynamicConv.IND))
            torch.save(M, './demo/M_{}.pth'.format(DynamicConv.IND))
            torch.save(subbox_pwconv, './demo/subbox_pwconv_{}.pth'.format(DynamicConv.IND))
            torch.save(subbox_dwconv, './demo/subbox_dwconv_{}.pth'.format(DynamicConv.IND))
            torch.save(query_vec, './demo/query_vec_{}.pth'.format(DynamicConv.IND))
            torch.save(params, './demo/params_{}.pth'.format(DynamicConv.IND))
            torch.save(self.parameter_generator[-1].weight, './demo/params_w_{}.pth'.format(DynamicConv.IND))
            torch.save(self.parameter_generator[-1].bias, './demo/params_b_{}.pth'.format(DynamicConv.IND))
        
        
        DynamicConv.IND += 1
        return feats_reg, feats_cls




class SubqueryFeatureExtractor(nn.Module):
    IND = 0

    def __init__(self,
                 content_dim,
                 in_points,
                 C_sub_q=64,
                 G_sub_q=4,
                 N_scale=5,
                 dim_feedforward=1024,
                 anchor_point_num=8,
                 anchor_channel=64,
                 num_heads=8,
                 dropout=0.,
                 num_queries=100,
                 featmap_dim=None,
                 subbox_poolsize=9,
                 ):
        super(SubqueryFeatureExtractor, self).__init__()
        
        self.featmap_dim = content_dim if featmap_dim is None else featmap_dim
        
        self.G = G_sub_q
        self.Cq = C_sub_q
        self.in_points = in_points
        self.content_dim = content_dim
        self.num_channel_heads = num_heads
        self.subbox_poolsize = subbox_poolsize
        
        
        
        self.offset_generator = nn.Sequential(
            nn.Linear(content_dim, 2 * G_sub_q * in_points),
        )
        self.scale_generator = nn.Sequential(
            nn.Linear(content_dim, 1 * G_sub_q * in_points),
        )
        
        
        
        
        #self.dypw_attendw = \
        #    DyPWAttenDW(
        #        query_dim=content_dim, 
        #        p_groups=num_heads, 
        #        num_queries=num_queries,
        #        dim_feedforward=dim_feedforward,
        #    )
        
        self.anchor_num = anchor_point_num
        self.anchor_channel = anchor_channel #content_dim // G_sub_q
        '''
        self.anchor_offset_generator = nn.Sequential(
            nn.Linear(content_dim, 4 * self.anchor_num),
        )
        self.anchor_feat_generator = nn.Sequential(
            nn.Linear(content_dim, 
                self.anchor_channel * self.anchor_num),
        )
        
        self.anchor_feat2offset = nn.Sequential(
            nn.Linear(self.anchor_channel, 
                2 * G_sub_q * in_points // self.anchor_num),
        )
        '''
        
        '''
        #self.featmap_shrink = nn.Sequential(
        #    nn.Linear(content_dim, self.anchor_channel),
        #)
        local_dim_fnn = dim_feedforward * self.anchor_channel // content_dim
        self.local_attention = MultiheadAttention(
            self.anchor_channel, self.num_channel_heads, dropout)
        self.local_attention_norm = nn.LayerNorm([self.anchor_num, self.anchor_channel])
        self.local_ffn = nn.Sequential(
            nn.Linear(self.anchor_channel, local_dim_fnn),
            nn.ReLU(inplace=True),
            nn.Linear(local_dim_fnn, self.anchor_channel),
        )
        self.local_ffn_norm = nn.LayerNorm([self.anchor_num, self.anchor_channel])
        '''
        
        #self.stage_subquery_bboxes = \
        #    nn.Embedding(self.anchor_num, 4)
        '''
        self.stage_query_specific_bboxes = \
            nn.Embedding(num_queries * G_sub_q * in_points, 2)
        self.stage_query_specific_scale = \
            nn.Embedding(num_queries * G_sub_q * in_points, 1)
        '''
        
        '''
        self.current_feat_q = nn.Sequential(
            nn.Linear(self.featmap_dim // self.G, content_dim),
        )
        self.last_subvec_k = nn.Sequential(
            nn.Linear(content_dim // self.G, content_dim),
        )
        self.current_xy_q = nn.Sequential(
            nn.Linear(2, content_dim, bias=False),
        )
        self.last_xy_k = nn.Sequential(
            nn.Linear(2, content_dim, bias=False),
        )
        self.current_z_q = nn.Sequential(
            nn.Linear(1, content_dim, bias=False),
        )
        self.last_z_k = nn.Sequential(
            nn.Linear(1, content_dim, bias=False),
        )
        '''
        
        self.subbox_poolsize = subbox_poolsize
        self.subbox_generator = nn.Sequential(
            nn.Linear(content_dim, 4 * G_sub_q * in_points),
        )
        
        self.gen_one_d = nn.Sequential(
            nn.Linear(content_dim, 3 * G_sub_q * in_points),
        )
        self.gen_double_d = nn.Sequential(
            #nn.Linear(content_dim, 3 * G_sub_q * in_points * subbox_poolsize),
            nn.Linear(content_dim, 3 * subbox_poolsize),
        )
        
        kernel_indices = self.create_box_element_indices(subbox_poolsize)
        self.register_buffer('kernel_indices', kernel_indices)


        #self.interact_sampler = InteractPointId(content_dim, G_sub_q, dropout, in_points)

        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.scale_generator[-1].weight)
        nn.init.zeros_(self.scale_generator[-1].bias)
        bias = self.scale_generator[-1].bias.data
        bias.mul_(0.0)
        
        nn.init.zeros_(self.offset_generator[-1].weight)
        nn.init.zeros_(self.offset_generator[-1].bias)
        bias = self.offset_generator[-1].bias.data.view(
            self.G, self.in_points, -1)
        bias.mul_(0.0)
        if int(self.in_points ** 0.5) ** 2 == self.in_points:
            h = int(self.in_points ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias, -bandwidth, bandwidth)
        
        
        nn.init.zeros_(self.subbox_generator[-1].weight)
        nn.init.zeros_(self.subbox_generator[-1].bias)
        bias = self.subbox_generator[-1].bias.data.view(
            self.G, -1, 4)
        bias.mul_(0.0)
        if int((self.in_points * self.G) ** 0.5) ** 2 == self.in_points * self.G:
            h = int((self.in_points * self.G) ** 0.5)
            y = torch.linspace(-0.5, 0.5, h + 1) + 0.5 / h
            yp = y[:-1]
            y = yp.view(-1, 1).repeat(1, h)
            x = yp.view(1, -1).repeat(h, 1)
            y = y.flatten(0, 1)[None, :, None]
            x = x.flatten(0, 1)[None, :, None]
            bias[:, :, 0:2] = torch.cat([y, x], dim=-1).view(self.G, -1, 2)
        else:
            bandwidth = 0.5 * 1.0
            nn.init.uniform_(bias[:, :, :2], -bandwidth, bandwidth)
        
        
        nn.init.zeros_(self.gen_one_d[-1].weight)
        nn.init.zeros_(self.gen_one_d[-1].bias)
        bias = self.gen_one_d[-1].bias.data.view(
            self.G, -1, 3)
        bias.mul_(0.0)
        bandwidth = 0.5 * 1.0
        nn.init.uniform_(bias[:, :, :-1], -bandwidth, bandwidth)
        
        nn.init.zeros_(self.gen_double_d[-1].weight)
        nn.init.zeros_(self.gen_double_d[-1].bias)
        bias = self.gen_double_d[-1].bias.data.view(
            -1, self.subbox_poolsize, 3)
        bias.mul_(0.0)
        bandwidth = 0.5 * 1.0 / math.sqrt(2.)
        nn.init.uniform_(bias[:, 1:, :-1], -bandwidth, bandwidth)
        
        #self.interact_sampler.init_weights()


    def create_box_element_indices(self, kernel_size):
        if int(kernel_size**0.5)**2 == kernel_size:
            kernel_size = int(kernel_size**0.5)
            if kernel_size % 2 == 0:
                start_idx = -kernel_size // 2
                end_idx = kernel_size // 2

                indices = torch.linspace(start_idx + 0.5, end_idx - 0.5, kernel_size)
                
            else:
                start_idx = -(kernel_size - 1) // 2
                end_idx = (kernel_size - 1) // 2

                indices = torch.linspace(start_idx, end_idx, kernel_size)
                
            tmp = indices[0]
            indices[0] = indices[len(indices)//2]
            indices[len(indices)//2] = tmp
            
            i, j = torch.meshgrid(indices, indices)
            kernel_indices = torch.stack([j, i], dim=-1).view(-1, 2) / kernel_size
        else:
            delta_theta = 2 * math.pi / kernel_size
            ids = dim_t = torch.arange(
                kernel_size, dtype=torch.float32)
            ids = torch.clamp(ids - 1, min=0.)
            delta_thetas = delta_theta * ids
            i = delta_thetas.clone()
            j = delta_thetas.clone()
            i = 0.5 * i.cos()
            j = 0.5 * j.sin()
            i[0] = 0.
            j[0] = 0.
            kernel_indices = torch.stack([i, j], dim=-1).view(-1, 2)
        
        return kernel_indices
    
    def get_subbox_feat_dxdydz(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            sub_query_xy,
            sub_query_z,
            sub_query_vec,
            imgs_whwh,
            roialign_func,
        ):
        
        
        C_map = featmap_list[0].shape[1]
        num_levels = len(featmap_list)
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        
        ori_xyzr = query_box.view(B, N, 1, 1, 4)
        orix = ori_xyzr[..., 0:1]
        oriy = ori_xyzr[..., 1:2]
        oriz = ori_xyzr[..., 2:3]
        orir = ori_xyzr[..., 3:4]
        
        #dxdydz00, query_content = self.interact_sampler(query_content, query_box, featmap_list, featmap_strides)
        dxdydz00 = self.gen_one_d(query_content)
        dxdydz00 = dxdydz00.view(B, N, -1, 1, 3) # B, N, G*P, 1, 4
        dx0 = dxdydz00[..., 0:1]
        dy0 = dxdydz00[..., 1:2]
        dz0 = dxdydz00[..., 2:3]
        
        dxdydz01 = self.gen_double_d(query_content)
        dxdydz01 = dxdydz01.view(B, N, -1, self.subbox_poolsize, 3) # B, N, G*P, 9, 4
        dx1 = dxdydz01[..., 0:1]
        dy1 = dxdydz01[..., 1:2]
        dz1 = dxdydz01[..., 2:3]
        
        newx = orix + 2**(oriz - 0.5*orir) * (dx0 + 2**dz0 * (dx1))
        newy = oriy + 2**(oriz + 0.5*orir) * (dy0 + 2**dz0 * (dy1))
        newz = oriz + dz0 + dz1
        
        grid = torch.cat([newx, newy], -1)
        grid = grid.view(B, N, G, -1, 2)# B, N, G, P*9, 2
        
        # grid = grid.view(B*N*G, -1, 2)
        # grid = grid.permute(0, 2, 1).contiguous()
        # grid = F.layer_norm(grid, [grid.size(-1), ]) ######
        # grid = grid.permute(0, 2, 1).contiguous()
        # grid = grid.view(B, N, G, -1, 2)
        
        grid = grid.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P*9, 2
        grid = grid.view(B*G, -1, self.subbox_poolsize, 2) # B*G, N*P, 9, 2
        
        
        newz = newz.view(B, N, G, -1, 1)
        newz = newz.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P*9, 2
        newz = newz.view(B*G, -1, self.subbox_poolsize, 1) # B*G, N*P, 9, 1
        
        
        featmap_list = [i.reshape(B*G, -1, i.shape[-2], i.shape[-1]) for i in featmap_list]
        # B*G, 64, H, W

        weight_z = self.regress_z(
            newz[..., 0], 
            stride_size=len(featmap_strides), 
            tau=2.0, 
            mask_size=None,
        )
        
        sample_points_lvl_weight_list = weight_z.unbind(-1)
        
        
        if DEBUG:
            torch.save(grid, 
                'demo/grid_{}.pth'.format(SubqueryFeatureExtractor.IND))
            torch.save(weight_z, 
                'demo/weight_z_{}.pth'.format(SubqueryFeatureExtractor.IND))
            torch.save(newz, 
                'demo/real_z_{}.pth'.format(SubqueryFeatureExtractor.IND))
        
        sample_feature = weight_z.new_zeros(B, G, C_map//G, N, P*self.subbox_poolsize) ###
        for i in range(num_levels):
            Hk, Wk = featmap_list[i].shape[-2:]
            
            featmap = featmap_list[i] # B*G, 64, H, W
            lvl_weights = sample_points_lvl_weight_list[i]  # B*G, N*P*9
            
            stride = featmap_strides[i]
            mapping_size = featmap.new_tensor(
                [featmap.size(3), featmap.size(2)]) * stride
            mapping_size = mapping_size.view(1, 1, -1)
            
            
            normalized_xyxy = grid / mapping_size 
            
            normalized_xyxy = normalized_xyxy.view(B*G, N, -1, 2)
            normalized_xyxy = normalized_xyxy*2.0-1.0
            sample_feats = F.grid_sample(
                featmap, normalized_xyxy,
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False,
            ) # B*G, C//G, N, P

            sample_feats = sample_feats.view(B, G, C_map//G, N, -1, self.subbox_poolsize) # B, G, C//G, N, P ,9
            lvl_weights = lvl_weights.reshape(B, G, 1, N, sample_feats.shape[4], -1)  # B, G, 1, N, P, 9
            sample_feats *= lvl_weights
            
            sample_feats = sample_feats.view(B, G, C_map//G, N, -1)
            
            
            sample_feature += sample_feats
        
        # B, G, C_map//G, N, P
        sample_feature = sample_feature.permute(0, 3, 1, 4, 2).contiguous()
        #B, N, G, P, C_map//G
        
        grid = grid.view(B, G, N, -1, 2)
        
        return sample_feature, grid, query_content
    
    def get_subbox_feat(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            sub_query_xy,
            sub_query_z,
            sub_query_vec,
            imgs_whwh,
            roialign_func,
        ):
        
        
        C_map = featmap_list[0].shape[1]
        num_levels = len(featmap_list)
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        if self.training:
            imgs_whwh = imgs_whwh.view(B, N, 1, -1)
        else:
            imgs_whwh = imgs_whwh.reshape(B, 1, 1, 4)
            imgs_whwh = imgs_whwh.expand(B, N, 1, 4)
        
        xyzr_delta = self.subbox_generator(query_content)
        xyzr_delta = xyzr_delta.view(B, N, -1, 4) # B, N, G*P//9=4*36/9=16, 4
        ori_box = query_box.view(B, N, 1, 4)
        #xyzr_delta = xyzr_delta.view(B, -1, 4)
        #ori_box = ori_box.expand(B, N, xyzr_delta.shape[-2], 4).reshape(B, -1, 4)
        sub_query_box_xyzr = self.subbox_refine_xyzr(ori_box, xyzr_delta, imgs_whwh, return_bbox=False)
        
        sub_query_box_xyzr = sub_query_box_xyzr.view(B, N, G, -1, 4) # B, N, G, P//9, 4
        sub_query_box_xyzr = sub_query_box_xyzr.permute(0, 2, 1, 3, 4).contiguous() # B, G, N, P//9, 4
        sub_query_box_xyzr = sub_query_box_xyzr.view(B*G, -1, 4) # B*G, N*P//9, 4
        featmap_list = [i.reshape(B*G, -1, i.shape[-2], i.shape[-1]) for i in featmap_list]
        # B*G, 64, H, W

        weight_z = self.regress_z(
            sub_query_box_xyzr[:, :, 2], 
            stride_size=len(featmap_strides), 
            tau=2.0, 
            mask_size=None, #self.subbox_poolsize
        )
        
        sample_points_lvl_weight_list = weight_z.unbind(-1)
        
        sub_query_box_xyxy = decode_box(sub_query_box_xyzr) # B*G, N*P//9, 4
        sub_query_box_xyxy = sub_query_box_xyxy.view(B*G, -1, 1, 4)
        cxcy = 0.5 * (sub_query_box_xyxy[..., :2] + sub_query_box_xyxy[..., 2:])
        wh = sub_query_box_xyxy[..., 2:] - sub_query_box_xyxy[..., :2]
        kernel_indices = self.kernel_indices.view(1, 1, -1, 2)
        grid = cxcy + kernel_indices * wh # B*G, N*P//9, 9, 2
        
        if DEBUG:
            torch.save(sub_query_box_xyxy, 
                'demo/sub_query_box_xyxy_{}.pth'.format(SubqueryFeatureExtractor.IND))
            torch.save(weight_z, 
                'demo/weight_z_{}.pth'.format(SubqueryFeatureExtractor.IND))
        
        sample_feature = weight_z.new_zeros(B, G, C_map//G, N, P*self.subbox_poolsize) ###
        for i in range(num_levels):
            Hk, Wk = featmap_list[i].shape[-2:]
            
            featmap = featmap_list[i] # B*G, 64, H, W
            lvl_weights = sample_points_lvl_weight_list[i]  # B*G, N*P//9
            
            stride = featmap_strides[i]
            mapping_size = featmap.new_tensor(
                [featmap.size(3), featmap.size(2)]) * stride
            mapping_size = mapping_size.view(1, 1, -1)
            
            
            normalized_xyxy = grid / mapping_size 
            
            normalized_xyxy = normalized_xyxy.view(B*G, N, -1, 2)
            normalized_xyxy = normalized_xyxy*2.0-1.0
            sample_feats = F.grid_sample(
                featmap, normalized_xyxy,
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=False,
            ) # B*G, C//G, N, P

            sample_feats = sample_feats.view(B, G, C_map//G, N, -1, self.subbox_poolsize) # B, G, C//G, N, P//9 ,9
            lvl_weights = lvl_weights.reshape(B, G, 1, N, -1, 1)  # B, G, 1, N, P//9
            sample_feats *= lvl_weights
            
            sample_feats = sample_feats.view(B, G, C_map//G, N, -1)
            
            # C, Ph, Pw = sample_feats.shape[-3:]
            # sample_feats = sample_feats.view(B, G, N, -1, C, Ph, Pw)
            # sample_feats = sample_feats.permute(0, 2, 1, 3, 5, 6, 4).contiguous()
            # # B, N, G, P//(Ph*Pw), Ph, Pw, C
            # 
            # lvl_weights = lvl_weights.reshape(B, G, N, -1, 1, 1, 1)  # B, G, N, P//9, 1, 1, 1
            # lvl_weights = lvl_weights.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
            # sample_feats = sample_feats * lvl_weights
            # sample_feats = sample_feats.view(B, N, G, -1, C)
            # # B, N, G, P, C
            # 
            # sample_feats = sample_feats.permute(0, 2, 4, 1, 3).contiguous()
            
            sample_feature += sample_feats
        
        # B, G, C_map//G, N, P
        sample_feature = sample_feature.permute(0, 3, 1, 4, 2).contiguous()
        return sample_feature #B, N, G, P, C_map//G
        
    def get_point_feat(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            sub_query_xy,
            sub_query_z,
            sub_query_vec,
            imgs_whwh,
        ):
        
        box_cxcy, box_wh, logscale = \
            self.get_cxcy_wh_logscale(query_box, box_ver='xyzr')
        logscale = logscale.view(B, N, 1, 1)
        
        
        ############################
        
        #query_content = self.dypw_attendw(query_content_attn, attn_mask=attn_bias,)
        
            
        # Every sub-query shares a same filter, 
        # only its vector makes a difference 

        
        #pe = self.get_spatial_info(query_content, query_box)
        
        
        offset, sub_query_xy = self.anchor_xy(
            query_content, sub_query_vec, sub_query_xy)
        offset = offset.reshape(B, N, P, G, 2)
        
        
        scale_logit = self.scale_generator(query_content)
        scale_logit = scale_logit.reshape(B, N, P, G, -1)
        
        # query_scale = self.stage_query_specific_scale.weight
        # query_scale = query_scale[None].expand(B, *query_scale.size())
        # query_scale = query_scale.reshape(B, N, P, G, -1)
        # scale_logit = scale_logit + query_scale
        

        
        box_cxcy = box_cxcy.view(B, N, 1, 1, 2)
        sample_points_xy, delta_sample_points_xy = \
            self.make_sample_points_xy(
                offset, box_cxcy,
                box_wh,
            )
        # sample_points_z, delta_sample_points_z = \
        #     self.make_sample_points_z(
        #         scale_logit,
        #         box_wh, wh_image, featmap_strides,
        #     )
        scale_logit = scale_logit.reshape(B, N, P, G)
        sample_points_z = scale_logit + logscale
        sample_points_z_w = self.regress_z(sample_points_z)
        delta_sample_points_z=None

        sample_feats = self.feat_extract(sample_points_xy, \
            sample_points_z_w, featmap_list, featmap_strides)

        sub_query_vec, sample_feats = \
            self.intercept_local_feat(sample_feats, sub_query_vec, offset, scale_logit)
        #sample_feats = sub_query_vec
        return sample_feats
        
    
    def forward(self,
            featmap_list,
            featmap_strides,
            query_content,
            query_box,
            sub_query_xy,
            sub_query_z,
            sub_query_vec,
            imgs_whwh,
            roialign_func=None,
        ):
        '''
            query_content: B, N, (C = G * Cs)
            query_box: B, N, 4 : x1y1x2y2
            sub_query_xy: B, N, P, G, 2
            sub_query_z: B, N, P, G, S
            sub_query_vec: B, N, P, G, Cs
            wh_image: B, N, 2
            
            sample_feats: B, N, G, P, C_map//G
            sample_points_xy: B, N, P, G, 2
            sample_points_z: B, N, P, G, num_levels
        '''

        wh_image = imgs_whwh[..., :2]
        
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        
        
        subbox_feat = None
        
        sample_points_xy = None
        offset = None
        sample_points_z = None
        scale_logit = None
        subbox_feat_xy = None

        subbox_feat, subbox_feat_xy, query_content = \
            self.get_subbox_feat_dxdydz(
                featmap_list,
                featmap_strides,
                query_content,
                query_box,
                sub_query_xy,
                sub_query_z,
                sub_query_vec,
                imgs_whwh,
                roialign_func,
            ) #get_subbox_feat 
        
        if subbox_feat is not None:
            sample_feats = subbox_feat.new_zeros(B, N, G, P, subbox_feat.shape[-1])
        
        # sample_feats = \
        #     self.get_point_feat(
        #         featmap_list,
        #         featmap_strides,
        #         query_content,
        #         query_box,
        #         sub_query_xy,
        #         sub_query_z,
        #         sub_query_vec,
        #         imgs_whwh,
        #     )
        
        if DEBUG:
            torch.save(
                subbox_feat, 'demo/subbox_feat_{}.pth'.format(SubqueryFeatureExtractor.IND))
        
        SubqueryFeatureExtractor.IND += 1
        
        return sample_feats, sub_query_xy, sub_query_z, sub_query_vec ,\
            query_content, sample_points_xy, offset, sample_points_z, scale_logit, \
            subbox_feat, subbox_feat_xy
    
    def anchor_xy(self, query_content, sub_query_vec, sub_query_xy):
        B, N = query_content.shape[:2]
        P = self.in_points
        G = self.G
        '''
        sub_query_xy = sub_query_xy[..., :2]
        sub_query_xy = sub_query_xy.view(B, N, self.anchor_num, -1, 2)
        
        #anchor_offset = self.anchor_offset_generator(query_content)
        #anchor_offset = sub_query_xy.detach()
        anchor_offset = sub_query_xy
        
        #sub_xy = self.stage_subquery_bboxes.weight.clone()
        #sub_xy = sub_xy[None, None].expand(B, N, *sub_xy.size())
        #sub_xy = sub_xy.reshape(B, N, self.anchor_num, 1, 4)
        #anchor_offset = sub_xy
        
        anchor_offset = anchor_offset.reshape(B, N, self.anchor_num, -1, 2) #dx dy dz dr: B, N, self.anchor_num, 2
        beta = anchor_offset[...,:2]
        #gamma = torch.sigmoid(anchor_offset[...,2:])
        
        anchor_feat = self.anchor_feat_generator(query_content)
        sub_query_vec = sub_query_vec + anchor_feat.view(B, N, self.anchor_num, -1)
        
        anchorbase_offset = self.anchor_feat2offset(sub_query_vec)
        anchorbase_offset = anchorbase_offset.reshape(B, N, self.anchor_num, -1, 2)
        # B, N, AN(=8), G*P/AN, 2 
        
        offset = anchorbase_offset
        #offset = offset * gamma + beta
        ####offset = offset + beta ##
        
        #sub_query_xy = offset[:, :, :, 0, :]
        sub_query_xy = offset[:, :, :, :, :].detach()
        '''
        
        
        
        ###########################################################
        offset = self.offset_generator(query_content)
        offset = offset.reshape(B, N, P, G, 2)
        '''
        query_offset = self.stage_query_specific_bboxes.weight
        query_offset = query_offset[None].expand(B, *query_offset.size())
        query_offset = query_offset.reshape(B, N, P, G, 2)
        
        offset = offset + query_offset
        '''
        return offset, sub_query_xy
    
    def make_sample_points_z(self, 
            predicted_deltas_z,  
            box_wh, 
            wh_image, 
            featmap_strides,
            tau=0.9,
        ):
        
        B, N, P, G, _ = predicted_deltas_z.shape
        
        original_area = box_wh[..., 0] * box_wh[..., 1]
        image_area = wh_image[..., 0] * wh_image[..., 1]
        
        original_z = torch.sqrt(
            torch.clamp(image_area / original_area, min=0, max=1)
        )
        original_z = torch.log2(original_z / featmap_strides[0])
        original_z = len(featmap_strides)-1 - \
            torch.clamp(original_z, min=0, max=len(featmap_strides)-1).long()

        original_z = F.one_hot(original_z, \
            len(featmap_strides)).type_as(predicted_deltas_z)
        original_z = original_z.view(B, N, 1, 1, -1)
        
        delta_z = predicted_deltas_z
        sample_z = (1 - tau) * F.softmax(delta_z, -1) + tau * original_z

        return sample_z, delta_z
    
    def intercept_local_feat(self, sample_feats, sub_query_vec, offset, scale_logit):
        '''
            offset: B, N, P, G, 2
            scale_logit: B, N, P, G
        '''
        B, N, G, P, C = sample_feats.shape
        AN = self.anchor_num
        '''
        offset = offset.permute(0, 1, 3, 2, 4).contiguous()
        scale_logit = scale_logit.permute(0, 1, 3, 2).contiguous()
        scale_logit = scale_logit.reshape(B, N, G, P, 1)
        
        sub_query_vec = sub_query_vec.reshape(B, N, G, P, -1)
        
        q = self.current_feat_q(sample_feats) + \
            self.current_xy_q(offset) + \
            self.current_z_q(scale_logit)
        k = self.last_subvec_k(sub_query_vec)
        q = q.view(B, N, G, P, 1, -1)
        k = k.view(B, N, G, P, -1, 1)
        l = torch.matmul(q, k).view(B, N, G, P, 1)
        
        s = torch.sigmoid(l)
        sub_query_vec = s * sub_query_vec + sample_feats
        '''
        
        '''
        sample_feats = sample_feats\
            .permute(0, 1, 3, 2, 4).contiguous()
        # B, N, P, G, -1
        
        #sample_feats = sample_feats.view(B, N, P*G, -1)
        sample_feats = sample_feats.reshape(B, N, AN, -1, C)
        # B, N, AN(=8), G*P/AN, 2 
        
        sample_feats = sample_feats.view(B*N*AN, -1, C)
        
        #sample_feats = self.featmap_shrink(sample_feats)
        sub_query_vec = sub_query_vec.view(B*N*AN, 1, -1)
        
        #B, N, P, G, 2
        #offset = offset.reshape(B, N, AN, -1, 2)
        offset = offset.reshape(B*N*AN, -1, 2)
        sample_xy_pe = position_embedding(
            offset, sample_feats.size(-1) // 2
        )
        
        #sample_feats = sample_feats + sample_xy_pe
        #sample_feats = sample_feats.reshape(B, N, AN, -1, C)
        
        
        #sample_feats = sample_feats.reshape(B, N, P, G, C)
        #sample_feats = sample_feats.permute(0, 1, 3, 2, 4).contiguous()
        
        sub_query_vec = sub_query_vec.reshape(B, N, AN, -1)
        
        # sample_xy_pe = sample_xy_pe.permute(1, 0, 2)
        # sample_feats = sample_feats.permute(1, 0, 2)
        # sub_query_vec = sub_query_vec.permute(1, 0, 2)
        # sub_query_vec = self.local_attention(
        #     sub_query_vec, 
        #     key=sample_feats + sample_xy_pe, 
        #     value=sample_feats,
        # )
        # sub_query_vec = sub_query_vec.permute(1, 0, 2)
        # sub_query_vec = sub_query_vec.reshape(B, N, AN, -1)
        # sub_query_vec = self.local_attention_norm(sub_query_vec)
        # 
        # sub_query_vec = self.local_ffn_norm(
        #     sub_query_vec + self.local_ffn(sub_query_vec)
        # )
        '''
        return sub_query_vec, sample_feats
        
    
    def get_cxcy_wh_logscale(self, query_box, box_ver='xyzr'):
        if box_ver == 'xyzr':

            box_cxcy = query_box[..., :2]
            scale = 2.00 ** query_box[..., 2:3]
            ratio = 2.00 ** torch.cat(
                [query_box[..., 3:4] * -0.5, 
                    query_box[..., 3:4] * 0.5], dim=-1)
            box_wh = scale * ratio
            
            logscale = query_box[..., 2:3]
            
        elif box_ver == 'xyxy':
        
            box_cxcy = (query_box[..., 0::2] + query_box[..., 1::2]) * 0.5
            box_wh = query_box[..., 2:] - query_box[..., :2]
            
            logscale = 0.5 * torch.log2(box_wh[..., 0] * box_wh[..., 1])
            
        return box_cxcy, box_wh, logscale
    
    def make_sample_points_xy(self, 
        predicted_deltas_xy, cxcy, wh):
        
        B, N, P, G, _ = predicted_deltas_xy.shape

        offset_xy = predicted_deltas_xy
        offset_xy = offset_xy * wh[:, :, None, None, :] #.view(B, N, 1, 1, 2)
        
        sample_xy = offset_xy + cxcy
        delta_xy = sample_xy - cxcy
        return sample_xy, delta_xy
    
    def regress_z(self, z, stride_size=4, tau=2.0, mask_size=None):
        def translate_to_linear_weight(ref, stride_size=4, tau=2.0, mask_size=None):
            grid = torch.arange(stride_size, device=ref.device, \
                dtype=ref.dtype).view(*[len(ref.shape)*[1, ]+[-1, ]])

            ref = ref.unsqueeze(-1).clone()
            l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
            if mask_size is not None:
                c = torch.argmax(l2, dim=-1, keepdim=False)

                r = c + mask_size // 2
                r = torch.clamp(r, min=0, max=stride_size-1)
                l = c - mask_size // 2
                l = torch.clamp(l, min=0, max=stride_size-1)
                
                mask_l = F.one_hot(l, num_classes=stride_size)
                mask_r = F.one_hot(r, num_classes=stride_size)
                cumsum_mask_l = torch.cumsum(mask_l, dim=-1)
                cumsum_mask_r = 1 - torch.cumsum(mask_r, dim=-1) + mask_r
                mask = -(cumsum_mask_l + cumsum_mask_r - 2)
                
                l2 = l2.masked_fill(mask.bool(), -np.inf)
                

            weight = torch.softmax(l2, dim=-1)
            return weight
        
        
        sample_points_lvl = z.clone()
        sample_points_lvl_mapped = sample_points_lvl - 3.
        sample_points_lvl_weight = \
            translate_to_linear_weight(
                sample_points_lvl_mapped,
                stride_size=stride_size,
                tau=tau,
                mask_size=mask_size,
            )
        
        return sample_points_lvl_weight
    
    
    def feat_extract(self, 
            sample_points_xy,
            sample_points_z,
            featmap_list,
            featmap_strides,
        ):
        
        B, N, P, G, num_levels = sample_points_z.shape
        B, C_map, H0, W0 = featmap_list[0].shape
        
        
        sample_points_lvl_weight_list = \
            sample_points_z.unbind(-1)
        
        sample_feature = \
            sample_points_z.new_zeros(B, G, C_map//G, N, P)
        
        for i in range(num_levels):
            featmap = featmap_list[i]
            lvl_weights = sample_points_lvl_weight_list[i]
            
            stride = featmap_strides[i]
            mapping_size = featmap.new_tensor(
                [featmap.size(3), featmap.size(2)]) * stride
            mapping_size = mapping_size.view(1, 1, 1, 1, -1)
            normalized_xy = sample_points_xy / mapping_size
            
            sample_feature += self.point_feat_sample(\
                normalized_xy, featmap, weight=lvl_weights)
        
        sample_feature = sample_feature\
            .permute(0, 3, 1, 4, 2).contiguous()
        return sample_feature #B, N, G, P, C_map//G
    
    def point_feat_sample(self, 
            sample_points, featmap, weight=None):
        
        B, N, P, G = sample_points.shape[:-1]
        B, Ck, Hk, Wk = featmap.shape
        
        sample_points = sample_points\
            .permute(0, 3, 1, 2, 4).contiguous()
        # B, n_heads, Hq, Wq, n_points, 2
        # B, n_heads, N, n_points, 2 
        # = (B, G, N, P, 2)
        
        sample_points = sample_points.view(B*G, N, P, 2)
        sample_points = sample_points*2.0-1.0
        featmap = featmap.view(B*G, -1, Hk, Wk)
        ans = F.grid_sample(
            featmap, sample_points,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False,
        )
        
        if weight is not None:
            weight = weight.permute(0, 3, 1, 2).contiguous()
            weight = weight.view(B*G, 1, N, P)
            ans *= weight
        
        ans = ans.view(B, G, -1, N, P)
        return ans
    
    def get_spatial_info(self, query_content, xyzr):
        with torch.no_grad():
            pe = position_embedding(xyzr, query_content.size(-1) // 4)
        return pe
    
    def subbox_refine_xyzr(self, xyzr, xyzr_delta, imgs_whwh, return_bbox=True):
        
        ori_x = xyzr[..., 0:1]
        ori_y = xyzr[..., 1:2]
        z = xyzr[..., 2:3]
        r = xyzr[..., 3:4]
        zr = xyzr[..., 2:4]
        ori_w = (2 ** (z - 0.5*r))
        ori_h = (2 ** (z + 0.5*r))
        
        
        # ori_x = imgs_whwh[..., 0:1] * 0.5
        # ori_y = imgs_whwh[..., 1:2] * 0.5
        # ori_w = imgs_whwh[..., 0:1]
        # ori_h = imgs_whwh[..., 1:2]
        # zr = 0.
        
        new_x = ori_x + xyzr_delta[..., 0:1] * ori_w
        new_y = ori_y + xyzr_delta[..., 1:2] * ori_h
        
        new_zr = zr + xyzr_delta[..., 2:4]
        #new_zr = xyzr_delta[..., 2:4]
        #new_zr = 0. * xyzr_delta[..., 2:4]
        
        xyzr = torch.cat([new_x, new_y, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr)
        else:
            return xyzr
    
    def box_roialign(self, feat_map, xyzr, bbox_roi_extractor, ver='xyzr'):
        if ver == 'xyzr':
            proposal_bboxes = decode_box(xyzr)
        else:
            proposal_bboxes = xyzr
        
        proposal_list = [bboxes for bboxes in proposal_bboxes]
        rois = bbox2roi(proposal_list)

        bbox_feats = bbox_roi_extractor(
            feat_map[:bbox_roi_extractor.num_inputs], rois,
        )
        return bbox_feats
    


class InteractPointId(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 point_num,
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 ):
        super(InteractPointId, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        self.point_num = point_num
        
        
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        
        
        self.W_in = GroupLinear(v_dim, query_dim, groups=p_groups, bias=True)
        
        self.W_out = nn.Linear(query_dim * point_num, query_dim, bias=True)
        
        self.q_generator = nn.Linear(q_dim, query_dim * point_num, bias=True)
        self.k_generator = GroupLinear(k_dim, query_dim, groups=p_groups, bias=True)
        
        
        self.gen_one_d = GroupLinear(
            query_dim * point_num, 3 * p_groups * point_num, 
            groups = p_groups * point_num, bias=True)

        self.layer_norm = nn.LayerNorm(query_dim)
        
        self.act = nn.ReLU(inplace=True)
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.W_in.weight)
        nn.init.zeros_(self.W_in.bias)
        
        nn.init.zeros_(self.q_generator.weight)
        nn.init.xavier_uniform_(self.k_generator.weight)
        nn.init.zeros_(self.q_generator.bias)
        nn.init.zeros_(self.k_generator.bias)
        
        nn.init.zeros_(self.W_out.bias)
        
        
        nn.init.zeros_(self.gen_one_d.weight)
        nn.init.zeros_(self.gen_one_d.bias)
        bias = self.gen_one_d.bias.data.view(-1, 3)
        bias.mul_(0.0)
        bandwidth = 0.5 * 1.0
        nn.init.uniform_(bias[:, :-1], -bandwidth, bandwidth)
    
    def forward(self,
            query_content,
            xyzr,
            featmap,
            featmap_strides,
        ):
        
        feats = featmap[-1]
        stride = featmap_strides[-1]
        
        
        feats = self.get_pos_embed_feat(xyzr, feats)
        feats_x = feats
        
        
        B, C, H, W = feats.shape
        M = H*W
        
        B, N, C = query_content.shape
        G = self.p_groups
        P = self.point_num
        
        q = self.q_generator(query_content)
        q_x = q
        q = q.view(B, N, G, P, -1)
        q = q.permute(0, 2, 3, 1, 4) # B*G, P*N, C//G
        q = q.contiguous().view(B*G, P*N, -1)
        
        feats = feats.view(B, C, -1)
        feats = feats.permute(2,0,1).contiguous() #HW, B, C
        
        k = feats
        k = self.k_generator(k)
        k = k.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, M
        
        S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        S = S.view(B, G, P, N, M)
        
        attn_mask = self.get_point_inside_box_mask(xyzr, feats_x, stride)
        S = S + attn_mask
        S = S.view(B*G, P*N, M)
        S = F.softmax(S, -1)
        
        
        value = feats
        value = self.W_in(value).view(M, B*G, -1)
        value = value.permute(1,0,2) # B*G, M, C//G
        value = torch.bmm(S, value) # B*G, P*N, C//G
        
        value = value.view(B, G, P, N, -1)
        value = value.permute(0, 3, 1, 2, 4)
        value = value.contiguous().view(B, N, G*P, -1)
        
        value = value + q_x.view(B, N, G*P, -1)
        value = value.view(B, N, -1)

        #value = F.layer_norm(value, [value.size(-2), value.size(-1)])
        
        dxyz = self.gen_one_d(value) # B, N, G*P, 1, 3
        dxyz = dxyz.view(B, N, G*P, 1, -1)
        
        value = value.view(B, N, -1)
        value = self.W_out(value)
        value = query_content + value # N,B,C
        value = self.layer_norm(value)
        return dxyz, value
    
    def get_point_inside_box_mask(self, xyzr, feats, stride):
        
        B, N = xyzr.shape[:2]
        B, C, H, W = feats.shape
        M = H*W
        
        mapping_size = feats.new_tensor([1, 1, 1, 1]) * stride #W, H, W, H
        mapping_size = mapping_size.view(1, 1, -1)
        
        grid_H = torch.arange(H, device=xyzr.device, \
            dtype=xyzr.dtype).view(1, 1, H)
        grid_W = torch.arange(W, device=xyzr.device, \
            dtype=xyzr.dtype).view(1, 1, W)
        
        xyxy = decode_box(xyzr)
        xyxy = xyxy / mapping_size
        
        attn_mask_W = ((xyxy[..., 0:1] < grid_W) & (grid_W < xyxy[..., 2:3]))
        attn_mask_H = ((xyxy[..., 1:2] < grid_H) & (grid_H < xyxy[..., 3:4]))
        attn_mask_W = attn_mask_W.view(B, N, 1, W)
        attn_mask_H = attn_mask_H.view(B, N, H, 1)
        attn_mask = (attn_mask_H & attn_mask_W).view(B, N, -1)
        attn_mask = (attn_mask * 1.)
        #print(attn_mask[0,0,:])
        attn_mask = attn_mask.clamp(min=1e-7).log()
        attn_mask = attn_mask.view(B, 1, 1, N, M) #B, G, P, N, M
        
        return attn_mask
    
    def get_pos_embed_feat(self, xyzr, feats):
        B, N = xyzr.shape[:2]
        B, C, H, W = feats.shape
        M = H*W
        
        grid_H = torch.arange(H, device=xyzr.device, \
            dtype=xyzr.dtype).view(1, 1, H)
        grid_W = torch.arange(W, device=xyzr.device, \
            dtype=xyzr.dtype).view(1, 1, W)
        
        posembed_H = self.position_embedding(grid_H, C, norm_size=H).view(1, H, C)
        posembed_W = self.position_embedding(grid_W, C, norm_size=W).view(1, W, C)
        posembed_H = posembed_H.permute(0, 2, 1).reshape(1, C, H, 1)
        posembed_W = posembed_W.permute(0, 2, 1).reshape(1, C, 1, W)
        feats = feats + posembed_H + posembed_W
        return feats
    
    def position_embedding(self, box, num_feats, norm_size=1000, temperature=10000):
        box = box / norm_size
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=box.device)
        dim_t = (temperature ** (2 * (dim_t // 2) / num_feats)).view(1, 1, 1, -1)
        pos_x = box[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=4).flatten(2)
        return pos_x



class GroupAtten(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups=8, 
                 key_dim=None, 
                 value_dim=None, 
                 ):
        super(GroupAtten, self).__init__()
        
        self.temper = (query_dim // p_groups) ** 0.5
        
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        if key_dim is None:
            key_dim = query_dim
        
        if value_dim is None:
            value_dim = query_dim
        
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(key_dim, query_dim, bias=True),
        )
        self.v_generator = nn.Sequential(
            nn.Linear(value_dim, value_dim, bias=True),
        )
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):

        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.xavier_uniform_(self.v_generator[-1].weight)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        nn.init.zeros_(self.v_generator[-1].bias)

    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=0.,
        ):
        N, B = query_content.shape[:2]
        G = self.p_groups

        if key is None:
            key = query_content
            M = N
        else:
            M = key.shape[0]
        
        if value is None:
            value = query_content
        
        v = self.v_generator(value)
        v = v.view(M, B, G, -1)
        v = v.permute(1, 2, 0, 3).contiguous() # B, G, M, C//G

        
        q = self.q_generator(query_content)
        q = q.view(N, B, G, -1)
        q = q.permute(1, 2, 0, 3).contiguous() # B, G, N, C//G
        
        
        k = self.k_generator(key)
        k = k.view(M, B, G, -1)
        k = k.permute(1, 2, 3, 0).contiguous() # B, G, C//G, M
        
        s = torch.matmul(q, k) # B, G, N, M
        s = F.softmax(s / self.temper + attn_mask, -1)
        v = torch.matmul(s, v) # B, G, N, C//G
        
        v = v.permute(2, 0, 1, 3).contiguous() # N, B, G, C//G
        v = v.view(N, B, -1)
        return v
        





class DyPWAttenDW2(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries=100,
                 ):
        super(DyPWAttenDW2, self).__init__()
        
        self.temper = (query_dim) ** 0.5
        
        
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        
        self.m_parameters = query_dim * (query_dim // p_groups)
        self.filter_generator_channel = nn.Sequential(
            nn.Linear(query_dim, self.m_parameters, bias=True),
        )
        
        
        
        #self.filter_generator_group = nn.Sequential(
        #    nn.Linear(num_queries, , bias=False),
        #)
        
        
        
        #self.filter_bias = nn.Embedding(query_dim, query_dim)
        
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.v_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        self.Wv = nn.Linear(query_dim, query_dim, bias=True)
        self.act = nn.ReLU(inplace=True)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.filter_generator_channel[-1].weight)
        #nn.init.zeros_(self.filter_generator_group[-1].weight)
        #nn.init.xavier_normal_(self.filter_bias.weight)
        
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.xavier_uniform_(self.v_generator[-1].weight)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        nn.init.zeros_(self.v_generator[-1].bias)
        
        nn.init.zeros_(self.Wv.bias)
        
        
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=0.,
            identity=None,
        ):
        query_content = query_content.permute(1,0,2)
        
        
        B, N = query_content.shape[:2]
        G = self.p_groups
        
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content
        
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        identity = identity.permute(1,0,2)
        
        if key is not None:
            M = key.shape[1]
        
        
        dy_c_proj = self.filter_generator_channel(value)
        dy_c_proj = dy_c_proj.view(B, M, G, self.query_dim//G, self.query_dim//G)
        dy_c_proj = dy_c_proj.permute(0, 2, 1, 3, 4).contiguous()
        dy_c_proj = dy_c_proj.view(B*G, M, self.query_dim//G, self.query_dim//G)
        dy_c_proj = dy_c_proj.mean(1)
        
        v = self.v_generator(value)
        v = v.view(B, M, G, -1)
        v = v.permute(0, 2, 1, 3).contiguous()
        
        v = v.view(B*G, M, -1)
        #v = torch.bmm(v, dy_c_proj).view(B*G, M, -1)
        
        
        q = self.q_generator(query_content)
        q = q.view(B, N, G, -1)
        q = q.permute(0, 2, 1, 3).contiguous() # B, G, N, C//G
        q = q.view(B*G, N, -1)
        
        k = self.k_generator(key)
        k = k.view(B, M, G, -1)
        k = k.permute(0, 2, 3, 1).contiguous() # B, G, C//G, M
        k = k.view(B*G, -1, M)
        
        s = torch.bmm(q, k) # B*G, N, M
        s = F.softmax(s / self.temper + attn_mask, -1)
        v = torch.bmm(s, v) # B*G, N, C//G
        v = v.view(B, G, N, -1)
        
        v = v.permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, -1)
        v = self.Wv(v)
        query_content = identity + v
        query_content = query_content.permute(1,0,2)
        return query_content
        





class DyPWAttenDW(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries=100,
                 ):
        super(DyPWAttenDW, self).__init__()
        
        self.temper = (query_dim) ** 0.5
        
        
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        
        self.m_parameters = query_dim * (query_dim // p_groups)
        self.filter_generator_channel = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, self.m_parameters, bias=True),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        
        self.attention = MultiheadAttention(query_dim, p_groups, dropout)
        
        self.act = nn.ReLU(inplace=True)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.filter_generator_channel[-1].weight)
        nn.init.zeros_(self.v_proj[-1].weight)
        
        
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=0.,
            identity=None,
        ):
        
        
        
        N, B = query_content.shape[:2]
        G = self.p_groups
        
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content
        
        query_content = query_content.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        identity = identity.permute(1,0,2)
        
        if key is not None:
            M = key.shape[1]
        
        #'''
        x_value = value.view(B, M, -1)
        
        dy_c_proj = self.filter_generator_channel(value)
        dy_c_proj = dy_c_proj.view(B, M, G, self.query_dim//G, self.query_dim//G)
        dy_c_proj = dy_c_proj.permute(0, 2, 1, 3, 4).contiguous()
        dy_c_proj = dy_c_proj.view(B*G, M, self.query_dim//G, self.query_dim//G)
        dy_c_proj = dy_c_proj.mean(1)
        
        value = self.v_proj(value)
        value = value.view(B, M, G, -1)
        value = value.permute(0, 2, 1, 3).contiguous()
        value = value.view(B*G, M, -1)
        value = torch.matmul(value, dy_c_proj).view(B*G, M, -1)
        value = value.view(B, G, M, -1)
        value = value.permute(0, 2, 1, 3).contiguous()
        value = value.view(B, M, -1)
        value = value + x_value
        #'''
        query_content = query_content.permute(1,0,2)
        key = key.permute(1,0,2)
        value = value.permute(1,0,2)
        identity = identity.permute(1,0,2)
        query_content = self.attention(
            query_content,
            key=key,
            value=value,
            attn_mask=attn_mask,
            identity=identity,
        )
        return query_content






class ReLUAtten(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries,
                 num_classes,
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 ):
        super(ReLUAtten, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        self.W_in = nn.Sequential(
            nn.Linear(v_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        self.q_generator = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )
        
        
        self.q_generator_relu1 = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator_relu1 = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )
        
        
        self.q_generator_relu2 = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator_relu2 = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )
        
        
        pe_feat_num = query_dim // 4
        self.pe_feat_num = pe_feat_num
        
        
        
        self.act = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        
        nn.init.xavier_uniform_(self.q_generator_relu1[-1].weight)
        nn.init.xavier_uniform_(self.k_generator_relu1[-1].weight)
        nn.init.zeros_(self.q_generator_relu1[-1].bias)
        nn.init.zeros_(self.k_generator_relu1[-1].bias)
        
        nn.init.xavier_uniform_(self.q_generator_relu2[-1].weight)
        nn.init.xavier_uniform_(self.k_generator_relu2[-1].weight)
        nn.init.zeros_(self.q_generator_relu2[-1].bias)
        nn.init.zeros_(self.k_generator_relu2[-1].bias)
        

        nn.init.xavier_uniform_(self.W_in[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)
        
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=None,
            identity=None,
            cls_logit=None,
            xyxy_q=None,
            xyxy_k=None,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content

        if key is not None:
            M = key.shape[0]
        
        q = query_content
        
        value = self.W_in(value).view(M*B, G, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*M, B, -1)
        local_feat = value
        
        q = self.q_generator(q)
        k = self.k_generator(key)
        q = q.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k = k.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        
        q_relu1 = self.q_generator_relu1(query_content)
        k_relu1 = self.k_generator_relu1(key)
        q_relu1 = q_relu1.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k_relu1 = k_relu1.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        #q_relu1 = self.act(q_relu1)
        #k_relu1 = self.act(k_relu1)
        #q = q - q_relu1
        #k = k - k_relu1
        S_relu1 = torch.bmm(q_relu1, k_relu1)
        S_relu1 = self.act(S_relu1)
        
        q_relu2 = self.q_generator_relu2(query_content)
        k_relu2 = self.k_generator_relu2(key)
        q_relu2 = q_relu2.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k_relu2 = k_relu2.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        #q_relu2 = self.act(q_relu2)
        #k_relu2 = self.act(k_relu2)
        #q = q - q_relu2
        #k = k - k_relu2
        S_relu2 = torch.bmm(q_relu2, k_relu2)
        S_relu2 = self.act(S_relu2)
        
        
        #S = S / torch.sum(S, dim=-1, keepdim=True).clamp(min=1e-7)
        
        #if attn_mask is None:
        #    S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        #else:
        #    S = torch.baddbmm(attn_mask, q / math.sqrt(q.shape[-1]), k) # B*G, N,N
        
        S = (torch.bmm(q, k) - S_relu1 - S_relu2)/ math.sqrt(q.shape[-1])
        #S = self.act(S)
        #S = torch.clamp(S, min=1e-7).log()
        if attn_mask is not None:
            S = S + attn_mask
        
        S = F.softmax(S, -1) #attn_mask: B*G, N, N
        
        
        local_feat = local_feat.permute(1, 0, 2) # B, G*N, C//G
        local_feat = local_feat.reshape(B*G, M, -1)
        local_feat = torch.bmm(S, local_feat) # B*G, N, -1
        
        
        local_feat = local_feat.reshape(B, G*N, -1)
        local_feat = local_feat.permute(1, 0, 2) # G*N, B, C//G
        

        local_feat = local_feat.reshape(G, N*B, -1)
        local_feat = local_feat.permute(1, 0, 2)
        local_feat = local_feat.reshape(N, B, C)
        
        
        local_feat = self.W_out(local_feat)
        
        
        local_feat = identity + local_feat # N,B,C
        
        return local_feat








class LocalSemanticAtt(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries,
                 num_classes,
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 ):
        super(LocalSemanticAtt, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        self.W_in = nn.Sequential(
            nn.Linear(v_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        self.q_generator = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )
        
        self.q_num_classes = nn.Sequential(
            nn.Linear(num_classes, query_dim, bias=True),
        )
        self.k_num_classes = nn.Sequential(
            nn.Linear(num_classes, query_dim, bias=True),
        )
        
        
        pe_feat_num = query_dim // 4
        self.pe_feat_num = pe_feat_num
        self.q_num_iof = nn.Sequential(
            GroupLinear(self.pe_feat_num * 4, query_dim * 4, groups=4),
        )
        self.k_num_iof = nn.Sequential(
            GroupLinear(self.pe_feat_num * 4, query_dim * 4, groups=4),
        )
        self.pe_embed_tau = nn.Parameter(torch.ones(self.p_groups, ))
        nn.init.uniform_(self.pe_embed_tau, 0.0, 4.0)
        
        
        
        self.q_l1_pe_proj = nn.Sequential(
            nn.Linear(pe_feat_num*4, query_dim, bias=True),
        )
        self.k_l1_pe_proj = nn.Sequential(
            nn.Linear(pe_feat_num*4, query_dim, bias=True),
        )
        self.l1_tau = nn.Parameter(torch.ones(self.p_groups, ))
        nn.init.uniform_(self.l1_tau, 0.0, 4.0)
        
        
        self.relative_pe_proj = nn.Sequential(
            nn.Linear(pe_feat_num*4, self.p_groups, bias=True),
        )
        
        
        self.act = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        
        #self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        #self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        #self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        #self.back_local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        #self.back_local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        #self.get_local_norm_out = nn.LayerNorm(query_dim // self.p_groups)
        
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        #nn.init.xavier_uniform_(self.local_proj)
        #nn.init.xavier_uniform_(self.back_local_proj)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        #nn.init.zeros_(self.local_bias)
        #nn.init.zeros_(self.back_local_bias)
        
        nn.init.xavier_uniform_(self.W_in[-1].weight)
        #nn.init.xavier_uniform_(self.W_out[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)
        
        nn.init.zeros_(self.q_num_classes[-1].weight)
        nn.init.zeros_(self.k_num_classes[-1].weight)
        nn.init.zeros_(self.q_num_classes[-1].bias)
        nn.init.zeros_(self.k_num_classes[-1].bias)
        
        nn.init.xavier_uniform_(self.q_num_iof[-1].weight) #eye_
        nn.init.xavier_uniform_(self.k_num_iof[-1].weight)
        nn.init.zeros_(self.q_num_iof[-1].bias)
        nn.init.zeros_(self.k_num_iof[-1].bias)
        
        nn.init.xavier_uniform_(self.q_l1_pe_proj[-1].weight) #eye_
        nn.init.xavier_uniform_(self.k_l1_pe_proj[-1].weight)
        nn.init.zeros_(self.q_l1_pe_proj[-1].bias)
        nn.init.zeros_(self.k_l1_pe_proj[-1].bias)
        
        nn.init.xavier_uniform_(self.relative_pe_proj[-1].weight)
        nn.init.zeros_(self.relative_pe_proj[-1].bias)
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=None,
            identity=None,
            cls_logit=None,
            xyxy_q=None,
            xyxy_k=None,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content

        if key is not None:
            M = key.shape[0]
        
        q = query_content
        
        if (xyxy_q is not None) and (xyxy_k is not None):
            G_pe = 1 #G
            with torch.no_grad():
                pe_feat_num = self.pe_feat_num
                xyxy_q = xyxy_q.view(N, B, 4)
                xyxy_k = xyxy_k.view(M, B, 4)
                q_box_embed = position_embedding_query(xyxy_q, pe_feat_num)
                k_box_embed = position_embedding_key(xyxy_k, pe_feat_num)
                # print(q_box_embed.shape, k_box_embed.shape)
                # #print(q_box_embed)
                # print(xyxy_q)
                # #assert False
                
                I_bias = torch.eye(N, 
                    device=xyxy_q.device, dtype=xyxy_q.dtype)
                I_bias = I_bias.view(1, N, N)
                
                q_box_l1_embed = position_embedding_query(xyxy_q, pe_feat_num)
                k_box_l1_embed = position_embedding_query(xyxy_k, pe_feat_num)
                
                relative_pe = relative_pe_interact_area(xyxy_q, xyxy_k, pe_feat_num)
                relative_pe = relative_pe.view(N, M, B, -1)
                
            
            # q_box_embed = q_box_embed.view(N, B, -1)
            # k_box_embed = k_box_embed.view(M, B, -1)
            # q_box_embed = self.q_num_iof(q_box_embed)
            # k_box_embed = self.k_num_iof(k_box_embed)
            q_box_embed = q_box_embed.view(N, B, 4, -1)
            k_box_embed = k_box_embed.view(M, B, 4, -1)
            
            pe_q_x1 = q_box_embed[:, :, 0, :]
            pe_q_y1 = q_box_embed[:, :, 1, :]
            pe_q_x2 = q_box_embed[:, :, 2, :]
            pe_q_y2 = q_box_embed[:, :, 3, :]
            
            pe_q_x1 = pe_q_x1.reshape(N, B*G_pe, -1)
            pe_q_y1 = pe_q_y1.reshape(N, B*G_pe, -1)
            pe_q_x2 = pe_q_x2.reshape(N, B*G_pe, -1)
            pe_q_y2 = pe_q_y2.reshape(N, B*G_pe, -1)
            
            pe_q_x1 = pe_q_x1.permute(1, 0, 2)
            pe_q_y1 = pe_q_y1.permute(1, 0, 2)
            pe_q_x2 = pe_q_x2.permute(1, 0, 2)
            pe_q_y2 = pe_q_y2.permute(1, 0, 2)
            
            
            pe_k_x1 = k_box_embed[:, :, 0, :]
            pe_k_y1 = k_box_embed[:, :, 1, :]
            pe_k_x2 = k_box_embed[:, :, 2, :]
            pe_k_y2 = k_box_embed[:, :, 3, :]
            
            pe_k_x1 = pe_k_x1.reshape(M, B*G_pe, -1)
            pe_k_y1 = pe_k_y1.reshape(M, B*G_pe, -1)
            pe_k_x2 = pe_k_x2.reshape(M, B*G_pe, -1)
            pe_k_y2 = pe_k_y2.reshape(M, B*G_pe, -1)
            
            pe_k_x1 = pe_k_x1.permute(1, 2, 0)
            pe_k_y1 = pe_k_y1.permute(1, 2, 0)
            pe_k_x2 = pe_k_x2.permute(1, 2, 0)
            pe_k_y2 = pe_k_y2.permute(1, 2, 0)
            
            x_q2k1 = torch.bmm(pe_q_x2, pe_k_x1) #/ math.sqrt(pe_q_x2.shape[-1])
            x_q2k2 = torch.bmm(pe_q_x2, pe_k_x2) #/ math.sqrt(pe_q_x2.shape[-1])
            x_q1k1 = torch.bmm(pe_q_x1, pe_k_x1) #/ math.sqrt(pe_q_x1.shape[-1])
            relu_x_q2k2 = self.relu(x_q2k2)
            relu_x_q1k1 = self.relu(x_q1k1)
            
            y_q2k1 = torch.bmm(pe_q_y2, pe_k_y1) #/ math.sqrt(pe_q_y2.shape[-1])
            y_q2k2 = torch.bmm(pe_q_y2, pe_k_y2) #/ math.sqrt(pe_q_y2.shape[-1])
            y_q1k1 = torch.bmm(pe_q_y1, pe_k_y1) #/ math.sqrt(pe_q_y1.shape[-1])
            relu_y_q2k2 = self.relu(y_q2k2)
            relu_y_q1k1 = self.relu(y_q1k1)
            
            # #print(y_q2k2)
            # #print(y_q1k1)
            # print(y_q2k1)
            # assert False
            
            pe_iof_x = x_q2k1 - relu_x_q2k2 - relu_x_q1k1
            pe_iof_y = y_q2k1 - relu_y_q2k2 - relu_y_q1k1
            pe_iof_x = torch.log(torch.clamp(self.relu(pe_iof_x), min=1e-7))
            pe_iof_y = torch.log(torch.clamp(self.relu(pe_iof_y), min=1e-7))
            pe_iof = pe_iof_x + pe_iof_y #B*G, N, N
            #pe_iof = distance_interact_area(pe_iof_x, pe_iof_y)
            
            #pe_iof_norm_q_x = torch.log(torch.clamp(self.relu(
            #    torch.bmm(pe_q_x2, pe_q_x1.permute(0,2,1))), min=1e-7))
            #pe_iof_norm_q_y = torch.log(torch.clamp(self.relu(
            #    torch.bmm(pe_q_y2, pe_q_y1.permute(0,2,1))), min=1e-7))
            #pe_iof = pe_iof - (pe_iof_norm_q_x + pe_iof_norm_q_y)
            
            # print(pe_iof)
            # print(pe_iof.shape)
            # assert False
            
            pe_iof = pe_iof.view(B, G_pe, N, N)
            pe_iof = pe_iof * self.pe_embed_tau.view(1, -1, 1, 1)
            pe_iof = pe_iof.view(B*G, N, N)
            
            #pe_iof = pe_iof + I_bias
            
            q_box_l1_embed = q_box_l1_embed.view(N, B*G_pe, -1)
            k_box_l1_embed = k_box_l1_embed.view(M, B*G_pe, -1)
            l1_S = torch.bmm(q_box_l1_embed.permute(1, 0, 2), k_box_l1_embed.permute(1, 2, 0))
            l1_S = torch.log(torch.clamp(l1_S, min=1e-7))
            l1_S = l1_S.view(B, G_pe, N, N)
            l1_S = l1_S * self.l1_tau.view(1, -1, 1, 1)
            l1_S = l1_S.view(B*G, N, N)
            
            
            relative_pe = self.relative_pe_proj(relative_pe) / math.sqrt(relative_pe.shape[-1])
            relative_pe = relative_pe.view(N, M, B*G)
            relative_pe = relative_pe.permute(2, 0, 1)
            
            # value = value + k_box_embed.view(N, B, -1) + k_box_l1_embed.view(N, B, -1)
            # q = q + q_box_embed.view(N, B, -1) + q_box_l1_embed.view(N, B, -1)
            # key = key + k_box_embed.view(N, B, -1) + k_box_l1_embed.view(N, B, -1)
        
        
        value = self.W_in(value).view(M*B, G, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*M, B, -1)
        local_feat = value
        # local_feat = torch.bmm(
        #     local_feat, 
        #     self.local_proj.view(G*M, C//G, C//G)
        # ) # G, N, B, C//G
        # local_feat = local_feat.view(G*M, B, -1)
        # local_feat = local_feat + self.local_bias.view(G*M, 1, -1) #G*N, B, -1
        # #local_feat = value + local_feat
        
        
        q = self.q_generator(q)
        k = self.k_generator(key)

        if cls_logit is not None:
            cls_logit = cls_logit.detach()
            q_cls = self.q_num_classes(cls_logit)
            q = q + q_cls
            k_cls = self.k_num_classes(cls_logit)
            k = k + q_cls
        
        q = q.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k = k.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        
        #l2_local_feat = torch.norm(local_feat, p = 2, dim = -1)
        #l2_q = torch.norm(q, p = 2, dim = -1)
        #l2_k = torch.norm(k, p = 2, dim = -1)
        #q = F.layer_norm(q, [q.size(-1), ])
        #k = F.layer_norm(k, [k.size(-1), ])
        
        
        if attn_mask is None:
            S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        else:
            S = torch.baddbmm(attn_mask, q / math.sqrt(q.shape[-1]), k) # B*G, N,N
            #S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        
        if (xyxy_q is not None) and (xyxy_k is not None):
            q_box_l1_embed = q_box_l1_embed.view(N, B, -1)
            k_box_l1_embed = k_box_l1_embed.view(M, B, -1)
            q_box_l1_embed = self.q_l1_pe_proj(q_box_l1_embed)
            k_box_l1_embed = self.k_l1_pe_proj(k_box_l1_embed)
            q_box_l1_embed = q_box_l1_embed.view(N, B*G, -1)
            k_box_l1_embed = k_box_l1_embed.view(M, B*G, -1)
            proj_l1_S = torch.bmm(
                q_box_l1_embed.permute(1, 0, 2), \
                k_box_l1_embed.permute(1, 2, 0) \
            ) / math.sqrt(q_box_l1_embed.shape[-1])
            
            S = S + relative_pe #+ proj_l1_S #+ l1_S + pe_iof
        
        S = F.softmax(S, -1) #attn_mask: B*G, N, N
        
        local_feat = local_feat.permute(1, 0, 2) # B, G*N, C//G
        local_feat = local_feat.reshape(B*G, M, -1)
        local_feat = torch.bmm(S, local_feat) # B*G, N, -1
        
        
        #local_feat = self.get_local_norm(local_feat)
        #local_feat = F.layer_norm(local_feat, [local_feat.size(-1), ])
        #local_feat = self.act(local_feat)
        
        
        local_feat = local_feat.reshape(B, G*N, -1)
        local_feat = local_feat.permute(1, 0, 2) # G*N, B, C//G
        
        
        
        # local_feat_out = torch.bmm(
        #     local_feat, 
        #     self.back_local_proj.view(G*N, C//G, C//G)
        # ) # G*N, B, C//G
        # local_feat_out = local_feat_out + self.back_local_bias.view(G*N, 1, -1)
        # local_feat = local_feat_out
        # #local_feat = local_feat + local_feat_out
        
        local_feat = local_feat.reshape(G, N*B, -1)
        local_feat = local_feat.permute(1, 0, 2)
        local_feat = local_feat.reshape(N, B, C)
        
        
        local_feat = self.W_out(local_feat)
        
        
        local_feat = identity + local_feat # N,B,C
        
        return local_feat




class Ada_projs(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 num_queries,
                 ):
        super(Ada_projs, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        self.W_in = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        
        self.act = nn.ReLU(inplace=True)
        
        
        self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        self.stage_tau = nn.Parameter(torch.ones(1, ))
        nn.init.uniform_(self.stage_tau, 0.0, 1.0)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.local_proj)

        nn.init.zeros_(self.local_bias)
        
        nn.init.xavier_uniform_(self.W_in[-1].weight)
        nn.init.xavier_uniform_(self.W_out[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)
        
    
    def forward(self,
            query_content,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        
        value = query_content
        
        value = self.W_in(value).view(N*B, G, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*N, B, -1)
        local_feat = torch.bmm(
            value, 
            self.local_proj.view(G*N, C//G, C//G)
        ) # G, N, B, C//G
        local_feat = local_feat.view(G*N, B, -1)
        local_feat = local_feat + self.local_bias.view(G*N, 1, -1) #G*N, B, -1
        #local_feat = value + local_feat
        
        local_feat = local_feat.view(G, N*B, -1)
        local_feat = local_feat.permute(1, 0, 2)
        local_feat = local_feat.reshape(N, B, C)
        
        
        local_feat = self.W_out(local_feat)
        local_feat = F.layer_norm(
            local_feat, [local_feat.size(-1), ]
        )
        local_feat = self.stage_tau * local_feat
        
        #s_pz = calc_distri_distance(local_feat.permute(1,0,2))
        #s_pz = s_pz.view(1, B, 1)
        #print(s_pz)
        
        local_feat = query_content + local_feat # N,B,C
        
        #local_feat = F.layer_norm(
        #    local_feat, [local_feat.size(-1), ]
        #)
        
        return local_feat


class LocalSemanticFusion(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 num_queries,
                 ):
        super(LocalSemanticFusion, self).__init__()
        
        self.temper = (query_dim) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        self.W_in = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.ln_out = nn.LayerNorm(query_dim)
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        
        self.act = nn.ReLU(inplace=True)
        
        
        self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        self.back_local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.back_local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        self.get_local_norm_out = nn.LayerNorm(query_dim // self.p_groups)
        
        self.iou_tau = nn.Parameter(torch.ones(p_groups, ))
        nn.init.uniform_(self.iou_tau, 0.0, 4.0)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.local_proj)
        nn.init.zeros_(self.back_local_proj)
        
        nn.init.zeros_(self.q_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        
    
    def forward(self,
            query_content,
            iof,
            eta=0.5,
        ):
        x = query_content
        
        B, N, C = query_content.shape
        G = self.p_groups
        
        iof = iof.view(1, B, N, N)
        
        query_content = self.W_in(query_content).view(B, N, G, -1)
        query_content = query_content.permute(2, 1, 0, 3).contiguous() #G,N,B,-1
        local_feat = torch.bmm(
            query_content.view(G*N, B, -1), 
            self.local_proj.view(G*N, C//G, C//G)
        ) # G, N, B, C//G
        local_feat = local_feat.view(G, N, B, -1)
        local_feat = local_feat + self.local_bias.view(G, N, 1, -1)
        
        
        q = self.q_generator(x).view(B, N, G, -1)
        q = q.permute(2, 0, 1, 3).contiguous() #G,N,B,-1
        q = q.view(G*B, N, -1)
        
        k = self.k_generator(x).view(B, N, G, -1)
        k = k.permute(2, 0, 1, 3).contiguous() #G,N,B,-1
        k = k.view(G*B, N, -1)
        
        
        #local_feat = local_feat.permute(2, 1, 0, 3).contiguous()
        #local_feat = local_feat.view(B, N, C)
        
        # l1 = torch.norm(iof, p = 1, dim = -1)
        # l1 = l1.view(B, N, 1)
        # iof = iof / torch.clamp(l1, min=1e-7)
        # #local_feat = torch.bmm(F.softmax(iof, -1), local_feat)
        # #local_feat = torch.bmm(1.0*(iof > eta), local_feat)
        # local_feat = torch.bmm(iof, local_feat)
        
        iof_score = self.iou_tau.view(-1, 1, 1, 1) * torch.clamp(iof, min=1e-7).log()
        iof_score = iof_score.view(G*B, N, N)
        
        local_feat = local_feat.permute(0, 2, 1, 3).contiguous() # G, B, N, C//G
        local_feat = local_feat.view(G*B, N, -1)
        
        #l2_local_feat = torch.norm(local_feat, p = 2, dim = -1)
        #l2_q = torch.norm(q, p = 2, dim = -1)
        #l2_k = torch.norm(k, p = 2, dim = -1)
        
        S = torch.bmm(q, k.permute(0, 2, 1)) # G*B, N,N
        #S = torch.bmm(local_feat, local_feat.permute(0, 2, 1))
        #S = S / (l2_local_feat.view(G*B, N, 1) * l2_local_feat.view(G*B, 1, N))
        
        S = F.softmax(S + iof_score, -1)
        local_feat = torch.bmm(S, local_feat) 
        
        local_feat = local_feat.view(G, B, N, -1)
        local_feat = local_feat.permute(0, 2, 1, 3).contiguous() # G, N, B, C//G
        
        
        local_feat = self.get_local_norm(local_feat)
        local_feat = self.act(local_feat)
        
        
        #local_feat = local_feat.view(B, N, G, -1)
        #local_feat = local_feat.permute(2, 1, 0, 3).contiguous() #G,N,B,-1
        
        
        local_feat = torch.bmm(
            local_feat.view(G*N, B, -1), 
            self.back_local_proj.view(G*N, C//G, C//G)
        ) # G, N, B, C//G
        local_feat = local_feat.view(G, N, B, -1)
        local_feat = local_feat + self.back_local_bias.view(G, N, 1, -1)
        
        #local_feat = self.get_local_norm_out(local_feat)
        #local_feat = self.act(local_feat)
        
        local_feat = local_feat.permute(2, 1, 0, 3).contiguous()
        local_feat = local_feat.view(B, N, C)
        
        
        local_feat = self.W_out(local_feat)
        
        local_feat = self.ln_out(x + local_feat)
        
        return local_feat
        


class LocalSemanticAttFusion(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries,
                 ):
        super(LocalSemanticAttFusion, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        self.W_in = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        
        self.act = nn.ReLU(inplace=True)
        
        
        self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        self.back_local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.back_local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        self.get_local_norm_out = nn.LayerNorm(query_dim)
        
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        #nn.init.zeros_(self.local_proj)
        #nn.init.zeros_(self.back_local_proj)
        #nn.init.zeros_(self.q_generator[-1].weight)
        #nn.init.zeros_(self.k_generator[-1].weight)
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.xavier_uniform_(self.local_proj)
        nn.init.xavier_uniform_(self.back_local_proj)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        nn.init.zeros_(self.local_bias)
        nn.init.zeros_(self.back_local_bias)
        
        nn.init.xavier_uniform_(self.W_in[-1].weight)
        #nn.init.xavier_uniform_(self.W_out[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)
        
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=None,
            identity=None,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content

        if key is not None:
            M = key.shape[0]
        
        q = query_content
       
        
        
        value = self.W_in(value).view(M*B, G, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*M, B, -1)
        local_feat = value
        local_feat = torch.bmm(
            local_feat, 
            self.local_proj.view(G*M, C//G, C//G)
        ) # G, N, B, C//G
        local_feat = local_feat.view(G*M, B, -1)
        local_feat = local_feat + self.local_bias.view(G*M, 1, -1) #G*N, B, -1
        #local_feat = value + local_feat
        
        q = self.q_generator(q).view(N, B*G, -1)
        q = q.permute(1, 0, 2) # B*G, N, C//G
        
        k = self.k_generator(key).view(M, B*G, -1)
        k = k.permute(1, 2, 0) # B*G, C//G, N
        
        #l2_local_feat = torch.norm(local_feat, p = 2, dim = -1)
        #l2_q = torch.norm(q, p = 2, dim = -1)
        #l2_k = torch.norm(k, p = 2, dim = -1)
        #q = F.layer_norm(q, [q.size(-1), ])
        #k = F.layer_norm(k, [k.size(-1), ])
        
        
        if attn_mask is None:
            S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        else:
            S = torch.baddbmm(attn_mask, q / math.sqrt(q.shape[-1]), k) # B*G, N,N
        S = F.softmax(S, -1) #attn_mask: B*G, N, N
        
        local_feat = local_feat.permute(1, 0, 2) # B, G*N, C//G
        local_feat = local_feat.reshape(B*G, M, -1)
        local_feat = torch.bmm(S, local_feat) # B*G, N, -1
        
        
        #local_feat = self.get_local_norm(local_feat)
        #local_feat = F.layer_norm(local_feat, [local_feat.size(-1), ])
        #local_feat = self.act(local_feat)
        
        
        local_feat = local_feat.reshape(B, G*N, -1)
        local_feat = local_feat.permute(1, 0, 2) # G*N, B, C//G
        
        
        
        local_feat_out = torch.bmm(
            local_feat, 
            self.back_local_proj.view(G*N, C//G, C//G)
        ) # G*N, B, C//G
        local_feat_out = local_feat_out + self.back_local_bias.view(G*N, 1, -1)
        local_feat = local_feat_out
        #local_feat = local_feat + local_feat_out
        
        local_feat = local_feat.reshape(G, N*B, -1)
        local_feat = local_feat.permute(1, 0, 2)
        local_feat = local_feat.reshape(N, B, C)
        
        
        local_feat = self.W_out(local_feat)
        
        
        local_feat = identity + local_feat # N,B,C
        
        return local_feat



class LocalSemanticFusion2(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 num_queries,
                 ):
        super(LocalSemanticFusion2, self).__init__()
        
        self.temper = (query_dim) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        
        
        self.q_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        
        self.act = nn.ReLU(inplace=True)
        
        
        self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim // self.p_groups))
        self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        self.back_local_proj = nn.Parameter(torch.zeros(num_queries * query_dim // self.p_groups, query_dim))
        self.back_local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        
        self.iou_tau = nn.Parameter(torch.ones(1, ))
        nn.init.uniform_(self.iou_tau, 0.0, 4.0)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.local_proj)
        nn.init.zeros_(self.back_local_proj)
        
        nn.init.zeros_(self.q_generator[-1].weight)
        nn.init.zeros_(self.k_generator[-1].weight)
        
    
    def forward(self,
            query_content,
            iof,
            eta=0.5,
        ):
        x = query_content
        
        B, N, C = query_content.shape
        G = self.p_groups
        
        iof = iof.view(1, B, N, N)
        
        query_content = query_content.permute(1, 0, 2)
        local_feat = torch.bmm(
            query_content.view(N, B, -1), 
            self.local_proj.view(N, C, C//G)
        ) # N, B, C//G
        local_feat = local_feat + self.local_bias.view(N, 1, -1)
        local_feat = local_feat.permute(1, 0, 2) # B, N, C//G
        
        q = self.q_generator(x)
        q = q.view(B, N, -1)
        
        k = self.k_generator(x)
        k = k.permute(0, 2, 1)
        k = k.view(B, -1, N)
        
        S = torch.bmm(q, k) # B, N,N
        
        iof_score = self.iou_tau.view(-1, 1, 1, 1) * torch.clamp(iof, min=1e-7).log()
        iof_score = iof_score.view(B, N, N)
        
        S = F.softmax(S + iof_score, -1)
        local_feat = torch.bmm(S, local_feat) 
        
        local_feat = local_feat.view(B, N, -1)
        local_feat = local_feat.permute(1, 0, 2) # N, B, C//G
        
        
        local_feat = self.get_local_norm(local_feat)
        local_feat = self.act(local_feat)
        
        
        
        local_feat = torch.bmm(
            local_feat.view(N, B, -1), 
            self.back_local_proj.view(N, C//G, C)
        ) # N, B, C
        local_feat = local_feat.view(N, B, -1)
        local_feat = local_feat + self.back_local_bias.view(N, 1, -1)
        
        
        local_feat = local_feat.permute(1, 0, 2).contiguous()
        local_feat = local_feat.view(B, N, C)
        
        
        local_feat = x + local_feat
        return local_feat
        
        
class RankingMaker(nn.Module):
    def __init__(self,
                 query_dim, 
                 num_queries,
                 ):
        super(RankingMaker, self).__init__()
        
        self.rank_proj_q = nn.Linear(query_dim, query_dim)
        self.rank_proj_k = nn.Linear(query_dim, query_dim)
        self.act = nn.ReLU(inplace=True)
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.rank_proj_q.weight)
        nn.init.zeros_(self.rank_proj_q.bias)
        
        nn.init.xavier_uniform_(self.rank_proj_k.weight)
        nn.init.zeros_(self.rank_proj_k.bias)
    
    def forward(self, query_content, xyzr):
        #onehot_max_score_rank, sorted_id_max_score = self.get_ranking(cls_logits) # B, N, N
        #rank_embed = self.rank_proj(onehot_max_score_rank)
        #relative_rank = self.get_relative_rank(sorted_id_max_score)
        B, N, C = query_content.shape
        rank_proj_q = self.rank_proj_q(query_content)
        rank_proj_k = self.rank_proj_k(query_content)
        count_zero = torch.bmm(rank_proj_q / math.sqrt(rank_proj_q.shape[-1]), rank_proj_k.permute(0,2,1))
        #count_zero = self.act(count_zero)
        
        rank_embed = count_zero
        rank_embed = F.softmax(rank_embed, -1)
        #query_content = torch.bmm(rank_embed, query_content)
        xyzr = torch.bmm(rank_embed, xyzr) 
        
        return query_content, xyzr
    
    def get_ranking(self, cls_logits):
        B, N, C = cls_logits.shape
        max_score, max_id = torch.max(cls_logits, dim=-1)
        sorted_max_score, sorted_id_max_score = torch.sort(max_score, dim=-1)
        onehot_max_score_rank = F.one_hot(sorted_id_max_score, \
            num_classes=N).to(cls_logits.device).type_as(cls_logits)
        return onehot_max_score_rank, sorted_id_max_score
    
    def get_relative_rank(self, sorted_id):
        B, N = sorted_id.shape[:2]
        relative_rank = sorted_id.view(B, N, 1) - sorted_id.view(B, 1, N)
        relative_rank = relative_rank / N
        return relative_rank
        
        




class SpatialGroupAtt(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries,
                 num_classes,
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 col_topk=32,
                 ):
        super(SpatialGroupAtt, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        self.W_in = nn.Sequential(
            nn.Linear(v_dim, query_dim, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
        )
        
        self.q_generator = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )
        
        
        
        pe_feat_num = query_dim // 4
        self.pe_feat_num = pe_feat_num
        
        
        
        
        
        self.act = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        col_topk = num_queries
        self.col_topk = col_topk
        
        #self.local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        #self.local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        #self.get_local_norm = nn.LayerNorm(query_dim // self.p_groups)
        
        #self.back_local_proj = nn.Parameter(torch.zeros(num_queries * query_dim, query_dim // self.p_groups))
        #self.back_local_bias = nn.Parameter(torch.zeros(num_queries, query_dim))
        #self.get_local_norm_out = nn.LayerNorm(query_dim // self.p_groups)
        
        
        # self.spatial_group_fusion = nn.Sequential(
        #     GroupLinear(col_topk * query_dim, query_dim, groups=self.p_groups),
        # )
        # #self.spatial_group_fuse_generator = nn.Sequential(
        # #    nn.Linear(query_dim, self.col_topk * self.p_groups, bias=True),
        # #)
        
        self.m_generator = nn.Sequential(
            nn.Linear(query_dim, query_dim * query_dim // self.p_groups, bias=True),
        )
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        #nn.init.xavier_uniform_(self.local_proj)
        #nn.init.xavier_uniform_(self.back_local_proj)
        
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        #nn.init.zeros_(self.local_bias)
        #nn.init.zeros_(self.back_local_bias)
        
        nn.init.xavier_uniform_(self.W_in[-1].weight)
        #nn.init.xavier_uniform_(self.W_out[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)
        
        #nn.init.zeros_(self.spatial_group_fuse_generator[-1].weight)
        nn.init.zeros_(self.m_generator[-1].weight)
        
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=None,
            identity=None,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content

        if key is not None:
            M = key.shape[0]
        
        q = query_content
        
        value_in = value
        value = self.W_in(value).view(M*B, G, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*M, B, -1)
        local_feat = value
        
        q = self.q_generator(q)
        k = self.k_generator(key)
        
        q = q.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k = k.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        
        #l2_local_feat = torch.norm(local_feat, p = 2, dim = -1)
        #l2_q = torch.norm(q, p = 2, dim = -1)
        #l2_k = torch.norm(k, p = 2, dim = -1)
        #q = F.layer_norm(q, [q.size(-1), ])
        #k = F.layer_norm(k, [k.size(-1), ])
        
        
        if attn_mask is None:
            S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        else:
            S = torch.baddbmm(attn_mask, q / math.sqrt(q.shape[-1]), k) # B*G, N,N
        
        
        S = F.softmax(S, -1) #attn_mask: B*G, N, N
        value = value.permute(1, 0, 2) # B, G*N, C//G
        value = value.reshape(B*G, M, -1)
        
        local_feat = value
        # local_feat = local_feat.view(B, G*M, -1).permute(1,0,2)
        # local_feat = torch.bmm(
        #     local_feat, 
        #     self.local_proj.view(G*M, C//G, C//G)
        # ) # G*N, B, C//G
        # local_feat = local_feat + self.local_bias.view(G*M, 1, -1) #G*N, B, -1
        # local_feat = local_feat.view(G*M, B, -1).permute(1,0,2)
        # local_feat = local_feat.reshape(B*G, M, -1)
        m = self.m_generator(value_in).view(M, B*G, C//G*C//G).permute(1, 0, 2)
        m = m.reshape(B*G*M, C//G, C//G)
        local_feat = local_feat.view(B*G*M, 1, -1)
        local_feat = torch.bmm(local_feat, m).view(B*G, M, -1)
        value = value + local_feat
        #print(local_feat)
        #local_feat = self.get_Stopk_feat(S, local_feat, query_content)
        # local_feat = self.spatial_group_fusion(local_feat)
        # #spatial_group_fuser = self.spatial_group_fuse_generator(query_content)
        # #spatial_group_fuser = spatial_group_fuser.view(N*B*G, 1, self.col_topk)
        # #local_feat = local_feat.view(N*B*G, self.col_topk, -1)
        # #local_feat = torch.bmm(spatial_group_fuser, local_feat).view(N, B, -1)
        # #value = value + local_feat.view(N, B*G, -1).permute(1, 0, 2)
        
        
        #local_feat = local_feat.view(N, B*G, -1).permute(1, 0, 2)
        #value = value + local_feat
        
        value = torch.bmm(S, value) # B*G, N, -1
        value = value.reshape(B, G*N, -1)
        value = value.permute(1, 0, 2) # G*N, B, C//G
        #local_feat = local_feat.view(N*B, G, -1).permute(1, 0, 2)
        
        
        #local_feat = self.get_local_norm(local_feat)
        #local_feat = F.layer_norm(local_feat, [local_feat.size(-1), ])
        #local_feat = self.act(local_feat)
        # local_feat_out = torch.bmm(
        #     local_feat, 
        #     self.back_local_proj.view(G*N, C//G, C//G)
        # ) # G*N, B, C//G
        # local_feat_out = local_feat_out + self.back_local_bias.view(G*N, 1, -1)
        # local_feat = local_feat_out
        # #local_feat = local_feat + local_feat_out
        
        value = value.reshape(G, N*B, -1)
        value = value.permute(1, 0, 2)
        value = value.reshape(N, B, C)
        
        #value = value + local_feat
        
        value = self.W_out(value)
        
        
        value = identity + value # N,B,C
        
        return value
    
    
    def get_Stopk_feat(self, S, value, query_content):
        N, B = query_content.shape[:2]
        G = self.p_groups
        M = value.shape[1]
        topk_S, id_topk_S = torch.topk(S, \
            self.col_topk, dim=-1, largest=True, sorted=True) # B*G,N,K
        topk_S = topk_S.view(B*G, N, self.col_topk, 1)
        id_topk_S = id_topk_S.view(B*G*N*self.col_topk)
        local_feat = value.view(B, G, M, -1)
        
        B_ids = torch.arange(B, device=value.device).view(B, 1).expand(B, G*N*self.col_topk).reshape(-1)
        G_ids = torch.arange(G, device=value.device).view(1, G, 1).expand(B, G, N*self.col_topk).reshape(-1)
        local_feat = local_feat[B_ids, G_ids, id_topk_S, :]
        local_feat = local_feat.view(B*G, N, self.col_topk, -1)
        
        #local_feat = local_feat * topk_S
        #local_feat = local_feat.view(B*G, N, -1)
        topk_S = topk_S.view(B*G*N, 1, self.col_topk)
        local_feat = local_feat.view(B*G*N, self.col_topk, -1)
        local_feat = torch.bmm(topk_S, local_feat)
        local_feat = local_feat.view(B*G, N, -1)
        
        local_feat = local_feat.permute(1, 0, 2).reshape(N, B, -1)
        return local_feat
        
        
class SpatialConsistencyAtt(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 dropout,
                 num_queries,
                 num_classes,
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 col_topk=32,
                 spatial_split_interval=0.125,
                 ):
        super(SpatialConsistencyAtt, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        self.spatial_split_interval = spatial_split_interval
        num_spatial_split_kernel = int(math.ceil(1. / spatial_split_interval))
        self.num_spatial_split_kernel = num_spatial_split_kernel
        spatial_split_intervals = spatial_split_interval * torch.arange(num_spatial_split_kernel)
        spatial_split_intervals_upbound = spatial_split_interval * torch.arange(1, num_spatial_split_kernel+1)
        spatial_split_intervals_upbound[-1] += 1e-7
        self.register_buffer('spatial_split_intervals', spatial_split_intervals)
        self.register_buffer('spatial_split_intervals_upbound', spatial_split_intervals_upbound)
        #self.local_proj = nn.Parameter(torch.zeros(num_spatial_split_kernel * query_dim, query_dim))
        #self.local_bias = nn.Parameter(torch.zeros(num_spatial_split_kernel, query_dim))
        self.iou_tau = nn.Parameter(torch.ones(self.p_groups, ))
        nn.init.uniform_(self.iou_tau, 0.0, 4.0)
        
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        self.W_in = nn.Sequential(
            nn.Linear(v_dim, query_dim * num_spatial_split_kernel, bias=True),
        )
        self.W_out = nn.Sequential(
            nn.Linear(query_dim * num_spatial_split_kernel, query_dim, bias=True),
        )
        
        self.q_generator = nn.Sequential(
            nn.Linear(q_dim, query_dim, bias=True),
        )
        self.k_generator = nn.Sequential(
            nn.Linear(k_dim, query_dim, bias=True),
        )

        pe_feat_num = query_dim // 4
        self.pe_feat_num = pe_feat_num
        
        self.act = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        col_topk = num_queries
        self.col_topk = col_topk
        
        # self.spatial_group_fusion = nn.Sequential(
        #     GroupLinear(col_topk * query_dim, query_dim, groups=self.p_groups),
        # )
        # #self.spatial_group_fuse_generator = nn.Sequential(
        # #    nn.Linear(query_dim, self.col_topk * self.p_groups, bias=True),
        # #)
        
        
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_generator[-1].weight)
        nn.init.xavier_uniform_(self.k_generator[-1].weight)
        nn.init.zeros_(self.q_generator[-1].bias)
        nn.init.zeros_(self.k_generator[-1].bias)
        
        nn.init.xavier_uniform_(self.W_in[-1].weight)
        #nn.init.xavier_uniform_(self.W_out[-1].weight)
        nn.init.zeros_(self.W_in[-1].bias)
        nn.init.zeros_(self.W_out[-1].bias)

        #nn.init.xavier_uniform_(self.local_bias)
    
    def forward(self,
            query_content,
            key=None,
            value=None,
            attn_mask=None,
            identity=None,
            iou=None,
        ):
        
        #query_content: N, B, C
        N, B, C = query_content.shape 
        G = self.p_groups
        T = self.num_spatial_split_kernel
        
        M = N
        if key is None:
            key = query_content
        
        if value is None:
            value = query_content
        
        if identity is None:
            identity = query_content

        if key is not None:
            M = key.shape[0]
        
        q = query_content
        
        value_in = value
        value = self.W_in(value).view(M*B, G*T, -1)
        value = value.permute(1, 0, 2) #G, N*B, -1
        value = value.reshape(G*T*M, B, -1)
        local_feat = value
        
        q = self.q_generator(q)
        k = self.k_generator(key)
        
        q = q.view(N, B*G, -1).permute(1, 0, 2) # B*G, N, C//G
        k = k.view(M, B*G, -1).permute(1, 2, 0) # B*G, C//G, N
        
        #l2_local_feat = torch.norm(local_feat, p = 2, dim = -1)
        #l2_q = torch.norm(q, p = 2, dim = -1)
        #l2_k = torch.norm(k, p = 2, dim = -1)
        #q = F.layer_norm(q, [q.size(-1), ])
        #k = F.layer_norm(k, [k.size(-1), ])
        
        spa_split_iou = self.get_iou_split(iou)
        spa_split_iou = spa_split_iou.view(B, 1, -1, N, N)
        #spa_split_iou = spa_split_iou.clamp(min=1e-7) * self.iou_tau.view(1, -1, 1, 1, 1)
        #spa_split_iou = spa_split_iou.view(-1, N, N) # B*G*T, N, N
        
        
        if attn_mask is None:
            S = torch.bmm(q / math.sqrt(q.shape[-1]), k)
        else:
            S = torch.baddbmm(attn_mask, q / math.sqrt(q.shape[-1]), k) # B*G, N,N

        S = F.softmax(S, -1) #attn_mask: B*G, N, N
        
        S = S.view(B, G, 1, N, M)
        spa_split_iou = spa_split_iou.view(B, -1, T, N, N)
        S = S * spa_split_iou
        S = S.view(B*G*T, N, M)
        
        value = value.permute(1, 0, 2) # B, G*N, C//G
        value = value.reshape(B*G*T, M, -1)
        
        #local_feat = value
        
        value = torch.bmm(S, value) # B*G, N, -1
        value = value.reshape(B, G*T*N, -1)
        value = value.permute(1, 0, 2) # G*N, B, C//G
        #local_feat = local_feat.view(N*B, G, -1).permute(1, 0, 2)
        
        
        #local_feat = self.get_local_norm(local_feat)
        #local_feat = F.layer_norm(local_feat, [local_feat.size(-1), ])
        #local_feat = self.act(local_feat)
        # local_feat_out = torch.bmm(
        #     local_feat, 
        #     self.back_local_proj.view(G*N, C//G, C//G)
        # ) # G*N, B, C//G
        # local_feat_out = local_feat_out + self.back_local_bias.view(G*N, 1, -1)
        # local_feat = local_feat_out
        # #local_feat = local_feat + local_feat_out
        
        value = value.reshape(G*T, N*B, -1)
        value = value.permute(1, 0, 2)
        value = value.reshape(N, B, -1)
        
        
        value = self.W_out(value)
        value = identity + value # N,B,C
        return value
    
    def get_iou_split(self, iou):
        B = iou.shape[0]
        N = iou.shape[-1]
        with torch.no_grad():
            iou = iou.view(B, 1, N, N)
            spa_interval = self.spatial_split_interval
            spa_split_intervals = self.spatial_split_intervals.view(1, -1, 1, 1)
            spa_split_intervals_up = self.spatial_split_intervals_upbound.view(1, -1, 1, 1)
            spa_split_mask = ((spa_split_intervals < iou) & (iou <= spa_split_intervals_up))
            spa_split_iou = iou * (1. * spa_split_mask) #B, T, N, N
        return spa_split_iou
        
    
    def get_Stopk_feat(self, S, value, query_content):
        N, B = query_content.shape[:2]
        G = S.shape[0] // B
        M = value.shape[1]
        topk_S, id_topk_S = torch.topk(S, \
            self.col_topk, dim=-1, largest=True, sorted=True) # B*G,N,K
        topk_S = topk_S.view(B*G, N, self.col_topk, 1)
        id_topk_S = id_topk_S.view(B*G*N*self.col_topk)
        local_feat = value.view(B, G, M, -1)
        
        B_ids = torch.arange(B, device=value.device).view(B, 1).expand(B, G*N*self.col_topk).reshape(-1)
        G_ids = torch.arange(G, device=value.device).view(1, G, 1).expand(B, G, N*self.col_topk).reshape(-1)
        local_feat = local_feat[B_ids, G_ids, id_topk_S, :]
        local_feat = local_feat.view(B*G, N, self.col_topk, -1)
        
        #local_feat = local_feat * topk_S
        #local_feat = local_feat.view(B*G, N, -1)
        topk_S = topk_S.view(B*G*N, 1, self.col_topk)
        local_feat = local_feat.view(B*G*N, self.col_topk, -1)
        local_feat = torch.bmm(topk_S, local_feat)
        local_feat = local_feat.view(B*G, N, -1)
        
        local_feat = local_feat.permute(1, 0, 2).reshape(N, B, -1)
        return local_feat
        

class QueryAllPointAttention(nn.Module):
    def __init__(self,
                 query_dim, 
                 p_groups, 
                 q_dim=None,
                 k_dim=None,
                 v_dim=None,
                 ):
        super(QueryAllPointAttention, self).__init__()
        
        self.temper = (query_dim / p_groups) ** 0.5
        self.query_dim = query_dim
        self.p_groups = p_groups
        
        if q_dim is None: q_dim = query_dim
        if k_dim is None: k_dim = query_dim
        if v_dim is None: v_dim = query_dim
        
        
        self.W_out = nn.Linear(query_dim, query_dim, bias=True)
        
        
        self.q_generator = nn.Linear(q_dim, query_dim, bias=True)
        self.k_generator = GroupLinear(k_dim, query_dim, groups=p_groups, bias=True)
        
        self.iou_tau = nn.Parameter(torch.ones(self.p_groups, ))
        nn.init.uniform_(self.iou_tau, 0.0, 4.0)
        
        
        self.layer_norm = nn.LayerNorm(query_dim)
        
        self.act = nn.ReLU(inplace=True)
        
        self.init_weights()
        
    @torch.no_grad()
    def init_weights(self):
        nn.init.xavier_uniform_(self.q_generator.weight)
        nn.init.xavier_uniform_(self.k_generator.weight)
        nn.init.zeros_(self.q_generator.bias)
        nn.init.zeros_(self.k_generator.bias)
        
        nn.init.zeros_(self.W_out.bias)

    def forward(self, query, feats, feats_xy, xyzr, imgs_whwh):
        B, N = query.shape[:2]
        _, G, P, S = feats.shape[:4] # B*N, G, P, self.subbox_poolsize, C_map//G
        
        v = feats.view(B, N, G, P*S, -1)
        v = v.permute(0, 2, 1, 3, 4).contiguous()
        v = v.view(B*G, N*P*S, -1)
        
        q = self.q_generator(query).view(B, N, G, -1) # B*N, G, C//G
        q = q.permute(0, 2, 1, 3).contiguous().view(B*G, N, -1)
        
        k = feats.permute(0, 2, 3, 1, 4).contiguous().view(B*N*P*S, -1)
        k = self.k_generator(k) # B*N, G, P, self.subbox_poolsize, C//G
        k = k.view(B, N*P*S, G, -1)
        k = k.permute(0, 2, 3, 1).contiguous().view(B*G, -1, N*P*S)
        
        
        
        xyxy = decode_box(xyzr)
        lt = xyxy[..., :2]
        rb = xyxy[..., 2:]
        xywh = torch.cat([(rb + lt) / 2, rb - lt], -1)
        #weight_xy = self.frame_sample_points(feats_xy, xyxy)
        #weight_xy = torch.log(weight_xy)
        #weight_xy = weight_xy * self.iou_tau.view(1, -1, 1, 1)
        #weight_xy = weight_xy.view(B*G, N, -1)
        
        #S = torch.baddbmm(weight_xy, q / math.sqrt(q.shape[-1]), k) # B*G, N, N*P*S
        ##S = torch.bmm(q / math.sqrt(q.shape[-1]), k) # B*G, N, N*P*S
        #S = F.softmax(S, -1)
        #v = torch.bmm(S, v)
        
        q = self.act(q)
        k = self.act(k)
        
        q,k = self.get_qk(q, k, xywh, feats_xy, self.iou_tau, imgs_whwh)
        #print(q.shape, k.shape)
        
        
        v = torch.bmm(k, v)
        v = torch.bmm(q, v) / torch.bmm(q, k.sum(-1, keepdim=True))
        
        
        v = v.view(B, G, N, -1)
        v = v.permute(0, 2, 1, 3).contiguous()
        v = v.view(B, N, -1)
        
        v = self.W_out(v)
        v = self.layer_norm(query + v)
        
        return v
        

    def frame_sample_points(self, feats_xy, xyxy):
        B, N = xyxy.shape[:2]
        B, G, M, PS = feats_xy.shape[:4]
        
        with torch.no_grad():
            lt = xyxy[..., :2]
            rb = xyxy[..., 2:]
            xywh = torch.cat([(rb + lt) / 2, rb - lt], -1)
            
            xywh = xywh.view(B, 1, N, 1, 4)
            xyxy = xyxy.view(B, 1, N, 1, 4)
            feats_xy = feats_xy.view(B, G, 1, -1, 2)
            
            #mask_x = ((xyxy[..., 0:1] < feats_xy) & (feats_xy < xyxy[..., 2:3]))
            #mask_y = ((xyxy[..., 1:2] < feats_xy) & (feats_xy < xyxy[..., 3:4]))
            #mask_feat_point = (mask_x & mask_y) #  B, G, N, N*PS
            weight_x = ((feats_xy[..., 0] - xywh[..., 0]) / xywh[..., 2])**2
            weight_y = ((feats_xy[..., 1] - xywh[..., 1]) / xywh[..., 3])**2
            weight_xy = -(weight_x + weight_y) / 2
            weight_xy = weight_xy.clamp(min=-7.)
            weight_xy = torch.exp(weight_xy) 
            #weight_xy = weight_xy / (2 * math.pi * xywh[..., 2] * xywh[..., 3])
            # B, G, N, N*PS
            
            weight_x_cx2 = xywh[..., 0] ** 2
            weight_x_cxpx = xywh[..., 0] ** 2
            weight_x_px2 = feats_xy[..., 0] ** 2
            
        
        return weight_xy
    
    def get_qk(self, q, k, xywh, feats_xy, alpha, imgs_whwh):
        """
            q: (B*G, N, C)
            k: (B*G, C, MPS)
        """
        B, N = xywh.shape[:2]
        B, G, M, PS = feats_xy.shape[:4]
        
        #xywh = xywh / imgs_whwh
        #feats_xy = feats_xy / imgs_whwh[..., :2].view(B, 1, N, 1, 2)
        
        q = q.view(B, G, N, -1)
        k = k.view(B, G, k.shape[-2], -1)
        
        AX = xywh[..., 0:1] # B, N, 1
        CX = xywh[..., 1:2] # B, N, 1
        AX = AX.view(B, 1, N, 1).expand(B, G, N, 1)
        CX = CX.view(B, 1, N, 1).expand(B, G, N, 1)
        
        BX = feats_xy[..., 0:1] # B, G, M, PS, 1
        DX = feats_xy[..., 1:2] # B, G, M, PS, 1
        BX = BX.view(B, G, 1, -1) # B, G, 1, MPS
        DX = DX.view(B, G, 1, -1) # B, G, 1, MPS
        
        q1 = torch.ones_like(AX) # B, G, N, 1
        k1 = torch.ones_like(BX) # B, G, 1, MPS
        
        row_norm_coeff_w = xywh[..., 2:3] # B, N, 1
        row_norm_coeff_h = xywh[..., 3:4] # B, N, 1
        row_norm_coeff_w = row_norm_coeff_w.view(B, 1, N, 1).expand(B, G, N, 1)
        row_norm_coeff_h = row_norm_coeff_h.view(B, 1, N, 1).expand(B, G, N, 1)
        WN = row_norm_coeff_w
        HN = row_norm_coeff_h
        WN = -0.5 / (WN * WN).clamp(min=1e-7)
        HN = -0.5 / (HN * HN).clamp(min=1e-7)
        
        alpha = alpha.view(1, -1, 1, 1) # 1, G, 1, 1
        a = alpha
        
        q_ = torch.cat(
            [
                q, 
                AX*AX * WN + CX*CX * HN, 
                -2*AX * WN, 
                -2*CX * HN, 
                q1*(WN + HN),
            ], -1
        )
        k_ = torch.cat(
            [
                k, 
                a*k1, 
                a*BX, 
                a*DX, 
                a*(BX*BX + DX*DX),
            ], -2
        )
        
        q_ = q_.view(B*G, N, -1)
        k_ = k_.view(B*G, k_.shape[-2], -1)
        
        return q_, k_