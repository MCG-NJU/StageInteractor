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
        weight = self.weight.view(self.groups, self.out_features // self.groups, -1) # G, Co//G, C//G
        weight = weight.permute(0, 2, 1) # G, C//G, Co//G
        x = input.view(-1, self.groups, input.shape[-1] // self.groups) # B*N*M, G, C//G
        x = x.permute(1, 0, 2) # G, B*N*M, C//G

        x = torch.bmm(x, weight) # G, B*N*M, Co//G
        x = x.permute(1, 0, 2) #B*N*M, G, Co//G
        x = x.reshape(*input.shape[:-1], -1) #B,N,M, Co
        if self.bias is not None:
            x = x + self.bias
        return x

class CascadeDynamicMixing(nn.Module):
    IND = 0
    
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 dropout=0.,
                 subbox_poolsize=9,
                 use_static_128=False,
                 use_static_S=False,
                 progress_filter=False,
                 stage_type=False,
                 gate_res=True,
                 inhibit_cls=False,
                 no_local_S=False,
                 last_in_point_num=None,
                 use_stage0_static_kernel=False,
                 stage_idx=None,
                 prefix_M_len=None,
                 lim_outpoints_times=None,
                 use_stage0_M=True,
                 use_axis_atten=False,
                 no_previous_filters=False,
                 decompose_S=False,
                 abla_use_static_spatial_mix=False,
                 abla_use_static_channel_mix=False,
                 use_pre_feats=False,
                 reuse_st_id=2,
                 share_W_dygen=True,
                 shrinkS1=False,
                 ):
        '''
            in_dim, out_dim: dim of featmap
        '''
        super(CascadeDynamicMixing, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim
        sampling_rate = sampling_rate if sampling_rate is not None else 1
        last_in_point_num = last_in_point_num 
        
        self.in_dim = in_dim
        self.query_dim = query_dim
        self.out_dim = out_dim
        self.p_groups = p_groups
        self.in_points = in_points
        self.sampling_rate = sampling_rate
        self.out_points = out_points
        
        self.subbox_poolsize = subbox_poolsize
        self.subbox_s2c = subbox_poolsize #num_heads # 8 # 4
        self.use_static_128 = use_static_128
        self.use_static_S = use_static_S
        self.progress_filter = progress_filter
        self.stage_type = stage_type
        self.gate_res = gate_res
        self.no_local_S = no_local_S
        self.no_previous_filters = no_previous_filters
        self.decompose_S = decompose_S
        self.abla_use_static_spatial_mix = abla_use_static_spatial_mix
        self.abla_use_static_channel_mix = abla_use_static_channel_mix
        self.use_pre_feats = use_pre_feats
        
        self.last_in_point_num = last_in_point_num
        self.use_S_reverse = False
        self.stage_idx = stage_idx
        self.reuse_st_id = reuse_st_id
        self.share_W_dygen = share_W_dygen
        self.prepare_share_W_dygen = share_W_dygen
        self.shrinkS1 = shrinkS1
        
        self.use_axis_atten = use_axis_atten
        self.use_stage0_M = use_stage0_M
        self.lim_outpoints_times = lim_outpoints_times if lim_outpoints_times is not None else out_points // (in_points * subbox_poolsize) - 1
        
        self.lim_outpoints_times = 0 if (self.stage_idx < self.reuse_st_id) else self.lim_outpoints_times

        Mlen = max(0, stage_idx-reuse_st_id+1)
        self.prefix_M_len = min(prefix_M_len, Mlen) if prefix_M_len is not None else Mlen
        
        self.subbox_dw_param = in_points * subbox_poolsize * self.subbox_s2c
        self.subbox_pw_param = (in_dim // p_groups) * (in_dim // p_groups)
        self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * out_points
        
        self.mix_bias = None
        if (prefix_M_len is None) or (prefix_M_len > 0):
            self.mix_bias = nn.Linear(query_dim, p_groups*(self.in_dim//p_groups + out_points))
        
        ori_out_points = out_points
        cur_out_points = out_points
        self.ori_out_points = ori_out_points
        
        if self.stage_idx>0 and self.lim_outpoints_times>0:
            cur_out_points = out_points // (self.lim_outpoints_times+1)
            out_points = cur_out_points * min(self.stage_idx, self.lim_outpoints_times+1)
            
        self.out_points = out_points
        self.cur_out_points = cur_out_points
        
        if (self.stage_idx < self.reuse_st_id):
            self.use_S_reverse = False
            self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * out_points
            if self.decompose_S:
                self.s_parameters = in_points * subbox_poolsize * out_points // in_points + \
                    subbox_poolsize * in_points * out_points // subbox_poolsize
        else:
            self.use_S_reverse = True
            self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * cur_out_points
            if self.decompose_S:
                self.s_parameters = in_points * subbox_poolsize * cur_out_points // in_points + \
                    subbox_poolsize * in_points * cur_out_points // subbox_poolsize

        if shrinkS1 and (self.stage_idx == self.reuse_st_id-1):
            self.s_parameters = (in_points * subbox_poolsize // sampling_rate) * cur_out_points
            if self.decompose_S:
                self.s_parameters = in_points * subbox_poolsize * cur_out_points // in_points + \
                    subbox_poolsize * in_points * cur_out_points // subbox_poolsize

        self.parameter_generator = None
        self.parameter_generator_bias = None
        self.parameter_generator_S = None
        if share_W_dygen and (self.stage_idx >= self.reuse_st_id):
            self.share_W_dygen = True
            self.parameter_generator_bias = nn.Parameter(torch.Tensor(p_groups * self.subbox_pw_param))
            self.parameter_generator_S = nn.Linear(query_dim, p_groups * self.s_parameters)
            pass 
        else:
            self.share_W_dygen = False
            self.parameter_generator = nn.Linear(query_dim, p_groups * (self.subbox_pw_param + self.s_parameters))
        
        self.Wv = nn.Linear(out_points * out_dim, out_dim, bias=True)
        self.Wv_layer_norm = nn.LayerNorm(out_dim)
        
        M_score_list_len = 0 if self.no_previous_filters else self.prefix_M_len
        self.M_score_list_len = M_score_list_len
        self.M_score_list = None
        self.S_score_list = None
        self.subbox_local_list = None
        self.axis_atten_list = None
        self.parameter_generator_self = None
        self.subbox_score_list = None
        self.pre_parameter_generator_bias = None
        self.channel_list = None
        if self.prefix_M_len > 0 or self.lim_outpoints_times > 0:

            self.subbox_local_list = nn.ModuleList()
            self.axis_atten_list = nn.ModuleList()
            if self.abla_use_static_spatial_mix:
                for i in range(self.prefix_M_len):
                    self.subbox_local_list.append(
                        nn.Linear(subbox_poolsize, subbox_poolsize, bias=True)
                    )

                for i in range(self.prefix_M_len):
                    self.axis_atten_list.append(
                        nn.Linear(in_points, in_points, bias=True)
                    )
            elif self.abla_use_static_channel_mix:
                for i in range(self.prefix_M_len):
                    self.subbox_local_list.append(
                        nn.Linear(in_dim // p_groups, in_dim // p_groups)
                    )

                for i in range(self.prefix_M_len):
                    self.axis_atten_list.append(
                        nn.Linear(in_dim // p_groups, in_dim // p_groups)
                    )
            elif self.share_W_dygen:
                for i in range(self.prefix_M_len):
                    self.subbox_local_list.append(
                        nn.Linear(in_dim * subbox_poolsize // p_groups, \
                            in_dim * subbox_poolsize // p_groups, bias=True)
                    )
                
                for i in range(self.prefix_M_len):
                    self.axis_atten_list.append(
                        nn.Linear(in_dim * in_points // p_groups, \
                            in_dim * in_points // p_groups, bias=True)
                    )
            else:
                for i in range(self.prefix_M_len):
                    self.subbox_local_list.append(
                        nn.Linear(in_dim * subbox_poolsize // p_groups, \
                            in_dim * subbox_poolsize // p_groups, bias=True)
                    )

                for i in range(self.prefix_M_len):
                    self.axis_atten_list.append(
                        nn.Linear(in_dim * in_points // p_groups, \
                            in_dim * in_points // p_groups, bias=True)
                    )
            
            self.subbox_score_list = nn.ModuleList()
            for i in range(self.prefix_M_len): ###!!!
                self.subbox_score_list.append(
                    nn.Linear(query_dim, p_groups * in_points * subbox_poolsize),
                )
            
            if self.use_pre_feats:
                self.channel_list = nn.ModuleList()
                for i in range(self.prefix_M_len):
                    self.channel_list.append(
                        nn.Linear(in_dim // p_groups, in_dim // p_groups)
                    )
            
            if M_score_list_len > 0:
                self.M_score_list = nn.Linear(query_dim, p_groups*in_dim//p_groups * 2 * M_score_list_len)
                self.M_score_list_mix_bias = nn.Linear(query_dim, p_groups*in_dim//p_groups * M_score_list_len)

            if self.lim_outpoints_times > 0:
                self.S_score_list = nn.Linear(query_dim, \
                    p_groups * cur_out_points * self.lim_outpoints_times, bias=True)
                
                if self.decompose_S:
                    self.S_score_list2 = nn.Linear(query_dim, \
                        p_groups * (in_points+subbox_poolsize) * self.lim_outpoints_times, bias=True)
                else:
                    self.S_score_list2 = nn.Linear(query_dim, \
                        p_groups * in_points*subbox_poolsize * self.lim_outpoints_times, bias=True)


        self.act = nn.ReLU(inplace=True)
        self.temper = (in_dim // p_groups) ** 0.5
        self.init_weights()
    
    @torch.no_grad()
    def init_weights(self):
        
        a = (6/(self.in_dim + self.p_groups*(self.subbox_pw_param+self.s_parameters)))**0.5
        
        if self.parameter_generator_self is not None:
            nn.init.zeros_(self.parameter_generator_self.weight)
            nn.init.uniform_(self.parameter_generator_self.bias, -a, a)
        
        if self.parameter_generator_S is not None:
            nn.init.zeros_(self.parameter_generator_S.weight)
            nn.init.uniform_(self.parameter_generator_S.bias, -a, a)
        
        if self.parameter_generator is not None:
            nn.init.zeros_(self.parameter_generator.weight)
        
        nn.init.zeros_(self.Wv.weight)
        
        if self.subbox_local_list is not None:
            for i in range(len(self.subbox_local_list)):
                nn.init.zeros_(self.subbox_local_list[i].weight)
                nn.init.zeros_(self.subbox_local_list[i].bias)
            
            for i in range(len(self.axis_atten_list)):
                nn.init.zeros_(self.axis_atten_list[i].weight)
                nn.init.zeros_(self.axis_atten_list[i].bias)
            
            for i in range(len(self.subbox_score_list)):
                nn.init.zeros_(self.subbox_score_list[i].weight)
                nn.init.zeros_(self.subbox_score_list[i].bias)
        
        if self.channel_list:
            for i in range(len(self.channel_list)):
                nn.init.zeros_(self.channel_list[i].weight)
                nn.init.zeros_(self.channel_list[i].bias)
        
        if self.pre_parameter_generator_bias is not None:
            for i in range(len(self.pre_parameter_generator_bias)):
                nn.init.uniform_(self.pre_parameter_generator_bias[i], -a, a)

        if self.S_score_list is not None:
            nn.init.zeros_(self.S_score_list.weight)
            nn.init.zeros_(self.S_score_list.bias)
            
            nn.init.zeros_(self.S_score_list2.weight)
            nn.init.zeros_(self.S_score_list2.bias)
        
        if self.M_score_list is not None:
            nn.init.zeros_(self.M_score_list.weight)
            nn.init.zeros_(self.M_score_list.bias)
            
            nn.init.zeros_(self.M_score_list_mix_bias.weight)
        
        if self.parameter_generator_bias is not None:
            nn.init.uniform_(self.parameter_generator_bias, -a, a)

        if self.mix_bias is not None:
            nn.init.zeros_(self.mix_bias.weight)

    def forward(self, feats, query_vec, dyconv_pack=None, last_feats=None, last_dyconv_gen_vec=None, last_query=None):
        '''
            feats (B, N, G, P, C//G)
            offset (B, N, P, G, 2)
            scale_logit (B, N, P, G)
            query_box (B, N, 4)
            origin, bbox_feats: (B*N, C, Pools, Pools): 200, 256, 7, 7
            subbox, bbox_feats: B, N, G, (P(//9)) *9, C_map//G
        '''
        
        
        P = self.in_points
        C_map = self.query_dim
        
        B, N, G = feats.shape[:3]
        if DEBUG:
            print(B*N*G* P*self.subbox_poolsize * self.in_dim // G * self.in_dim // G)

        feats = feats.view(
            B*N, G, P, self.subbox_poolsize, C_map//G
        )
        ori_feats = feats
        
        PS = P * self.subbox_poolsize
        
        dy_filters = None
        if dyconv_pack is not None:
            dy_filters = dyconv_pack

        
        feat_res = feats
        
        if self.share_W_dygen:
            query_vec_t = query_vec
            
            last_parameter_generator_weight = dy_filters[-1][-1]
            
            Wv_w = last_parameter_generator_weight
            Wv_w = Wv_w.view(C_map, G, -1)
            Wv_w = Wv_w[:, :, :self.subbox_pw_param]
            Wv_w = Wv_w.reshape(-1, G*self.subbox_pw_param)
            subbox_pwconv = torch.matmul(query_vec_t, Wv_w) + self.parameter_generator_bias
            S = self.parameter_generator_S(query_vec_t)
        else:
            params = self.parameter_generator(query_vec)
            params = params.reshape(B*N, G, -1)
            subbox_pwconv, S = params.split(
                [self.subbox_pw_param, self.s_parameters], 2
            )
        
        M_bias, S_bias = 0, 0
        if self.mix_bias is not None:
            M_bias, S_bias = self.mix_bias(query_vec).view(B*N*G, -1).split([self.in_dim//G, self.ori_out_points], -1)
            M_bias = M_bias.view(B*N*G, 1, self.in_dim // G) 
            S_bias = S_bias.view(B*N*G, -1, 1) 
        
        feats_subbox = feats.view(B*N*G, -1, C_map//G) #B*N, G, P*9, C_map//G
        subbox_pwconv = subbox_pwconv.view(B*N*G, self.in_dim // G, -1)
        feats_pw = torch.bmm(feats_subbox, subbox_pwconv) + M_bias#B*N,G,P*9,(in_dim//G)
        feats_pw = feats_pw.view(B*N*G, -1, self.in_dim // G)
        feats_pw = F.layer_norm(
            feats_pw, 
            [feats_pw.size(-2), feats_pw.size(-1)]
        )
        feats_pw = self.act(feats_pw)
        feats = feats_pw
        
        if self.decompose_S: S = S.reshape(B*N*G, -1, P+self.subbox_poolsize) ###!!!!
        else: S = S.reshape(B*N*G, -1, P, self.subbox_poolsize)
        
        if self.prefix_M_len > 0 or self.lim_outpoints_times > 0:
            
            M_now = None
            if self.parameter_generator_self is not None:
                M_now = self.parameter_generator_self(query_vec)
                M_now = M_now.reshape(B*N*G, C_map//G, C_map//G)
            
            if self.M_score_list_len > 0:
                Mscores = self.M_score_list(query_vec).sigmoid()
                Mscores = Mscores.view(B*N*G, self.in_dim//G, 2, self.M_score_list_len)
                Mscores2 = Mscores[:, :, 0:1, :]
                Mscores = Mscores[:, :, 1:2, :]
                Mscores = Mscores.split(Mscores.shape[-1]*[1,], -1)
                Mscores2 = Mscores2.view(B*N*G, 1, self.in_dim//G, self.M_score_list_len)
                Mscores2 = Mscores2.split(Mscores2.shape[-1]*[1,], -1)
                M_mix_bias = self.M_score_list_mix_bias(query_vec).view(B*N*G, self.M_score_list_len, self.in_dim//G)
                M_mix_bias = M_mix_bias.split(M_mix_bias.shape[-2]*[1,], -2)
            
            if self.lim_outpoints_times > 0:
                Sscores = self.S_score_list(query_vec).sigmoid()
                if self.decompose_S:
                    Sscores = Sscores.view(B*N*G, self.cur_out_points, 1, self.lim_outpoints_times)
                else:
                    Sscores = Sscores.view(B*N*G, self.cur_out_points, 1, 1, self.lim_outpoints_times)
                Sscores = Sscores.split(Sscores.shape[-1]*[1,], -1)
                
                Sscores2 = self.S_score_list2(query_vec).sigmoid()
                if self.decompose_S:
                    Sscores2 = Sscores2.view(B*N*G, 1, -1, self.lim_outpoints_times)
                else:
                    Sscores2 = Sscores2.view(B*N*G, 1, P, self.subbox_poolsize, self.lim_outpoints_times)
                Sscores2 = Sscores2.split(Sscores2.shape[-1]*[1,], -1)

            S_lst_list = [S, ]
            len_last_gen = len(dy_filters)
            mlp_counts = 0
            S_counts = 0
            
            for idx in range(len(dy_filters)):
                M_lst, feat_lst, S_lst, pre_Wv = dy_filters[idx]
                if self.use_pre_feats:
                    feat_lst = feat_lst.view_as(feats)
                
                if idx >= len_last_gen - self.prefix_M_len:
                    
                    subbox_score = self.subbox_score_list[mlp_counts](query_vec)
                    subbox_score = subbox_score.view(B*N, G, P, self.subbox_poolsize, -1)
                    subbox_score = F.softmax(subbox_score, -1)
                    feat_res = feat_res.view(B*N, G, P, self.subbox_poolsize, -1) * subbox_score
                    feat_res = feat_res.view(B*N*G, -1, C_map//G)
                    
                    feats_axis = feats
                    feats_subb = feats
                    if self.use_pre_feats:
                        feats_axis = feat_lst
                        feats_subb = feat_lst
                    
                    if self.abla_use_static_spatial_mix:
                        feats_axis = feats_axis.view(B*N*G, P, self.subbox_poolsize, -1)
                        feats_axis = feats_axis.permute(0, 2, 3, 1).reshape(B*N*G, self.subbox_poolsize, -1, P)
                        feats_axis = self.axis_atten_list[mlp_counts](feats_axis)
                        feats_axis = feats_axis.permute(0, 3, 1, 2).reshape(B*N*G, P*self.subbox_poolsize, -1)
                        
                        feats_subb = feats_subb.view(B*N*G*P, self.subbox_poolsize, -1)
                        feats_subb = feats_subb.permute(0, 2, 1)
                        feats_subb = self.subbox_local(feats_subb)
                        feats_subb = feats_subb.permute(0, 2, 1).reshape(B*N*G, P*self.subbox_poolsize, -1)
                    elif self.abla_use_static_channel_mix:
                        feats_axis = feats_axis.view(B*N*G, P*self.subbox_poolsize, -1)
                        feats_axis = self.axis_atten_list[mlp_counts](feats_axis)

                        feats_subb = feats_subb.view(B*N*G, P*self.subbox_poolsize, -1)
                        feats_subb = self.subbox_local_list[mlp_counts](feats_subb)
                    elif self.share_W_dygen:
                        
                        feats_axis = feats_axis.view(B*N*G, P, self.subbox_poolsize, -1) # B*N,G,P, K*(in_dim // G)
                        feats_axis = feats_axis.permute(0, 2, 1, 3).reshape(B*N*G, self.subbox_poolsize, -1)
                        feats_axis = self.axis_atten_list[mlp_counts](feats_axis)
                        feats_axis = feats_axis.view(B*N*G, self.subbox_poolsize, P, -1)
                        feats_axis = feats_axis.permute(0, 2, 1, 3).reshape(B*N*G, P*self.subbox_poolsize, -1)
                        
                        feats_subb = feats_subb.view(B*N, G, P, -1) # B*N,G,P, K*(in_dim // G)
                        feats_subb = self.subbox_local_list[mlp_counts](feats_subb)

                    else:
                        feats_axis = feats_axis.view(B*N*G, P, self.subbox_poolsize, -1) # B*N,G,P, K*(in_dim // G)
                        feats_axis = feats_axis.permute(0, 2, 1, 3).reshape(B*N*G, self.subbox_poolsize, -1)
                        feats_axis = self.axis_atten_list[mlp_counts](feats_axis)
                        feats_axis = feats_axis.view(B*N*G, self.subbox_poolsize, P, -1)
                        feats_axis = feats_axis.permute(0, 2, 1, 3).reshape(B*N*G, P*self.subbox_poolsize, -1)
                        
                        feats_subb = feats_subb.view(B*N, G, P, -1) # B*N,G,P, K*(in_dim // G)
                        feats_subb = self.subbox_local_list[mlp_counts](feats_subb)
                    
                    feats = feats.view(B*N*G, -1, C_map//G)
                    feats_subb = feats_subb.view(B*N*G, -1, C_map//G)
                    if self.use_pre_feats:
                        feats = self.channel_list[mlp_counts](feats)
                        feats = feats + feats_subb + feats_axis + feat_res
                    else:
                        feats = feats_subb + feats_axis + feat_res
                    
                    feat_res = feats
                    
                    if self.M_score_list_len > 0:
                        
                        Mscore = Mscores[mlp_counts].squeeze(-1)
                        Mscore2 = Mscores2[mlp_counts].squeeze(-1)
                        M = M_lst * Mscore + subbox_pwconv * (1 - Mscore)
                        M = M * Mscore2 + subbox_pwconv * (1 - Mscore2)
                    else:
                        M = M_lst
                    
                    if M_now is not None:
                        M = M_now 
                    
                    if self.no_previous_filters:
                        feats = feats
                    else:
                        feats = torch.bmm(feats, M) + M_mix_bias[mlp_counts]# B*N*G, P, outdim//G

                    feats = F.layer_norm(feats, [feats.size(-2), feats.size(-1)])
                    feats = self.act(feats)

                    mlp_counts = mlp_counts + 1


                if (idx >= len_last_gen-self.lim_outpoints_times):
                    
                    if self.decompose_S:
                        S_last = S_lst[:, :S.shape[1], :]
                    else:
                        S_last = S_lst[:, :S.shape[1], :P, :]

                    Sscore = Sscores[S_counts].squeeze(-1)
                    Sscore2 = Sscores2[S_counts].squeeze(-1)
                    
                    S_last = S_last * Sscore + S * (1 - Sscore)
                    S_last = S_last * Sscore2 + S * (1 - Sscore2)
                    S_lst_list.append(S_last)
                    S_counts = S_counts + 1

            S = torch.cat(S_lst_list, 1)
        
        feats_M = feats
        if self.use_pre_feats:
            ori_feats = feats_M

        if self.decompose_S:
            Sp, Ss = S.split([P, self.subbox_poolsize], -1)
            Sp = Sp.reshape(B*N*G*P, -1, self.subbox_poolsize)
            Ss = Ss.reshape(B*N*G*self.subbox_poolsize, -1, P)
            
            feats_Mp = feats_M.view(B*N*G*P, -1, self.in_dim // G)
            feats_Mb = feats_M.view(B*N*G, P, -1, self.in_dim // G)
            feats_Mb = feats_Mb.permute(0, 2, 1, 3).contiguous()
            feats_Mb = feats_Mb.view(B*N*G*self.subbox_poolsize, -1, self.in_dim // G)
            
            feats_Mp = torch.bmm(Sp, feats_Mp)
            feats_Mb = torch.bmm(Ss, feats_Mb)
            
            feats_Mp = F.layer_norm(feats_Mp, [feats_Mp.size(-2), feats_Mp.size(-1)])
            feats_Mb = F.layer_norm(feats_Mb, [feats_Mb.size(-2), feats_Mb.size(-1)])
            
            feats_Mp = feats_Mp.view(B*N*G, -1, self.in_dim // G)
            feats_Mb = feats_Mb.view(B*N*G, -1, self.in_dim // G)
            feats_MS = feats_Mb + feats_Mp
            
            feats_MS = self.act(feats_MS)
        else:
            S = S.view(B*N*G, -1, feats_M.shape[-2])
            feats_M = feats_M.view(B*N*G, -1, self.in_dim // G)
            feats_MS = torch.bmm(S, feats_M) + S_bias # B*N*G, outP, outdim//G
            feats_MS = feats_MS.view(B*N*G, -1, self.in_dim // G)
            feats_MS = F.layer_norm(feats_MS, [feats_MS.size(-2), feats_MS.size(-1)])
            feats_MS = self.act(feats_MS)
            
            if DEBUG:
                print(B*N*G*self.in_dim // G * P*self.subbox_poolsize * self.out_dim)


        
        feats_MS_flat = feats_MS.reshape(B, N, -1)

        feats_MS_flat = self.Wv(feats_MS_flat)
        
        feats_MS_q = self.Wv_layer_norm(query_vec + feats_MS_flat)
        
        feats_reg = feats_MS_q
        feats_cls = feats_MS_q
        

        subbox_pwconv = subbox_pwconv.view(B*N*G, self.in_dim // G, -1)
        if self.decompose_S:
            S = S.view(B*N*G, -1, P+self.subbox_poolsize)
        else:
            S = S.view(B*N*G, -1, P, self.subbox_poolsize)
        
        last_weight = self.Wv.weight if self.prepare_share_W_dygen else None
        if self.use_S_reverse:
            dy_filter = (subbox_pwconv, ori_feats, S, last_weight)
            dy_filters.append(dy_filter)
        else:
            dy_filters = []
            dy_filter = (subbox_pwconv, ori_feats, S, last_weight)
            dy_filters.append(dy_filter)
        
        
        
        CascadeDynamicMixing.IND += 1
        return feats_reg, feats_cls, dy_filters
    
    def get_topk_delta_feat(self, q2, q1, w1, b1=None, k=4):
        '''
            w1: (D_in*D_out, C)
            b1: (D_in*D_out, )
        '''
        w1 = w1.permute(1, 0)
        B, N, C = q2.shape
        dq = q2 - q1
        dq = dq.reshape(B*N, k, -1)
        w1 = w1.view(k, C//k, -1)
        
        dq = dq[:, 0, :]
        w1 = w1[0, :, :]
        dq = torch.matmul(dq, w1)
        if b1 is not None:
            dq = dq + b1
        
        return dq



class SRShadowForFlops(nn.Module):
    def __init__(self, 
                 in_dim=256, 
                 in_points=32, 
                 p_groups=4, 
                 num_queries=100,
                 dim_feedforward=2048,
                 query_dim=None,
                 out_dim=None, 
                 out_points=None, 
                 sampling_rate=None,
                 dropout=0.,
                 subbox_poolsize=9,
                 use_static_128=False,
                 use_static_S=False,
                 progress_filter=False,
                 stage_type=False,
                 gate_res=True,
                 inhibit_cls=False,
                 no_local_S=False,
                 last_in_point_num=None,
                 use_stage0_static_kernel=False,
                 stage_idx=None,
                 prefix_M_len=None,
                 lim_outpoints_times=None,
                 use_stage0_M=True,
                 use_axis_atten=False,
                 **kwargs
                ):
        super(SRShadowForFlops, self).__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.p_groups = p_groups
        self.out_dim = out_dim
        self.out_points = out_points
        
        self.stage_idx = stage_idx
        self.lim_outpoints_times = lim_outpoints_times if lim_outpoints_times is not None else out_points // (in_points * subbox_poolsize) - 1
        self.prefix_M_len = prefix_M_len if prefix_M_len is not None else stage_idx

    def forward(self, x, query):
        pass

    @staticmethod
    def __user_flops_handle__(module, input, output):
        B, num_query, num_group, num_point, num_channel = input.shape

        eff_in_dim = module.in_dim//num_group
        eff_out_dim = module.out_dim//num_group
        in_points = module.in_points
        out_points = module.out_points

        step1 = B*num_query*num_group*in_points*eff_in_dim*eff_out_dim * max(2, module.stage_idx)
        step2 = B*num_query*num_group*eff_out_dim*in_points*out_points
        module.__flops__ += int(step1+step2)