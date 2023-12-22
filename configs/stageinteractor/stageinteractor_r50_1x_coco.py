def __get_debug():
    import os
    return 'C_DEBUG' in os.environ


debug = __get_debug()

log_interval = 100

if debug:
    _base_ = [
        '../_base_/datasets/coco_detection_tiny.py',
        '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
    log_interval = 50
    work_dir_prefix = './work_dirs/stageinteractor_debug'
else:
    _base_ = [
        '../_base_/datasets/coco_detection.py',
        '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
    work_dir_prefix = './work_dirs/stageinteractor_mmdet'

IMAGE_SCALE = (1333, 800)

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=IMAGE_SCALE, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMAGE_SCALE,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, #2,
    workers_per_gpu=2, #2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline),
)
evaluation = dict(interval=1, metric='bbox')


num_stages = 6
num_proposals = 100
QUERY_DIM = 256
FEAT_DIM = 256
FF_DIM = 2048

num_interact_heads=8
num_interact_channel_groups=4
N_scale=4

in_point_sampling_rate = 1
subbox_poolsize = 8 #5 #9

subbox_poolsize_rcnn = 8

# P_in for spatial mixing in the paper.
in_points_list = [64 // subbox_poolsize * in_point_sampling_rate, ] + [32 // subbox_poolsize_rcnn * in_point_sampling_rate, ] * (num_stages-1)

# P_out for spatial mixing in the paper. Also named as `out_points` in this codebase.
out_patterns_list = [128, ] * num_stages

# G for the mixer grouping in the paper. Also named as n_head (distinguished from num_heads in MHSA) in this codebase.
n_group_list = [4, ] * num_stages


anchor_point_num = 8
anchor_channel = QUERY_DIM // n_group_list[0]


stageinteractor_head_division = [0, 1, 2, 2, 2, 2]

kd_coeff = 1 #0.25

box_basic_heads = [
    dict(
        type='StageInteractorHead',
        num_classes=80,
        num_ffn_fcs=2,
        num_heads=8,
        num_cls_fcs=1,
        num_reg_fcs=1,
        feedforward_channels=FF_DIM,
        content_dim=QUERY_DIM,
        feat_channels=FEAT_DIM,
        dropout=0.0,
        in_points=in_points_list[stage_idx],
        out_points=out_patterns_list[stage_idx],
        n_heads=n_group_list[stage_idx],
        ffn_act_cfg=dict(type='ReLU', inplace=True),
        num_queries=num_proposals,
        num_interact_heads=num_interact_heads,
        num_interact_channel_groups=num_interact_channel_groups, 
        N_scale=N_scale,
        anchor_point_num=anchor_point_num,
        anchor_channel=anchor_channel,
        roi_inpoint_h = 7,
        roi_inpoint_w = 7,
        in_point_sampling_rate = in_point_sampling_rate,
        subbox_poolsize = subbox_poolsize,
        stage_type = stageinteractor_head_division[stage_idx],
        use_bg_idx_classifier = False, 
        loss_bbox=dict(type='WeightedL1Loss', loss_weight=5.0), #L1Loss
        loss_iou=dict(type='SoftIou', loss_weight=2.0), #GIoULoss
        iou_eval_eta=0.5,  
        use_soft_iou=False, 
        use_topk_labels=False, 
        use_thres_filter=True,  
        use_iof=False,
        use_hard_label=True,
        soft2hard_label=True,  
        targets_candi_ids= [-1, ] + [j for j in range(1, num_stages-stage_idx)],
        stage_idx=stage_idx, 
        use_from_gt_perspective=True,  
        progress_filter=True,  
        use_static_128=True, 
        last_in_point_num=None,  
        lim_outpoints_times=1,  
        use_axis_atten=True, 
        loss_cls=dict(
            type='BoundAlphaQualityFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            alpha_bgfg2fg=0.25,
            loss_weight=2.0),
        ori_focal_loss=False,
        # NOTE: The following argument is a placeholder to hack the code. No real effects for decoding or updating bounding boxes.
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=False,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.5, 0.5, 1., 1.])
    ) for stage_idx in range(num_stages)
]

for i in range(1, num_stages):
    box_basic_heads[i]['subbox_poolsize'] = subbox_poolsize_rcnn
    box_basic_heads[i]['last_in_point_num'] = in_points_list[i-1] * box_basic_heads[i-1]['subbox_poolsize']  
    

# for i in range(0, 1):
#     box_basic_heads[i]['targets_candi_ids'] = None
#     box_basic_heads[i]['ori_focal_loss'] = True
#     box_basic_heads[i]['loss_cls'] = \
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=2.0)
#     box_basic_heads[i]['loss_bbox']=dict(type='L1Loss', loss_weight=5.0)
#     box_basic_heads[i]['loss_iou']=dict(type='GIoULoss', loss_weight=2.0)
#     box_basic_heads[i]['use_soft_iou']=False

# for i in range(0, 2):
#     box_basic_heads[i]['ori_focal_loss'] = True
#     box_basic_heads[i]['loss_cls'] = \
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=2.0)


model = dict(
    type='SubQueryDeformableRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        #with_cp=True, 
        style='pytorch'),
    neck=dict(
        type='ChannelMapping',
        in_channels=[256, 512, 1024, 2048],
        out_channels=FEAT_DIM,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4),
    rpn_head=dict(
        type='SubInitialQueryGenerator',
        num_query=num_proposals,
        content_dim=QUERY_DIM,
        scale_num=N_scale,
        per_group_point_num=in_points_list[0],
        point_group_num=n_group_list[0],
        anchor_point_num=anchor_point_num,
        anchor_channel=anchor_channel,
        ),
    roi_head=dict(
        type='StageInteractorDecoder',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        content_dim=QUERY_DIM,
        use_iou=True, 
        bbox_head=box_basic_heads),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='SeqHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.000025,
    weight_decay=0.0001,
)

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=1.0, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[8, 11],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001
)
runner = dict(type='EpochBasedRunner', max_epochs=12)


def __date():
    import datetime
    return datetime.datetime.now().strftime('%m%d_%H%M')


log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

postfix = '_' + __date()

find_unused_parameters = True
#find_unused_parameters = False 


resume_from = None
