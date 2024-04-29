default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=200),
    checkpoint=dict(
        type='CheckpointHook', interval=2, save_best='auto', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
file_client_args = dict(backend='disk')
_file_client_args = dict(backend='disk')
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
img_scales = [(640, 640), (320, 320), (960, 960)]
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(320, 320)),
            dict(
                type='LetterResize',
                scale=(320, 320),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
            dict(
                type='LetterResize',
                scale=(960, 960),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ])
]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (640, 640)
            }, {
                'type': 'LetterResize',
                'scale': (640, 640),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (320, 320)
            }, {
                'type': 'LetterResize',
                'scale': (320, 320),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (960, 960)
            }, {
                'type': 'LetterResize',
                'scale': (960, 960),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }], [{
                        'type': 'mmdet.LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'pad_param', 'flip', 'flip_direction')
                    }]])
]
data_root = './data/piglet/'
train_ann_file = 'annotations/coco_train.json'
train_data_prefix = 'trainImages/JPEGImages/'
val_ann_file = 'annotations/coco_val.json'
val_data_prefix = 'valImages/JPEGImages/'
test_ann_file = 'annotations/coco_test.json'
test_data_prefix = 'testImages/JPEGImages/'
num_classes = 2
train_batch_size_per_gpu = 8
train_num_workers = 4
persistent_workers = True
anchors = [[(68, 69), (154, 91), (143, 162)],
           [(242, 160), (189, 287), (391, 207)],
           [(353, 337), (539, 341), (443, 432)]]
num_last_epochs = 15
base_lr = 0.00625
max_epochs = 200
model_test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.1,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=300)
img_scale = (640, 640)
dataset_type = 'YOLOv5CocoDataset'
val_batch_size_per_gpu = 1
val_num_workers = 2
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=640,
    size_divisor=32,
    extra_pad_ratio=0.5)
deepen_factor = 0.33
widen_factor = 1.0
strides = [8, 16, 32]
num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
affine_scale = 0.5
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0
obj_level_weights = [4.0, 1.0, 0.4]
lr_factor = 0.01
weight_decay = 0.0001
save_checkpoint_intervals = 10
max_keep_ckpts = 3
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=0.33,
        widen_factor=1.0,
        in_channels=[192, 384, 768],
        out_channels=[192, 384, 768],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=10,
            in_channels=[192, 384, 768],
            widen_factor=1.0,
            featmap_strides=[8, 16, 32],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.0625),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-07,
            reduction='mean',
            loss_weight=0.05,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.4]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.1,
        nms=dict(type='nms', iou_threshold=0.3),
        max_per_img=300))
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-320, -320),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=True,
        pad_val=dict(img=114)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(0.5, 1.5),
        max_shear_degree=0.0),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/piglet/',
        ann_file='annotations/coco_train.json',
        data_prefix=dict(img='trainImages/JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(640, 640),
                pad_val=114.0,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(0.5, 1.5),
                border=(-320, -320),
                border_val=(114, 114, 114)),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='Blur', p=0.01),
                    dict(type='MedianBlur', p=0.01),
                    dict(type='ToGray', p=0.01),
                    dict(type='CLAHE', p=0.01)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap=dict(img='image', gt_bboxes='bboxes')),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ],
        metainfo=dict(classes=('sow', 'piglet'))))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
    dict(
        type='LetterResize',
        scale=(640, 640),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/piglet/',
        test_mode=True,
        data_prefix=dict(img='valImages/JPEGImages/'),
        ann_file='annotations/coco_val.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        metainfo=dict(classes=('sow', 'piglet'))))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='./data/piglet/',
        test_mode=True,
        data_prefix=dict(img='testImages/JPEGImages/'),
        ann_file='annotations/coco_test.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=640,
            size_divisor=32,
            extra_pad_ratio=0.5),
        metainfo=dict(classes=('sow', 'piglet'))))
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.00625,
        momentum=0.937,
        weight_decay=0.0001,
        nesterov=True,
        batch_size_per_gpu=16),
    constructor='YOLOv5OptimizerConstructor')
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=185,
        switch_pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=True,
                pad_val=dict(img=114)),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_translate_ratio=0.1,
                scaling_ratio_range=(0.5, 1.5),
                max_shear_degree=0.0),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ])
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='./data/piglet/annotations/coco_val.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='./data/piglet/annotations/coco_test.json',
    metric='bbox')
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=200, val_interval=2, val_begin=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
save_epoch_intervals = 2
class_name = ('sow', 'piglet')
metainfo = dict(classes=('sow', 'piglet'))
launcher = 'none'
work_dir = './work_dirs/yolov5_s_swintrans-2classes'
