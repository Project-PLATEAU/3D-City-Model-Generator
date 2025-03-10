auto_scale_lr = dict(base_batch_size=32)
data_preprocessor = dict(
    mean=[
        122.7709383,
        116.7460125,
        104.09373615000001,
    ],
    num_classes=5,
    std=[
        68.5005327,
        66.6321579,
        70.32316304999999,
    ],
    to_rgb=True)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        out_dir='checkpoints/eva02-large/20241218',
        save_best='auto',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = './checkpoints/eva02-large/20241210/eva02-large-p14_my/best_accuracy_top1_epoch_24.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='l',
        final_norm=False,
        img_size=448,
        out_type='avg_featmap',
        patch_size=14,
        sub_ln=True,
        type='ViTEVA02'),
    head=dict(
        in_channels=1024,
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        num_classes=5,
        type='LinearClsHead'),
    init_cfg=[
        dict(layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    neck=None,
    train_cfg=dict(augments=[
        dict(alpha=0.8, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(lr=2.3e-05, type='AdamW', weight_decay=0.3),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=15,
        start_factor=0.001,
        type='LinearLR'),
    dict(begin=15, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=150,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/test2.txt',
        classes=[
            'FL',
            'SFL',
            'FD',
            'HP',
            'GB',
        ],
        data_prefix='test2',
        data_root=
        '/home/mdxuser/Desktop/jupyter/mmpretrain/dataset/roof_type_v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=448,
                type='ResizeEdge'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=448,
        type='ResizeEdge'),
    dict(crop_size=448, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='meta/train.txt',
        classes=[
            'FL',
            'SFL',
            'FD',
            'HP',
            'GB',
        ],
        data_prefix='train',
        data_root=
        '/home/mdxuser/Desktop/jupyter/mmpretrain/dataset/roof_type_v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=448,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=448,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=100,
    dataset=dict(
        ann_file='meta/val.txt',
        classes=[
            'FL',
            'SFL',
            'FD',
            'HP',
            'GB',
        ],
        data_prefix='val',
        data_root=
        '/home/mdxuser/Desktop/jupyter/mmpretrain/dataset/roof_type_v2',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=448,
                type='ResizeEdge'),
            dict(crop_size=448, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
warmup_epochs = 8
work_dir = './work_dirs/eva02-large-p14_my'
