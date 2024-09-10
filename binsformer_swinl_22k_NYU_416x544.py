backbone_norm_cfg = dict(requires_grad=True, type='LN')
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'
autodl_tmp = '/root/autodl-tmp/'
data_root = '/root/autodl-tmp/data/nyu'
work_dir = autodl_tmp+'work_dirs/binsformer_swinl_22k_NYU_416x544'
batch_size=10
max_depth_eval=20.0
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=100)
dataset_type = 'NYUDataset'
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=0.001, type='AdamW', weight_decay=0.0001)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
default_hooks = dict(
    #only save the best model.
    checkpoint=dict(by_epoch=False, type='CheckpointHook',save_best=['abs_rel','rmse'], rule=['less','less']),
    logger=dict(interval=10, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmdepth'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'pytorch'
load_from = '/root/autodl-tmp/work_dirs/binsformer_swinl_22k_NYU_416x544/best_abs_rel_iter_12000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
crop_size = (
    416,
    544,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.Binsformer.decode_head',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        127.5,
        127.5,
        127.5,
    ],
    pad_val=0,
    seg_pad_val=0,
    size=(
        416,
        544,
    ),
    std=[
        127.5,
        127.5,
        127.5,
    ],
    type='SegDataPreProcessor')
model = dict(
    backbone=dict(
        depths=[
            2,
            2,
            18,
            2,
        ],
        embed_dims=192,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth',
            type='Pretrained'),
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        pretrain_img_size=224,
        type='SwinTransformer',
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=0,
        size=(
            416,
            544,
        ),
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        enforce_decoder_input_project=False,
        max_depth=20,
        feat_channels=256,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        num_classes=1,
        num_queries=64,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(num_groups=32, type='GN'),
            type='mmdet.PixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    add_identity=True,
                    dropout_layer=None,
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.1,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_drop=0.1,
                    batch_first=True,
                    dropout_layer=None,
                    embed_dims=256,
                    num_heads=8,
                    proj_drop=0.1)),
            num_layers=9,
            return_intermediate=True),
        type='BinsFormerDecodeHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(
        aux_index=[
            2,
            5,
        ], aux_loss=True, aux_weight=[
            0.25,
            0.5,
        ]),
    type='DepthEstimator')
norm_cfg = dict(requires_grad=True, type='SyncBN')

resume = False
test_cfg = dict(type='TestLoop')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
    # dict(
    #     transforms=[
    #         dict(type='CenterCrop',height=673,width=1197, p=1),
    #         dict(type='PadIfNeeded',min_height=720,min_width=1280,position='center', border_mode=0,value=0,p=1),
    #     ],
    #     type='Albu'),
    dict(
        meta_keys=(
            'img_path',
            'depth_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'category_id',
        ),
        type='PackSegInputs'),
]
test_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_prefix=dict(
            #depth_map_path='annotations/test', img_path='images/test',
            depth_map_path='depth/test', img_path='images/test'
            ),
        #data_root='/root/autodl-tmp/data/nyu',
        data_root='/root/autodl-tmp/ZED',
        pipeline=test_pipeline,
        test_mode=True,
        type='NYUDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    depth_scale_factor=1000.0,
    max_depth_eval=max_depth_eval,
    min_depth_eval=0.001,
    type='DepthMetric')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
    dict(prob=0.5, type='RandomFlip'),
    dict(crop_size=(
        416,
        544,
    ), type='RandomCrop'),
    # dict(
    #     transforms=[
    #         dict(type='RandomBrightnessContrast',brightness_limit = 0.3, contrast_limit= 0.3, p=0.25),
    #         dict(type='RandomGamma',gamma_limit=(50,200),p=0.25),
    #         dict(type='HueSaturationValue',hue_shift_limit=30,sat_shift_limit=30,val_shift_limit=30,p=0.25),
    #         dict(type='RGBShift',r_shift_limit=20,g_shift_limit=20,b_shift_limit=20,p=0.25)
    #     ],
    #     type='Albu'),
    dict(
        meta_keys=(
            'img_path',
            'depth_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'category_id',
        ),
        type='PackSegInputs'),
]
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='annotations/train', img_path='images/train'),
        data_root='/root/autodl-tmp/data/nyu',
        pipeline=train_pipeline,
        type='NYUDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))

tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_prefix=dict(
            depth_map_path='annotations/test', img_path='images/test'),
        data_root='/root/autodl-tmp/data/nyu',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(depth_rescale_factor=0.001, type='LoadDepthAnnotation'),
            dict(
                meta_keys=(
                    'img_path',
                    'depth_map_path',
                    'ori_shape',
                    'img_shape',
                    'pad_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'category_id',
                ),
                type='PackSegInputs'),
        ],
        test_mode=True,
        type='NYUDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    depth_scale_factor=1000.0,
    max_depth_eval=max_depth_eval,
    min_depth_eval=0.001,
    type='DepthMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])

