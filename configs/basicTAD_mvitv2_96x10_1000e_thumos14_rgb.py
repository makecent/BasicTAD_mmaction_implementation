_base_ = ['./basicTAD_slowonly_96x10_1200e_thumos14_rgb.py']

# model settings
model = dict(
    backbone=dict(
        type='MViT_TRN',
        arch='small',
        drop_path_rate=0.2,
        spatial_size=112,
        temporal_size=96,
        with_cls_token=False,
        output_cls_token=False,
        patch_cfg=dict(kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3)),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb_20230201-23284ff3.pth'
        )
    ),
    neck=[
        dict(
            type='VDM',
            in_channels=768,
            out_channels=256,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='SyncBN'),
            kernel_sizes=(3, 1, 1),
            strides=(2, 1, 1),
            paddings=(1, 0, 0),
            stage_layers=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3, 4),
            out_pooling=True),
        dict(type='mmdet.FPN',
             in_channels=[768, 256, 256, 256, 256],
             out_channels=256,
             num_outs=5,
             conv_cfg=dict(type='Conv1d'),
             norm_cfg=dict(type='SyncBN'))],
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

clip_len = 96
frame_interval = 10
img_shape = (112, 112)
img_shape_test = (112, 112)

train_pipeline = [
    dict(type='Time2Frame'),
    # dict(type='RandSlideAug'),
    dict(type='TemporalRandomCrop',
         clip_len=clip_len,
         frame_interval=frame_interval,
         iof_th=0.75),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1), keep_ratio=True),  # scale images' short-side to 128, keep aspect ratio
    dict(type='SpatialRandomCrop', crop_size=img_shape),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18,
         p=0.5),
    dict(type='Rotate',
         limit=(-45, 45),
         border_mode='reflect_101',
         p=0.5),
    dict(type='Pad', size=(clip_len, *img_shape)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MyPackInputs')]
val_pipeline = [
    dict(type='Time2Frame'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 112), keep_ratio=True),
    dict(type='SpatialCenterCrop', crop_size=img_shape_test),
    dict(type='Pad', size=(clip_len, *img_shape_test)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MyPackInputs')
]
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader
train_dataloader = dict(dataset=dict(pipeline=train_pipeline), batch_size=2)

# optimizer settings
train_cfg = dict(max_epochs=1000)
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.01,
         by_epoch=True,
         begin=0,
         end=200,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR',
         eta_min_ratio=0.01,
         by_epoch=True,
         begin=200,
         end=1000,
         convert_to_iter_based=True)]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=1, norm_type=2))
# compile=True