_base_ = ['./basicTAD_slowonly_96x10_1200e_thumos14_rgb.py']

# model settings
model = dict(
    backbone=dict(
        type='MViT',
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
            out_channels=768,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='SyncBN'),
            kernel_sizes=(3, 1, 1),
            strides=(2, 1, 1),
            paddings=(1, 0, 0),
            stage_layers=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3, 4),
            out_pooling=True),
        dict(type='mmdet.FPN',
             in_channels=[768] * 5,
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
img_shape_test = (112, 112)

val_pipeline = [
    dict(type='Time2Frame'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 112), keep_ratio=True),
    dict(type='SpatialCenterCrop', crop_size=img_shape_test),
    dict(type='Pad', size=(clip_len, *img_shape_test)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MyPackInputs')
]
val_dataloader = dict( dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader
train_dataloader = dict(batch_size=2)

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
    clip_grad=dict(max_norm=40, norm_type=2))
