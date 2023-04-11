_base_ = ['./basicTAD_slowonly_96x10_1200e_thumos14_rgb.py']

# model settings
model = dict(
    type='SegmentDetector',
    # bbox_head=dict(
    #     type='RetinaHead1D',
    #     anchor_generator=dict(
    #         type='Anchor1DGenerator',
    #         octave_base_scale=2,
    #         scales_per_octave=4,
    #         strides=[1, 2, 4, 8])),
    backbone=dict(
        type='MViT',
        arch='small',
        drop_path_rate=0.2,
        spatial_size=112,
        temporal_size=96,
        t_downscale=True,
        with_cls_token=False,
        output_cls_token=False,
        out_scales=(0, 1, 2, 3),
        patch_cfg=dict(kernel_size=(3, 7, 7), stride=(1, 4, 4), padding=(1, 3, 3)),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint='work_dirs/mvit-small_p112_32x3_20e_k400-rgb/epoch_20.pth')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        spatial_pooling=True,
        add_extra_convs='on_input',
        num_outs=5,
        start_level=0),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))

# dataset settings
clip_len = 96
frame_interval = 10
img_shape = (112, 112)
img_shape_test = (112, 112)
overlap_ratio = 0.25

data_root = 'my_data/thumos14'  # Root path to data for training
data_prefix_train = 'rawframes/val'  # path to data for training
data_prefix_val = 'rawframes/test'  # path to data for validation and testing
ann_file_train = 'annotations/basicTAD/val.json'  # Path to the annotation file for training
ann_file_val = 'annotations/basicTAD/test.json'  # Path to the annotation file for validation
ann_file_test = ann_file_val
val_pipeline = [
    dict(type='Time2Frame'),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 112), keep_ratio=True),
    dict(type='SpatialCenterCrop', crop_size=img_shape_test),
    dict(type='Pad', size=(clip_len, *img_shape_test)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MyPackInputs')
]
val_dataloader = dict(  # Config of validation dataloader
    batch_size=2,  # Batch size of each single GPU during validation
    num_workers=6,  # Workers to pre-fetch data for each single GPU during validation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing
    # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py)  # Randomly shuffle the training data in each epoch
    dataset=dict(  # Config of validation dataset
        type='Thumos14ValDataset',
        clip_len=clip_len, frame_interval=frame_interval, overlap_ratio=0.25,
        filename_tmpl='img_{:05}.jpg',
        ann_file=ann_file_val,  # Path of annotation file
        data_root=data_root,
        data_prefix=dict(imgs=data_prefix_val),  # Prefix of specific data components
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader
train_dataloader = dict(batch_size=2)
#
train_cfg = dict(max_epochs=1000)
# learning policy
param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
    # Linear learning rate warm-up scheduler
    dict(type='LinearLR',
         start_factor=0.01,
         by_epoch=True,
         begin=0,
         end=200,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
         eta_min_ratio=0.01,
         by_epoch=True,
         begin=200,
         end=1000,
         convert_to_iter_based=True)]  # Convert to update by iteration.

# optimizer
optim_wrapper = dict(  # Config of optimizer wrapper
    _delete_=True,
    type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
    optimizer=dict(
        # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='AdamW',  # Name of optimizer
        lr=1e-4,  # Learning rate
        betas=(0.9, 0.999),
        weight_decay=0.05),  # Weight decay
    clip_grad=dict(max_norm=40, norm_type=2))
