_base_ = ['./basicTAD_slowonly_96x10_1200e_thumos14_rgb.py']
# model settings
model = dict(type='SegmentDetector',
             neck=[dict(type='MaxPool3d', kernel_size=(2, 1, 1), stride=(2, 1, 1)), dict(type='VDM'), dict(type='FPN')],
             bbox_head=dict(
                 type='RetinaHead1D',
                 anchor_generator=dict(
                     type='Anchor1DGenerator',
                     octave_base_scale=2,
                     scales_per_octave=5,
                     strides=[2, 4, 8, 16, 32])))

# dataset settings
data_root = 'my_data/thumos14'  # Root path to data for training
data_prefix_train = 'rawframes/val'  # path to data for training
data_prefix_val = 'rawframes/test'  # path to data for validation and testing
ann_file_train = 'annotations/basicTAD/val.json'  # Path to the annotation file for training
ann_file_val = 'annotations/basicTAD/test.json'  # Path to the annotation file for validation
ann_file_test = ann_file_val

clip_len = 192
frame_interval = 5
img_shape = (112, 112)
img_shape_test = (128, 128)
overlap_ratio = 0.25

train_pipeline = [
    dict(type='Time2Frame'),
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
    dict(type='Resize', scale=(128, -1), keep_ratio=True),
    dict(type='SpatialCenterCrop', crop_size=img_shape_test),
    dict(type='Pad', size=(clip_len, *img_shape_test)),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MyPackInputs')
]
# test_pipeline = val_pipeline

train_dataloader = dict(  # Config of train dataloader
    batch_size=2,  # Batch size of each single GPU during training
    num_workers=2,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,
    # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(  # Config of train dataset
        type='Thumos14Dataset',
        filename_tmpl='img_{:05}.jpg',
        ann_file=ann_file_train,  # Path of annotation file
        data_root=data_root,  # Root path to data, including both frames and ann_file
        data_prefix=dict(imgs=data_prefix_train),  # Prefix of specific data, e.g., frames and ann_file
        pipeline=train_pipeline))
val_dataloader = dict(  # Config of validation dataloader
    batch_size=2,  # Batch size of each single GPU during validation
    num_workers=2,  # Workers to pre-fetch data for each single GPU during validation
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
