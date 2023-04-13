_base_ = ['./basicTAD_slowonly_192x5_1200e_thumos14_rgb.py']

clip_len = 192
frame_interval = 10
img_shape = (112, 112)

model = dict(
    neck=[
        dict(type='VDM',
             in_channels=2048,
             out_channels=512,
             conv_cfg=dict(type='Conv3d'),
             norm_cfg=dict(type='SyncBN'),
             kernel_sizes=(3, 1, 1),
             strides=(2, 1, 1),
             paddings=(1, 0, 0),
             stage_layers=(1, 1, 1, 1, 1),
             out_indices=(0, 1, 2, 3, 4, 5),
             out_pooling=True),
        dict(type='mmdet.FPN',
             in_channels=[2048, 512, 512, 512, 512, 512],
             out_channels=256,
             num_outs=6,
             conv_cfg=dict(type='Conv1d'),
             norm_cfg=dict(type='SyncBN'))],
    bbox_head=dict(anchor_generator=dict(strides=[1, 2, 4, 8, 16, 32])))

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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(clip_len=clip_len, frame_interval=frame_interval))
test_dataloader = val_dataloader
