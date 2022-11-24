_base_ = 'solov2_r50_fpn_3x_coco.py'

# model settings
model = dict(
    backbone=dict(
        in_channels=4,
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    mask_head=dict(
        feat_channels=256,
        num_classes=1,
        stacked_convs=3,
        scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
        mask_feature_head=dict(out_channels=128),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=False))  # light solov2 head

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=50)

# data
dataset_type = 'BoxesCocoStyleDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 1521.0897], std=[58.395, 57.12, 57.375, 652.615], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(640,480), (768, 512), (768, 480), (768, 448), (768, 416), (768, 384),
                   (768, 352)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 480),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, override=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(type=dataset_type,
               ann_file=data_root + 'train/depth_annotations.json',
               img_prefix=data_root + 'train/',
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             ann_file=data_root + 'test/depth_annotations.json',
             img_prefix=data_root + 'test/',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              ann_file=data_root + 'test/depth_annotations.json',
              img_prefix=data_root + 'test/',
              pipeline=test_pipeline))
