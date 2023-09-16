model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        in_channels=20,  # 17 channels for pose, 3 for rgb image
        base_channels=32,
        num_stages=4,
        temporal_downsample=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=256,
        num_classes=60,
        dropout=0.5),
    test_cfg=dict(average_clips='prob'))


dataset_type = 'PoseDataset'
data_root = '../../datasets/nturgbd/nturgb+d/'
ann_file = './data/nturgbd/ntu60_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=32, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='StackFrames'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=32, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='StackFrames'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=32, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='StackFrames'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]


data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=4),
    train=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', data_prefix=data_root,
               pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root, pipeline=test_pipeline))

optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/early_fusion/hrnet/xsub/'
