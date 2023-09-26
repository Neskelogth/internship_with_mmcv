# model_cfg
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=False,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        base_channels=32,
        conv1_kernel=(1, 7, 7),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 0, 1, 1),
        out_indices=(2, )),
    pose_pathway=dict(
        num_stages=4,
        lateral=False,
        lateral_inv=False,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        in_channels=17,
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 0, 1, 1)))
head_cfg = dict(
    type='RGBPoseHeadCat',
    num_classes=60,
    in_channels=2048,
    temporal=True)
test_cfg = dict(average_clips='prob')
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    test_cfg=test_cfg)

dataset_type = 'PoseDataset'
data_root = '../../datasets/nturgbd/nturgb+d_rgb/'
ann_file = './data/nturgbd/ntu60_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=False, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=False, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=8), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(128, 128), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=False, with_kp=True, with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=8),
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=8),
    train=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', data_prefix=data_root,
               pipeline=train_pipeline, origin='json'),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root,
             pipeline=val_pipeline, origin='json'),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root,
              pipeline=test_pipeline, origin='json'))
# optimizer
optimizer = dict(type='Adam', lr=1e-3, fused=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 25
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5),
                  key_indicator='RGBPose_1:1_top1_acc')
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/late_only/late_only_cat/openpose/xsub'
load_from = 'https://download.openmmlab.com/mmaction/pyskl/ckpt/rgbpose_conv3d/rgbpose_conv3d_init.pth'
