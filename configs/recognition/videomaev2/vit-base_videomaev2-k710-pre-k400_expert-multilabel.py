_base_ = ["../../_base_/default_runtime.py"]

load_from = "pretrained_models/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth"

dataset_type = "VideoDataset"
data_root = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/all"
ann_file_train = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/train.txt"
ann_file_val = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/val.txt"
ann_file_test = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/test.txt"

file_client_args = dict(io_backend="disk")

num_frames = 16
batch_size = 8
num_classes = 4
base_batch_size = 512

# model settings
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="VisionTransformer",
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=num_frames,
        norm_cfg=dict(type="LN", eps=1e-6),
    ),
    cls_head=dict(
        type="TimeSformerHead",
        average_clips="prob",
        num_classes=num_classes,
        in_channels=768,
        loss_cls=dict(type="BCELossWithLogits"),
        multi_class=True,
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape="NCTHW",
    ),
)

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=num_frames, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="PytorchVideoWrapper", op="RandAugment", magnitude=7, num_layers=4),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    dict(type="DecordInit"),
    dict(type="UniformSample", clip_len=num_frames, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

test_pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=16, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
        num_classes=num_classes,
        multi_class=True,
    ),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True,
        num_classes=num_classes,
        multi_class=True,
    ),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True,
        num_classes=num_classes,
        multi_class=True,
    ),
)

base_lr = 2e-5
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=20, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1 / 20,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min_ratio=1 / 20,
        by_epoch=True,
        begin=5,
        end=24,
        convert_to_iter_based=True,
    ),
]

val_evaluator = dict(
    type="AccMetric",
    metric_list=("mean_average_precision"),
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=48, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(
        interval=5, max_keep_ckpts=1, save_best="acc/mean_average_precision"
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (3 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=base_batch_size)

vis_backends = [dict(type="LocalVisBackend"), dict(type="WandbVisBackend")]

visualizer = dict(
    type="ActionVisualizer",
    vis_backends=vis_backends,
)
