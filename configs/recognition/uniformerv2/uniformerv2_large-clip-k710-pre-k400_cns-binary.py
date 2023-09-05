_base_ = ["../../_base_/default_runtime.py"]

load_from = "pretrained_models/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics400-rgb_20221219-6dc86d05.pth"

dataset_type = "VideoDataset"
data_root = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/panaf_20k/data"
ann_file_train = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/cns_binary/train.txt"
ann_file_val = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/cns_binary/val.txt"
ann_file_test = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/cns_binary/test.txt"

file_client_args = dict(io_backend="disk")

num_frames = 16
batch_size = 3
num_classes = 2

# model settings
model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="UniFormerV2",
        input_resolution=224,
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.0,
        temporal_downsample=False,
        no_lmhra=True,
        double_lmhra=True,
        return_list=[20, 21, 22, 23],
        n_layers=4,
        n_dim=1024,
        n_head=16,
        mlp_factor=4.0,
        drop_path_rate=0.0,
        mlp_dropout=[0.5, 0.5, 0.5, 0.5],
    ),
    cls_head=dict(
        type="TimeSformerHead",
        in_channels=1024,
        num_classes=num_classes,
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
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
    dict(type="UniformSample", clip_len=num_frames, num_clips=1, test_mode=True),
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
    ),
)


base_lr = 2e-5
optim_wrapper = dict(
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

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_evaluator = dict(type="AccMetric", metric_list=("mean_average_precision"))
val_evaluator = dict(type="AccMetric", metric_list=("mean_average_precision"))
test_evaluator = dict(type="AccMetric", metric_list=("mean_average_precision"))

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=batch_size)
