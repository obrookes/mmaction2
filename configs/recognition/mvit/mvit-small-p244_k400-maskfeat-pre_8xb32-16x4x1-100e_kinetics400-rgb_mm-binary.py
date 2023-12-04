_base_ = ["../../_base_/models/mvit_small.py", "../../_base_/default_runtime.py"]

model = dict(
    backbone=dict(
        drop_path_rate=0.1,
        dim_mul_in_attention=False,
        pretrained="pretrained_models/maskfeat_mvit-small_16xb32-amp-coslr-300e_k400_20230131-87d60b6f.pth",  # noqa: E251  # noqa
        pretrained_type="maskfeat",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor",
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        blending=dict(
            type="RandomBatchAugment",
            augments=[
                dict(type="MixupBlending", alpha=0.8, num_classes=2),
                dict(type="CutmixBlending", alpha=1, num_classes=2),
            ],
        ),
        format_shape="NCTHW",
    ),
    cls_head=dict(dropout_ratio=0.0, init_scale=0.001),
)

# dataset settings
dataset_type = "VideoDataset"
data_root = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/chimp_videos/all"
ann_file_train = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/train.txt"
ann_file_val = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/val.txt"
ann_file_test = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/test.txt"

file_client_args = dict(io_backend="disk")
train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=16, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="PytorchVideoWrapper", op="RandAugment", magnitude=7),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="Uniform", clip_len=16, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=16, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs"),
]

repeat_sample = 2
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    collate_fn=dict(type="repeat_pseudo_collate"),
    dataset=dict(
        type="RepeatAugDataset",
        num_repeats=repeat_sample,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="AccMetric",
    metric_list=("mean_average_precision"),
)
test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=100, val_begin=1, val_interval=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

base_lr = 9.6e-3  # for batch size 512
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={"decay_rate": 0.75, "decay_type": "layer_wise", "num_layers": 16},
    clip_grad=dict(max_norm=5, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1 / 600,
        by_epoch=True,
        begin=0,
        end=20,
        convert_to_iter_based=True,
    ),
    dict(
        type="CosineAnnealingLR",
        T_max=80,
        eta_min_ratio=1 / 600,
        by_epoch=True,
        begin=20,
        end=100,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(interval=3, max_keep_ckpts=1), logger=dict(interval=100)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=512 // repeat_sample)
