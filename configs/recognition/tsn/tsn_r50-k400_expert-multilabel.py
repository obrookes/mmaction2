_base_ = [
    "../../_base_/models/tsn_r50.py",
    "../../_base_/schedules/sgd_100e.py",
    "../../_base_/default_runtime.py",
]

load_from = "pretrained_models/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"

# dataset settings
dataset_type = "VideoDataset"
data_root = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/all"
ann_file_train = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/train.txt"
ann_file_val = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/val.txt"
ann_file_test = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/expert_multilabel/test.txt"

file_client_args = dict(io_backend="disk")

model = dict(
    cls_head=dict(
        type="TSNHead",
        num_classes=4,
        loss_cls=dict(type="BCELossWithLogits"),
        multi_class=True,
    )
)

num_frames = 16

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=num_frames, num_clips=1),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="PytorchVideoWrapper", op="RandAugment", magnitude=7, num_layers=4),
    dict(type="RandomResizedCrop"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=num_frames, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="UniformSample", clip_len=num_frames, num_clips=1, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 224)),
    dict(type="ThreeCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
        multi_class=True,
        num_classes=4,
    ),
)
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True,
        multi_class=True,
        num_classes=4,
    ),
)
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True,
        multi_class=True,
        num_classes=4,
    ),
)

val_evaluator = dict(type="AccMetric", metric_list=("mean_average_precision"))
test_evaluator = dict(type="AccMetric", metric_list=("mean_average_precision"))

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (1 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
