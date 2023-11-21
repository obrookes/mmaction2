_base_ = [
    "../../_base_/models/tsn_r50.py",
    "../../_base_/schedules/sgd_100e.py",
    "../../_base_/default_runtime.py",
]

# model settings
model = dict(cls_head=dict(type="TSNHead", num_classes=2))  # change from 400 to 101

dataset_type = "VideoDataset"
data_root = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/chimp_videos/all"
ann_file_train = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/train.txt"
ann_file_val = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/val.txt"
ann_file_test = "/jmain02/home/J2AD001/wwp02/oxb63-wwp02/data/camera_reaction/annotations/mmaction2/mm_binary/test.txt"

file_client_args = dict(io_backend="disk")

train_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(
        type="MultiScaleCrop",
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
    ),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
val_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=3, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="CenterCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]
test_pipeline = [
    dict(type="DecordInit", **file_client_args),
    dict(
        type="SampleFrames", clip_len=1, frame_interval=1, num_clips=25, test_mode=True
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="TenCrop", crop_size=224),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="PackActionInputs"),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
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

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,  # change from 100 to 50
    val_begin=1,
    val_interval=1,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning policy
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=50,  # change from 100 to 50
        by_epoch=True,
        milestones=[20, 40],  # change milestones
        gamma=0.1,
    )
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="SGD",
        lr=0.005,  # change from 0.01 to 0.005
        momentum=0.9,
        weight_decay=0.0001,
    ),
    clip_grad=dict(max_norm=40, norm_type=2),
)

val_evaluator = dict(type="AccMetric")
test_evaluator = val_evaluator

default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=256)

# use the pre-trained model for the whole TSN network
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth"
