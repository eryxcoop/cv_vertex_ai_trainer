import json
import logging
import os
import random
import shutil
import datetime
from pathlib import Path

import torch


class TrainingScript:
    def __init__(self) -> None:
        self.image_size = [int(size) for size in os.environ["IMAGE_SIZE"].replace(" ", "").split(",")]
        self.epochs = int(os.environ["EPOCHS"])
        self.model = os.environ["MODEL"]
        self.label_studio_token = os.environ["LABEL_STUDIO_TOKEN"]
        self.label_studio_project_url = os.environ["LABEL_STUDIO_PROJECT_URL"]
        self.images_bucket_path = os.environ["IMAGES_BUCKET_PATH"]
        self.base_path = os.getcwd()
        self.bucket_path = os.environ["BUCKET_PATH"]
        self.dataset_path = Path("dataset")
        self.number_folds = int(os.environ["NUMBER_OF_FOLDS"])
        self.save_path = Path(
            self.dataset_path / f"{datetime.date.today().isoformat()}_{self.number_folds}-Fold_Cross-val")
        self.accelerator_count = int(os.environ["ACCELERATOR_COUNT"])
        self.training_results_path = self.save_path / "training_results"
        self.fold_datasets_path = self.save_path / "folds_datasets"

    def run(self):
        self._check_if_gpu_is_available()
        self._prepare_dataset()
        self._train_model()
        self._export_results()

    def _check_if_gpu_is_available(self):
        gpu_available = torch.cuda.is_available()
        logging.info(f"Checking if GPU is available: {gpu_available}")
        if self.accelerator_count > 0 and not gpu_available:
            logging.error(f"GPU is not available, accelerator count: {self.accelerator_count}")
            exit(1)

    def _prepare_dataset(self):
        os.system('/root/.local/bin/mim install -q "mmengine>=0.6.0"')
        os.system('/root/.local/bin/mim install -q "mmcv>=2.0.0rc4,<2.1.0"')
        os.system('/root/.local/bin/mim install -q "mmdet>=3.0.0rc6,<3.1.0"')
        os.system('git clone https://github.com/open-mmlab/mmyolo.git')
        os.system('pip install -e mmyolo')

        os.system(f'mkdir -p /data')
        os.makedirs("data/custom-dataset", exist_ok=True)

        os.system(f"curl -X GET '{self.label_studio_project_url}/export?exportType=COCO&download_resources=true&download_all_tasks=true' -H 'Authorization: Token {self.label_studio_token}' --output 'annotations.zip'")

        os.system('unzip annotations -d data/custom-dataset/')

        os.system(f'gsutil -m cp -r "{self.images_bucket_path}" /data/custom-dataset')

        all_images = os.listdir(f"/data/custom-dataset/images")

        file_path = f'/data/custom-dataset/result.json'
        with open(file_path, "r") as json_file:
            annotations = json.load(json_file)

        for image in annotations["images"]:
            image['file_name'] = image['file_name'].split('/')[-1]

        with open(file_path, "w") as json_file:
            json.dump(annotations, json_file)

        val_percentage = 0.2
        random.seed(1)
        val_set = random.sample(all_images, int(len(all_images) * val_percentage))
        print(f"There are {len(val_set)} images in your val set")

        os.mkdir(f"/data/custom-dataset/val/")
        os.mkdir(f"/data/custom-dataset/train/")

        for image in all_images:
            if image in val_set:
                shutil.copy(f"/data/custom-dataset/images/{image}", f"/data/custom-dataset/val/")
            else:
                shutil.copy(f"/data/custom-dataset/images/{image}", f"/data/custom-dataset/train/")

        train_annotations = {'categories': annotations["categories"], "info": annotations["info"], "images": [],
                             "annotations": []}
        for image in os.listdir(f"/data/custom-dataset/train"):
            image_id = None
            for i in annotations["images"]:
                if image == i["file_name"]:
                    image_id = i["id"]
                    train_annotations["images"].append(i)
            for a in annotations["annotations"]:
                if image_id == a["image_id"]:
                    train_annotations["annotations"].append(a)

        val_annotations = {'categories': annotations["categories"], "info": annotations["info"], "images": [],
                           "annotations": []}
        for image in os.listdir(f"/data/custom-dataset/val"):
            image_id = None
            for i in annotations["images"]:
                if image == i["file_name"]:
                    image_id = i["id"]
                    val_annotations["images"].append(i)
            for a in annotations["annotations"]:
                if image_id == a["image_id"]:
                    val_annotations["annotations"].append(a)

        with open(f"/data/custom-dataset/train/annotations.json", "w") as json_file:
            json.dump(train_annotations, json_file)

        with open(f"/data/custom-dataset/val/annotations.json", "w") as json_file:
            json.dump(val_annotations, json_file)

        BATCH_SIZE = 8
        classes = tuple([value["name"] for value in annotations["categories"]])

        CUSTOM_CONFIG_PATH = f"/mmyolo/configs/rtmdet/rtmdet_l_syncbn_fast_8xb32-300e_coco.py"

        # FIXME: Embedding the config file here is not a good practice, but it's the only way we found to make it work
        # the problem is that the config vars are reused within the same file, so if we load the config from the file:
        #
        # from mmengine.config import Config
        # config = Config.fromfile(CUSTOM_CONFIG_PATH)
        # config["class_name"] = classes
        # config["num_classes"] = len(classes)
        # config["max_epochs"] = self.epochs
        # config["train_batch_size_per_gpu"] = BATCH_SIZE
        # config.dump(CUSTOM_CONFIG_PATH)
        #
        # the config vars are not replaced in the right order, so the config file is not valid

        CUSTOM_CONFIG = f"""
_base_ = ['../_base_/default_runtime.py', '../_base_/det_p5_tta.py']

# ========================Frequently modified parameters======================
# -----data related-----
data_root = f'/data/custom-dataset/'

train_ann_file = 'train/annotations.json'
train_data_prefix = 'train/'

val_ann_file = 'val/annotations.json'
val_data_prefix = 'val/'

class_name = {classes}
num_classes = {len(classes)}

metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

train_batch_size_per_gpu = {BATCH_SIZE}
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 4
# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.004
max_epochs = {self.epochs}  # Maximum training epochs
# Change train_pipeline for final 20 epochs (stage 2)
num_epochs_stage2 = 20

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (960, 540)  # width, height
# ratio range for random resize
random_resize_ratio_range = (0.1, 2.0)
# Cached images number in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixup
mixup_max_cached_images = 20
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 10

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 1.0
# The scaling factor that controls the width of the network structure
widen_factor = 1.0
# Strides of multi-scale prior box
strides = [8, 16, 32]

norm_cfg = dict(type='BN')  # Normalization config

# -----train val related-----
lr_start_factor = 1.0e-5
dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss
weight_decay = 0.05

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
# https://mmengine.readthedocs.io/en/latest/api/visualization.html
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='CSPNeXtPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetHead',
        head_module=dict(
            type='RTMDetSepBNHeadModule',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=qfl_beta,
            loss_weight=loss_cls_weight),
        loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=loss_bbox_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchDynamicSoftLabelAssigner',
            num_classes=num_classes,
            topk=dsl_topk,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=model_test_cfg,
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=img_scale,
        use_cached=True,
        max_cached_images=mosaic_max_cached_images,
        pad_val=114.0),
    dict(
        type='mmdet.RandomResize',
        # img_scale is (width, height)
        scale=(img_scale[0] * 2, img_scale[1] * 2),
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=mixup_max_cached_images),
    dict(type='mmdet.PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=img_scale,
        ratio_range=random_resize_ratio_range,
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=img_scale),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    collate_fn=dict(type='yolov5_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_data_prefix),
        test_mode=True,
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Reduce evaluation time
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=weight_decay),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_checkpoint_intervals,
        max_keep_ckpts=max_keep_ckpts  # only keep latest 3 checkpoints
    ))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - num_epochs_stage2,
        switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_checkpoint_intervals,
    dynamic_intervals=[(max_epochs - num_epochs_stage2, val_interval_stage2)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
        """

        with open(CUSTOM_CONFIG_PATH, 'w') as file:
            file.write(CUSTOM_CONFIG)

    def _train_model(self):
        os.system(f'python /mmyolo/tools/train.py /mmyolo/configs/rtmdet/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py')

    def _export_results(self):
        os.system(f'gcloud storage cp -r "/work_dirs/" "{self.bucket_path}/rtmdet/{datetime.datetime.now().isoformat()}/"')


if __name__ == "__main__":
    training_script = TrainingScript()
    training_script.run()
