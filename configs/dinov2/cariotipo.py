_base_ = [
  'vit-small-p14_dinov2-pre_headless.py'
]







load_from = "https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth"

num_classes = 2
data_preprocessor = dict(
    num_classes=num_classes,
    _delete_=True
    )

model = dict(
    head=dict(
        num_classes=num_classes,
        _delete_=True
    ))

warmup_epochs = 10
base_lr = 5e-4

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.001),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs)
]


data_root = 'data/cariotipo/'
train_image_folder = "train"
val_image_folder = "test" 
IMAGENET_CATEGORIES = ["alterado", "normal"]
METAINFO = {'classes': IMAGENET_CATEGORIES}

train_dataloader = dict(
    dataset=dict(
        metainfo=METAINFO,
        data_root=data_root,
        data_prefix=train_image_folder
        )
)

val_dataloader = dict(
    dataset=dict(
        metainfo=METAINFO,
        data_root=data_root,
        data_prefix=val_image_folder
        )
)

val_evaluator = dict(type='Accuracy', topk=(1, ))
#test_evaluator = val_evaluator


max_epochs = 30
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
#test_cfg = dict()

# local path to saving the models and logs
work_dir = "./out"

# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', max_keep_ckpts=1),

)

mlflow_tags = {
        "model_type": "Dino v2",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="Dino v2" , params=mlflow_tags)
]
