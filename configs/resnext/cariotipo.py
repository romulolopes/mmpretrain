_base_ = [
  'resnext50-32x4d_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth"


mlflow_tags = {
        "model_type": "Resnext",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="Resnext" , params=mlflow_tags)
]


