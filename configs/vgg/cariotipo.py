_base_ = [
  'vgg11_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth"


mlflow_tags = {
        "model_type": "VGG",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="VGG" , params=mlflow_tags)
]
