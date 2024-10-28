_base_ = [
    './seresnet101_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth"

mlflow_tags = {
        "model_type": "SeresNet",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="SeresNet" , params=mlflow_tags)
]
