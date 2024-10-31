_base_ = [
    './hrnet-w18_4xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w18_3rdparty_8xb32_in1k_20220120-0c10b180.pth"

mlflow_tags = {
        "model_type": "HRNet",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="HRNet" , params=mlflow_tags)
]
