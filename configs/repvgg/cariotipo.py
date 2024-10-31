_base_ = [
  'repvgg-A0_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth"


mlflow_tags = {
        "model_type": "RepVGG",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="RepVGG" , params=mlflow_tags)
]


