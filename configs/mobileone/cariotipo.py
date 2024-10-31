_base_ = [
  'mobileone-s0_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/mobileone/mobileone-s0_8xb32_in1k_20221110-0bc94952.pth"


mlflow_tags = {
        "model_type": "Mobile One",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="Mobile One" , params=mlflow_tags)
]
