_base_ = [
  'res2net50-w14-s8_8xb32_in1k.py'
]

load_from = "https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth"


mlflow_tags = {
        "model_type": "Res2net",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="Res2net" , params=mlflow_tags)
]


