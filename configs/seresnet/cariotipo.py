_base_ = [
    './seresnet101_8xb32_in1k.py'
]


mlflow_tags = {
        "model_type": "SeresNet",
        "dataset": "Cariotipo",
}

custom_hooks = [
    dict(type='MlflowLoggerHook',  exp_name="SeresNet" , params=mlflow_tags)
]
