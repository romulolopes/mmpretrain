# hivit-tiny-p16_16xb64_in1k
#modificado dataset imagenet_bs64_hivit_224
_base_ = [
    '../_base_/models/hivit/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_hivit_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_hivit.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

