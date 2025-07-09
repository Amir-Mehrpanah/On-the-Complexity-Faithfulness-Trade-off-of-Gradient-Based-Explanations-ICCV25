# %% imports
import os

import torch

cwd = "/proj/azizpour-group/users/x_amime/projects/kernel-view-to-explainability/"
os.chdir(cwd)

from submission.utils import (
    submit_measurements,
    submit_training,
    submit_grads,
    submit_explainers,
)
from src.utils import (
    ActivationSwitch,
    ExplainerSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    convert_str_to_explainer,
)

force_run = True  # set to True to force run even if checkpoints exist

if False:  # debug
    port = 5678
    block_main = True
    timeout = 10
else:
    port = None
    block_main = False
    timeout = 60

seed = [
    0,
    # 1,
    # 2,
    # 3,
    # 4,
    # 5,
    # 6,
    # 7,
    # 8,
    # 9,
]
activation = [
    ActivationSwitch.RELU,
    ActivationSwitch.GELU,
    # Only for ablation studies:
    # ActivationSwitch.LEAKY_RELU,
    # ActivationSwitch.SIGMOID,
    # ActivationSwitch.TANH,
    #
    #
    #### order matters! in this a very ugly list :/
    #### because we do string matching sometimes
    #
    #
    # ActivationSwitch.SOFTPLUS_B_9,
    # ActivationSwitch.SOFTPLUS_B_8,
    # ActivationSwitch.SOFTPLUS_B_7,
    # ActivationSwitch.SOFTPLUS_B_6,
    # ActivationSwitch.SOFTPLUS_B_5,
    # ActivationSwitch.SOFTPLUS_B_4,
    # ActivationSwitch.SOFTPLUS_B_3,
    # ActivationSwitch.SOFTPLUS_B_2,
    # ActivationSwitch.SOFTPLUS_B_1,
    # ActivationSwitch.SOFTPLUS_B100,
    # ActivationSwitch.SOFTPLUS_B50,
    # ActivationSwitch.SOFTPLUS_B10,
    # ActivationSwitch.SOFTPLUS_B7,
    # ActivationSwitch.SOFTPLUS_B5,
    # ActivationSwitch.SOFTPLUS_B3,
    # ActivationSwitch.SOFTPLUS_B2,
    # ActivationSwitch.SOFTPLUS_B1,
]
loss = [
    LossSwitch.CE,
    # LossSwitch.MSE
]
add_inverse = [
    # True,
    False,
]
model_name = [
    # ModelSwitch.SIMPLE_CNN_DEPTH
    # ModelSwitch.SIMPLE_CNN,
    # ModelSwitch.SIMPLE_CNN_BN,
    # ModelSwitch.SIMPLE_CNN_SK,
    # ModelSwitch.SIMPLE_CNN_SK_BN,
    # ModelSwitch.RESNET_BASIC,
    # ModelSwitch.RESNET_BOTTLENECK,
    # ModelSwitch.RESNET50,
    ModelSwitch.VIT_16,
    # ModelSwitch.VIT_32,
]
layers = [
    [],
    # [1],
    # [2],
    # [3],
    # [4],
    # [5],
    # [6],
    # [7],
    # [1, 1, 1, 1],
    # [2, 2, 2, 2],
    # [3, 3, 3, 3],
    # [3, 4, 6, 3],
    # [6, 1, 1, 1],
    # [1, 6, 1, 1],
    # [1, 1, 6, 1],
    # [1, 1, 1, 6],
]
dataset = [
    # DatasetSwitch.FASHION_MNIST,
    # DatasetSwitch.CIFAR10,
    DatasetSwitch.IMAGENETTE,
    # DatasetSwitch.IMAGENET,
]
bias = [False]
pre_act = [
    False,
    # True,
]

num_workers = [16]
prefetch_factor = [8]
img_size = [
    # 28,
    # 32,
    # 46,
    # 64,
    # 112,
    224,
]
l2_reg = [
    0,
    # 1e-2,
    # 1e-3,
]
lr = [
    # 5e-3,
    # 3e-3,
    # 1e-3,
    # 5e-4,
    # 4e-4,
    3e-4,
    # 2e-4,
    # 1e-4,
    # 5e-5,
    # 3e-5,
    # 4e-5,
    # 1e-5,
]

# %% submit training
batch_size = [256]
min_test_acc = [0.6]
patience = [1]

ckpt_mod = [1]  # checkpoint if epoch % ckpt_mod == 0
epochs = [100]
lr_decay_gamma = [1 - 1e-4]
warmup_epochs_ratio = 0.0
gaussian_noise_var = [0.0]
gaussian_blur_var = [0.0]
timeout = 60
submit_training(
    seed=seed,
    activation=activation,
    loss=loss,
    add_inverse=add_inverse,
    model_name=model_name,
    layers=layers,
    dataset=dataset,
    bias=bias,
    pre_act=pre_act,
    port=port,
    block_main=block_main,
    timeout=timeout,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    batch_size=batch_size,
    patience=patience,
    lr=lr,
    l2_reg=l2_reg,
    ckpt_mod=ckpt_mod,
    epochs=epochs,
    warmup_epochs_ratio=warmup_epochs_ratio,
    gaussian_noise_var=gaussian_noise_var,
    gaussian_blur_var=gaussian_blur_var,
    img_size=img_size,
    lr_decay_gamma=lr_decay_gamma,
    min_test_acc=min_test_acc,
    force_run=force_run,  # set to True to force run even if checkpoints exist
)

# %% submit grads
gaussian_noise_var = [0.0]
gaussian_blur_var = [0.0]
e_gaussian_noise_var = [
    # 0.0,  # vg
    0.1,  # sg
]
e_gaussian_blur_var = [0.0]
num_batches = [1]
batch_size = [
    # 1, # VG
    64, # SG
]
epoch = [0]
num_distinct_images = [1000]
eval_only_on_test = [True]
stats = [
    {
        "mean_rank": None,
        # "mean": None,
        # "var_rank": None,
        # "var": None,
        "correct": None,
        "image": None,
        "label": None,
        "batch_size": None,
    }
]
timeout = 10
submit_grads(
    timeout=timeout,
    batch_size=batch_size,
    activation=activation,
    dataset=dataset,
    model_name=model_name,
    layers=layers,
    img_size=img_size,
    bias=bias,
    port=port,
    block_main=block_main,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    gaussian_noise_var=gaussian_noise_var,  # training noise
    gaussian_blur_var=gaussian_blur_var,  # training blur
    e_gaussian_noise_var=e_gaussian_noise_var,  # explainer noise
    e_gaussian_blur_var=e_gaussian_blur_var,  # explainer blur
    num_batches=num_batches,
    add_inverse=add_inverse,
    seed=seed,
    eval_only_on_test=eval_only_on_test,
    pre_act=pre_act,
    stats=stats,
    epoch=epoch,
    num_distinct_images=num_distinct_images,
    l2_reg=l2_reg,
    lr=lr,
)

# %% submit explainers
explainer = [
    ExplainerSwitch.GRAD_CAM,
    ExplainerSwitch.DEEP_LIFT,
    ExplainerSwitch.INTEGRATED_GRAD,
    ExplainerSwitch.GUIDED_BPP,
    # ExplainerSwitch.LRP,
]
gaussian_noise_var = [0.0]
gaussian_blur_var = [0.0]
num_batches = [1]
batch_size = [1]
epoch = [0]
num_distinct_images = [1000]
eval_only_on_test = [True]
stats = [
    {
        "mean_rank": None,
        # "mean": None,
        "correct": None,
        "image": None,
        "label": None,
        "batch_size": None,
    }
]
timeout = 10
submit_explainers(
    timeout=timeout,
    explainer=explainer,
    batch_size=batch_size,
    activation=activation,
    dataset=dataset,
    model_name=model_name,
    layers=layers,
    img_size=img_size,
    bias=bias,
    port=port,
    block_main=block_main,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    gaussian_noise_var=gaussian_noise_var,  # training noise
    gaussian_blur_var=gaussian_blur_var,  # training blur
    num_batches=num_batches,
    add_inverse=add_inverse,
    seed=seed,
    eval_only_on_test=eval_only_on_test,
    pre_act=pre_act,
    stats=stats,
    epoch=epoch,
    num_distinct_images=num_distinct_images,
    l2_reg=l2_reg,
    lr=lr,
)

# %% run measurements on grads
hook_samples = [
    [
        2,
        # 100,
        # 101,
        # 201,
        # 202,
        # 302,
        # 303,
        # 403,
        # 404,
        # 504,
        # 505,
        # 605,
        # 606,
        # 706,
        # 707,
        # 807,
        # 808,
        # 908,
        # 909,
    ]
]
name = [
    # "IMAGENETTE",
    # "IMAGENETTE/NONE",
    # "IMAGENETTE/SK",
    # "IMAGENETTE/BN",
    # "IMAGENETTE/SKBN",
    # "IMAGENETTE/32",
    # "IMAGENETTE/46",
    # "IMAGENETTE/64",
    # "IMAGENETTE/112",
    "IMAGENETTE/224",
    # "IMAGENETTE/j0",
    # "IMAGENETTE/j1",
    # "IMAGENETTE/j2",
    # "IMAGENETTE/j3",
    # "CIFAR10",
    # "FASHION_MNIST",
    # "IMAGENET",
]
timeout = 20
submit_measurements(
    name=name,
    timeout=timeout,
    port=0,
    block_main=block_main,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    hook_samples=hook_samples,
)
# %% remove stuff
import os

os.chdir(cwd)

# !mkdir checkpoints/
# !ls checkpoints/
# !rm checkpoints/*
# !rm -r visualizations/*
# !rm -r logs/runs/*
# !rm -r logs/.su*
# !rm -r logs/12*
# !rm -r .tmp/extracted/*
# !rm -r .tmp/outputs/*

# %% only for debugging
from src.models.utils import get_model
from captum.attr import GuidedGradCam

arch = get_model(
    model_name=ModelSwitch.VIT_16,
    input_shape=(3, 224, 224),
    num_classes=1000,
    activation_fn=ActivationSwitch.RELU,
    bias=False,
    pre_act=False,
    layers=[],
    checkpoint_path="checkpoints/IMAGENET/224/VIT_16::::RELU::0::0::0.0::0.0::0.0001.pt",
    device="cpu",
)
explainer = convert_str_to_explainer(
    explainer=ExplainerSwitch.GRAD_CAM, model=arch, model_name=ModelSwitch.VIT_16
)
inputs = torch.randn(1, 3, 224, 224)
explanation = explainer.attribute(inputs, target=0)
print(explanation.shape)
# %%
