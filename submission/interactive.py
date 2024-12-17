# %% imports
import torch
from itertools import product
import os

cwd = "/proj/azizpour-group/users/x_amime/projects/kernel-view-to-explainability/"
os.chdir(cwd)

from submission.utils import submit_training, submit_grads

from datetime import datetime
from src.utils import (
    ActivationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    AugmentationSwitch,
)

seed = [0]
activation = [
    ActivationSwitch.RELU,
    ActivationSwitch.LEAKY_RELU,
    ActivationSwitch.SOFTPLUS_B_1,
    ActivationSwitch.SOFTPLUS_B1,
    ActivationSwitch.SOFTPLUS_B5,
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
    ModelSwitch.SIMPLE_CNN_DEPTH  ## different depths L @1
    # ModelSwitch.SIMPLE_CNN,
    # ModelSwitch.SIMPLE_CNN_BN,
    # ModelSwitch.SIMPLE_CNN_SK,
    # ModelSwitch.SIMPLE_CNN_SK_BN,
    # ModelSwitch.RESNET_BASIC,
    # ModelSwitch.RESNET_BOTTLENECK,
]
layers = [
    [1],
    [2],
    [3],
    [4],
    # [1, 1, 1, 1],
    # [2, 2, 2, 2],
    # [3, 3, 3, 3],
    # [3, 4, 6, 3],
    # [6, 1, 1, 1],
    # [1, 6, 1, 1],
    # [1, 1, 6, 1],
    # [1, 1, 1, 6],
]
dataset = [DatasetSwitch.FASHION_MNIST]
bias = [False]
pre_act = [
    False,
    # True,
]

if 0:  # debug
    port = 5678
    block_main = True
    timeout = 10
else:
    port = None
    block_main = False
    timeout = 20

num_workers = [16]
prefetch_factor = [8]
img_size = [
    28,
    37,
    46,
    55,
    64,
]
l2_reg = [
    1e-2,
    5e-3,
    1e-3,
]
# %% submit training
batch_size = {
    ActivationSwitch.RELU: 256,
    ActivationSwitch.LEAKY_RELU: 256,
    ActivationSwitch.SOFTPLUS_B_1: 256,
    ActivationSwitch.SOFTPLUS_B1: 256,
    ActivationSwitch.SOFTPLUS_B5: 256,
}

patience = [5]
lr = [1e-3]
ckpt_mod = [1]  # checkpoint if epoch % ckpt_mod == 0
epochs = [20]
lr_decay_gamma = [0.98]
warmup_epochs_ratio = 0.6
gaussian_noise_var = [0.01]
augmentation = [
    AugmentationSwitch.TRAIN,
]

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
    augmentation=augmentation,
    img_size=img_size,
    lr_decay_gamma=lr_decay_gamma,
)

# %% submit grads
num_batches = [4]
batch_size = {
    ActivationSwitch.RELU: 32,
    ActivationSwitch.LEAKY_RELU: 32,
    ActivationSwitch.SOFTPLUS_B_1: 32,
    ActivationSwitch.SOFTPLUS_B1: 32,
    ActivationSwitch.SOFTPLUS_B5: 32,
}
augmentation = [
    AugmentationSwitch.EXP_GEN,
]
epoch = [0]
num_distinct_images = [4]
gaussian_noise_var = [1e-5]
eval_only_on_test = [True]
stats = [
    {
        "mean_rank": None,
        "var_rank": None,
        "mean": None,
        "var": None,
        "correct": None,
        "image": None,
        "label": None,
        "batch_size": None,
    }
]

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
    gaussian_noise_var=gaussian_noise_var,
    num_batches=num_batches,
    add_inverse=add_inverse,
    seed=seed,
    augmentation=augmentation,
    eval_only_on_test=eval_only_on_test,
    pre_act=pre_act,
    stats=stats,
    epoch=epoch,
    num_distinct_images=num_distinct_images,
    l2_reg=l2_reg,
)

# %% extract the grad results
os.system("bash src/ext.sh")

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


# %% run measurements on grads
for activation, add_inverse, model_name, augmentation, pre_act, layers in product(
    activations,
    add_inverses,
    model_names,
    augmentations,
    pre_acts,
    layerss,
):
    print(f"time: {datetime.now()}")
    if len(layers) > 0:
        layers_ = " ".join(map(str, layers))
        layers_ = f"--layers {layers_}"
    else:
        layers_ = ""

    batch_size_ = batch_sizes[activation]
    now = datetime.now().strftime("%Y%m%d-%H")
    output = os.system(
        f"python submission/quant_measures_grads.py "
        f" {port} {block_main}"
        f" --num_workers {num_workers} --prefetch_factor {prefetch_factor}"
    )
    if output != 0:
        print(f"Error: {activation} {model_name}")
        break

# %% visualize
import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

keys = ["var", "mean", "var_rank", "mean_rank", "image"]

for j in range(4):
    os.makedirs(f"visualizations/{j}", exist_ok=True)
    glob_path = f"{cwd}.tmp/extracted/*/outputs_{j}.pt"
    for path in glob(glob_path):
        data = torch.load(path)
        corrects = data["correct"]
        batch_size = data["batch_size"]
        print(os.path.basename(glob_path), "corrects", corrects, batch_size)
        prefix = path.split("/")[-2]
        for key in keys:
            if key == "image":
                if os.path.exists(f"visualizations/{j}/{key}.png"):
                    continue
                temp = np.transpose(data[key], (1, 2, 0))
                plt.imshow(temp)
                plt.savefig(f"visualizations/{j}/{key}.png")
            else:
                temp = data[key]
                plt.imshow(temp)
                plt.savefig(f"visualizations/{j}/{key}_{prefix}.png")
            plt.close()

# %% debug
import os

os.chdir(cwd)
from src.models.utils import get_model
import torch.nn as nn
import torch

model = get_model(
    input_shape=(1, 28, 28),
    model_name=ModelSwitch.SIMPLE_CNN_DEPTH,
    num_classes=10,
    activation_fn=nn.ReLU(),
    bias=False,
    pre_act=False,
    layers=[4],
)

x = torch.randn(1, 1, 28, 28)
model(x)

# %%
