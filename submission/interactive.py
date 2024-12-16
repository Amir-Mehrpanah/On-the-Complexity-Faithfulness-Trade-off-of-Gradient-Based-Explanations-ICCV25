# %% imports
import torch
from itertools import product
import os

cwd = "/proj/azizpour-group/users/x_amime/projects/kernel-view-to-explainability/"
os.chdir(cwd)

from datetime import datetime
from src.utils import (
    ActivationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    AugmentationSwitch,
)

seeds = [0]
activations = [
    ActivationSwitch.RELU,
    ActivationSwitch.LEAKY_RELU,
    ActivationSwitch.SOFTPLUS_B_1,
    ActivationSwitch.SOFTPLUS_B1,
    ActivationSwitch.SOFTPLUS_B5,
]
losses = [
    LossSwitch.CE,
    # LossSwitch.MSE
]
add_inverses = [
    # "--add_inverse",
    "",
]
model_names = [
    ModelSwitch.SIMPLE_CNN_DEPTH  ## different depths L @1
    # ModelSwitch.SIMPLE_CNN,
    # ModelSwitch.SIMPLE_CNN_BN,
    # ModelSwitch.SIMPLE_CNN_SK,
    # ModelSwitch.SIMPLE_CNN_SK_BN,
    # ModelSwitch.RESNET_BASIC,
    # ModelSwitch.RESNET_BOTTLENECK,
]
layerss = [
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
dataset = DatasetSwitch.FASHION_MNIST  # MNIST ## D @4 @5
bias = "--nobias"
pre_acts = [
    "",
    # "--pre_act",
]

if 0:  # debug
    port = "--port 5678"
    block_main = "--block_main"
    timeout = 10
else:
    port = ""
    block_main = ""
    timeout = 20

num_workers = 16
prefetch_factor = 8

# %% submit training
batch_sizes = {
    ActivationSwitch.RELU: 256,
    ActivationSwitch.LEAKY_RELU: 256,
    ActivationSwitch.SOFTPLUS_B_1: 256,
    ActivationSwitch.SOFTPLUS_B1: 256,
    ActivationSwitch.SOFTPLUS_B5: 256,
}

patience = 5
lr = 1e-3
l2_reg = 1e-3
ckpt_mod = 1  # checkpoint if epoch % ckpt_mod == 0
epochs = 20
warmup_epochs = epochs // 4
gaussian_noise_var = 0.01
augmentations = [
    AugmentationSwitch.TRAIN,
]
for (
    activation,
    loss,
    add_inverse,
    model_name,
    augmentation,
    pre_act,
    layers,
    seed,
) in product(
    activations,
    losses,
    add_inverses,
    model_names,
    augmentations,
    pre_acts,
    layerss,
    seeds,
):
    torch.manual_seed(seed)
    print(f"time: {datetime.now()}")
    if len(layers) > 0:
        layers_ = " ".join(map(str, layers))
        layers_ = f"--layers {layers_}"
        layers_tb = "_".join(map(str, layers))
    else:
        layers_ = ""
        layers_tb = ""
    activation_tb = activation.name
    dataset_tb = dataset.name
    now = datetime.now().strftime("%Y%m%d-%H")
    batch_size_ = batch_sizes[activation]
    output = os.system(
        f"python submission/training.py {block_main}"
        f" {port} --dataset {dataset} --ckpt_mod {ckpt_mod}"
        f" {bias} --activation {activation} --loss {loss}"
        f" --lr {lr} --epochs {epochs} --batch_size {batch_size_}"
        f" {add_inverse} --num_workers {num_workers} {layers_}"
        f" --prefetch_factor {prefetch_factor} {pre_act} --seed {seed}"
        f" --tb_postfix {now}_{model_name}_{layers_tb}_{activation_tb}_{dataset_tb}"
        f" --patience {patience} --model_name {model_name}"
        f" --augmentation {augmentation} --gaussian_noise_var {gaussian_noise_var}"
        f" --l2_reg {l2_reg} --timeout {timeout} --warmup_epochs {warmup_epochs}"
    )
    if output != 0:
        print(f"Error: {activation} {loss}")
        break

# %% submit grads
num_batches = 4
batch_sizes = {
    ActivationSwitch.RELU: 32,
    ActivationSwitch.LEAKY_RELU: 32,
    ActivationSwitch.SOFTPLUS_B_1: 32,
    ActivationSwitch.SOFTPLUS_B1: 32,
    ActivationSwitch.SOFTPLUS_B5: 32,
}
augmentations = [
    AugmentationSwitch.EXP_GEN,
]
epoch = 0
num_distinct_images = 25
gaussian_noise_var = 1e-5
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
        f"python submission/grads.py {block_main}"
        f" {port} --dataset {dataset} {bias} {pre_act}"
        f" --activation {activation} --epoch {epoch} {layers_}"
        f" --batch_size {batch_size_} --num_batches {num_batches}"
        f" {add_inverse} --num_workers {num_workers} --augmentation {augmentation}"
        f" --prefetch_factor {prefetch_factor} --model_name {model_name}"
        f" --num_distinct_images {num_distinct_images} --eval_only_on_test"
        f" --gaussian_noise_var {gaussian_noise_var} "
    )
    if output != 0:
        print(f"Error: {activation} {model_name}")
        break

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
