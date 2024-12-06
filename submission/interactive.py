#%% imports 
from itertools import product
import os
from glob import glob
from datetime import datetime
from src.utils import (
    ActivationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
    AugmentationSwitch,
)

activations = [
    ActivationSwitch.RELU,
    ActivationSwitch.LEAKY_RELU,
    ActivationSwitch.SOFTPLUS_B1,
    ActivationSwitch.SOFTPLUS_B5,
    ActivationSwitch.SOFTPLUS_B10,
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
    ModelSwitch.RESNET34,
    ModelSwitch.RESNET50,
    ModelSwitch.SIMPLE_CNN,
]
dataset = DatasetSwitch.IMAGENETTE
bias = "--nobias"
port = ""  # "--port 5678"
block_main = ""  # "--block_main"
# batch_size = 256
batch_sizes = {
    ActivationSwitch.RELU: 256,
    ActivationSwitch.LEAKY_RELU: 256,
    ActivationSwitch.SOFTPLUS_B_1: 128,
    ActivationSwitch.SOFTPLUS_B1: 128,
    ActivationSwitch.SOFTPLUS_B5: 128,
    ActivationSwitch.SOFTPLUS_B10: 128,
}
lr = 1e-3
epochs = 100
num_workers = 16
prefetch_factor = 16
patience = 5
ckpt_mod = 5  # checkpoint if epoch % ckpt_mod == 0


#%% submit training

augmentations = [
    AugmentationSwitch.TRAIN,
]
for activation, loss, add_inverse, model_name, augmentation in product(
    activations,
    losses,
    add_inverses,
    model_names,
    augmentations,
):
    print(f"time: {datetime.now()}")
    add_inverse_ture = add_inverse == "--add_inverse"
    now = datetime.now().strftime("%Y%m%d-%H")
    augmentation_tb = "aug" if augmentation == "--augmentation" else "noaug"
    batch_size_ = batch_sizes[activation]
    output = os.system(
        f"python submission/training.py {block_main}"
        f" {port} --dataset {dataset} --ckpt_mod {ckpt_mod}"
        f" {bias} --activation {activation} --loss {loss}"
        f" --lr {lr} --epochs {epochs} --batch_size {batch_size_}"
        f" {add_inverse} --num_workers {num_workers}"
        f" --prefetch_factor {prefetch_factor}"
        f" --tb_postfix {now}_{model_name}_{activation}_{augmentation_tb}"
        f" --patience {patience} --model_name {model_name}"
        f" {augmentation}"
    )
    if output != 0:
        print(f"Error: {activation} {loss}")
        break

#%% submit grads 

augmentations = [
    AugmentationSwitch.EXP_GEN,
]
checkpoint_paths = glob("checkpoints/*.pth")
exp_batch_size = 1
exp_num_samples = 128
for checkpoint_path in checkpoint_paths:
    print(f"time: {datetime.now()}")
    now = datetime.now().strftime("%Y%m%d-%H")
    output = os.system(
        f"python submission/curvature_summary.py {block_main}"
        f" {port} --dataset {dataset} --ckpt_mod {ckpt_mod}"
        f" {bias} --activation {activation} --loss {loss}"
        f" --lr {lr} --epochs {epochs} --batch_size {batch_size_}"
        f" {add_inverse} --num_workers {num_workers}"
        f" --prefetch_factor {prefetch_factor}"
        f" --tb_postfix {now}_{model_name}_{activation}_{augmentation_tb}"
        f" --patience {patience} --model_name {model_name}"
        f" {augmentation}"
    )
    if output != 0:
        print(f"Error: {activation} {loss}")
        break


# %% remove stuff

# !mkdir checkpoints/
# !ls checkpoints/
# !rm checkpoints/*
# !rm -r ../logs/.su*
# !rm -r logs/12*
# !rm -r logs/runs/*


# %%
