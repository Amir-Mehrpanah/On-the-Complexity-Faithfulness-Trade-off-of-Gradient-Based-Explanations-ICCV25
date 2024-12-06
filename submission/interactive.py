# %% CD
# %cd '/proj/azizpour-group/users/x_amime/projects/kernel-view-to-explainability/'

# %% imports

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
    # ActivationSwitch.LEAKY_RELU,
    # ActivationSwitch.SOFTPLUS_B1,
    # ActivationSwitch.SOFTPLUS_B5,
    # ActivationSwitch.SOFTPLUS_B10,
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
    # ModelSwitch.RESNET50,
    # ModelSwitch.SIMPLE_CNN,
]
dataset = DatasetSwitch.IMAGENETTE
bias = "--nobias"
port = "--port 5678"  # --port 5678
block_main = "--block_main"  # "--block_main"
# batch_size = 256
batch_sizes = {
    ActivationSwitch.RELU: 128,
    ActivationSwitch.LEAKY_RELU: 128,
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


# %% submit training

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
        f"--augmentation {augmentation}"
    )
    if output != 0:
        print(f"Error: {activation} {loss}")
        break

# %% submit grads
num_batches = 2
augmentations = [
    AugmentationSwitch.EXP_GEN,
]
epoch = 0
num_distinct_images = 4
gaussian_noise_var = 1e-4
for activation, add_inverse, model_name, augmentation in product(
    activations,
    add_inverses,
    model_names,
    augmentations,
):
    print(f"time: {datetime.now()}")
    augmentation_tb = "aug" if augmentation == "--augmentation" else "noaug"
    batch_size_ = batch_sizes[activation]
    now = datetime.now().strftime("%Y%m%d-%H")
    output = os.system(
        f"python submission/grads.py {block_main}"
        f" {port} --dataset {dataset} {bias}"
        f" --activation {activation} --epoch {epoch}"
        f" --batch_size {batch_size_} --num_batches {num_batches}"
        f" {add_inverse} --num_workers {num_workers} --augmentation {augmentation}"
        f" --prefetch_factor {prefetch_factor} --model_name {model_name}"
        f" --num_distinct_images {num_distinct_images} --eval_only_on_test"
        f" --gaussian_noise_var {gaussian_noise_var} "
    )
    if output != 0:
        print(f"Error: {activation} {model_name}")
        break


# %% remove stuff

# !mkdir checkpoints/    # type: ignore
# !ls checkpoints/    # type: ignore
# !rm checkpoints/*    # type: ignore
!rm -r logs/.su*    # type: ignore
!rm -r logs/12*    # type: ignore
# !rm -r logs/runs/*    # type: ignore


# %%
