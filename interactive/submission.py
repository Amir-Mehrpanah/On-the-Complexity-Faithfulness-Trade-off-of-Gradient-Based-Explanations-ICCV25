# %% imports
import os

cwd = "/proj/azizpour-group/users/x_amime/projects/kernel-view-to-explainability/"
os.chdir(cwd)

from submission.utils import (
    submit_measurements,
    submit_training,
    submit_grads,
    visualize_hooks,
)
from src.utils import (
    ActivationSwitch,
    LossSwitch,
    DatasetSwitch,
    ModelSwitch,
)

seed = [0]
activation = [
    ActivationSwitch.RELU,
    ActivationSwitch.LEAKY_RELU,
    # ActivationSwitch.SOFTPLUS_B_1,
    ActivationSwitch.SOFTPLUS_B1,
    # ActivationSwitch.SOFTPLUS_B3,
    ActivationSwitch.SOFTPLUS_B5,
    # ActivationSwitch.SOFTPLUS_B7,
    ActivationSwitch.SOFTPLUS_B10,
    ActivationSwitch.SOFTPLUS_B50,
    ActivationSwitch.SOFTPLUS_B100,
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
    ModelSwitch.SIMPLE_CNN_DEPTH
    # ModelSwitch.SIMPLE_CNN,
    # ModelSwitch.SIMPLE_CNN_BN,
    # ModelSwitch.SIMPLE_CNN_SK,
    # ModelSwitch.SIMPLE_CNN_SK_BN,
    # ModelSwitch.RESNET_BASIC,
    # ModelSwitch.RESNET_BOTTLENECK,
]
layers = [
    # [],
    # [1],
    # [2],
    # [3],
    # [4],
    [5],
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
]
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
    timeout = 60

num_workers = [16]
prefetch_factor = [8]
img_size = [
    # 28,
    # 46,
    # 32,
    64,
    # 112,
    # 224,
]
l2_reg = [
    0,
    # 1e-2,
    # 1e-3,
]
lr = [
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    4e-5,
    5e-5,
    # 1e-5,
]
# %% submit training
batch_size = [256]
min_test_acc = [0.6]
patience = [1]

ckpt_mod = [1]  # checkpoint if epoch % ckpt_mod == 0
epochs = [100]
lr_decay_gamma = [0.98]
warmup_epochs_ratio = 0.0
gaussian_noise_var = [0.0]

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
    img_size=img_size,
    lr_decay_gamma=lr_decay_gamma,
    min_test_acc=min_test_acc,
)

# %% submit grads
num_batches = [1]
batch_size = [1]
epoch = [0]
num_distinct_images = [1000]
gaussian_noise_var = [
    0.0,
    # 1e-5,
    # 1e-4,
    # 1e-3,
    # 1e-2,
    # 1e-1,
]
eval_only_on_test = [True]
stats = [
    {
        "mean_rank": None,
        # "var_rank": None,
        "mean": None,
        # "var": None,
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
    eval_only_on_test=eval_only_on_test,
    pre_act=pre_act,
    stats=stats,
    epoch=epoch,
    num_distinct_images=num_distinct_images,
    l2_reg=l2_reg,
    lr=lr,
)


# %% run measurements on grads
hook_samples = [[]]
name = ["IMAGENETTE"]
num_workers = [8]
submit_measurements(
    name=name,
    timeout=timeout,
    port=port,
    block_main=block_main,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    hook_samples=hook_samples,
)

# %% visualize
hook_samples = [[]]
keys = ["var", "mean", "var_rank", "mean_rank", "image"]
visualize_hooks(DatasetSwitch.IMAGENETTE, hook_samples, keys)

# %% extract the grad results
# os.system("bash src/ext.sh")

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

# %% debug
import torch
from src.models.utils import get_model

g = 224
model = get_model(
    input_shape=(3, g, g),
    model_name=ModelSwitch.SIMPLE_CNN_DEPTH,
    num_classes=10,
    activation_fn=torch.nn.ReLU(),
    bias=False,
    pre_act=False,
    layers=[7],
)
x = torch.randn(1, 3, g, g)
paramsnorm = torch.norm(torch.cat([p.view(-1) for p in model.parameters()]))
model(x)
