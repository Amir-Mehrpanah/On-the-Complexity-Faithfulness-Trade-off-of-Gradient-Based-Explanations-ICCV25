# %% imports read quants
import itertools
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.ticker as mticker
from scipy import stats
import torch
import matplotlib.pyplot as plt
import os
os.chdir("/home/x_amime/x_amime/projects/kernel-view-to-explainability/")

from src import paths as local_pathlib
from src.datasets import get_imagenette_dataset
from src.utils import AugmentationSwitch, DatasetSwitch


output_dir = ".tmp/visualizations/paper/"
input_size = 112
dataset = [
    # f"*/IMAGENETTE::_*",
    # f"IMAGENETTE/{input_size}::_*",
    # f"IMAGENETTE/*::_*",
    # f"IMAGENETTE::_*",
    # f"IMAGENET::_*",
    # f"CIFAR10::_*",
    f"FASHION_MNIST::_*",
][0]
quants_path = f".tmp/quants/{dataset}quants.pt"
paths = glob(quants_path)
print("found quants paths", paths)
quants = []
for path in tqdm(paths):
    quants.extend(torch.load(path))

quants = pd.DataFrame(quants)

quants["noise_scale"] = quants["noise_scale"].astype(float)
quants["index"] = quants["index"].astype(int)
temp = quants["address"].apply(lambda x: x[0].split("/")[-2].split("::"))
print("columns:", temp.iloc[0])
temp = pd.DataFrame(
    temp.tolist(),
    columns=[
        "model_name",
        "layers",
        "activation",
        "seed",
        "l2reg",
        # "tgnv",
        # "tgbv",
        # "input_size",
        "lr",
        # "explainer_name",
        "egnv",
        # "egbv",
    ],
)
# temp.drop(columns=["egnv"], inplace=True)
# temp.drop(columns=["egbv"], inplace=True)
temp["input_size"] = input_size
temp["lr"] = temp["lr"].astype(float)
if "explainer_name" in temp.columns:
    temp.explainer_name.fillna("VG", inplace=True)
quants = pd.concat([quants, temp], axis=1)

quants = quants.set_index(
    [
        "layers",
        "activation",
        "seed",
        "l2reg",
        "input_size",
        "noise_scale",
        # "tgnv",
        "model_name",
        "lr",
        "index",
    ]
)
quants = quants.sort_index()
quants.value_counts("activation")
# %% constants
faint_grids_alpha = 0.5
activation_betas = {
    "RELU": np.inf,
    "LEAKY_RELU": np.inf,
    "SOFTPLUS_B_1": 0.1,
    "SOFTPLUS_B_2": 0.2,
    "SOFTPLUS_B_3": 0.3,
    "SOFTPLUS_B_4": 0.4,
    "SOFTPLUS_B_5": 0.5,
    "SOFTPLUS_B_6": 0.6,
    "SOFTPLUS_B_7": 0.7,
    "SOFTPLUS_B_8": 0.8,
    "SOFTPLUS_B_9": 0.9,
    "SOFTPLUS_B1": 1,
    "SOFTPLUS_B2": 2,
    "SOFTPLUS_B3": 3,
    "SOFTPLUS_B4": 4,
    "SOFTPLUS_B5": 5,
    "SOFTPLUS_B7": 7,
    "SOFTPLUS_B10": 10,
    "SOFTPLUS_B50": 50,
    "SOFTPLUS_B100": 100,
}
activation_nice_names = {
    "RELU": "ReLU",
    "TANH": "Tanh",
    "SIGMOID": "Sigmoid",
    "LEAKY_RELU": "Leaky ReLU",
}
for activation, beta in activation_betas.items():
    if beta == np.inf:
        activation_nice_names[activation] = "ReLU"
    else:
        activation_nice_names[activation] = f"SP(β={beta})"

activations = [
    # "LEAKY_RELU",
    # "SIGMOID",
    # "TANH",
    "SOFTPLUS_B_1",
    # "SOFTPLUS_B_2",
    "SOFTPLUS_B_3",
    # "SOFTPLUS_B_4",
    "SOFTPLUS_B_5",
    # "SOFTPLUS_B_6",
    "SOFTPLUS_B_7",
    # "SOFTPLUS_B_8",
    # "SOFTPLUS_B_9",
    "SOFTPLUS_B1",
    "SOFTPLUS_B2",
    # "SOFTPLUS_B3",
    # "SOFTPLUS_B4",
    "SOFTPLUS_B5",
    # "SOFTPLUS_B7",
    # "SOFTPLUS_B10",
    # "SOFTPLUS_B50",
    # "SOFTPLUS_B100",
    "RELU",
]
markers = ["o", "s", "D", "v", "^", "<", ">", "p", "P", "*", "X", "d", "H", "h"]
# colors = plt.cm.tab20(np.arange(len(activations)))
colors = plt.cm.get_cmap("turbo_r")(np.linspace(0.1, 0.9, len(activations)))
activation_colors = {
    activation: color for activation, color in zip(activations, colors)
}

models = [
    "SIMPLE_CNN",
    "SIMPLE_CNN_SK",
    "SIMPLE_CNN_BN",
    "SIMPLE_CNN_SK_BN",
    # "RESNET_BASIC",
    # "RESNET_BOTTLENECK",
]
colors = plt.cm.tab10(np.arange(len(models)))
model_colors = {model: color for model, color in zip(models, colors)}
model_nice_names = {
    "SIMPLE_CNN": "None",
    "SIMPLE_CNN_BN": "BN",
    "SIMPLE_CNN_SK": "SK",
    "SIMPLE_CNN_SK_BN": "SK+BN",
    "RESNET_BASIC": "ResNet Basic",
    "RESNET_BOTTLENECK": "ResNet Bottleneck",
}
explainer_nice_names = {
    "DEEP_LIFT": "DeepLIFT",
    "GRAD_CAM": "GradCAM",
    "GUIDED_BPP": "GuidedBP",
    "INTEGRATED_GRAD": "IntGrad",
    "SG": "SmoothGrad",
    "LRP": "LRP",
    "VG": "VanillaGrad",
    "0.0": "VanillaGrad",
    "0.1": "SmoothGrad",
}


# %% defs
def density_depth_inputsize(
    quants, spec_type="mr_spectral_density", for_noise_scale=1e-05
):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale", "lr"],
        values=spec_type,
        aggfunc=np.mean,
    )
    factor = 2.5
    n_rows = len(temp.columns.levels[1])
    n_cols = len(temp.columns.levels[0])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.1, n_rows * factor),
        tight_layout=True,
    )
    spec = "mean rank" if spec_type == "mr_spectral_density" else "var rank"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for nid, noise_scale in enumerate(temp.columns.levels[3]):
            for lid, layers in enumerate(temp.columns.levels[1]):
                # skip keys
                if noise_scale != for_noise_scale:
                    continue

                if len(temp.columns.levels[0]) == 1:
                    ax = axes[lid]
                else:
                    ax = axes[lid, sid]

                if lid == 0:
                    ax.set_title(f"density at size {input_size}")
                if sid == 0:
                    # ax.set_ylabel("spectral density")
                    ax.set_ylabel(f"depth {layers}")
                    # ax.text(
                    #     -0.3,
                    #     0.5,
                    #     f"image size {input_size}",
                    #     va="center",
                    #     ha="center",
                    #     rotation=90,
                    #     transform=ax.transAxes,
                    #     fontsize=12,
                    # )
                if lid == n_rows - 1:
                    ax.set_xlabel("frequency")

                ax.set_xscale("log")
                ax.set_yscale("log")
                for color_indx, activation in enumerate(activations):
                    for rid, lr in enumerate(temp.columns.levels[4]):
                        try:
                            ax.plot(
                                temp[input_size][layers][activation][noise_scale][
                                    lr
                                ].values[0],
                                color=activation_colors[activation],
                                # alpha=(rid + 1) / len(temp.columns.levels[4]),
                            )
                            vline = (
                                len(
                                    temp[input_size][layers][activation][noise_scale][
                                        lr
                                    ].values[0]
                                )
                                * 0.5
                            )
                            print(
                                "addr",
                                input_size,
                                layers,
                                activation,
                                noise_scale,
                                rid,
                                markers[rid],
                                lr,
                                vline,
                            )
                            ax.vlines(
                                x=vline,
                                ymin=0,
                                ymax=1,
                                color="r",
                                linestyle="--",
                                alpha=0.5,
                            )
                        except KeyError:
                            print(
                                f"missing {input_size} {layers} {activation} {noise_scale}"
                            )
                            pass

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [],
                            label=activation_nice_names[activation],
                            color=activation_colors[activation],
                        )
                        ax.legend()


def density_depth_l2reg(quants, spec_type="mr_spectral_density", for_noise_scale=1e-05):
    temp = quants.pivot_table(
        columns=["input_size", "l2reg", "activation", "noise_scale"],
        values=spec_type,
        aggfunc=np.mean,
    )
    factor = 3
    n_rows = len(temp.columns.levels[1])
    n_cols = len(temp.columns.levels[0])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.1, n_rows * factor),
        tight_layout=True,
    )
    spec = "mean rank" if spec_type == "mr_spectral_density" else "var rank"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for nid, noise_scale in enumerate(temp.columns.levels[3]):
            for lid, layers in enumerate(temp.columns.levels[1]):
                # skip keys
                if noise_scale != for_noise_scale:
                    continue

                if len(temp.columns.levels[0]) == 1:
                    ax = axes[lid]
                else:
                    ax = axes[lid, sid]

                if lid == 0:
                    ax.set_title(f"density at size {input_size}")
                if sid == 0:
                    # ax.set_ylabel("spectral density")
                    ax.set_ylabel(f"l2 {layers}")
                    # ax.text(
                    #     -0.3,
                    #     0.5,
                    #     f"image size {input_size}",
                    #     va="center",
                    #     ha="center",
                    #     rotation=90,
                    #     transform=ax.transAxes,
                    #     fontsize=12,
                    # )
                if lid == n_rows - 1:
                    ax.set_xlabel("frequency")

                ax.set_xscale("log")
                ax.set_yscale("log")
                for color_indx, activation in enumerate(activations):
                    try:
                        ax.plot(
                            temp[input_size][layers][activation][noise_scale].values[0],
                            color=activation_colors[activation],
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {activation} {noise_scale}"
                        )
                        pass

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [],
                            label=activation_nice_names[activation],
                            color=activation_colors[activation],
                        )
                        ax.legend()


def expected_freq_depth(quants, spec_type="mr_expected_spectral_density"):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "noise_scale", "activation"],
        values=spec_type,
        aggfunc=np.mean,
    )
    temp_var = quants.pivot_table(
        columns=["input_size", "layers", "noise_scale", "activation"],
        values=spec_type,
        aggfunc=np.var,
    )
    factor = 2.5
    width = 0.1
    min_val = np.inf
    max_val = -np.inf
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for noise_scale in temp.columns.levels[2]:
            fig, ax = plt.subplots(
                figsize=(len(temp.columns.levels[1]) * factor * 0.6, factor),
                tight_layout=True,
            )
            ax.set_xticks(range(len(temp.columns.levels[1])))
            ax.set_xticklabels(f"depth {x}" for x in temp.columns.levels[1])
            ax.set_title(f"input size {input_size} and noise scale {noise_scale}")
            ax.set_ylabel("expected frequency")
            for lid, layers in enumerate(temp.columns.levels[1]):
                for color_indx, activation in enumerate(activations):
                    try:

                        if (
                            min_val
                            > temp[input_size][layers][noise_scale][activation].values[
                                0
                            ]
                        ):
                            min_val = temp[input_size][layers][noise_scale][
                                activation
                            ].values[0]

                        if (
                            max_val
                            < temp[input_size][layers][noise_scale][activation].values[
                                0
                            ]
                        ):
                            max_val = temp[input_size][layers][noise_scale][
                                activation
                            ].values[0]

                        ax.bar(
                            x=lid + (color_indx - 1) * width,
                            height=temp[input_size][layers][noise_scale][
                                activation
                            ].values[0],
                            yerr=temp_var[input_size][layers][noise_scale][
                                activation
                            ].values[0],
                            color=activation_colors[activation],
                            width=width,
                        )
                        print(
                            input_size,
                            layers,
                            noise_scale,
                            activation,
                            "var",
                            temp_var[input_size][layers][noise_scale][
                                activation
                            ].values[0],
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {noise_scale} {activation}"
                        )
                        pass

            for activation in temp.columns.levels[3]:
                ax.bar(
                    height=0,
                    x=1,
                    label=activation_nice_names[activation],
                    color=activation_colors[activation],
                )
            ax.set_ylim(min_val * 0.99, max_val * 1.01)
            ax.legend(loc="upper center")
            plt.savefig(f"{output_dir}{dataset}_ef_{input_size}_{noise_scale}.pdf")
            plt.close()


def cosine_similarity(
    quants,
    spec_type="cosine_similarity",
):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale"],
        values=spec_type,
        aggfunc=np.mean,
    )
    factor = 2.5
    n_rows = len(temp.columns.levels[1])
    n_cols = len(temp.columns.levels[0])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.5, n_rows * factor),
        tight_layout=True,
    )
    spec = "cosine similarity"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for lid, layers in enumerate(temp.columns.levels[1]):

            ax = axes[lid, sid]

            if lid == 0:
                ax.set_title(f"density at size {input_size}")
            if sid == 0:
                # ax.set_ylabel("spectral density")
                ax.set_ylabel(f"depth {layers}")
                # ax.text(
                #     -0.3,
                #     0.5,
                #     f"image size {input_size}",
                #     va="center",
                #     ha="center",
                #     rotation=90,
                #     transform=ax.transAxes,
                #     fontsize=12,
                # )
            if lid == n_rows - 1:
                ax.set_xlabel("frequency")

            ax.set_xscale("log")
            ax.set_yscale("log")
            for color_indx, activation in enumerate(activations):
                try:
                    ax.plot(
                        temp.columns.levels[3],
                        temp[input_size][layers][activation].values[0],
                        color=activation_colors[activation],
                    )
                except KeyError:
                    print(f"missing {input_size} {layers} {activation}")
                    pass

                if lid == 0 and sid == 0:
                    ax.plot(
                        [],
                        label=activation_nice_names[activation],
                        color=activation_colors[activation],
                    )
                    ax.legend()


def density_depth_noisescale(
    quants, spec_type="mr_spectral_density", for_input_size=28
):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale", "lr"],
        values=spec_type,
        aggfunc=np.mean,
    )
    factor = 2.5
    n_rows = len(temp.columns.levels[1])
    n_cols = len(temp.columns.levels[3])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.1, n_rows * factor),
        tight_layout=True,
    )
    spec = "mean rank" if spec_type == "mr_spectral_density" else "var rank"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for nid, noise_scale in enumerate(temp.columns.levels[3]):
            for lid, layers in enumerate(temp.columns.levels[1]):
                # skip keys

                if input_size != for_input_size:
                    continue

                if len(temp.columns.levels[3]) == 1:
                    if len(temp.columns.levels[1]) == 1:
                        ax = axes
                    else:
                        ax = axes[lid]
                else:
                    ax = axes[lid, nid]

                if lid == 0:
                    ax.set_title(f"density at noise scale {noise_scale}")
                if nid == 0:
                    # ax.set_ylabel("spectral density")
                    ax.set_ylabel(f"depth {layers}")
                    # ax.text(
                    #     -0.3,
                    #     0.5,
                    #     f"image size {input_size}",
                    #     va="center",
                    #     ha="center",
                    #     rotation=90,
                    #     transform=ax.transAxes,
                    #     fontsize=12,
                    # )
                if lid == n_rows - 1:
                    ax.set_xlabel("frequency")

                ax.set_xscale("log")
                ax.set_yscale("log")
                for color_indx, activation in enumerate(activations):
                    for rid, lr in enumerate(temp.columns.levels[4]):
                        try:
                            ax.plot(
                                temp[input_size][layers][activation][noise_scale][
                                    lr
                                ].values[0],
                                color=activation_colors[activation],
                                # marker=markers[rid],
                                # markersize=0.1,
                                # alpha=0.5,
                            )
                            print(
                                "addr",
                                input_size,
                                layers,
                                activation,
                                noise_scale,
                                rid,
                                markers[rid],
                                lr,
                            )
                        except KeyError:
                            print(
                                f"missing {input_size} {layers} {activation} {noise_scale}"
                            )
                            pass

                    if lid == 0 and nid == 0:
                        ax.plot(
                            [],
                            label=activation_nice_names[activation],
                            color=activation_colors[activation],
                        )
                        ax.legend()


# %% plot expected_freq_depth
expected_freq_depth(quants)

# %% plot density_depth_inputsize
for ns in quants.index.levels[5]:
    for spec_type in [
        # "m_expected_spectral_density",
        "mr_spectral_density",
        # "v_expected_spectral_density",
        # "vr_spectral_density",
    ]:
        density_depth_inputsize(quants, spec_type=spec_type, for_noise_scale=ns)
        plt.savefig(f"{output_dir}{dataset}_{spec_type}_{ns}.pdf")
        plt.close()

# %% plot density_depth_noisescale
for isz in quants.index.levels[4]:
    for spec_type in [
        # "m_expected_spectral_density",
        "mr_spectral_density",
        # "v_expected_spectral_density",
        # "vr_spectral_density",
    ]:
        density_depth_noisescale(quants, spec_type=spec_type, for_input_size=isz)
        plt.savefig(f"{output_dir}{dataset}_{spec_type}_{isz}.pdf")
        plt.close()


# %% plot cosine similarity

dataset = quants.dataset[0]
for spec_type in ["cosine_similarity"]:
    cosine_similarity(quants, spec_type=spec_type)

    plt.savefig(f"{output_dir}{dataset}_{spec_type}.pdf")
    plt.close()

# %% density_depth_l2reg
for ns in quants.index.levels[5]:
    for spec_type in ["mr_spectral_density", "vr_spectral_density"]:
        density_depth_l2reg(quants, spec_type=spec_type, for_noise_scale=ns)
        plt.savefig(f"{output_dir}{dataset}_l2reg_{spec_type}_{ns}.pdf")
        plt.close()


# %% density_inputsize_model
def density_inputsize_model(quants, spec_type="mr_spectral_density"):
    temp = quants.pivot_table(
        columns=["input_size", "noise_scale", "model_name"],
        values=spec_type,
        aggfunc="mean",
    )
    factor = 3
    n_rows = len(temp.columns.levels[0])
    n_cols = len(temp.columns.levels[1])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        # sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.1, n_rows * factor),
        tight_layout=True,
    )
    spec = "mean rank" if spec_type == "mr_spectral_density" else "var rank"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for nid, noise_scale in enumerate(temp.columns.levels[1]):
            for lid, model_name in enumerate(temp.columns.levels[2]):

                if len(temp.columns.levels[0]) == 1:
                    ax = axes[nid]
                else:
                    ax = axes[sid, nid]

                if nid == 0:
                    ax.set_ylabel(f"input size {input_size}")
                if sid == 0:
                    ax.set_title(f"density at noise scale {noise_scale}")
                try:
                    ax.plot(
                        temp[input_size][noise_scale][model_name].values[0][1:],
                        color=model_colors[model_name],
                    )
                except KeyError:
                    print(f"missing {input_size} {noise_scale} {model_name}")
                    pass

                ax.set_xscale("log")
                ax.set_yscale("log")
                if sid == n_rows - 1:
                    ax.set_xlabel("frequency")

                if nid == 0 and sid == 0:
                    ax.plot(
                        [],
                        label=model_name,
                        color=model_colors[model_name],
                    )
                    ax.legend()


for spec_type in [
    "mr_spectral_density",
    "vr_spectral_density",
    "m_spectral_density",
    "v_spectral_density",
]:
    density_inputsize_model(quants, spec_type=spec_type)
    # plt.show()
    plt.savefig(f"{output_dir}{dataset}_{spec_type}_model.pdf")
    plt.close()

# %% expected_freq_model


def expected_freq_model(quants, spec_type="mr_expected_spectral_density"):
    temp = quants.pivot_table(
        columns=["input_size", "noise_scale", "model_name"],
        values=spec_type,
        aggfunc=np.mean,
    )

    factor = 4
    width = 0.1
    for sid, input_size in enumerate(temp.columns.levels[0]):
        fig, ax = plt.subplots(
            figsize=(len(temp.columns.levels[1]) * factor * 0.6, factor),
            tight_layout=True,
        )
        ax.set_xticks(range(len(temp.columns.levels[1])))
        ax.set_xticklabels(f"noise scale {x}" for x in temp.columns.levels[1])
        ax.set_title(f"{spec_type} for input size {input_size}")
        ax.set_ylabel("expected frequency")
        for nid, noise_scale in enumerate(temp.columns.levels[1]):
            for lid, model_name in enumerate(temp.columns.levels[2]):
                try:
                    ax.bar(
                        x=nid + (lid - 2) * width,
                        height=temp[input_size][noise_scale][model_name].values[0],
                        color=model_colors[model_name],
                        width=width,
                    )
                except KeyError:
                    print(f"missing {input_size} {noise_scale} {model_name}")
                    pass

        for model in temp.columns.levels[2]:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.bar(
                height=0,
                x=1,
                label=model_nice_names[model],
                color=model_colors[model],
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.legend(loc="upper center")
        plt.show()
        # plt.savefig(f"{output_dir}{dataset}_ef_{input_size}_{noise_scale}.pdf")
        plt.close()


for spec_type in [
    "mr_expected_spectral_density",
    "vr_expected_spectral_density",
    "m_expected_spectral_density",
    "v_expected_spectral_density",
]:
    expected_freq_model(quants, spec_type=spec_type)


# %% plot tails input size


def plot_activations(quants, spec_type, for_lr=None):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale", "lr"],
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale", "lr"],
        values=spec_type,
        aggfunc="mean",
    )
    factor = 3.5 # 2.5 for banner and 3.5 for main experiments
    n_rows = len(temp.columns.levels[1])
    n_cols = len(temp.columns.levels[0])
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex=True,
        # sharey=True,
        figsize=(n_cols * factor * 1.1, n_rows * factor),
        tight_layout=True,
    )
    spec = "mean rank" if spec_type == "mr_spectral_density" else "var rank"
    # fig.suptitle(f"{spec} at {for_noise_scale}", fontsize=16)
    for sid, input_size in enumerate(temp.columns.levels[0]):
        for nid, noise_scale in enumerate(temp.columns.levels[3]):
            for lid, layers in enumerate(temp.columns.levels[1]):
                if (
                    len(temp.columns.levels[0]) == 1
                    and len(temp.columns.levels[1]) == 1
                ):
                    ax = axes

                elif len(temp.columns.levels[0]) == 1:
                    ax = axes[lid]
                else:
                    ax = axes[lid, sid]

                if lid == 0:
                    pass
                    # ax.set_title(f"density at size {input_size}")
                if sid == 0:
                    ax.set_ylabel("spectral density")
                    # ax.set_ylabel(f"depth {layers}")
                    # ax.text(
                    #     -0.3,
                    #     0.5,
                    #     f"image size {input_size}",
                    #     va="center",
                    #     ha="center",
                    #     rotation=90,
                num_lrs = len(temp.columns.levels[4])
                for color_indx, activation in enumerate(activations):
                    for rid, lr in enumerate(temp.columns.levels[4]):
                        if for_lr is not None and lr != for_lr:
                            continue
                        alpha = color_indx / len(activations)
                        try:
                            if spec_type == "mr_spectral_density":
                                # max_val = (
                                #     temp[input_size][layers][activation][noise_scale][
                                #         lr
                                #     ]
                                #     .values[0]
                                #     .max()
                                # )
                                max_val = 1
                            else:
                                max_val = 1
                            ax.plot(
                                temp[input_size][layers][activation][noise_scale][
                                    lr
                                ].values[0]
                                / max_val,
                                color=activation_colors[activation],
                                # alpha=alpha,
                                # marker=markers[rid],
                                # markersize=0.3,
                            )
                            print(
                                "addr",
                                input_size,
                                layers,
                                activation,
                                noise_scale,
                                rid,
                                "alpha",
                                (rid + 1) / num_lrs,
                                markers[rid],
                                lr,
                            )
                        except KeyError:
                            print(
                                f"missing {input_size} {layers} {activation} {noise_scale}"
                            )
                            pass

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [],
                            label=activation_nice_names[activation],
                            color=activation_colors[activation],
                        )
                        ax.legend(loc="upper right")

                if lid == n_rows - 1:
                    ax.set_xlabel("frequency")

                # ax.set_xscale("log")
                ax.set_yscale("log")


# for spec_type in [
#     "mr_spectral_density",
#     # "vr_spectral_density",
#     # "m_spectral_density",
#     # "v_spectral_density",
# ]:
#     plot_activations(quants, spec_type=spec_type)
#     plt.savefig(f"{output_dir}{dataset}_{spec_type}_all.pdf")
#     plt.close()

# for lr in quants.index.levels[8]:
#     for spec_type in [
#         "mr_spectral_density",
#         # "vr_spectral_density",
#         # "m_spectral_density",
#         # "v_spectral_density",
#     ]:
#         plot_activations(quants, spec_type=spec_type, for_lr=lr)
#         plt.savefig(f"{output_dir}{dataset}_{spec_type}_{lr}.pdf")
#         plt.close()

# import matplotlib as mpl

# lr = 0.0001  # IMAEGNETTE_224
# r = 0.0005
# lr = 0.0005 # IMAEGNETTE_112
# r = 0.005
# lr = 0.0003 # IMAEGNETTE_64
# r = 0.005
# lr = 0.0005 # IMAEGNETTE_46
# r = 0.005
# lr = 0.003  # CIFAR10
# r = 0.003
lr = 0.0001  # FASHION_MNIST
r = 0.05
for spec_type in [
    "mr_spectral_density",
    # "vr_spectral_density",
    # "m_spectral_density",
    # "v_spectral_density",
]:
    plot_activations(quants, spec_type=spec_type, for_lr=lr)
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    scale = r if spec_type == "mr_spectral_density" else 0.5
    y_lim = (y_lim[0], y_lim[1] * scale)
    print("ylim", y_lim)

    # enable gridlines
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=faint_grids_alpha)

    plt.ylim(y_lim)
    plt.savefig(f"{output_dir}{dataset}_{spec_type}_{lr}.pdf", bbox_inches="tight")
    plt.close()

# only for banner remove xticks and yticks
    # plt.gca().set_ylabel("(log) spectral density")
    # set x axis and y axis off
    # plt.gca().set_xticks([])
    # plt.gca().set_yticks([])
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    # get objects in the plot
    # artists = plt.gca().get_children()
    # remove objects except the legeneds
    # for i,artist in enumerate(artists):
    #     print(i,artist)
    #     if i == 13:
    #         for j,art in enumerate(artist.get_children()):
    #             print(i,j,art)
    #             if isinstance(art, mpl.axis.YTick):
    #                 art.set_visible(False)

# %% plot expected_freq
def plot_ef(quants, spec_type):
    temp = quants.pivot_table(
        columns=["input_size", "seed", "layers", "noise_scale", "lr"],
        index="activation",
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["input_size", "seed", "layers", "noise_scale", "lr"],
        values=spec_type,
        index="activation",
        aggfunc="mean",
    )
    temp = temp.map(
        lambda x: np.mean((x / x.sum()) * np.arange(len(x))), na_action="ignore"
    )
    mean = temp.apply(lambda x: np.mean(x), axis=1)
    std = temp.apply(lambda x: np.std(x), axis=1)
    return mean, std, temp


spec_type = "mr_spectral_density"
temp_mean, temp_std, temp = plot_ef(quants, spec_type=spec_type)
# sort rows of temp according to activation functions
# temp_mean = temp_mean.loc[activations]
# temp_std = temp_std.loc[activations]

# temp_mean = temp_mean.squeeze()
# temp_std = temp_std.squeeze()

# replace index with nice names
if "FASHION_MNIST" in dataset:
    print("dropping SOFTPLUS_B50 for FASHION_MNIST")
    temp = temp.drop(index="SOFTPLUS_B50")  # ONLY for FASHION_MNIST
x_ticks = np.array([activation_betas[x] for x in temp.index])
finite_max = 2 * np.max(x_ticks[x_ticks != np.inf])
x_ticks = x_ticks.clip(0, finite_max)
temp["x"] = x_ticks
temp = temp.set_index("x")
temp = temp.sort_index()
temp.columns = temp.columns.droplevel([0, 2, 3, 4])
temp = temp.stack()
temp = temp.reset_index().drop(columns="seed")
x = np.log(temp["x"])
y = temp[0]

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
reg_line = slope * x + intercept

# # Compute 95% confidence interval for the regression line
n = len(x)
mean_x = np.mean(x)
t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
s_err = np.sqrt(np.sum((y - reg_line) ** 2) / (n - 2))
ci = t_val * s_err * np.sqrt(1 / n + (x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))
upper = reg_line + ci
lower = reg_line - ci
# # Plot the data, regression line, and confidence interval
plt.figure(figsize=(3, 2))
color = "tab:blue"
plt.scatter(x, y, alpha=0.3, color=color, s=5)
plt.fill_between(x, lower, upper, alpha=0.3)
plt.plot(x, reg_line)
plt.ylabel("Expected Frequency")
plt.xlabel(r"$\beta$")

x_ticks = x.unique()
print(len(x_ticks))
# good_indices = [0,3,5,7,8,9] # IMAGENETTE_224
# good_indices = [0,5,6,9,11,12] # IMAGENETTE_112
# good_indices = [0,7,9,12,13,14] # CIFAR10
good_indices = [0, 2, 9, 11, 14, 15]  # FAHION_MNIST
x_ticks = x_ticks[good_indices]
x_labels = np.exp(x_ticks)
x_labels = [f"{x:.1f}" if x < 1 else f"{x:.0f}" for x in x_labels]
x_labels = [x if x != f"{finite_max:.0f}" else r"$\infty$" for x in x_labels]
plt.xticks(x_ticks, x_labels)

ylims = plt.gca().get_ylim()
xlims = plt.gca().get_xlim()
# plt.ylim(ylims[0], ylims[1]*0.93) # IMAGENETTE_112
# plt.ylim(ylims[0]*1.08, ylims[1]) # CIFAR10
plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(2, 4))

plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=faint_grids_alpha)

plt.savefig(f"{output_dir}{dataset}_{spec_type}_{lr}_ef.pdf", bbox_inches="tight")
# %% plot depths and activations


def plot_depth(quants, spec_type):
    temp = quants.pivot_table(
        columns=["lr", "layers", "activation"],
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["lr", "layers", "activation"],
        values=spec_type,
        aggfunc="mean",
    )
    return temp


spec_type = "mr_spectral_density"
temp = plot_depth(quants, spec_type=spec_type)

# sort rows of temp according to activation functions
lrs = temp.columns.levels[0]
lrs = lrs[1:-3]
dpths = temp.columns.levels[1]
dpths = dpths[1:]
n_rows = len(dpths)
n_cols = len(lrs)
factor = 3

fig, axes = plt.subplots(
    nrows=n_rows, ncols=n_cols, figsize=(factor * n_cols, factor * n_rows), sharex=True
)

for i, lr in enumerate(lrs):
    for j, dpt in enumerate(dpths):
        ax = axes[j, i]
        for activation in activations:
            try:
                ax.plot(
                    temp[lr][dpt][activation].values[0],
                    label=activation_nice_names[activation],
                    color=activation_colors[activation],
                )
            except KeyError:
                print(f"missing {activation} {dpt} {lr}")
        ax.set_yscale("log")
        if i == 0:
            ax.set_ylabel("Power Spectral Density")
        if j == len(dpths) - 1:
            ax.set_xlabel("Frequency")

        if j == 0 and i == len(lrs) - 1:
            ax.legend()
        ax.set_title(f"lr={lr}, depth={dpt}")

        ylim = ax.get_ylim()
        rate = 0.001
        ax.set_ylim(ylim[0], ylim[1] * rate)

dataset_path = os.path.basename(dataset)
plt.savefig(f"{output_dir}{dataset_path}_{spec_type}_depth.pdf", bbox_inches="tight")

# %% plot SKBN and activations


def plot_skbn(quants, spec_type):
    temp = quants.pivot_table(
        columns=["lr", "activation", "model_name"],
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["lr", "activation", "model_name"],
        values=spec_type,
        aggfunc="mean",
    )
    return temp


spec_type = "mr_spectral_density"
temp = plot_skbn(quants, spec_type=spec_type)

lrs = temp.columns.levels[0]
activations = temp.columns.levels[1]
models = temp.columns.levels[2]
unique_models = models.unique()
unique_activations = activations.unique()
prod_act_model = list(itertools.product(unique_activations, unique_models))
combination_colors = plt.cm.tab20(np.linspace(0.1, 0.9, len(prod_act_model)))
combination_colors = {
    (activation, model): color
    for (activation, model), color in zip(prod_act_model, combination_colors)
}
dataset_path = os.path.basename(dataset)
for lr in lrs:
    if lr != 0.0001:
        print("skipping", lr)
        continue
    else:
        print(lr)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    for model in models:
        for activation in activations:
            try:
                plt.plot(
                    temp[lr][activation][model].values[0],
                    label=model_nice_names[model] if activation != "RELU" else None,
                    color=model_colors[model],
                    alpha=0.5 if activation == "RELU" else 1,
                )
            except Exception as e:
                print(f"missing {activation} {model} {lr}")
                print(e)

    plt.yscale("log")
    ylim = plt.gca().get_ylim()
    rate = 0.0005
    plt.ylim(ylim[0], ylim[1] * rate)

    plt.plot(
        [],
        label="RELU",
        color="black",
        alpha=0.5,
    )
    plt.plot(
        [],
        label=r"SP($\beta$=1)",
        color="black",
        alpha=1,
    )

    plt.ylabel("Spectral Density")
    plt.xlabel("Frequency")
    plt.legend()
    plt.savefig(
        f"{output_dir}{dataset_path}_{spec_type}_{lr}_skbn.pdf", bbox_inches="tight"
    )


# %% plot SKBN and activations ef
def plot_skbn(quants, spec_type):
    temp = quants.pivot_table(
        columns=["lr", "activation", "model_name"],
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["lr", "activation", "model_name"],
        values=spec_type,
        aggfunc="mean",
    )
    temp = temp.map(
        lambda x: np.mean((x / x.sum()) * np.arange(len(x))), na_action="ignore"
    )
    return temp


spec_type = "mr_spectral_density"
temp = plot_skbn(quants, spec_type=spec_type)

lrs = temp.columns.levels[0]
activations = temp.columns.levels[1]
models = temp.columns.levels[2]
unique_models = models.unique()
unique_activations = activations.unique()
prod_act_model = list(itertools.product(unique_activations, unique_models))
combination_colors = plt.cm.tab20(np.linspace(0.1, 0.9, len(prod_act_model)))
combination_colors = {
    (activation, model): color
    for (activation, model), color in zip(prod_act_model, combination_colors)
}
dataset_path = os.path.basename(dataset)
for lr in lrs:
    if lr != 0.0001:
        print("skipping", lr)
        continue
    else:
        print(lr)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    for model in models:
        for activation in activations:
            try:
                plt.bar(
                    x=model_nice_names[model],
                    width=0.5,
                    height=temp[lr][activation][model].values[0],
                    # label=model_nice_names[model]+" "+activation_nice_names[activation],
                    # color=combination_colors[(activation, model)],
                    color=model_colors[model],
                    alpha=0.5,
                )
            except Exception as e:
                print(f"missing {activation} {model} {lr}")
                print(e)

    plt.bar(
        x=0,
        height=0,
        width=0.5,
        label="RELU",
        color="black",
        alpha=0.5,
    )
    plt.bar(
        x=0,
        height=0,
        width=0.5,
        label=r"SP($\beta=1$)",
        color="black",
    )

    plt.ylabel("Expected Frequency")
    plt.xlabel("Architecture")
    plt.legend(loc="lower right")
    # plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(-4, -4))
    ylim = plt.gca().get_ylim()
    plt.ylim(3.1 * 1e-4, 4.7 * 1e-4)
    plt.savefig(
        f"{output_dir}{dataset_path}_{spec_type}_{lr}_skbn_ef.pdf", bbox_inches="tight"
    )


# %% plot banner
train_dataset, test_dataset = get_imagenette_dataset(
    local_pathlib.get_local_data_dir(DatasetSwitch.IMAGENETTE),
    224,
    augmentation=AugmentationSwitch.EXP_VIS,
    gaussian_noise_var=0,
    gaussian_blur_var=0,
)


def plot_mean_rank(path, image_path, is_abs):
    data = torch.load(path)
    fig, ax = plt.subplots(figsize=(4, 4))
    mean_rank = data["mean_rank"]
    if is_abs:
        mean_rank = np.abs(mean_rank)
        ax.imshow(mean_rank, cmap="viridis", vmax=np.quantile(mean_rank, 1 - q))
    else:
        ax.imshow(
            mean_rank,
            cmap="bwr",
        )
    # disable ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(image_path.replace(".pdf", "_mean_rank.pdf"), bbox_inches="tight")
    plt.close()


def plot_mean(path, image_path, is_abs):
    data = torch.load(path)
    fig, ax = plt.subplots(figsize=(4, 4))
    mean = data["mean"]
    if is_abs:
        mean = np.abs(mean)
        ax.imshow(mean, cmap="viridis", vmax=np.quantile(mean, 1 - q))
    else:
        mean = mean - mean.median()
        vmax = np.quantile(mean, 1 - q)
        vmin = np.quantile(mean, q)
        vabs = max(abs(vmax), abs(vmin))
        ax.imshow(mean, cmap="bwr", vmax=vabs, vmin=-vabs)

    # disable ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(image_path.replace(".pdf", "_mean.pdf"), bbox_inches="tight")
    plt.close()


def plot_image_from_dataset(path, image_path):
    data = torch.load(path)
    index = data["index"]
    image = test_dataset[index][0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image.squeeze().permute(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


quants_path = f".tmp/quants/hooks/*/*.pt"
paths = glob(quants_path)
is_abs = False
q = 0.005

for path in paths:
    parent_dir = os.path.dirname(path)
    parent_dir = os.path.basename(parent_dir)
    image_path = os.path.join(
        output_dir, os.path.basename(path).replace(".pt", f"_image.pdf")
    )
    print(image_path)
    if not os.path.exists(image_path):
        plot_image_from_dataset(path, image_path)

    image_path = image_path.replace(".pdf", f"{parent_dir}.pdf")
    # plot_mean_rank(path, image_path, is_abs=is_abs)
    plot_mean(path, image_path, is_abs=is_abs)
    
    # break

# %% plot accuracy
checkpoints_path = "checkpoints/**/*.pt"
paths = glob(checkpoints_path, recursive=True)
print(paths)

finite_max = 20
beta_train, acc_train = [], []
beta_test, acc_test = [], []
for path in paths:
    data = torch.load(path, map_location="cpu")
    val_acc = data["test_acc"]
    train_acc = data["train_acc"]
    activation = path.split("::")[2]
    if activation == "RELU":
        beta = finite_max
    else:
        beta = activation_betas[activation]
    beta = np.log(beta)
    beta_train.append(beta)
    acc_train.append(train_acc)
    beta_test.append(beta)
    acc_test.append(val_acc)

print(beta_test)


def scatter_plot_with_regression(
    x,
    y,
    good_indices,
    ax,
    xlabel=r"$\beta$",
    color="tab:blue",
    ylabel="forgotten",
    label="forgotten",
    finite_max=20,
):
    # sort x values and y according to x
    x, y = zip(*sorted(zip(x, y)))
    x = np.array(x)
    y = np.array(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    reg_line = slope * x + intercept

    # Compute 95% confidence interval for the regression line
    n = len(x)
    mean_x = np.mean(x)
    t_val = stats.t.ppf(0.975, n - 2)  # 95% confidence
    s_err = np.sqrt(np.sum((y - reg_line) ** 2) / (n - 2))
    ci = t_val * s_err * np.sqrt(1 / n + (x - mean_x) ** 2 / np.sum((x - mean_x) ** 2))
    upper = reg_line + ci
    lower = reg_line - ci

    # Plot the data, regression line, and confidence interval
    ax.scatter(x, y, alpha=0.3, color=color, s=5, label=label)
    ax.fill_between(x, lower, upper, color=color, alpha=0.3)
    ax.plot(x, reg_line, color=color)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    x_ticks = np.unique(x)
    x_ticks = x_ticks[good_indices]
    x_labels = np.exp(x_ticks)
    x_labels = [f"{x:.1f}" if x < 1 else f"{x:.0f}" for x in x_labels]
    x_labels = [x if x != f"{finite_max:.0f}" else r"$\infty$" for x in x_labels]
    ax.set_xticks(x_ticks, x_labels)

fig, ax = plt.subplots(figsize=(5, 3))

good_indices = [0,3, 4, 6, 7]
scatter_plot_with_regression(
    beta_train,
    acc_train,
    good_indices,
    ax,
    label="Train",
)
scatter_plot_with_regression(
    beta_test,
    acc_test,
    good_indices,
    ax,
    ylabel="Accuracy",
    color="tab:orange",
    label="Test",
)
ax.legend()
plt.savefig(f"{output_dir}{dataset}_acc.pdf", bbox_inches="tight")

# %% plot tail gaussian noise var

def plot_tgnv_ef(quants, spec_type, for_lr=None):
    temp = quants.pivot_table(
        columns=["lr","tgnv"],
        index="activation",
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp = quants.pivot_table(
        columns=["lr","tgnv"],
        index="activation",
        values=spec_type,
        aggfunc="mean",
    )
    temp = temp.map(
        lambda x: np.mean((x / x.sum()) * np.arange(len(x))), na_action="ignore"
    )
    mean = temp.apply(lambda x: np.mean(x), axis=1)
    std = temp.apply(lambda x: np.std(x), axis=1)
    return mean, std, temp

spec_type = "mr_spectral_density"
temp_mean, temp_std, temp = plot_tgnv_ef(quants, spec_type=spec_type)
print(temp.columns.get_level_values(0))
temp = temp[0.0005]
x_ticks = np.array([activation_betas[x] for x in temp.index])
finite_max = 2 * np.max(x_ticks[x_ticks != np.inf])
x_ticks = x_ticks.clip(0, finite_max)
temp["x"] = x_ticks
temp = temp.set_index("x")
temp = temp.sort_index()
# temp.columns = temp.columns.droplevel([0])
temp = temp.stack()
temp = temp.reset_index()
temp.rename(columns={0: "0"}, inplace=True)
temp.sort_values(by="x", inplace=True)

import seaborn as sns
sns.heatmap(temp.pivot_table(index="x", columns="tgnv", values="0"))

# plt.savefig(f"{output_dir}{dataset}_{spec_type}_{lr}_ef.pdf", bbox_inches="tight")
# %% create tables explanation methods
def pivot_expliners(quants, spec_type, for_explainer=None):
    temp = quants.pivot_table(
        columns=["explainer_name"],
        index="activation",
        values=spec_type,
        aggfunc="count",
    )
    # print(temp)
    temp_mean = quants.pivot_table(
        columns=["explainer_name"],
        index="activation",
        values=spec_type,
        aggfunc="mean",
    )
    temp_std = quants.pivot_table(
        columns=["explainer_name"],
        index="activation",
        values=spec_type,
        aggfunc="std",
    )
    return temp_mean, temp_std

spec_type ="mr_expected_spectral_density"    
temp_mean,temp_var = pivot_expliners(quants, spec_type=spec_type)
# print(temp.columns.get_level_values(0))

# display the tables in ipython notebook
from IPython.display import display
display(((temp_mean)*1e5).round(3))
display(((temp_mean-temp_mean["VG"][0])*1e5).round(3))

# %% plots for explanation methods
def pivot_expliners(quants, spec_type, for_explainer=None):
    temp = quants.pivot_table(
        columns=["explainer_name"],
        index="activation",
        values=spec_type,
        aggfunc="count",
    )
    print(temp)
    temp_mean = quants.pivot_table(
        columns=["explainer_name"],
        index="activation",
        values=spec_type,
        aggfunc="mean",
    )
    return temp_mean

spec_type = "mr_spectral_density"
temp_mean = pivot_expliners(quants, spec_type=spec_type)
print(temp.columns.get_level_values(0))

for activation in activations:
    plt.figure(figsize=(5, 5))
    for explainer in temp_mean.columns:
        plt.plot(temp_mean[explainer][activation], 
                 label=explainer_nice_names[explainer],
                 )
        # check numpy nd array is not nan
        if temp_mean[explainer][activation] is np.nan:
            continue
        ef = np.mean(np.arange(len(temp_mean[explainer][activation]))*temp_mean[explainer][activation])
        print(explainer,activation,ef)
    plt.yscale("log")
    plt.ylabel("Power Spectral Density")
    plt.xlabel("Frequency")
    # plt.title(f"{explainer} {activation}")
    ylim = plt.gca().get_ylim()
    rate = 0.00003
    plt.ylim(ylim[0]*(1.9), ylim[1] * rate)

    plt.legend()
    plt.savefig(f"{output_dir}{dataset}_{spec_type}_{explainer}_{activation}_ex.pdf", bbox_inches="tight")
    # plt.close()

# %% imshow explainer ranks

import torch
import matplotlib.pyplot as plt
import os


paths = glob.glob(".tmp/quants/hooks/*/*.pt")
for path in paths:
    parent_dir = path.split("/")[-2]
    data = torch.load(path)
    print(data.keys())
    plt.imshow(data["image"].permute(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f".tmp/visualizations/paper/{parent_dir}_{os.path.basename(path)}.pdf", bbox_inches="tight")

    # plt.imshow(data["mean_rank"])
    # plt.title(f"{data['mean_rank'].shape}")
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(f".tmp/visualizations/paper/{parent_dir}_{os.path.basename(path)}_mean_rank.pdf", bbox_inches="tight")
    
    plt.imshow(data["mean"])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f".tmp/visualizations/paper/{parent_dir}_{os.path.basename(path)}_mean.pdf", bbox_inches="tight")

    break
# %%
