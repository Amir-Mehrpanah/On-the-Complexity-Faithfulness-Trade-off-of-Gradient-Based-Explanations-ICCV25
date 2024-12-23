# %% imports read quants
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from src.utils import DatasetSwitch

os.chdir("/home/x_amime/x_amime/projects/kernel-view-to-explainability/")

output_dir = ".tmp/visualizations/paper/"
dataset = DatasetSwitch.FASHION_MNIST
quants_path = f".tmp/quants/{dataset}::l2_reg::quants.pt"
quants = torch.load(quants_path)
quants = pd.DataFrame(quants)
quants["noise_scale"] = quants["noise_scale"].astype(float)
quants["index"] = quants["index"].astype(int)
temp = quants["address"].apply(lambda x: x[0].split("/")[-2].split("::"))
temp = pd.DataFrame(
    temp.tolist(),
    columns=[
        "dataset",
        "model_name",
        "layers",
        "activation",
        "seed",
        "l2reg",
        "input_size",
        "ns",
    ],
)
temp.drop(columns=["ns"], inplace=True)
quants = pd.concat([quants, temp], axis=1)
quants = quants.set_index(
    [
        "layers",
        "activation",
        "seed",
        "l2reg",
        "input_size",
        "noise_scale",
        "model_name",
        "index",
    ]
)

nice_name = {
    "RELU": "ReLU",
    "LEAKY_RELU": "Leaky ReLU",
    "SOFTPLUS_B_1": "Softplus 0.1",
    "SOFTPLUS_B1": "Softplus 1",
    "SOFTPLUS_B5": "Softplus 5",
}
activations = [
    "RELU",
    "LEAKY_RELU",
    "SOFTPLUS_B5",
    "SOFTPLUS_B1",
    "SOFTPLUS_B_1",
]
colors = plt.cm.tab10(np.arange(len(activations)))
colors = {activation: color for activation, color in zip(activations, colors)}
quants


# %% def density_depth_inputsize
def density_depth_inputsize(
    quants, spec_type="mr_spectral_density", for_noise_scale=1e-05
):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale"],
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
                            temp[input_size][layers][activation][noise_scale].values[0],
                            color=colors[activation],
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {activation} {noise_scale}"
                        )
                        pass

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [], label=nice_name[activation], color=colors[activation]
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
                            color=colors[activation],
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {activation} {noise_scale}"
                        )
                        pass

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [], label=nice_name[activation], color=colors[activation]
                        )
                        ax.legend()


def expected_freq_depth(quants):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "noise_scale", "activation"],
        values="mr_expected_spectral_density",
        aggfunc=np.mean,
    )

    factor = 4
    width = 0.1
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
                        ax.bar(
                            x=lid + (color_indx - 2) * width,
                            height=temp[input_size][layers][noise_scale][
                                activation
                            ].values[0],
                            color=colors[activation],
                            width=width,
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {noise_scale} {activation}"
                        )
                        pass

            for activation in temp.columns.levels[3]:
                ax.bar(
                    height=0, x=1, label=nice_name[activation], color=colors[activation]
                )
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
                        color=colors[activation],
                    )
                except KeyError:
                    print(f"missing {input_size} {layers} {activation}")
                    pass

                if lid == 0 and sid == 0:
                    ax.plot([], label=nice_name[activation], color=colors[activation])
                    ax.legend()


def density_depth_noisescale(
    quants, spec_type="mr_spectral_density", for_noise_scale=1e-05
):
    temp = quants.pivot_table(
        columns=["input_size", "layers", "activation", "noise_scale"],
        values=spec_type,
        aggfunc=np.mean,
    )
    factor = 3
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

                if len(temp.columns.levels[3]) == 1:
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
                    try:
                        ax.plot(
                            temp[input_size][layers][activation][noise_scale].values[0],
                            color=colors[activation],
                        )
                    except KeyError:
                        print(
                            f"missing {input_size} {layers} {activation} {noise_scale}"
                        )
                        pass

                    if lid == 0 and nid == 0:
                        ax.plot(
                            [], label=nice_name[activation], color=colors[activation]
                        )
                        ax.legend()


# %% plot expected_freq_depth
expected_freq_depth(quants)

# %% plot density_depth_inputsize
for ns in quants.index.levels[5]:
    for spec_type in ["mr_spectral_density", "vr_spectral_density"]:
        density_depth_inputsize(quants, spec_type=spec_type, for_noise_scale=ns)
        plt.savefig(f"{output_dir}{dataset}_{spec_type}_{ns}.pdf")
        plt.close()

# %% plot density_depth_noisescale
for isz in quants.index.levels[4]:
    for spec_type in ["mr_spectral_density", "vr_spectral_density"]:
        density_depth_noisescale(quants, spec_type=spec_type, for_noise_scale=isz)
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
