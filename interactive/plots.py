# %% imports read quants
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

os.chdir("/home/x_amime/x_amime/projects/kernel-view-to-explainability/")

output_dir = ".tmp/visualizations/paper/"
dataset = [
    # "IMAGENETTE_122_beta_1k",
    "IMAGENETTE_224_depth_1k"
][0]
quants_path = f".tmp/quants/{dataset}::quants.pt"
quants = torch.load(quants_path)
quants = pd.DataFrame(quants)
quants["noise_scale"] = quants["noise_scale"].astype(float)
quants["index"] = quants["index"].astype(int)
temp = quants["address"].apply(lambda x: x[0].split("/")[-2].split("::"))
temp = pd.DataFrame(
    temp.tolist(),
    columns=[
        "model_name",
        "layers",
        "activation",
        "seed",
        "l2reg",
        "input_size",
        "lr",
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
        "lr",
        "index",
    ]
)

activation_nice_names = {
    "RELU": "ReLU",
    "LEAKY_RELU": "Leaky ReLU",
    "SOFTPLUS_B_1": "Softplus 0.1",
    "SOFTPLUS_B1": "Softplus 1",
    "SOFTPLUS_B3": "Softplus 3",
    "SOFTPLUS_B5": "Softplus 5",
    "SOFTPLUS_B7": "Softplus 7",
    "SOFTPLUS_B10": "Softplus 10",
    "SOFTPLUS_B50": "Softplus 50",
    "SOFTPLUS_B100": "Softplus 100",
}
activations = [
    "RELU",
    "LEAKY_RELU",
    "SOFTPLUS_B10",
    # "SOFTPLUS_B7",
    "SOFTPLUS_B5",
    "SOFTPLUS_B50",
    "SOFTPLUS_B100",
    # "SOFTPLUS_B3",
    # "SOFTPLUS_B1",
    # "SOFTPLUS_B_1",
]
markers = ["o", "s", "D", "v", "^", "<", ">", "p", "P", "*", "X", "d", "H", "h"]
colors = plt.cm.tab10(np.arange(len(activations)))
activation_colors = {
    activation: color for activation, color in zip(activations, colors)
}

models = [
    "SIMPLE_CNN",
    "SIMPLE_CNN_BN",
    "SIMPLE_CNN_SK",
    "SIMPLE_CNN_SK_BN",
    "RESNET_BASIC",
    "RESNET_BOTTLENECK",
]
colors = plt.cm.tab10(np.arange(len(models)))
model_colors = {model: color for model, color in zip(models, colors)}
model_nice_names = {
    "SIMPLE_CNN": "CNN",
    "SIMPLE_CNN_BN": "CNN BN",
    "SIMPLE_CNN_SK": "CNN SK",
    "SIMPLE_CNN_SK_BN": "CNN SK+BN",
    "RESNET_BASIC": "ResNet Basic",
    "RESNET_BOTTLENECK": "ResNet Bottleneck",
}

quants


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
                            vline = len(
                                    temp[input_size][layers][activation][noise_scale][lr].values[0]
                                )*0.5
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


# %%
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


# %%


def plot_activations(quants, spec_type, for_lr=0.001):
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

                for color_indx, activation in enumerate(activations):
                    for rid, lr in enumerate(temp.columns.levels[4]):
                        if lr != "0.001":
                            continue
                        # if activation == "SOFTPLUS_B5" and lr != "0.001":
                        #     continue
                        # if activation == "SOFTPLUS_B10" and lr != "0.001":
                        #     continue
                        # if activation == "SOFTPLUS_B50" and lr == "0.005":
                        #     continue
                        # if activation == "SOFTPLUS_B100" and lr == "0.005":
                        #     continue
                        # if activation == "RELU" and lr != "0.0001":
                        #     continue
                        # if activation == "LEAKY_RELU" and lr != "0.0001":
                        #     continue

                        try:
                            ax.plot(
                                temp[input_size][layers][activation][noise_scale][
                                    lr
                                ].values[0],
                                color=activation_colors[activation],
                                alpha=0.5,
                                marker=markers[rid],
                                markersize=0.3,
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

                    if lid == 0 and sid == 0:
                        ax.plot(
                            [],
                            label=activation_nice_names[activation],
                            color=activation_colors[activation],
                        )
                        ax.legend()

                if lid == n_rows - 1:
                    ax.set_xlabel("frequency")

                ax.set_xscale("log")
                ax.set_yscale("log")


plot_activations(
    quants,
    spec_type="mr_spectral_density",
)
plt.savefig(f"{output_dir}{dataset}_activations.pdf")

# %%
