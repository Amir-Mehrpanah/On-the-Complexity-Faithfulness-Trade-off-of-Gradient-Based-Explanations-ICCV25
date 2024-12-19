# %% imports read quants
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

quants_path = ".tmp/quants/quants.pt"
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

# %% plot cosine similarity
temp = quants.pivot_table(
    index=["noise_scale"],
    columns=["input_size", "layers", "activation"],
    values="cosine_similarity",
    aggfunc="mean",
)

for sizes in temp.columns.levels[0]:
    for layers in temp.columns.levels[1]:
        temp[sizes][layers].plot()
        plt.title(f"cosim i{sizes} l{layers}")
        plt.xscale("log")
        # plt.yscale("log")
        plt.show()


# %% plot mr_spectral density
temp = quants.pivot_table(
    index=["noise_scale"],
    columns=["input_size", "layers", "activation", "noise_scale"],
    values="mr_spectral_density",
    aggfunc=np.mean,
)
quants.loc["1", "SOFTPLUS_B1", "0", "0.005", "28", 0.01]["mr_spectral_density"]
# colors = plt.cm.rainbow(np.linspace(0, 1, len(temp.columns.levels[2])))
# ls_markers = ["-", "--", "-.", ":"]
# for sizes in temp.columns.levels[0]:
#     for lid,layers in enumerate(temp.columns.levels[1]):
#         fig = plt.figure()
#         for color_indx, activation in enumerate(temp.columns.levels[2]):
#             for alpha, noise_scale in enumerate(temp.index):
#                 try:
#                     plt.plot(
#                         temp[sizes][layers][activation][noise_scale],
#                         alpha=(alpha+1) / (len(temp.index)+1),
#                         color=colors[color_indx],
#                     )
#                 except KeyError:
#                     print(f"missing {sizes} {layers} {activation} {noise_scale}")
#                     pass
#             # if lid == 0:
#             plt.plot([], label=activation, color=colors[color_indx])
#             plt.legend()
#         plt.title(f"psd for i{sizes} l{layers}")
#         plt.xscale("log")
#         plt.yscale("log")
# %% debug
# mean_psd = []
# for noise_scale in quants.index.levels[5]:
#     print(noise_scale)
#     mean_psd.append(
#         np.array(
#             quants.loc[
#                 "1", "SOFTPLUS_B1", "0", "0.005", "64", noise_scale, "SIMPLE_CNN_DEPTH"
#             ]["mr_spectral_density"].tolist()
#         )
#     )
# mean_psd = np.array(mean_psd)
# plt.plot(mean_psd.T)
noise_scale = 1e-05
A = np.array(quants.loc[
    "1",
    "SOFTPLUS_B1",
    "0",
    "0.005",
    "64",
    noise_scale,
    "SIMPLE_CNN_DEPTH",
]["mr_spectral_density"].tolist())
noise_scale = 0.001
B = np.array(quants.loc[
    "1",
    "SOFTPLUS_B1",
    "0",
    "0.005",
    "64",
    noise_scale,
    "SIMPLE_CNN_DEPTH",
]["mr_spectral_density"].tolist())
A
B
