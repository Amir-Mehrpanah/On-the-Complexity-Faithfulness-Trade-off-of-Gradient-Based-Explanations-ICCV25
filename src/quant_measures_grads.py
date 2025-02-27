import os
import numpy as np
import torch
from src.datasets import get_grad_dataloader
from src.utils import EXPERIMENT_PREFIX_SEP
import gc


def cosine_similarity(data):
    return (data["mean_rank"] * data["var_rank"]).sum() / (
        data["var_rank"].sum() * data["mean_rank"].sum()
    )


def spectral_density(image):
    if image.ndim == 3:
        image = image.squeeze(0)
    # Compute the 2D Fourier Transform of the image
    ft_image = np.fft.fft2(image)
    ft_magnitude = np.abs(ft_image) ** 2  # Power spectrum (magnitude squared)

    # Shift zero frequencies to the center
    ft_magnitude_shifted = np.fft.fftshift(ft_magnitude)

    # Generate radial distances for the spectrum
    center = np.array(ft_magnitude_shifted.shape) // 2
    y, x = np.indices(ft_magnitude_shifted.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)

    # Radial average
    radial_sum = np.bincount(r.ravel(), ft_magnitude_shifted.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / radial_count  # Average power in each radius
    return radial_profile


def measure_grads(data):
    results = {
        "mr_spectral_density": spectral_density(data["mean_rank"]),
        "m_spectral_density": spectral_density(data["mean"]),
        # "cosine_similarity": cosine_similarity(data),
        # "vr_spectral_density": spectral_density(data["var_rank"]),
        # "v_spectral_density": spectral_density(data["var"]),
    }
    freq = np.linspace(0, 1, len(results["mr_spectral_density"]))
    # see paper eq 5
    results["mr_expected_spectral_density"] = (
        freq * results["mr_spectral_density"] / results["mr_spectral_density"].sum()
    ).mean()

    # results["vr_expected_spectral_density"] = (
    #     freq * results["vr_spectral_density"]
    # ).mean()
    # results["v_expected_spectral_density"] = (
    #     freq * results["v_spectral_density"]
    # ).mean()
    return results


def main(
    *,
    root_path,
    name,
    num_workers,
    prefetch_factor,
    hook_samples,
    output_dir,
    **kwargs,
):
    print(f"num_workers: {num_workers}")
    print(f"prefetch_factor: {prefetch_factor}")
    print(f"kwargs: {kwargs}")
    hooks_dir = os.path.join(output_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)

    # check if name is a path like path_files/file_prefix or only file_prefix
    # create path_files if it does not exist
    if os.path.dirname(name):
        temp = os.path.join(output_dir, os.path.dirname(name))
        os.makedirs(temp, exist_ok=True)
        print(f"created directory under {output_dir} with name {os.path.dirname(name)}")

    dataloader = get_grad_dataloader(
        root_path,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    measurements = []
    print(f"len(dataloader): {len(dataloader)}")
    q10_dataloader = len(dataloader) // 10
    for i, data in enumerate(dataloader):
        quant = measure_grads(data)
        quant["index"] = data["index"]
        quant["address"] = data["address"]
        quant["noise_scale"] = data["noise_scale"]
        measurements.append(quant)
        if data["index"] in hook_samples:
            address = data["address"][0]
            parent_dir = os.path.basename(os.path.dirname(address))
            os.system(f"rsync -a {address} {hooks_dir}/{parent_dir}/")  # faster
            # torch.save(data, f"{hooks_dir}/{data['address']}") # slower

        if ((i > 0) and (i % q10_dataloader == 0)) or (i == len(dataloader) - 1):
            print(f"{i / len(dataloader):.2%} is processed")
            gc.collect()

            file_name = os.path.join(
                output_dir, f"{name}{EXPERIMENT_PREFIX_SEP}_{i}_quants.pt"
            )

            torch.save(measurements, file_name)
            measurements = []
