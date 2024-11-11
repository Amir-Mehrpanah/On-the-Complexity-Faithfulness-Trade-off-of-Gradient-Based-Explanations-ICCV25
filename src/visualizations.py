import numpy as np


def compute_1d_power_spectrum(image):
    # Step 1: Compute the 2D Fourier Transform of the image
    ft_image = np.fft.fft2(image)
    ft_magnitude = np.abs(ft_image) ** 2  # Power spectrum (magnitude squared)

    # Step 2: Shift zero frequencies to the center
    ft_magnitude_shifted = np.fft.fftshift(ft_magnitude)

    # Step 3: Generate radial distances for the spectrum
    center = np.array(ft_magnitude_shifted.shape) // 2
    y, x = np.indices(ft_magnitude_shifted.shape)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r = r.astype(np.int32)

    # Step 4: Radial average
    radial_sum = np.bincount(r.ravel(), ft_magnitude_shifted.ravel())
    radial_count = np.bincount(r.ravel())
    radial_profile = radial_sum / radial_count  # Average power in each radius

    return radial_profile
