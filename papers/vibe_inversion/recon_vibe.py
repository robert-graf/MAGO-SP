from pathlib import Path

import numpy as np
from TPTBox import NII


def smooth_gaussian(self: np.ndarray, sigma: float | list[float] | tuple[float], truncate: float = 4.0, nth_derivative=0):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(self, sigma, order=nth_derivative, cval=0, truncate=truncate)


def vibe_separate_phase_from_guess_nii(inphase: NII, outphase: NII, water_guess: NII, smooth=False):
    """
    Separates water and fat signals from in-phase and out-phase MRI images
    based on an initial guess for the water signal, and optionally smooths the results.

    Parameters:
        inphase (NII): NIfTI object containing the in-phase MRI image data.
        outphase (NII): NIfTI object containing the out-phase MRI image data.
        water_guess (NII): NIfTI object containing an initial guess for the water signal.
        smooth (bool, optional): If True, applies Gaussian smoothing to the predicted water
                                  and fat images. Defaults to False.

    Returns:
        Tuple[NII, NII]: Two NIfTI objects containing the separated water and fat images.
    """
    w, f = vibe_separate_phase_from_guess(inphase.get_array(), outphase.get_array(), water_guess.get_array(), smooth)
    return inphase.set_array(w), inphase.set_array(f)


def vibe_separate_phase_from_guess(inphase: np.ndarray, outphase: np.ndarray, water_guess: np.ndarray, smooth=False):
    """
    Separates water and fat signals from in-phase and out-phase MRI images
    based on an initial guess for the water signal.
    See: https://onlinelibrary.wiley.com/doi/pdf/10.1002/jmri.21492
    Parameters:
        inphase (np.ndarray): In-phase MRI image data as a NumPy array.
        outphase (np.ndarray): Out-phase MRI image data as a NumPy array.
        water_guess (np.ndarray): Initial guess for the water signal as a NumPy array.
        smooth (bool, optional): If True, applies Gaussian smoothing to the predicted water
                                  and fat arrays. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two NumPy arrays containing the separated water
                                       and fat signals.
    """

    s_0 = outphase.astype(float)
    s_1 = inphase.astype(float)
    w = water_guess

    a_pred = (s_0 + s_1) * 0.5
    b_pred = abs(s_0 - s_1) * 0.5
    msk = abs(a_pred - w) < abs(b_pred - w)
    w_pred = a_pred * msk + b_pred * (1 - msk)
    f_pred = b_pred * msk + a_pred * (1 - msk)
    if smooth:
        w_pred = smooth_gaussian(w_pred, 0.45)
        f_pred = smooth_gaussian(f_pred, 0.45)
    return w_pred, f_pred


def vibe_separate_water_fat_from_guess(water: NII, fat: NII, water_guess: NII):
    """
    Separates water and fat signals by swapping pixels based on an initial guess for the water signal.

    Parameters:
        water (NII): NIfTI object containing the water signal.
        fat (NII): NIfTI object containing the fat signal.
        water_guess (NII): NIfTI object containing an initial guess for the water signal.

    Returns:
        Tuple[NII, NII]: Two NumPy arrays containing the updated water
                                       and fat signals after swapping pixels.
    """
    w = water_guess.copy().set_dtype_(float)

    msk = abs(water - w) < abs(fat - w)
    w_pred = water * msk + fat * (1 - msk)
    f_pred = fat * msk + water * (1 - msk)
    return w_pred, f_pred


if __name__ == "__main__":
    from TPTBox import to_nii

    nako_dataset = "/media/data/NAKO/dataset-nako/"
    c = Path("/media/data/NAKO/dataset-nako/rawdata/100/100000/vibe/")
    crop = slice(0, None), slice(0, None), slice(50, 52)

    w = to_nii(c / "sub-100000_acq-ax_chunk-3_part-water_vibe.nii.gz")
    f = to_nii(c / "sub-100000_acq-ax_chunk-3_part-fat_vibe.nii.gz")
    s_0 = to_nii(c / "sub-100000_acq-ax_chunk-3_part-outphase_vibe.nii.gz")
    s_1 = to_nii(c / "sub-100000_acq-ax_chunk-3_part-inphase_vibe.nii.gz")
    water_prior = w + np.random.rand(*w.shape)  # TODO: Replace with signal prior
    w_pred, f_pred = vibe_separate_phase_from_guess(s_1, s_0, water_prior, smooth=False)
    (w_pred).save("predictions/vibe/water_vibe.nii.gz")
    (f_pred).save("predictions/vibe/fat_vibe.nii.gz")
    w = to_nii(c / "sub-100000_acq-ax_chunk-3_part-water_vibe.nii.gz")
    f = to_nii(c / "sub-100000_acq-ax_chunk-3_part-fat_vibe.nii.gz")

    w_pred, f_pred = vibe_separate_water_fat_from_guess(w, f, water_prior)
    (f_pred).save("predictions/vibe/fat_vibe2.nii.gz")
