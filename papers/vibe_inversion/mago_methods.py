import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.append(str(Path(__file__).parent))
from recon_mevibe import alpha_p, freqs_ppm, multipeak_fat_model_from_guess, multipeak_fat_model_smooth, ti_ms_default
from recon_mevibe_rician_sigma import estimate_rician_sigma, estimate_rician_sigma_from_guess


def load_ISMRM(data: dict | str | Path, absolute: bool = True) -> dict:
    """
    Load and process ISMRM (International Society for Magnetic Resonance in Medicine) fat-water toolbox data.

    The function reads input data from a dictionary or a file path (MAT file) and extracts the required
    imaging data and metadata for further processing. Only single-coil data is supported.

    Parameters:
    ----------
    data : dict | str | Path
        The input data to be processed. Can be a dictionary, a file path to a .mat file, or a Path object.
    absolute : bool, optional
        If True, the magnitude of the signal data is taken (default is True).

    Returns:
    -------
    dict
        A dictionary containing:
        - "s_magnitude_arr": List of processed signal magnitudes (absolute or complex).
        - "MagneticFieldStrength": Magnetic field strength of the MRI system.
        - "ti_ms": Echo times (TE) in milliseconds.

    Notes:
    -----
    - The ISMRM fat-water toolbox format is described here:
      https://www.ismrm.org/workshops/FatWater12/data.htm.
    - This function assumes the input data follows the specific structure provided by the toolbox.

    Example:
    --------
    >>> result = load_ISMRM("path/to/data.mat", absolute=True)
    >>> print(result["s_magnitude_arr"])
    """
    # Load data from file if necessary
    if isinstance(data, (Path, str)):
        im_data: dict = sio.loadmat(data)
    else:
        im_data = data

    # Extract the main data structure
    if "imDataAll" in im_data:
        im_data = im_data["imDataAll"][0, 0]

    # Unpack the expected data structure
    MagneticFieldStrength, ti_ms, unk, unk2, unk3, raw = im_data

    # Print raw shape for debugging purposes
    print("Raw data shape:", raw.shape)

    # Process the signal magnitudes
    s_magnitude_arr = [raw[..., i] for i in range(raw.shape[-1])]
    if absolute:
        s_magnitude_arr = [np.abs(signal) for signal in s_magnitude_arr]

    return {"s_magnitude_arr": s_magnitude_arr, "MagneticFieldStrength": MagneticFieldStrength, "ti_ms": ti_ms}


def mago_ISMRM(data: dict | str | Path, **args):
    """
    Wrapper function for the `mago` function that integrates with the ISMRM data loader.

    Parameters:
    ----------
    data : dict | str | Path
        The input data to be processed. Can be a dictionary, a file path to a .mat file, or a Path object.
    **args : dict
        Additional keyword arguments to be passed to the `mago` function.

    Notes:
    -----
    This function relies on the `load_ISMRM` function to parse the input data.
    """
    return mago(**load_ISMRM(data), **args)


def magorino_ISMRM(data: dict | str | Path, **args):
    """
    Wrapper function for the `magorino` function that integrates with the ISMRM data loader.

    Parameters:
    ----------
    data : dict | str | Path
        The input data to be processed. Can be a dictionary, a file path to a .mat file, or a Path object.
    **args : dict
        Additional keyword arguments to be passed to the `magorino` function.

    Notes:
    -----
    This function relies on the `load_ISMRM` function to parse the input data.
    """
    return magorino(**load_ISMRM(data), **args)


def mago_sp_ISMRM(data: dict | str | Path, water_guess: np.ndarray, fat_guess: np.ndarray | None = None, **args):
    """
    Wrapper function for the `mago_sp` function that integrates with the ISMRM data loader.

    Parameters:
    ----------
    data : dict | str | Path
        The input data to be processed. Can be a dictionary, a file path to a .mat file, or a Path object.
    water_guess : np.ndarray
        Initial guess for water signal.
    **args : dict
        Additional keyword arguments to be passed to the `mago_sp` function.

    Notes:
    -----
    This function relies on the `load_ISMRM` function to parse the input data.
    """
    return mago_sp(**load_ISMRM(data), water_guess=water_guess, fat_guess=fat_guess, **args)


def mago(
    s_magnitude_arr: list[np.ndarray],
    MagneticFieldStrength: float = 3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    smooth: bool = False,
    sigma_smooth: int = 1,
    factor: float = 1.0,
):
    """
    Perform fat-water separation using a multipeak fat model with optional smoothing.

    Parameters:
    ----------
    s_magnitude_arr : list[np.ndarray]
        List of signal magnitude arrays (e.g., MRI images).
    MagneticFieldStrength : float, optional
        Magnetic field strength of the MRI system in Tesla (default is 3.0).
    ti_ms : list[float] | np.ndarray | None, optional
        Echo times (TE) in milliseconds (default is `ti_ms_default`).
    alpha_p : float
        Parameter for the fat model fitting (default is `alpha_p`).
    freqs_ppm : list[float]
        Fat resonance frequencies in ppm (default is `freqs_ppm`).
    smooth : bool, optional
        Whether to apply smoothing to the signal (default is False).
    sigma_smooth : int, optional
        Standard deviation for smoothing kernel (default is 1).
    factor : float, optional
        Reweighting factor for solution preference (default is 1.0).

    Notes:
    -----
    This function assumes at least one signal magnitude array is provided.
    """
    assert len(s_magnitude_arr) >= 1, "At least one signal magnitude array is required."

    return multipeak_fat_model_smooth(
        s_magnitude_arr,
        MagneticFieldStrength=MagneticFieldStrength,
        ti_ms=ti_ms,
        sigma_smooth=sigma_smooth,
        alpha_p=alpha_p,
        freqs_ppm=freqs_ppm,
        smooth=smooth,
        factor=factor,
        rician_loss=False,
    )


def magorino(
    s_magnitude_arr: list[np.ndarray],
    MagneticFieldStrength: float = 3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    smooth: bool = False,
    sigma_smooth: int = 1,
    factor: float = 1.0,
    subset_size=10000,
    rician_over_estimation=1.163,
    sigma=16,
):
    """
    Perform fat-water separation with Rician noise modeling.

    Parameters:
    ----------
    s_magnitude_arr : list[np.ndarray]
        List of signal magnitude arrays (e.g., MRI images).
    MagneticFieldStrength : float, optional
        Magnetic field strength of the MRI system in Tesla (default is 3.0).
    ti_ms : list[float] | np.ndarray | None, optional
        Echo times (TE) in milliseconds (default is `ti_ms_default`).
    alpha_p : float
        Parameter for the fat model fitting (default is `alpha_p`).
    freqs_ppm : list[float]
        Fat resonance frequencies in ppm (default is `freqs_ppm`).
    smooth : bool, optional
        Whether to apply smoothing to the signal (default is False).
    sigma_smooth : int, optional
        Standard deviation for smoothing kernel (default is 1).
    factor : float, optional
        Reweighting factor for solution preference (default is 1.0).

    Notes:
    -----
    - This function first estimates the Rician noise standard deviation (sigma).
    - Rician noise modeling is used during the fat-water separation.
    """
    if sigma <= 0:
        _, sigma = estimate_rician_sigma(
            s_magnitude_arr=s_magnitude_arr,
            MagneticFieldStrength=MagneticFieldStrength,
            ti_ms=ti_ms,
            alpha_p=alpha_p,
            freqs_ppm=freqs_ppm,
            subset_size=subset_size,
            rician_over_estimation=rician_over_estimation,
        )
        # sigmas = sorted(list(sigmas))
        # print([sigmas[i] for i in range(0, len(sigmas), len(sigmas) // 10)])
        print("rician_noise estimation", sigma)
        # return None
    return multipeak_fat_model_smooth(
        s_magnitude_arr,
        MagneticFieldStrength=MagneticFieldStrength,
        ti_ms=ti_ms,
        sigma_smooth=sigma_smooth,
        alpha_p=alpha_p,
        freqs_ppm=freqs_ppm,
        smooth=smooth,
        factor=factor,
        rician_loss=True,
        sigma_rician=sigma,
    )


def mago_sp(
    s_magnitude_arr: list[np.ndarray],
    water_guess: np.ndarray,
    fat_guess: np.ndarray | None = None,
    MagneticFieldStrength: float = 3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    use_rician: bool = True,
    sigma: float = 16.0,
    subset_size=20000,
    rician_over_estimation=1.163,
):
    """
    Perform fat-water separation with initial water and optional fat guesses.

    Parameters:
    ----------
    s_magnitude_arr : list[np.ndarray]
        List of signal magnitude arrays (e.g., MRI images).
    water_guess : np.ndarray
        Initial guess for water signal.
    fat_guess : np.ndarray | None, optional
        Initial guess for fat signal. If None, it is assumed that the first image
        is an in-phase image, and the fat guess is calculated as `s_magnitude_arr[0] - water_guess`.
    MagneticFieldStrength : float, optional
        Magnetic field strength of the MRI system in Tesla (default is 3.0).
    ti_ms : list[float] | np.ndarray | None, optional
        Echo times (TE) in milliseconds (default is `ti_ms_default`).
    alpha_p : float
        Parameter for the fat model fitting (default is `alpha_p`).
    freqs_ppm : list[float]
        Fat resonance frequencies in ppm (default is `freqs_ppm`).
    use_rician : bool, optional
        Whether to use Rician noise modeling (default is True). If False, the solution is biased for <5% noise.
    sigma : float, optional
        Standard deviation of Rician noise. If 0 and `use_rician` is True, it will be estimated.

    Notes:
    -----
    - This function allows for fat-water separation with pre-defined guesses for water and fat.
    - If `use_rician` is True and `sigma` is 0, the noise standard deviation will be estimated.
    """
    if fat_guess is None:
        # Assume first image is an in-phase image, if a fat_guess is not given
        fat_guess = s_magnitude_arr[0] - water_guess

    if use_rician and sigma == 0:
        _, sigma = estimate_rician_sigma_from_guess(
            s_magnitude_arr=s_magnitude_arr,
            water_guess=water_guess,
            fat_guess=fat_guess,
            MagneticFieldStrength=MagneticFieldStrength,
            ti_ms=ti_ms,
            alpha_p=alpha_p,
            freqs_ppm=freqs_ppm,
            subset_size=subset_size,
            rician_over_estimation=rician_over_estimation,
        )
        print("rician_noise estimation", sigma)
    return multipeak_fat_model_from_guess(
        s_magnitude_arr,
        water_guess=water_guess,
        fat_guess=fat_guess,
        MagneticFieldStrength=MagneticFieldStrength,
        ti_ms=ti_ms,
        alpha_p=alpha_p,
        freqs_ppm=freqs_ppm,
        rician_loss=use_rician,
        sigma=sigma,
    )


if __name__ == "__main__":
    load_ISMRM("/media/data/robert/code/MEVIBE_inversion/datasets/site4_1p5T_protocol1.mat")
