import os
import random
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import least_squares
from scipy.special import i0
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

sys.path.append(str(Path(__file__).parents[2]))
from papers.vibe_inversion.recon_mevibe import RicianLogLik, alpha_p, freqs_ppm, get_freqs_hz, rss, signal_model_magnitude, ti_ms_default

## This is a copy of recon_mevibe for RICIAN-Noise estimation. On in vivo data the estimation did not yield realistic values (>200).
## We use sigma = 16 to not get artifacts but still improve PDFF
## With a signal prior we get 10-50 optimal rician noise.


def loss_function(params, s_magnitude, ti, alpha_p, freqs_hz, rician_loss=True):
    p_w, p_f, R2_star, sigma = params
    model_signal = signal_model_magnitude(ti, p_w, p_f, R2_star, alpha_p, freqs_hz)
    return -RicianLogLik(s_magnitude, model_signal, sigma=sigma)


def optimize_voxel_magnitude(s_magnitude: np.ndarray, ti: np.ndarray, r2, initial_guess: tuple[float, float], alpha_p, freqs_hz, rician_loss=True, sigma=2):
    """
    Optimize p_w, p_f, and R2* for a given voxel using least squares minimization.

    Parameters:
    -----------
    s_magnitude : ndarray
        Observed magnitude signals for a voxel.
    ti : ndarray
        Echo times (in seconds).
    initial_guess : list or ndarray
        Initial guess for the parameters [p_w, p_f, R2_star].

    Returns:
    --------
    ndarray
        Optimized parameters [p_w, p_f, R2_star].

    Math:
    -----
    The function minimizes the loss function to estimate the water and fat proton densities and \\( R_2^* \\).
    """
    s_magnitude = s_magnitude.astype(np.float32)
    # Ensure sigma is a numpy array

    guess = [*initial_guess, r2, sigma]
    p_w = 100000

    try:
        if len(s_magnitude) >= 3:
            res = least_squares(loss_function, guess, args=(s_magnitude, ti, alpha_p, freqs_hz, rician_loss), method="lm")
            p_w, p_f, r2s, sigma = res.x
    except Exception:
        p_w = 10000000
        p_f = 10000000
    if p_w > 1000 or p_f > 1000:
        guess = [max(min(g, 1000.0), 0.0) for g in guess]
        try:
            res = least_squares(
                loss_function, guess, args=(s_magnitude, ti, alpha_p, freqs_hz, rician_loss), bounds=((0, 0, 0, -10), (1000, 1000, 1000, 100000000)), method="dogbox"
            )
        except Exception:
            # print(f"({guess=}, {s_magnitude=}, {ti=}, {r2=}, {freqs_hz=})")
            return 0, 0, 0, -1
        p_w, p_f, r2s, sigma = res.x
        if p_w >= 1000 or p_f >= 1000:
            return 0, 0, 0, -1
    p_w = max(p_w, 0)
    p_f = max(p_f, 0)
    return p_w, p_f, r2s, sigma


def _process_voxel(idx, s_magnitude, ti, alpha_p, freqs_hz, rician_loss=True, sigma=2):
    if s_magnitude.sum() <= 20:
        return -1
    if np.prod(s_magnitude) < 0:
        return -1
    initial_guess = (0.0, 1000.0)
    p_w, p_f, r2s, sigma = optimize_voxel_magnitude(s_magnitude, ti, 100, initial_guess, alpha_p, freqs_hz, rician_loss=rician_loss, sigma=sigma)
    loss = rss(p_w, p_f, s_magnitude, ti, r2s, alpha_p, freqs_hz)

    # Try second initial guess
    initial_guess2 = (1000.0, 0.0)
    p_w2, p_f2, r2s2, sigma2 = optimize_voxel_magnitude(s_magnitude, ti, 100, initial_guess2, alpha_p, freqs_hz, rician_loss=rician_loss, sigma=sigma)
    loss2 = rss(p_w2, p_f2, s_magnitude, ti, r2s2, alpha_p, freqs_hz)
    return sigma if loss < loss2 else sigma2  # abs(p_w), abs(p_f), abs(p_w2), abs(p_f2), loss, loss2, r2s, r2s2, idx


def _refine_prediction(idx, s_magnitude: np.ndarray, guess_water, guess_fat, r2s, ti, alpha_p, freqs_hz, sigma=2):
    if s_magnitude.sum() <= 20:
        return 0, 0, 0, -1, idx
    if np.prod(s_magnitude) < 0:
        return 0, 0, 0, -1, idx

    initial_guess = (guess_water, guess_fat)
    p_w, p_f, r2s, sigma = optimize_voxel_magnitude(s_magnitude, ti, r2s, initial_guess, alpha_p, freqs_hz, rician_loss=True, sigma=sigma)
    # loss = rss(p_w, p_f, s_magnitude, ti, r2s, alpha_p, freqs_hz)
    return abs(p_w), abs(p_f), r2s, sigma, idx


def estimate_rician_sigma(
    s_magnitude_arr: list[np.ndarray],
    MagneticFieldStrength=3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    rician_loss=True,
    subset_size=None,
    rician_over_estimation=1.163,
):
    assert len(s_magnitude_arr) > 1, len(s_magnitude_arr)
    if ti_ms is None:
        ti_ms = ti_ms_default
    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]
    ti = np.array(ti_ms)
    shape = s_magnitude_arr[0].shape
    all_indices = list(np.ndindex(shape))
    if subset_size is not None and subset_size > np.prod(shape):
        subset_size = None
    selected_indices = random.sample(all_indices, subset_size) if subset_size is not None else all_indices

    # Use joblib to parallelize the process_voxel function
    freqs_hz = get_freqs_hz(freqs_ppm, MagneticFieldStrength)
    with tqdm_joblib(tqdm(desc="Processing voxels sigmar", total=len(selected_indices))):
        results = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
            delayed(_process_voxel)(idx, np.array([i[idx] for i in s_magnitude_arr]), ti, alpha_p, freqs_hz, rician_loss, 2) for idx in selected_indices
        )
    out_l = []
    for sigma_ in results:
        if sigma_ is None:
            continue
        if sigma_ > 0:
            out_l.append(sigma_)

    out = np.array(out_l)
    return out * rician_over_estimation, out.mean() * rician_over_estimation  # Overestimate from the paper


def estimate_rician_sigma_from_guess(
    s_magnitude_arr: list[np.ndarray],
    water_guess: np.ndarray,
    fat_guess: np.ndarray,
    MagneticFieldStrength=3.0,
    ti_ms: list[float] | np.ndarray | None = None,
    alpha_p=None,
    freqs_ppm=None,
    sigma=2,
    subset_size=None,
    rician_over_estimation=1.163,
):
    if ti_ms is None:
        ti_ms = ti_ms_default
    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]
    shape = s_magnitude_arr[0].shape
    ti = np.array(ti_ms)
    freqs_hz = get_freqs_hz(freqs_ppm, MagneticFieldStrength)
    # Generate random subset of indices if subset_fraction < 1.0
    all_indices = list(np.ndindex(shape))
    if subset_size is not None and subset_size > np.prod(shape):
        subset_size = None
    selected_indices = random.sample(all_indices, subset_size) if subset_size is not None else all_indices

    with tqdm_joblib(tqdm(desc="Processing voxels", total=len(selected_indices))):
        results = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
            delayed(_refine_prediction)(
                idx,
                np.array([i[idx] for i in s_magnitude_arr]),
                water_guess[idx],
                fat_guess[idx],
                50,
                ti,
                alpha_p,
                freqs_hz,
                sigma,
            )
            for idx in selected_indices
        )

    out_l = []
    # Assign results back to output arrays
    for p_w, p_f, r2s, sigma, idx in results:
        if sigma > 0:
            out_l.append(sigma)

    out = np.array(out_l)
    return out * rician_over_estimation, out.mean() * rician_over_estimation  # Overestimate from the paper
