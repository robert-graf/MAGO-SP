import os

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.optimize import least_squares
from scipy.special import i0
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# Constants (adjust these based on your system and acquisition parameters)
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
# freqs_ppm = np.array([5.20, 4.21, 2.66, 2.00, 1.20, 0.80])
# freqs_ppm = np.array([5.30, 4.20, 2.75, 2.10, 1.30, 0.90])
# freqs_ppm = np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50])  # Hernando et al.

alpha_p = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306])
# alpha_p = np.array([0.048, 0.039, 0.004, 0.128, 0.694, 0.087]) #UKBB
# alpha_p = np.array([0.047, 0.039, 0.006, 0.12, 0.7, 0.088]) #UKBB
# alpha_p = np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048])  # Hernando et al.
gyromagnetic_ratio = 42.577478518  # MHz/T for hydrogen
ti_ms_default = np.array([1.23, 2.46, 3.69, 4.92, 6.15, 7.38]) / 1000


@njit
def fat_model(ti, alpha_p, freqs_hz):
    """(sum over p: alpha_p*e^(j*2*PI*f_p*t_i))"""
    # fat_component
    return np.sum(alpha_p[:, None] * np.exp(1j * 2 * np.pi * freqs_hz[:, None] * ti), axis=0)


@njit
def fat_model_vibe(ti):
    """(sum over p: alpha_p*e^(j*2*PI*f_p*t_i))"""
    # fat_component
    return np.sum(np.exp(1j * ti), axis=0)


@njit
def signal_model_magnitude(ti, p_w, p_f, R2_star, alpha_p, freqs_hz):
    fat_component = fat_model(ti, alpha_p, freqs_hz)
    return np.abs(p_w + p_f * fat_component) * np.exp(-R2_star * ti)


@njit
def signal_model_magnitude_vibe(ti, p_w, p_f):
    fat_component = -1 if abs(ti - 1.23) < 0.1 else 1
    return np.abs(p_w + p_f * fat_component) * np.exp(ti)


@njit
def rss(p_w, p_f, s_magnitude: np.ndarray, ti, R2_star, alpha_p, freqs_hz):
    model_signal = signal_model_magnitude(ti, p_w, p_f, R2_star, alpha_p, freqs_hz)
    assert model_signal.shape == s_magnitude.shape, (model_signal.shape, s_magnitude.shape)
    return np.square(np.abs(s_magnitude) - np.abs(model_signal)).sum()


def loss_function(params, s_magnitude, ti, alpha_p, freqs_hz, rician_loss=True, sigma=2):
    p_w, p_f, R2_star = params
    model_signal = signal_model_magnitude(ti, p_w, p_f, R2_star, alpha_p, freqs_hz)
    if rician_loss:  # noqa: SIM108
        # MAGORINO https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29493
        res = -RicianLogLik(s_magnitude, model_signal, sigma=sigma)
    else:
        # MAGO https://pubmed.ncbi.nlm.nih.gov/30874334/
        res = np.abs(s_magnitude - model_signal)
    return res


def RicianLogLik(s_magnitude, model_signal, sigma):
    """
    Computes the log likelihood of the measurements given the model
    predictions for the Rician noise model with the noise standard
    deviation sigma.

    Parameters:
    -----------
    measurements : array-like, shape (N,)
        The N-by-1 array storing the measurements (inclusive of noise).

    predictions : array-like, shape (N,)
        The N-by-1 array, the same size as the measurements, storing the predictions computed from a model.

    sigma : array-like, shape (N,) or scalar
        The standard deviation of the Gaussian distributions underlying the Gaussian distribution.
        This can either be a single (positive) value or a N-by-1 array.

    Returns:
    --------
    logliks : array-like, shape (N,)
        The log likelihood for individual measurements.
    """
    epsilon = 0.0000000001

    sigma = max(sigma, epsilon)
    sigmaSquared = np.asarray(sigma**2)
    # sum of squared measurements and predictions, normalized by squared sigma(s) (halved)
    sumsqsc = (s_magnitude**2 + model_signal**2) / (2 * sigmaSquared)
    # product of measurements and predictions, normalized by squared sigma(s)
    scp = s_magnitude.astype(np.float32) * model_signal / sigmaSquared
    # logarithm of the product just computed (using Bessel function I0)
    scp2 = np.maximum(scp, epsilon)
    lb0 = np.where(scp < 700, np.log(i0(scp)), scp - 0.5 * np.log(2 * np.pi * scp2))
    # valid = s_magnitude > 0
    # return np.log(s_magnitude, out=np.zeros_like(s_magnitude), where=valid) - np.log(sigmaSquared) - sumsqsc + lb0
    s_magnitude[s_magnitude == 0] = epsilon  # Prevent log(0)
    return np.log(s_magnitude) - np.log(sigmaSquared) - sumsqsc + lb0


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
    guess = [*initial_guess, r2]
    p_w = 100000

    # try:
    if len(s_magnitude) >= 3:
        res = least_squares(loss_function, guess, args=(s_magnitude, ti, alpha_p, freqs_hz, rician_loss, sigma), method="lm")
        p_w, p_f, r2s = res.x
    # except ValueError:
    #    p_w = 10000000
    #    p_f = 10000000
    if p_w > 1000 or p_f > 1000:
        guess = [max(min(g, 1000.0), 0.0) for g in guess]
        # try:
        res = least_squares(loss_function, guess, args=(s_magnitude, ti, alpha_p, freqs_hz, rician_loss, sigma), bounds=((0, 0, 0), (1000, 1000, 1000)), method="dogbox")
        # except ValueError:
        #    print(f"({guess=}, {s_magnitude=}, {ti=}, {r2=}, {freqs_hz=})")
        #    return 0, 0, 0
        p_w, p_f, r2s = res.x
        if p_w >= 1000 or p_f >= 1000:
            return 0, 0, 0
    p_w = max(p_w, 0)
    p_f = max(p_f, 0)
    return p_w, p_f, r2s


def _process_voxel(idx, s_magnitude, ti, alpha_p, freqs_hz, rician_loss=True, sigma=2):
    if s_magnitude.sum() <= 20:
        return 0, 0, 0, 0, 0, 0, 0, 0, idx

    initial_guess = (0.0, 1000.0)
    p_w, p_f, r2s = optimize_voxel_magnitude(s_magnitude, ti, 100, initial_guess, alpha_p, freqs_hz, rician_loss=rician_loss, sigma=sigma)
    loss = rss(p_w, p_f, s_magnitude, ti, r2s, alpha_p, freqs_hz)

    # Try second initial guess
    initial_guess2 = (1000.0, 0.0)
    p_w2, p_f2, r2s2 = optimize_voxel_magnitude(s_magnitude, ti, 100, initial_guess2, alpha_p, freqs_hz, rician_loss=rician_loss, sigma=sigma)
    loss2 = rss(p_w2, p_f2, s_magnitude, ti, r2s2, alpha_p, freqs_hz)
    return abs(p_w), abs(p_f), abs(p_w2), abs(p_f2), loss, loss2, r2s, r2s2, idx


def _refine_prediction(idx, s_magnitude: np.ndarray, guess_water, guess_fat, r2s, ti, alpha_p, freqs_hz, rician_loss=True, sigma=2):
    if s_magnitude.sum() <= 20:
        return 0, 0, 0, idx
    initial_guess = (guess_water, guess_fat)
    p_w, p_f, r2s = optimize_voxel_magnitude(s_magnitude, ti, r2s, initial_guess, alpha_p, freqs_hz, rician_loss=rician_loss, sigma=sigma)
    # loss = rss(p_w, p_f, s_magnitude, ti, r2s, alpha_p, freqs_hz)
    return abs(p_w), abs(p_f), r2s, idx


def get_freqs_hz(freqs_ppm, MagneticFieldStrength=3.0):
    # hz Larmor frequency
    center_freq = gyromagnetic_ratio * MagneticFieldStrength  # * 1e6
    return freqs_ppm * center_freq  # Convert ppm to Hz


def get_freqs_ppm(freqs_hz, MagneticFieldStrength=3.0):
    # hz Larmor frequency
    center_freq = gyromagnetic_ratio * MagneticFieldStrength  # * 1e6
    return freqs_hz / center_freq  # Convert Hz to ppm


def smooth_gaussian(self: np.ndarray, sigma: float | list[float] | tuple[float], truncate: float = 4.0, nth_derivative=0):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(self, sigma, order=nth_derivative, cval=0, truncate=truncate)


def multipeak_fat_model_smooth(
    s_magnitude_arr: list[np.ndarray],
    MagneticFieldStrength=3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    smooth=True,
    sigma_smooth=1,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    factor=1.0,
    rician_loss=True,
    sigma_rician=2,
):
    assert len(s_magnitude_arr) > 1, len(s_magnitude_arr)
    if ti_ms is None:
        ti_ms = ti_ms_default

    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]

    shape = s_magnitude_arr[0].shape
    z = s_magnitude_arr[0]
    w1 = z * 0
    f1 = z * 0
    w2 = z * 0
    f2 = z * 0
    l1 = z * 0
    l2 = z * 0
    r1 = z * 0
    r2 = z * 0
    ti = np.array(ti_ms)

    # Use joblib to parallelize the process_voxel function
    freqs_hz = get_freqs_hz(freqs_ppm, MagneticFieldStrength)
    with tqdm_joblib(tqdm(desc="Processing voxels", total=np.prod(shape))):
        results = Parallel(n_jobs=os.cpu_count())(
            delayed(_process_voxel)(idx, np.array([i[idx] for i in s_magnitude_arr]), ti, alpha_p, freqs_hz, rician_loss, sigma_rician) for idx in np.ndindex(shape)
        )
    r1 = r1.astype(np.int16)
    r2 = r2.astype(np.int16)
    # Assign results back to output arrays
    for p_w, p_f, p_w2, p_f2, l1_, l2_, r2s, r2s2, idx in results:
        w1[idx] = p_w
        w2[idx] = p_w2
        f1[idx] = p_f
        f2[idx] = p_f2
        l1[idx] = l1_
        l2[idx] = l2_
        r1[idx] = r2s
        r2[idx] = r2s2
    if smooth:
        l1 = smooth_gaussian(l1, sigma=sigma_smooth, truncate=3) * factor
        l2 = smooth_gaussian(l2, sigma=sigma_smooth, truncate=3)
    msk = l1 * 0
    msk[l2 > l1] = 1
    out_w = w1 * msk + w2 * (-msk + 1)
    out_f = f1 * msk + f2 * (-msk + 1)
    out_r = r1 * msk + r2 * (-msk + 1)
    out_l = l1 * msk + l2 * (-msk + 1)
    return out_w, out_f, out_r, out_l


def multipeak_fat_model_from_guess(
    s_magnitude_arr: list[np.ndarray],
    water_guess: np.ndarray,
    fat_guess: np.ndarray,
    MagneticFieldStrength=3.0,
    ti_ms: list[float] | np.ndarray | None = ti_ms_default,
    alpha_p=alpha_p,
    freqs_ppm=freqs_ppm,
    rician_loss=True,
    sigma=2.0,
):
    if ti_ms is None:
        ti_ms = ti_ms_default

    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]

    shape = s_magnitude_arr[0].shape
    out_w = s_magnitude_arr[0] * 0
    out_f = s_magnitude_arr[0] * 0
    # out_l = s_magnitude_arr[0] * 0
    out_r = s_magnitude_arr[0] * 0
    r2s_arr = s_magnitude_arr[0] * 0  # + 100
    ti = np.array(ti_ms)  # TODO read from json

    # Use joblib to parallelize the process_voxel function
    freqs_hz = get_freqs_hz(freqs_ppm, MagneticFieldStrength)
    # n_jobs=32
    with tqdm_joblib(tqdm(desc="Processing voxels", total=np.prod(shape))):
        results = Parallel(n_jobs=max(os.cpu_count() - 1, 1))(
            delayed(_refine_prediction)(
                idx, np.array([i[idx] for i in s_magnitude_arr]), water_guess[idx], fat_guess[idx], r2s_arr[idx], ti, alpha_p, freqs_hz, rician_loss, sigma
            )
            for idx in np.ndindex(shape)
        )
    out_r = out_r.astype(np.int16)

    # Assign results back to output arrays
    for p_w, p_f, r2s, idx in results:
        out_w[idx] = max(p_w, 0)
        out_f[idx] = max(p_f, 0)
        out_r[idx] = max(r2s * 10, 0)
        # out_l[idx] = loss
    return out_w, out_f, out_r, None
