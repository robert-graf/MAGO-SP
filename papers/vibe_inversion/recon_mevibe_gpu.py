"""GPU (PyTorch) batched multi-peak fat model fits.

Mirrors the two entry points used by `pipeline.recon_fat_water_model`:

- :func:`multipeak_fat_model_from_guess_torch` — mirrors
  :func:`recon_mevibe.multipeak_fat_model_from_guess`.
- :func:`multipeak_fat_model_smooth_torch` — mirrors
  :func:`recon_mevibe.multipeak_fat_model_smooth` (two-initial-guess fit,
  loss-smoothed selection).

Both accept the same peak model kwargs (`alpha_p`, `freqs_ppm`, `MagneticFieldStrength`)
and the same echo-time list (`ti_ms`, in seconds — same unit convention as
the CPU path in `recon_mevibe.py`; the ms→s conversion is done by
`pipeline.pipeline` before this function is reached).

Design:
    - All voxels are packed into `S ∈ (Nvox, Necho)` and fit in one Adam call.
    - Autograd handles the Jacobian; no manual derivative code — makes it easy
      to keep the loss function in sync with the CPU version (Gaussian by
      default, Rician log-likelihood when `rician_loss=True`).
    - Two initial guesses per voxel (fat-dominant vs water-dominant) are run
      as two batched passes; the lower-loss branch wins per voxel, matching
      the CPU `multipeak_fat_model_smooth` behavior.
    - Parameters are clamped to [0, 1000] each step, same range as the CPU
      `dogbox` bounded fit.
"""

from __future__ import annotations

import numpy as np
import torch

from papers.vibe_inversion.recon_mevibe import alpha_p as _DEFAULT_ALPHA_P
from papers.vibe_inversion.recon_mevibe import freqs_ppm as _DEFAULT_FREQS_PPM
from papers.vibe_inversion.recon_mevibe import (
    get_freqs_hz,
    smooth_gaussian,
    ti_ms_default,
)


def _rician_neg_loglik_torch(s: torch.Tensor, r: torch.Tensor, sigma: float) -> torch.Tensor:
    """Rician negative log-likelihood, same math as the CPU `RicianLogLik`.

    Both `s` and `r` are (Nvox, Necho) magnitudes. Returns a tensor of the
    same shape (per-echo per-voxel neg-log-lik); the caller sums it.
    """
    eps = 1e-10
    sigma_ = max(float(sigma), eps)
    s2 = sigma_ * sigma_
    sumsqsc = (s * s + r * r) / (2.0 * s2)
    scp = s * r / s2
    # log I0(x): stable evaluation. For small/mid scp, torch.special.i0; for
    # very large scp, use the asymptotic log I0(x) ~ x - 0.5*log(2*pi*x).
    scp_safe = torch.clamp(scp, min=eps)
    lb0 = torch.where(
        scp < 700.0,
        torch.log(torch.special.i0(scp) + eps),
        scp - 0.5 * torch.log(2.0 * torch.pi * scp_safe),
    )
    s_safe = torch.where(s == 0, torch.full_like(s, eps), s)
    log_pdf = torch.log(s_safe) - np.log(s2) - sumsqsc + lb0
    return -log_pdf


def _predict(theta: torch.Tensor, ti_row: torch.Tensor, fr_row: torch.Tensor, fi_row: torch.Tensor) -> torch.Tensor:
    """Batched magnitude signal model. `theta[:, 0]=p_w, 1=p_f, 2=R2*`."""
    p_w = theta[:, 0:1]
    p_f = theta[:, 1:2]
    r2s = theta[:, 2:3]
    re = p_w + p_f * fr_row
    im = p_f * fi_row
    mag = torch.sqrt(re * re + im * im + 1e-12)
    return mag * torch.exp(-r2s * ti_row)


def _fit_batch(
    s_target: torch.Tensor,
    ti: torch.Tensor,
    f_re: torch.Tensor,
    f_im: torch.Tensor,
    theta_init: torch.Tensor,
    rician: bool,
    sigma: float,
    n_iter: int,
    lr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batched Adam fit. Returns (fitted theta, per-voxel final loss).

    NaN-hardening applied throughout so downstream stitching / PDFF math
    doesn't receive garbage:
      - Non-finite input magnitudes (rare, but happen for reconstructed
        priors) are zeroed once.
      - No hard upper clamp on p_w/p_f during optimisation — scanner
        magnitudes routinely exceed 1000, so the previous per-step
        clamp([0, 1000]) collapsed real signal to a flat 1000 and made
        Adam diverge into NaN. We only enforce non-negativity per step.
      - Non-finite gradients are zeroed so a single blown-up voxel
        doesn't poison neighbouring parameters via Adam's moment buffers.
      - Voxels that still ended non-finite are reverted to their init.
      - Per-voxel final loss uses ``inf`` for reverted voxels so the
        two-guess selector in the caller picks the other branch.
    """
    theta_init = torch.nan_to_num(theta_init, nan=0.0, posinf=0.0, neginf=0.0)
    theta = theta_init.clone().detach().requires_grad_(True)
    s_target = torch.nan_to_num(s_target, nan=0.0, posinf=0.0, neginf=0.0)
    opt = torch.optim.Adam([theta], lr=lr)
    ti_row = ti.unsqueeze(0)
    fr_row = f_re.unsqueeze(0)
    fi_row = f_im.unsqueeze(0)
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        pred = _predict(theta, ti_row, fr_row, fi_row)
        if rician:
            loss = _rician_neg_loglik_torch(s_target, pred, sigma).sum()
        else:
            loss = ((pred - s_target) ** 2).sum()
        if not torch.isfinite(loss):
            # Loss blew up — leave theta at its last finite state, bail.
            break
        loss.backward()
        if theta.grad is not None:
            theta.grad = torch.nan_to_num(theta.grad, nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        with torch.no_grad():
            # Only non-negativity is a hard physical constraint. NO upper
            # cap — scanner magnitudes are commonly in the thousands.
            theta.clamp_(min=0.0)
            # Belt-and-braces: kill any NaN that snuck through moment buffers.
            theta.copy_(torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0))
    with torch.no_grad():
        bad = ~torch.isfinite(theta).all(dim=-1, keepdim=True)
        if bad.any():
            theta = torch.where(bad.expand_as(theta), theta_init, theta)
        pred = _predict(theta, ti_row, fr_row, fi_row)
        loss_per_voxel = ((pred - s_target) ** 2).sum(dim=-1)
        loss_per_voxel = torch.where(
            torch.isfinite(loss_per_voxel),
            loss_per_voxel,
            torch.full_like(loss_per_voxel, float("inf")),
        )
    return theta.detach(), loss_per_voxel


def _prep(
    s_magnitude_arr: list[np.ndarray],
    magnetic_field_strength: float,
    ti_ms: list[float] | np.ndarray,
    alpha_p: np.ndarray,
    freqs_ppm: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    n_echo = len(s_magnitude_arr)
    shape = s_magnitude_arr[0].shape
    ti_np = np.asarray(ti_ms, dtype=np.float64)[:n_echo]
    ti = torch.as_tensor(ti_np, dtype=torch.float32, device=device)
    freqs_hz = get_freqs_hz(np.asarray(freqs_ppm, dtype=np.float64), magnetic_field_strength)
    # F(t_e) = sum_p alpha_p * exp(1j·2π·f_p·t_e), complex
    f_complex = (np.asarray(alpha_p, dtype=np.float64)[:, None] * np.exp(1j * 2.0 * np.pi * freqs_hz[:, None] * ti_np[None, :])).sum(axis=0)
    f_re = torch.as_tensor(f_complex.real.astype(np.float32), device=device)
    f_im = torch.as_tensor(f_complex.imag.astype(np.float32), device=device)
    s_stack = np.stack([a.astype(np.float32, copy=False) for a in s_magnitude_arr])
    s_tensor = torch.as_tensor(s_stack, device=device).reshape(n_echo, -1).T.contiguous()
    return s_tensor, ti, f_re, f_im, shape


def _low_signal_mask(s_magnitude_arr: list[np.ndarray], threshold: float = 20.0) -> np.ndarray:
    """Mirrors `_process_voxel`'s `s.sum() <= 20` skip: voxels below stay zero."""
    return sum(a.astype(np.float32) for a in s_magnitude_arr) <= threshold


def multipeak_fat_model_from_guess_torch(
    s_magnitude_arr: list[np.ndarray],
    water_guess: np.ndarray,
    fat_guess: np.ndarray,
    MagneticFieldStrength: float = 3.0,
    ti_ms: list[float] | np.ndarray | None = None,
    alpha_p: np.ndarray = _DEFAULT_ALPHA_P,
    freqs_ppm: np.ndarray = _DEFAULT_FREQS_PPM,
    rician_loss: bool = True,
    sigma: float = 2.0,
    device: str = "cuda",
    n_iter: int = 100,
    lr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """GPU-batched refinement fit from a water/fat prior; mirrors the CPU signature."""
    if ti_ms is None:
        ti_ms = ti_ms_default
    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]
    dev = torch.device(device)
    s_tensor, ti, f_re, f_im, shape = _prep(s_magnitude_arr, MagneticFieldStrength, ti_ms, alpha_p, freqs_ppm, dev)
    n_vox = s_tensor.shape[0]
    w0 = torch.as_tensor(water_guess.astype(np.float32, copy=False).reshape(-1), device=dev)
    f0 = torch.as_tensor(fat_guess.astype(np.float32, copy=False).reshape(-1), device=dev)
    r0 = torch.full((n_vox,), 100.0, dtype=torch.float32, device=dev)
    theta_init = torch.stack([w0, f0, r0], dim=-1).contiguous()
    theta, _ = _fit_batch(s_tensor, ti, f_re, f_im, theta_init, rician_loss, sigma, n_iter, lr)
    w = np.nan_to_num(theta[:, 0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    f = np.nan_to_num(theta[:, 1].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    r = np.nan_to_num(theta[:, 2].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    out_w = np.clip(w, 0, None).reshape(shape)
    out_f = np.clip(f, 0, None).reshape(shape)
    out_r = np.clip(r * 10.0, 0, None).reshape(shape).astype(np.int16)
    low = _low_signal_mask(s_magnitude_arr)
    out_w[low] = 0
    out_f[low] = 0
    out_r[low] = 0
    return out_w, out_f, out_r, None


def multipeak_fat_model_smooth_torch(
    s_magnitude_arr: list[np.ndarray],
    MagneticFieldStrength: float = 3.0,
    ti_ms: list[float] | np.ndarray | None = None,
    smooth: bool = True,
    sigma_smooth: float = 1,
    alpha_p: np.ndarray = _DEFAULT_ALPHA_P,
    freqs_ppm: np.ndarray = _DEFAULT_FREQS_PPM,
    factor: float = 1.0,
    rician_loss: bool = True,
    sigma_rician: float = 2,
    device: str = "cuda",
    n_iter: int = 100,
    lr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GPU-batched two-guess fit; mirrors the CPU `multipeak_fat_model_smooth`."""
    if ti_ms is None:
        ti_ms = ti_ms_default
    if len(ti_ms) != len(s_magnitude_arr):
        ti_ms = ti_ms[: len(s_magnitude_arr)]
    dev = torch.device(device)
    s_tensor, ti, f_re, f_im, shape = _prep(s_magnitude_arr, MagneticFieldStrength, ti_ms, alpha_p, freqs_ppm, dev)
    n_vox = s_tensor.shape[0]
    r2_init = 100.0
    theta_a = torch.tensor([[0.0, 1000.0, r2_init]], dtype=torch.float32, device=dev).expand(n_vox, 3).contiguous()
    theta_b = torch.tensor([[1000.0, 0.0, r2_init]], dtype=torch.float32, device=dev).expand(n_vox, 3).contiguous()
    theta_a, l_a = _fit_batch(s_tensor, ti, f_re, f_im, theta_a, rician_loss, sigma_rician, n_iter, lr)
    theta_b, l_b = _fit_batch(s_tensor, ti, f_re, f_im, theta_b, rician_loss, sigma_rician, n_iter, lr)
    w_a = np.nan_to_num(theta_a[:, 0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    f_a = np.nan_to_num(theta_a[:, 1].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    r_a = np.nan_to_num(theta_a[:, 2].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    w_b = np.nan_to_num(theta_b[:, 0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    f_b = np.nan_to_num(theta_b[:, 1].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    r_b = np.nan_to_num(theta_b[:, 2].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    # Losses may be `inf` for voxels the fit gave up on; keep them finite for
    # the smoothing / max comparison downstream.
    max_finite = float(np.finfo(np.float32).max)
    l1 = np.nan_to_num(l_a.cpu().numpy().reshape(shape), nan=max_finite, posinf=max_finite, neginf=max_finite)
    l2 = np.nan_to_num(l_b.cpu().numpy().reshape(shape), nan=max_finite, posinf=max_finite, neginf=max_finite)
    if smooth:
        l1 = smooth_gaussian(l1, sigma=sigma_smooth, truncate=3) * factor
        l2 = smooth_gaussian(l2, sigma=sigma_smooth, truncate=3)
    msk = (l2 > l1).astype(np.float32)
    out_w = w_a.reshape(shape) * msk + w_b.reshape(shape) * (1 - msk)
    out_f = f_a.reshape(shape) * msk + f_b.reshape(shape) * (1 - msk)
    out_r = (r_a.reshape(shape) * msk + r_b.reshape(shape) * (1 - msk)).astype(np.int16)
    out_l = l1 * msk + l2 * (1 - msk)
    low = _low_signal_mask(s_magnitude_arr)
    out_w[low] = 0
    out_f[low] = 0
    out_r[low] = 0
    out_l[low] = 0
    return out_w, out_f, out_r, out_l
