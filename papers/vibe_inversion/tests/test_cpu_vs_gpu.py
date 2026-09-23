"""test_cpu_vs_gpu.py — CPU vs GPU convergence check for MAGO / MAGORINO.

Pulls a small ROI of voxels out of one NAKO MEVIBE volume, runs the CPU fit
(`multipeak_fat_model_smooth` / `multipeak_fat_model_from_guess`) and the
batched torch fit (`multipeak_fat_model_smooth_torch` /
`multipeak_fat_model_from_guess_torch`) side-by-side with identical inputs,
and reports how close the water / fat / R2* / PDFF maps come out.

Also writes ``sample_points.txt`` — a portable text dump of the sample
voxels (echoes, TIs, field strength, priors, peak model) so the same
comparison can be re-run somewhere else (e.g. on a machine with Philips
data where the GPU path fails), without needing the NAKO source or TPTBox.

Run:
    python -m papers.vibe_inversion.tests.test_cpu_vs_gpu

The NAKO subject / ROI can be swapped via CLI flags — see ``--help``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[3]))

from papers.vibe_inversion.recon_mevibe import (
    gyromagnetic_ratio,
    multipeak_fat_model_from_guess,
    multipeak_fat_model_smooth,
    ti_ms_default,
)
from papers.vibe_inversion.recon_mevibe_gpu import (
    multipeak_fat_model_from_guess_torch,
    multipeak_fat_model_smooth_torch,
)

# Hamilton 9-peak liver — the current default (see recon_mevibe.py).
HAMILTON_FREQS_PPM = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
HAMILTON_ALPHA_P = np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037])

SIEMENS_B0 = 123.2400047 / gyromagnetic_ratio  # ~2.8936 T, from the NAKO json

NAKO_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako/rawdata")
MEVIBE_ECHO_PARTS = ("eco0-opp1", "eco1-pip1", "eco2-opp2", "eco3-in1", "eco4-pop1", "eco5-arb1")


def load_nako_sample(sub: int = 100000, sequ: str = "me1") -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return (list of 6 echo arrays, water, fat) for one NAKO MEVIBE volume."""
    import nibabel as nib

    d = NAKO_ROOT / str(sub)[:3] / str(sub) / "mevibe"
    echoes = [
        nib.load(d / f"sub-{sub}_sequ-{sequ}_acq-ax_part-{p}_mevibe.nii.gz").get_fdata().astype(np.float32) for p in MEVIBE_ECHO_PARTS
    ]
    water = nib.load(d / f"sub-{sub}_sequ-{sequ}_acq-ax_part-water_mevibe.nii.gz").get_fdata().astype(np.float32)
    fat = nib.load(d / f"sub-{sub}_sequ-{sequ}_acq-ax_part-fat_mevibe.nii.gz").get_fdata().astype(np.float32)
    return echoes, water, fat


def carve_roi(
    echoes: list[np.ndarray],
    water: np.ndarray,
    fat: np.ndarray,
    n: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pick `n` voxels with meaningful signal, return them shaped (n, 1, 1) so the
    volumetric fit APIs work unchanged. Also returns the original indices."""
    rng = np.random.default_rng(seed)
    total = sum(a.astype(np.float32) for a in echoes)
    keep = np.argwhere((total > 100) & np.isfinite(total))
    idx = keep[rng.choice(keep.shape[0], size=n, replace=False)]
    zs, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
    echoes_r = [e[zs, ys, xs].reshape(n, 1, 1) for e in echoes]
    water_r = water[zs, ys, xs].reshape(n, 1, 1)
    fat_r = fat[zs, ys, xs].reshape(n, 1, 1)
    return echoes_r, water_r, fat_r, (zs, ys, xs)


def _pdff(w: np.ndarray, f: np.ndarray) -> np.ndarray:
    tot = w + f
    out = np.zeros_like(tot, dtype=np.float32)
    m = tot > 1e-6
    out[m] = f[m] / tot[m]
    return out


def _stats(cpu: np.ndarray, gpu: np.ndarray, name: str) -> str:
    cpu = np.asarray(cpu, dtype=np.float64).ravel()
    gpu = np.asarray(gpu, dtype=np.float64).ravel()
    d = gpu - cpu
    ad = np.abs(d)
    scale = np.abs(cpu).max() or 1.0
    lines = [
        f"[{name}] n={cpu.size}",
        f"  cpu:  min={cpu.min():.4g}  max={cpu.max():.4g}  mean={cpu.mean():.4g}",
        f"  gpu:  min={gpu.min():.4g}  max={gpu.max():.4g}  mean={gpu.mean():.4g}",
        f"  diff: mean_abs={ad.mean():.4g}  median_abs={np.median(ad):.4g}  max_abs={ad.max():.4g}  (rel max={ad.max() / scale:.3%})",
    ]
    return "\n".join(lines)


def compare_smooth(
    echoes_r: list[np.ndarray],
    *,
    use_rician: bool,
    sigma_rician: float,
    device: str,
    gpu_iters: int,
    gpu_lr: float,
) -> dict[str, np.ndarray]:
    """Run smooth (two-guess) fit on CPU + GPU; return the flattened maps."""
    tag = "MAGORINO" if use_rician else "MAGO"
    print(f"\n=== {tag} — smooth (two-guess) fit ===")
    print("running CPU ...")
    w_c, f_c, r_c, _ = multipeak_fat_model_smooth(
        echoes_r,
        MagneticFieldStrength=SIEMENS_B0,
        ti_ms=ti_ms_default,
        smooth=False,
        alpha_p=HAMILTON_ALPHA_P,
        freqs_ppm=HAMILTON_FREQS_PPM,
        rician_loss=use_rician,
        sigma_rician=sigma_rician,
    )
    print("running GPU ...")
    w_g, f_g, r_g, _ = multipeak_fat_model_smooth_torch(
        echoes_r,
        MagneticFieldStrength=SIEMENS_B0,
        ti_ms=ti_ms_default,
        smooth=False,
        alpha_p=HAMILTON_ALPHA_P,
        freqs_ppm=HAMILTON_FREQS_PPM,
        rician_loss=use_rician,
        sigma_rician=sigma_rician,
        device=device,
        n_iter=gpu_iters,
        lr=gpu_lr,
    )
    print(_stats(w_c, w_g, f"{tag} smooth water"))
    print(_stats(f_c, f_g, f"{tag} smooth fat"))
    print(_stats(r_c, r_g, f"{tag} smooth R2* (int16*10)"))
    print(_stats(_pdff(w_c, f_c), _pdff(w_g, f_g), f"{tag} smooth PDFF"))
    return {"w_cpu": w_c.ravel(), "f_cpu": f_c.ravel(), "r_cpu": r_c.ravel(), "w_gpu": w_g.ravel(), "f_gpu": f_g.ravel(), "r_gpu": r_g.ravel()}


def compare_from_guess(
    echoes_r: list[np.ndarray],
    water_r: np.ndarray,
    fat_r: np.ndarray,
    *,
    use_rician: bool,
    sigma: float,
    device: str,
    gpu_iters: int,
    gpu_lr: float,
) -> dict[str, np.ndarray]:
    """Run from-guess (mago_sp-style) fit on CPU + GPU; return the flattened maps."""
    tag = "MAGO-SP-Rician" if use_rician else "MAGO-SP-Gauss"
    print(f"\n=== {tag} — from-guess fit ===")
    print("running CPU ...")
    w_c, f_c, r_c, _ = multipeak_fat_model_from_guess(
        echoes_r,
        water_guess=water_r,
        fat_guess=fat_r,
        MagneticFieldStrength=SIEMENS_B0,
        ti_ms=ti_ms_default,
        alpha_p=HAMILTON_ALPHA_P,
        freqs_ppm=HAMILTON_FREQS_PPM,
        rician_loss=use_rician,
        sigma=sigma,
    )
    print("running GPU ...")
    w_g, f_g, r_g, _ = multipeak_fat_model_from_guess_torch(
        echoes_r,
        water_guess=water_r,
        fat_guess=fat_r,
        MagneticFieldStrength=SIEMENS_B0,
        ti_ms=ti_ms_default,
        alpha_p=HAMILTON_ALPHA_P,
        freqs_ppm=HAMILTON_FREQS_PPM,
        rician_loss=use_rician,
        sigma=sigma,
        device=device,
        n_iter=gpu_iters,
        lr=gpu_lr,
    )
    print(_stats(w_c, w_g, f"{tag} water"))
    print(_stats(f_c, f_g, f"{tag} fat"))
    print(_stats(r_c, r_g, f"{tag} R2* (int16*10)"))
    print(_stats(_pdff(w_c, f_c), _pdff(w_g, f_g), f"{tag} PDFF"))
    return {"w_cpu": w_c.ravel(), "f_cpu": f_c.ravel(), "r_cpu": r_c.ravel(), "w_gpu": w_g.ravel(), "f_gpu": f_g.ravel(), "r_gpu": r_g.ravel()}


def write_sample_txt(
    path: Path,
    echoes_r: list[np.ndarray],
    water_r: np.ndarray,
    fat_r: np.ndarray,
    idx: tuple[np.ndarray, np.ndarray, np.ndarray],
    sub: int,
    sequ: str,
    smooth_res: dict[str, np.ndarray] | None,
    guess_res: dict[str, np.ndarray] | None,
) -> None:
    """Portable text dump. Header block carries every constant needed to
    reproduce the fit (TIs, field strength, peak model, sigma). Body has one
    row per voxel with the 6 echo magnitudes, the water/fat prior, the
    source (z,y,x) index, and — when provided — the CPU + GPU fit outputs.
    """
    zs, ys, xs = idx
    n = zs.size
    echoes_flat = np.stack([e.reshape(n) for e in echoes_r], axis=1)  # (n, 6)
    water_flat = water_r.reshape(n)
    fat_flat = fat_r.reshape(n)

    lines = [
        f"# CPU-vs-GPU MAGO / MAGORINO sample dump — {n} voxels",
        f"# source: NAKO sub-{sub} sequ-{sequ} (mevibe)",
        "# Hamilton 9-peak liver, Siemens 3T (via ImagingFrequency)",
        f"# MagneticFieldStrength_T = {SIEMENS_B0:.10f}",
        f"# gyromagnetic_ratio_MHz_per_T = {gyromagnetic_ratio}",
        f"# ti_ms = {ti_ms_default.tolist()}   # already in seconds inside the code",
        f"# freqs_ppm = {HAMILTON_FREQS_PPM.tolist()}",
        f"# alpha_p = {HAMILTON_ALPHA_P.tolist()}",
        "#",
        "# Columns:",
        "#   idx  z  y  x  eco0 eco1 eco2 eco3 eco4 eco5  water_prior fat_prior",
    ]
    header_extra = []
    if smooth_res is not None:
        header_extra += ["mago_w_cpu mago_f_cpu mago_r_cpu  mago_w_gpu mago_f_gpu mago_r_gpu"]
    if guess_res is not None:
        header_extra += ["sp_w_cpu sp_f_cpu sp_r_cpu  sp_w_gpu sp_f_gpu sp_r_gpu"]
    if header_extra:
        lines[-1] = lines[-1] + "  " + "  ".join(header_extra)
    lines.append("#")

    for i in range(n):
        row = [
            str(i),
            str(int(zs[i])),
            str(int(ys[i])),
            str(int(xs[i])),
            *(f"{v:.4f}" for v in echoes_flat[i]),
            f"{water_flat[i]:.4f}",
            f"{fat_flat[i]:.4f}",
        ]
        if smooth_res is not None:
            for k in ("w_cpu", "f_cpu", "r_cpu", "w_gpu", "f_gpu", "r_gpu"):
                row.append(f"{smooth_res[k][i]:.4f}")
        if guess_res is not None:
            for k in ("w_cpu", "f_cpu", "r_cpu", "w_gpu", "f_gpu", "r_gpu"):
                row.append(f"{guess_res[k][i]:.4f}")
        lines.append("\t".join(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {n}-voxel sample dump to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", type=int, default=100000, help="NAKO subject id")
    ap.add_argument("--sequ", default="me1", help="MEVIBE sequ tag")
    ap.add_argument("--n", type=int, default=200, help="number of voxels to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="torch device for the GPU fit")
    ap.add_argument("--gpu-iters", type=int, default=100)
    ap.add_argument("--gpu-lr", type=float, default=0.5)
    ap.add_argument("--sigma-rician", type=float, default=16.0)
    ap.add_argument("--skip-magorino", action="store_true")
    ap.add_argument("--skip-mago-sp", action="store_true")
    ap.add_argument(
        "--sweep-iters",
        type=int,
        nargs="*",
        default=None,
        help="Also re-run the GPU smooth MAGO fit at each of these gpu-iter values (compares PDFF back to CPU each time).",
    )
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "sample_points.txt")
    args = ap.parse_args()

    print(f"loading NAKO sub-{args.sub} sequ-{args.sequ} ...")
    echoes, water, fat = load_nako_sample(args.sub, args.sequ)
    echoes_r, water_r, fat_r, idx = carve_roi(echoes, water, fat, args.n, args.seed)
    print(f"sampled {args.n} voxels; echo shape {echoes_r[0].shape}")

    mago_res = compare_smooth(
        echoes_r,
        use_rician=False,
        sigma_rician=args.sigma_rician,
        device=args.device,
        gpu_iters=args.gpu_iters,
        gpu_lr=args.gpu_lr,
    )
    if not args.skip_magorino:
        _ = compare_smooth(
            echoes_r,
            use_rician=True,
            sigma_rician=args.sigma_rician,
            device=args.device,
            gpu_iters=args.gpu_iters,
            gpu_lr=args.gpu_lr,
        )
    if args.sweep_iters:
        print("\n=== GPU MAGO smooth iteration sweep (CPU reference reused from above) ===")
        w_c = mago_res["w_cpu"]
        f_c = mago_res["f_cpu"]
        pdff_ref = _pdff(w_c, f_c)
        for n_it in args.sweep_iters:
            w_g, f_g, _, _ = multipeak_fat_model_smooth_torch(
                echoes_r,
                MagneticFieldStrength=SIEMENS_B0,
                ti_ms=ti_ms_default,
                smooth=False,
                alpha_p=HAMILTON_ALPHA_P,
                freqs_ppm=HAMILTON_FREQS_PPM,
                rician_loss=False,
                sigma_rician=args.sigma_rician,
                device=args.device,
                n_iter=n_it,
                lr=args.gpu_lr,
            )
            pdff_gpu = _pdff(w_g.ravel(), f_g.ravel())
            d = np.abs(pdff_gpu - pdff_ref)
            print(f"  gpu_iters={n_it:>5}  PDFF |gpu-cpu|: mean={d.mean():.4f}  median={np.median(d):.4f}  max={d.max():.4f}")

    sp_res = None
    if not args.skip_mago_sp:
        sp_res = compare_from_guess(
            echoes_r,
            water_r,
            fat_r,
            use_rician=True,
            sigma=args.sigma_rician,
            device=args.device,
            gpu_iters=args.gpu_iters,
            gpu_lr=args.gpu_lr,
        )

    write_sample_txt(args.out, echoes_r, water_r, fat_r, idx, args.sub, args.sequ, mago_res, sp_res)


if __name__ == "__main__":
    main()
