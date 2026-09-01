# MAGO-SP: Detection and Correction of Water-Fat Swaps in Magnitude-Only VIBE MRI

Paper (MICCAI 2025, open access): <https://papers.miccai.org/miccai-2025/paper/0269_paper.pdf>

The trained networks can be downloaded automatically.

You find the MAGO-SP code in /papers/vibe_inversion

## Installation

```
# Run this commands by coping in to the Terminal
# Recommended: make a virtual Python environment (example shows Anaconda)
conda create -n "TotalVibeSegmentator" python=3.11.0
conda activate TotalVibeSegmentator

# Install PyTorch that works with your GPU (follow instructions at https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install required Python packages
pip install TPTBox ruamel.yaml configargparse
pip install nnunetv2 
# We recommend the newest versions. Tested versions: TPTBox==1.6, ruamel.yaml==0.18.6, configargparse==1.7, nnunetv2==2.4.2

# If e. g. nnunetv2 does not work, try version 2.4.2
# Uninstall the current version and reinstall with the specified version
#pip uninstall nnunetv2
#pip install nnunetv2==2.4.2

# Download the scripts (they will be downloaded to your current folder)
git clone https://github.com/robert-graf/MAGO-SP.git
cd TotalVibeSegmentator
```

## Individual Functions

### MAGO like methods

if you want to use numpy-array or ISMRM (International Society for Magnetic Resonance in Medicine) fat-water toolbox data we made wrapper classes in mago_methods.py. 
```python
from mago_methods import (
    mago,
    mago_ISMRM,
    magorino,
    magorino_ISMRM,
    mago_sp,
    mago_sp_ISMRM,  # There is a boolen flag to use MAGO or MAGORINO
)
```

### Water fat swap detection

```python
from pipeline import detect_inversion_seg, make_swap_statistic_single

# Paths
name = "example_subject"
out_phase = "PATH to out-phase"
in_phase = "PATH to in-phase"
water_image = "Path to water image (not the PDWF)"
fat_image = "Path to fat image (not the PDFF)"
# Compute detection
water_detection, fat_detection = detect_inversion_seg(
    out_phase,
    in_phase,
    water_image,
    fat_image,
    out_detection_water="water_msk.nii.gz",
    out_detection_fat="fat_msk.nii.gz",
    override=False,
    ddevice="cuda",
    gpu=0,
)
swap_static = make_swap_statistic_single(name, water_detection, fat_detection, total_vibe=None)

print(f"{swap_static.percent * 100:.2f} % of the image is swapped")
```

```python
### RUN Only on a single reconstruction ###
from pipeline import run_nnunet

run_nnunet(
    i=[
        "Path to water/fat image (not the PDFF)",
        "PATH to out-phase",
        "PATH to in-phase",
    ],
    out_seg="OUT-PATH",
    override=False,
)
# You can run it via TotalVibeSegmentor (https://github.com/robert-graf/TotalVibeSegmentator) with run_TotalVibeSegmentator_multi.py [...] --dataset_id 282
```

### SIGNAL PRIOR

```python
from pipeline import predict_signal_prior

def predict_signal_prior(
    s_magnitude: Sequence[Image-Paths], # Path to nii files
    #Default: VIBE: outphase/inphase
    #       MEVIBE: 0 to 5
    out_signal_prior: str | Path | None = None,
    steps_signal_prior: int = 50,
    override: bool = False,
    gpu: int = 0,
    ddevice: str = "cuda",
):
    #if len(s_magnitude) == 6:
    #    return signal_prior_mevibe(s_magnitude, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    #if len(s_magnitude) == 2:
    #    return signal_prior_vibe(s_magnitude, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    #raise NotImplementedError(len(s_magnitude))

```

## Full Pipeline

```python
from pipeline import pipeline_bids, pipeline

# pipeline_bids automatic generates BIDS names, if the input name is BIDS compliant.

# Note:
#    ti_ms and s_magnitude must be in the same order.
#    Default: VIBE: outphase/inphase
#             MEVIBE: 0 to 5
"""
    Pipeline for water-fat separation and reconstruction from MRI data.
    This function processes MRI magnitude images to detect and correct for
    swapped or mismatched water and fat signals, applies deep learning models
    for predicting a prior, reconstructs the image with the prior, and evaluates the reconstructed data.

    If `len(s_magnitude) == 2`, VIBE reconstruction is used, while if `len(s_magnitude) == 6`, MEVIBE reconstruction is applied.
    Other lengths are possible, but the deep learning models are fixed in sizes.

    Parameters:
        s_magnitude (Sequence[Image_Reference]): List of magnitude image references,
            ordered according to `ti_ms`. For VIBE, this typically includes outphase and
            inphase images (in this order). For MEVIBE, the list may contain images from different echo times
            (e.g., from 0 to 5).
        water_image (Image_Reference): Reference to the initial water image.
        fat_image (Image_Reference): Reference to the initial fat image.
        out_reconstruction_water (str | Path): Path to save the reconstructed water image.
        out_reconstruction_fat (str | Path): Path to save the reconstructed fat image.
        out_reconstruction_r2s (str | Path): Path to save the reconstructed R2* image.
        total_vibe (str | Path | None, optional): Path to save total VIBE segmentation.
            Defaults to None.
        out_detection_water (str | Path | None, optional): Path to save detected water image.
            Defaults to None.
        out_detection_fat (str | Path | None, optional): Path to save detected fat image.
            Defaults to None.
        out_signal_prior (str | Path | None, optional): Path to save signal prior image.
            Defaults to None.
        out_reconstruction_loss (str | Path | None, optional): Path to save reconstruction loss image.
            Defaults to None.
        out_detection_water_reconstructed (str | Path | None, optional): Path to save detected
            water image after reconstruction. Defaults to None.
        out_detection_fat_reconstructed (str | Path | None, optional): Path to save detected
            fat image after reconstruction. Defaults to None.
        steps_signal_prior (int, optional): Number of steps for signal prior prediction.
            Defaults to 50.
        override (bool, optional): If True, overwrite existing outputs. Defaults to False.
        ddevice (Literal["cpu", "cuda", "mps"], optional): Device for computation
            ("cpu", "cuda", or "mps"). Defaults to "cuda".
        gpu (int, optional): GPU index to use for computation. Defaults to 0.
        threshold_swapped_voxels (int, optional): Threshold for swapped voxel count to
            trigger correction. Defaults to 100.
        threshold_disagree_voxels (int, optional): Threshold for disagreement voxel count
            to trigger correction. Defaults to 2000.
        ti_ms (list[float] | None, optional): List of echo times in milliseconds.
            Must correspond to the order of `s_magnitude`. Defaults to None.
        evaluate_reconstructed (bool, optional): If True, evaluate the reconstructed
            images for errors. Defaults to True.

    Returns:
        Result: A result object containing information about the processing steps, including:
            - original_swap_stat: Statistics on the initial detection of swaps.
            - needs_correction (bool): Indicates whether a correction was applied.
            - out_w_nii: Reconstructed water image (NIfTI format).
            - out_f_nii: Reconstructed fat image (NIfTI format).
            - out_r_nii: Reconstructed R2* image (NIfTI format).
            - out_l_nii: Reconstruction loss image (NIfTI format).
            - reconstructed_swap_stat: Statistics on the detection after reconstruction.
            - needs_manuel_intervention (bool): Indicates whether manual intervention
            is needed after reconstruction.

    Notes:
        - `s_magnitude` and `ti_ms` must be in the same order.
        - For VIBE, `s_magnitude` typically includes outphase and inphase images.
        - This function attempts to minimize manual intervention through automated correction.
    """
```

## Fat-peak models

MEVIBE reconstruction fits a multi-peak fat spectrum. Both `pipeline` and `pipeline_bids` accept the peak model through three kwargs, all passed straight through to `recon_fat_water_model` → `multipeak_fat_model_*`:

```python
pipeline_bids(
    s_magnitude=[eco0, eco1, eco2, eco3, eco4, eco5],
    water_image=water,
    fat_image=fat,
    ti_ms=[1.14, 1.99, 2.85, 3.70, 4.55, 5.41],  # milliseconds; converted to seconds internally
    MagneticFieldStrength=3.0,  # Tesla (used to convert ppm → Hz)
    freqs_ppm=numpy.array([...]),  # chemical shifts of each peak
    alpha_p=numpy.array([...]),  # relative amplitudes, same length as freqs_ppm
    use_gpu=True,  # opt-in PyTorch batched fit — see "GPU acceleration" below
)
```

- Both arrays must have the same length; `alpha_p` typically sums to ~1.
- The active default set in `recon_mevibe.py` is now **Zhong 7-peak** (liver). The **MAGO-SP paper** results were produced with **Ren marrow (9-peak)** — that set is still in the file, just as a commented alternate. Bring your own for liver, subcutaneous fat, brown adipose, phantom, or scanner-specific calibrations.

These are some examples for Water-Fat-Models we where using. Pick one that matches your tissue of interest, or drop in your own set. Each entry is either the active default in `papers/vibe_inversion/recon_mevibe.py` or a commented alternate in the same file / in `papers/vibe_inversion/notebook/estemate_rican_sigma.ipynb`.

| Name | Peaks | Validated on | `freqs_ppm` | `alpha_p` | Source |
|---|---|---|---|---|---|
| **Ren marrow** (MAGO-SP paper default) | 9 | bone marrow & subcutaneous adipose tissue | `[-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]` | `[0.08991, 0.58342, 0.05994, 0.08492, 0.05994, 0.01499, 0.03996, 0.00999, 0.05694]` | Ren et al., *J Lipid Res* 2008 — <https://doi.org/10.1194/jlr.D700041-JLR200>; `+0.05 ppm` shift variant used in `tests/run_mevibe_test.py` — <https://doi.org/10.1002/jmri.25453> |
| **Hamilton liver** | 9 | in vivo human liver | same 9 shifts as Ren | `[0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037]` | Hamilton et al., *NMR Biomed* 2011 — <https://doi.org/10.1002/nbm.1622> |
| **Hernando** | 6 | in vivo human liver (R2* / iron overload) | `[-3.9, -3.5, -2.7, -2.04, -0.49, 0.50]` | `[0.087, 0.694, 0.128, 0.004, 0.039, 0.048]` | Hernando et al., *MRM* 2013 — <https://doi.org/10.1002/mrm.24593> |
| **UKBB v1** | 6 | in vivo human liver (UK Biobank cohort) | `[5.20, 4.21, 2.66, 2.00, 1.20, 0.80]` | `[0.048, 0.039, 0.004, 0.128, 0.694, 0.087]` | MAGO — Triay Bagur et al., *MRM* 2019 — <https://doi.org/10.1002/mrm.27728> |
| **UKBB v2** | 6 | in vivo human liver (UK Biobank cohort) | same 6 shifts as UKBB v1 | `[0.047, 0.039, 0.006, 0.12, 0.7, 0.088]` | MAGO — Triay Bagur et al., *MRM* 2019 — <https://doi.org/10.1002/mrm.27728> |
| **Zhong 7-peak** (current default) | 7 | in vivo human liver (PDFF + R2*) | `[-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68]` | `[0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04]` | Zhong et al., *MRM* 2014 — <https://doi.org/10.1002/mrm.25054> |

> **Disclaimer:** Please check the paper yourself, to not miss details.

For a visual side-by-side comparison of the Water-Fat models above on 6-Point NAKO, see our ECR 2026 poster: <https://epos.myesr.org/poster/esr/ecr2026/C-24425>


### Sign convention gotcha

`freqs_ppm` is used inside `get_freqs_hz(freqs_ppm, MagneticFieldStrength)` and multiplied by the scanner center frequency directly. Two conventions exist in the wild — shifts relative to water (0 ppm) or relative to TMS (~4.7 ppm). The `Ren`/`Hamilton`/`Hernando`/`Zhong` sets above are **water-referenced** (mostly negative values, methyl-methylene near 0 ppm as small positive lobe). The `UKBB` and `Alt-6` sets are **TMS-referenced** — the `tests/run_mevibe_test.py` file subtracts `4.7` from them precisely to convert. If you drop a set in from a new paper without checking, you can silently get everything ~600 Hz off at 3 T.

### `use_rician`

Same layer also exposes `use_rician=True|False`. `True` (default) fits with a Rician log-likelihood — this is what makes the method "MAGO*RINO*"; `False` collapses back to Gaussian residual ("MAGO"). See `notebook/compare_fatmodel.ipynb` for the naming.

### `ti_ms` — echo times

`pipeline` and `pipeline_bids` both take `ti_ms: list[float] | None`. The values are echo times in **milliseconds** — converted to seconds by `pipeline` before being handed to `recon_fat_water_model`, because the signal model computes `exp(1j·2π·f_Hz·t)` and needs `t` in seconds. Order must match `s_magnitude`. `None` uses the built-in default (`recon_mevibe.ti_ms_default`, six echoes at 1.23–7.38 ms).

### Suppress structures from the swap analysis (`ignore_vibe_labels`)

The 3-channel swap detector (nnU-Net 282) can produce false positives in air-filled or fluid-only regions (lungs, trachea, spinal channel) where the water/fat contrast is degenerate. Pass a VIBESeg label id (or a list) to zero those voxels out **before** the swap counts (`count_water`, `count_fat`, `count_disagree`) and **before** the `affected_structures` report:

```python
pipeline_bids(
    ...,
    ignore_vibe_labels=[10, 11, 12, 13, 14, 16, 71],  # lungs, trachea, spinal channel
)
```

Works on both `pipeline` and `pipeline_bids`; only takes effect when a `total_vibe` segmentation is available (via `derivative_total` in `pipeline_bids`, or explicitly in `pipeline`). Label ids are from `TPTBox.segmentation.VibeSeg.vibeseg.VibeSeg_map` (1=spleen, 5=liver, 10-14=lung lobes, 16=trachea, 52=spinal_cord, 71=spinal_channel, 65=subcutaneous_fat, 66=muscle, 67=inner_fat, …). Accepts a single int or any `Sequence[int]`. Affects the reconstruction gate too — voxels ignored here don't push `needs_correction` past `threshold_swapped_voxels` / `threshold_disagree_voxels`.

### GPU acceleration (`use_gpu`)

The per-voxel `scipy.optimize.least_squares` fit is now optionally routed through a batched PyTorch implementation in `papers/vibe_inversion/recon_mevibe_gpu.py`. Set `use_gpu=True` on `pipeline` or `pipeline_bids` to enable it:

```python
pipeline_bids(
    ...,
    use_gpu=True,  # off by default
    gpu_device="cuda",  # or "cuda:1", "cpu", "mps"
    gpu_iters=100,  # Adam iterations
    gpu_lr=0.5,  # Adam learning rate
)
```

- Same signal model, same peak arrays, same Rician / Gaussian loss — the maths mirrors `multipeak_fat_model_from_guess` and `multipeak_fat_model_smooth` one-for-one.
- Two initial guesses (fat-dominant and water-dominant) run as two batched Adam passes; the lower-loss branch wins per voxel, matching the CPU behaviour.
- Parameters are clamped to `[0, 1000]` at every step (same range as the CPU `dogbox` bounded fit).
- Speedup grows with volume size — a ~10 M-voxel body-composition volume is minutes on CPU and seconds on GPU.
- Falls back cleanly to CPU by setting `use_gpu=False` (the default) or `gpu_device="cpu"`.

Reproducing the paper: leave `use_gpu=False`. The GPU path is a numerical rewrite (batched Adam vs. per-voxel Levenberg-Marquardt); results agree within noise on the phantom test but are not bit-exact. Not recommended when the exact CPU numbers are the deliverable.

### Configuring the peak model without going through the pipeline

If you want the fit but not the detection/DL layers, `papers/vibe_inversion/mago_methods.py` exposes standalone `mago(...)`, `magorino(...)`, `mago_sp(...)` (and `*_ISMRM` variants for ISMRM fat-water-toolbox test data). Each takes the same `alpha_p` / `freqs_ppm` / `MagneticFieldStrength` / `ti_ms` / `use_rician` kwargs and forwards them to `multipeak_fat_model_*`. Handy for benchmarking a new peak set on a single image without invoking the full pipeline.

The GPU fit is also usable without the pipeline. Import it directly:

```python
from papers.vibe_inversion.recon_mevibe_gpu import (
    multipeak_fat_model_from_guess_torch,  # refinement fit from a water/fat prior
    multipeak_fat_model_smooth_torch,  # two-guess fit + loss-smoothed selection
)
```

Both accept the same `alpha_p` / `freqs_ppm` / `MagneticFieldStrength` / `ti_ms` / `rician_loss` / `sigma` kwargs as the CPU functions, plus `device`, `n_iter`, `lr`.
