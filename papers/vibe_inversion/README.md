# MAGO-SP: Detection and Correction of Water-Fat Swaps in Magnitude-Only VIBE MRI

The trained networks can be downloaded automatically.

You find the MAGO-SP code in /papers/vibe_inversion

## Installation

TODO

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
from pipeline import detect_inversion_seg,make_swap_statistic_single
# Paths
name = "example_subject"
out_phase = "PATH to out-phase"
in_phase =  "PATH to in-phase"
water_image = "Path to water image (not the PDWF)"
fat_image = "Path to fat image (not the PDFF)"
# Compute detection
water_detection, fat_detection = detect_inversion_seg(out_phase, in_phase, water_image, fat_image, out_detection_water="water_msk.nii.gz", out_detection_fat="fat_msk.nii.gz", override=False, ddevice="cuda", gpu=0)
swap_static = make_swap_statistic_single(name, water_detection, fat_detection, total_vibe=None)

print(f"{swap_static.percent*100:.2f} % of the image is swapped")    
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
    s_magnitude: Sequence[Image_Reference], # Path to nii files
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
from pipeline import pipeline_bids,pipeline

#pipeline_bids automatic generates BIDS names, if the input name is BIDS compliant.

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
