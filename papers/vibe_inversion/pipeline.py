import random
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from TPTBox import BIDS_FILE, NII, Image_Reference, to_nii
from TPTBox.segmentation.TotalVibeSeg import run_totalvibeseg, total_vibe_map
from TPTBox.segmentation.TotalVibeSeg.inference_nnunet import run_inference_on_file

sys.path.append(str(Path(__file__).parents[2]))
from papers.vibe_inversion.recon_mevibe import multipeak_fat_model_from_guess, multipeak_fat_model_smooth
from papers.vibe_inversion.recon_vibe import vibe_separate_phase_from_guess, vibe_separate_water_fat_from_guess
from papers.vibe_inversion.signal_prior import signal_prior_mevibe, signal_prior_vibe


def run_nnunet(i: list[Image_Reference], out_seg: str | Path | None, override=False, dataset_id=282, **args):
    return run_inference_on_file(dataset_id, [to_nii(i) for i in i], out_file=out_seg, override=override, **args)[0]


@dataclass
class Swap_statistic:
    """Represents an image with counts for water, fat, and disagreements (inversions),
    and calculates related percentages.
    """

    sub: str
    count_water: float
    count_fat: float  # water pixels that are segmented as fat
    count_disagree: float  # water and fat prediction disagree
    affected_structures: list[str] | None

    @property
    def percent(self) -> float:
        """Calculates the proportion of fat in the total water-fat composition."""
        total = self.count_fat + self.count_water
        if total == 0:
            return 1
        return self.count_fat / total

    @property
    def percent_dis(self) -> float:
        """Calculates the proportion of disagreements in the total composition."""
        total = self.count_fat + self.count_water + self.count_disagree
        return self.count_disagree / total


def detect_inversion_seg(
    inphase_file: Image_Reference,
    outphase_file: Image_Reference,
    water_image: Image_Reference,
    fat_image: Image_Reference,
    out_detection_water: str | Path | None = None,
    out_detection_fat: str | Path | None = None,
    override=False,
    ddevice="cuda",
    gpu=0,
) -> tuple[Image_Reference, Image_Reference]:
    """
    Detect inversion segmentation for water and fat images using nnUNet.

    Args:
        inphase_file (Image_Reference): Path to the in-phase image file.
        outphase_file (Image_Reference): Path to the out-phase image file.
        water_image (Image_Reference): Path to the water image file.
        fat_image (Image_Reference): Path to the fat image file.
        out_detection_water (str | Path | None): Output path for the water segmentation.
        out_detection_fat (str | Path | None): Output path for the fat segmentation.
        override (bool): If True, overwrites existing files.
        gpu (int): GPU ID for computation.

    Returns:
        tuple[Image_Reference, Image_Reference]: Paths to the segmented water and fat images.
    """
    # Ensure output paths are Path objects
    if out_detection_water is not None:
        out_detection_water = Path(out_detection_water)
    if out_detection_fat is not None:
        out_detection_fat = Path(out_detection_fat)

    # Skip processing if outputs exist and override is False
    if not override and out_detection_water and out_detection_fat and out_detection_water.exists() and out_detection_fat.exists():
        return out_detection_water, out_detection_fat
    # Run segmentation
    a, _ = run_nnunet([water_image, inphase_file, outphase_file], out_detection_water, override=override, dataset_id=282, gpu=gpu, ddevice=ddevice)
    b, _ = run_nnunet([fat_image, inphase_file, outphase_file], out_detection_fat, override=override, dataset_id=282, gpu=gpu, ddevice=ddevice)
    # Delete files if we
    return a, b


def make_swap_statistic_single(
    subj_name: str,
    seg1_water: Image_Reference,
    seg2_fat: Image_Reference,
    total_vibe: Image_Reference | None = None,
) -> Swap_statistic:
    """
    Identifies water-fat inversions in the segmentation, optionally using total VIBE segmentation.

    This function analyzes two segmentation files (`seg1_water` and `seg2_fat`) to detect
    regions where water-fat labels are inverted. If provided, a total VIBE segmentation
    can be used to identify affected structures.

    Args:
        subj_name (str): Subject identifier.
        seg1_water (Path): Path to the water segmentation file.
        seg2_fat (Path): Path to the fat segmentation file.
        total_vibe (Path | str | None): Optional path to the total VIBE segmentation file.
                                        If None, affected structures will not be analyzed.

    Returns:
        Swap_statistic: An object containing inversion counts and affected structures if applicable.
    """
    # Convert segmentations to NII format
    nii_water: NII = to_nii(seg1_water, seg=True)
    nii_fat: NII = to_nii(seg2_fat, seg=True)

    # Map labels in the fat segmentation: 1 <-> 2 (invert water-fat labels)
    nii_fat.map_labels_({1: 2, 2: 1}, verbose=False)

    # Ensure both segmentations have the same shape
    if nii_water.shape != nii_fat.shape:
        nii_fat.resample_from_to_(nii_water)

    # Create an inversion mask
    nii_fat[nii_water == 0] = 0  # Background remains as 0
    nii_water[nii_fat != nii_water] = 3  # Disagreement mask
    nii_water[nii_fat == 0] = 0  # Background remains as 0

    # Analyze affected structures if `total_vibe` is provided
    affected_structures = None
    if total_vibe:
        vibe_nii: NII = to_nii(total_vibe, seg=True)
        if nii_water.shape != vibe_nii.shape:
            vibe_nii.resample_from_to_(nii_water)

        # Retain only regions affected by label inversions
        disagreement_mask = nii_water.extract_label(2).erode_msk(1, verbose=False)
        vibe_nii[disagreement_mask != 1] = 0

        # Map unique labels to corresponding structures
        affected_structures = [total_vibe_map[label] for label in vibe_nii.unique() if label in total_vibe_map]

    # Calculate inversion statistics
    return Swap_statistic(
        sub=subj_name,
        count_water=nii_water.extract_label(1).sum(),
        count_fat=nii_water.extract_label(2).sum(),
        count_disagree=nii_water.extract_label(3).sum(),
        affected_structures=affected_structures,
    )


def predict_signal_prior(
    s_magnitude: Sequence[Image_Reference],
    out_signal_prior: str | Path | None = None,
    steps_signal_prior: int = 50,
    override: bool = False,
    gpu: int = 0,
    ddevice: str = "cuda",
):
    if len(s_magnitude) == 6:
        return signal_prior_mevibe(s_magnitude, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    if len(s_magnitude) == 2:
        return signal_prior_vibe(s_magnitude, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    raise NotImplementedError(len(s_magnitude))


def make_pdff_pdwf(water_image: Image_Reference, fat_image: Image_Reference, pdff_out: str | Path | None = None, pdwf_out: str | Path | None = None, override=False):
    if not override and pdff_out is not None and Path(pdff_out).exists() and pdwf_out is not None and Path(pdwf_out).exists():
        return pdff_out, pdwf_out
    fat = to_nii(fat_image)
    water = to_nii(water_image)
    water.set_dtype_()
    fat.set_dtype_()
    nii_pdff = fat / (water + fat)
    nii_pdff[water + fat == 0] = 0
    nii_pdff *= 1000
    nii_pdff.set_dtype_("smallest_int")
    nii_pdff.save(pdff_out) if pdff_out is not None else None
    nii_pdwf = water / (water + fat)
    nii_pdwf[water + fat == 0] = 0
    nii_pdwf *= 1000
    nii_pdwf.set_dtype_("smallest_int")
    nii_pdwf.save(pdwf_out) if pdwf_out is not None else None
    return nii_pdff, nii_pdwf


def recon_fat_water_model(
    s_magnitude: Sequence[Image_Reference],
    water_image: Image_Reference | None,
    fat_image: Image_Reference | None,
    signal_prior: Image_Reference | None,
    out_reconstruction_water: str | Path | None = None,
    out_reconstruction_fat: str | Path | None = None,
    out_reconstruction_r2s: str | Path | None = None,
    out_reconstruction_loss: str | Path | None = None,
    ti_ms: list[float] | None = None,
    override=False,
    vibe_from_signal=False,
):
    """
    MEVIBE:
    Applies a multipeak fat model to reconstruct water, fat, R2* (decay), and loss maps from MRI magnitude images.

    This function supports two types of reconstruction:
    - **VIBE reconstruction** (when `len(s_magnitude) == 2`), using inphase and outphase images, with an optional signal prior for water and fat separation.
    - **MEVIBE reconstruction** (when `len(s_magnitude) == 6`), using multiple echo times (ti_ms) and performing a multipeak fat model reconstruction.

    Parameters:
        s_magnitude (Sequence[Image_Reference]): A sequence of magnitude images used for the reconstruction.
            For VIBE, this typically includes outphase and inphase images. For MEVIBE, this should include multiple
            echo-time images (e.g., 0 to 5).
        water_image (Image_Reference): Water image (Used for computing Fat prior from water prior and for VIBE if vibe_from_signal=False)
        fat_image (Image_Reference): Fat image (Used for computing Fat prior from water prior and for VIBE if vibe_from_signal=False)
        signal_prior (Image_Reference | None): NII file containing the water signal prior, clamped to a valid range.
            Needed for VIBE reconstruction.
        out_reconstruction_water (Union[str, Path, None]): Path to save the reconstructed water map. If None, the map is not saved.
        out_reconstruction_fat (Union[str, Path, None]): Path to save the reconstructed fat map. If None, the map is not saved.
        out_reconstruction_r2s (Union[str, Path, None]): Path to save the reconstructed R2* map. If None, the map is not saved.
        out_reconstruction_loss (Union[str, Path, None]): Path to save the reconstruction loss map. If None, the map is not saved.
        ti_ms (list[float] | None, optional): List of echo times in milliseconds for MEVIBE. Defaults to None.
        override (bool, optional): If True, overwrite existing outputs. Defaults to False.
        vibe_from_signal (bool, optional): If True, use VIBE reconstruction from the signal prior. Defaults to False.

    Returns:
        Tuple[NII, NII, NII, NII]: A tuple containing the reconstructed water, fat, R2*, and loss maps as NII objects.

    Raises:
        AssertionError: If the length of `s_magnitude` is invalid or does not match the expected configuration.
        RuntimeError: If any of the reconstruction steps fail, such as during fat model processing.

    Notes:
        - For VIBE reconstruction, `signal_prior` is required and used to separate water and fat based on phase information.
        - For MEVIBE, if no `signal_prior` is provided, the function uses a multipeak fat model based on multiple echo times.
        - The function returns the reconstructed images as NIfTI objects and saves them if output paths are specified.
        - The order of `s_magnitude` and `ti_ms` must align.
        - If `vibe_from_signal` is enabled, the VIBE separation is done based on the signal prior.

    Example:
        out_w_nii, out_f_nii, out_r_nii, out_l_nii = multipeak_fat_model(
            s_magnitude=[inphase_image, outphase_image],
            water_image=water_image,
            fat_image=fat_image,
            signal_prior=water_signal_prior,
            out_reconstruction_water='water.nii',
            out_reconstruction_fat='fat.nii',
            out_reconstruction_r2s='r2s.nii',
            out_reconstruction_loss='loss.nii'
        )
    """
    # Validate the input
    assert len(s_magnitude) > 1, "s_magnitude must contain at least two magnitude image."
    if not override and all(
        i is not None and Path(i).exists() for i in [out_reconstruction_water, out_reconstruction_fat, out_reconstruction_r2s, out_reconstruction_loss]
    ):
        out_w_nii = NII.load(out_reconstruction_water, False) if out_reconstruction_water is not None else None
        out_f_nii = NII.load(out_reconstruction_fat, False) if out_reconstruction_fat is not None else None
        out_r_nii = NII.load(out_reconstruction_r2s, False) if out_reconstruction_r2s is not None else None
        out_l_nii = NII.load(out_reconstruction_loss, False) if out_reconstruction_loss is not None else None
        return out_w_nii, out_f_nii, out_r_nii, out_l_nii
    if len(s_magnitude) == 2 and not override and all(i is not None and Path(i).exists() for i in [out_reconstruction_water, out_reconstruction_fat]):
        out_w_nii = NII.load(out_reconstruction_water, False)
        out_f_nii = NII.load(out_reconstruction_fat, False)
        return out_w_nii, out_f_nii, None, None

    # Convert s_magnitude images to numpy arrays
    s_magnitude_arr = [to_nii(image).get_array() for image in s_magnitude]
    if len(s_magnitude) == 2:
        # VIBE
        assert signal_prior is not None, "signal_prior is needed for VIBE reconstruction on absolute values"
        grid = water_prior = to_nii(signal_prior).clamp(0, 1000)

        if vibe_from_signal:
            # Use inphase and outphase to compute water and fat
            out_w, out_f = vibe_separate_phase_from_guess(s_magnitude_arr[1].squeeze(), s_magnitude_arr[0].squeeze(), water_prior.get_array(), smooth=True)
        else:
            assert water_image is not None and fat_image is not None, "use vibe_from_signal=True if you do not want ot use water_image/fat_image"
            # Use water and fat and just swap pixels
            out_w, out_f = vibe_separate_water_fat_from_guess(to_nii(water_image), to_nii(fat_image), water_prior)
        out_r = None
        out_l = None
    elif signal_prior is None:
        # MEVIBE
        # Perform multipeak fat model reconstruction
        print("no signal prior")
        out_w, out_f, out_r, out_l = multipeak_fat_model_smooth(s_magnitude_arr, ti_ms=ti_ms)
        grid = to_nii(s_magnitude[0])  # just for the affine info
    else:
        # MEVIBE
        # Perform multipeak fat model reconstruction
        # Prepare water and fat priors

        grid = water_prior = to_nii(signal_prior).clamp(0, 1000)
        # water and fat should give proton density. So we can estimate fat_pror by 'fat_prior = pd - water_prior'
        assert water_image is not None and fat_image is not None, "Need water_image/fat_image for estimating fat prior"
        pd_sum = to_nii(water_image).get_array().astype(float) + to_nii(fat_image).get_array().astype(float)
        fat_prior = pd_sum - water_prior.get_array().astype(float)
        fat_prior[fat_prior < 0] = 0
        out_w, out_f, out_r, out_l = multipeak_fat_model_from_guess(s_magnitude_arr, water_prior.get_array(), fat_guess=fat_prior, ti_ms=ti_ms)

    out_w_nii = grid.set_array(out_w)
    out_f_nii = grid.set_array(out_f)
    out_r_nii = grid.set_array(out_r) if out_r is not None else None
    out_l_nii = grid.set_array(out_l) if out_l is not None else None
    # Save the output NII files if paths are provided
    out_w_nii.save(out_reconstruction_water) if out_reconstruction_water is not None else None
    out_f_nii.save(out_reconstruction_fat) if out_reconstruction_fat is not None else None
    if out_r_nii is not None:
        out_r_nii.save(out_reconstruction_r2s) if out_reconstruction_r2s is not None else None
    if out_l_nii is not None:
        out_l_nii.save(out_reconstruction_loss) if out_reconstruction_loss is not None else None
    # Return the reconstructed maps as NII objects
    return out_w_nii, out_f_nii, out_r_nii, out_l_nii


@dataclass
class Result:
    original_swap_stat: Swap_statistic | None = None
    needs_correction: bool = True
    reconstructed_swap_stat: Swap_statistic | None = None
    needs_manuel_intervention: bool = False
    out_w_nii: NII | None = None
    out_f_nii: NII | None = None
    out_r_nii: NII | None = None
    out_l_nii: NII | None = None


def pipeline(
    s_magnitude: Sequence[Image_Reference],  # same order as ti_ms
    # Note:
    #    ti_ms and s_magnitude must be in the same order.
    #    Default: VIBE: outphase/inphase
    #             MEVIBE: 0 to 5
    water_image: Image_Reference,
    fat_image: Image_Reference,
    out_reconstruction_water: str | Path,
    out_reconstruction_fat: str | Path,
    out_reconstruction_r2s: str | Path | None,
    out_reconstruction_pdwf: str | Path | None = None,
    out_reconstruction_pdff: str | Path | None = None,
    total_vibe: str | Path | None = None,
    out_detection_water: str | Path | None = None,
    out_detection_fat: str | Path | None = None,
    out_signal_prior: str | Path | None = None,
    out_reconstruction_loss: str | Path | None = None,
    out_detection_water_reconstructed: str | Path | None = None,
    out_detection_fat_reconstructed: str | Path | None = None,
    steps_signal_prior=50,
    override=False,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    gpu=0,
    threshold_swapped_voxels=100,
    threshold_disagree_voxels=2000,
    ti_ms: list[float] | None = None,
    evaluate_reconstructed=True,
    vibe_from_signal=True,
) -> Result:
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
    # TODO limit FOV to speed up correction

    # Final output exists
    # if not override and all(i is not None and Path(i).exists() for i in [out_reconstruction_water, out_reconstruction_fat, out_reconstruction_r2s]):
    #    return Result()
    if total_vibe:
        run_totalvibeseg(s_magnitude[1], total_vibe, gpu=gpu, ddevice=ddevice)
    # Compute detection
    water_detection, fat_detection = detect_inversion_seg(
        s_magnitude[0], s_magnitude[1], water_image, fat_image, out_detection_water, out_detection_fat, override, ddevice, gpu
    )
    swap_static = make_swap_statistic_single(str(out_reconstruction_water), water_detection, fat_detection, total_vibe)
    # count_fat is the amount of swapped pixels
    # count_disagree is the amount of pixels, where the detection disagrees
    needs_correction = swap_static.count_fat >= threshold_swapped_voxels or swap_static.count_disagree >= threshold_disagree_voxels
    if not needs_correction:
        return Result(original_swap_stat=swap_static, needs_correction=False)
    # Take s_magnitude to compute with DL a water image
    signal_prior = predict_signal_prior(s_magnitude, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    out_w_nii, out_f_nii, out_r_nii, out_l_nii = recon_fat_water_model(
        s_magnitude,
        water_image,
        fat_image,
        signal_prior,
        out_reconstruction_water,
        out_reconstruction_fat,
        out_reconstruction_r2s,
        out_reconstruction_loss,
        ti_ms=ti_ms,
        override=override,
        vibe_from_signal=vibe_from_signal,
    )
    if out_reconstruction_pdwf is not None or out_reconstruction_pdff is not None:
        make_pdff_pdwf(out_w_nii, out_f_nii, out_reconstruction_pdff, out_reconstruction_pdwf)
    if not evaluate_reconstructed:
        return Result(original_swap_stat=swap_static, needs_correction=True, out_w_nii=out_w_nii, out_f_nii=out_f_nii, out_r_nii=out_r_nii, out_l_nii=out_l_nii)
    # Compute detection
    water_detection, fat_detection = detect_inversion_seg(
        s_magnitude[0], s_magnitude[1], out_w_nii, out_f_nii, out_detection_water_reconstructed, out_detection_fat_reconstructed, override, ddevice, gpu
    )
    swap_static_rec = make_swap_statistic_single(str(out_reconstruction_water), water_detection, fat_detection, total_vibe)
    needs_manuel_intervention = swap_static_rec.count_fat >= threshold_swapped_voxels or swap_static_rec.count_disagree >= threshold_disagree_voxels
    return Result(
        original_swap_stat=swap_static,
        needs_correction=True,
        out_w_nii=out_w_nii,
        out_f_nii=out_f_nii,
        out_r_nii=out_r_nii,
        out_l_nii=out_l_nii,
        reconstructed_swap_stat=swap_static_rec,
        needs_manuel_intervention=needs_manuel_intervention,
    )


def pipeline_bids(
    s_magnitude: Sequence[BIDS_FILE],
    water_image: BIDS_FILE,
    fat_image: BIDS_FILE,
    derivative="derivatives_inversion",
    derivative_total="derivatives_Abdominal-Segmentation",
    steps_signal_prior=50,
    override=False,
    ddevice: Literal["cpu", "cuda", "mps"] = "cuda",
    gpu=0,
    threshold_swapped_voxels=200,
    threshold_disagree_voxels=400,
    evaluate_reconstructed=True,
    non_strict_mode=False,
    vibe_from_signal=True,
) -> Result:
    args = {
        "file_type": "nii.gz",
        "parent": derivative,
        "info": {
            "desc": "reconstructed",
        },
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    args["info"]["part"] = "water"
    out_reconstruction_water = water_image.get_changed_path(**args)
    args["info"]["part"] = "fat"
    out_reconstruction_fat = water_image.get_changed_path(**args)
    args["info"]["part"] = "r2s"
    out_reconstruction_r2s = water_image.get_changed_path(**args)
    args["info"]["part"] = "water-fraction"
    out_reconstruction_pdwf = water_image.get_changed_path(**args)
    args["info"]["part"] = "fat-fraction"
    out_reconstruction_pdff = water_image.get_changed_path(**args)
    args["info"]["part"] = "recon-loss"
    args["bids_format"] = "msk"
    out_reconstruction_loss = water_image.get_changed_path(**args)
    args["info"]["part"] = "water"
    args["info"]["desc"] = None
    args["info"]["seg"] = "fat-water-inversion-detection"
    out_detection_water = water_image.get_changed_path(**args)
    args["info"]["part"] = "fat"
    out_detection_fat = water_image.get_changed_path(**args)
    args["info"]["part"] = "water"
    args["bids_format"] = None
    args["info"]["seg"] = None
    args["info"]["desc"] = "signal_prior"
    out_signal_prior_ = water_image.get_changed_path(**args)
    args["info"]["desc"] = "signal-prior"
    out_signal_prior = water_image.get_changed_path(**args)
    if out_signal_prior_.exists() and out_signal_prior_ != out_signal_prior:
        out_signal_prior_.rename(out_signal_prior)

    args["bids_format"] = "msk"
    args["info"]["desc"] = "reconstructed"
    args["info"]["seg"] = "fat-water-inversion-detection"
    out_detection_water_reconstructed = fat_image.get_changed_path(**args)
    args["info"]["part"] = "fat"
    out_detection_fat_reconstructed = fat_image.get_changed_path(**args)
    args = {
        "file_type": "nii.gz",
        "bids_format": "msk",
        "parent": derivative_total,
        "info": {"seg": "TotalVibeSegmentator80", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    total_vibe = water_image.get_changed_path(**args)
    # total_vibe

    return pipeline(
        s_magnitude=s_magnitude,
        water_image=water_image,
        fat_image=fat_image,
        out_reconstruction_water=out_reconstruction_water,
        out_reconstruction_fat=out_reconstruction_fat,
        out_reconstruction_r2s=out_reconstruction_r2s,
        out_reconstruction_pdwf=out_reconstruction_pdwf,
        out_reconstruction_pdff=out_reconstruction_pdff,
        total_vibe=total_vibe,
        out_detection_water=out_detection_water,
        out_detection_fat=out_detection_fat,
        out_signal_prior=out_signal_prior,
        out_reconstruction_loss=out_reconstruction_loss,
        out_detection_water_reconstructed=out_detection_water_reconstructed,
        out_detection_fat_reconstructed=out_detection_fat_reconstructed,
        steps_signal_prior=steps_signal_prior,
        override=override,
        ddevice=ddevice,
        gpu=gpu,
        threshold_swapped_voxels=threshold_swapped_voxels,
        threshold_disagree_voxels=threshold_disagree_voxels,
        evaluate_reconstructed=evaluate_reconstructed,
        vibe_from_signal=vibe_from_signal,
    )


if __name__ == "__main__":
    import os

    from TPTBox import Print_Logger

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    nako_dataset = "/media/data/NAKO/dataset-nako/"
    if not Path(nako_dataset).exists():
        nako_dataset = "/DATA/NAS/datasets_processed/NAKO/dataset-nako/"

    def get_mevibe_dict(name):
        sub = str(name)
        sub_sup = sub[:3]
        path = Path(nako_dataset, f"rawdata/{sub_sup}/{sub}/mevibe/")
        files = {}

        if not path.exists():
            print(path, "does not exits")
            return None

        for i in path.glob("sub-*_sequ-me1_acq-ax_part-*_mevibe.nii.gz"):
            files[i.name.split("part-")[1].split("_")[0]] = BIDS_FILE(i, nako_dataset)
        if len(files) == 0:
            for i in path.glob("sub-*_acq-ax_part-*_mevibe.nii.gz"):
                files[i.name.split("part-")[1].split("_")[0]] = BIDS_FILE(i, nako_dataset)

        if len(files) != 11:
            print("files != 11", len(files))
            return None

        return files

    def get_vibe_dict(name):
        sub = str(name)
        sub_sup = sub[:3]
        path = Path(nako_dataset, f"rawdata_stitched/{sub_sup}/{sub}/vibe/")
        files = {}

        if not path.exists():
            print(path, "does not exits")
            return None

        for i in path.glob("sub-*_acq-ax_part-*_vibe.nii.gz"):
            files[i.name.split("part-")[1].split("_")[0]] = BIDS_FILE(i, nako_dataset)

        if len(files) != 4:
            print("files != 4", len(files))
            return None

        return files

    mevibe = False

    for i in [
        "102051",
        "104057",
        "113516",
        "102137",
        "103706",
        "123047",
        "119007",
        "128983",
        "123205",
        "124497",
        "128889",
        "114182",
        "126665",
        "123686",
        "115207",
        "124844",
        "125450",
        "108121",
        "128868",
        "115976",
        "104875",
        "125159",
        "119203",
        "107008",
        "118416",
        "116666",
        "108644",
        "127014",
        "126437",
        "126527",
        "128093",
        "121126",
        "124948",
        "127006",
        "106721",
        "125003",
        "100532",
        "127533",
        "101153",
        "129163",
        "107182",
        "104952",
        "110345",
        "104507",
        "125905",
        "107669",
        "129664",
        "121029",
        "102934",
        "116110",
        "110806",
        "115338",
        "107523",
        "108358",
        "130753",
        "129294",
        "122475",
        "104905",
        "100139",
        "124923",
        "114695",
        "117780",
        "115959",
        "113232",
        "116139",
        "115378",
        "130190",
        "118173",
        "123436",
        "120544",
        "104449",
        "120972",
        "124215",
        "105528",
        "118162",
        "102503",
        "115316",
        "122192",
        "109078",
        "102898",
        "115234",
        "115128",
        "109080",
        "106457",
        "125366",
        "100113",
        "120530",
        "101821",
        "125252",
        "125578",
        "121775",
        "126924",
        "107386",
        "110310",
        "100534",
        "125729",
        "106158",
        "126920",
        "127337",
        "111491",
        "102240",
        "129577",
        "117712",
        "113236",
        "129951",
        "123375",
        "115610",
        "113710",
    ]:
        if mevibe:
            batch_nii = get_mevibe_dict(i)
            if batch_nii is None:
                continue
            s_magnitude = [batch_nii[i] for i in ["eco0-opp1", "eco1-pip1", "eco2-opp2", "eco3-in1", "eco4-pop1", "eco5-arb1"]]
        else:
            batch_nii = get_vibe_dict(i)
            if batch_nii is None:
                continue
            s_magnitude = [batch_nii[i] for i in ["outphase", "inphase"]]
        try:
            r = pipeline_bids(s_magnitude, batch_nii["water"], batch_nii["fat"])
            print(r)
            print()
        except Exception:
            Print_Logger().print_error()
