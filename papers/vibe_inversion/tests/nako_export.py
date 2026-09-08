"""nako_export.py — build the corrected-NAKO release.

Runs *after* nako_vibe.py / nako_mevibe.py — reuses their outputs in
``derivatives_inversion`` (detection masks + mevibe signal prior) so this
pass never re-runs anything that peak-model choice does not invalidate.

VIBE
    Iterate per chunk (not the stitched volume). For each chunk look at the
    matching per-chunk detection produced by the earlier pass; skip chunks
    whose swap counts stay below the correction thresholds. Only chunks
    that actually need correction are reconstructed and emitted.

MEVIBE
    Reuse the existing detection masks AND the DL signal prior (peak-model
    independent). Re-run only the multi-peak fit, using the **Hamilton
    9-peak liver** model with **MAGO** (Gaussian residual, i.e.
    ``use_rician=False``). Rename the sequ tag in every emitted filename
    from ``me1`` to the DICOM ``SeriesNumber`` pulled from the raw json;
    record the original sequ in each emitted json under ``original_sequ``.

Nothing is written under ``dataset-nako``. Final images (+ their jsons)
land in ``dataset-nako-canonical/rawdata-corrected``; scratch outputs
(pipeline intermediates, run log) go in ``rawdata-corrected-temp``.

``--test`` stops after 10 sequences/chunks were actually corrected.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[3]))

from TPTBox import Print_Logger, to_nii  # noqa: E402

from papers.vibe_inversion.pipeline import (  # noqa: E402
    detect_inversion_seg,
    make_pdff_pdwf,
    make_swap_statistic_single,
    predict_signal_prior,
    recon_fat_water_model,
)
from papers.vibe_inversion.recon_mevibe import gyromagnetic_ratio  # noqa: E402
from papers.vibe_inversion.recon_vibe import vibe_separate_water_fat_from_guess  # noqa: E402

os.nice(15)

# --- paths -------------------------------------------------------------------

NAKO_DATASET = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako")
CANONICAL_ROOT = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako-canonical")
OUT_IMAGES_DIR = CANONICAL_ROOT / "rawdata-corrected"
OUT_TEMP_DIR = CANONICAL_ROOT / "rawdata-corrected-temp"

# --- fit config --------------------------------------------------------------

# Hamilton 9-peak liver (NMR Biomed 2011, https://doi.org/10.1002/nbm.1622)
HAMILTON_FREQS_PPM = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
HAMILTON_ALPHA_P = np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037])
RECONSTRUCTION_NAME = "Hamilton"
# MAGO (Gaussian residual). MAGORINO would set this True.
USE_RICIAN = False
SIEMENS_MAGNETIC_FIELD_STRENGTH = 123.2400047 / gyromagnetic_ratio

# thresholds — kept identical to the prior nako_vibe / nako_mevibe passes so
# "needs correction" here matches what those runs already decided.
VIBE_SWAPPED_VOXELS = 200
VIBE_DISAGREE_VOXELS = 100000
MEVIBE_SWAPPED_VOXELS = 200
MEVIBE_DISAGREE_VOXELS = 100000

# --- json handling -----------------------------------------------------------

_EXTRA_JSON_FIELDS = {
    "reconstruction_model": RECONSTRUCTION_NAME,
    "reconstruction_optimizer": "MAGO" if not USE_RICIAN else "MAGORINO",
    "reconstruction_alpha_p": HAMILTON_ALPHA_P.tolist(),
    "reconstruction_freqs_ppm": HAMILTON_FREQS_PPM.tolist(),
}


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def _series_number(raw_json_path: Path) -> str | None:
    try:
        return str(_load_json(raw_json_path)["SeriesNumber"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


# --- filesystem helpers ------------------------------------------------------


def _sub_dirs(sub: int | str, mod: str) -> tuple[Path, Path, Path, Path, Path]:
    """(raw, deriv_inversion, deriv_seg, out_images, out_temp) for a subject."""
    s = str(sub)
    ss = s[:3]
    return (
        NAKO_DATASET / "rawdata" / ss / s / mod,
        NAKO_DATASET / "derivatives_inversion" / ss / s / mod,
        NAKO_DATASET / "derivatives_Abdominal-Segmentation" / ss / s / mod,
        OUT_IMAGES_DIR / ss / s / mod,
        OUT_TEMP_DIR / ss / s / mod,
    )


def _pick_first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


# --- VIBE --------------------------------------------------------------------

VIBE_CHUNK_PARTS = ("outphase", "inphase", "water", "fat")


def _vibe_chunk_files(raw_dir: Path, sub: int | str, chunk: int) -> dict[str, Path] | None:
    files: dict[str, Path] = {}
    for part in VIBE_CHUNK_PARTS:
        p = raw_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-{part}_vibe.nii.gz"
        if not p.exists():
            return None
        files[part] = p
    return files


def _vibe_stitched_detection(deriv_inv_dir: Path, sub: int | str) -> tuple[Path, Path] | None:
    """Preferred source of truth going forward: whole-body stitched detection."""
    w = deriv_inv_dir / f"sub-{sub}_sequ-stitched_acq-ax_part-water_seg-fat-water-inversion-detection_msk.nii.gz"
    f = deriv_inv_dir / f"sub-{sub}_sequ-stitched_acq-ax_part-fat_seg-fat-water-inversion-detection_msk.nii.gz"
    return (w, f) if w.exists() and f.exists() else None


def _vibe_stitched_total_vibe(deriv_seg_dir: Path, sub: int | str) -> Path | None:
    """Whole-body semantic seg used for `affected_structures` in the swap
    report and for the `ignore_vibe_labels` filter. Uses VibeSeg-100 (label
    ids match ``TPTBox.segmentation.VibeSeg.vibeseg.VibeSeg_map``)."""
    return _pick_first_existing(
        deriv_seg_dir / f"sub-{sub}_sequ-stitched_acq-ax_mod-vibe_part-inphase_seg-VibeSeg-100_msk.nii.gz",
        deriv_seg_dir / f"sub-{sub}_sequ-stitched_acq-ax_mod-vibe_part-outphase_seg-VibeSeg-100_msk.nii.gz",
        deriv_seg_dir / f"sub-{sub}_sequ-stitched_acq-ax_mod-vibe_part-water_seg-VibeSeg-100_msk.nii.gz",
        deriv_seg_dir / f"sub-{sub}_sequ-stitched_acq-ax_mod-vibe_part-fat_seg-VibeSeg-100_msk.nii.gz",
    )


def _vibe_stitched_roi(deriv_seg_dir: Path, sub: int | str) -> Path | None:
    """ROI seg with arm labels 9/10 that we mask out of the swap counts."""
    p = deriv_seg_dir / f"sub-{sub}_sequ-stitched_acq-ax_mod-vibe_seg-ROI_msk.nii.gz"
    return p if p.exists() else None


def _resample_seg_to(src: Path, target_nii, temp_dir: Path, tag: str) -> Path:
    """Crop/resample a segmentation NIfTI onto ``target_nii``'s grid and cache
    the result in ``temp_dir``. Returns the cached path (make_swap_statistic
    accepts Path directly)."""
    out = temp_dir / f"{src.stem.replace('.nii', '')}_{tag}.nii.gz"
    if not out.exists():
        seg = to_nii(src, seg=True).resample_from_to(target_nii)
        temp_dir.mkdir(parents=True, exist_ok=True)
        seg.save(out)
    return out


@dataclass
class VibePrep:
    """Result of the CPU-only preprocessing step for one VIBE chunk.

    ``outcome`` is one of ``already_done`` / ``not_needed`` / ``ready`` /
    ``needs_detection_gpu`` / ``error``. Only ``ready`` and
    ``needs_detection_gpu`` require the GPU stage.
    """

    sub: int | str
    chunk: int
    files: dict[str, Path]
    out_water: Path
    out_fat: Path
    out_water_json: Path
    out_fat_json: Path
    temp_dir: Path
    raw_dir: Path
    outcome: str
    # Chunk-grid-resampled versions of the stitched masks — reused in the GPU
    # stage for the verification detection's swap-stat (arm exclusion etc.).
    total_vibe: Path | None = None
    roi: Path | None = None
    error: str | None = None


def _vibe_prep(
    sub: int | str,
    chunk: int,
    files: dict[str, Path],
    *,
    stitched_det: tuple[Path, Path] | None,
    stitched_total_vibe: Path | None,
    stitched_roi: Path | None,
) -> VibePrep:
    """CPU-only stage. Safe to run from a worker thread — does resampling +
    swap-stat + the up-front already-done / not-needed decision."""
    raw_dir, _deriv_inv_dir, _deriv_seg_dir, out_dir, temp_dir = _sub_dirs(sub, "vibe")
    out_water = out_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-water_desc-corrected_vibe.nii.gz"
    out_fat = out_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-fat_desc-corrected_vibe.nii.gz"
    out_water_json = out_water.with_suffix("").with_suffix(".json")
    out_fat_json = out_fat.with_suffix("").with_suffix(".json")
    prep = VibePrep(
        sub=sub,
        chunk=chunk,
        files=files,
        out_water=out_water,
        out_fat=out_fat,
        out_water_json=out_water_json,
        out_fat_json=out_fat_json,
        temp_dir=temp_dir,
        raw_dir=raw_dir,
        outcome="error",
    )
    try:
        if out_water.exists() and out_fat.exists() and out_water_json.exists() and out_fat_json.exists():
            prep.outcome = "already_done"
            return prep

        if stitched_det is None:
            # nnUNet detection is GPU work — defer to the main thread.
            prep.outcome = "needs_detection_gpu"
            return prep

        temp_dir.mkdir(parents=True, exist_ok=True)
        water_nii = to_nii(files["water"])
        det_water = _resample_seg_to(stitched_det[0], water_nii, temp_dir, f"chunk-{chunk}")
        det_fat = _resample_seg_to(stitched_det[1], water_nii, temp_dir, f"chunk-{chunk}")
        total_vibe = (
            _resample_seg_to(stitched_total_vibe, water_nii, temp_dir, f"chunk-{chunk}") if stitched_total_vibe is not None else None
        )
        roi = _resample_seg_to(stitched_roi, water_nii, temp_dir, f"chunk-{chunk}") if stitched_roi is not None else None
        stat = make_swap_statistic_single(
            f"sub-{sub}_chunk-{chunk}",
            det_water,
            det_fat,
            total_vibe,
            roi=roi,
            roi_exclude=(9, 10),
        )
        needs = stat.count_fat >= VIBE_SWAPPED_VOXELS or stat.count_disagree >= VIBE_DISAGREE_VOXELS
        prep.total_vibe = total_vibe
        prep.roi = roi
        prep.outcome = "ready" if needs else "not_needed"
        return prep  # noqa: TRY300
    except Exception as e:  # noqa: BLE001
        prep.outcome = "error"
        prep.error = f"{type(e).__name__}: {e}"
        return prep


def _vibe_finish(prep: VibePrep, *, ddevice: str, gpu: int, verify: bool, log: Print_Logger) -> str:
    """GPU stage. Runs sequentially on the main thread.

    ``verify=True``: after prep flags a chunk from the resampled stitched
    detection, re-run detection on the native chunk grid (with the same
    arm/ROI exclusion) and drop chunks that no longer meet the thresholds.
    This suppresses the false positives that resampling introduces at chunk
    boundaries.
    """
    sub, chunk, files = prep.sub, prep.chunk, prep.files
    raw_dir, temp_dir = prep.raw_dir, prep.temp_dir

    # Chunk-native nnUNet detection: (a) the mandatory fallback path when no
    # stitched detection existed; (b) the verification pass when the resampled
    # stitched detection flagged the chunk (`verify=True`, default).
    run_native_detection = prep.outcome == "needs_detection_gpu" or verify
    if run_native_detection:
        temp_dir.mkdir(parents=True, exist_ok=True)
        det_water = temp_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-water_seg-fat-water-inversion-detection_msk.nii.gz"
        det_fat = temp_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-fat_seg-fat-water-inversion-detection_msk.nii.gz"
        detect_inversion_seg(
            files["outphase"],
            files["inphase"],
            files["water"],
            files["fat"],
            det_water,
            det_fat,
            override=False,
            ddevice=ddevice,
            gpu=gpu,
        )
        stat = make_swap_statistic_single(
            f"sub-{sub}_chunk-{chunk}",
            det_water,
            det_fat,
            prep.total_vibe,
            roi=prep.roi,
            roi_exclude=(9, 10),
        )
        if not (stat.count_fat >= VIBE_SWAPPED_VOXELS or stat.count_disagree >= VIBE_DISAGREE_VOXELS):
            log.print(f"vibe sub-{sub} chunk-{chunk}: verification cleared (false positive)")
            return "not_needed_verified"

    prep.out_water.parent.mkdir(parents=True, exist_ok=True)

    sig_prior = temp_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-water_desc-signal-prior_vibe.nii.gz"
    predict_signal_prior(
        [files["outphase"], files["inphase"]],
        sig_prior,
        steps_signal_prior=50,
        override=False,
        gpu=gpu,
        ddevice=ddevice,
    )
    water_nii = to_nii(files["water"])
    fat_nii = to_nii(files["fat"])
    prior = to_nii(sig_prior).clamp(0, 1000)
    out_w, out_f = vibe_separate_water_fat_from_guess(water_nii, fat_nii, prior)
    out_w.save(prep.out_water)
    out_f.save(prep.out_fat)

    for part, out_json in (("water", prep.out_water_json), ("fat", prep.out_fat_json)):
        raw_json = raw_dir / f"sub-{sub}_acq-ax_chunk-{chunk}_part-{part}_vibe.json"
        base = _load_json(raw_json) if raw_json.exists() else {}
        base.update(_EXTRA_JSON_FIELDS)
        _dump_json(out_json, base)

    log.print(f"vibe corrected sub-{sub} chunk-{chunk}")
    return "corrected"


# --- MEVIBE ------------------------------------------------------------------

MEVIBE_ECHO_PARTS = ("eco0-opp1", "eco1-pip1", "eco2-opp2", "eco3-in1", "eco4-pop1", "eco5-arb1")


def _mevibe_sequs(raw_dir: Path, sub: int | str) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {}
    for p in raw_dir.glob(f"sub-{sub}_sequ-*_acq-ax_part-*_mevibe.nii.gz"):
        sequ = p.name.split("sequ-")[1].split("_")[0]
        part = p.name.split("part-")[1].split("_")[0]
        out.setdefault(sequ, {})[part] = p
    return out


def _mevibe_total_seg(deriv_seg_dir: Path, sub: int | str, sequ: str) -> Path | None:
    stem = f"sub-{sub}_sequ-{sequ}_acq-ax_mod-mevibe_seg-"
    return _pick_first_existing(
        deriv_seg_dir / f"{stem}TotalVibeSegmentator80_msk.nii.gz",
        deriv_seg_dir / f"{stem}TotalVibeSegmentator_msk.nii.gz",
        deriv_seg_dir / f"{stem}VibeSeg-100_msk.nii.gz",
    )


def process_mevibe_sequ(
    sub: int | str,
    sequ: str,
    parts: dict[str, Path],
    *,
    ddevice: str,
    gpu: int,
    use_gpu: bool,
    log: Print_Logger,
) -> str:
    """Return "skipped" / "not_needed" / "corrected" / "error"."""
    raw_dir, deriv_inv_dir, deriv_seg_dir, out_dir, temp_dir = _sub_dirs(sub, "mevibe")

    # need all echoes + water + fat
    try:
        s_magnitude = [parts[e] for e in MEVIBE_ECHO_PARTS]
    except KeyError:
        return "skipped"
    if "water" not in parts or "fat" not in parts:
        return "skipped"

    # -- new sequ name from DICOM SeriesNumber -------------------------------
    seq_num = None
    for e in MEVIBE_ECHO_PARTS:
        j = raw_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-{e}_mevibe.json"
        seq_num = _series_number(j)
        if seq_num is not None:
            break
    if seq_num is None:
        log.print(f"mevibe sub-{sub} sequ-{sequ}: no SeriesNumber in any echo json, skipping")
        return "skipped"

    # -- decide need-for-correction using the existing detection -------------
    det_water = deriv_inv_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-water_seg-fat-water-inversion-detection_msk.nii.gz"
    det_fat = deriv_inv_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-fat_seg-fat-water-inversion-detection_msk.nii.gz"
    have_detection = det_water.exists() and det_fat.exists()
    total_vibe = _mevibe_total_seg(deriv_seg_dir, sub, sequ)

    if have_detection:
        stat = make_swap_statistic_single(
            f"sub-{sub}_sequ-{sequ}",
            det_water,
            det_fat,
            total_vibe,
        )
        needs = stat.count_fat >= MEVIBE_SWAPPED_VOXELS or stat.count_disagree >= MEVIBE_DISAGREE_VOXELS
        if not needs:
            return "not_needed"
    else:
        det_water = temp_dir / det_water.name
        det_fat = temp_dir / det_fat.name
        temp_dir.mkdir(parents=True, exist_ok=True)
        detect_inversion_seg(
            parts["eco0-opp1"],
            parts["eco3-in1"],
            parts["water"],
            parts["fat"],
            det_water,
            det_fat,
            override=False,
            ddevice=ddevice,
            gpu=gpu,
        )
        stat = make_swap_statistic_single(
            f"sub-{sub}_sequ-{sequ}",
            det_water,
            det_fat,
            total_vibe,
        )
        needs = stat.count_fat >= MEVIBE_SWAPPED_VOXELS or stat.count_disagree >= MEVIBE_DISAGREE_VOXELS
        if not needs:
            return "not_needed"

    # -- output filenames (sequ renamed to SeriesNumber) ---------------------
    base = f"sub-{sub}_sequ-{seq_num}_acq-ax"
    out_water = out_dir / f"{base}_part-water_desc-corrected_mevibe.nii.gz"
    out_fat = out_dir / f"{base}_part-fat_desc-corrected_mevibe.nii.gz"
    out_r2s = out_dir / f"{base}_part-r2s_desc-corrected_mevibe.nii.gz"
    out_pdff = out_dir / f"{base}_part-fat-fraction_desc-corrected_mevibe.nii.gz"
    out_pdwf = out_dir / f"{base}_part-water-fraction_desc-corrected_mevibe.nii.gz"

    already_done = all(p.exists() for p in (out_water, out_fat, out_r2s, out_pdff, out_pdwf))
    if already_done:
        return "not_needed"

    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # -- reuse the DL signal prior from the earlier mevibe pass --------------
    src_prior = deriv_inv_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-water_desc-signal-prior_mevibe.nii.gz"
    if src_prior.exists():
        prior_path = src_prior  # read-only reuse, no copy
    else:
        prior_path = temp_dir / src_prior.name
        predict_signal_prior(
            s_magnitude,
            prior_path,
            steps_signal_prior=50,
            override=False,
            gpu=gpu,
            ddevice=ddevice,
        )

    # -- run Hamilton + MAGO fit --------------------------------------------
    recon_fat_water_model(
        s_magnitude,
        parts["water"],
        parts["fat"],
        prior_path,
        out_reconstruction_water=out_water,
        out_reconstruction_fat=out_fat,
        out_reconstruction_r2s=out_r2s,
        out_reconstruction_loss=None,
        ti_ms=None,  # falls back to ti_ms_default
        override=False,
        vibe_from_signal=False,
        MagneticFieldStrength=SIEMENS_MAGNETIC_FIELD_STRENGTH,
        alpha_p=HAMILTON_ALPHA_P,
        freqs_ppm=HAMILTON_FREQS_PPM,
        use_rician=USE_RICIAN,
        use_gpu=use_gpu,
        gpu_device="cuda" if use_gpu and ddevice == "cuda" else "cpu",
    )
    make_pdff_pdwf(out_water, out_fat, out_pdff, out_pdwf, override=False)

    # -- jsons ---------------------------------------------------------------
    for part, out_nii in (
        ("water", out_water),
        ("fat", out_fat),
        ("r2s", out_r2s),
        ("fat-fraction", out_pdff),
        ("water-fraction", out_pdwf),
    ):
        raw_json = raw_dir / f"sub-{sub}_sequ-{sequ}_acq-ax_part-{part}_mevibe.json"
        base_json = _load_json(raw_json) if raw_json.exists() else {}
        base_json["original_sequ"] = sequ
        base_json.update(_EXTRA_JSON_FIELDS)
        out_json = out_nii.with_suffix("").with_suffix(".json")
        _dump_json(out_json, base_json)

    log.print(f"mevibe corrected sub-{sub} sequ-{sequ} -> {seq_num}")
    return "corrected"


# --- driver ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=("vibe", "mevibe", "both"), default="both")
    ap.add_argument("--start", type=int, default=100000)
    ap.add_argument("--stop", type=int, default=140000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ddevice", choices=("cpu", "cuda", "mps"), default="cuda")
    ap.add_argument("--use-gpu-fit", action="store_true", help="batched torch multi-peak fit for MEVIBE")
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
        help="Threads used to preprocess VIBE chunks (resample + swap-stat) ahead of the GPU stage.",
    )
    ap.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="Re-run detection on the native chunk grid after prep flags a swap, to drop false positives from resampling (default).",
    )
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--test", action="store_true", help="stop after 10 corrections")
    args = ap.parse_args()

    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEMP_DIR.mkdir(parents=True, exist_ok=True)[]
    log = Print_Logger()

    corrected = 0
    seen = 0

    # Producer/consumer for VIBE: CPU workers preprocess chunks (resample the
    # stitched detection/ROI/total-vibe onto the chunk grid, then compute the
    # swap statistic) ahead of the main thread, which then runs the GPU
    # signal-prior + reconstruction sequentially in submission order.
    pool = ThreadPoolExecutor(max_workers=max(1, args.cpu_workers)) if args.cpu_workers > 0 else None
    in_flight: deque = deque()  # of (fut, sub, chunk) tuples
    max_in_flight = max(2, args.cpu_workers * 2) if pool else 0

    def _drain_one(force: bool) -> bool:
        """Handle the oldest preprocessed chunk (blocking if ``force``).
        Returns True if the test-stop was hit."""
        nonlocal corrected, seen
        if not in_flight:
            return False
        if not force and not in_flight[0][0].done():
            return False
        fut, s, c = in_flight.popleft()
        prep = fut.result()
        seen += 1
        outcome = prep.outcome
        try:
            if outcome in ("ready", "needs_detection_gpu"):
                outcome = _vibe_finish(prep, ddevice=args.ddevice, gpu=args.gpu, verify=args.verify, log=log)
            elif outcome == "error":
                log.print(f"vibe sub-{s} chunk-{c}: prep error: {prep.error}")
        except Exception:  # noqa: BLE001
            Print_Logger().print_error()
            outcome = "error"
        if outcome == "corrected":
            corrected += 1
        print(f"vibe sub-{s} chunk-{c}: {outcome} (corrected={corrected}, seen={seen})            ", end="\r")
        return bool(args.test and corrected >= 10)

    for sub in range(args.start, args.stop):
        for modality in ["vibe", "mevibe"] if args.modality == "both" else [args.modality]:
            if modality == "vibe":
                raw_dir, deriv_inv_dir, deriv_seg_dir, *_ = _sub_dirs(sub, "vibe")
                if not raw_dir.exists():
                    continue
                # Load whole-body stitched sources once per subject.
                stitched_det = _vibe_stitched_detection(deriv_inv_dir, sub)
                stitched_total_vibe = _vibe_stitched_total_vibe(deriv_seg_dir, sub)
                stitched_roi = _vibe_stitched_roi(deriv_seg_dir, sub)
                for chunk in range(1, 16):
                    files = _vibe_chunk_files(raw_dir, sub, chunk)
                    if files is None:
                        continue
                    if pool is None:
                        # Serial fallback (--cpu-workers 0).
                        prep = _vibe_prep(
                            sub,
                            chunk,
                            files,
                            stitched_det=stitched_det,
                            stitched_total_vibe=stitched_total_vibe,
                            stitched_roi=stitched_roi,
                        )
                        seen += 1
                        outcome = prep.outcome
                        try:
                            if outcome in ("ready", "needs_detection_gpu"):
                                outcome = _vibe_finish(prep, ddevice=args.ddevice, gpu=args.gpu, verify=args.verify, log=log)
                            elif outcome == "error":
                                log.print(f"vibe sub-{sub} chunk-{chunk}: prep error: {prep.error}")
                        except Exception:  # noqa: BLE001
                            Print_Logger().print_error()
                            outcome = "error"
                        if outcome == "corrected":
                            corrected += 1
                        print(f"vibe sub-{sub} chunk-{chunk}: {outcome} (corrected={corrected}, seen={seen})", end="\r")
                        if args.test and corrected >= 10:
                            print()
                            log.print(f"--test hit corrected>=10 (seen={seen}); stopping")
                            return
                        continue
                    # Parallel path — submit prep, drain when buffer full.
                    fut = pool.submit(
                        _vibe_prep,
                        sub,
                        chunk,
                        files,
                        stitched_det=stitched_det,
                        stitched_total_vibe=stitched_total_vibe,
                        stitched_roi=stitched_roi,
                    )
                    in_flight.append((fut, sub, chunk))
                    while len(in_flight) >= max_in_flight:
                        if _drain_one(force=True):
                            print()
                            log.print(f"--test hit corrected>=10 (seen={seen}); stopping")
                            assert pool is not None
                            pool.shutdown(wait=False, cancel_futures=True)
                            return
                # Between subjects, opportunistically drain anything already done
                # so the console line stays close to the current subject.
                while in_flight and in_flight[0][0].done():
                    if _drain_one(force=True):
                        print()
                        log.print(f"--test hit corrected>=10 (seen={seen}); stopping")
                        assert pool is not None
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
            else:
                raw_dir, *_ = _sub_dirs(sub, "mevibe")
                if not raw_dir.exists():
                    continue
                sequs = _mevibe_sequs(raw_dir, sub)
                for sequ, parts in sequs.items():
                    try:
                        outcome = process_mevibe_sequ(
                            sub,
                            sequ,
                            parts,
                            ddevice=args.ddevice,
                            gpu=args.gpu,
                            use_gpu=args.use_gpu_fit,
                            log=log,
                        )
                    except Exception:
                        Print_Logger().print_error()
                        outcome = "error"
                    seen += 1
                    if outcome == "corrected":
                        corrected += 1
                    print(f"mevibe sub-{sub} sequ-{sequ}: {outcome} (corrected={corrected}, seen={seen})", end="\r")
                    if args.test and corrected >= 10:
                        print()
                        log.print(f"--test hit corrected>=10 (seen={seen}); stopping")
                        if pool is not None:
                            pool.shutdown(wait=False, cancel_futures=True)
                        return

    # Final drain of the VIBE prep buffer.
    while in_flight:
        if _drain_one(force=True):
            print()
            log.print(f"--test hit corrected>=10 (seen={seen}); stopping")
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            return
    if pool is not None:
        pool.shutdown(wait=True)
    print()


if __name__ == "__main__":
    main()
