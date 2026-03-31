#!/usr/bin/env python3
"""
evaluate_pdff_agreement.py

Produces:
 - Agreement scatter panels (one panel / water-fat model) comparing pdff_old (reference)
   vs pdff_new (reconstruction). Each point = mean PDFF inside an organ in one 3D image.
 - Per-organ aggregated mean/std and bias; prints a LaTeX table.

Assumptions:
 - `subs` is an iterable of subject identifiers already defined in your environment.
 - `recons` is an iterable of tuples: (water_fat_model_name, derivative, freqs_ppm, alpha_p)
   used exactly the same way as in your pipeline.
 - `get_mevibe_dict(sub)` returns a dict mapping sequence names to a dict containing keys
   "water" and "fat" (each either numpy arrays or nibabel-like objects).
 - `pipeline_bids(water, fat, derivative=...)` returns (vibeseg, out_reconstruction_water, ..., out_reconstruction_pdff)
   where out_reconstruction_pdff is a path-like object or array accepted by to_nii().
 - `to_nii(x, as_array=True)` will return a numpy array when as_array True, or nibabel image if False.
 - `make_pdff_pdwf(water, fat)` returns (pdff_old, something). pdff_old as numpy array.
 - `vibeseg` supports `extract_label(idx)` returning a boolean or {0,1} mask or array-like mask.
 - `Full_Body_Instance_Vibe` is iterable of label enums/objects with .name and comparable indices.

Adjust `LOW_SIGNAL_THRESHOLD`, `MIN_VOXELS_PER_ORG`, and `MAX_SAMPLES` for your data.
"""

import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TPTBox import BIDS_FILE, Print_Logger, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance_Vibe
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
sys.path.append(str(Path(__file__).parents[4]))
sys.path.append(str(Path(__file__).parents[2]))
from papers.vibe_inversion.pipeline import make_pdff_pdwf
from papers.vibe_inversion.tests.nako_mevibe import get_mevibe_dict

log = Print_Logger()
# ---- BEGIN CONFIG ----
MAX_SAMPLES = 1000000  # evaluate on up to this many images (random sample if more)
RANDOM_SEED = 42
LOW_SIGNAL_WATER_MEAN = 1e-3  # threshold for mean water intensity (inside organ mask) below which we reject sample
MIN_VOXELS_PER_ORG = 50  # minimum voxels in organ mask to consider the organ present
EXCLUDE_ORG_NAMES = {"lung", "lungs", "esophagus", "trachea", "pulmonary_vein"}  # organ names to ignore (case-insensitive)
OUTPUT_DIR = Path(__file__).parent / "pdff_evaluation_results"
# ---- END CONFIG ----

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pipeline_bids(
    water_image: BIDS_FILE,
    fat_image: BIDS_FILE,
    derivative="derivatives_inversion",
    derivative_total="derivatives_Abdominal-Segmentation",
    non_strict_mode=False,
):
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
    args["info"]["desc"] = None
    args["info"]["seg"] = "fat-water-inversion-detection"
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
        "info": {"seg": "VIBESeg-100", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    total_vibe = water_image.get_changed_path(**args)
    args = {
        "file_type": "nii.gz",
        "bids_format": "msk",
        "parent": derivative_total,
        "info": {"seg": "ROI", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    roi = water_image.get_changed_path(**args)
    args = {
        "file_type": "nii.gz",
        "bids_format": "msk",
        "parent": derivative_total,
        "info": {"seg": "TotalVibeSegmentator", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    total_vibe2 = water_image.get_changed_path(**args)
    if total_vibe2.exists():
        total_vibe = total_vibe2
    args = {
        "file_type": "nii.gz",
        "bids_format": "msk",
        "parent": derivative_total,
        "info": {"seg": "TotalVibeSegmentator80", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    total_vibe2 = water_image.get_changed_path(**args)
    if total_vibe2.exists():
        total_vibe = total_vibe2
    # print(total_vibe2.exists())
    ####### old name
    args["bids_format"] = "msk"
    args["parent"] = derivative
    args["info"]["mod"] = water_image.format
    args["info"]["part"] = None
    args["info"]["seg"] = "water-fat-map"
    old_name = water_image.get_changed_path(**args)
    from TPTBox import Print_Logger

    log = Print_Logger()
    if old_name.exists():
        log.on_save("rename", old_name, "->", out_detection_water)
        old_name.rename(out_detection_water)
    args["info"]["mod"] = None
    #######
    ####### old name
    args["info"]["mod"] = water_image.format
    args["info"]["desc"] = None
    args["info"]["part"] = "fat"
    args["info"]["seg"] = "water-fat-map"
    old_name = water_image.get_changed_path(**args)
    if old_name.exists():
        log.on_save("rename", old_name, "->", out_detection_fat)
        old_name.rename(out_detection_fat)
    args["info"]["mod"] = None
    return (total_vibe, out_reconstruction_water, out_reconstruction_fat, out_reconstruction_r2s, out_reconstruction_pdwf, out_reconstruction_pdff, out_signal_prior)


def is_low_signal(water_array: np.ndarray, mask: np.ndarray, threshold=LOW_SIGNAL_WATER_MEAN) -> bool:
    """
    Decide if the sample should be rejected for low signal.
    Use the mean water intensity inside the mask.
    """
    if mask.sum() < MIN_VOXELS_PER_ORG:
        return True
    mean_water = float(np.nanmean(water_array[mask > 0]))
    return mean_water < threshold


# ------------- main evaluation logic -------------
def collect_subject_list(all_subs):
    """Given 'subs' from your workspace, produce a deterministic list and optionally sample it."""
    subs_list = list(all_subs)
    if len(subs_list) > MAX_SAMPLES:
        # random.shuffle(subs_list)
        subs_list = subs_list[:MAX_SAMPLES]
    return list(subs_list)


def evaluate(subs, recons):
    """
    Walk over selected subjects and recons. Collect organ-wise PDFF (ref and new).
    Returns:
      results_by_model: dict mapping model_name -> list of dicts:
            { 'sub': sub, 'org_name': org_name, 'ref_mean': float, 'new_mean': float }
      organ_names: sorted list of organs encountered
    """
    results_by_model = defaultdict(list)
    results_by_model_px = defaultdict(list)
    organs_seen = set()
    # select sample of subjects
    used_subs = collect_subject_list(subs)

    for sub in tqdm(used_subs, "subs"):
        batch_niis = get_mevibe_dict(sub)
        if batch_niis is None:
            continue

        # try to find 'water' and 'fat' keys somewhere: assume pipeline works on per-sequence
        for water_fat_model_name, derivative in recons:
            for sequ, batch_nii in batch_niis.items():
                if batch_nii is None:
                    continue
                try:
                    # run pipeline same as in your snippet
                    (
                        vibeseg,
                        out_reconstruction_water,
                        out_reconstruction_fat,
                        out_reconstruction_r2s,
                        out_reconstruction_pdwf,
                        out_reconstruction_pdff,
                        out_signal_prior,
                    ) = pipeline_bids(
                        batch_nii["water"],
                        batch_nii["fat"],
                        derivative=derivative,
                    )
                    # if water_fat_model_name == "AI":
                    #    water = to_nii(out_signal_prior)
                    #    fat = to_nii(batch_nii["water"]) + to_nii(batch_nii["fat"]) - water
                    #    out_reconstruction_pdff, _ = make_pdff_pdwf(water, fat)
                    #    out_reconstruction_pdff = to_nii(out_reconstruction_pdff)
                    # convert pdff_new into array (use to_nii like you did)
                    pdff_new_arr = to_nii(out_reconstruction_pdff, False)
                    # if pdff_new doesn't exist or empty skip
                    if pdff_new_arr is None or np.isnan(pdff_new_arr.get_array()).all():
                        continue

                    # reference pdff_old from analytic function
                    pdff_old_arr, _ = make_pdff_pdwf(batch_nii["water"], batch_nii["fat"])
                    pdff_old_arr = to_nii(pdff_old_arr)

                    # water array for signal check
                    water_arr = to_nii(batch_nii["water"]).get_array()
                    fat_arr = to_nii(batch_nii["fat"]).get_array()
                    seg = to_nii(vibeseg, True)
                    l = seg.extract_label(
                        [
                            Full_Body_Instance_Vibe.lung_upper_lobe_left,
                            Full_Body_Instance_Vibe.lung_lower_lobe_left,
                            Full_Body_Instance_Vibe.lung_upper_lobe_right,
                            Full_Body_Instance_Vibe.lung_middle_lobe_right,
                            Full_Body_Instance_Vibe.lung_lower_lobe_right,
                            Full_Body_Instance_Vibe.esophagus,
                            Full_Body_Instance_Vibe.trachea,
                            Full_Body_Instance_Vibe.pulmonary_vein,
                        ]
                    )
                    seg[l != 0] = 0
                    u = seg.unique()
                    pdff_old_arr[l != 0] = 0
                    pdff_new_arr[l != 0] = 0
                    water_arr[l != 0] = 0
                    fat_arr[l != 0] = 0
                    # iterate organs
                    for idx in Full_Body_Instance_Vibe:  # [Full_Body_Instance_Vibe.liver, Full_Body_Instance_Vibe.subcutaneous_fat]:  # Full_Body_Instance_Vibe TODO
                        if idx.value not in u:
                            continue
                        org_name = getattr(idx, "name", str(idx)).lower()
                        if org_name in EXCLUDE_ORG_NAMES:
                            continue

                        # get mask; accept boolean or numeric mask
                        mask = seg.extract_label(idx).erode_msk(2, verbose=False)
                        if mask.sum() < MIN_VOXELS_PER_ORG:
                            continue

                        # compute reference and new means inside organ (exclude zeros where mask=0)
                        # to mimic original code's (l * pdff_old).mean() we compute mean over mask voxels
                        ref_mean = float(pdff_old_arr.mean(where=mask))
                        new_mean = float(pdff_new_arr.mean(where=mask))

                        # low-signal rejection (use water inside mask)
                        if is_low_signal(water_arr, mask.get_array()):
                            continue

                        results_by_model[water_fat_model_name].append({"sub": sub, "sequence": sequ, "org_name": org_name, "ref_mean": ref_mean, "new_mean": new_mean})
                        organs_seen.add(org_name)

                    # get mask; accept boolean or numeric mask

                    mask = np.asarray(seg) > 0
                    mask[water_arr + fat_arr < 100] = 0
                    # compute reference and new means inside organ (exclude zeros where mask=0)
                    # to mimic original code's (l * pdff_old).mean() we compute mean over mask voxels
                    ref_vals = pdff_old_arr[mask]
                    new_vals = pdff_new_arr[mask]
                    colors = seg[mask]
                    k = 500

                    # Ensure we don't sample more pixels than available
                    num_samples = min(k, len(ref_vals))

                    # Randomly sample the same indices for both arrays
                    idx = np.random.choice(len(ref_vals), size=num_samples, replace=False)

                    ref_mean = ref_vals[idx].tolist()
                    new_mean = new_vals[idx].tolist()
                    colors = colors[idx].tolist()

                    results_by_model_px[water_fat_model_name].append({"sub": sub, "sequence": sequ, "ref_mean": ref_mean, "new_mean": new_mean, "colors": colors})

                except Exception as exc:
                    # print minimal info, but continue
                    try:
                        Print_Logger().print_error()
                    except Exception:
                        print(f"Error processing sub {sub}, seq {sequ}, model {water_fat_model_name}: {exc}", file=sys.stderr)
                    continue

    organ_list = sorted(organs_seen)
    return results_by_model, results_by_model_px, organ_list


# ------------- stats & plotting -------------
def plot_agreement(results_by_model_px: dict, outdir=OUTPUT_DIR, max_points_plot=20000000):
    """
    Create a multi-panel figure: one panel per model_name
    Each panel: scatter of ref_mean (x) vs new_mean (y) for each organ-instance point.
    """
    model_names = list(results_by_model_px.keys())
    n_models = len(model_names)
    if n_models == 0:
        print("No results to plot.")
        return

    # determine grid shape
    cols = min(3, n_models)
    rows = math.ceil(n_models / cols)
    fig_w = cols * 4
    fig_h = rows * 4
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, model_name in enumerate(model_names):
        ax = axes_flat[ax_idx]
        points = results_by_model_px[model_name]
        if len(points) == 0:
            ax.set_title(f"{model_name} (no points)")
            continue

        ref = np.array([p["ref_mean"] for p in points])
        new = np.array([p["new_mean"] for p in points])
        c = np.array([p["colors"] for p in points])
        # optionally subsample for plotting performance
        npts = len(ref)
        if npts > max_points_plot:
            idx = np.random.choice(npts, size=max_points_plot, replace=False)
            ref_plot = ref[idx]
            new_plot = new[idx]
        else:
            ref_plot = ref
            new_plot = new

        ax.scatter(ref_plot, new_plot, alpha=0.4, s=6, c=c, cmap="jet")
        max_lim = max(np.nanmax(ref_plot), np.nanmax(new_plot), 0.5)
        ax.plot([0, max_lim], [0, max_lim], linestyle="--")  # identity line
        ax.set_xlabel("PDFF reference")
        ax.set_ylabel("PDFF model")
        ax.set_title(model_name)
        ax.set_xlim(-0.01, max_lim + 0.01)
        ax.set_ylim(-0.01, max_lim + 0.01)

    # hide unused axes
    for i in range(len(model_names), rows * cols):
        axes_flat[i].axis("off")

    plt.tight_layout()
    outpath = os.path.join(outdir, "pdff_agreement_panels.png")
    fig.savefig(outpath, dpi=400)
    print(f"Saved agreement panels to {outpath}")


def aggregate_stats(results_by_model: dict, organ_list: list[str]):
    """
    Produce per-organ aggregated stats across models:
      - for each model and organ, gather ref and new lists, compute mean/std and bias.
    Returns:
      stats[model_name][org] = dict(N, ref_mean, ref_std, new_mean, new_std, bias_mean, bias_std)
    """
    stats = {}
    for model_name, points in results_by_model.items():
        df = pd.DataFrame(points)
        model_stats = {}
        for org in organ_list:
            subdf = df[df["org_name"] == org]
            if subdf.empty:
                continue
            ref_vals = subdf["ref_mean"].to_numpy(dtype=float)
            new_vals = subdf["new_mean"].to_numpy(dtype=float)
            diff = new_vals - ref_vals
            model_stats[org] = {
                "N": len(ref_vals),
                "ref_mean": float(np.nanmean(ref_vals)),
                "ref_std": float(np.nanstd(ref_vals, ddof=0)),
                "new_mean": float(np.nanmean(new_vals)),
                "new_std": float(np.nanstd(new_vals, ddof=0)),
                "bias_mean": float(np.nanmean(diff)),
                "bias_std": float(np.nanstd(diff, ddof=0)),
            }
        stats[model_name] = model_stats
    return stats


import csv
import os


def print_latex_table(stats: dict, organ_list: list[str], outdir=OUTPUT_DIR, filename_tex="pdff_summary_all.tex", filename_csv="pdff_summary_all.csv", short=True):
    """
    Build one LaTeX + CSV table summarizing per-organ statistics for all models side-by-side.
    Each model contributes three columns: Ref mean±SD, Model mean±SD, Bias±SD.
    """

    os.makedirs(outdir, exist_ok=True)
    model_names = list(stats.keys())

    # ---- Build dynamic header ----
    header = ["Organ", "N"]
    for model_name in model_names:
        header.extend(
            [f"Bias_mean_{model_name}"]
            if short
            else [
                f"Ref_mean_{model_name}",
                f"Ref_std_{model_name}",
                f"Model_mean_{model_name}",
                f"Model_std_{model_name}",
                f"Bias_mean_{model_name}",
                f"Bias_std_{model_name}",
            ]
        )

    # ---- Prepare rows for both LaTeX and CSV ----
    csv_rows = []
    latex_header = ["Organ", "N"]
    for model_name in model_names:
        short_name = model_name.replace("_", "\\_")
        if short:
            latex_header.extend([f"Bias ({short_name})"])
        else:
            latex_header.extend([f"Ref ({short_name})", f"Model ({short_name})", f"Bias ({short_name})"])

    colspec = "l r " + " ".join(["c" if short else "c c c" for _ in model_names])

    tex_lines = []
    tex_lines.append("% Auto-generated PDFF summary table for all models")
    tex_lines.append("\\begin{table}[ht]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{PDFF per-organ summary across all models}")
    tex_lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    tex_lines.append("\\hline")
    tex_lines.append(" & ".join(latex_header) + " \\\\")
    tex_lines.append("\\hline")

    # ---- Fill data ----
    for org in organ_list:
        row_csv = [org]
        N = None
        for model_name in model_names:
            if org in stats[model_name]:
                N = stats[model_name][org]["N"]
                break
        row_csv.append(N if N is not None else "-")

        row_tex = [org, str(N) if N is not None else "-"]

        for model_name in model_names:
            s = stats[model_name].get(org)
            if s is None:
                row_tex.extend(["-", "-", "-"])
                row_csv.extend(["", "", "", "", "", ""])
            else:
                ref_mean, ref_std = s["ref_mean"], s["ref_std"]
                new_mean, new_std = s["new_mean"], s["new_std"]
                bias_mean, bias_std = s["bias_mean"], s["bias_std"]
                if short:
                    # Add to LaTeX
                    row_tex.extend([f"{bias_mean:.3f} $\\pm$ {bias_std:.3f}"])

                    # Add to CSV
                    row_csv.extend([f"{bias_mean:.3f}"])

                else:
                    # Add to LaTeX
                    row_tex.extend([f"{ref_mean:.3f} $\\pm$ {ref_std:.3f}", f"{new_mean:.3f} $\\pm$ {new_std:.3f}", f"{bias_mean:.3f} $\\pm$ {bias_std:.3f}"])

                    # Add to CSV
                    row_csv.extend([f"{ref_mean:.3f}", f"{ref_std:.3f}", f"{new_mean:.3f}", f"{new_std:.3f}", f"{bias_mean:.3f}", f"{bias_std:.3f}"])

        tex_lines.append(" & ".join(row_tex) + " \\\\")
        csv_rows.append(row_csv)

    tex_lines.append("\\hline")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\label{tab:pdff_summary_all}")
    tex_lines.append("\\end{table}")

    # ---- Write LaTeX ----
    out_tex = os.path.join(outdir, filename_tex)
    with open(out_tex, "w") as fh:
        fh.write("\n".join(tex_lines))
    print(f"Wrote single combined LaTeX table to {out_tex}")

    # ---- Write CSV ----
    out_csv = os.path.join(outdir, filename_csv)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"Wrote summary CSV to {out_csv}")


def make_bland_altman_plots(results_by_model: dict, organ_list: list, outdir=OUTPUT_DIR, min_presence_rate: float = 0.5, min_N: int = 10, dpi: int = 300):
    """
    Create one Bland-Altman plot per organ appearing in >= min_presence_rate fraction of subjects.
    Units are converted from promille (‰) to percent (%).
    Two subplots per organ: left = non-MAGO models, right = MAGO models.
    """
    os.makedirs(os.path.join(outdir, "Bland-Altman"), exist_ok=True)

    # Collect subjects
    all_subs = {e["sub"] for m in results_by_model.values() for e in m}
    total_subjects = len(all_subs)
    if total_subjects == 0:
        print("No subject-level results available for Bland-Altman plots.")
        return

    # Per-organ accumulation
    organ_to_subs = defaultdict(set)
    organ_to_points = defaultdict(list)  # organ -> list of (model_name, ref, new, sub)

    for model_name, entries in results_by_model.items():
        for e in entries:
            org, sub = e["org_name"], e["sub"]
            if np.isnan(e["ref_mean"]) or np.isnan(e["new_mean"]):
                continue
            organ_to_subs[org].add(sub)
            organ_to_points[org].append((model_name, float(e["ref_mean"]), float(e["new_mean"]), sub))

    # Iterate organs
    for org in sorted(organ_list):
        n_subs_with_org = len(organ_to_subs.get(org, set()))
        presence_rate = n_subs_with_org / float(total_subjects)
        points = organ_to_points.get(org, [])

        if presence_rate < min_presence_rate or len(points) < min_N:
            continue

        # Split into MAGO / non-MAGO groups
        group_points = {
            "MAGORINO": [p for p in points if "MAGO" not in p[0]],
            "MAGO": [p for p in points if "MAGO" in p[0]],
        }

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, (group_name, pts) in zip(axes, group_points.items(), strict=False):
            if not pts:
                ax.set_visible(False)
                continue

            # model_names = sorted({p[0] for p in pts})
            model_names = []
            model_to_xy = {}
            for model_name, ref, new, _ in pts:
                if model_name not in model_to_xy:
                    model_names.append(model_name)
                    model_to_xy[model_name] = {"mean": [], "diff": []}
                mean_val = 0.5 * (ref + new) / 10.0  # convert ‰ → %
                diff_val = (new - ref) / 10.0  # convert ‰ → %
                model_to_xy[model_name]["mean"].append(mean_val)
                model_to_xy[model_name]["diff"].append(diff_val)

            colors = plt.cm.tab10.colors
            handles = []
            labels = []

            for i, model_name in enumerate(model_names):
                mx = np.array(model_to_xy[model_name]["mean"])
                dy = np.array(model_to_xy[model_name]["diff"])
                if mx.size == 0:
                    continue
                color = colors[i % len(colors)]

                # Model-specific stats
                bias = np.nanmean(dy)
                sd = np.nanstd(dy, ddof=1) if len(dy) > 1 else 0.0
                loa_upper, loa_lower = bias + 1.96 * sd, bias - 1.96 * sd

                # Scatter + bias lines
                sc = ax.scatter(mx, dy, s=18, alpha=0.6, color=color)
                ax.axhline(bias, linestyle="--", linewidth=1.3, color=color)
                ax.axhline(loa_upper, linestyle=":", linewidth=1.0, color=color)
                ax.axhline(loa_lower, linestyle=":", linewidth=1.0, color=color)

                # Legend text for this model
                label = f"{model_name} (N={len(mx)}; Bias={bias:.2f}±{sd:.2f} %; LoA=[{loa_lower:.2f}, {loa_upper:.2f}]) %"
                handles.append(sc)
                labels.append(label)

            # Format axes
            ax.set_title(group_name)
            ax.set_xlabel("Mean PDFF (%)")
            ax.grid(alpha=0.15)
            ax.legend(handles, labels, frameon=False, fontsize=8, loc="upper left")

        axes[0].set_ylabel("Difference (model - reference) [%]")
        fig.suptitle(f"Bland-Altman — {org.replace('_', ' ')}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        outpath = os.path.join(outdir, "Bland-Altman", f"{org.replace(' ', '_')}_bland_altman.png")
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Bland-Altman for organ '{org}' → {outpath}")


# ------------- run everything -------------
def main():
    subs = [
        100000,
        118190,
        118207,
        118909,
        118866,
        118480,
        118712,
        118687,
        118893,
        118008,
        118189,
        118235,
        118918,
        118106,
        118688,
        118055,
        118133,
        118870,
        118665,
        118617,
        118395,
        118067,
        118519,
        118146,
        118534,
        118135,
        118793,
        118036,
        118188,
        111731,
        111676,
        111513,
        111123,
        111364,
        111529,
        111096,
        111786,
        111017,
        111778,
        111983,
        111028,
        111599,
        111797,
        111857,
        111036,
        111176,
        111722,
        111190,
        111799,
        111965,
        111613,
        111796,
        111358,
        111249,
        111950,
        111969,
        111647,
        123911,
        123708,
        123201,
        123030,
        123408,
        123468,
        123631,
        123817,
        123963,
        105727,
        105923,
        105570,
        105501,
        105767,
        105621,
        105110,
        105503,
        105485,
        105917,
        105701,
        105840,
        105345,
        105845,
        105981,
        105898,
        105683,
        105462,
        105891,
        105772,
        105543,
        105367,
        105386,
        105086,
        119717,
        119879,
        119288,
        119360,
        119231,
        119249,
        119850,
        119099,
        119680,
        119588,
        # 119880,
    ]
    subs = subs
    recons = [
        # ("Ren et Al. + 0.05", "derivatives_inversion_test_campi2"),
        ("Hamilton et Al.", "derivatives_inversion_test_Hamilton"),
        ("Ren et Al.", "derivatives_inversion_test_campi"),
        ("Hernando et Al.", "derivatives_inversion_test_Hernando"),
        ("Zhong et Al.", "derivatives_inversion_test_Zhong"),
        # ("Ren et Al. + 0.05 (3 Tesla)", "derivatives_inversion_test_campi2_3t"),
        # ("Ren et Al. (3 Tesla)", "derivatives_inversion_test_campi_3t"),
        # ("Hamilton et Al. (3 Tesla)", "derivatives_inversion_test_Hamilton_3t"),
        # ("Hernando et Al. (3 Tesla)", "derivatives_inversion_test_Hernando_3t"),
        # ("Zhong et Al. (3 Tesla)", "derivatives_inversion_test_Zhong_3t"),
        # ("Ren et Al. + 0.05 (MAGO)", "derivatives_inversion_test_campi2_mago"),
        ("Hamilton et Al. (MAGO)", "derivatives_inversion_test_Hamilton_mago"),
        ("Ren et Al. (MAGO)", "derivatives_inversion_test_campi_mago"),
        ("Hernando et Al. (MAGO)", "derivatives_inversion_test_Hernando_mago"),
        ("Zhong et Al. (MAGO)", "derivatives_inversion_test_Zhong_mago"),
        # (
        #    "AI",
        #    "derivatives_inversion_test_Zhong",
        #    np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68]),
        #    np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04]),
        # ),
    ]

    # The following names must exist in your environment (same as your earlier script).
    # If you run this as a module, import or define these earlier in the script / kernel.
    try:
        results_by_model, results_by_model_px, organ_list = evaluate(subs, recons)
        if not results_by_model:
            print("No results collected. Try loosening filters or check if pipeline outputs pdff arrays.")
            return

        plot_agreement(results_by_model_px)
        stats = aggregate_stats(results_by_model, organ_list)
        print_latex_table(stats, organ_list)
        # NEW: produce Bland-Altman plots per organ (stack models together)
        make_bland_altman_plots(results_by_model, organ_list, outdir=OUTPUT_DIR, min_presence_rate=0.5, min_N=10)

        # Also print a small summary to console
        print("\nAggregate summary (model -> organ -> N, bias_mean ± bias_std):")
        for model_name, model_stats in stats.items():
            print(f"\nModel: {model_name}")
            for org, s in model_stats.items():
                print(f"  {org:20s} N={s['N']:3d}  bias={s['bias_mean']:.4f} ± {s['bias_std']:.4f}")

    except Exception as exc:
        print("Fatal error in evaluation script:", exc)
        raise


if __name__ == "__main__":
    main()
