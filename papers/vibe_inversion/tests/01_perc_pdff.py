import os
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from TPTBox import BIDS_FILE, NII, to_nii

sys.path.append(str(Path(__file__).parents[-3]))
sys.path.append(str(Path(__file__).parents[-4]))
sys.path.append(str(Path(__file__).parents[-5]))
sys.path.append(str(Path(__file__).parents[-6]))
sys.path.append(str(Path(__file__).parents[-7]))

from datasets.mevibe import MEVIBE_dataset
from papers.vibe_inversion.mago_methods import mago_sp, magorino
from papers.vibe_inversion.pipeline import predict_signal_prior, recon_fat_water_model
from papers.vibe_inversion.recon_mevibe import multipeak_fat_model_from_guess, multipeak_fat_model_smooth

batch_able = ["SSIM", "VIFp", "DISTS"]
MagneticFieldStrength = 3

c = MEVIBE_dataset(256, gray=True, test=True, validation=False, create_dataset=False)
metric: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    # "L1": lambda x, y: torch.mean(torch.abs(x - y)),
    "MSE": torch.nn.functional.mse_loss,
    "PSNR": peak_signal_noise_ratio,
    "SSIM": structural_similarity_index_measure,  # type: ignore
    # https://piq.readthedocs.io/en/latest/overview.html
}

gpu = 0
ddevice = "cuda"
override = False
steps_signal_prior = 50
derivative = "derivatives_inversion"
non_strict_mode = False


def make_pdff(water: NII, fat: NII):
    s = water + fat
    valid = (s) != 0
    out_f = water * 0
    out_f[valid] = fat[valid] / s[valid] * 100
    out_w = water * 0
    out_w[valid] = water[valid] / s[valid] * 100
    return out_w, out_f


def add_pdff(water_p: NII, fat_p: NII, out, water_gt: NII, fat_gt: NII, total_vibe: NII | None = None):
    pwff_p, pdff_p = make_pdff(water_p, fat_p)
    pwff_gt, pdff_gt = make_pdff(water_gt, fat_gt)
    if total_vibe is not None:
        liver = total_vibe.extract_label(5)
        paraspinal = total_vibe.extract_label([60, 59])
        kidny = total_vibe.extract_label([1, 2])
        vert = total_vibe.extract_label([69])
        abs_delta_pdff = np.abs(pdff_p - pdff_gt)
        abs_delta_pdff_liver = np.mean(abs_delta_pdff, where=liver)
        abs_delta_pdff_paraspinal = np.mean(abs_delta_pdff, where=paraspinal)
        abs_delta_pdff_vert = np.mean(abs_delta_pdff, where=vert)
        abs_delta_pdff_kidny = np.mean(abs_delta_pdff, where=kidny)
        print(pdff_p.max(), pdff_gt.max(), abs_delta_pdff_liver)
    out_ = {"kidney": abs_delta_pdff_kidny, "liver": abs_delta_pdff_liver, "paraspinal": abs_delta_pdff_paraspinal, "vertbody": abs_delta_pdff_vert}
    for k, v in out_.items():
        out[k] = v


def count_percent(water_p: NII, fat_p: NII, water_gt: NII, fat_gt: NII, seg: np.ndarray | None = None, total_vibe: NII | None = None):
    if isinstance(water_p, np.ndarray):
        water_p = water_gt.set_array(water_p)
    if isinstance(fat_p, np.ndarray):
        fat_p = water_gt.set_array(fat_p)
    water_gt = water_gt.set_dtype_(np.float32)
    fat_gt = fat_gt.set_dtype_(np.float32)
    water_p = water_p.set_dtype_(np.float32)
    fat_p = fat_p.set_dtype_(np.float32)
    if seg is None:
        seg = np.array(water_gt.get_array() + fat_gt.get_array() >= 50)  # Mask away low signal
    seg[water_gt + fat_gt <= 50] = 0  # Mask low signal

    total = seg.sum()
    dif_w = np.abs(water_p.get_array() - water_gt.get_array().astype(float))
    dif_f = np.abs(water_p.get_array() - fat_gt.get_array().astype(float))
    res = np.zeros_like(dif_f)
    res[dif_w < dif_f] = 1
    res[seg != 1] = 0
    print(res.sum(), total, res.sum() / total)
    out = {"res": res.sum(), "total": total, "p": res.sum() / total}
    add_pdff(water_p, fat_p, out, water_gt, fat_gt, total_vibe)
    eval_images(water_p, water_gt, out)
    return out


@torch.no_grad()
def eval_images(out_nii: NII, target_nii: NII, metric_list: dict[str, list[float]]):
    out_nii = out_nii.normalize_mri(quantile=0.95)
    target_nii = target_nii.normalize_mri(quantile=0.95)
    arr = target_nii.get_array()
    target = Tensor(arr.astype(float))
    out = Tensor(out_nii.get_array())
    assert out.min() >= -1, f"[{out.min()} - {out_nii.max()}]"
    assert out.min() <= 1, f"[{out.min()} - {out_nii.max()}]"
    for key, metric_fun in metric.items():
        if key not in metric_list:
            metric_list[key] = []
        # m = metric_fun(target_s, out_s)
        # metric_list[key].append(m.detach().cpu().item())
        m = metric_fun(out.unsqueeze(1), target.unsqueeze(1)) if key in batch_able else metric_fun(out, target)

        metric_list[key].append(m.detach().cpu().item())


results = {"MAGO": {}, "MAGO-smoothed": {}, "MAGORINO": {}, "MAGO-SP": {}, "MAGORINO-SP": {}}


def stats():
    print(
        "######################################################################################################################################################################"
    )
    for k, d in results.items():
        print(f"{k:15}", end="")
        for m_key in ["p", *metric.keys(), "liver", "paraspinal", "vertbody", "kidney"]:  # []:
            v = np.array([v[m_key] for v in d.values()])

            print(f"--{m_key:6} {np.mean(v):1.3f}±{np.std(v):1.2f} ", end="")
        print()
    print(
        "######################################################################################################################################################################"
    )


buffer_file = Path("01_perc_pdff.pkl")
if buffer_file.exists():
    with open(buffer_file, "rb") as f:
        keys = results.keys()
        results = pickle.load(f)
        for key in keys:
            if key not in results:
                results[key] = {}
print("total-len", len(results["MAGO"]), len(results["MAGORINO-SP"]))
stats()
out_path = "/DATA/NAS/ongoing_projects/robert/code/image2image/papers/vibe_inversion/outputs"
try:
    os.nice(10)
except Exception:
    pass


def save(out_w, out_f, out_r, key, total_vibe):
    out_w = water_gt.set_array(out_w)
    out_w.save(f"{out_path}/{sub}/w_{key}.nii.gz")
    water_gt.set_array(out_f).save(f"{out_path}/{sub}/f_{key}.nii.gz")
    water_gt.set_array(out_r).save(f"{out_path}/{sub}/r_{key}.nii.gz")
    results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
    return out_w


def reload(key):
    if Path(f"{out_path}/{sub}/r_{key}.nii.gz").exists():
        return to_nii(f"{out_path}/{sub}/w_{key}.nii.gz"), to_nii(f"{out_path}/{sub}/f_{key}.nii.gz"), to_nii(f"{out_path}/{sub}/r_{key}.nii.gz")
    return None


import random

# random.seed(42)
# random.shuffle(c.subjects)
# print(c.subjects[:100])
# exit()
subjs = c.subjects
old = [i.name for i in Path(out_path).iterdir()]
subjs = [*old, *[s for s in subjs if s not in old]]
for sub in subjs:
    derivative_total = "derivatives_Abdominal-Segmentation"
    d = c.get_dict(sub, load=False)
    if d is None:
        continue
    if sub in results["MAGORINO-SP"]:
        continue
    f = d["water"]
    dataset = Path(str(f).split("/dataset-")[0], "dataset-nako")
    print(f)
    print(dataset)
    water_image = BIDS_FILE(Path(f), dataset)
    args = {"file_type": "nii.gz", "parent": derivative, "info": {"desc": "reconstructed"}, "make_parent": False, "non_strict_mode": non_strict_mode}
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
    args["info"]["desc"] = "signal-prior"
    out_signal_prior = water_image.get_changed_path(**args)
    args = {
        "file_type": "nii.gz",
        "bids_format": "msk",
        "parent": derivative_total,
        "info": {"seg": "TotalVibeSegmentator80", "part": None, "mod": water_image.bids_format},
        "make_parent": False,
        "non_strict_mode": non_strict_mode,
    }
    total_vibe = to_nii(water_image.get_changed_path(**args), True)
    magnitude_nii: list[NII] = [
        to_nii(d["eco0-opp1"], False),
        to_nii(d["eco1-pip1"], False),
        to_nii(d["eco2-opp2"], False),
        to_nii(d["eco3-in1"], False),
        to_nii(d["eco4-pop1"], False),
        to_nii(d["eco5-arb1"], False),
    ]  #'fat', 'fat-fraction', 'r2s', 'water', 'water-fraction']

    magnitude = [a.get_array() for a in magnitude_nii]
    water_gt = to_nii(to_nii(d["water"], False))
    fat_gt = to_nii(d["fat"], False)
    r2s_gt = to_nii(d["r2s"]) * 0.1
    mask = to_nii(out_detection_water, True).clamp(0, 1).get_array()
    updated = False
    ### MAGO ###
    key = "MAGO"
    if sub not in results[key]:
        r = reload(key)
        if r is None:
            out_w, out_f, out_r, _ = multipeak_fat_model_smooth(magnitude, smooth=False)
            out_w = save(out_w, out_f, out_r, key, total_vibe)
        else:
            out_w, out_f, out_r = r
            results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
        updated = True
        # exit()
    ### MAGO-SMOTH ###
    key = "MAGO-smoothed"
    if sub not in results[key]:
        r = reload(key)
        if r is None:
            out_w, out_f, out_r, _ = multipeak_fat_model_smooth(magnitude, smooth=True)
            out_w = save(out_w, out_f, out_r, key, total_vibe)
        else:
            out_w, out_f, out_r = r
            results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
        updated = True
    ### MAGORINO ###
    key = "MAGORINO"
    if sub not in results[key]:
        r = reload(key)
        if r is None:
            out_w, out_f, out_r, _ = magorino(magnitude)
            out_w = save(out_w, out_f, out_r, key, total_vibe)
        else:
            out_w, out_f, out_r = r
            results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
        updated = True
    signal_prior = predict_signal_prior(magnitude_nii, out_signal_prior, steps_signal_prior, override, gpu, ddevice)
    signal_prior = to_nii(signal_prior, False)
    ### MAGO-SP ###
    key = "MAGO-SP"
    if sub not in results[key]:
        r = reload(key)
        if r is None:
            out_w, out_f, out_r, _ = recon_fat_water_model(magnitude_nii, water_gt, fat_gt, signal_prior=signal_prior)
            assert out_w is not None, out_w
            out_w = save(out_w, out_f, out_r, key, total_vibe)
        else:
            out_w, out_f, out_r = r
            results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
        updated = True
    ### MAGORINO-SP ###
    key = "MAGORINO-SP"
    if sub not in results[key]:
        r = reload(key)
        if r is None:
            magnitude = [a.get_array() for a in magnitude_nii]
            out_w, out_f, out_r, _ = mago_sp(magnitude, signal_prior.get_array(), ((water_gt + fat_gt) - signal_prior).get_array(), use_rician=True)
            out_w = save(out_w, out_f, out_r, key, total_vibe)
        else:
            out_w, out_f, out_r = r
            results[key][sub] = count_percent(out_w, out_f, water_gt, fat_gt, seg=mask, total_vibe=total_vibe)
        updated = True

    if updated:
        with open(buffer_file, "wb") as f:
            pickle.dump(results, f)

    print("total-len", len(results["MAGO"]), len(results["MAGORINO-SP"]))
    stats()
    # exit()

# print(c.subjects)
print(len(c))
print("Test", c.count_subj())
