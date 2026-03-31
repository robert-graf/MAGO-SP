import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, Logger, to_nii
from TPTBox.core.vert_constants import Full_Body_Instance_Vibe

sys.path.append(str(Path(__file__).parents[3]))
sys.path.append(str(Path(__file__).parents[4]))
sys.path.append(str(Path(__file__).parents[2]))
from papers.vibe_inversion.pipeline import make_pdff_pdwf
from papers.vibe_inversion.tests.nako_mevibe import get_mevibe_dict

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
    119880,
]


# siemens_MagneticFieldStrength = 123.2400047 / gyromagnetic_ratio
# Constants (adjust these based on your system and acquisition parameters)
# freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]) + 0.05  # Ren marrow aus https://doi.org/10.1002/jmri.25453
# freqs_ppm = np.array([5.20, 4.21, 2.66, 2.00, 1.20, 0.80]) - 4.7
# freqs_ppm = np.array([5.30, 4.20, 2.75, 2.10, 1.30, 0.90]) - 4.7
# freqs_ppm = np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50])  # Hernando et al.
# freqs_ppm = np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])  # Hamilton


alpha_p = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306])
# alpha_p = np.array([0.048, 0.039, 0.004, 0.128, 0.694, 0.087])  # UKBB
# alpha_p = np.array([0.047, 0.039, 0.006, 0.12, 0.7, 0.088])  # UKBB
# alpha_p = np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048])  # Hernando et al.
# alpha_p = np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
alpha_p = np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037])  # Hamilton

recons = [
    (
        "Ren et Al.",
        "derivatives_inversion_test_campi",
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306]),
    ),
    (
        "Hamilton et Al.",
        "derivatives_inversion_test_Hamilton",
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037]),
    ),
    (
        "Hernando et Al.",
        "derivatives_inversion_test_Hernando",
        np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50]),
        np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048]),
    ),
    (
        "Zhong et Al.",
        "derivatives_inversion_test_Zhong",
        np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68]),
        np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04]),
    ),
]
# (
#    "Ren et Al. + 0.05",
#    "derivatives_inversion_test_campi2",
#    np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]) + 0.05,
#    np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306]),
# ),


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


if __name__ == "__main__":
    from TPTBox import Print_Logger

    log = Print_Logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    mevibe = False
    needs_manuel_intervention = 0
    needs_correction = 0

    for e, sub in enumerate(subs):
        batch_niis = get_mevibe_dict(sub)
        if batch_niis is None:
            continue

        s = str(sub)
        for water_fat_model_name, derivative, freqs_ppm, alpha_p in recons:
            for sequ, batch_nii in batch_niis.items():
                if batch_nii is None:
                    continue
                try:
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

                    seg = to_nii(vibeseg, True)
                    if not out_reconstruction_pdff.exists():
                        continue
                    # if water_fat_model_name == "AI":
                    # water = to_nii(out_signal_prior)
                    # fat = to_nii(batch_nii["fat"])  # - water
                    # out_reconstruction_pdff, _ = make_pdff_pdwf(water, fat)
                    pdff_new = to_nii(out_reconstruction_pdff, True)

                    pdff_old, _ = make_pdff_pdwf(batch_nii["water"], batch_nii["fat"])
                    for idx in Full_Body_Instance_Vibe:  # [Full_Body_Instance_Vibe.liver]:
                        l = seg.extract_label(idx)
                        if l.sum() == 0:
                            continue
                        ref = (pdff_old).mean(where=l)
                        if ref < 0.01:
                            continue
                        print(f"{idx.name:40}{(pdff_new).mean(where=l):.5}\t; {ref:.5}")
                except Exception:
                    Print_Logger().print_error()
                exit()
    print()
    print()
