import os
import random
import sys
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, Logger

sys.path.append(str(Path(__file__).parents[3]))


from papers.vibe_inversion.pipeline import pipeline_bids
from papers.vibe_inversion.recon_mevibe import gyromagnetic_ratio

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


siemens_MagneticFieldStrength = 123.2400047 / gyromagnetic_ratio
# Constants (adjust these based on your system and acquisition parameters)
# freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]) + 0.05  # Ren marrow aus https://doi.org/10.1002/jmri.25453
# freqs_ppm = np.array([5.20, 4.21, 2.66, 2.00, 1.20, 0.80]) - 4.7
# freqs_ppm = np.array([5.30, 4.20, 2.75, 2.10, 1.30, 0.90]) - 4.7
# freqs_ppm = np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50])  # Hernando et al.
# freqs_ppm = np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])  # Hamilton


alpha_p = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306])  # Ren
# alpha_p = np.array([0.048, 0.039, 0.004, 0.128, 0.694, 0.087])  # UKBB
# alpha_p = np.array([0.047, 0.039, 0.006, 0.12, 0.7, 0.088])  # UKBB
# alpha_p = np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048])  # Hernando et al.
# alpha_p = np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
alpha_p = np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037])  # Hamilton


recons = [
    (
        "derivatives_inversion_test_campi",  # Ren
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306]),
        siemens_MagneticFieldStrength,
        True,
    ),
    (
        "derivatives_inversion_test_Hamilton",
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037]),
        siemens_MagneticFieldStrength,
        True,
    ),
    (
        "derivatives_inversion_test_Hernando",
        np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50]),
        np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048]),
        siemens_MagneticFieldStrength,
        True,
    ),
    (
        "derivatives_inversion_test_Zhong",
        np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68]),
        np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04]),
        siemens_MagneticFieldStrength,
        True,
    ),
    # (
    #    "derivatives_inversion_test_campi2",
    #    np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]) + 0.05,
    #    np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306]),
    #    siemens_MagneticFieldStrength,
    #    True,
    # ),
    (
        "derivatives_inversion_test_campi_mago",
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306]),
        siemens_MagneticFieldStrength,
        False,
    ),
    (
        "derivatives_inversion_test_Hamilton_mago",
        np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]),
        np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037]),
        siemens_MagneticFieldStrength,
        False,
    ),
    (
        "derivatives_inversion_test_Hernando_mago",
        np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50]),
        np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048]),
        siemens_MagneticFieldStrength,
        False,
    ),
    (
        "derivatives_inversion_test_Zhong_mago",
        np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68]),
        np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04]),
        siemens_MagneticFieldStrength,
        False,
    ),
]
os.nice(15)
nako_dataset = "/media/data/NAKO/dataset-nako/"
if not Path(nako_dataset).exists():
    nako_dataset = "/DATA/NAS/datasets_processed/NAKO/dataset-nako/"
    log = Logger("/DATA/NAS/datasets_processed/NAKO/notes/invers_logs", "mevibe")


def get_mevibe_dict(name):
    rawdata = "rawdata"
    sub = str(name)
    sub_sup = sub[:3]
    path = Path(nako_dataset, f"{rawdata}/{sub_sup}/{sub}/mevibe/")
    files: dict[str, dict] = {}

    if not path.exists():
        # print(path, "does not exits")
        return None
    for i in path.glob("sub-*_part-*_mevibe.nii.gz"):
        sequ = i.name.split("sequ-")[1].split("_")[0]
        if sequ not in files:
            files[sequ] = {}
        files[sequ][i.name.split("part-")[1].split("_")[0]] = BIDS_FILE(i, nako_dataset)
    if len(files) == 0:
        return None

    return files


stop_after_signal_prior = False
if __name__ == "__main__":
    import os

    from TPTBox import Print_Logger

    log = Print_Logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    mevibe = False
    # subs = list(range(100000, 140000))
    # subs = list(range(100000, 100058))
    # random.shuffle(subs)
    needs_manuel_intervention = 0
    needs_correction = 0

    for e, sub in enumerate(subs):
        batch_niis = get_mevibe_dict(sub)
        if batch_niis is None:
            continue
        # if len(batch_niis) >= 2:
        #    print(sub)
        s = str(sub)
        # for a in Path(f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_inversion/{s[:3]}/{sub}/mevibe").glob("*mod-me*"):
        #    a.unlink(missing_ok=True)
        #    exit()
        # continue

        for sequ, batch_nii in batch_niis.items():
            if batch_nii is None:
                continue
            try:
                s_magnitude = [
                    batch_nii[i]
                    for i in [
                        "eco0-opp1",
                        "eco1-pip1",
                        "eco2-opp2",
                        "eco3-in1",
                        "eco4-pop1",
                        "eco5-arb1",
                    ]
                ]
            except KeyError:
                print(batch_nii)
                continue
            try:
                for derivative, freqs_ppm, alpha_p, MagneticFieldStrength, rician in recons:
                    r = pipeline_bids(
                        s_magnitude,
                        batch_nii["water"],
                        batch_nii["fat"],
                        vibe_from_signal=False,
                        threshold_disagree_voxels=-1,
                        threshold_swapped_voxels=10000,
                        stop_after_signal_prior=stop_after_signal_prior,
                        derivative=derivative,  # "derivatives_inversion_test_Hernando", #"derivatives_inversion_test_campi",
                        MagneticFieldStrength=MagneticFieldStrength,
                        alpha_p=alpha_p,
                        freqs_ppm=freqs_ppm,
                        use_rician=rician,
                    )
                    # if r.original_swap_stat is not None and r.original_swap_stat.affected_structures is not None and len(r.original_swap_stat.affected_structures) != 0:
                    #    log.print(sub, sequ, r.original_swap_stat.affected_structures)
                    #    print(sub, sequ, r.original_swap_stat.affected_structures)
                    # if r.needs_correction:
                    #    needs_correction += 1
                    # if r.needs_manuel_intervention:
                    #    needs_manuel_intervention += 1
                    #    log.print("--->", sub, sequ, " <---")
                    # print(e, sub, sequ, f"{needs_correction=}, {needs_manuel_intervention=}", end="\r")
            except Exception:
                Print_Logger().print_error()
            # exit()
    print()
    print()
