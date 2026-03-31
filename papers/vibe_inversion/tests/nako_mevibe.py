import os
import random
import sys
from pathlib import Path

import numpy as np
from TPTBox import BIDS_FILE, Logger

sys.path.append(str(Path(__file__).parents[3]))


from papers.vibe_inversion.pipeline import pipeline_bids
from papers.vibe_inversion.recon_mevibe import gyromagnetic_ratio

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


siemens_MagneticFieldStrength = 123.2400047 / gyromagnetic_ratio
# Constants (adjust these based on your system and acquisition parameters)
# freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
# freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59]) + 0.05  # Ren marrow aus https://doi.org/10.1002/jmri.25453
# freqs_ppm = np.array([5.20, 4.21, 2.66, 2.00, 1.20, 0.80]) - 4.7
# freqs_ppm = np.array([5.30, 4.20, 2.75, 2.10, 1.30, 0.90]) - 4.7
# freqs_ppm = np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50])  # Hernando et al.
# freqs_ppm = np.array([-3.73, -3.33, -3.04, -2.60, -2.38, -1.86, 0.68])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
freqs_ppm = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])  # Hamilton


# alpha_p = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006, 0.01498501, 0.03996004, 0.00999001, 0.05694306])  # Ren
# alpha_p = np.array([0.048, 0.039, 0.004, 0.128, 0.694, 0.087])  # UKBB
# alpha_p = np.array([0.047, 0.039, 0.006, 0.12, 0.7, 0.088])  # UKBB
# alpha_p = np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048])  # Hernando et al.
# alpha_p = np.array([0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04])  # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.25054
alpha_p = np.array([0.088, 0.642, 0.058, 0.062, 0.058, 0.006, 0.039, 0.01, 0.037])  # Hamilton

reconstruction_name = "Hamilton"
stop_after_signal_prior = False
if __name__ == "__main__":
    import os

    from TPTBox import Print_Logger

    log = Print_Logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    mevibe = False
    subs = list(range(100000, 140000))
    # subs = list(range(100000, 100058))
    # random.shuffle(subs)
    # subs = [115345]
    needs_manuel_intervention = 0
    needs_correction = 0

    for e, sub in enumerate(subs):
        batch_niis = get_mevibe_dict(sub)
        if batch_niis is None:
            continue
        # if len(batch_niis) >= 2:
        #    print(sub)
        s = str(sub)
        for a in Path(f"/DATA/NAS/datasets_processed/NAKO/dataset-nako/derivatives_inversion/{s[:3]}/{sub}/mevibe").glob("*mod-me*"):
            a.unlink(missing_ok=True)
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
                r = pipeline_bids(
                    s_magnitude,
                    batch_nii["water"],
                    batch_nii["fat"],
                    vibe_from_signal=False,
                    threshold_disagree_voxels=100000,
                    threshold_swapped_voxels=10000,
                    stop_after_signal_prior=stop_after_signal_prior,
                    alpha_p=alpha_p,
                    MagneticFieldStrength=siemens_MagneticFieldStrength,
                    freqs_ppm=freqs_ppm,
                    reconstruction_name=reconstruction_name,
                )
                if r.original_swap_stat is not None and r.original_swap_stat.affected_structures is not None and len(r.original_swap_stat.affected_structures) != 0:
                    log.print(sub, sequ, r.original_swap_stat.affected_structures)
                    print(sub, sequ, r.original_swap_stat.affected_structures)
                if r.needs_correction:
                    needs_correction += 1
                if r.needs_manuel_intervention:
                    needs_manuel_intervention += 1
                    log.print("--->", sub, sequ, " <---")
                print(e, sub, sequ, f"{needs_correction=}, {needs_manuel_intervention=}", end="\r")
            except Exception:
                Print_Logger().print_error()

    print()
    print()
