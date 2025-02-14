import random
import sys
from pathlib import Path

from TPTBox import BIDS_FILE

sys.path.append(str(Path(__file__).parents[2]))
from papers.vibe_inversion.pipeline import pipeline_bids

if __name__ == "__main__":
    import os

    from TPTBox import Print_Logger

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    nako_dataset = "/media/data/NAKO/dataset-nako/"
    if not Path(nako_dataset).exists():
        nako_dataset = "/DATA/NAS/datasets_processed/NAKO/dataset-nako/"

    def get_vibe_dict(name, i):
        sub = str(name)
        sub_sup = sub[:3]
        path = Path(nako_dataset, f"rawdata/{sub_sup}/{sub}/vibe/")
        files = {}

        if not path.exists():
            print(path, "does not exits")
            return None

        for i in path.glob(f"sub-*_chunk-{i}_part-*_vibe.nii.gz"):
            files[i.name.split("part-")[1].split("_")[0]] = BIDS_FILE(i, nako_dataset)
        if len(files) == 0:
            return None
        if len(files) != 4:
            print("files != 4", len(files))
            return None

        return files

    mevibe = False
    subs = list(range(100000, 140000))
    random.shuffle(subs)
    # subs = [100113]
    needs_manuel_intervention = 0
    needs_correction = 0

    for sub in subs:
        for i in range(1, 16):
            batch_nii = get_vibe_dict(sub, i)
            if batch_nii is None:
                continue
            s_magnitude = [batch_nii[i] for i in ["outphase", "inphase"]]
            try:
                r = pipeline_bids(s_magnitude, batch_nii["water"], batch_nii["fat"], vibe_from_signal=False)
                if r.needs_correction:
                    needs_correction += 1
                if r.needs_manuel_intervention:
                    needs_manuel_intervention += 1
                    print("--->", sub, i, " <---")
                print(sub, i, f"{needs_correction=}, {needs_manuel_intervention=}", end="\r")
            except Exception:
                Print_Logger().print_error()
    print()
    print()
