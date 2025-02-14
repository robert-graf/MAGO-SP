import os
import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from TPTBox import NII
from tqdm import tqdm

nth_parent = Path(__file__).resolve().parents[1]

# Add it to sys.path if it's not already there
if str(nth_parent) not in sys.path:
    sys.path.insert(0, str(nth_parent))
from datasets.mevibe import MEVIBE_dataset  # noqa: E402


class SHIP_dataset(MEVIBE_dataset):
    def __init__(
        self,
        size: int,
        gray=True,
        class_labels=False,
        validation=False,
        test=False,
        vflip=True,
        hflip=True,
        dflip=False,
        rotation=None,
        random_zoom=False,
        zoom_min=0.8,
        zoom_max=1.2,
        padding="constant",
        split_file=Path(os.environ.get("DATASET_SHIP", ""), "notes/nako_split.xlsx"),
        black_list=Path(os.environ.get("DATASET_SHIP", ""), "notes/water_fat_inversion_vibe.xlsx"),
        dataset_path=Path(os.environ.get("DATASET_SHIP", ""), "training_data/all/"),
        nako_dataset=Path(os.environ.get("DATASET_SHIP", "")),
        create_dataset=False,
        num_slices=2,
    ):
        super().__init__(
            size=size,
            gray=gray,
            validation=validation,
            test=test,
            class_labels=class_labels,
            vflip=vflip,
            hflip=hflip,
            dflip=dflip,
            rotation=rotation,
            random_zoom=random_zoom,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            padding=padding,
            split_file=split_file,
            black_list=black_list,
            dataset_path=dataset_path,
            nako_dataset=nako_dataset,
            create_dataset=create_dataset,
            num_slices=num_slices,
        )

    def get_dict(self, name):
        sub = str(name)
        sub_sup = sub[:5]
        path = Path(self.nako_dataset, f"rawdata-registered/{sub_sup}/sub-{sub}/")
        if not path.exists():
            return None
        niis = {}
        niis_path = {}
        ## T1w
        niis_path["T1w"] = next((path / "T1w").glob(f"sub-{sub}_sequ-T1w*_T1w.nii.gz"))
        niis_path["T2w"] = next((path / "T2w").glob(f"sub-{sub}_sequ-T2w*_T2w.nii.gz"))
        assert niis_path["T1w"].exists(), niis_path["T1w"]
        for k, i in niis_path.items():
            nii = NII.load(i, True).reorient_(("S", "L", "A"))
            nii.set_dtype_("smallest_int")
            niis[k] = nii
        for i in (path / "vibe").glob("sub-*vibe.nii.gz"):
            # extract_vertebra_bodies_from_totalVibe()
            nii = NII.load(i, True).reorient_(("S", "L", "A"))
            nii.set_dtype_("smallest_int")
            key = i.name.split("part-")[1].split("_")[0]
            niis_path[key] = i
            niis[key] = nii
        if len(niis) != 6:
            print("nii_paths != 6", len(niis))
            return None
        return niis

    def get_as_batch(self, name):
        batch = self.get_dict(name)
        if batch is None:
            return None, None

        def _help(n: NII):
            v = n.get_array() - n.min()
            v = v / n.max()
            v = 2 * v - 1
            v = torch.from_numpy(v)
            return v.unsqueeze(1)

        return {b: _help(batch) for b, batch in batch.items()}, batch


if __name__ == "__main__":
    os.environ["DATASET_SHIP"] = "/DATA/NAS/datasets_processed/SHIP/"
    c = SHIP_dataset(256, gray=True, test=True, validation=False, create_dataset=False)
    print(c.get_dict(10105004).keys())
    # print(c[0].keys())
    # c = SHIP_dataset(256, gray=True, test=False, validation=False, create_dataset=True)
    # c = SHIP_dataset(256, gray=True, test=False, validation=True, create_dataset=True)
