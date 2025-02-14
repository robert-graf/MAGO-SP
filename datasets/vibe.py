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


class VIBE_dataset(MEVIBE_dataset):
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
        split_file=Path(os.environ.get("DATASET_NAKO", ""), "notes/nako_split.xlsx"),
        black_list=Path(os.environ.get("DATASET_NAKO", ""), "notes/water_fat_inversion_vibe.xlsx"),
        dataset_path=Path(os.environ.get("DATASET_NAKO", ""), "dataset-nako/training_data/vibe/"),
        nako_dataset=Path(os.environ.get("DATASET_NAKO", ""), "dataset-nako"),
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

    def get_dict(self, name, chunk=None):
        if chunk is None:
            chunk = random.randint(1, 4)
        sub = str(name)
        sub_sup = sub[:3]
        path = Path(self.nako_dataset, f"rawdata/{sub_sup}/{sub}/vibe/")
        nii_paths = {}

        if not path.exists():
            print(path, "does not exits")
            return None

        for i in path.glob(f"sub-*_acq-ax_chunk-{chunk}_part-*_vibe.nii.gz"):
            nii = NII.load(i, True).reorient_(("S", "L", "A"))
            nii.set_dtype_("smallest_int")
            nii_paths[i.name.split("part-")[1].split("_")[0]] = nii
        if len(nii_paths) == 0:
            for i in path.glob("sub-*_acq-ax_part-*_vibe.nii.gz"):
                nii = NII.load(i, True).reorient_(("S", "L", "A"))
                nii.set_dtype_("smallest_int")
                nii_paths[i.name.split("part-")[1].split("_")[0]] = nii

        if len(nii_paths) != 4:
            print("nii_paths != 4", len(nii_paths))
            return None

        if len({i.shape for i in nii_paths.values()}) != 1:
            print({i.shape for i in nii_paths.values()})
            return None

        return nii_paths

    def get_as_batch(self, name, chunk):
        batch = self.get_dict(name, chunk)
        if batch is None:
            return None, None

        def _help(n: NII):
            v = n.get_array() - n.min()
            v = v / 1000
            v = 2 * v - 1
            v = torch.from_numpy(v)
            return v.unsqueeze(1)

        return {b: _help(batch) for b, batch in batch.items()}, batch


if __name__ == "__main__":
    c = VIBE_dataset(256, gray=True, test=True, validation=False, create_dataset=True)
    print(c[0].keys())
    # exit()
    # subj: list[str] = [str(s) for s in c.subjects]
    # df = pd.read_excel("/DATA/NAS/datasets_processed/NAKO/notes/water_fat_inversion_mevibe.xlsx")
    # filtered_df = df[df["sub"].isin(subj)].sort_values(by="percent", ascending=False)
    # print(filtered_df)
    c = VIBE_dataset(256, gray=True, test=False, validation=False, create_dataset=True)
    # subj: list[str] = [str(s) for s in c.subjects]
    # df = pd.read_excel("/DATA/NAS/datasets_processed/NAKO/notes/water_fat_inversion_mevibe.xlsx")
    # filtered_df = df[df["sub"].isin(subj)].sort_values(by="percent", ascending=False)
    # print(filtered_df)

    c = VIBE_dataset(256, gray=True, test=False, validation=True, create_dataset=True)
    # subj: list[str] = [str(s) for s in c.subjects]
    # df = pd.read_excel("/DATA/NAS/datasets_processed/NAKO/notes/water_fat_inversion_mevibe.xlsx")
    # filtered_df = df[df["sub"].isin(subj)].sort_values(by="percent", ascending=False)
    # print(filtered_df)

    # TODO limit df by colum "sub" that are conatine in subj list
    # c = MEVIBE_dataset(256, gray=True, test=False, validation=True)
    # c.create_dataset()
    # c = MEVIBE_dataset(256, gray=True, test=False, validation=False)
    # c.create_dataset()
    # c.prepare_data()
    # print(c[0].keys())
    # print(c[0]["fat"].shape, c[0]["fat"].dtype)
