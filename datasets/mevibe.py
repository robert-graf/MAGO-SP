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
from datasets.dataset_base import BaseDataset  # noqa: E402


class MEVIBE_dataset(BaseDataset):
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
        black_list=Path(os.environ.get("DATASET_NAKO", ""), "notes/water_fat_inversion_mevibe.xlsx"),
        dataset_path=Path(os.environ.get("DATASET_NAKO", ""), "dataset-nako/training_data/mevibe/"),
        nako_dataset=Path(os.environ.get("DATASET_NAKO", ""), "dataset-nako"),
        create_dataset=False,
        num_slices=2,
    ):
        super().__init__(
            size=size,
            gray=gray,
            class_labels=class_labels,
            vflip=vflip,
            hflip=hflip,
            dflip=dflip,
            rotation=rotation,
            random_zoom=random_zoom,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
            padding=padding,
        )
        assert not (validation and test), f"test and validation can not be true at the same time, {validation=}; {test=}"

        # self.size = (size, size)
        if validation:
            phase = "val"
        elif test:
            phase = "test"
        else:
            phase = "train"
        Path(dataset_path).mkdir(exist_ok=True)
        dataset_path = Path(dataset_path, phase)
        dataset_path.mkdir(exist_ok=True)
        self.dataset_path = dataset_path
        self.phase = phase
        self.split_file = split_file
        self.black_list = black_list
        self.nako_dataset = nako_dataset
        self.num_slices = num_slices
        if create_dataset:
            self.create_dataset()

        file_list_p = self.dataset_path / "files.pkl"
        assert file_list_p.exists(), f"call create_dataset = True, {file_list_p}"
        with open(file_list_p, "rb") as f:
            file_list = pickle.load(f)

        self.file_list: list[str] = file_list["filename"]
        self.subjects: list[str] = file_list["subjects"]

    def __len__(self):
        return len(self.file_list)

    def load_file(self, path: Path | str, default_key="img", norm=True):
        path = str(path)
        end = path.split(".")[-1]
        dict_mods = {}
        if end in ("npz", "npy"):
            f = np.load(path, allow_pickle=True)
            for k, v2 in f.items():  # type: ignore
                v: np.ndarray = v2.astype("f")
                if norm:
                    v -= v.min()
                    v /= 1000
                    v = 2 * v - 1
                dict_mods[k] = v
            f.close()  # type: ignore
            return dict_mods
        assert False

    def __getitem__(self, i):
        path = self.dataset_path / self.file_list[i]
        image_dict = self.load_file(path)
        return self.data_argumentation_2D(image_dict)

    def get_dict(self, name, load=True):
        sub = str(name)
        sub_sup = sub[:3]
        path = Path(self.nako_dataset, f"rawdata/{sub_sup}/{sub}/mevibe/")
        nii_paths = {}

        if not path.exists():
            print(path, "does not exits")
            return None

        for i in path.glob("sub-*_sequ-me1_acq-ax_part-*_mevibe.nii.gz"):
            k = i.name.split("part-")[1].split("_")[0]
            if load:
                nii = NII.load(i, True).reorient_(("S", "L", "A"))
                nii.set_dtype_("smallest_int")

                nii_paths[k] = nii
            else:
                nii_paths[k] = i
        if len(nii_paths) == 0:
            for i in path.glob("sub-*_acq-ax_part-*_mevibe.nii.gz"):
                k = i.name.split("part-")[1].split("_")[0]
                if load:
                    nii = NII.load(i, True).reorient_(("S", "L", "A"))
                    nii.set_dtype_("smallest_int")
                    nii_paths[k] = nii
                else:
                    nii_paths[k] = i

        if len(nii_paths) != 11:
            print("nii_paths != 11", len(nii_paths))
            return None
        if load and len({i.shape for i in nii_paths.values()}) != 1:
            print({i.shape for i in nii_paths.values()})
            return None

        return nii_paths

    def get_as_batch(self, name):
        batch = self.get_dict(name)
        if batch is None:
            return None, None

        def _help(n: NII):
            v = n.get_array() - n.min()
            v = v / 1000
            v = 2 * v - 1
            v = torch.from_numpy(v)
            return v.unsqueeze(1)

        return {b: _help(batch) for b, batch in batch.items()}, batch

    def count_subj(self):
        random.seed(1235)
        split_df = pd.read_excel(self.split_file)
        names = split_df[split_df["split"] == self.phase]["name"]
        bl_df = pd.read_excel(self.black_list)
        black_list = bl_df[bl_df["percent"] > 0.00001]["sub"]
        black_list = {str(i).split("_")[0] for i in black_list}
        i = 0
        for name in tqdm(names):
            if str(name) in black_list:
                continue
            # nii_paths = self.get_dict(name)
            # if nii_paths is None:
            #    continue
            i += 1
        print(i)
        return i

    def create_dataset(self):
        buffer_path = self.dataset_path / "files.pkl"
        if buffer_path.exists():
            print(buffer_path, "exist. Did not override the dataset")
            return None

        random.seed(1235)
        split_df = pd.read_excel(self.split_file)
        names = split_df[split_df["split"] == self.phase]["name"]
        bl_df = pd.read_excel(self.black_list)
        black_list = bl_df[bl_df["percent"] > 0.00001]["sub"]
        black_list = {str(i).split("_")[0] for i in black_list}
        dic_dataset = {"filename": [], "subjects": []}

        def process_subject(name):
            # Check if subject is in blacklist
            if str(name) in black_list:
                return None
            nii_paths = self.get_dict(name)
            if nii_paths is None:
                return None
            sub = str(name)
            sub_sup = sub[:3]
            subject_data = {"filename": [], "subject": name}
            for slice_index in random.sample(range(next(iter(nii_paths.values())).shape[0]), self.num_slices):
                save_path = self.dataset_path / f"{sub_sup}/{sub}_{slice_index}.npz"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if not save_path.exists():
                    slice_data = {part: nii[slice_index, :, :] for part, nii in nii_paths.items()}
                    np.savez_compressed(save_path, **slice_data)
                subject_data["filename"].append(save_path.relative_to(self.dataset_path))

            return subject_data

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_subject, name): name for name in names}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
                result = future.result()
                if result:
                    dic_dataset["filename"].extend(result["filename"])
                    dic_dataset["subjects"].append(result["subject"])

        with open(buffer_path, "wb") as f:
            pickle.dump(dic_dataset, f)
        return dic_dataset


if __name__ == "__main__":
    #
    #
    #
    #
    # exit()
    c = MEVIBE_dataset(256, gray=True, test=True, validation=False, create_dataset=True)
    print(c[0].keys())
    print("Test", c.count_subj())
    # subj: list[str] = [str(s) for s in c.subjects]
    # df = pd.read_excel("/DATA/NAS/datasets_processed/NAKO/notes/water_fat_inversion_mevibe.xlsx")
    # filtered_df = df[df["sub"].isin(subj)].sort_values(by="percent", ascending=False)
    # print(filtered_df)
    c = MEVIBE_dataset(256, gray=True, test=False, validation=False, create_dataset=True)
    print("Train", c.count_subj())
    # subj: list[str] = [str(s) for s in c.subjects]
    # df = pd.read_excel("/DATA/NAS/datasets_processed/NAKO/notes/water_fat_inversion_mevibe.xlsx")
    # filtered_df = df[df["sub"].isin(subj)].sort_values(by="percent", ascending=False)
    # print(filtered_df)

    c = MEVIBE_dataset(256, gray=True, test=False, validation=True, create_dataset=True)
    print("Validation", c.count_subj())
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
