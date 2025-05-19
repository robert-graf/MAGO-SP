import math
import os
import random
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from TPTBox import Print_Logger, to_nii

# sys.path.append(str(Path(__file__).parent.parent))

# from inference.inference_all import get_all_files_of_subj

sub_list = []  # blinded subject list
random.seed(999)
logger = Print_Logger()


def rand_perlin_3d(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    """
    Generate 3D Perlin noise using given shape and resolution.

    Args:
        shape (tuple): Shape of the 3D volume.
        res (tuple): Resolution of the Perlin grid.
        fade (callable): Fade function for interpolation smoothing.

    Returns:
        torch.Tensor: 3D tensor of Perlin noise.
    """
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])

    grid = (
        torch.stack(
            torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1]), torch.arange(0, res[2], delta[2]), indexing="ij"),
            dim=-1,
        )
        % 1
    )

    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1, slice2, slice3):
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1], slice3[0] : slice3[1]]
            .repeat_interleave(d[0], 0)
            .repeat_interleave(d[1], 1)
            .repeat_interleave(d[2], 2)
        )

    def dot(grad, shift):
        return (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], : shape[2], 0] + shift[0],
                    grid[: shape[0], : shape[1], : shape[2], 1] + shift[1],
                    grid[: shape[0], : shape[1], : shape[2], 2] + shift[2],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1], : shape[2]]
        ).sum(dim=-1)

    # Compute noise at each corner of the cube
    n000 = dot(tile_grads([0, -1], [0, -1], [0, -1]), [0, 0, 0])
    n100 = dot(tile_grads([1, None], [0, -1], [0, -1]), [-1, 0, 0])
    n010 = dot(tile_grads([0, -1], [1, None], [0, -1]), [0, -1, 0])
    n110 = dot(tile_grads([1, None], [1, None], [0, -1]), [-1, -1, 0])
    n001 = dot(tile_grads([0, -1], [0, -1], [1, None]), [0, 0, -1])
    n101 = dot(tile_grads([1, None], [0, -1], [1, None]), [-1, 0, -1])
    n011 = dot(tile_grads([0, -1], [1, None], [1, None]), [0, -1, -1])
    n111 = dot(tile_grads([1, None], [1, None], [1, None]), [-1, -1, -1])

    t = fade(grid[: shape[0], : shape[1], : shape[2]])

    n0 = torch.lerp(torch.lerp(n000, n100, t[..., 0]), torch.lerp(n010, n110, t[..., 0]), t[..., 1])
    n1 = torch.lerp(torch.lerp(n001, n101, t[..., 0]), torch.lerp(n011, n111, t[..., 0]), t[..., 1])
    return math.sqrt(3) * torch.lerp(n0, n1, t[..., 2])


def make_sample(idx, out_base: Path, data):
    """
    Generate a single synthetic training sample with optional Perlin noise mixing and save it.

    Args:
        idx (int): Subject index.
        out_base (Path): Output base directory.
        data (dict): Dataset settings.
    """
    if (out_base / f"labelsTr/{idx:04}.nii.gz").exists() and (out_base / f"imagesTr/{idx:04}_{2:04}.nii.gz").exists():
        return
    import random

    zoom = [float(z) for z in data["spacing"][::-1]]
    random.seed(idx)
    np.random.seed(idx)
    while True:
        try:
            out = get_all_files_of_subj(sub_list[idx])  # TODO Load known-good data
            if idx % 2 == 0:
                if "water" in out:
                    water_fraction = out["water"]
                    water_img = water_fraction
                    fat_fraction = out["fat"]
                    eco0_opp1 = out["inphase"]
                    eco1_pip1 = out["outphase"]
                    water_img_nii = to_nii(water_img)
                    water_img_nii.assert_affine(to_nii(water_fraction))
                    water_img_nii.assert_affine(to_nii(fat_fraction))
                    water_img_nii.assert_affine(to_nii(eco0_opp1))
                    water_img_nii.assert_affine(to_nii(eco1_pip1))
                    break
            elif "water-fraction-mevibe" in out:
                water_img = out["water-mevibe"]
                water_fraction = out["water-fraction-mevibe"]
                fat_fraction = out["fat-fraction-mevibe"]
                eco0_opp1 = out["eco0-opp1-mevibe"]
                eco1_pip1 = out["eco1-pip1-mevibe"]
                water_img_nii = to_nii(water_img)
                water_img_nii.assert_affine(to_nii(water_fraction))
                water_img_nii.assert_affine(to_nii(fat_fraction))
                water_img_nii.assert_affine(to_nii(eco0_opp1))
                water_img_nii.assert_affine(to_nii(eco1_pip1))
                break

        except Exception:
            pass

    from TPTBox.segmentation.TotalVibeSeg import run_totalvibeseg

    ###########################################################################################
    ################ Make Synthetic Sample from Perlin Noise ##################################
    ###########################################################################################
    roi_path = water_img.get_changed_path(info={"seg": "roi"}, bids_format="msk", parent="derivatives-totalvibeseg", non_strict_mode=True)
    run_totalvibeseg(water_img, roi_path, dataset_id=278)  # lock in https://github.com/robert-graf/TotalVibeSegmentator for the newest ID
    roi = to_nii(roi_path, True).clamp(0, 1).reorient(("L", "P", "S"))
    roi.rescale_(zoom)
    fat_fraction = to_nii(fat_fraction, False).reorient(("L", "P", "S")).rescale_(zoom)
    water_fraction = to_nii(water_fraction, False).reorient(("L", "P", "S")).rescale_(zoom)
    eco0_opp1 = to_nii(eco0_opp1).reorient(("L", "P", "S")).rescale_(zoom)
    eco1_pip1 = to_nii(eco1_pip1).reorient(("L", "P", "S")).rescale_(zoom)
    crop = None  # type: ignore
    if idx % 2 == 0:
        axis = roi.get_axis("S")

        from TPTBox.core.np_utils import _select_axis_dynamically

        off = random.randint(0, roi.shape[axis] - 64)
        crop: tuple[slice, ...] = _select_axis_dynamically(axis, slice(off, off + 80))  # type: ignore
        print(crop)

    if crop is not None:
        roi.apply_crop_(crop)
        fat_fraction.apply_crop_(crop)
        water_fraction.apply_crop_(crop)
        eco0_opp1.apply_crop_(crop)
        eco1_pip1.apply_crop_(crop)
        print("CROP", roi.shape)

    arr = rand_perlin_3d(fat_fraction.shape, res=(1, 1, 1)).numpy()
    if random.random() > 0.5:
        arr += np.random.randn(*fat_fraction.shape) * 0.05 * random.random()
    if random.random() > 0.5:
        arr = np.roll(arr, random.randint(0, arr.shape[0]), axis=random.randint(0, len(arr.shape) - 1))
    arr -= arr.min()
    arr /= arr.max()
    a = random.random() * 0.7 + 0.15
    arr[arr > a] = 1
    arr[arr <= a] = 0
    brocken_img = fat_fraction.set_array(fat_fraction.get_array() * arr + water_fraction.get_array() * (1 - arr))
    gt = roi * (arr + 1)
    gt = fat_fraction.set_array(gt)
    try:
        logger.on_ok(eco1_pip1, idx)
        out_d = {"seg": gt, "img": brocken_img, "img2": eco0_opp1, "img3": eco1_pip1}

        img_num = -1
        for _, (name, nii) in enumerate(out_d.items()):
            out = out_base
            if name == "seg":
                out = out / f"labelsTr/{idx:04}.nii.gz"
                print("seg", out)
            else:
                img_num += 1
                out = out / f"imagesTr/{idx:04}_{img_num:04}.nii.gz"
                print(img_num, out)
            Path(out).parent.mkdir(exist_ok=True, parents=True)
            assert out_d["seg"].shape == nii.shape, (out_d, nii)
            nii.save(out)
    except Exception:
        logger.on_fail("FAILED", eco1_pip1)
        logger.print_error()
        raise


def run_make_ds(n, idx, make_sample=make_sample, cpu=os.cpu_count() // 2 + 3, setting=None):
    """
    Generate a full synthetic dataset and save in nnUNet format.

    Args:
        n (int): Number of samples to generate.
        idx (int): Dataset index.
        make_sample (callable): Sample creation function.
        cpu (int): Number of CPUs to use in parallel.
        setting (dict, optional): Additional dataset config to include in dataset.json.
    """
    if setting is None:
        setting = {}
    with Pool(cpu) as p:
        out_base = Path(__file__).parent.parent / f"nnUNet/nnUNet_raw/Dataset{idx:03}/"
        Path(out_base).mkdir(exist_ok=True, parents=True)
        import json

        files = {"0": "fat_fraction", "1": "eco0-opp1", "2": "eco1-pip1"}
        labels = {"background": 0, "water_fraction": 1, "fat_fraction": 2}
        data = {
            "channel_names": files,
            "labels": labels,
            "numTraining": n,
            "file_ending": ".nii.gz",
            "reference": "deep-spine.de",
            "licence": "https://github.com/robert-graf/TotalVibeSegmentator",
            "regions_class_order": list(labels.values()),
            "dataset_release": "1.0",
            "orientation": ("L", "P", "S"),
            "spacing": ("3.0", "1.641", "1.641"),
            "turn_on_mirroring": True,
            **setting,
        }
        with open(out_base / "dataset.json", "w") as f:
            json.dump(data, f, indent=4)
        p.map(partial(make_sample, out_base=out_base, data=data), range(n))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    run_make_ds(n=500, idx=282)
