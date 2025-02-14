import os
import pickle
import random
from collections.abc import Sequence
from functools import partial
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(
        self,
        size: int | tuple[int, int],
        gray=False,
        class_labels=False,
        vflip=True,
        hflip=True,
        dflip=False,
        rotation=None,
        random_zoom=False,
        zoom_min=0.8,
        zoom_max=1.2,
        padding="constant",
        linspace=False,
    ):
        """
        Super-resolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop

        :param size: resizing to size after cropping
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        # Pad to the larger dimension and then resize
        self.class_labels = class_labels
        self.vflip = vflip
        self.hflip = hflip
        self.dflip = dflip
        self.padding = padding
        self.rotation = rotation
        self.random_zoom = random_zoom
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.gray = gray
        self.linspace = linspace
        scale = (self.zoom_min, self.zoom_max)
        self.resize_corp = transforms.RandomResizedCrop(self.size, scale=scale)

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
                    v /= max(float(v.max()), 0.0000001)
                    v = 2 * v - 1
                dict_mods[k] = v
            f.close()  # type: ignore
            return dict_mods
        if end in ("jpg", "png", "jepg"):
            image_pil = Image.open(path)  # type: ignore
            if not self.gray and image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            image = tf.pil_to_tensor(image_pil).to(torch.uint8)
            if norm:
                image = image.to(torch.float32) / 127.5 - 1.0
            return {default_key: image.to(torch.float32)}

        assert False, f"Expected a not a {end} file; {path}"

    def data_argumentation_3D():
        raise NotImplementedError()
        # l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        # l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        # l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        # l1 = Tensor(l1).permute(2, 0, 1)
        # l2 = Tensor(l2).permute(0, 2, 1)
        # l3 = Tensor(l3)

    def data_argumentation_2D(self, target2: dict[str, torch.Tensor], segmentations: Sequence[str] = ()):
        # Separate target and segmentation tensors
        target_items = {k: v for k, v in target2.items() if k not in segmentations}
        segmentation_items = {k: v for k, v in target2.items() if k in segmentations}

        # Stack tensors for targets and segmentations separately
        target = torch.Tensor(np.stack(list(target_items.values()), 0))
        original_shape = target.shape[-2:]
        segmentation = torch.Tensor(torch.stack(list(segmentation_items.values()), 0)) if len(segmentation_items) != 0 else None
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            segmentation = segmentation.unsqueeze(1) if segmentation is not None else None
        # Random zoom (RandomResizedCrop) for both target and segmentation
        if self.random_zoom:
            c_t = self.resize_corp
            i, j, h, w = c_t.get_params(target, c_t.scale, c_t.ratio)  # type: ignore
            tf.resized_crop(target, i, j, h, w, c_t.size, c_t.interpolation, antialias=c_t.antialias)  # type: ignore
            tf.resized_crop(segmentation, i, j, h, w, c_t.size, c_t.interpolation, antialias=c_t.antialias) if segmentation is not None else None  # type: ignore

        # Padding
        w, h = target.shape[-2], target.shape[-1]
        hp = max((self.size[0] - w) / 2, 0)
        vp = max((self.size[1] - h) / 2, 0)
        padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]
        target = tf.pad(target, padding, padding_mode=self.padding)
        segmentation = tf.pad(segmentation, padding, padding_mode=self.padding) if segmentation is not None else None

        # Random rotation for both target and segmentation
        if self.rotation:
            angle = random.uniform(-self.rotation, self.rotation)
            target = tf.rotate(target, angle, tf.InterpolationMode.BILINEAR)
            segmentation = tf.rotate(segmentation, angle, tf.InterpolationMode.NEAREST) if segmentation is not None else None

        # Random crop for both target and segmentation
        i, j, h, w = transforms.RandomCrop.get_params(target, output_size=self.size)  # type: ignore
        target = tf.crop(target, i, j, h, w)
        segmentation = tf.crop(segmentation, i, j, h, w) if segmentation is not None else None

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            target = tf.hflip(target)
            segmentation = tf.hflip(segmentation) if segmentation is not None else None

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            target = tf.vflip(target)
            segmentation = tf.vflip(segmentation) if segmentation is not None else None

        # Random diagonal flipping
        if self.dflip and random.random() > 0.5:
            target = target.swapaxes(-1, -2)
            segmentation = segmentation.swapaxes(-1, -2) if segmentation is not None else None
        # Reassemble targets and segmentations into a single dictionary with original keys
        processed_target = {key: target[i] for i, key in enumerate(target_items.keys())}
        processed_segmentation = {key: segmentation[i] for i, key in enumerate(segmentation_items.keys())} if segmentation is not None else {}

        if self.linspace:
            l1 = np.tile(np.linspace(0, 1, original_shape[0]), (1, original_shape[1]))
            l2 = np.tile(np.linspace(0, 1, original_shape[1]), (original_shape[0], 1))
            processed_target["linspace1"] = tf.crop(torch.Tensor(l1), i, j, h, w)
            processed_target["linspace2"] = tf.crop(torch.Tensor(l2), i, j, h, w)

        return {**processed_target, **processed_segmentation}
