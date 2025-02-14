import os
import pickle
import random
from functools import partial
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageSR(Dataset):
    def __init__(
        self,
        size: int,
        gray=False,
        class_labels=False,
        validation=False,
        vflip=True,
        hflip=True,
        dflip=False,
        rotation=None,
        random_zoom=False,
        zoom_min=0.8,
        zoom_max=1.2,
        padding="constant"
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
        
        self.size = (size, size)
        dataset_path = "/media/data/robert/datasets/multimodal_large/"
        if os.path.exists("/DATA/NAS/datasets_processed/Natural/multimodal_large/"):
            dataset_path = "/DATA/NAS/datasets_processed/Natural/multimodal_large/"
        labels_pkl = os.path.join(dataset_path, "labels.pkl")

        if not os.path.exists(labels_pkl):
            self.labels_name = sorted([p.name for p in Path(dataset_path + "train").iterdir() if p.is_dir()])
            with open(labels_pkl, "wb") as f:
                pickle.dump(self.labels_name, f)
            print(self.labels_name)
        else:
            with open(labels_pkl, "rb") as f:
                self.labels_name = pickle.load(f)

        dataset_path += "train" if not validation else "val"

        buffer_path = os.path.join(dataset_path, ".buffer.pkl")

        # Check if the buffer file exists
        if os.path.exists(buffer_path):
            # Load the buffer file
            with open(buffer_path, "rb") as f:
                file_list = pickle.load(f)
            print(f"Loaded file list from buffer. n = {len(file_list)}")
        else:
            # Traverse the directory and collect all file paths
            file_list: list[tuple[str, int]] = []
            for p in Path(dataset_path).iterdir():
                if not p.is_dir():
                    continue
                idx = self.labels_name.index(p.name)
                print(p.name)
                for root, _, files in os.walk(p):
                    for file in files:
                        if ".png" in file or ".jpg" in file:
                            file_list.append((os.path.join(root, file), idx))

            # Save the list to a pickle file
            with open(buffer_path, "wb") as f:
                pickle.dump(file_list, f)
            print("Buffer file created and file list saved:", buffer_path)
        self.file_list: list[tuple[str, int]] = file_list

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

    def __len__(self):
        return len(self.file_list)

    def data_argumentation(self, target):
        # Random zoom (RandomResizedCrop)
        if self.random_zoom:
            scale = (self.zoom_min, self.zoom_max)
            target = transforms.RandomResizedCrop(self.size, scale=scale)(target)
        # Padding
        w, h = target.shape[-2], target.shape[-1]
        hp = max((self.size[0] - w) / 2, 0)
        vp = max((self.size[1] - h) / 2, 0)
        padding = [int(floor(vp)), int(floor(hp)), int(ceil(vp)), int(ceil(hp))]
        if self.rotation:
            angle = random.uniform(-self.rotation, self.rotation)  # Random rotation within range
            target = tf.rotate(target, angle, tf.InterpolationMode.BILINEAR)
        target = tf.pad(target, padding, padding_mode=self.padding)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(target, output_size=self.size)  # type: ignore
        target = tf.crop(target, i, j, h, w)

        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            target = tf.hflip(target)

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            target = tf.vflip(target)
        # Random vertical flipping
        if self.dflip and random.random() > 0.5:
            target = target.swapaxes(-1, -2)
        return target

    def __getitem__(self, i):
        path, idx = self.file_list[i]
        image_pil = Image.open(path)  # type: ignore

        if image_pil.mode != "RGB":
            if not self.gray:
                image_pil = image_pil.convert("RGB")
        else:
            assert not self.gray
        image = tf.pil_to_tensor(image_pil).to(torch.uint8)

        # Add padding
        # image, _ = self.padd(image)
        # image = self.cropper(image)
        # image: torch.Tensor = self.image_rescale(image)

        # if self.pil_interpolation:
        #    lr_image = image.clone()
        #    lr_image = self.degradation_process(lr_image)  # type: ignore
        #    lr_image = tf.pil_to_tensor(lr_image).to(torch.uint8)
        # else:
        #    lr_image = image

        img = (image.to(torch.float32) / 127.5 - 1.0).to(torch.float32)
        img = self.data_argumentation(img)
        img = img.permute((1, 2, 0)).to(torch.float32).clone().numpy()

        example: dict = {
            "image": img
        }
        # example["LR_image"] = (lr_image / 127.5 - 1.0).to(torch.float32).permute((1, 2, 0))  # type: ignore
        if self.class_labels:
            example["class_label"] = idx
            example["human_label"] = self.labels_name[idx]
        return example


class ImageSRTrain(ImageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ImageSRValidation(ImageSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, validation=True)


if __name__ == "__main__":
    c = ImageSR(256, gray=True)
    print(c[0]["image"].shape, c[0]["image"].dtype)
    assert c[0]["image"].shape[-1] == 1
    c = ImageSR(256, gray=True, validation=True)
    print("classes", len(c.labels_name))
