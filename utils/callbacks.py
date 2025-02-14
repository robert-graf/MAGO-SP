import os
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.rank_zero import rank_zero_only

########################################################################################


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, _pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, _pl_module):
        if trainer.global_rank == 0:
            # Create log dirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config and "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]:
                os.makedirs(os.path.join(self.ckptdir, "trainstep_checkpoints"), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))

        elif not self.resume and os.path.exists(self.logdir):
            dst, name = os.path.split(self.logdir)
            dst = os.path.join(dst, "child_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            try:
                os.rename(self.logdir, dst)
            except FileNotFoundError:
                pass


########################################################################################


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        max_num_img=100,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {CSVLogger: self._tensorboard}
        self.log_steps = [1] + [2**n * 100 for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.max_num_img = max_num_img
        self.img_files: list[list[Path]] = []
        self.first_epoch = True

    @rank_zero_only
    def _tensorboard(self, pl_module, images, _batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            try:
                pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)
            except AttributeError:
                from pprint import pprint

                pprint(vars(pl_module.logger))
                raise

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        path_list = []
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = f"gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}_{k}.png"
            path = Path(root, filename)
            path.parent.mkdir(exist_ok=True, parents=True)
            path_list.append(path)
            Image.fromarray(grid).save(path)
        # Delete image if there are to many
        if len(self.img_files) >= self.max_num_img:
            l = self.img_files.pop(random.randint(0, int(self.max_num_img * 0.8)))
            for x in l:
                [x.unlink(missing_ok=True) for x in l]

        self.img_files.append(path_list)

    def log_img(self, pl_module: pytorch_lightning.LightningModule, batch, batch_idx, split="train"):
        self.first_epoch = False
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.check_frequency(check_idx) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images: list[torch.Tensor] = pl_module.log_images(batch, split=split, **self.log_images_kwargs)  # type: ignore

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)  # type: ignore

            logger_log_images = self.logger_log_images.get(logger, lambda *_args, **_kwargs: None)  # type: ignore
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm") and (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
            self.log_gradients(trainer, pl_module, batch_idx=batch_idx)  # type: ignore


########################################################################################


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):  # noqa: ARG002
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(pl_module.device)
        torch.cuda.synchronize(pl_module.device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(pl_module.device)
        max_memory = torch.cuda.max_memory_allocated(pl_module.device) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)  # type: ignore
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)  # type: ignore

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
