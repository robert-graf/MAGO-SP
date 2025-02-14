from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from networks.backbone.ema import LitEma
from networks.utils import instantiate_from_config


class BaseLightningModule(pl.LightningModule):
    def __init__(self, use_ema, back_bone_config, dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_dims = dims
        self.model: torch.nn.Module = instantiate_from_config(back_bone_config, dims=dims)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self._buffer_dict = {}

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]["lr"]  # type: ignore
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def log_dict(
        self,
        dictionary: dict,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        rank_zero_only: bool = False,
    ) -> None:
        d = {}
        for k, v in dictionary.items():
            if isinstance(v, torch.Tensor):
                v = v.mean().clone().detach().cpu().numpy().item()  # noqa: PLW2901

            if k not in self._buffer_dict:
                self._buffer_dict[k] = []
            self._buffer_dict[k].append(v)
            if len(self._buffer_dict[k]) == 101:
                (self._buffer_dict[k]).pop(0)
            d[k] = np.mean(np.array(self._buffer_dict[k])).item()

        super().log_dict(d, prog_bar, logger, on_step, on_epoch, reduce_fx, enable_graph, sync_dist, sync_dist_group, add_dataloader_idx, batch_size, rank_zero_only)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *_args, **_kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = [*params, self.logvar]
        return torch.optim.AdamW(params, lr=lr)

    def get_input(self, batch: dict[str, torch.Tensor], k: str | list[str]):
        try:
            x = batch[k] if isinstance(k, str) else torch.concat([batch[k] for k in k], 1)
        except KeyError:
            print(batch.keys(), k)
            raise
        if len(x.shape) == self.n_dims:
            raise ValueError(f"Shape missmatch {x.shape}, {(self.n_dims+1)=}")
            # x = x[None, ...]
        # assert x.shape[1] == self.channels if self.channels is not None else True, x.shape
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
