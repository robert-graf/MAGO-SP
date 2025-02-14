# %reload_ext tensorboard
# %tensorboard --logdir lightning_logs/
import glob
import itertools
from collections import deque
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

# from torchmetrics import StructuralSimilarityIndexMeasure  # type: ignore
from models.pix2pix.cut.patchnce import PatchNCELoss, PatchSampleF
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure

from networks.backbone.optimizer_utils import LambdaLR, weights_init_normal
from utils.config_loading import instantiate_from_config


class Pix2Pix(pl.LightningModule):
    def __init__(
        self,
        in_channel,
        output_channels,
        lr,
        gan_config,
        discriminator_config,
        decay_epoch=None,
        lambda_paired=10.0,
        lambda_GAN=1.0,
        lambda_NCE=1.0,
        lambda_NCE_ID=0.0,
        lambda_ssim=1,
        batch_size: int | None = None,
        netF_nc=256,
        nce_T=0.07,
        mode: Literal["cut", "pix2pix", "paired_cut"] = "pix2pix",
        size: tuple[int, ...] = (256, 256),
        max_epochs=1000,
        start_epoch=0,
        nce_layers: list[int] = None,  # type: ignore  # noqa: RUF013
        num_patches=256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.logger: TensorBoardLogger
        self.output_channels = output_channels
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_NCE_ID = lambda_NCE_ID
        self.channels = in_channel
        self.lambda_ssim = lambda_ssim
        self.lambda_paired = lambda_paired
        self.batch_size = batch_size
        self.netF_nc = netF_nc
        self.nce_T = nce_T
        self.lr = lr
        self.decay_epoch = max_epochs // 2 if decay_epoch is None else decay_epoch
        self.max_epochs = max_epochs
        self.start_epoch = start_epoch
        self.num_patches = num_patches
        ### MODE SELECT ###
        self.use_contrastive = mode in ("cut", "paired_cut")
        self.use_paired = mode in ("pix2pix", "paired_cut")

        #### Initialize Models ####
        self.gan: torch.nn.Module = instantiate_from_config(gan_config)
        self.discriminator: torch.nn.Module = instantiate_from_config(discriminator_config)

        #### Initial Weights ####
        self.gan.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        #### Losses ####
        # Using LSGAN variants hardcoded ([vanilla| lsgan | wgangp]) TODO test Wasserstein GANs
        self.criterion_GAN = torch.nn.MSELoss()

        if self.use_paired:
            self.criterion_paired = torch.nn.L1Loss()
        ##### contrastive loss ######
        if self.use_contrastive:
            self.nce_layers = nce_layers
            assert self.nce_layers is not None
            # normal | xavier | kaiming | orthogonal
            self.patch_SampleF_MLP = PatchSampleF(use_mlp=True, init_type="normal", init_gain=0.02, nc=self.netF_nc)
            self.criterion_NCE = [PatchNCELoss(self.batch_size, self.nce_T).to(self.device) for _ in self.nce_layers]
            # A janky way to initialize but this was given to me...
            with torch.no_grad():
                if self.lambda_NCE > 0.0:
                    a = torch.zeros((self.batch_size, in_channel, *size), device=self.device)  # type: ignore
                    _ = self.calculate_NCE_loss(a, a)
                if self.lambda_NCE_ID and self.lambda_NCE > 0.0:
                    a = torch.zeros((self.batch_size, in_channel, *size), device=self.device)  # type: ignore
                    _ = self.calculate_NCE_loss(a, a)

        self.counter = 0
        self.automatic_optimization = False
        self.q = deque(maxlen=1000)

    def forward(self, x: Tensor) -> Tensor:
        return self.gan(x)

    def configure_optimizers(self) -> tuple[list[torch.optim.Adam], list[torch.optim.lr_scheduler.LambdaLR]]:
        para = itertools.chain(self.gan.parameters(), self.patch_SampleF_MLP.parameters()) if self.use_contrastive else self.gan.parameters()
        optimizer_G = torch.optim.Adam(para, lr=self.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if self.decay_epoch == -1:
            self.decay_epoch = self.max_epochs // 2
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(self.max_epochs, self.start_epoch, self.decay_epoch).step)
        lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(self.max_epochs, self.start_epoch, self.decay_epoch).step)

        return [optimizer_G, optimizer_D], [lr_scheduler_G, lr_scheduler_D]

    def training_step(self, train_batch, batch_idx):
        opt_g, opt_d = self.optimizers()  # type: ignore
        opt_g: torch.optim.Adam
        opt_d: torch.optim.Adam
        #### Get batch ####
        real_A = train_batch["condition"]
        real_B = train_batch["target"]

        assert real_A is not None
        assert (
            real_B.shape[1] == self.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {self.output_channels}"
        #### In case of multiple optimizer fork ###
        # Compute forward and loss. Log loss. return one loss value.
        opt_g.zero_grad()
        loss = self.training_step_G(real_A, real_B, batch_idx)
        self.manual_backward(loss)
        opt_g.step()

        # Compute forward and loss. Log loss. return one loss value.
        opt_d.zero_grad()
        loss = self.training_step_D(real_A, real_B)
        self.manual_backward(loss)
        opt_d.step()

    def training_step_G(self, real_A: Tensor, real_B: Tensor, _batch_idx):
        fake_B: Tensor = self.gan(real_A)
        # First, G(A) should fake the discriminator
        ZERO = torch.zeros(1, device=self.device)
        loss_G_GAN = ZERO
        fake = torch.cat([fake_B, real_A], dim=1) if self.use_paired else fake_B

        if self.lambda_GAN > 0.0:
            pred_fake: Tensor = self.discriminator(fake)
            real_label = torch.ones((pred_fake.shape[0], 1), device=self.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * self.lambda_GAN
        loss_NCE_both = 0
        if self.use_contrastive:
            if real_A.shape == real_B:
                idt: Tensor = self.gan(real_B)
            else:
                idt: Tensor = self.gan(torch.cat([real_B, real_A[:, 1:]], 1))

            loss_NCE = ZERO
            if self.lambda_NCE > 0.0:
                loss_NCE = self.calculate_NCE_loss(real_A, fake_B)

            loss_NCE_both = loss_NCE
            loss_NCE_Y = ZERO
            if self.lambda_NCE_ID and self.lambda_NCE > 0.0:
                loss_NCE_Y = self.calculate_NCE_loss(real_B, idt)
                loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
            self.log("train/GAN", loss_G_GAN.detach() / self.lambda_GAN)
            self.log("train/NCE", loss_NCE.detach() / self.lambda_NCE)
            self.log("train/NCE_Y", loss_NCE_Y.detach() / self.lambda_NCE)

        loss_paired = 0
        loss_ssim = 0
        if self.use_paired:
            loss_paired = self.criterion_paired(real_B, fake_B)
            self.log("train/loss_paired", loss_paired.detach())
            if self.lambda_ssim > 0.0:
                loss_ssim = self.lambda_ssim * (1 - structural_similarity_index_measure(real_B + 1, fake_B + 1, data_range=2.0))  # type: ignore
                self.log("train/loss_ssim", loss_ssim.detach())
            loss_paired = self.lambda_paired * (loss_ssim + loss_paired)
        self.fake_B_buffer = fake.detach()

        loss_G = loss_G_GAN + loss_NCE_both + loss_paired
        self.log("train/All", loss_G.detach().cpu())
        self.q.append(loss_G.detach().cpu())
        self.log("train_avg", sum(self.q) / len(self.q), prog_bar=True)
        return loss_G

    def training_step_D(self, real_A, real_B) -> Tensor:
        # Fake loss, will be fake_B if unpaired and fake_B||real_A if paired
        fake = self.fake_B_buffer
        pred_fake = self.discriminator(fake)

        assert not np.any(
            np.isnan(pred_fake.detach().cpu().numpy())  # type: ignore
        ), "NAN detected! (ʘᗩʘ'), if this happened at the start of your training, than the init is instable. Try again, or change init_type and try again."
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=self.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?

        # Real loss
        real = torch.cat([real_B, real_A], dim=1) if self.use_paired else real_B

        pred_real = self.discriminator(real)
        assert not np.any(np.isnan(pred_real.detach().cpu().numpy()))  # type: ignore
        real_label = torch.ones((pred_real.shape[0], 1), device=self.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.log("train/D_real", loss_D_real.detach())
        self.log("train/D_fake", loss_D_fake.detach())
        return loss_D

    def calculate_NCE_loss(self, src, tgt) -> Tensor:
        n_layers = len(self.nce_layers)
        feat_q = self.forward_GAN_with_Intermediate(tgt, self.nce_layers)
        feat_k = self.forward_GAN_with_Intermediate(src, self.nce_layers)
        feat_k_pool, sample_ids = self.patch_SampleF_MLP(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.patch_SampleF_MLP(feat_q, self.num_patches, sample_ids)
        ZERO = torch.zeros(1, device=self.device)
        total_nce_loss = ZERO
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterion_NCE, self.nce_layers, strict=False):
            loss = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def forward_GAN_with_Intermediate(self, input, target_layers) -> list[Tensor]:
        raise NotImplementedError()
        if isinstance(self.gan, Generator):  # self.opt.model_name == "resnet"
            if -1 in target_layers:
                target_layers.append(len(self.gan.model))
            assert len(target_layers)

            partial_forward = input
            feats = []
            for layer_id, layer in enumerate(self.gan.model):
                partial_forward = layer(partial_forward)
                if layer_id in target_layers:
                    feats.append(partial_forward)
                else:
                    pass
            return feats
        _, features = self.gan(input, return_intermediate=True, layers=target_layers)
        return features

    def validation_step(self, batch, batch_idx):
        real_A = batch["condition"]
        real_B = batch["target"]
        print(real_B.shape, real_A.shape)
        assert (
            real_B.shape[1] == self.output_channels
        ), f"real_a and output_channels are unequal. This feature is not supported every were. Shape:{real_A.shape}, output_channels = {self.output_channels}"
        fake_B = self.gan(real_A)
        assert real_A is not None
        # fake_id = self.gan(real_B)
        out = (
            [real_B[:, [i]] for i in range(real_B.shape[1])] + [fake_B[:, [i]] for i in range(fake_B.shape[1])] + [real_A[:, [i]] for i in range(real_A.shape[1])]
        )  # , fake_id, real_B
        out = [denormalize(i) for i in out]

        grid = torch.cat(out, dim=-1).cpu()
        grid = torchvision.utils.make_grid(grid, nrow=2)
        self.logger.experiment.add_image("A2B", grid, self.counter)
        self.counter += 1


def normalize(tensor) -> Tensor:
    return (tensor * 2) - 1  # map [0,1]->[-1,1]


def denormalize(tensor) -> Tensor:
    return torch.clamp((tensor + 1) / 2, 0, 1)  # map [-1,1]->[0,1]
