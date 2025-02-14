import glob
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

_interop = False


def get_device(ddevice, gpu, init_threads=False):
    if ddevice == "cpu":
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count()) if init_threads else None
        device = torch.device("cpu")
    elif ddevice == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        try:
            torch.set_num_threads(1) if init_threads else None
            global _interop  # noqa: PLW0603
            if not _interop:
                torch.set_num_interop_threads(1) if init_threads else None
                _interop = True
        except Exception as e:
            print(e)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("mps")
    return device


def get_config(resume: str | Path):
    if str(resume).startswith("logs/"):
        resume = Path(__file__).parent / resume

    resume = Path(resume)
    if not resume.exists():
        raise ValueError(f"Cannot find {resume}")

    if resume.is_file():
        logdir = resume.parent.parent
        ckpt = resume
    else:
        logdir = resume
        ckpt = resume / "checkpoints" / "last.ckpt"
        assert ckpt.exists(), ckpt
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    if len(base_configs) == 0:
        base_configs = [os.path.join(logdir, "config.yaml")]
        assert Path(base_configs[0]).exists(), base_configs

    configs = [OmegaConf.load(c) for c in base_configs]  # type: ignore
    config: dict = OmegaConf.merge(*configs)  # type: ignore
    m = config["model"]
    m["params"]["ckpt_path"] = ckpt
    return config
