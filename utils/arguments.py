import argparse
import datetime
import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

from utils.config_loading import instantiate_from_config


@dataclass()
class MainOpt:
    config: list[str]
    seed: int
    logdir_name: str
    name: str
    resume: str
    resume_from_checkpoint: str | None = None
    logdir: Path = None  # type: ignore
    gpus: list[int] = field(default_factory=list)
    verbose: bool = False
    now: str | None = None
    accelerator: str = "auto"

    @property
    def ckptdir(self):
        return self.logdir / "checkpoints"

    @property
    def cfgdir(self):
        return self.logdir / "configs"

    def get_config(self):
        if self.now is None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.now = now

        configs = [OmegaConf.load(c) for c in self.config]  # type: ignore
        config: dict = OmegaConf.merge(*configs)  # type: ignore
        model_type = config.get("model_type", "model")
        additional_name = config.get("additional_name", "")
        if additional_name != "":
            additional_name = "_" + additional_name
        name = self.name if self.name != "" else Path(self.config[0]).name.split(".")[0]
        self.logdir = Path(__file__).parent.parent / self.logdir_name / model_type / f"{now}-{name}{additional_name}"
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.validate_resume()
        self.cfgdir.mkdir(exist_ok=True)
        OmegaConf.save(config, self.cfgdir / "config.yaml")

        return config

    def validate_resume(self):
        if self.now is None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.now = now

        if self.name and self.resume:
            raise ValueError(
                "-n/--name and -r/--resume cannot be specified both."
                "If you want to resume training in a new log folder, "
                "use -n/--name in combination with --resume_from_checkpoint"
            )
        if self.resume:
            if not os.path.exists(self.resume):
                raise ValueError(f"Cannot find {self.resume}")
            if os.path.isfile(self.resume):
                paths = self.resume.split("/")
                logdir = "/".join(paths[:-2])
                ckpt = self.resume
            else:
                assert os.path.isdir(self.resume), self.resume
                logdir = self.resume.rstrip("/")
                ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

            self.resume_from_checkpoint = ckpt
            base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
            if len(base_configs) == 0:
                base_configs = [os.path.join(logdir, "config.yaml")]
                assert Path(base_configs[0]).exists(), base_configs

            self.config = base_configs + self.config
            _tmp = logdir.split("/")
            now_name = _tmp[-1]
        else:
            if self.name:
                name = "_" + self.name
            elif self.config:
                cfg_fname = os.path.split(self.config[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = "_" + cfg_name
            else:
                name = ""
            now_name = self.now + name  # + opt.postfix
            logdir = os.path.join(self.logdir, now_name)
        return now_name, logdir


def reload_model(resume):
    if not os.path.exists(resume):
        raise ValueError(f"Cannot find {resume}")
    if os.path.isfile(resume):
        paths = str(resume).split("/")
        logdir = "/".join(paths[:-2])
        ckpt = resume
    else:
        assert os.path.isdir(resume), resume
        logdir = resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    if len(base_configs) == 0:
        base_configs = [os.path.join(logdir, "config.yaml")]
        assert Path(base_configs[0]).exists(), base_configs
    from utils.config_loading import instantiate_from_config

    configs = [OmegaConf.load(c) for c in base_configs]  # type: ignore
    config: dict = OmegaConf.merge(*configs)  # type: ignore
    m = config["model"]
    m["params"]["ckpt_path"] = ckpt
    return m


def get_opt(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-c", "--config", type=Path, nargs="*", default=[])

    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-acc", "--accelerator", type=str, default="auto")
    # help_ = "paths to base configs. Loaded from left-to-right. " "Parameters can be overwritten or added with command-line options of the form `--key value`."
    # parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", help=help_, default=[])
    # parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?", help="train")
    # parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    # parser.add_argument("-p", "--project", help="name of new or path to existing project")
    # parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=42, help="seed for seed_everything")
    parser.add_argument("-gpu", "--gpus", type=int, nargs="*")
    # parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir_name", type=str, default="logs", help="directory for logging")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Enable verbose logging")
    # parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by n_gpu * batch_size * n_accumulate")
    p = parser.parse_args()
    return MainOpt(config=p.config, seed=p.seed, logdir_name=p.logdir_name, name=p.name, gpus=p.gpus, resume=p.resume, verbose=p.verbose, accelerator=p.accelerator)  # type: ignore


def add_default_config(config, opt: MainOpt, model):
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) if "trainer" in lightning_config else config.get("trainer", OmegaConf.create())
    trainer_config = {**trainer_config}
    if opt.gpus is None:
        opt.gpus = []
    if len(opt.gpus) == 0:
        cpu = True
    else:
        trainer_config["accelerator"] = opt.accelerator
        gpu_info = opt.gpus
        print(f"Running on GPUs {gpu_info}")
        cpu = False
    lightning_config.trainer = trainer_config

    # default logger configs
    default_logger_cfg = {"target": "pytorch_lightning.loggers.TensorBoardLogger", "params": {"name": "tbl", "save_dir": opt.logdir}}
    logger_cfg = lightning_config.logger if "logger" in lightning_config else OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

    trainer_config["logger"] = instantiate_from_config(logger_cfg)  # type: ignore

    default_model_ckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {"dirpath": opt.ckptdir, "filename": "{epoch:06}", "verbose": True, "save_last": True},
    }
    assert hasattr(model, "monitor")
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.") if opt.verbose else None
        default_model_ckpt_cfg["params"]["monitor"] = model.monitor
        default_model_ckpt_cfg["params"]["save_top_k"] = 3

    model_ckpt_cfg = lightning_config.modelcheckpoint if "modelcheckpoint" in lightning_config else OmegaConf.create()
    model_ckpt_cfg = OmegaConf.merge(default_model_ckpt_cfg, model_ckpt_cfg)
    print(f"Merged modelckpt-cfg: \n{model_ckpt_cfg}") if opt.verbose else None

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "utils.callbacks.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": opt.now,
                "logdir": opt.logdir,
                "ckptdir": opt.ckptdir,
                "cfgdir": opt.cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "image_logger": {"target": "utils.callbacks.ImageLogger", "params": {"batch_frequency": 1000, "max_images": 8, "clamp": True}},
        "learning_rate_logger": {"target": "pytorch_lightning.callbacks.LearningRateMonitor", "params": {"logging_interval": "step"}},
        "cuda_callback": {"target": "utils.callbacks.CUDACallback"},
    }
    default_callbacks_cfg.update({"checkpoint_callback": model_ckpt_cfg})
    callbacks_cfg = lightning_config.callbacks if "callbacks" in lightning_config else OmegaConf.create()

    if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
        print("Caution: Saving checkpoints every n train steps without deleting. This might require some free space.")
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": opt.ckptdir / "trainstep_checkpoints",
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_config, "resume_from_checkpoint"):
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_config.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]  # type: ignore

    trainer_config["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]  # type: ignore
    return trainer_config, lightning_config, cpu
