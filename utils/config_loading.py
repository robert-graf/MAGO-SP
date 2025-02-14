import importlib

from omegaconf import OmegaConf

blanks = ("__is_first_stage__", "__is_unconditional__")


def instantiate_from_config(config: dict):
    if "config_path" in config:
        config_new = OmegaConf.load(config["config_path"])["model"]  # type: ignore
        OmegaConf.merge(config_new, config)
    if "target" not in config:
        if config in blanks:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


module_replacements = {
    "ddpm": "networks.models.diffusion.ddpm",
    "ddim": "networks.models.diffusion.utils.ddim_sampler",
    "LatentDiffusion": "networks.models.diffusion.latent_diffusion",
    "attention": "networks.backbone.substructures.attention",
    "openaimodel": "networks.backbone.openaimodel",
    "util": "networks.backbone.substructures.nd_layers",
    "ema": "networks.backbone.ema",
    "autoencoder": "networks.models.autoencoders.autoencoder",
}


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except Exception:
        print(module, "-->", module_replacements.get(module.split(".")[-1]))
        module = module_replacements.get(module.split(".")[-1])

        return getattr(importlib.import_module(module, package=None), cls)
