import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from TPTBox import NII, Image_Reference, to_nii
from TPTBox.segmentation.VibeSeg.auto_download import _download

root = str(Path(__file__).parents[2])
sys.path.append(root)
import networks
import networks.models
import networks.models.diffusion
import networks.models.diffusion.ddpm
from utils.config_loading import instantiate_from_config
from utils.reload import get_config, get_device


def _norm_input(s_magnitude: Sequence[Image_Reference], cond, is_3d=False):
    def _help(i: Image_Reference):
        n = to_nii(i)
        v = n.get_array() - n.min()
        v = v / 1000
        v = 2 * v - 1
        v = torch.from_numpy(v)
        if is_3d:
            return v.unsqueeze(0).unsqueeze(0)
        return v.unsqueeze(1)

    return {b: _help(batch) for b, batch in zip(cond, s_magnitude, strict=True)}


# @torch.no_grad()
# def _inference(model: networks.models.diffusion.ddpm.DDPM, batch: dict, ref: NII, ddim_steps=250, ddevice="cuda"):
#    batch = {k: v.to(model.device) for k, v in batch.items()}
#    with torch.autocast(device_type=ddevice, dtype=torch.float16):
#        shape = (ref.shape[0], 1, *ref.shape[1:])
#        with model.ema_scope("Plotting"):
#            samples, _ = model.sample(
#                batch, batch_size=shape[0], return_intermediates=False, ddim=True, ddim_steps=ddim_steps, shape=shape, clamp=lambda x: torch.clamp(x, -1, 1)
#            )
#        arr: torch.Tensor = (samples + 1) * 500
#        nii = ref.set_array(arr.squeeze(1).cpu().numpy())
#        nii.clamp_(min=0)
#        return nii
#
def pad_to_divisible_by_16(tensor: torch.Tensor, is_3d: bool = False, size=16):
    """
    Pads the spatial dimensions of a tensor to make them divisible by 16.

    Args:
        tensor (torch.Tensor):
            Input tensor of shape (N, C, H, W) if is_3d=False,
            or (N, C, D, H, W) if is_3d=True.
        is_3d (bool): Whether the tensor is 3D (with depth).

    Returns:
        torch.Tensor: Padded tensor.
        tuple: Padding applied for each spatial dimension.
            - 2D: ((pad_left, pad_right), (pad_top, pad_bottom))
            - 3D: ((pad_left, pad_right), (pad_top, pad_bottom), (pad_front, pad_back))
    """
    if is_3d:
        depth, height, width = tensor.shape[-3:]

        pad_d = (size - depth % size) % size
        pad_h = (size - height % size) % size
        pad_w = (size - width % size) % size

        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_tensor = torch.nn.functional.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back),  # (W, H, D)
            mode="constant",
            value=0,
        )

        return padded_tensor, ((pad_left, pad_right), (pad_top, pad_bottom), (pad_front, pad_back))

    height, width = tensor.shape[-2:]

    pad_h = (size - height % size) % size
    pad_w = (size - width % size) % size

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_tensor = torch.nn.functional.pad(
        tensor,
        (pad_left, pad_right, pad_top, pad_bottom),  # (W, H)
        mode="constant",
        value=0,
    )
    return padded_tensor, ((pad_left, pad_right), (pad_top, pad_bottom))


def crop_to_original(tensor: torch.Tensor, padding: tuple):
    """
    Crops the last two dimensions of a tensor to remove previously added padding.

    Args:
        tensor (torch.Tensor): Input padded tensor.
        padding (tuple): Padding applied as ((pad_left, pad_right), (pad_top, pad_bottom)).

    Returns:
        torch.Tensor: Cropped tensor.
    """
    pad_left, pad_right = padding[0]
    pad_top, pad_bottom = padding[1]

    return tensor[..., pad_top : tensor.shape[-2] - pad_bottom, pad_left : tensor.shape[-1] - pad_right]


@torch.no_grad()
def _inference(
    model: networks.models.diffusion.ddpm.DDPM,
    batch: dict,
    ref: NII,
    ddim_steps=250,
    ddevice="cuda",
):
    batch_size = next(iter(batch.values())).size(0)  # Assuming all inputs have the same batch size
    batch = {k: pad_to_divisible_by_16(v.to(model.device)) for k, v in batch.items()}
    # Estimate sub-batch size from currently free GPU RAM (not the "ever
    # allocated" high-water mark, which under-estimates free memory badly
    # after the first call and lets the multiplier below OOM the sampler).
    # `mem_get_info` returns (free, total) in bytes for the current device.
    if str(model.device).startswith("cuda"):
        try:
            device_idx = model.device.index if getattr(model.device, "index", None) is not None else 0
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_idx)
        except Exception:  # noqa: BLE001
            free_bytes = torch.cuda.get_device_properties(0).total_memory
        # ~1 GiB per sub-batch item, halve for a safety margin against fragmentation.
        sub_batch_size = max(1, int(free_bytes * 0.5) // 10**9)
    else:
        sub_batch_size = batch_size  # CPU has no such limit
    print(f"Estimated sub-batch size: {sub_batch_size}")

    # Process sub-batches. On CUDA OOM, halve the sub-batch and retry down to 1;
    # frees fragmented cache between tries so the estimator's guess is only a hint.
    results = []
    i = 0
    while i < batch_size:
        step = sub_batch_size
        while True:
            sub_batch = {k: v[0][i : i + step] for k, v in batch.items()}
            try:
                with torch.autocast(device_type=str(ddevice), dtype=torch.float16):
                    key = next(iter(sub_batch.keys()))
                    shape = sub_batch[key].shape
                    with model.ema_scope("Plotting"):
                        model.clamp = (-1, 1)
                        samples = model.sample(
                            sub_batch,
                            batch_size=shape[0],
                            return_intermediates=False,
                            ddim=True,
                            ddim_steps=ddim_steps,
                            shape=shape,
                            # clamp=lambda x: torch.clamp(x, -1, 1),
                        )
                        # DDIM sampler in networks/models/diffusion/utils/ddim_sampler.py
                        # ignores `return_intermediates` and always returns
                        # `(samples, intermediates)`; drop the intermediates.
                        if isinstance(samples, tuple):
                            samples = samples[0]
                        samples = crop_to_original(samples, batch[key][1])
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if step == 1:
                    raise
                step = max(1, step // 2)
                print(f"CUDA OOM at sub-batch {sub_batch_size}; retrying with {step}")
        results.append((samples + 1) * 500)
        i += step
        sub_batch_size = step  # stick with the smaller size for the remaining slices

    # Merge results
    merged = torch.cat(results, dim=0)
    arr: torch.Tensor = merged
    nii = ref.set_array(arr.squeeze(1).cpu().numpy())
    nii.clamp_(min=0)
    return nii


models: dict[str, networks.models.diffusion.ddpm.DDPM] = {}  # type: ignore


def signal_prior_vibe(
    s_magnitude: Sequence[Image_Reference],
    out_signal_prior: str | Path | None = None,
    steps_signal_prior: int = 50,
    override: bool = False,
    gpu: int = 0,
    ddevice: str = "cuda",
    model_folder: str = "logs/palette/2024-11-29T20-13-53-vibe/",
):
    assert len(s_magnitude) == 2
    # Trained it inphase/outphase so we have to flip
    return signal_prior_mevibe(s_magnitude[::-1], out_signal_prior, steps_signal_prior, override, gpu, ddevice, model_folder)


def signal_prior_mevibe(
    s_magnitude: Sequence[Image_Reference],
    out_signal_prior: str | Path | None = None,
    steps_signal_prior: int = 50,
    override: bool = False,
    gpu: int = 0,
    ddevice: str = "cuda",
    model_folder: str = "logs/palette/2024-11-21T10-25-35-mevibe_water/",
):
    """
    Generates a signal prior using the ME-VIBE model based on the provided magnitude images.

    Args:
        s_magnitude (Sequence[Image_Reference]): Sequence of six magnitude images as input.
        out_signal_prior (Union[str, Path, None]): Output path for the generated signal prior. If None,
                                                   the result is returned without saving.
        steps_signal_prior (int): Number of diffusion steps for generating the signal prior. Default is 50.
        override (bool): If True, forces regeneration even if the output file exists. Default is False.
        gpu (int): GPU ID to use for computation. Default is 0.
        ddevice (str): Device to use ("cuda" or "cpu"). Default is "cuda".
        model_folder (str): Path to the folder containing the ME-VIBE model configuration. Default is pre-defined.

    Returns:
        Path: Path to the generated signal prior image.

    Raises:
        AssertionError: If the input sequence `s_magnitude` does not contain exactly six images.
        FileNotFoundError: If the model folder does not exist.
        RuntimeError: If the model fails to load or inference fails.
    """
    # Check for existing output
    if not override and out_signal_prior is not None and Path(out_signal_prior).exists():
        return Path(out_signal_prior)

    # Load model configuration
    model_path = Path(root, model_folder)
    if not model_path.exists():
        if model_folder == "logs/palette/2024-11-21T10-25-35-mevibe_water/":
            url = "https://github.com/robert-graf/MAGO-SP/releases/download/0.0.0/2024-11-21T10-25-35-mevibe_water.zip"
            download(url)
        if model_folder == "logs/palette/2024-11-29T20-13-53-vibe/":
            url = "https://github.com/robert-graf/MAGO-SP/releases/download/0.0.0/2024-11-29T20-13-53-vibe.zip "
            download(url)
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder '{model_path}' does not exist.")

    config = get_config(model_path)
    cond = config["model"]["params"]["conditioning_key_concat"]
    # Validate input sequence length
    assert len(s_magnitude) == len(cond), f"Expected {len(cond)} magnitude images, but got {len(s_magnitude)}."

    # Lazy load model
    global models  # noqa: PLW0602
    if str(model_path) not in models:
        try:
            models[str(model_path)] = instantiate_from_config(config["model"])
        except Exception as e:
            raise FileNotFoundError(f"Failed to instantiate the model: {e}") from e
    model = models[str(model_path)]
    spacing = config["model"].get("spacing", 1.641)
    example_nii = to_nii(s_magnitude[0]).copy()
    use_spacing = abs(example_nii.zoom[example_nii.get_axis("L")] - spacing) > 0.2

    # spacing
    if use_spacing:
        s_magnitude = [to_nii(i).reorient(("S", "L", "A")).rescale_((-1, spacing, spacing)) for i in s_magnitude]
    else:
        s_magnitude = [to_nii(i).reorient(("S", "L", "A")) for i in s_magnitude]

    # Configure device and move model to device
    device = get_device(ddevice, gpu)
    print("predict on", device)
    model.to(device)

    # Normalize input images and prepare for inference
    s_magnitude_torch = _norm_input(s_magnitude, cond)

    # Perform inference to generate signal prior
    signal_prior_water = _inference(model, s_magnitude_torch, to_nii(s_magnitude[0]), ddim_steps=steps_signal_prior, ddevice=device)
    if use_spacing:
        signal_prior_water.resample_from_to_(example_nii)
    else:
        signal_prior_water.reorient_(example_nii.orientation)
    # Save output if path is specified
    if out_signal_prior is not None:
        signal_prior_water.save(out_signal_prior)

    return Path(out_signal_prior) if out_signal_prior is not None else signal_prior_water


def download(url):
    ## Note There is no override protection.
    p = Path(__file__).parents[2] / "logs" / "palette"
    p.mkdir(exist_ok=True, parents=True)
    _download(url, p)


if __name__ == "__main__":
    model_path = Path(root, "logs_/palette/2024-11-21T10-25-35-mevibe_water/")
    if not model_path.exists():
        url = "https://github.com/robert-graf/MAGO-SP/releases/download/0.0.0/2024-11-21T10-25-35-mevibe_water.zip"
        p = Path(__file__).parents[2] / "logs_" / "palette"
        p.mkdir(exist_ok=True, parents=True)
        _download(url, p)
    model_path = Path(root, "logs_/palette/2024-11-29T20-13-53-vibe/")
    if not model_path.exists():
        url = "https://github.com/robert-graf/MAGO-SP/releases/download/0.0.0/2024-11-29T20-13-53-vibe.zip"
        p = Path(__file__).parents[2] / "logs_" / "palette"
        p.mkdir(exist_ok=True, parents=True)
        _download(url, p)
