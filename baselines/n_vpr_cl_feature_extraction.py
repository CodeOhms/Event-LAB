import os, glob
from os.path import isfile
from collections import OrderedDict
import torch
import numpy as np

from N_VPR_CL.model import SynSSparkEncoder


def _device(dev=None):
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _coerce_items(images):
    """
    Accept: list/tuple/ndarray of paths, a single path, a directory, a glob, or a .txt/.lst file of paths.
    Returns a sorted list of file paths.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm")
    # list/tuple already
    if isinstance(images, (list, tuple, np.ndarray)):
        return list(images)

    # single string cases
    if isinstance(images, str):
        # text list of paths
        if images.lower().endswith((".txt", ".lst")) and isfile(images):
            with open(images, "r") as f:
                paths = [ln.strip() for ln in f if ln.strip()]
            return paths
        # directory
        if os.path.isdir(images):
            files = [os.path.join(images, f) for f in os.listdir(images)]
            files = [p for p in files if isfile(p) and p.lower().endswith(exts)]
            files.sort()
            return files
        # glob
        g = sorted(glob.glob(images))
        if len(g):
            return [p for p in g if isfile(p)]
        # single file path
        return [images]

    # fallback
    return [images]


def _remap_state_keys(mdl_state_d):
    remapped_state = OrderedDict()
    for k, v in mdl_state_d.items():
        prefix = "module.encoder."
        if k.startswith(prefix):
            k = k[len(prefix) :]
        remapped_state[k] = v
    return remapped_state


def _extract_encoder_state(mdl_state_d):
    encoder_state = OrderedDict()
    for k, v in mdl_state_d.items():
        if not "fc" in k:
            encoder_state[k] = v
    return encoder_state


def load_simclr_encoder(weights_path: str, device=None):
    assert isfile(weights_path), f"Missing weights file: {weights_path}"
    dev = _device(device)

    model = SynSSparkEncoder()
    simclr_save = torch.load(weights_path)
    mdl_state_dict = _remap_state_keys(simclr_save["state_dict"])
    mdl_state_dict = _extract_encoder_state(mdl_state_dict)
    model.load_state_dict(mdl_state_dict)
    model.to(dev).eval()
    return model


@torch.inference_mode()
def extract_features(
    model,
    images_dir,
    batch_size=16,
    num_workers=4,
    device=None,
    rgb_input=True,
):
    dev = _device(device)
    model = model.to(dev).eval()

    image_list = _coerce_items(images_dir)
    if len(image_list) == 0:
        raise ValueError("No images found to process.")
    ds = _EventVGGPreprocess(image_list, rgb_input=rgb_input)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(num_workers > 0),
        # drop_last=False  # keep default; we’ll fix the 1-sample case below
    )

    feats = []
    for batch in tqdm(dl, desc="EventVLAD", leave=False):
        batch = batch.to(dev, non_blocking=True)
        out = model.feature_extract(batch)  # may be [B, D] or [D] when B==1
        if out.dim() == 1:  # <— ensure 2-D
            out = out.unsqueeze(0)  # -> [1, D]
        out_cpu = out.detach().cpu().to(torch.float32).numpy()
        feats.append(out_cpu)

    return np.concatenate(feats, axis=0)  # now all chunks are [b_i, D]
