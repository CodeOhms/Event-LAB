import os, glob
from os.path import isfile
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import cv2

from .N_VPR_CL.model import SynSSparkEncoder


class _EventVGGPreprocess(Dataset):
    """
    - Resizes to 224x224
    - Converts to RGB by default (see note), uint8-like scale (no /255)
    - Subtracts channel means from EventVLAD Imagenet VGG meta (std=1)
    """

    def __init__(self, items, rgb_input=True):
        self.items = items
        self.rgb_input = rgb_input
        # From Imagenet_matconvnet_vgg_verydeep_16_dag.meta (RGB order)
        self.mean = np.array([122.7449417, 114.9440994, 101.6417770], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def _load(self, it):
        if isinstance(it, (np.ndarray, np.generic)):
            img = it
        else:
            img = cv2.imread(it, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {it}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if self.rgb_input:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if self.rgb_input:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        # matconv-style: keep [0..255]-scale floats, subtract means
        img = img.astype(np.float32)
        img -= self.mean  # RGB means
        img = np.transpose(img, (2, 0, 1))  # CHW
        return torch.from_numpy(img)

    def __getitem__(self, idx):
        return self._load(self.items[idx])


# class _Dataset(Dataset):
#     def __init__(self, np_imgs, transform=None):
#         self.np_img_arr = np_imgs

#     def __len__(self):
#         return len(self.np_img_arr)

#     def __getitem__(self, idx):
#         return torch.from_numpy(self.np_img_arr[:, :, idx, :])


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
    model.load_state_dict(mdl_state_dict, strict=True)
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
        out = model(batch)  # may be [B, D] or [D] when B==1
        if out.dim() == 1:  # <— ensure 2-D
            out = out.unsqueeze(0)  # -> [1, D]
        out_cpu = out.detach().cpu().to(torch.float32).numpy()
        feats.append(out_cpu)

    return np.concatenate(feats, axis=0)  # now all chunks are [b_i, D]
