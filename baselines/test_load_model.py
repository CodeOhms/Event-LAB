from collections import OrderedDict
import torch

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


if __name__ == "__main__":
    dev = _device()

    model = SynSSparkEncoder()
    simclr_save = torch.load("test_model.pth.tar")
    mdl_state_dict = _remap_state_keys(simclr_save["state_dict"])
    mdl_state_dict = _extract_encoder_state(mdl_state_dict)
    model.load_state_dict(mdl_state_dict)
    model.to(dev).eval()
