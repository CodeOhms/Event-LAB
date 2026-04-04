import os
from baselines import n_vpr_cl
from baselines.download_baseline import clone_repo

# Ensure the standardized metrics repo (VPR-Tutorial) is cloned
if not os.path.exists("./baselines/VPR_Tutorial"):
    clone_repo(
        "https://github.com/stschubert/VPR_Tutorial.git",
        destination="./baselines/VPR_Tutorial",
    )

from baselines.lens import LENS_baseline
from baselines.sparse_event import sparse_event_baseline
from baselines.ensemble import ensemble_baseline
from baselines.eventvlad import eventvlad_baseline
from baselines.vprmethods import vprmethods_baseline
from baselines.n_vpr_cl import n_vpr_cl_baseline


def get_baseline_switcher():
    return {
        "lens": lambda: LENS_baseline(),
        "sparse_event": lambda: sparse_event_baseline(),
        "ensemble": lambda: ensemble_baseline(),
        "eventvlad": lambda: eventvlad_baseline(),
        "vprmethods": lambda: vprmethods_baseline(),
        "n_vpr_cl": lambda: n_vpr_cl_baseline(),
    }


def get_baseline(baseline_name):
    baseline_name = baseline_name.lower()
    switcher = get_baseline_switcher()
    return switcher.get(baseline_name, lambda: "Invalid")()


def list_available_baselines():
    return list(get_baseline_switcher().keys())
