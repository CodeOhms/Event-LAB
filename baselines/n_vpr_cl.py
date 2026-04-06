import os
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from baselines.Neuormorphic_VPR_Contrastive_Learning.n_vpr_cl.math import normalise_np_vector
from baselines.Neuormorphic_VPR_Contrastive_Learning.n_vpr_cl.vpr_evaluation import distance_to_similarity_matrix
import utils.functional as FUNC
from baselines.download_baseline import clone_repo
from baselines.EventBaselineLab import EventBaseline
from baselines.Neuormorphic_VPR_Contrastive_Learning.n_vpr_cl.feature_extraction import (
    compute_NN_similarity_matrix,
    compute_descriptors,
)
from baselines.Neuormorphic_VPR_Contrastive_Learning.n_vpr_cl.generalised_datasets import (
    CustomDataset,
)
from baselines.Neuormorphic_VPR_Contrastive_Learning.n_vpr_cl.utils import (
    HWC_to_CHW,
    get_torch_device,
    init_encoder_and_model,
    load_np_sample,
)


class n_vpr_cl_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "n_vpr_cl"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/Neuormorphic_VPR_Contrastive_Learning"
        # Baseline URL
        self.url = (
            "https://github.com/CodeOhms/Neuormorphic_VPR_Contrastive_Learning.git"
        )
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.baseline_config_path = "./baselines/n_vpr_cl.yaml"
        self.matrix_type = 'similarity' # options are 'similarity' or 'distance'
        # Load the baseline configuration
        with open(self.baseline_config_path, "r") as file:
            self.baseline_config = yaml.safe_load(file)
        # Create the data output folder
        self.outdir = "./output/n_vpr_cl"
        os.makedirs(self.outdir, exist_ok=True)

    # Copied from eventvlad.py bead3ab019a4af2bf7131f2a6f54c78ff54bd8b5
    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the baseline.
        """
        self.config = config
        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info["sequence_name"]
        query_name = query_info["sequence_name"]

        # from ref_info['file_path'] dict, find the directory that matches ref/query name and timewindow
        self.ref_key = [
            d for d in ref_info["file_path"] if ref_name in d and str(timewindow) in d
        ]
        self.query_key = [
            d
            for d in query_info["file_path"]
            if query_name in d and str(timewindow) in d
        ]
        self.ref_directory = ref_info["file_path"][self.ref_key[0]]
        self.query_directory = query_info["file_path"][self.query_key[0]]
        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]

        import re
        from pathlib import Path

        _RX_FRAME = re.compile(r"^frame_(\d+)\.npy$")

        def list_frame_files(dirpath: str):
            paths = []
            for p in Path(dirpath).iterdir():
                m = _RX_FRAME.fullmatch(p.name)
                if m:
                    paths.append((int(m.group(1)), p))
            paths.sort(key=lambda t: t[0])  # numeric sort
            return [p for _, p in paths]

        # usage
        self.ref_files = list_frame_files(self.ref_directory)
        self.query_files = list_frame_files(self.query_directory)
        # after you have ref_files, query_files and min_gap_sec
        min_gap_sec = float(config.get("filter_places_sec", 60))

        ref_res = FUNC._apply_time_filter_to_files(
            self.ref_files, self.ref_directory, min_gap_sec, debug=False
        )
        query_res = FUNC._apply_time_filter_to_files(
            self.query_files, self.query_directory, min_gap_sec, debug=False
        )

        # Replace file lists with filtered ones
        self.ref_files = ref_res["files"]
        self.query_files = query_res["files"]

        self.output_dir = os.path.join(
            self.outdir,
            f"{ref_info['dataset_name']}",
            f"{ref_info['sequence_name']}_{query_info['sequence_name']}",
            f"{config['frame_generator']}_{timewindow}",
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def build_execute(self, config, data_config, ground_truth):
        """
        Build a commandline execute for the baseline with the provided reference, query, and ground truth data.
        """
        pass

    def run(self):
        """
        Run the baseline.
        """
        """
        Implement run logic here to retrieve distance matrix and save it for analysis.
        """
        batch_size = 8
        num_workers = 4
        Args = namedtuple("Args", ["num_input_channels", "model_path"])
        model_args = Args(
            2,
            "./baselines/Neuormorphic_VPR_Contrastive_Learning/models/encoder_event-22-03-26.pth.tar",
        )
        encoder, model = init_encoder_and_model(model_args)
        dev = get_torch_device()
        model.to(dev).eval()

        ref_ds = CustomDataset(
            samples_dir=None,
            samples_paths=self.ref_files,
            sample_loader=load_np_sample,
            transform=HWC_to_CHW,
        )
        qry_ds = CustomDataset(
            samples_dir=None,
            samples_paths=self.query_files,
            sample_loader=load_np_sample,
            transform=HWC_to_CHW,
        )
        ref_loader = DataLoader(
            ref_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
            persistent_workers=(num_workers > 0),
            drop_last=False,  # keep default; we’ll fix the 1-sample case below (compute_descriptors)
        )
        qry_loader = DataLoader(
            qry_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev.type == "cuda"),
            persistent_workers=(num_workers > 0),
            drop_last=False,  # keep default; we’ll fix the 1-sample case below (compute_descriptors)
        )

        similarity_matrix = compute_NN_similarity_matrix(dev, model, ref_loader=ref_loader, qry_loader=qry_loader)

        # Save the similarity matrix
        np.save(f"{self.output_dir}/similarity_matrix.npy", similarity_matrix)

        # ref_feats = compute_descriptors(dev, model, ref_loader)
        # qry_feats = compute_descriptors(dev, model, qry_loader)
        # ref_feats = ref_feats / np.linalg.norm(ref_feats, axis=1, keepdims=True)
        # qry_feats = qry_feats / np.linalg.norm(qry_feats, axis=1, keepdims=True)
        # distance_matrix = (1 - (qry_feats @ ref_feats.T)).T

        # # Save the distance matrices
        # distance_matrix_mine = distance_to_similarity_matrix(distance_matrix)
        # np.save(f"{self.output_dir}/distance_matrix_mine.npy", distance_matrix_mine)

        # distance_matrix_other = distance_matrix.max() - distance_matrix
        # np.save(f"{self.output_dir}/distance_matrix_other.npy", distance_matrix_other)

    # Copied from eventvlad.py bead3ab019a4af2bf7131f2a6f54c78ff54bd8b5
    def parse_results(self, GT):
        """
        Summary sheet: upsert by (run_name, ref_query, array_name)
        Per-run sheet (self.name): upsert summary by (ref_query, array_name),
        and upsert each PR block keyed by "PR curve for {ref_query} :: {array_name}".
        """
        # gather files
        all_files = sorted(list(Path(self.output_dir).glob("*.npy")))
        all_names = [os.path.basename(f).replace(".npy", "") for f in all_files]
        all_arrays = [np.load(f) for f in all_files]
        GThard = np.load(GT)
        if not all_arrays:
            print("No .npy result files found in", self.output_dir)
            return

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        # Run evaluation metrics
        rows, pr_curves = self.run_metrics(
            all_names,
            all_arrays,
            GThard,
            timestamp,
            self.name,
            f"{self.ref_name}_{self.query_name}",
            matrix_type=self.matrix_type,
            outdir=self.output_dir,
            tolerance=self.config.get("ground_truth_tolerance", 0.0),
        )

        # Save results to excel spreadsheet
        self.save_results(
            rows, pr_curves, self.name, f"{self.ref_name}_{self.query_name}"
        )

    def cleanup(self):
        """
        Clean up temporary files.
        """
        import shutil

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
