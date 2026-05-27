import os, tempfile, subprocess, yaml
import numpy as np
from pathlib import Path
from baselines.EventBaselineLab import EventBaseline
from baselines.download_baseline import clone_repo
from baselines.VPR_Tutorial.evaluation.metrics import recallAtK, createPR
import prettytable
import openpyxl
from datetime import datetime, timezone
import re, gdown, time
import utils.functional as FUNC
import shutil
from datasets.dataloader import make_frame_source
from loguru import logger



class n_vpr_cl_baseline(EventBaseline):
    def __init__(self):
        super().__init__()

        self.name = "n_vpr_cl"
        # Check if the baseline repository is already cloned
        self.repo_path = "./baselines/N_VPR_CL"
        # Set the type of matrix generated for evaluation (distance or similarity)
        self.matrix_type = 'distance'
        # Baseline URL
        self.url = "https://github.com/<name>"
        if not os.path.exists(self.repo_path):
            clone_repo(self.url, destination=self.repo_path)
        self.baseline_config_path = "./baselines/n_vpr_cl.yaml"
        # Load the baseline configuration
        with open(self.baseline_config_path, "r") as file:
            self.baseline_config = yaml.safe_load(file)
        # Create the data output folder
        self.outdir = "./output/n_vpr_cl"
        os.makedirs(self.outdir, exist_ok=True)

    def format_data(self, config, dataset_config, reference, query, timewindow):
        """
        Format the reference and query data for the EventVLAD baseline.

        Frame loading is now storage-agnostic through:

            from datasets.dataloader import make_frame_source

        This supports the new frames.h5 storage system without manually listing
        frame_*.npy files.
        """

        self.config = config
        self.dataset_config = dataset_config
        self.reference = reference
        self.query = query

        # Get experimental details
        ref_info = reference.get_dataset_info()
        query_info = query.get_dataset_info()

        ref_name = ref_info["sequence_name"]
        query_name = query_info["sequence_name"]

        # Find the frame-store directories matching sequence name + timewindow
        self.ref_key = [
            d for d in ref_info["file_path"]
            if ref_name in d and str(timewindow) in d
        ]
        self.query_key = [
            d for d in query_info["file_path"]
            if query_name in d and str(timewindow) in d
        ]

        if not self.ref_key:
            raise FileNotFoundError(
                f"No reference frame store found for sequence={ref_name}, timewindow={timewindow}"
            )

        if not self.query_key:
            raise FileNotFoundError(
                f"No query frame store found for sequence={query_name}, timewindow={timewindow}"
            )

        self.ref_name = self.ref_key[0]
        self.query_name = self.query_key[0]

        self.ref_directory = ref_info["file_path"][self.ref_name]
        self.query_directory = query_info["file_path"][self.query_name]

        # Build frame sources using the new generalized loader.
        # This replaces all manual frame_*.npy listing/loading logic.
        min_gap_sec = float(config.get("filter_places_sec", 60))

        self.ref_frame_source = make_frame_source(
            self.ref_directory,
            min_gap_sec=min_gap_sec,
            collapse_polarity=False,
        )

        self.query_frame_source = make_frame_source(
            self.query_directory,
            min_gap_sec=min_gap_sec,
            collapse_polarity=False,
        )

        # Preserve kept/dropped indices if the loader exposes them.
        self.ref_kept_idx = getattr(self.ref_frame_source, "kept_idx", None)
        self.ref_dropped_idx = getattr(self.ref_frame_source, "dropped_idx", None)

        self.query_kept_idx = getattr(self.query_frame_source, "kept_idx", None)
        self.query_dropped_idx = getattr(self.query_frame_source, "dropped_idx", None)

        # Main baseline output directory
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
        from baselines.n_vpr_cl_feature_extraction import (
            load_simclr_encoder,
            extract_features_h5,
        )

        model = load_simclr_encoder(
            weights_path="./baselines/N_VPR_CL/models/checkpoint_0019.pth.tar",
            in_channels=2,
        )

        ref_feats = extract_features_h5(
            model, self.ref_frame_source.iter_batches(batch_size=8)
        )
        query_feats = extract_features_h5(
            model, self.query_frame_source.iter_batches(batch_size=8)
        )
        # ref_feats = extract_features(
        #     model, self.ref_directory, batch_size=8, num_workers=4
        # )
        # query_feats = extract_features(
        #     model, self.query_directory, batch_size=8, num_workers=4
        # )

        distance_matrix = (1 - (query_feats @ ref_feats.T)).T

        # Save the distance matrices
        np.save(f"{self.output_dir}/distance_matrix.npy", distance_matrix)

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
            logger.warning("No .npy result files found in", self.output_dir)
            return

        timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

        # Run evaluation metrics
        rows, pr_curves = self.run_metrics(
                all_names,
                all_arrays,
                GThard,
                timestamp,
                self.name,
                f'{self.ref_name}_{self.query_name}',
                matrix_type=self.matrix_type,
                outdir=self.output_dir,
                tolerance=self.config.get('ground_truth_tolerance', 0.0)
        )

        # Save results to excel spreadsheet
        self.save_results(rows, pr_curves, self.name, f'{self.ref_name}_{self.query_name}')

    def cleanup(self):
        """
        Clean up temporary files.
        """
        import shutil

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
