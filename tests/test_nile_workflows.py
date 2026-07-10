"""CPU-only regression tests for the NILE experiment workflow scripts."""

import ast
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import eval_multiview_consistency, run_nile_grid


class TestNILEGridWorkflow(unittest.TestCase):
    def test_dry_run_deduplicates_baselines_and_sweeps_callback_parameters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            # Dry-run discovery only checks that the input is an existing image path.
            input_path = root / "input.png"
            input_path.write_bytes(b"synthetic dry-run placeholder")
            output_root = root / "outputs"
            manifest_path = root / "manifest.json"

            argv = [
                "--input",
                str(input_path),
                "--methods",
                "iid",
                "shared",
                "flat_sobol",
                "nile_vt",
                "--seeds",
                "7",
                "--rhos",
                "0.0",
                "0.5",
                "--experiment-id",
                "workflow-unit-test",
                "--code-revision",
                "unit-test-revision",
                "--base-model-revision",
                "base-revision",
                "--vae-model-revision",
                "vae-revision",
                "--unet-model",
                "example/unet",
                "--unet-model-revision",
                "unet-revision",
                "--lora-model",
                "example/lora/weights.safetensors",
                "--lora-model-revision",
                "lora-revision",
                "--adapter-revision",
                "adapter-revision",
                "--adapter-sha256",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "--birefnet-model",
                "example/birefnet",
                "--birefnet-revision",
                "birefnet-revision",
                "--rho-start",
                "0.2",
                "0.4",
                "--active-ratio",
                "0.3",
                "0.6",
                "--dry-run",
                "--output-root",
                str(output_root),
                "--manifest",
                str(manifest_path),
            ]
            with redirect_stdout(io.StringIO()):
                return_code = run_nile_grid.main(argv)

            self.assertEqual(return_code, 0)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["experiment_id"], "workflow-unit-test")
            self.assertEqual(payload["code_revision"], "unit-test-revision")
            records = payload["runs"]

            by_method = {}
            for record in records:
                by_method.setdefault(record["method"], []).append(record)

            # Built-in rho-independent baselines are each planned once when the
            # default baseline de-duplication is enabled, despite two rho values.
            for method in ("iid", "shared", "flat_sobol"):
                self.assertEqual(len(by_method[method]), 1)
                self.assertEqual(by_method[method][0]["rho_geo"], 0.0)
                self.assertEqual(by_method[method][0]["nile_callback"], "none")

            callback_records = by_method["nile_vt"]
            self.assertEqual(len(callback_records), 8)
            self.assertEqual(
                {
                    (
                        record["rho_geo"],
                        record["rho_start"],
                        record["active_ratio"],
                    )
                    for record in callback_records
                },
                {
                    (rho, rho_start, active_ratio)
                    for rho in (0.0, 0.5)
                    for rho_start in (0.2, 0.4)
                    for active_ratio in (0.3, 0.6)
                },
            )
            self.assertTrue(
                all(record["nile_mode"] == "nile_v" for record in callback_records)
            )
            self.assertTrue(
                all(
                    record["nile_callback"] == "nile_vt"
                    for record in callback_records
                )
            )

            required_fields = {
                "schema_version",
                "run_id",
                "experiment_id",
                "code_revision",
                "base_model",
                "base_model_revision",
                "vae_model",
                "vae_model_revision",
                "unet_model",
                "unet_model_revision",
                "lora_model",
                "lora_model_revision",
                "adapter_path",
                "adapter_revision",
                "adapter_sha256",
                "birefnet_model",
                "birefnet_revision",
                "status",
                "input",
                "method",
                "nile_mode",
                "nile_callback",
                "seed",
                "rho_geo",
                "rho_start",
                "rho_end",
                "active_ratio",
                "command",
                "output",
                "metadata_path",
                "views_dir",
                "reference_output",
            }
            self.assertEqual(len(records), 11)
            for record in records:
                self.assertTrue(required_fields.issubset(record))
                self.assertEqual(record["status"], "dry_run")
                self.assertEqual(record["seed"], 7)
                self.assertEqual(record["experiment_id"], "workflow-unit-test")
                self.assertEqual(record["code_revision"], "unit-test-revision")
                self.assertEqual(record["base_model_revision"], "base-revision")
                self.assertEqual(record["vae_model_revision"], "vae-revision")
                self.assertEqual(record["unet_model_revision"], "unet-revision")
                self.assertEqual(record["lora_model_revision"], "lora-revision")
                self.assertEqual(record["adapter_revision"], "adapter-revision")
                self.assertEqual(record["adapter_sha256"], "a" * 64)
                self.assertEqual(record["birefnet_model"], "example/birefnet")
                self.assertEqual(record["birefnet_revision"], "birefnet-revision")
                self.assertIsInstance(record["command"], list)
                self.assertIn("--output", record["command"])
                for flag, value in (
                    ("--base_model_revision", "base-revision"),
                    ("--vae_model_revision", "vae-revision"),
                    ("--unet_model_revision", "unet-revision"),
                    ("--lora_model_revision", "lora-revision"),
                    ("--adapter_revision", "adapter-revision"),
                    ("--birefnet_model", "example/birefnet"),
                    ("--birefnet_revision", "birefnet-revision"),
                ):
                    index = record["command"].index(flag)
                    self.assertEqual(record["command"][index + 1], value)

    def test_run_id_changes_across_experiment_and_code_revision(self):
        base = {
            "experiment_id": "full-a",
            "code_revision": "commit-a",
            "method": "iid_default",
            "seed": 0,
        }
        run_id = run_nile_grid._run_id(base)
        self.assertNotEqual(
            run_id,
            run_nile_grid._run_id({**base, "experiment_id": "full-b"}),
        )
        self.assertNotEqual(
            run_id,
            run_nile_grid._run_id({**base, "code_revision": "commit-b"}),
        )


class TestDistributionPreservingNotebookWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[1]
            / "notebooks"
            / "mvadapter_distribution_preserving_colab.ipynb"
        )
        cls.notebook = json.loads(path.read_text(encoding="utf-8"))
        sources = []
        for index, cell in enumerate(cls.notebook["cells"]):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            sources.append(source)
            if cell.get("cell_type") == "code":
                ast.parse(source, filename="notebook-cell-{}".format(index))
        cls.source = "\n".join(sources)

    def test_notebook_is_clean_and_unexecuted(self):
        for cell in self.notebook["cells"]:
            if cell.get("cell_type") == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_repository_check_rejects_untracked_experiment_code(self):
        self.assertIn('"--untracked-files=all"', self.source)
        self.assertNotIn('"--untracked-files=no"', self.source)
        self.assertIn('"ls-files"', self.source)
        self.assertIn('"--error-unmatch"', self.source)

    def test_hugging_face_revisions_are_resolved_and_forwarded(self):
        self.assertIn("from huggingface_hub import hf_hub_download, model_info", self.source)
        self.assertIn("resolved = info.sha", self.source)
        self.assertIn("PRELOAD_MODEL_SNAPSHOTS = True", self.source)
        self.assertIn(
            "model_snapshots.append((BIREFNET_MODEL, BIREFNET_REVISION))",
            self.source,
        )
        for name in (
            "BASE_MODEL_REVISION",
            "VAE_MODEL_REVISION",
            "ADAPTER_REVISION",
            "BIREFNET_REVISION",
        ):
            self.assertIn("{} =".format(name), self.source)
        for flag in (
            "--base-model-revision",
            "--vae-model-revision",
            "--adapter-revision",
            "--birefnet-revision",
        ):
            self.assertIn(flag, self.source)


class TestMultiviewEvaluationWorkflow(unittest.TestCase):
    @staticmethod
    def _write_views(directory):
        directory.mkdir()
        height, width = 12, 16
        y, x = np.mgrid[:height, :width]
        relative_paths = []
        for index in range(6):
            image = np.stack(
                (
                    (x * 13 + index * 17) % 256,
                    (y * 19 + index * 29) % 256,
                    ((x + y) * 11 + index * 7) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
            path = directory / "view_{:02d}.png".format(index)
            Image.fromarray(image, mode="RGB").save(path)
            relative_paths.append(str(Path(directory.name) / path.name))
        return relative_paths

    def test_metadata_input_reports_expected_pairs_and_normalized_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            view_paths = self._write_views(root / "views")
            metadata_path = root / "sample_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "run_id": "synthetic-six-view",
                        "view_files": view_paths,
                        "azimuth_deg": [0, 45, 90, 180, 270, 315],
                        "seed": 9,
                        "nile": {
                            "mode": "nile_v",
                            "callback": "nile_vt",
                            "rho_geo": 0.65,
                            "rho_start": 0.4,
                            "rho_end": 0.0,
                            "active_ratio": 0.6,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "metrics.json"

            with redirect_stdout(io.StringIO()):
                return_code = eval_multiview_consistency.main(
                    [
                        "--input",
                        str(metadata_path),
                        "--metrics",
                        "lightweight",
                        "--image-size",
                        "16",
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(return_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("not geometry-aware substitutes for MEt3R", payload["metric_notice"])
            self.assertIn("collapse detectors", payload["metric_notice"])
            self.assertEqual(payload["settings"]["metrics"], "lightweight")
            self.assertEqual(
                payload["settings"]["lightweight_role"], "collapse_detector_only"
            )
            self.assertEqual(
                payload["settings"]["angle_bins_deg"], [45.0, 90.0, 135.0, 180.0]
            )
            self.assertIsNone(payload["settings"]["met3r"])

            self.assertEqual(len(payload["samples"]), 1)
            sample = payload["samples"][0]
            self.assertEqual(sample["status"], "succeeded")
            self.assertEqual(sample["num_views"], 6)
            self.assertEqual(sample["adjacent_pair_count"], 6)
            self.assertEqual(sample["opposite_pair_count"], 2)
            self.assertEqual(sample["method"], "nile_vt")
            self.assertEqual(sample["nile_mode"], "nile_v")
            self.assertEqual(sample["nile_callback"], "nile_vt")
            self.assertEqual(sample["rho_geo"], 0.65)
            self.assertEqual(sample["rho_start"], 0.4)
            self.assertEqual(sample["active_ratio"], 0.6)
            self.assertEqual(sample["angle_all_pair_count"], 15)
            self.assertEqual(sample["angle_45_pair_count"], 4)
            self.assertEqual(sample["angle_90_pair_count"], 5)
            self.assertEqual(sample["angle_135_pair_count"], 4)
            self.assertEqual(sample["angle_180_pair_count"], 2)
            self.assertIn(
                sample["collapse_detector_label"],
                {"no_collapse_signal", "view_collapse_alert"},
            )

            pair_groups = [row["pair_group"] for row in payload["pairs"]]
            self.assertEqual(pair_groups.count("adjacent"), 6)
            self.assertEqual(pair_groups.count("opposite"), 2)
            self.assertTrue(
                all("lowfreq_l1_similarity" in row for row in payload["pairs"])
            )
            self.assertTrue(all("met3r_score" not in row for row in payload["pairs"]))
            self.assertEqual(len(payload["angle_pairs"]), 15)
            self.assertEqual(
                {
                    angle: sum(
                        row["angle_bin_deg"] == angle
                        for row in payload["angle_pairs"]
                    )
                    for angle in (45.0, 90.0, 135.0, 180.0)
                },
                {45.0: 4, 90.0: 5, 135.0: 4, 180.0: 2},
            )
            self.assertEqual(len(payload["angle_bin_summaries"]), 4)
            self.assertTrue(
                all(
                    row["r_hf_status"] == "missing_iid_reference"
                    for row in payload["angle_bin_summaries"]
                )
            )

    def test_met3r_numpy_preprocessing_shape_range_and_layout(self):
        black = np.zeros((5, 7, 3), dtype=np.float32)
        white = np.ones((5, 7, 3), dtype=np.float32)
        batch = eval_multiview_consistency._prepare_met3r_numpy_batch(
            [(black, white), (white, black)], image_size=8
        )

        self.assertEqual(batch.shape, (2, 2, 3, 8, 8))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(batch.flags.c_contiguous)
        self.assertEqual(float(batch.min()), -1.0)
        self.assertEqual(float(batch.max()), 1.0)
        self.assertTrue(np.all(batch[0, 0] == -1.0))
        self.assertTrue(np.all(batch[0, 1] == 1.0))

    def test_r_hf_uses_iid_default_per_angle_and_emits_guardrail_labels(self):
        rows = [
            {
                "method": "iid_default",
                "inference_method": "iid_default",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.2,
            },
            {
                "method": "camera_rbf_corr",
                "inference_method": "camera_rbf_corr",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.16,
            },
            {
                "method": "nested_tree_ab",
                "inference_method": "nested_tree_ab",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.08,
            },
            {
                "method": "shared_full",
                "inference_method": "shared_full",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.02,
            },
        ]
        eval_multiview_consistency._annotate_relative_high_frequency(
            rows,
            iid_method="iid_default",
            metric_name="highfreq_l1_distance",
            match_fields=("angle_bin_deg",),
        )

        self.assertAlmostEqual(rows[0]["r_hf"], 1.0)
        self.assertEqual(rows[0]["r_hf_status"], "healthy")
        self.assertAlmostEqual(rows[1]["r_hf"], 0.8)
        self.assertEqual(rows[1]["r_hf_status"], "healthy")
        self.assertAlmostEqual(rows[2]["r_hf"], 0.4)
        self.assertEqual(rows[2]["r_hf_status"], "overcoupling_alert")
        self.assertAlmostEqual(rows[3]["r_hf"], 0.1)
        self.assertEqual(rows[3]["r_hf_status"], "likely_view_collapse")

    def test_r_hf_pairs_iid_reference_per_input_before_aggregation(self):
        rows = [
            {
                "experiment_id": "exp-a",
                "code_revision": "commit-a",
                "input_image": "small-iid.png",
                "seed": 0,
                "method": "iid_default",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.2,
            },
            {
                "experiment_id": "exp-a",
                "code_revision": "commit-a",
                "input_image": "small-iid.png",
                "seed": 0,
                "method": "camera_rbf_corr",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.16,
            },
            {
                "experiment_id": "exp-a",
                "code_revision": "commit-a",
                "input_image": "large-iid.png",
                "seed": 0,
                "method": "iid_default",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.8,
            },
            {
                "experiment_id": "exp-a",
                "code_revision": "commit-a",
                "input_image": "large-iid.png",
                "seed": 0,
                "method": "camera_rbf_corr",
                "angle_bin_deg": 45.0,
                "highfreq_l1_distance": 0.64,
            },
        ]
        match_fields = (
            "experiment_id",
            "code_revision",
            "input_image",
            "seed",
            "angle_bin_deg",
        )
        eval_multiview_consistency._annotate_relative_high_frequency(
            rows,
            iid_method="iid_default",
            metric_name="highfreq_l1_distance",
            match_fields=match_fields,
        )

        self.assertAlmostEqual(rows[1]["r_hf"], 0.8)
        self.assertAlmostEqual(rows[3]["r_hf"], 0.8)
        self.assertAlmostEqual(
            rows[1]["r_hf_reference_highfreq_l1_distance"], 0.2
        )
        self.assertAlmostEqual(
            rows[3]["r_hf_reference_highfreq_l1_distance"], 0.8
        )

    def test_camera_response_monotonic_requires_strict_complete_order(self):
        bins = [45.0, 90.0, 135.0, 180.0]
        passing = {
            "angle_45_lowfreq_l1_similarity": 0.9,
            "angle_90_lowfreq_l1_similarity": 0.8,
            "angle_135_lowfreq_l1_similarity": 0.7,
            "angle_180_lowfreq_l1_similarity": 0.6,
        }
        self.assertEqual(
            eval_multiview_consistency._camera_response_monotonic(passing, bins),
            "passed",
        )

        failing = dict(passing)
        failing["angle_135_lowfreq_l1_similarity"] = 0.8
        self.assertEqual(
            eval_multiview_consistency._camera_response_monotonic(failing, bins),
            "failed",
        )

        incomplete = dict(passing)
        incomplete.pop("angle_180_lowfreq_l1_similarity")
        self.assertEqual(
            eval_multiview_consistency._camera_response_monotonic(incomplete, bins),
            "not_available",
        )


if __name__ == "__main__":
    unittest.main()
