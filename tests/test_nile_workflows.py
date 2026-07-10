"""CPU-only regression tests for the NILE experiment workflow scripts."""

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
                self.assertIsInstance(record["command"], list)
                self.assertIn("--output", record["command"])


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
            self.assertEqual(payload["settings"]["metrics"], "lightweight")
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

            pair_groups = [row["pair_group"] for row in payload["pairs"]]
            self.assertEqual(pair_groups.count("adjacent"), 6)
            self.assertEqual(pair_groups.count("opposite"), 2)
            self.assertTrue(
                all("lowfreq_l1_similarity" in row for row in payload["pairs"])
            )
            self.assertTrue(all("met3r_score" not in row for row in payload["pairs"]))

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


if __name__ == "__main__":
    unittest.main()
