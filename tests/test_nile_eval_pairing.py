"""Regression tests for paired IID high-frequency guardrails."""

import unittest

from scripts import eval_multiview_consistency as evaluation


class TestPairedHighFrequencyGuardrail(unittest.TestCase):
    def test_input_image_is_normalized_from_inference_metadata(self):
        normalized = evaluation._normalize_metadata(
            {
                "input": {"image": "/drive/inputs/chair.png", "text": "chair"},
                "method": "iid_default",
            }
        )
        self.assertEqual(normalized["input_image"], "/drive/inputs/chair.png")

    def test_ratios_are_paired_before_cross_input_aggregation(self):
        rows = []
        cases = (
            ("cup.png", "iid_default", 0.10),
            ("cup.png", "camera_rbf_corr", 0.08),  # ratio 0.80
            ("chair.png", "iid_default", 0.40),
            ("chair.png", "camera_rbf_corr", 0.20),  # ratio 0.50
        )
        for input_image, method, distance in cases:
            rows.append(
                {
                    "sample_id": "{}:{}".format(input_image, method),
                    "experiment_id": "formal-v1",
                    "code_revision": "abc123",
                    "input_image": input_image,
                    "seed": 0,
                    "method": method,
                    "inference_method": method,
                    "max_correlation": 0.0 if method == "iid_default" else 0.3,
                    "frequency_scale": 0.12,
                    "camera_length_scale": 0.8,
                    "nile_mode": method,
                    "nile_callback": "none",
                    "rho_geo": 0.0 if method == "iid_default" else 0.3,
                    "angle_bin_deg": 45.0,
                    "highfreq_l1_distance": distance,
                }
            )

        per_sample = evaluation._build_sample_angle_bin_summaries(rows)
        evaluation._annotate_relative_high_frequency(
            per_sample,
            iid_method="iid_default",
            metric_name="highfreq_l1_distance",
            match_fields=(
                "experiment_id",
                "code_revision",
                "input_image",
                "seed",
                "angle_bin_deg",
            ),
        )
        summaries = evaluation._build_angle_bin_summaries(per_sample)
        camera = next(
            row
            for row in summaries
            if row["inference_method"] == "camera_rbf_corr"
        )

        # Mean of paired ratios: (0.80 + 0.50) / 2.  A ratio of global means
        # would be (0.08 + 0.20) / (0.10 + 0.40) == 0.56 instead.
        self.assertAlmostEqual(camera["r_hf"], 0.65)
        self.assertEqual(camera["r_hf_status"], "visual_check")
        self.assertEqual(camera["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
