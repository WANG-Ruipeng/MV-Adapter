import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from mvadapter.nile.basis import build_dct2_basis
from mvadapter.nile.covariance import mix_covariance_with_identity
from mvadapter.nile.diagnostics import (
    diagnose_lowrank_latents,
    empirical_basis_coefficient_covariance,
    evaluate_lowrank_distribution_gates,
)
from mvadapter.nile.lowrank_coupling import correlate_orthonormal_subspace
from scripts.diagnose_nile_lowrank import (
    build_pilot_configurations,
    main as diagnose_main,
    run_configuration_preflight,
    summarize_configuration_records,
    write_preflight_artifacts,
)
from scripts.run_nile_lowrank_study import (
    build_pilot_configurations as runner_pilot_configurations,
)


def _study_config():
    return {
        "model": {"views_deg": [0, 45, 90, 180, 270, 315]},
        "preflight": {
            "batch_size": 4,
            "channels": 1,
            "latent_height": 4,
            "latent_width": 4,
            "seeds": [0, 1],
            "mean_abs_max": 0.01,
            "std_min": 0.99,
            "std_max": 1.01,
            "lag_abs_max": 0.02,
            "radial_psd_max": 0.05,
            "covariance_mae_max": 0.03,
            "basis_orthonormality_max": 1e-6,
            "kl_relative_error_max": 1e-5,
            "min_eigenvalue": 1e-8,
        },
        "pilot": {
            "ranks": [8, 16],
            "target_kls": [1.0, 5.0],
            "rbf_length_scales_deg": [45.0, 90.0],
            "expected_configs_per_input_seed": 18,
        },
    }


class LowRankDistributionGateTests(unittest.TestCase):
    def test_cli_uses_configured_coefficient_observation_count(self):
        config = _study_config()
        config["preflight"]["coefficient_min_observations"] = 1234
        summary = {
            "passed": True,
            "requested_configuration_count": 18,
            "attempted_configuration_count": 15,
            "passed_configuration_count": 15,
            "failed_configuration_count": 0,
            "excluded_unattainable_count": 3,
            "eligible_configuration_count": 15,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "scripts.diagnose_nile_lowrank._load_config", return_value=config
        ), mock.patch(
            "scripts.diagnose_nile_lowrank.run_study_preflight",
            return_value=summary,
        ) as run_mock:
            code = diagnose_main(
                [
                    "--config",
                    str(Path(directory) / "config.yaml"),
                    "--output-dir",
                    str(Path(directory) / "output"),
                    "--no-plots",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            run_mock.call_args.kwargs["coefficient_min_observations"], 1234
        )

    def test_preflight_matrix_exactly_matches_runner(self):
        config = _study_config()
        observed = build_pilot_configurations(config)
        expected = runner_pilot_configurations(config)
        self.assertEqual(len(observed), 18)
        self.assertEqual(observed, expected)
        self.assertEqual(len({item["config_id"] for item in observed}), 18)

    def test_empirical_basis_coefficient_covariance_tracks_target(self):
        batch_size, num_views = 512, 3
        generator = torch.Generator().manual_seed(41)
        iid = torch.randn(
            (batch_size * num_views, 1, 4, 4), generator=generator
        )
        basis = build_dct2_basis(1, 4, 4, 3)
        target = torch.tensor(
            [[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]],
            dtype=torch.float64,
        )
        output = correlate_orthonormal_subspace(
            iid, basis, target, num_views
        )
        empirical = empirical_basis_coefficient_covariance(
            output,
            basis,
            batch_size=batch_size,
            num_views=num_views,
        )
        self.assertLess(float((empirical - target).abs().mean()), 0.035)

    def test_combined_lowrank_gates_cover_basis_covariance_kl_and_eigen(self):
        batch_size, num_views = 128, 3
        generator = torch.Generator().manual_seed(12)
        iid = torch.randn(
            (batch_size * num_views, 1, 8, 8), generator=generator
        )
        basis = build_dct2_basis(1, 8, 8, 4)
        target = torch.tensor(
            [[1.0, 0.5, 0.1], [0.5, 1.0, 0.25], [0.1, 0.25, 1.0]],
            dtype=torch.float64,
        )
        effective = mix_covariance_with_identity(target, 0.35)
        output = correlate_orthonormal_subspace(
            iid, basis, effective, num_views
        )
        coefficient_generator = torch.Generator().manual_seed(99)
        coefficient_iid = torch.randn(
            (20000, num_views),
            generator=coefficient_generator,
            dtype=torch.float64,
        )
        factor = torch.linalg.cholesky(effective)
        supplemental = coefficient_iid @ factor.mT
        report = diagnose_lowrank_latents(
            output,
            batch_size=batch_size,
            num_views=num_views,
            basis=basis,
            coefficient_target_covariance=effective,
            additional_coefficient_samples=supplemental,
            target_kl=1.0,
            achieved_kl=1.0 + 1e-7,
            alpha=0.35,
        )
        relaxed_distribution = {
            "max_abs_mean": 0.1,
            "min_std": 0.8,
            "max_std": 1.2,
            "max_abs_lag_autocorrelation": 0.2,
            "max_radial_psd_deviation": 0.5,
            "max_axis_stripe_score": 1.0,
            "max_cross_view_covariance_mae": 0.1,
        }
        gates = evaluate_lowrank_distribution_gates(
            report, distribution_thresholds=relaxed_distribution
        )
        self.assertTrue(gates["passed"], gates)
        for name in (
            "finite_higher_moments",
            "basis_orthonormality",
            "basis_coefficient_covariance",
            "joint_kl",
            "minimum_eigenvalue",
            "covariance_condition_number",
        ):
            self.assertIn(name, gates["checks"])

        damaged = copy.deepcopy(report)
        damaged["basis"]["orthonormality_max_error"] = 1e-3
        failed = evaluate_lowrank_distribution_gates(
            damaged, distribution_thresholds=relaxed_distribution
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["basis_orthonormality"]["passed"])

    def test_unattainable_kl_is_excluded_before_sampling(self):
        config = _study_config()
        configuration = next(
            item
            for item in build_pilot_configurations(config)
            if item["method"] == "lowrank_nested_tree_a"
            and item["rank"] == 8
            and item["target_kl"] == 5.0
        )
        record = run_configuration_preflight(
            configuration,
            config,
            coefficient_min_observations=32,
        )
        self.assertEqual(record["status"], "excluded")
        self.assertEqual(record["exclusion_reason"], "unattainable_target_kl")
        self.assertFalse(record["sampling_performed"])
        self.assertFalse(record["eligible_for_generation"])
        self.assertNotIn("report", record)

    def test_unattainable_does_not_mask_real_gate_failure(self):
        unattainable = {
            "config_id": "u",
            "sampling_performed": False,
            "passed": False,
            "exclusion_reason": "unattainable_target_kl",
        }
        passing = {
            "config_id": "p",
            "sampling_performed": True,
            "passed": True,
            "exclusion_reason": None,
        }
        summary = summarize_configuration_records([unattainable, passing])
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["excluded_unattainable_count"], 1)

        failing = {
            "config_id": "f",
            "sampling_performed": True,
            "passed": False,
            "exclusion_reason": "distribution_gate_failed",
        }
        failed_summary = summarize_configuration_records(
            [unattainable, passing, failing]
        )
        self.assertFalse(failed_summary["passed"])
        self.assertEqual(failed_summary["failed_config_ids"], ["f"])

        exception = {
            "config_id": "e",
            "sampling_performed": False,
            "passed": False,
            "exclusion_reason": "preflight_exception",
        }
        exception_summary = summarize_configuration_records(
            [unattainable, passing, exception]
        )
        self.assertFalse(exception_summary["passed"])
        self.assertEqual(exception_summary["failed_config_ids"], ["e"])

    def test_artifact_writer_emits_json_csv_and_plots(self):
        record = {
            "config_id": "baseline",
            "method": "lowrank_camera_rbf",
            "rank": 8,
            "target_kl": 1.0,
            "rbf_length_scale_deg": 45.0,
            "effective_covariance_metadata": {
                "eigenvalues": [0.6, 0.8, 0.9, 1.0, 1.2, 1.5]
            },
            "sampling_performed": True,
            "passed": True,
            "eligible_for_generation": True,
            "exclusion_reason": None,
            "failed_checks": [],
            "report": {
                "global": {
                    "mean": 0.0,
                    "std": 1.0,
                    "skewness": 0.0,
                    "excess_kurtosis": 0.0,
                },
                "lag_autocorrelation": {"values": {"0,1": 0.0, "1,0": 0.0}},
                "radial_psd_deviation": 0.0,
                "axis_stripe_score": {"max": 0.0},
                "cross_view_covariance": [[1.0, 0.0], [0.0, 1.0]],
            },
            "gates": {"checks": {"mean": {"passed": True}}},
        }
        payload = {
            "schema_version": 1,
            "passed": True,
            "configurations": [record],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_preflight_artifacts(root, payload, plots=True)
            self.assertTrue((root / "configuration_gates.json").is_file())
            self.assertTrue((root / "configuration_gates.csv").is_file())
            self.assertTrue((root / "preflight_summary.json").is_file())
            stored = json.loads(
                (root / "configuration_gates.json").read_text(encoding="utf-8")
            )
            self.assertIn("diagnostic_plots", stored)
            self.assertTrue(stored["diagnostic_plots_complete"])
            self.assertTrue(stored["diagnostic_plot_audit"]["complete"])
            self.assertTrue(
                (root / "diagnostics" / "configuration_gate_matrix.png").is_file()
            )
            eigenvalue_plot = (
                root / "diagnostics" / "covariance_eigenvalue_spectra.png"
            )
            self.assertTrue(eigenvalue_plot.is_file())
            self.assertGreater(eigenvalue_plot.stat().st_size, 0)
            self.assertEqual(
                stored["diagnostic_plots"]["covariance_eigenvalue_spectra"],
                "diagnostics/covariance_eigenvalue_spectra.png",
            )


if __name__ == "__main__":
    unittest.main()
