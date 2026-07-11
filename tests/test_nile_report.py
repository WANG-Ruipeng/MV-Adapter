"""Fixture tests for evidence-bounded low-rank study reporting."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.report_nile_lowrank_study import (
    _preflight_passed,
    classify_scientific_result,
    generate_report,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metric_row(input_hash: str, method: str, score: float) -> dict:
    return {
        "input_hash": input_hash,
        "input_image": input_hash + ".png",
        "seed": 0,
        "method": method,
        "status": "succeeded",
        "angle_all_met3r_score": score,
        "dino_identity_mean_delta": 0.0,
        "small_component_ratio_delta": 0.0,
        "component_failure_rate_delta": 0.0,
        "foreground_area_cv_delta": 0.0,
        "artifact_failure": False,
        "r_hf": 1.0,
        "collapse_detector_label": "no_collapse_signal",
    }


def _comparison_row(
    method: str,
    baseline: str,
    delta: float,
    *,
    ci=None,
    win_rate: float = 1.0,
    holm_p: float = 0.04,
) -> dict:
    return {
        "comparison_id": "{}__vs__{}__cfg".format(method, baseline),
        "method": method,
        "config_id": method + "-cfg",
        "comparison_baseline": baseline,
        "pair_count": 2,
        "object_cluster_count": 2,
        "mean_delta": delta,
        "median_delta": delta,
        "std_delta": 0.0,
        "win_rate": win_rate,
        "bootstrap_95_ci": list(ci if ci is not None else (delta, delta)),
        "wilcoxon_p": 0.02,
        "holm_bonferroni_p": holm_p,
        "effect_size_dz": None,
    }


def _current_comparison_rows() -> list:
    return [
        _comparison_row("selected_camera_rbf", "iid_external", -0.05),
        _comparison_row("selected_nested_tree_a", "iid_external", -0.10),
        _comparison_row("selected_nested_tree_ab", "iid_external", -0.08),
        _comparison_row(
            "selected_nested_tree_a", "selected_camera_rbf", -0.05
        ),
        _comparison_row(
            "selected_nested_tree_ab", "selected_camera_rbf", -0.03
        ),
    ]


class NileReportFixtureTests(unittest.TestCase):
    def test_current_preflight_requires_diagnostic_plots(self):
        self.assertTrue(
            _preflight_passed(
                {"passed": True, "diagnostic_plots_complete": True}
            )
        )
        self.assertFalse(
            _preflight_passed(
                {"passed": True, "diagnostic_plots_complete": False}
            )
        )

    def _complete_fixture(self, root: Path) -> None:
        config = {
            "data": {"pilot_count": 1, "full_count": 2, "min_distinct_inputs": 3},
            "pilot": {
                "seeds": [0],
                "expected_configs_per_input_seed": 5,
            },
            "full": {
                "seeds": [0],
                "methods": [
                    "iid_external",
                    "shared_full",
                    "selected_camera_rbf",
                    "selected_nested_tree_a",
                    "selected_nested_tree_ab",
                ],
            },
            "selection": {
                "max_dino_drop_abs": 0.02,
                "max_small_component_ratio_increase": 0.02,
                "max_component_failure_rate_increase": 0.10,
                "max_foreground_area_cv_increase": 0.05,
                "rhf_min": 0.5,
                "rhf_max": 1.5,
            },
            "evaluation": {
                "met3r_repository": "https://github.com/mohammadasim98/met3r",
                "met3r_revision": "ee0e1752898559e1a3e85e2e151d3edeb9b55f73",
            },
        }
        _write_json(root / "configs" / "resolved_config.json", config)
        canonical = json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        _write_json(
            root / "configs" / "config_lock.json",
            {"config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()},
        )
        _write_json(
            root / "inputs" / "input_validation.json",
            {
                "formal_ready": True,
                "distinct_count": 3,
                "required_pilot_count": 1,
                "required_full_count": 2,
                "pilot_count": 1,
                "full_count": 2,
                "missing_distinct_inputs": 0,
            },
        )
        _write_json(
            root / "distribution_gates" / "configuration_gates.json",
            {
                "passed": True,
                "configurations": [
                    {"config_id": "lowrank-0", "passed": True},
                    {"config_id": "lowrank-1", "passed": True},
                ],
            },
        )
        pilot_summaries = [
            {
                "method": method,
                "rank": 8 if "lowrank" in method else None,
                "target_kl": 1.0 if "lowrank" in method else None,
                "alpha": 0.2 if "lowrank" in method else None,
                "met3r_all_pair_mean": score,
                "dino_identity_mean_delta": 0.0,
                "r_hf": 1.0,
            }
            for method, score in (
                ("iid_external", 0.50),
                ("shared_full", 0.49),
                ("lowrank_camera_rbf", 0.45),
                ("lowrank_nested_tree_a", 0.40),
                ("lowrank_nested_tree_ab", 0.42),
            )
        ]
        _write_json(
            root / "pilot" / "metrics" / "lowrank_metrics.json",
            {
                "met3r_required": True,
                "met3r_score_direction": "lower_is_better",
                "configuration_summaries": pilot_summaries,
                "plots": {
                    "complete": True,
                    "plots_dir": str(root / "plots" / "pilot"),
                    "artifacts": [
                        str(root / "plots" / "pilot" / "pareto.png")
                    ],
                },
                "contact_sheets": {
                    "complete": True,
                    "directory": str(root / "contact_sheets" / "pilot"),
                    "paired_sheet_count": 1,
                    "failure_row_count": 0,
                    "failure_gallery": str(
                        root
                        / "contact_sheets"
                        / "pilot"
                        / "failure_gallery.jpg"
                    ),
                    "artifacts": [
                        str(
                            root
                            / "contact_sheets"
                            / "pilot"
                            / "paired.jpg"
                        ),
                        str(
                            root
                            / "contact_sheets"
                            / "pilot"
                            / "failure_gallery.jpg"
                        ),
                    ],
                },
            },
        )
        selections = {}
        for topology, method in (
            ("camera_rbf", "lowrank_camera_rbf"),
            ("nested_tree_a", "lowrank_nested_tree_a"),
            ("nested_tree_ab", "lowrank_nested_tree_ab"),
        ):
            selections[topology] = {
                "status": "selected",
                "diagnostic_only": False,
                "configuration": {
                    "method": method,
                    "rank": 8,
                    "target_kl": 1.0,
                    "achieved_kl": 1.0,
                    "alpha": 0.2,
                    "rbf_length_scale_deg": 45.0 if topology == "camera_rbf" else None,
                },
            }
        _write_json(
            root / "selected_candidates" / "selected_candidates.json",
            {"configuration_hash": "abc123", "selections": selections},
        )
        samples = []
        for input_hash in ("object-a", "object-b"):
            samples.extend(
                [
                    _metric_row(input_hash, "iid_external", 0.50),
                    _metric_row(input_hash, "shared_full", 0.49),
                    _metric_row(input_hash, "selected_camera_rbf", 0.45),
                    _metric_row(input_hash, "selected_nested_tree_a", 0.40),
                    _metric_row(input_hash, "selected_nested_tree_ab", 0.42),
                ]
            )
        _write_json(
            root / "full" / "metrics" / "lowrank_metrics.json",
            {
                "met3r_required": True,
                "met3r_score_direction": "lower_is_better",
                "samples": samples,
                "plots": {
                    "complete": True,
                    "plots_dir": str(root / "plots" / "full"),
                    "artifacts": [
                        str(root / "plots" / "full" / "paired_delta.png")
                    ],
                },
                "contact_sheets": {
                    "complete": True,
                    "directory": str(root / "contact_sheets" / "full"),
                    "paired_sheet_count": 2,
                    "failure_row_count": 0,
                    "failure_gallery": str(
                        root
                        / "contact_sheets"
                        / "full"
                        / "failure_gallery.jpg"
                    ),
                    "artifacts": [
                        str(
                            root
                            / "contact_sheets"
                            / "full"
                            / "paired_a.jpg"
                        ),
                        str(
                            root
                            / "contact_sheets"
                            / "full"
                            / "paired_b.jpg"
                        ),
                        str(
                            root
                            / "contact_sheets"
                            / "full"
                            / "failure_gallery.jpg"
                        ),
                    ],
                },
                "paired_statistics": [
                    {
                        "method": "selected_camera_rbf",
                        "pair_count": 2,
                        "mean_delta": -0.05,
                        "bootstrap_95_ci": [-0.05, -0.05],
                        "wilcoxon_p": 0.5,
                        "holm_bonferroni_p": 1.0,
                    },
                    {
                        "method": "selected_nested_tree_a",
                        "pair_count": 2,
                        "mean_delta": -0.10,
                        "bootstrap_95_ci": [-0.10, -0.10],
                        "wilcoxon_p": 0.5,
                        "holm_bonferroni_p": 1.0,
                    },
                ],
            },
        )
        _write_json(
            root / "trajectory" / "trajectory_summary.json",
            {"complete": True, "correlation_state": "retained"},
        )
        pilot_runs = []
        for index in range(5):
            pilot_runs.append({"run_id": "run-{}".format(index), "status": "succeeded"})
        full_runs = []
        for index in range(10):
            full_runs.append({"run_id": "run-{}".format(index), "status": "succeeded"})
        _write_json(root / "pilot" / "manifest.json", {"split": "pilot", "runs": pilot_runs})
        _write_json(root / "full" / "manifest.json", {"split": "full", "runs": full_runs})
        _write_json(
            root / "trajectory" / "manifest.json",
            {"split": "trajectory", "runs": [{"run_id": "trajectory-0", "status": "succeeded"}]},
        )
        _write_json(
            root / "runtime_status.json",
            {
                "implementation_complete": True,
                "tests_complete": True,
                "pilot_complete": True,
                "full_complete": True,
                "met3r_complete": True,
                "blockers": [],
            },
        )

    def test_complete_fixture_generates_four_reports_and_nested_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            report = generate_report(root)

            self.assertTrue(report["overall_complete"])
            self.assertEqual(report["scientific_judgment"]["label"], "nested_positive")
            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["run_counts"]["pilot"]["succeeded"], 5)
            self.assertEqual(report["run_counts"]["full"]["succeeded"], 10)
            for name in (
                "FULL_EXPERIMENT_REPORT.md",
                "FULL_EXPERIMENT_REPORT.json",
                "FINAL_STATUS.json",
                "REPRODUCE.md",
            ):
                self.assertTrue((root / name).is_file(), name)
            status = json.loads((root / "FINAL_STATUS.json").read_text(encoding="utf-8"))
            self.assertTrue(status["implementation_complete"])
            self.assertTrue(status["tests_complete"])
            self.assertTrue(status["full_complete"])
            self.assertTrue(status["completion"]["report_complete"])
            self.assertTrue(status["completion"]["met3r_complete"])
            markdown = (root / "FULL_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("strict NILE/SZ", markdown)
            self.assertIn("-0.1", markdown)
            reproduce = (root / "REPRODUCE.md").read_text(encoding="utf-8")
            self.assertIn("--stage all --resume", reproduce)
            self.assertNotIn("--output-root", reproduce)
            self.assertIn("Recommended: Colab Run All", reproduce)
            self.assertIn(
                "notebooks/mvadapter_nile_lowrank_full_colab.ipynb",
                reproduce,
            )
            self.assertIn("Runtime > Run all", reproduce)
            self.assertIn(
                "python -m pip install -r requirements-colab.txt",
                reproduce,
            )
            self.assertIn(
                "ee0e1752898559e1a3e85e2e151d3edeb9b55f73",
                reproduce,
            )
            self.assertIn("CHECKPOINT_MANIFEST=", reproduce)
            self.assertIn("INPUT_HASH_MANIFEST=", reproduce)
            self.assertIn("CANDIDATE_FILE=", reproduce)
            self.assertIn("test_results.json", reproduce)
            self.assertIn('json.dumps(payload, indent=2) + "\\n"', reproduce)
            self.assertIn("HF_TOKEN", reproduce)
            self.assertIn("token value must not be", reproduce)
            self.assertIn("printed.", reproduce)
            self.assertEqual(
                report["full_statistics_schema"]["source"], "paired_statistics"
            )
            self.assertFalse(
                report["full_statistics_schema"]["authoritative"]
            )
            visuals = report["evaluation_visual_artifacts"]
            self.assertEqual(visuals["pilot"]["paired_sheet_count"], 1)
            self.assertEqual(visuals["full"]["paired_sheet_count"], 2)
            self.assertTrue(
                visuals["full"]["failure_gallery"].endswith(
                    "failure_gallery.jpg"
                )
            )
            self.assertEqual(report["failure_cases"]["total_count"], 0)
            self.assertIn("Evaluator visual artifacts", markdown)
            self.assertIn("failure_gallery.jpg", markdown)
            self.assertIn("[failure_gallery.jpg](", markdown)
            self.assertIn("[pareto.png](", markdown)

    def test_current_schema_reports_iid_and_nested_rbf_global_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = _current_comparison_rows()
            _write_json(metrics_path, metrics)

            report = generate_report(root)

            self.assertTrue(report["completion"]["full_complete"])
            self.assertEqual(
                report["scientific_judgment"]["label"], "nested_positive"
            )
            schema = report["full_statistics_schema"]
            self.assertEqual(schema["source"], "paired_comparison_statistics")
            self.assertTrue(schema["authoritative"])
            self.assertTrue(schema["complete"])
            self.assertEqual(schema["holm_scope"], "global_iid_and_nested_vs_rbf")
            self.assertEqual(len(report["full_statistics"]), 5)
            self.assertEqual(len(report["full_iid_statistics"]), 3)
            self.assertEqual(len(report["full_nested_vs_rbf_statistics"]), 2)
            self.assertEqual(
                report["full_nested_vs_rbf_statistics"][0][
                    "comparison_baseline"
                ],
                "selected_camera_rbf",
            )
            self.assertEqual(
                report["full_nested_vs_rbf_statistics"][0][
                    "holm_bonferroni_p"
                ],
                0.04,
            )
            markdown = (root / "FULL_EXPERIMENT_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Methods vs IID", markdown)
            self.assertIn("Nested methods vs selected camera RBF", markdown)
            self.assertIn("95% cluster-bootstrap CI", markdown)
            self.assertIn("global Holm p", markdown)
            self.assertIn("selected_camera_rbf", markdown)

    def test_current_schema_missing_nested_rbf_is_a_formal_blocker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = [
                row
                for row in _current_comparison_rows()
                if row["comparison_baseline"] == "iid_external"
            ]
            _write_json(metrics_path, metrics)

            report = generate_report(root)

            self.assertFalse(report["completion"]["full_complete"])
            self.assertEqual(
                report["scientific_judgment"]["label"], "full_blocked"
            )
            self.assertIn(
                "paired_comparison_statistics_incomplete",
                {item["code"] for item in report["blockers"]},
            )

    def test_current_statistics_override_raw_deltas_without_overclaiming(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            rows = _current_comparison_rows()
            for row in rows:
                if row["comparison_baseline"] == "selected_camera_rbf":
                    row.update(
                        {
                            "mean_delta": 0.01,
                            "median_delta": 0.01,
                            "win_rate": 0.25,
                            "bootstrap_95_ci": [-0.02, 0.04],
                            "wilcoxon_p": 0.8,
                            "holm_bonferroni_p": 1.0,
                        }
                    )
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = rows
            _write_json(metrics_path, metrics)

            report = generate_report(root)

            self.assertTrue(report["completion"]["full_complete"])
            self.assertEqual(
                report["scientific_judgment"]["label"],
                "generic_coupling_only",
            )
            paired = report["scientific_judgment"]["evidence"]["paired"]
            self.assertEqual(
                paired["selected_nested_tree_a"]["vs_rbf"]["mean_delta"],
                0.01,
            )
            self.assertEqual(
                report["scientific_judgment"]["evidence"][
                    "statistics_source"
                ],
                "paired_comparison_statistics",
            )

    def test_nested_positive_requires_equal_rank_and_target_kl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = _current_comparison_rows()
            _write_json(metrics_path, metrics)
            selected_path = (
                root / "selected_candidates" / "selected_candidates.json"
            )
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            selected["selections"]["nested_tree_a"]["configuration"]["rank"] = 16
            selected["selections"]["nested_tree_ab"]["configuration"][
                "target_kl"
            ] = 5.0
            _write_json(selected_path, selected)

            report = generate_report(root)

            self.assertFalse(report["completion"]["full_complete"])
            self.assertEqual(
                report["scientific_judgment"]["label"],
                "full_blocked",
            )
            self.assertEqual(report["run_counts"]["full"]["succeeded"], 10)
            self.assertIn(
                "equal_rank_kl_fairness_mismatch",
                {item["code"] for item in report["blockers"]},
            )
            audit = report["formal_equal_rank_kl_audit"]
            self.assertFalse(audit["complete"])
            by_method = {
                row["method"]: row for row in audit["records"]
            }
            self.assertFalse(by_method["selected_nested_tree_a"]["valid"])
            self.assertFalse(by_method["selected_nested_tree_ab"]["valid"])
            markdown = (root / "FULL_EXPERIMENT_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Equal-rank/equal-target-KL fairness audit", markdown)
            self.assertIn("fairness_mismatch", markdown)

    def test_diagnostic_no_eligible_nested_is_exempt_from_formal_fairness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = _current_comparison_rows()
            _write_json(metrics_path, metrics)
            selected_path = (
                root / "selected_candidates" / "selected_candidates.json"
            )
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            diagnostic = selected["selections"]["nested_tree_ab"]
            diagnostic["status"] = "no_eligible_candidate"
            diagnostic["diagnostic_only"] = True
            diagnostic["configuration"]["rank"] = 32
            diagnostic["configuration"]["target_kl"] = 5.0
            _write_json(selected_path, selected)

            report = generate_report(root)

            self.assertTrue(report["completion"]["full_complete"])
            self.assertNotIn(
                "equal_rank_kl_fairness_mismatch",
                {item["code"] for item in report["blockers"]},
            )
            audit = report["formal_equal_rank_kl_audit"]
            self.assertTrue(audit["complete"])
            self.assertEqual(audit["formal_comparison_count"], 1)
            self.assertEqual(
                audit["records"][0]["method"], "selected_nested_tree_a"
            )

    def test_nested_positive_uses_method_level_trajectory_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = _current_comparison_rows()
            _write_json(metrics_path, metrics)
            _write_json(
                root / "trajectory" / "trajectory_summary.json",
                {
                    "schema_version": 2,
                    "complete": True,
                    "correlation_state": "retained",
                    "classification_thresholds": {
                        "washout_max_final_g": 0.5,
                        "amplified_min_final_g": 1.5,
                    },
                    "method_summaries": [
                        {
                            "method": "selected_camera_rbf",
                            "pair_count": 2,
                            "rank": 8,
                            "final_g_t": 1.0,
                        },
                        {
                            "method": "selected_nested_tree_a",
                            "pair_count": 2,
                            "rank": 8,
                            "final_g_t": 0.2,
                        },
                        {
                            "method": "selected_nested_tree_ab",
                            "pair_count": 2,
                            "rank": 8,
                            "final_g_t": 1.8,
                        },
                    ],
                },
            )

            report = generate_report(root)

            self.assertEqual(
                report["scientific_judgment"]["label"],
                "generic_coupling_only",
            )
            states = report["scientific_judgment"]["evidence"][
                "method_trajectory"
            ]
            self.assertEqual(
                states["selected_nested_tree_a"]["state"], "wash_out"
            )
            self.assertEqual(
                states["selected_nested_tree_ab"]["state"], "amplified"
            )
            self.assertEqual(
                states["selected_camera_rbf"]["state"], "retained"
            )

    def test_full_guardrails_include_r_hf_and_collapse_alert(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["paired_comparison_statistics"] = _current_comparison_rows()
            for row in metrics["samples"]:
                if row["method"] == "selected_nested_tree_a":
                    row["r_hf"] = 1.75
                if row["method"] == "selected_nested_tree_ab":
                    row["collapse_detector_label"] = "view_collapse_alert"
            _write_json(metrics_path, metrics)

            report = generate_report(root)

            self.assertEqual(
                report["scientific_judgment"]["label"],
                "generic_coupling_only",
            )
            guardrails = report["scientific_judgment"]["evidence"][
                "guardrails_safe"
            ]
            self.assertFalse(guardrails["selected_nested_tree_a"])
            self.assertFalse(guardrails["selected_nested_tree_ab"])
            self.assertTrue(guardrails["selected_camera_rbf"])

    def test_failure_cases_only_include_explicit_sample_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            metrics_path = root / "full" / "metrics" / "lowrank_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["samples"][0]["status"] = "failed"
            metrics["samples"][1]["guardrail_error"] = "mask backend failed"
            metrics["samples"][2]["artifact_failure"] = True
            metrics["samples"][3][
                "collapse_detector_label"
            ] = "view_collapse_alert"
            _write_json(metrics_path, metrics)

            report = generate_report(root)

            failures = report["failure_cases"]
            self.assertEqual(failures["total_count"], 4)
            self.assertEqual(failures["reported_count"], 4)
            self.assertFalse(failures["truncated"])
            reason_sets = {
                reason
                for row in failures["records"]
                for reason in row["reasons"]
            }
            self.assertEqual(
                reason_sets,
                {
                    "status_failed",
                    "guardrail_error",
                    "artifact_failure",
                    "view_collapse_alert",
                },
            )
            markdown = (root / "FULL_EXPERIMENT_REPORT.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Explicitly recorded sample failures", markdown)
            self.assertIn("mask backend failed", markdown)
            self.assertIn("view_collapse_alert", markdown)

    def test_missing_artifacts_are_blocked_without_fabricated_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = generate_report(root)

            self.assertEqual(report["scientific_judgment"]["label"], "full_blocked")
            self.assertFalse(report["completion"]["pilot_complete"])
            self.assertFalse(report["completion"]["full_complete"])
            self.assertFalse(report["completion"]["met3r_complete"])
            self.assertTrue(report["completion"]["report_complete"])
            self.assertEqual(report["pilot_summaries"], [])
            self.assertEqual(report["full_statistics"], [])
            self.assertEqual(report["failure_cases"]["total_count"], 0)
            self.assertIsNone(
                report["evaluation_visual_artifacts"]["full"][
                    "failure_gallery"
                ]
            )
            codes = {row["code"] for row in report["blockers"]}
            self.assertIn("manifest_missing", codes)
            self.assertIn("full_metrics_missing", codes)
            reproduce = (root / "REPRODUCE.md").read_text(encoding="utf-8")
            self.assertIn("Recommended: Colab Run All", reproduce)
            self.assertIn(
                "<missing immutable revision in resolved config>", reproduce
            )
            self.assertIn(
                "BLOCKED: resolved config lacks a safe repository", reproduce
            )
            markdown = (root / "FULL_EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("未估算任何指标", markdown)

    def test_reproduce_redacts_repository_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / "configs" / "resolved_config.json",
                {
                    "evaluation": {
                        "met3r_repository": (
                            "https://user:super-secret@github.com/"
                            "mohammadasim98/met3r?token=also-secret"
                        ),
                        "met3r_revision": (
                            "ee0e1752898559e1a3e85e2e151d3edeb9b55f73"
                        ),
                    }
                },
            )

            generate_report(root)
            reproduce = (root / "REPRODUCE.md").read_text(encoding="utf-8")

            self.assertNotIn("super-secret", reproduce)
            self.assertNotIn("also-secret", reproduce)
            self.assertIn(
                "git+https://github.com/mohammadasim98/met3r@"
                "ee0e1752898559e1a3e85e2e151d3edeb9b55f73",
                reproduce,
            )

    def test_runtime_blocker_prevents_full_completion_despite_stale_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._complete_fixture(root)
            _write_json(
                root / "runtime_status.json",
                {
                    "implementation_complete": True,
                    "tests_complete": True,
                    "pilot_complete": True,
                    "full_complete": True,
                    "met3r_complete": True,
                    "blockers": [
                        {
                            "code": "model_revisions_not_immutable",
                            "fields": ["adapter_revision"],
                        }
                    ],
                },
            )
            report = generate_report(root)

            self.assertFalse(report["completion"]["full_complete"])
            self.assertEqual(
                report["scientific_judgment"]["label"], "full_blocked"
            )
            self.assertIn(
                "model_revisions_not_immutable",
                {item["code"] for item in report["blockers"]},
            )

    def test_generic_and_no_go_labels_do_not_overstate_nested(self):
        samples = []
        for index in range(2):
            key = "object-{}".format(index)
            samples.extend(
                [
                    _metric_row(key, "iid_external", 0.50),
                    _metric_row(key, "selected_camera_rbf", 0.40),
                    _metric_row(key, "selected_nested_tree_a", 0.45),
                ]
            )
        selected = {
            "selections": {
                "camera_rbf": {
                    "status": "selected",
                    "diagnostic_only": False,
                    "configuration": {
                        "method": "selected_camera_rbf",
                        "rank": 8,
                        "target_kl": 1.0,
                    },
                },
                "nested_tree_a": {
                    "status": "selected",
                    "diagnostic_only": False,
                    "configuration": {
                        "method": "selected_nested_tree_a",
                        "rank": 8,
                        "target_kl": 1.0,
                    },
                },
                "nested_tree_ab": {
                    "status": "selected",
                    "diagnostic_only": False,
                    "configuration": {
                        "method": "selected_nested_tree_ab",
                        "rank": 8,
                        "target_kl": 1.0,
                    },
                },
            }
        }
        generic = classify_scientific_result(
            full_complete=True,
            met3r_complete=True,
            trajectory_complete=True,
            full_metrics={"samples": samples},
            trajectory={"correlation_state": "retained"},
            selected_candidates=selected,
        )
        self.assertEqual(generic["label"], "generic_coupling_only")

        diagnostic_samples = [dict(row) for row in samples]
        for row in diagnostic_samples:
            if row["method"] == "selected_nested_tree_a":
                row["angle_all_met3r_score"] = 0.35
        diagnostic_selected = json.loads(json.dumps(selected))
        diagnostic_selected["selections"]["nested_tree_a"].update(
            {"status": "no_eligible_candidate", "diagnostic_only": True}
        )
        diagnostic = classify_scientific_result(
            full_complete=True,
            met3r_complete=True,
            trajectory_complete=True,
            full_metrics={"samples": diagnostic_samples},
            trajectory={"correlation_state": "retained"},
            selected_candidates=diagnostic_selected,
        )
        self.assertEqual(diagnostic["label"], "generic_coupling_only")

        for row in samples:
            if row["method"] != "iid_external":
                row["angle_all_met3r_score"] = 0.55
        no_go = classify_scientific_result(
            full_complete=True,
            met3r_complete=True,
            trajectory_complete=True,
            full_metrics={"samples": samples},
            trajectory={"correlation_state": "wash_out"},
            selected_candidates=selected,
        )
        self.assertEqual(no_go["label"], "initial_noise_no_go")

        missing_guardrails = [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "dino_identity_mean_delta",
                    "small_component_ratio_delta",
                    "component_failure_rate_delta",
                    "foreground_area_cv_delta",
                }
            }
            for row in samples
        ]
        blocked = classify_scientific_result(
            full_complete=True,
            met3r_complete=True,
            trajectory_complete=True,
            full_metrics={"samples": missing_guardrails},
            trajectory={"correlation_state": "retained"},
            selected_candidates=selected,
        )
        self.assertEqual(blocked["label"], "full_blocked")


if __name__ == "__main__":
    unittest.main()
