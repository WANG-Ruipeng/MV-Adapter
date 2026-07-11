import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts.eval_nile_lowrank_study import (
    _row_grid_path,
    add_metric_comparison_deltas,
    add_paired_deltas,
    generate_contact_sheets,
    paired_statistics,
    reconcile_manifest_samples,
    resolve_contact_sheets_directory,
    resolve_plots_directory,
    verify_installed_met3r_revision,
)
from scripts.run_nile_lowrank_study import (
    _execute_persistent_worker,
    _formal_blockers,
    _hash,
    audit_checkpoint_manifest,
    audit_input_split_isolation,
    audit_selected_candidates,
    audit_test_results_receipt,
    audit_pilot_met3r_prerequisite,
    audit_run_artifacts,
    build_pilot_configurations,
    build_run_plan,
    build_trajectory_configurations,
    execute_plan,
    gated_pilot_configurations,
    lock_resolved_config,
    summarize_trajectory_stage,
    validate_inputs_stage,
    write_manifest,
)
from scripts.select_nile_lowrank_candidates import select_candidates
from scripts.validate_nile_inputs import discover_images, inspect_inputs, stable_split


def _config():
    return {
        "experiment": {"strict_full_requires_met3r": True},
        "model": {
            "height": 768,
            "width": 768,
            "views_deg": [0, 45, 90, 180, 270, 315],
            "num_inference_steps": 30,
            "guidance_scale": 3.0,
            "scheduler": None,
            "base_model": "base",
            "vae_model": "vae",
            "adapter_path": "adapter",
            "mv_adapter_checkpoint": "mvadapter_i2mv_sdxl.safetensors",
            "device": "cuda",
        },
        "pilot": {
            "ranks": [8, 16],
            "target_kls": [1.0, 5.0],
            "rbf_length_scales_deg": [45.0, 90.0],
            "expected_configs_per_input_seed": 18,
        },
        "preflight": {"expected_unattainable_count": 3},
        "runtime": {"max_retries": 0},
    }


def _eligible_row(method, config_id, score, target_kl=1.0, rank=8):
    return {
        "method": method,
        "config_id": config_id,
        "rank": rank,
        "target_kl": target_kl,
        "achieved_kl": target_kl,
        "alpha": 0.2,
        "rbf_length_scale_deg": 45.0 if method == "lowrank_camera_rbf" else None,
        "basis_checksum": "basis",
        "covariance_checksum": "covariance",
        "distribution_gate_passed": True,
        "output_missing": False,
        "met3r_all_pair_mean": score,
        "met3r_standard_error": 0.0,
        "met3r_failure_rate": 0.0,
        "dino_identity_mean_delta": 0.0,
        "small_component_ratio_delta": 0.0,
        "component_failure_rate_delta": 0.0,
        "foreground_area_cv_delta": 0.0,
        "r_hf": 1.0,
        "collapse_detector_label": "no_collapse_signal",
    }


def _write_selection_proof(
    root, config, *, rank_mismatch=False, diagnostic_tree_ab=False
):
    requested = build_pilot_configurations(config)
    excluded_ids = {item["config_id"] for item in requested[-3:]}
    gate_rows = []
    summaries = []
    for item in requested:
        excluded = item["config_id"] in excluded_ids
        gate = {
            **item,
            "passed": not excluded,
            "eligible_for_generation": not excluded,
            "exclusion_reason": (
                "unattainable_target_kl" if excluded else None
            ),
            "achieved_kl": item.get("target_kl"),
            "alpha": 0.2,
            "basis_checksum": "basis-" + item["config_id"],
            "covariance_checksum": "cov-" + item["config_id"],
        }
        gate_rows.append(gate)
        if excluded or item["method"] not in (
            "lowrank_camera_rbf",
            "lowrank_nested_tree_a",
            "lowrank_nested_tree_ab",
        ):
            continue
        score = (
            float(item["target_kl"])
            + float(item["rank"]) / 1000.0
            + float(item.get("rbf_length_scale_deg") or 0.0) / 100000.0
        )
        if (
            rank_mismatch
            and item["method"].startswith("lowrank_nested_")
            and item["rank"] == 16
            and item["target_kl"] == 1.0
        ):
            score = 0.0
        summary = _eligible_row(
            item["method"],
            item["config_id"],
            score,
            target_kl=item["target_kl"],
            rank=item["rank"],
        )
        summary.update(
            {
                field: gate.get(field)
                for field in (
                    "achieved_kl",
                    "alpha",
                    "rbf_length_scale_deg",
                    "basis_checksum",
                    "covariance_checksum",
                )
            }
        )
        if diagnostic_tree_ab and item["method"] == "lowrank_nested_tree_ab":
            summary["r_hf"] = 2.0
        summaries.append(summary)

    gates_path = root / "distribution_gates" / "configuration_gates.json"
    gates_path.parent.mkdir(parents=True)
    gates_path.write_text(
        json.dumps(
            {
                "diagnostic_plots_complete": True,
                "configurations": gate_rows,
            }
        ),
        encoding="utf-8",
    )
    metrics_path = root / "metrics" / "pilot" / "lowrank_metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        json.dumps({"configuration_summaries": summaries}),
        encoding="utf-8",
    )
    selected = select_candidates(summaries, config.get("selection", {}))
    selected["study_config_hash"] = _hash(config)
    selected["pilot_metrics_sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    selected["candidate_configuration_hash"] = selected["configuration_hash"]
    selected["configuration_hash"] = _hash(
        {
            "candidate_configuration_hash": selected[
                "candidate_configuration_hash"
            ],
            "study_config_hash": selected["study_config_hash"],
            "pilot_metrics_sha256": selected["pilot_metrics_sha256"],
        }
    )
    selected_dir = root / "selected_candidates"
    selected_dir.mkdir(parents=True)
    for name in ("selected_candidates.json", "selected_candidates.yaml"):
        (selected_dir / name).write_text(json.dumps(selected), encoding="utf-8")
    return selected, metrics_path


def _worker_record(root, run_id="worker-run"):
    input_path = root / "input.png"
    input_path.write_bytes(b"input")
    output = root / run_id / "grid.png"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "config_id": "worker-config",
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": 0,
        "method": "iid_external",
        "distribution_gate_passed": True,
        "camera_list": [0, 45, 90, 180, 270, 315],
        "output": str(output),
        "metadata_path": str(output.with_name("grid_metadata.json")),
        "views_dir": str(output.with_name("grid_views")),
        "mask_dir": str(output.with_name("grid_masks")),
        "status": "succeeded",
    }


def _write_worker_bundle(record, config, marker="bundle"):
    output = Path(record["output"])
    reference = output.with_name("grid_reference.png")
    views_dir = Path(record["views_dir"])
    masks_dir = Path(record["mask_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), marker if marker in {"white", "black"} else "white").save(
        output
    )
    Image.new("RGB", (4, 4), "white").save(reference)
    views = []
    masks = []
    for index in range(len(config["model"]["views_deg"])):
        view = views_dir / "view-{}.png".format(index)
        mask = masks_dir / "mask-{}.png".format(index)
        Image.new("RGB", (4, 4), "white").save(view)
        Image.new("L", (4, 4), 255).save(mask)
        views.append(str(view.resolve()))
        masks.append(str(mask.resolve()))
    Path(record["metadata_path"]).write_text(
        json.dumps(
            {
                "config_id": record["config_id"],
                "output": str(output.resolve()),
                "reference_output": str(reference.resolve()),
                "view_files": views,
                "mask_files": masks,
                "azimuth_deg": config["model"]["views_deg"],
                "seed": record["seed"],
                "input": {
                    "image": str(Path(record["input_path"]).resolve()),
                    "sha256": record["input_sha256"],
                },
                "models": {
                    key: config["model"].get(key)
                    for key in (
                        "base_model_revision",
                        "vae_model_revision",
                        "unet_model_revision",
                        "lora_model_revision",
                        "adapter_revision",
                        "birefnet_revision",
                        "scheduler",
                        "mv_adapter_checkpoint",
                    )
                },
                "inference": {
                    "num_inference_steps": config["model"]["num_inference_steps"],
                    "guidance_scale": config["model"]["guidance_scale"],
                },
                "distribution": {
                    "method": record["method"],
                    "target_joint_kl": record.get("target_kl"),
                },
                "marker": marker,
            }
        ),
        encoding="utf-8",
    )


def _write_trajectory_npz(path, *, scale, offdiag):
    milestones = np.asarray(["initial", "50%", "final"], dtype="U32")
    progress = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    coefficients = (
        np.arange(18, dtype=np.float64).reshape(3, 1, 3, 2) + 1.0
    ) * float(scale)
    correlations = []
    for multiplier in (1.0, 0.8, 0.6):
        matrix = np.full((3, 3), float(offdiag) * multiplier, dtype=np.float64)
        np.fill_diagonal(matrix, 1.0)
        correlations.append(matrix)
    view_correlation = np.asarray(correlations, dtype=np.float64)[:, None, :, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray("nile_trajectory_v1"),
        milestones=milestones,
        target_progress=progress,
        actual_progress=progress,
        steps=np.asarray([-1, 0, 1], dtype=np.int64),
        timesteps=np.asarray([1000.0, 500.0, 0.0], dtype=np.float64),
        basis_coefficients=coefficients,
        view_correlation=view_correlation,
        offdiag_frobenius=np.asarray([[1.0], [0.8], [0.6]]),
        per_view_coefficient_norm=np.ones((3, 1, 3), dtype=np.float64),
        g_t=np.asarray([[1.0], [0.8], [0.6]], dtype=np.float64),
        num_views=np.asarray(3, dtype=np.int64),
        batch_size=np.asarray(1, dtype=np.int64),
        basis_rank=np.asarray(2, dtype=np.int64),
        latent_dimension=np.asarray(4, dtype=np.int64),
        basis_checksum=np.asarray("shared-basis"),
    )


def _write_valid_trajectory_pair(root, config_hash):
    iid_path = root / "trajectory" / "iid" / "trajectory.npz"
    correlated_path = root / "trajectory" / "correlated" / "trajectory.npz"
    _write_trajectory_npz(iid_path, scale=1.0, offdiag=0.05)
    _write_trajectory_npz(correlated_path, scale=1.2, offdiag=0.45)
    records = [
        {
            "run_id": "iid",
            "input_sha256": "object",
            "seed": 0,
            "trajectory_pair_id": "pair",
            "trajectory_role": "iid_control",
            "paired_method": "lowrank_camera_rbf",
            "method": "iid_external",
            "rank": 2,
            "status": "succeeded",
            "trajectory_output": str(iid_path),
        },
        {
            "run_id": "correlated",
            "input_sha256": "object",
            "seed": 0,
            "trajectory_pair_id": "pair",
            "trajectory_role": "correlated",
            "method": "lowrank_camera_rbf",
            "rank": 2,
            "target_kl": 1.0,
            "status": "succeeded",
            "trajectory_output": str(correlated_path),
        },
    ]
    write_manifest(
        root / "trajectory" / "manifest.json",
        records,
        config_hash,
        "trajectory",
    )


class WorkflowIdentityTests(unittest.TestCase):
    def test_run_id_is_stable_and_changes_with_full_config_hash(self):
        config = _config()
        input_record = {"path": "object.png", "sha256": "a" * 64}
        method = build_pilot_configurations(config)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kwargs = dict(
                split="pilot",
                inputs=[input_record],
                seeds=[0],
                configurations=[method],
                root=root,
                config=config,
                git_commit="abc",
                code_revision="abc-dirty-state",
                gpu={"name": "cpu"},
            )
            first = build_run_plan(config_hash="1" * 64, **kwargs)
            second = build_run_plan(config_hash="1" * 64, **kwargs)
            changed = build_run_plan(config_hash="2" * 64, **kwargs)
            changed_code = build_run_plan(
                config_hash="1" * 64,
                **{**kwargs, "code_revision": "different-dirty-state"},
            )
        self.assertEqual(first[0]["run_id"], second[0]["run_id"])
        self.assertNotEqual(first[0]["run_id"], changed[0]["run_id"])
        self.assertNotEqual(first[0]["run_id"], changed_code[0]["run_id"])

    def test_config_lock_rejects_different_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"value": 1}
            self.assertEqual(lock_resolved_config(root, original), _hash(original))
            self.assertEqual(lock_resolved_config(root, original), _hash(original))
            with self.assertRaises(ValueError):
                lock_resolved_config(root, {"value": 2})

    def test_resume_skips_only_integrity_verified_success(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_run_plan(
                split="pilot",
                inputs=[{"path": "object.png", "sha256": "b" * 64}],
                seeds=[0],
                configurations=[build_pilot_configurations(config)[0]],
                config_hash="c" * 64,
                root=root,
                config=config,
                git_commit="abc",
                code_revision="abc-dirty-state",
                gpu={},
            )
            previous = dict(plan[0])
            previous["status"] = "succeeded"
            manifest = root / "pilot" / "manifest.json"
            write_manifest(manifest, [previous], "c" * 64, "pilot")
            with mock.patch(
                "scripts.run_nile_lowrank_study.audit_run_artifacts",
                return_value={"complete": True, "issues": []},
            ), mock.patch(
                "scripts.run_nile_lowrank_study.subprocess.run"
            ) as run_mock:
                result = execute_plan(
                    plan,
                    manifest_path=manifest,
                    config_hash="c" * 64,
                    split="pilot",
                    config=config,
                    resume=True,
                    force_rerun=False,
                    dry_run=False,
                    max_runs=None,
                )
            run_mock.assert_not_called()
            self.assertEqual(result["executed"], 0)
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(result["artifact_incomplete"], 0)

    def test_resume_reschedules_corrupt_success(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_run_plan(
                split="pilot",
                inputs=[{"path": "object.png", "sha256": "d" * 64}],
                seeds=[0],
                configurations=[build_pilot_configurations(config)[0]],
                config_hash="e" * 64,
                root=root,
                config=config,
                git_commit="abc",
                code_revision="abc-dirty-state",
                gpu={},
            )
            previous = dict(plan[0])
            previous["status"] = "succeeded"
            manifest = root / "pilot" / "manifest.json"
            write_manifest(manifest, [previous], "e" * 64, "pilot")
            result = execute_plan(
                plan,
                manifest_path=manifest,
                config_hash="e" * 64,
                split="pilot",
                config=config,
                resume=True,
                force_rerun=False,
                dry_run=True,
                max_runs=None,
            )
            persisted = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(result["would_execute"], 1)
        self.assertEqual(result["succeeded"], 0)
        self.assertEqual(persisted["runs"][0]["status"], "succeeded")

    def test_dry_run_does_not_create_a_formal_manifest(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_run_plan(
                split="pilot",
                inputs=[{"path": "object.png", "sha256": "f" * 64}],
                seeds=[0],
                configurations=[build_pilot_configurations(config)[0]],
                config_hash="a" * 64,
                root=root,
                config=config,
                git_commit="abc",
                gpu={},
            )
            manifest = root / "pilot" / "manifest.json"
            result = execute_plan(
                plan,
                manifest_path=manifest,
                config_hash="a" * 64,
                split="pilot",
                config=config,
                resume=True,
                force_rerun=False,
                dry_run=True,
                max_runs=None,
            )
            self.assertFalse(manifest.exists())
        self.assertEqual(result["planned"], 1)
        self.assertEqual(result["would_execute"], 1)
        self.assertGreater(result["estimated_output_bytes"], 0)


class PersistentWorkerMergeTests(unittest.TestCase):
    @staticmethod
    def _execute(root, record, config, popen):
        manifest = root / "pilot" / "manifest.json"
        records = {record["run_id"]: record}
        with mock.patch(
            "scripts.run_nile_lowrank_study.subprocess.Popen", side_effect=popen
        ):
            result = _execute_persistent_worker(
                executable=[record],
                plan=[record],
                records=records,
                order=[record["run_id"]],
                manifest_path=manifest,
                config_hash="a" * 64,
                split="pilot",
                config=config,
                estimate_bytes=1,
            )
        saved = json.loads(manifest.read_text(encoding="utf-8"))["runs"][0]
        return result, records[record["run_id"]], saved

    def test_force_rerun_cannot_recover_unchanged_stale_complete_bundle(self):
        config = _config()
        config["runtime"]["model_load_strategy"] = "persistent_worker"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _worker_record(root, "stale")
            record["recovered_after_worker_exit"] = True
            _write_worker_bundle(record, config, marker="stale")
            manifest = root / "pilot" / "manifest.json"
            write_manifest(manifest, [record], "a" * 64, "pilot")
            process = mock.Mock()
            process.poll.return_value = 17
            with mock.patch(
                "scripts.run_nile_lowrank_study.subprocess.Popen",
                return_value=process,
            ):
                result = execute_plan(
                    [record],
                    manifest_path=manifest,
                    config_hash="a" * 64,
                    split="pilot",
                    config=config,
                    resume=True,
                    force_rerun=True,
                    dry_run=False,
                    max_runs=None,
                )
            saved = json.loads(manifest.read_text(encoding="utf-8"))["runs"][0]

        self.assertEqual(result["failed"], 1)
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["worker_artifact_freshness"]["refreshed"])
        self.assertIn("grid", saved["worker_artifact_freshness"]["stale_components"])
        self.assertFalse(saved.get("recovered_after_worker_exit", False))
        self.assertEqual(saved["worker_returncode"], 17)

    def test_no_terminal_event_recovers_only_fresh_complete_bundle(self):
        config = _config()
        config["runtime"]["model_load_strategy"] = "persistent_worker"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _worker_record(root, "fresh")
            process = mock.Mock()
            process.poll.return_value = 137

            def launch(*args, **kwargs):
                _write_worker_bundle(record, config, marker="fresh")
                return process

            result, observed, saved = self._execute(
                root, record, config, launch
            )

        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(observed["status"], "succeeded")
        self.assertTrue(observed["worker_artifact_freshness"]["refreshed"])
        self.assertTrue(observed["recovered_after_worker_exit"])
        self.assertEqual(observed["worker_returncode"], 137)
        self.assertEqual(saved["worker_returncode"], 137)

    def test_partial_directory_refresh_cannot_recover_stale_views_and_masks(self):
        config = _config()
        config["runtime"]["model_load_strategy"] = "persistent_worker"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _worker_record(root, "partial")
            _write_worker_bundle(record, config, marker="stale")
            process = mock.Mock()
            process.poll.return_value = 9

            def launch(*args, **kwargs):
                output = Path(record["output"])
                output.write_bytes(b"fresh-grid-content")
                output.with_name("grid_reference.png").write_bytes(
                    b"fresh-reference-content"
                )
                metadata_path = Path(record["metadata_path"])
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["marker"] = "fresh"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                next(Path(record["views_dir"]).glob("*.png")).write_bytes(
                    b"fresh-view-content"
                )
                next(Path(record["mask_dir"]).glob("*.png")).write_bytes(
                    b"fresh-mask-content"
                )
                return process

            result, observed, _ = self._execute(root, record, config, launch)

        self.assertEqual(result["failed"], 1)
        self.assertFalse(observed["worker_artifact_freshness"]["refreshed"])
        self.assertIn("views", observed["worker_artifact_freshness"]["stale_components"])
        self.assertIn("masks", observed["worker_artifact_freshness"]["stale_components"])
        self.assertTrue(observed["worker_artifact_freshness"]["unchanged_files"])

    def test_terminal_worker_record_persists_returncode(self):
        config = _config()
        config["runtime"]["model_load_strategy"] = "persistent_worker"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _worker_record(root, "terminal")
            process = mock.Mock()
            process.poll.return_value = 0

            def launch(*args, **kwargs):
                _write_worker_bundle(record, config, marker="terminal")
                events = root / "pilot" / "worker_events.jsonl"
                events.write_text(
                    json.dumps(
                        {
                            "sequence": 1,
                            "event": "run_succeeded",
                            "run_id": record["run_id"],
                            "attempt": 1,
                            "timestamp": "2026-07-11T00:00:00+00:00",
                            "duration_seconds": 1.0,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return process

            result, observed, saved = self._execute(
                root, record, config, launch
            )

        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(observed["status"], "succeeded")
        self.assertEqual(observed["worker_returncode"], 0)
        self.assertEqual(saved["worker_returncode"], 0)
        self.assertEqual(observed["worker_event_sequence"], 1)


class FormalTestReceiptTests(unittest.TestCase):
    @staticmethod
    def _valid_receipt():
        return {
            "schema_version": 1,
            "passed": True,
            "tests_complete": True,
            "compileall_returncode": 0,
            "pytest_returncode": 0,
            "finished_at": "2026-07-11T00:00:00+00:00",
            "command": [["python", "-m", "compileall"], ["python", "-m", "pytest"]],
        }

    def test_missing_failed_and_valid_test_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment" / "test_results.json"
            missing = audit_test_results_receipt(path)
            self.assertFalse(missing["verified"])
            self.assertIn("test_results_missing", missing["issues"])

            path.parent.mkdir(parents=True)
            failed_payload = self._valid_receipt()
            failed_payload["pytest_returncode"] = 1
            failed_payload["passed"] = False
            path.write_text(json.dumps(failed_payload), encoding="utf-8")
            failed = audit_test_results_receipt(path)
            self.assertFalse(failed["verified"])
            self.assertIn("pytest_failed_or_missing", failed["issues"])

            path.write_text(json.dumps(self._valid_receipt()), encoding="utf-8")
            valid = audit_test_results_receipt(path)
            self.assertTrue(valid["verified"], valid)

    def test_tests_not_verified_blocker_is_cleared_only_by_valid_receipt(self):
        config = _config()
        config["data"] = {"min_distinct_inputs": 25}
        input_validation = {"formal_ready": True}
        environment = {
            "python_cuda": {"cuda_available": True},
            "disk_free_bytes": 100 * (1024 ** 3),
        }
        missing_codes = {
            item["code"]
            for item in _formal_blockers(
                config,
                input_validation,
                environment,
                test_results_audit={
                    "verified": False,
                    "receipt_path": "missing",
                    "issues": ["test_results_missing"],
                },
            )
        }
        valid_codes = {
            item["code"]
            for item in _formal_blockers(
                config,
                input_validation,
                environment,
                test_results_audit={"verified": True, "issues": []},
            )
        }
        self.assertIn("tests_not_verified", missing_codes)
        self.assertNotIn("tests_not_verified", valid_codes)


class CheckpointProvenanceTests(unittest.TestCase):
    @staticmethod
    def _frozen_config(adapter_sha256):
        config = _config()
        config["data"] = {"min_distinct_inputs": 25}
        config["model"].update(
            {
                "base_model_revision": "a" * 40,
                "vae_model_revision": "b" * 40,
                "adapter_revision": "c" * 40,
                "birefnet_model": "birefnet",
                "birefnet_revision": "d" * 40,
                "adapter_sha256": adapter_sha256,
                "mv_adapter_checkpoint": "mvadapter_i2mv_sdxl.safetensors",
            }
        )
        config["evaluation"] = {
            "identity_model": "identity",
            "identity_model_revision": "e" * 40,
            "met3r_revision": "f" * 40,
        }
        return config

    @staticmethod
    def _manifest(config, checkpoint_path, sha256):
        model = config["model"]
        evaluation = config["evaluation"]
        return {
            "schema_version": 1,
            "resolved_revisions": {
                "base_model": {
                    "repo_id": model["base_model"],
                    "revision": model["base_model_revision"],
                },
                "vae_model": {
                    "repo_id": model["vae_model"],
                    "revision": model["vae_model_revision"],
                },
                "adapter_path": {
                    "repo_id": model["adapter_path"],
                    "revision": model["adapter_revision"],
                },
                "birefnet_model": {
                    "repo_id": model["birefnet_model"],
                    "revision": model["birefnet_revision"],
                },
                "identity_model": {
                    "repo_id": evaluation["identity_model"],
                    "revision": evaluation["identity_model_revision"],
                },
            },
            "adapter_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256,
            },
            "met3r_revision": evaluation["met3r_revision"],
        }

    def test_manifest_revisions_and_actual_adapter_hash_must_all_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "mvadapter_i2mv_sdxl.safetensors"
            checkpoint.write_bytes(b"verified adapter bytes")
            sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config = self._frozen_config(sha256)
            manifest_path = root / "checkpoint_manifest.json"
            manifest_path.write_text(
                json.dumps(self._manifest(config, checkpoint, sha256)),
                encoding="utf-8",
            )
            valid = audit_checkpoint_manifest(manifest_path, config)
            self.assertTrue(valid["verified"], valid)

            mismatched_revision = json.loads(manifest_path.read_text(encoding="utf-8"))
            mismatched_revision["resolved_revisions"]["adapter_path"][
                "revision"
            ] = "0" * 40
            manifest_path.write_text(
                json.dumps(mismatched_revision), encoding="utf-8"
            )
            revision_audit = audit_checkpoint_manifest(manifest_path, config)
            self.assertFalse(revision_audit["verified"])
            self.assertIn(
                "resolved_revision_mismatch:adapter_path",
                revision_audit["issues"],
            )

            manifest_path.write_text(
                json.dumps(self._manifest(config, checkpoint, sha256)),
                encoding="utf-8",
            )
            checkpoint.write_bytes(b"tampered adapter bytes")
            hash_audit = audit_checkpoint_manifest(manifest_path, config)
            self.assertFalse(hash_audit["verified"])
            self.assertIn(
                "adapter_sha256_file_manifest_mismatch", hash_audit["issues"]
            )

    def test_checkpoint_provenance_blocker_requires_verified_audit(self):
        config = self._frozen_config("a" * 64)
        input_validation = {"formal_ready": True}
        environment = {
            "python_cuda": {"cuda_available": True},
            "disk_free_bytes": 100 * (1024 ** 3),
        }
        common = {
            "config": config,
            "input_validation": input_validation,
            "environment": environment,
            "test_results_audit": {"verified": True, "issues": []},
        }
        missing_codes = {
            item["code"]
            for item in _formal_blockers(
                **common,
                checkpoint_audit={
                    "verified": False,
                    "manifest_path": "missing",
                    "issues": ["checkpoint_manifest_missing"],
                },
            )
        }
        valid_codes = {
            item["code"]
            for item in _formal_blockers(
                **common,
                checkpoint_audit={"verified": True, "issues": []},
            )
        }
        self.assertIn("checkpoint_provenance_not_verified", missing_codes)
        self.assertNotIn("checkpoint_provenance_not_verified", valid_codes)

    def test_verified_cache_rehashes_on_manifest_config_or_file_identity_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "mvadapter_i2mv_sdxl.safetensors"
            checkpoint.write_bytes(b"cacheable adapter bytes")
            sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            config = self._frozen_config(sha256)
            manifest = self._manifest(config, checkpoint, sha256)
            manifest_path = root / "checkpoint_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            cache_path = root / "environment" / "checkpoint_audit_cache.json"

            with mock.patch(
                "scripts.run_nile_lowrank_study._sha256_file",
                wraps=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            ) as hash_mock:
                first = audit_checkpoint_manifest(
                    manifest_path, config, cache_path=cache_path
                )
                second = audit_checkpoint_manifest(
                    manifest_path, config, cache_path=cache_path
                )
                self.assertTrue(first["verified"])
                self.assertFalse(first["cache_hit"])
                self.assertTrue(second["cache_hit"])
                self.assertEqual(hash_mock.call_count, 1)

                manifest["token_source"] = "changed-but-valid"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                changed_manifest = audit_checkpoint_manifest(
                    manifest_path, config, cache_path=cache_path
                )
                self.assertFalse(changed_manifest["cache_hit"])
                self.assertEqual(hash_mock.call_count, 2)

                changed_config = json.loads(json.dumps(config))
                changed_config["prompt"] = "new prompt changes config hash"
                changed_config_audit = audit_checkpoint_manifest(
                    manifest_path, changed_config, cache_path=cache_path
                )
                self.assertTrue(changed_config_audit["verified"])
                self.assertFalse(changed_config_audit["cache_hit"])
                self.assertEqual(hash_mock.call_count, 3)

                stat = checkpoint.stat()
                os.utime(
                    checkpoint,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000),
                )
                changed_file = audit_checkpoint_manifest(
                    manifest_path, changed_config, cache_path=cache_path
                )
                self.assertTrue(changed_file["verified"])
                self.assertFalse(changed_file["cache_hit"])
                self.assertEqual(hash_mock.call_count, 4)

                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                cache["verified"] = False
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                unverified_cache = audit_checkpoint_manifest(
                    manifest_path, changed_config, cache_path=cache_path
                )
                self.assertFalse(unverified_cache["cache_hit"])
                self.assertEqual(hash_mock.call_count, 5)


class InputAndGateTests(unittest.TestCase):
    def test_exact_and_rotation_duplicates_are_rejected_and_splits_disjoint(self):
        generator = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            duplicate = root / "duplicate.png"
            rotated = root / "rotated.png"
            distinct = root / "distinct.png"
            array = generator.integers(0, 256, size=(40, 64, 3), dtype=np.uint8)
            Image.fromarray(array).save(source)
            shutil.copyfile(source, duplicate)
            Image.fromarray(np.rot90(array)).save(rotated)
            Image.fromarray(
                generator.integers(0, 256, size=(40, 64, 3), dtype=np.uint8)
            ).save(distinct)
            unique, rejected = inspect_inputs(discover_images(root))
            split = stable_split(unique, pilot_count=1, full_count=1)
        self.assertEqual(len(unique), 2)
        self.assertEqual(
            {item["reason"] for item in rejected},
            {"duplicate_sha256", "perceptual_or_rotation_duplicate"},
        )
        pilot = {item.sha256 for item in split if item.split == "pilot"}
        full = {item.sha256 for item in split if item.split == "full"}
        self.assertTrue(pilot)
        self.assertTrue(full)
        self.assertFalse(pilot.intersection(full))

    def test_formal_gate_matrix_filters_three_unattainable_configs(self):
        config = _config()
        requested = build_pilot_configurations(config)
        records = []
        excluded_ids = {item["config_id"] for item in requested[2:5]}
        for item in requested:
            excluded = item["config_id"] in excluded_ids
            records.append(
                {
                    **item,
                    "passed": not excluded,
                    "eligible_for_generation": not excluded,
                    "exclusion_reason": "unattainable_target_kl" if excluded else None,
                    "achieved_kl": item.get("target_kl"),
                    "alpha": 0.2,
                    "basis_checksum": "basis",
                    "covariance_checksum": "cov",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "distribution_gates" / "configuration_gates.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "diagnostic_plots_complete": True,
                        "configurations": records,
                    }
                ),
                encoding="utf-8",
            )
            selected = gated_pilot_configurations(root, config)
        self.assertEqual(len(selected), 15)
        self.assertFalse(excluded_ids.intersection(item["config_id"] for item in selected))


class InputManifestLockTests(unittest.TestCase):
    @staticmethod
    def _input_config(directory):
        config = _config()
        config["data"] = {
            "input_dir": str(directory),
            "drive_input_dir": str(directory),
            "pilot_count": 1,
            "full_count": 1,
            "min_distinct_inputs": 2,
        }
        return config

    def test_add_delete_and_modify_are_rejected_without_overwriting_frozen_manifest(self):
        generator = np.random.default_rng(91)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            root = base / "artifacts"
            source.mkdir()
            first = source / "first.png"
            second = source / "second.png"
            Image.fromarray(
                generator.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
            ).save(first)
            Image.fromarray(
                generator.integers(0, 256, size=(35, 45, 3), dtype=np.uint8)
            ).save(second)
            second_bytes = second.read_bytes()
            config = self._input_config(source)
            initial = validate_inputs_stage(root, config, source)
            frozen_path = root / "inputs" / "input_validation.json"
            frozen_bytes = frozen_path.read_bytes()
            self.assertTrue(initial["formal_ready"])
            self.assertFalse(
                validate_inputs_stage(root, config, source).get(
                    "input_manifest_changed", False
                )
            )

            added = source / "added.png"
            Image.fromarray(
                generator.integers(0, 256, size=(31, 47, 3), dtype=np.uint8)
            ).save(added)
            added_audit = validate_inputs_stage(root, config, source)
            self.assertTrue(added_audit["input_manifest_changed"])
            self.assertEqual(frozen_path.read_bytes(), frozen_bytes)

            added.unlink()
            second.unlink()
            deleted_audit = validate_inputs_stage(root, config, source)
            self.assertTrue(deleted_audit["input_manifest_changed"])
            self.assertEqual(frozen_path.read_bytes(), frozen_bytes)

            second.write_bytes(second_bytes)
            Image.fromarray(
                generator.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
            ).save(first)
            modified_audit = validate_inputs_stage(root, config, source)
            self.assertTrue(modified_audit["input_manifest_changed"])
            self.assertEqual(frozen_path.read_bytes(), frozen_bytes)
            blockers = _formal_blockers(
                config,
                modified_audit,
                {
                    "python_cuda": {"cuda_available": True},
                    "disk_free_bytes": 100 * (1024 ** 3),
                },
                test_results_audit={"verified": True, "issues": []},
                checkpoint_audit={"verified": True, "issues": []},
            )
            self.assertIn(
                "input_manifest_changed", {item["code"] for item in blockers}
            )

    def test_manifest_overlap_blocks_full_and_valid_manifests_pass(self):
        config = _config()
        config["data"] = {
            "pilot_count": 1,
            "full_count": 1,
            "min_distinct_inputs": 2,
        }
        pilot_sha = "1" * 64
        full_sha = "2" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_path = root / "inputs" / "input_validation.json"
            frozen_path.parent.mkdir(parents=True)
            frozen_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "formal_ready": True,
                        "pilot_count": 1,
                        "full_count": 1,
                        "records": [
                            {"path": "pilot.png", "sha256": pilot_sha, "split": "pilot"},
                            {"path": "full.png", "sha256": full_sha, "split": "full"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            pilot_record = {"run_id": "pilot", "input_sha256": pilot_sha}
            write_manifest(
                root / "pilot" / "manifest.json",
                [pilot_record],
                "a" * 64,
                "pilot",
            )
            valid_full = audit_input_split_isolation(root, config, stage="full")
            self.assertTrue(valid_full["ready"], valid_full)

            write_manifest(
                root / "full" / "manifest.json",
                [{"run_id": "full", "input_sha256": full_sha}],
                "a" * 64,
                "full",
            )
            valid_trajectory = audit_input_split_isolation(
                root, config, stage="trajectory"
            )
            self.assertTrue(valid_trajectory["ready"], valid_trajectory)

            write_manifest(
                root / "pilot" / "manifest.json",
                [{"run_id": "leak", "input_sha256": full_sha}],
                "a" * 64,
                "pilot",
            )
            leaked = audit_input_split_isolation(root, config, stage="full")
            self.assertFalse(leaked["ready"])
            self.assertIn("pilot_manifest_contains_full_inputs", leaked["reasons"])


class CandidateAndFullGateTests(unittest.TestCase):
    def test_zero_score_selection_is_deterministic(self):
        rows = []
        for method in (
            "lowrank_camera_rbf",
            "lowrank_nested_tree_a",
            "lowrank_nested_tree_ab",
        ):
            rows.extend(
                [
                    _eligible_row(method, method + "-zero", 0.0, target_kl=0.0),
                    _eligible_row(method, method + "-higher", 0.1, target_kl=1.0),
                ]
            )
        first = select_candidates(rows, {})
        second = select_candidates(list(reversed(rows)), {})
        self.assertEqual(first["configuration_hash"], second["configuration_hash"])
        for selection in first["selections"].values():
            self.assertTrue(selection["configuration"]["config_id"].endswith("-zero"))
            self.assertEqual(selection["pilot_metrics"]["met3r_all_pair_mean"], 0.0)

    def test_full_readiness_rejects_missing_or_unverified_met3r(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics" / "pilot").mkdir(parents=True)
            (root / "pilot").mkdir(parents=True)
            (root / "metrics" / "pilot" / "lowrank_metrics.json").write_text(
                json.dumps(
                    {
                        "met3r_required": True,
                        "met3r_score_direction": "lower_is_better",
                        "formal_evaluation_complete": False,
                        "samples": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "pilot" / "manifest.json").write_text(
                json.dumps({"runs": []}), encoding="utf-8"
            )
            audit = audit_pilot_met3r_prerequisite(root, config)
        self.assertFalse(audit["ready"])
        self.assertIn("pilot_met3r_revision_unverified", audit["reasons"])
        self.assertIn("pilot_formal_evaluation_incomplete", audit["reasons"])

    def test_selected_candidate_proof_is_recomputed_from_current_artifacts(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, _ = _write_selection_proof(root, config)
            valid = audit_selected_candidates(root, config)
            self.assertTrue(valid["ready"], valid)

            selected["selections"]["camera_rbf"]["configuration"]["alpha"] = 0.9
            core = dict(selected)
            for field in (
                "configuration_hash",
                "candidate_configuration_hash",
                "study_config_hash",
                "pilot_metrics_sha256",
            ):
                core.pop(field, None)
            selected["candidate_configuration_hash"] = _hash(core)
            selected["configuration_hash"] = _hash(
                {
                    "candidate_configuration_hash": selected[
                        "candidate_configuration_hash"
                    ],
                    "study_config_hash": selected["study_config_hash"],
                    "pilot_metrics_sha256": selected[
                        "pilot_metrics_sha256"
                    ],
                }
            )
            for name in ("selected_candidates.json", "selected_candidates.yaml"):
                (root / "selected_candidates" / name).write_text(
                    json.dumps(selected), encoding="utf-8"
                )
            tampered = audit_selected_candidates(root, config)
        self.assertFalse(tampered["ready"])
        self.assertIn(
            "selected_candidate_not_current_policy_result",
            tampered["issues"],
        )
        self.assertIn(
            "selected_pilot_field_mismatch:camera_rbf:alpha",
            tampered["issues"],
        )

    def test_selected_candidate_audit_rejects_formal_rank_mismatch(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_selection_proof(root, config, rank_mismatch=True)
            audit = audit_selected_candidates(root, config)
        self.assertFalse(audit["ready"])
        self.assertIn("selected_equal_rank_kl_mismatch", audit["issues"])

    def test_diagnostic_only_candidate_does_not_block_integrity_audit(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, _ = _write_selection_proof(
                root, config, diagnostic_tree_ab=True
            )
            audit = audit_selected_candidates(root, config)
        self.assertEqual(
            selected["selections"]["nested_tree_ab"]["status"],
            "no_eligible_candidate",
        )
        self.assertTrue(audit["ready"], audit)

    def test_trajectory_plans_exact_rank_matched_pairs(self):
        selections = {}
        for topology, method, rank in (
            ("camera_rbf", "lowrank_camera_rbf", 8),
            ("nested_tree_a", "lowrank_nested_tree_a", 16),
            ("nested_tree_ab", "lowrank_nested_tree_ab", 8),
        ):
            selections[topology] = {
                "status": "selected",
                "diagnostic_only": False,
                "configuration": {
                    "config_id": topology,
                    "method": method,
                    "rank": rank,
                    "target_kl": 1.0,
                    "distribution_gate_passed": True,
                },
            }
        planned = build_trajectory_configurations(
            {"configuration_hash": "frozen", "selections": selections}
        )
        self.assertEqual(len(planned), 6)
        pair_ids = {item["trajectory_pair_id"] for item in planned}
        for pair_id in pair_ids:
            members = [item for item in planned if item["trajectory_pair_id"] == pair_id]
            self.assertEqual({item["trajectory_role"] for item in members}, {"iid_control", "correlated"})
            self.assertEqual(len({item["rank"] for item in members}), 1)

    def test_trajectory_summary_rejects_rank_mismatch_before_artifact_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "run_id": "iid",
                    "input_sha256": "object",
                    "seed": 0,
                    "trajectory_pair_id": "pair",
                    "trajectory_role": "iid_control",
                    "paired_method": "lowrank_camera_rbf",
                    "method": "iid_external",
                    "rank": 8,
                    "status": "succeeded",
                },
                {
                    "run_id": "corr",
                    "input_sha256": "object",
                    "seed": 0,
                    "trajectory_pair_id": "pair",
                    "trajectory_role": "correlated",
                    "method": "lowrank_camera_rbf",
                    "rank": 16,
                    "status": "succeeded",
                },
            ]
            write_manifest(
                root / "trajectory" / "manifest.json",
                records,
                "d" * 64,
                "trajectory",
            )
            summary = summarize_trajectory_stage(root, "d" * 64)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["failures"][0]["reason"], "observer_rank_mismatch")

    def test_trajectory_summary_writes_pair_heatmaps_and_aggregate_curves(self):
        config_hash = "e" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_valid_trajectory_pair(root, config_hash)
            summary = summarize_trajectory_stage(root, config_hash)

            self.assertTrue(summary["complete"])
            self.assertTrue(summary["visualization_complete"])
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(
                summary["visualization_audit"],
                {
                    "expected_pair_heatmap_count": 1,
                    "pair_heatmap_count": 1,
                    "aggregate_plot_complete": True,
                },
            )
            heatmap = Path(
                summary["pairs"][0]["visualizations"][
                    "view_correlation_heatmaps"
                ]
            )
            aggregate = Path(summary["artifacts"]["aggregate_g_delta_plot"])
            self.assertTrue(heatmap.is_file())
            self.assertGreater(heatmap.stat().st_size, 0)
            self.assertTrue(aggregate.is_file())
            self.assertGreater(aggregate.stat().st_size, 0)
            self.assertEqual(
                aggregate.parent.resolve(),
                (root / "plots" / "trajectory").resolve(),
            )
            self.assertEqual(
                summary["artifacts"]["pair_view_correlation_heatmaps"],
                [str(heatmap)],
            )
            saved = json.loads(
                (root / "trajectory" / "trajectory_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(saved["complete"])
            self.assertEqual(
                saved["artifacts"]["aggregate_g_delta_plot"], str(aggregate)
            )

    def test_trajectory_summary_is_incomplete_when_aggregate_plot_fails(self):
        config_hash = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_valid_trajectory_pair(root, config_hash)
            with mock.patch(
                "scripts.run_nile_lowrank_study._save_aggregate_trajectory_plot",
                side_effect=RuntimeError("plot failed"),
            ):
                summary = summarize_trajectory_stage(root, config_hash)

        self.assertFalse(summary["complete"])
        self.assertFalse(summary["visualization_complete"])
        self.assertIsNone(summary["artifacts"]["aggregate_g_delta_plot"])
        self.assertIn(
            "aggregate_trajectory_plot_failed",
            [item["reason"] for item in summary["failures"]],
        )


class EvaluationPairingTests(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        for object_index in range(3):
            common = {
                "experiment_id": "exp",
                "code_revision": "rev",
                "input_sha256": str(object_index),
                "input_image": "object-{}.png".format(object_index),
                "seed": 0,
                "status": "succeeded",
            }
            for method, config_id, score in (
                ("iid_external", "iid", 0.50),
                ("lowrank_camera_rbf", "rbf", 0.45),
                ("lowrank_nested_tree_a", "tree-a", 0.35),
                ("lowrank_nested_tree_ab", "tree-ab", 0.40),
            ):
                rows.append(
                    {
                        **common,
                        "method": method,
                        "config_id": config_id,
                        "angle_all_met3r_score": score,
                    }
                )
        return rows

    def test_iid_and_nested_vs_rbf_statistics_are_strictly_paired(self):
        rows = add_paired_deltas(self._rows())
        iid_stats = paired_statistics(rows, iterations=200, seed=4)
        rbf_pairs = add_metric_comparison_deltas(
            rows,
            baseline_method="lowrank_camera_rbf",
            target_methods=("lowrank_nested_tree_a", "lowrank_nested_tree_ab"),
            output_field="met3r_vs_rbf_delta",
        )
        rbf_stats = paired_statistics(
            rbf_pairs,
            delta_field="met3r_vs_rbf_delta",
            iterations=200,
            seed=4,
            comparison_baseline="lowrank_camera_rbf",
        )
        self.assertTrue(all(item["mean_delta"] < 0 for item in iid_stats))
        self.assertTrue(all(item["mean_delta"] < 0 for item in rbf_stats))
        self.assertTrue(all(item["object_cluster_count"] == 3 for item in rbf_stats))

    def test_ambiguous_rbf_baseline_is_not_silently_overwritten(self):
        rows = self._rows()
        duplicate = dict(next(item for item in rows if item["method"] == "lowrank_camera_rbf"))
        duplicate["config_id"] = "another-rbf"
        rows.append(duplicate)
        compared = add_metric_comparison_deltas(
            rows,
            baseline_method="lowrank_camera_rbf",
            target_methods=("lowrank_nested_tree_a",),
        )
        affected = [item for item in compared if item["input_sha256"] == "0"]
        self.assertEqual(affected[0]["comparison_pair_status"], "ambiguous")
        self.assertIsNone(affected[0]["comparison_delta"])

    def test_manifest_reconciliation_keeps_missing_runs_and_conflicts(self):
        manifest = [
            {"run_id": "one", "status": "succeeded", "method": "iid_external", "config_id": "iid"},
            {"run_id": "two", "status": "failed", "method": "lowrank_camera_rbf", "config_id": "rbf"},
        ]
        evaluated = [
            {"sample_id": "one", "status": "succeeded", "method": "wrong", "config_id": "iid"}
        ]
        rows, audit = reconcile_manifest_samples(manifest, evaluated)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["method"], "iid_external")
        self.assertIn("method", rows[0]["metadata_config_conflicts"])
        self.assertEqual(rows[1]["status"], "generation_failed")
        self.assertEqual(audit["missing_evaluation_run_ids"], ["two"])

    def test_met3r_revision_uses_pep610_commit(self):
        fake_distribution = mock.Mock()
        fake_distribution.version = "0.1"
        fake_distribution.metadata = {"Name": "met3r"}
        fake_distribution.read_text.return_value = json.dumps(
            {
                "url": "https://github.com/mohammadasim98/met3r",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
            }
        )
        with mock.patch(
            "scripts.eval_nile_lowrank_study.importlib_metadata.distribution",
            return_value=fake_distribution,
        ):
            verified = verify_installed_met3r_revision("abc123")
            self.assertTrue(verified["verified"])
            with self.assertRaises(RuntimeError):
                verify_installed_met3r_revision("different")

    def test_evaluation_plots_live_under_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = resolve_plots_directory(root / "metrics" / "full", None)
            self.assertEqual(observed, root / "plots" / "full")

    def test_real_metadata_grids_generate_paired_sheet_and_failure_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "experiment"
            output_dir = artifact_root / "metrics" / "full"
            contact_dir = resolve_contact_sheets_directory(output_dir, None)
            self.assertEqual(
                contact_dir, (artifact_root / "contact_sheets" / "full").resolve()
            )
            explicit = Path(directory) / "explicit-contact-sheets"
            self.assertEqual(
                resolve_contact_sheets_directory(output_dir, explicit),
                explicit.resolve(),
            )

            rows = []
            cases = (
                ("iid_external", "iid", "#d9eaf7", False, True),
                ("lowrank_camera_rbf", "rbf", "#e5f2d0", False, False),
                ("lowrank_nested_tree_a", "tree-a", "#f5d5d5", True, True),
            )
            for method, config_id, colour, is_failure, relative_output in cases:
                run_dir = artifact_root / "full" / method / "seed_000007"
                run_dir.mkdir(parents=True, exist_ok=True)
                grid_path = run_dir / "grid.png"
                Image.new("RGB", (96, 48), colour).save(grid_path)
                metadata_path = run_dir / "grid_metadata.json"
                metadata_path.write_text(
                    json.dumps(
                        {
                            "output": (
                                grid_path.name if relative_output else str(grid_path.resolve())
                            )
                        }
                    ),
                    encoding="utf-8",
                )
                row = {
                    "source": str(metadata_path),
                    "input_sha256": "same-object-sha256",
                    "input_image": "same-object.png",
                    "seed": 7,
                    "method": method,
                    "config_id": config_id,
                    "status": "succeeded",
                    "artifact_failure": is_failure,
                }
                rows.append(row)
                self.assertEqual(_row_grid_path(row), grid_path.resolve())

            result = generate_contact_sheets(rows, contact_dir)

            self.assertTrue(result["complete"], result)
            self.assertEqual(result["directory"], str(contact_dir))
            self.assertEqual(result["paired_sheet_count"], 1)
            self.assertEqual(result["failure_row_count"], 1)
            self.assertEqual(len(result["artifacts"]), 2)
            self.assertEqual(
                result["failure_gallery"], str(contact_dir / "failure_gallery.jpg")
            )
            paired_paths = [
                Path(item)
                for item in result["artifacts"]
                if Path(item).name.startswith("paired_")
            ]
            self.assertEqual(len(paired_paths), 1)
            for artifact in map(Path, result["artifacts"]):
                self.assertTrue(artifact.is_file())
                self.assertGreater(artifact.stat().st_size, 0)
                with Image.open(artifact) as image:
                    image.verify()


class ArtifactIntegrityTests(unittest.TestCase):
    def test_complete_bundle_requires_parseable_metadata_and_all_views_masks(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            Image.new("RGB", (4, 4), "black").save(input_path)
            input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
            output = root / "grid.png"
            reference = root / "grid_reference.png"
            views = []
            masks = []
            Image.new("RGB", (4, 4), "white").save(output)
            Image.new("RGB", (4, 4), "white").save(reference)
            for index in range(6):
                view = root / "view-{}.png".format(index)
                mask = root / "mask-{}.png".format(index)
                Image.new("RGB", (4, 4), "white").save(view)
                Image.new("L", (4, 4), 255).save(mask)
                views.append(str(view))
                masks.append(str(mask))
            metadata = root / "grid_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "config_id": "iid",
                        "output": str(output),
                        "reference_output": str(reference),
                        "view_files": views,
                        "mask_files": masks,
                        "azimuth_deg": config["model"]["views_deg"],
                        "seed": 0,
                        "input": {
                            "image": str(input_path),
                            "sha256": input_sha256,
                        },
                        "inference": {
                            "num_inference_steps": config["model"]["num_inference_steps"],
                            "guidance_scale": config["model"]["guidance_scale"],
                        },
                        "models": {
                            key: config["model"].get(key)
                            for key in (
                                "base_model_revision",
                                "vae_model_revision",
                                "unet_model_revision",
                                "lora_model_revision",
                                "adapter_revision",
                                "birefnet_revision",
                                "scheduler",
                                "mv_adapter_checkpoint",
                            )
                        },
                        "distribution": {"method": "iid_external"},
                    }
                ),
                encoding="utf-8",
            )
            record = {
                "output": str(output),
                "metadata_path": str(metadata),
                "config_id": "iid",
                "input_path": str(input_path),
                "input_sha256": input_sha256,
                "seed": 0,
                "camera_list": config["model"]["views_deg"],
                "method": "iid_external",
            }
            audit = audit_run_artifacts(record, config)
            swapped_payload = json.loads(metadata.read_text(encoding="utf-8"))
            swapped_payload["seed"] = 99
            metadata.write_text(
                json.dumps(swapped_payload), encoding="utf-8"
            )
            swapped = audit_run_artifacts(record, config)
        self.assertTrue(audit["complete"], audit)
        self.assertFalse(swapped["complete"])
        self.assertIn("seed_metadata_mismatch", swapped["issues"])


if __name__ == "__main__":
    unittest.main()
