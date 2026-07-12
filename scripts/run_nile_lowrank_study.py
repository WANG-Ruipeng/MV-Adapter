"""Run the resumable equal-KL low-rank MV-Adapter study.

Stages are explicit and resumable. A formal FULL result is never declared when
inputs, CUDA, frozen candidates, or required MEt3R metrics are missing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

try:
    import yaml
except ImportError:  # reported clearly by load_config
    yaml = None

from scripts.select_nile_lowrank_candidates import freeze_candidates, select_candidates
from scripts.validate_nile_inputs import resolve_input_directory, validate_input_directory
from scripts.run_nile_grid import _detect_code_revision


STAGES = ("preflight", "pilot", "select", "full", "trajectory", "evaluate", "report", "all")
LOWRANK_METHODS = (
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
FORMAL_PILOT_METHODS = (
    "iid_external",
    "shared_full",
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
FORMAL_FULL_METHODS = (
    "iid_external",
    "shared_full",
    "selected_camera_rbf",
    "selected_nested_tree_a",
    "selected_nested_tree_ab",
)
FORMAL_RUNTIME_POLICY = {
    "retry_failed": True,
    "max_retries": 1,
    "retry_oom_once": True,
    "clear_cuda_cache_on_oom": True,
    "atomic_manifest": True,
    "model_load_strategy": "persistent_worker",
}
FORMAL_SELECTION_TIE_BREAK_ORDER = [
    "target_kl",
    "rank",
    "rbf_stability",
    "rbf_length_scale_deg",
]
FORMAL_FIXED_POLICY_FIELDS = (
    ("full", "allow_sweep", False),
    ("trajectory", "enabled", True),
    ("selection", "require_distribution_gate", True),
    ("selection", "one_standard_error_tie_break", True),
    ("selection", "tie_break_order", FORMAL_SELECTION_TIE_BREAK_ORDER),
    ("evaluation", "identity_backend", "dinov2"),
    ("evaluation", "mask_backend", "birefnet"),
    ("evaluation", "save_pair_rows", True),
)
STATEMENT = (
    "NILE-inspired nested Gaussian element topology; strict NILE/SZ is not "
    "implemented in this study."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _slug(value: Any) -> str:
    text = "".join(character if str(character).isalnum() else "-" for character in str(value))
    return "-".join(part for part in text.lower().split("-") if part) or "item"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for the study runner; install requirements-colab.txt"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("study config must contain a YAML mapping")
    return payload


def validate_formal_protocol_config(config: Mapping[str, Any]) -> None:
    """Reject declarative overrides that the formal runner does not implement."""

    for section_name, expected in (
        ("pilot", FORMAL_PILOT_METHODS),
        ("full", FORMAL_FULL_METHODS),
    ):
        section = config.get(section_name, {})
        observed = section.get("methods") if isinstance(section, Mapping) else None
        if not isinstance(observed, list) or tuple(observed) != expected:
            raise ValueError(
                "{}.methods must exactly match the formal protocol: {}".format(
                    section_name, list(expected)
                )
            )

    model = config.get("model", {})
    if not isinstance(model, Mapping) or model.get("scheduler") is not None:
        raise ValueError(
            "model.scheduler must be null; scheduler overrides are not supported "
            "by the formal protocol"
        )

    for section_name, field, expected in FORMAL_FIXED_POLICY_FIELDS:
        section = config.get(section_name, {})
        observed = section.get(field) if isinstance(section, Mapping) else None
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                "{}.{} must be {!r} for the formal protocol; got {!r}".format(
                    section_name, field, expected, observed
                )
            )

    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be a mapping")
    supported_runtime_keys = set(FORMAL_RUNTIME_POLICY).union({"min_free_disk_gib"})
    unknown_runtime_keys = sorted(set(runtime).difference(supported_runtime_keys))
    if unknown_runtime_keys:
        raise ValueError(
            "unsupported runtime policy keys: {}".format(unknown_runtime_keys)
        )
    for field, expected in FORMAL_RUNTIME_POLICY.items():
        observed = runtime.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                "runtime.{} must be {!r} for the formal protocol; got {!r}".format(
                    field, expected, observed
                )
            )
    min_free_disk_gib = runtime.get("min_free_disk_gib")
    if (
        isinstance(min_free_disk_gib, bool)
        or not _finite_number(min_free_disk_gib)
        or float(min_free_disk_gib) <= 0
    ):
        raise ValueError("runtime.min_free_disk_gib must be a positive finite number")

    evaluation = config.get("evaluation", {})
    angle_bins = (
        evaluation.get("angle_bins_deg")
        if isinstance(evaluation, Mapping)
        else None
    )
    if (
        not isinstance(angle_bins, list)
        or not angle_bins
        or any(
            isinstance(value, bool)
            or not _finite_number(value)
            or not 0 < float(value) <= 180
            for value in angle_bins
        )
        or len({float(value) for value in angle_bins}) != len(angle_bins)
    ):
        raise ValueError(
            "evaluation.angle_bins_deg must contain unique finite values in (0, 180]"
        )


def resolve_config(
    config: Mapping[str, Any],
    *,
    input_dir: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("experiment", {})
    resolved.setdefault("data", {})
    if input_dir is not None:
        resolved["data"]["input_dir"] = str(input_dir.expanduser().resolve())
    if output_root is not None:
        resolved["experiment"]["output_root"] = str(output_root.expanduser().resolve())
    resolved["statement"] = {
        "text": STATEMENT,
        "strict_nile_sz_implemented": False,
    }
    resolved.setdefault("provenance", {})["code_revision"] = _detect_code_revision(
        Path(__file__).resolve().parents[1]
    )
    validate_formal_protocol_config(resolved)
    return resolved


def experiment_root(config: Mapping[str, Any]) -> Path:
    config_hash = _hash(config)
    section = config.get("experiment", {})
    base = Path(section.get("output_root", "outputs/nile_lowrank_kl_full")).expanduser()
    name = _slug(section.get("name", "nile_lowrank_kl_full"))
    return (base / "{}-{}".format(name, config_hash[:12])).resolve()


def lock_resolved_config(root: Path, config: Mapping[str, Any]) -> str:
    config_hash = _hash(config)
    lock_path = root / "configs" / "config_lock.json"
    resolved_path = root / "configs" / "resolved_config.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != config_hash:
            raise ValueError(
                "output root is locked to config hash {}; requested {}".format(
                    existing.get("config_hash"), config_hash
                )
            )
    else:
        _atomic_json(
            lock_path,
            {"schema_version": 1, "config_hash": config_hash, "locked_at": _utc_now()},
        )
    _atomic_json(resolved_path, dict(config))
    return config_hash


def detect_git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def detect_gpu() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        name, memory, driver = [part.strip() for part in output.splitlines()[0].split(",")]
        return {"available": True, "name": name, "memory_mib": int(memory), "driver": driver}
    except Exception as error:
        return {"available": False, "error": repr(error)}


def python_cuda_status() -> Dict[str, Any]:
    try:
        import torch

        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as error:
        return {"torch_available": False, "cuda_available": False, "error": repr(error)}


def capture_environment(root: Path, repo_root: Path) -> Dict[str, Any]:
    def capture(command: Sequence[str], *, cwd: Optional[Path] = None) -> str:
        try:
            process = subprocess.run(
                list(command),
                cwd=str(cwd) if cwd is not None else None,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return process.stdout + (
                "\nSTDERR\n" + process.stderr if process.stderr else ""
            )
        except Exception as error:
            return "capture failed: {!r}\n".format(error)

    environment_dir = root / "environment"
    git_commit = detect_git_revision(repo_root)
    captures = {
        "git_status_initial.txt": capture(
            ["git", "status", "--short", "--untracked-files=all"], cwd=repo_root
        ),
        "git_commit.txt": git_commit + "\n",
        "worktree.diff": capture(["git", "diff", "--binary"], cwd=repo_root),
        "pip_freeze.txt": capture([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi.txt": capture(["nvidia-smi"]),
    }
    for name, value in captures.items():
        _atomic_text(environment_dir / name, value)
    environment = {
        "captured_at": _utc_now(),
        "git_commit": git_commit,
        "code_revision": _detect_code_revision(repo_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "gpu": detect_gpu(),
        "python_cuda": python_cuda_status(),
        "disk_free_bytes": shutil.disk_usage(root.parent if root.parent.exists() else repo_root).free,
    }
    _atomic_json(environment_dir / "environment.json", environment)
    return environment


def build_pilot_configurations(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pilot = config["pilot"]
    configurations: List[Dict[str, Any]] = [
        {"method": "iid_external", "rank": None, "target_kl": 0.0, "rbf_length_scale_deg": None},
        {"method": "shared_full", "rank": None, "target_kl": None, "rbf_length_scale_deg": None},
    ]
    for rank in pilot["ranks"]:
        for target_kl in pilot["target_kls"]:
            for ell in pilot["rbf_length_scales_deg"]:
                configurations.append(
                    {
                        "method": "lowrank_camera_rbf",
                        "rank": int(rank),
                        "target_kl": float(target_kl),
                        "rbf_length_scale_deg": float(ell),
                    }
                )
            for method in ("lowrank_nested_tree_a", "lowrank_nested_tree_ab"):
                configurations.append(
                    {
                        "method": method,
                        "rank": int(rank),
                        "target_kl": float(target_kl),
                        "rbf_length_scale_deg": None,
                    }
                )
    for item in configurations:
        item["config_id"] = _hash(item)[:20]
    expected = int(pilot.get("expected_configs_per_input_seed", 18))
    if len(configurations) != expected:
        raise ValueError(
            "PILOT matrix has {} configurations, expected {}".format(
                len(configurations), expected
            )
        )
    return configurations


def build_full_configurations(selected: Mapping[str, Any]) -> List[Dict[str, Any]]:
    configurations = [
        {"method": "iid_external", "rank": None, "target_kl": 0.0, "rbf_length_scale_deg": None, "selection_status": "baseline", "distribution_gate_passed": True},
        {"method": "shared_full", "rank": None, "target_kl": None, "rbf_length_scale_deg": None, "selection_status": "diagnostic_upper_bound", "distribution_gate_passed": True},
    ]
    for topology in ("camera_rbf", "nested_tree_a", "nested_tree_ab"):
        selection = selected.get("selections", {}).get(topology, {})
        configuration = selection.get("configuration")
        if configuration and configuration.get("distribution_gate_passed", False):
            item = dict(configuration)
            item["selection_status"] = selection.get("status")
            item["diagnostic_only"] = bool(selection.get("diagnostic_only", False))
            configurations.append(item)
    for item in configurations:
        item.setdefault("config_id", _hash(item)[:20])
    return configurations


def build_trajectory_configurations(selected: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build strict IID/correlated pairs with an identical observer basis rank."""

    configurations: List[Dict[str, Any]] = []
    for topology in ("camera_rbf", "nested_tree_a", "nested_tree_ab"):
        selection = selected.get("selections", {}).get(topology, {})
        configuration = selection.get("configuration")
        if not configuration or configuration.get("rank") is None:
            continue
        correlated = dict(configuration)
        if not correlated.get("distribution_gate_passed", False):
            continue
        pair_id = _hash(
            {
                "topology": topology,
                "configuration": correlated,
                "selection_hash": selected.get("configuration_hash"),
            }
        )[:20]
        correlated.update(
            {
                "trajectory_pair_id": pair_id,
                "trajectory_role": "correlated",
                "selection_status": selection.get("status"),
                "diagnostic_only": bool(selection.get("diagnostic_only", False)),
            }
        )
        control_core = {
            "method": "iid_external",
            "rank": int(correlated["rank"]),
            "target_kl": 0.0,
            "rbf_length_scale_deg": None,
            "trajectory_pair_id": pair_id,
            "trajectory_role": "iid_control",
            "paired_method": correlated.get("method"),
            "distribution_gate_passed": True,
        }
        control = dict(control_core)
        control["config_id"] = _hash(control_core)[:20]
        configurations.extend((control, correlated))
    return configurations


def _read_input_records(path: Path, split: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload.get("records", []) if item.get("split") == split]


def build_run_plan(
    *,
    split: str,
    inputs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    configurations: Sequence[Mapping[str, Any]],
    config_hash: str,
    root: Path,
    config: Mapping[str, Any],
    git_commit: str,
    gpu: Mapping[str, Any],
    code_revision: Optional[str] = None,
) -> List[Dict[str, Any]]:
    model = config["model"]
    records = []
    for input_record in inputs:
        for seed in seeds:
            for method_config in configurations:
                identity = {
                    "config_hash": config_hash,
                    "code_revision": code_revision or git_commit,
                    "split": split,
                    "input_sha256": input_record["sha256"],
                    "seed": int(seed),
                    **dict(method_config),
                    "views_deg": list(model["views_deg"]),
                    "steps": int(model["num_inference_steps"]),
                    "scheduler": model.get("scheduler"),
                    "model_revisions": {
                        key: model.get(key)
                        for key in (
                            "base_model_revision",
                            "vae_model_revision",
                            "adapter_revision",
                            "birefnet_revision",
                        )
                    },
                }
                run_id = _hash(identity)[:24]
                output_dir = (
                    root
                    / split
                    / "{}-{}".format(_slug(Path(input_record["path"]).stem), input_record["sha256"][:8])
                    / _slug(method_config["method"])
                    / str(method_config["config_id"])
                    / "seed_{:06d}".format(int(seed))
                )
                output = output_dir / "grid.png"
                records.append(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "experiment_id": root.name,
                        "config_hash": config_hash,
                        "split": split,
                        "input_path": input_record["path"],
                        "input_image": input_record["path"],
                        "input_sha256": input_record["sha256"],
                        "seed": int(seed),
                        **dict(method_config),
                        "camera_list": list(model["views_deg"]),
                        "steps": int(model["num_inference_steps"]),
                        "scheduler": model.get("scheduler"),
                        "status": "planned",
                        "git_commit": git_commit,
                        "code_revision": code_revision or git_commit,
                        "gpu": dict(gpu),
                        "model_checkpoint": model.get("mv_adapter_checkpoint"),
                        "output": str(output),
                        "metadata_path": str(output.with_name("grid_metadata.json")),
                        "views_dir": str(output.with_name("grid_views")),
                        "mask_dir": str(output.with_name("grid_masks")),
                        "retry_count": 0,
                    }
                )
    return records


def _manifest_payload(records: Sequence[Mapping[str, Any]], config_hash: str, split: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "config_hash": config_hash,
        "split": split,
        "runs": list(records),
    }


def read_manifest(path: Path, config_hash: str, split: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_hash") != config_hash or payload.get("split") != split:
        raise ValueError("manifest belongs to another config hash or split")
    return [dict(item) for item in payload.get("runs", [])]


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(item) for item in payload.get("runs", [])]


def _artifact_path(value: Any, base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def audit_run_artifacts(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate the complete generated bundle, not merely the grid file."""

    issues: List[str] = []
    output = Path(str(record.get("output", "")))
    metadata_path = Path(str(record.get("metadata_path", "")))
    expected_views = len(config.get("model", {}).get("views_deg", []))
    if not output.is_file() or output.stat().st_size <= 0:
        issues.append("grid_missing_or_empty")
    metadata: Dict[str, Any] = {}
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        issues.append("metadata_missing_or_empty")
    else:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("metadata root is not an object")
            metadata = loaded
        except Exception as error:
            issues.append("metadata_invalid:{!r}".format(error))
    base = metadata_path.parent
    model = config.get("model", {})
    metadata_output = _artifact_path(metadata.get("output"), base)
    if metadata_output is None or metadata_output.resolve() != output.resolve():
        issues.append("output_metadata_mismatch")
    if metadata.get("config_id") != record.get("config_id"):
        issues.append("config_id_metadata_mismatch")
    try:
        if int(metadata.get("seed")) != int(record.get("seed")):
            issues.append("seed_metadata_mismatch")
    except (TypeError, ValueError):
        issues.append("seed_metadata_mismatch")
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, Mapping):
        input_metadata = {}
        issues.append("input_metadata_missing")
    expected_input = Path(str(record.get("input_path", ""))).expanduser()
    observed_input = _artifact_path(input_metadata.get("image"), base)
    if (
        observed_input is None
        or observed_input.resolve() != expected_input.resolve()
    ):
        issues.append("input_path_metadata_mismatch")
    expected_input_sha = str(record.get("input_sha256") or "").lower()
    if expected_input_sha:
        if str(input_metadata.get("sha256") or "").lower() != expected_input_sha:
            issues.append("input_sha256_metadata_mismatch")
        if not expected_input.is_file():
            issues.append("input_file_missing_during_audit")
        else:
            try:
                if _sha256_file(expected_input).lower() != expected_input_sha:
                    issues.append("input_sha256_file_mismatch")
            except Exception as error:
                issues.append("input_sha256_audit_failed:{!r}".format(error))
    inference = metadata.get("inference")
    if not isinstance(inference, Mapping):
        inference = {}
        issues.append("inference_metadata_missing")
    if inference.get("num_inference_steps") != int(
        model.get("num_inference_steps", 30)
    ):
        issues.append("steps_metadata_mismatch")
    if not _selection_value_matches(
        inference.get("guidance_scale"), model.get("guidance_scale", 3.0)
    ):
        issues.append("guidance_scale_metadata_mismatch")
    models_metadata = metadata.get("models")
    if not isinstance(models_metadata, Mapping):
        models_metadata = {}
        issues.append("models_metadata_missing")
    if models_metadata.get("scheduler") != model.get("scheduler"):
        issues.append("scheduler_metadata_mismatch")
    for field in (
        "base_model_revision",
        "vae_model_revision",
        "unet_model_revision",
        "lora_model_revision",
        "adapter_revision",
        "birefnet_revision",
    ):
        if models_metadata.get(field) != model.get(field):
            issues.append("{}_metadata_mismatch".format(field))
    if models_metadata.get("mv_adapter_checkpoint") != model.get(
        "mv_adapter_checkpoint"
    ):
        issues.append("mv_adapter_checkpoint_metadata_mismatch")
    view_paths = [
        _artifact_path(item, base) for item in metadata.get("view_files", [])
    ]
    mask_paths = [
        _artifact_path(item, base) for item in metadata.get("mask_files", [])
    ]
    reference = _artifact_path(metadata.get("reference_output"), base)
    if expected_views <= 0 or len(view_paths) != expected_views:
        issues.append("view_count_mismatch")
    if expected_views <= 0 or len(mask_paths) != expected_views:
        issues.append("mask_count_mismatch")
    if any(path is None or not path.is_file() or path.stat().st_size <= 0 for path in view_paths):
        issues.append("view_artifact_missing_or_empty")
    if any(path is None or not path.is_file() or path.stat().st_size <= 0 for path in mask_paths):
        issues.append("mask_artifact_missing_or_empty")
    if reference is None or not reference.is_file() or reference.stat().st_size <= 0:
        issues.append("reference_missing_or_empty")
    metadata_angles = metadata.get("azimuth_deg")
    try:
        angles_match = [float(item) for item in metadata_angles] == [
            float(item) for item in record.get("camera_list", [])
        ]
    except (TypeError, ValueError):
        angles_match = False
    if not angles_match:
        issues.append("camera_list_mismatch")
    distribution = metadata.get("distribution")
    if not isinstance(distribution, Mapping):
        issues.append("distribution_metadata_missing")
        distribution = {}
    if distribution.get("method") != record.get("method"):
        issues.append("method_metadata_mismatch")
    if record.get("method") in LOWRANK_METHODS:
        try:
            rank_matches = int(distribution.get("basis_rank")) == int(record.get("rank"))
        except (TypeError, ValueError):
            rank_matches = False
        if not rank_matches:
            issues.append("rank_metadata_mismatch")
        if not _finite_number(distribution.get("achieved_kl")):
            issues.append("achieved_kl_missing")
        if not _finite_number(distribution.get("alpha")):
            issues.append("alpha_missing")
        if not _selection_value_matches(
            distribution.get("target_joint_kl"), record.get("target_kl")
        ):
            issues.append("target_kl_metadata_mismatch")
        if not distribution.get("basis_checksum"):
            issues.append("basis_checksum_missing")
        if not distribution.get("covariance_checksum"):
            issues.append("covariance_checksum_missing")
    trajectory_path = _artifact_path(record.get("trajectory_output"), output.parent)
    if record.get("trajectory_output"):
        if trajectory_path is None or not trajectory_path.is_file():
            issues.append("trajectory_npz_missing")
        else:
            try:
                with np.load(trajectory_path, allow_pickle=False) as archive:
                    required = {"basis_coefficients", "milestones", "g_t", "basis_rank"}
                    if required.difference(archive.files):
                        issues.append("trajectory_npz_incomplete")
                    elif int(archive["basis_rank"].item()) != int(record.get("rank")):
                        issues.append("trajectory_rank_mismatch")
            except Exception as error:
                issues.append("trajectory_npz_invalid:{!r}".format(error))
    return {
        "complete": not issues,
        "issues": sorted(set(issues)),
        "expected_view_count": expected_views,
        "view_count": len(view_paths),
        "mask_count": len(mask_paths),
        "metadata_path": str(metadata_path),
    }


def write_manifest(path: Path, records: Sequence[Mapping[str, Any]], config_hash: str, split: str) -> None:
    _atomic_json(path, _manifest_payload(records, config_hash, split))


def _add_optional(command: List[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_inference_command(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> List[str]:
    model = config["model"]
    if int(model.get("height", 768)) != 768 or int(model.get("width", 768)) != 768:
        raise ValueError("current MV-Adapter inference entry point requires 768x768")
    scheduler = model.get("scheduler")
    if scheduler not in (None, "ddpm", "lcm"):
        raise ValueError(
            "model.scheduler must be one of null, ddpm, or lcm; got {!r}".format(
                scheduler
            )
        )
    checkpoint = model.get("mv_adapter_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("model.mv_adapter_checkpoint must be a non-empty filename")
    command = [
        sys.executable,
        "-m",
        "scripts.inference_i2mv_sdxl_nile",
        "--image",
        str(record["input_path"]),
        "--text",
        str(config.get("prompt", "high quality, detailed object")),
        "--seed",
        str(record["seed"]),
        "--method",
        str(record["method"]),
        "--num_inference_steps",
        str(model["num_inference_steps"]),
        "--guidance_scale",
        str(model["guidance_scale"]),
        "--azimuth_deg",
        *[str(item) for item in model["views_deg"]],
        "--base_model",
        str(model["base_model"]),
        "--vae_model",
        str(model["vae_model"]),
        "--adapter_path",
        str(model["adapter_path"]),
        "--mv_adapter_checkpoint",
        checkpoint,
        "--birefnet_model",
        str(model.get("birefnet_model", "ZhengPeng7/BiRefNet")),
        "--device",
        str(model.get("device", "cuda")),
        "--output",
        str(record["output"]),
        "--views_dir",
        str(record["views_dir"]),
        "--mask_dir",
        str(record["mask_dir"]),
        "--save_views",
        "--save_masks",
        "--config_id",
        str(record["config_id"]),
        "--input_sha256",
        str(record["input_sha256"]),
    ]
    for key, flag in (
        ("base_model_revision", "--base_model_revision"),
        ("vae_model_revision", "--vae_model_revision"),
        ("adapter_revision", "--adapter_revision"),
        ("birefnet_revision", "--birefnet_revision"),
        ("scheduler", "--scheduler"),
    ):
        _add_optional(command, flag, model.get(key))
    if model.get("remove_background", True):
        command.append("--remove_bg")
    if record["method"] in LOWRANK_METHODS:
        command.extend(
            [
                "--basis_rank",
                str(record["rank"]),
                "--target_joint_kl",
                str(record["target_kl"]),
            ]
        )
        _add_optional(
            command,
            "--rbf_length_scale_deg",
            record.get("rbf_length_scale_deg"),
        )
    trajectory_output = record.get("trajectory_output")
    if trajectory_output:
        if record["method"] not in LOWRANK_METHODS and record.get("rank") is not None:
            command.extend(["--basis_rank", str(record["rank"])])
        command.extend(["--trajectory_output", str(trajectory_output)])
        milestones = config.get("trajectory", {}).get("milestones")
        if milestones is not None:
            command.extend(
                ["--trajectory_milestones", *[str(value) for value in milestones]]
            )
    return command


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_worker_events(path: Path, after_sequence: int = 0) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            sequence = int(event.get("sequence", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if sequence > after_sequence:
            events.append(dict(event))
    return sorted(events, key=lambda item: int(item.get("sequence", 0)))


def _last_worker_sequence(path: Path) -> int:
    events = _read_worker_events(path)
    return max((int(item.get("sequence", 0)) for item in events), default=0)


def _worker_file_signature(path: Path) -> Optional[Dict[str, int]]:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _worker_directory_signature(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_dir():
        return None
    stat = path.stat()
    files = []
    for child in sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    ):
        files.append(
            {
                "path": child.relative_to(path).as_posix(),
                "signature": _worker_file_signature(child),
            }
        )
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "files": files,
    }


def _worker_artifact_snapshot(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Snapshot every worker-owned bundle component before one invocation."""

    output = Path(str(record.get("output", ""))).expanduser()
    metadata = Path(str(record.get("metadata_path", ""))).expanduser()
    reference = output.with_name(output.stem + "_reference" + output.suffix)
    snapshot: Dict[str, Any] = {
        "grid": _worker_file_signature(output),
        "metadata": _worker_file_signature(metadata),
        "reference": _worker_file_signature(reference),
        "views": _worker_directory_signature(
            Path(str(record.get("views_dir", ""))).expanduser()
        ),
        "masks": _worker_directory_signature(
            Path(str(record.get("mask_dir", ""))).expanduser()
        ),
    }
    if record.get("trajectory_output"):
        snapshot["trajectory"] = _worker_file_signature(
            Path(str(record["trajectory_output"])).expanduser()
        )
    return snapshot


def _worker_artifact_freshness(
    record: Mapping[str, Any], baseline: Mapping[str, Any]
) -> Dict[str, Any]:
    """Require every required bundle component to change this invocation."""

    current = _worker_artifact_snapshot(record)
    required = ["grid", "metadata", "reference", "views", "masks"]
    if record.get("trajectory_output"):
        required.append("trajectory")
    refreshed = []
    stale = []
    missing = []
    unchanged_files: Dict[str, List[str]] = {}
    for name in required:
        current_component = current.get(name)
        baseline_component = baseline.get(name)
        if current_component is None:
            missing.append(name)
            continue
        if name not in {"views", "masks"}:
            if current_component != baseline_component:
                refreshed.append(name)
            else:
                stale.append(name)
            continue
        if baseline_component is None:
            refreshed.append(name)
            continue
        baseline_files = {
            item["path"]: item.get("signature")
            for item in baseline_component.get("files", [])
        }
        current_files = {
            item["path"]: item.get("signature")
            for item in current_component.get("files", [])
        }
        unchanged = sorted(
            path
            for path, signature in current_files.items()
            if baseline_files.get(path) == signature
        )
        if unchanged:
            stale.append(name)
            unchanged_files[name] = unchanged
        elif current_component != baseline_component:
            refreshed.append(name)
        else:
            stale.append(name)
    return {
        "refreshed": not stale and not missing,
        "required_components": required,
        "refreshed_components": refreshed,
        "stale_components": stale,
        "missing_components": missing,
        "unchanged_files": unchanged_files,
    }


def _copy_distribution_metadata(record: Dict[str, Any]) -> None:
    metadata_path = Path(str(record.get("metadata_path", "")))
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return
    distribution = metadata.get("distribution", {})
    if not isinstance(distribution, Mapping):
        return
    for field in (
        "achieved_kl",
        "alpha",
        "basis_checksum",
        "covariance_checksum",
    ):
        if field in distribution:
            record[field] = distribution[field]


def _merge_worker_events(
    *,
    events_path: Path,
    after_sequence: int,
    records: MutableMapping[str, Dict[str, Any]],
    order: Sequence[str],
    manifest_path: Path,
    config_hash: str,
    split: str,
    config: Mapping[str, Any],
) -> tuple:
    """Merge terminal worker events and atomically checkpoint the manifest."""

    latest = after_sequence
    terminal_ids = set()
    for event in _read_worker_events(events_path, after_sequence):
        latest = max(latest, int(event.get("sequence", 0)))
        if event.get("event") not in {"run_succeeded", "run_failed"}:
            continue
        run_id = str(event.get("run_id", ""))
        record = records.get(run_id)
        if record is None:
            continue
        terminal_ids.add(run_id)
        attempt = int(event.get("attempt", 1) or 1)
        record.update(
            {
                "finished_at": event.get("timestamp", _utc_now()),
                "duration_seconds": event.get("duration_seconds"),
                "retry_count": max(0, attempt - 1),
                "worker_event_sequence": event.get("sequence"),
                "worker_events_path": str(events_path),
            }
        )
        if event.get("event") == "run_succeeded":
            integrity = audit_run_artifacts(record, config)
            record["artifact_integrity"] = integrity
            if integrity.get("complete"):
                record["status"] = "succeeded"
                record["error"] = None
                _copy_distribution_metadata(record)
            else:
                record["status"] = "failed"
                record["error"] = "artifact integrity failed: {}".format(
                    integrity.get("issues", [])
                )
        else:
            record["status"] = "failed"
            record["artifact_integrity"] = event.get("artifact_integrity") or {
                "complete": False,
                "issues": ["worker_run_failed"],
            }
            record["error"] = event.get("error")
            record["traceback"] = str(event.get("traceback", ""))[-12000:]
            record["oom"] = bool(event.get("oom", False))
        write_manifest(
            manifest_path,
            [records[item] for item in order],
            config_hash,
            split,
        )
    return latest, terminal_ids


def _latest_worker_fatal_event(
    events_path: Path, after_sequence: int
) -> Optional[Dict[str, Any]]:
    fatal = None
    for event in _read_worker_events(events_path, after_sequence):
        if event.get("event") in {"plan_rejected", "model_load_failed"}:
            fatal = dict(event)
    return fatal


def _execute_persistent_worker(
    *,
    executable: Sequence[Dict[str, Any]],
    plan: Sequence[Mapping[str, Any]],
    records: MutableMapping[str, Dict[str, Any]],
    order: Sequence[str],
    manifest_path: Path,
    config_hash: str,
    split: str,
    config: Mapping[str, Any],
    estimate_bytes: int,
) -> Dict[str, Any]:
    worker_plan_path = manifest_path.with_name("worker_plan.json")
    events_path = manifest_path.with_name("worker_events.jsonl")
    log_path = manifest_path.with_name("worker.log")
    start_sequence = _last_worker_sequence(events_path)
    artifact_baselines = {
        str(record["run_id"]): _worker_artifact_snapshot(record)
        for record in executable
    }
    _atomic_json(
        worker_plan_path,
        {
            "schema_version": 1,
            "config_hash": config_hash,
            "split": split,
            "resolved_config": dict(config),
            "records": [dict(item) for item in executable],
        },
    )
    command = [
        sys.executable,
        "-m",
        "scripts.nile_lowrank_inference_worker",
        "--plan",
        str(worker_plan_path),
        "--events",
        str(events_path),
    ]
    for record in executable:
        for stale_field in (
            "finished_at",
            "duration_seconds",
            "error",
            "traceback",
            "oom",
            "artifact_integrity",
            "worker_artifact_freshness",
            "worker_event_sequence",
            "worker_returncode",
            "recovered_after_worker_exit",
        ):
            record.pop(stale_field, None)
        record.update(
            {
                "status": "running",
                "started_at": _utc_now(),
                "worker_command": command,
                "worker_plan_path": str(worker_plan_path),
                "worker_events_path": str(events_path),
                "log_path": str(log_path),
            }
        )
    write_manifest(
        manifest_path, [records[item] for item in order], config_hash, split
    )

    terminal_ids = set()
    worker_returncode: Optional[int] = None
    launch_error = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write("\nWORKER COMMAND {}\n".format(json.dumps(command)))
            log_handle.flush()
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            sequence = start_sequence
            while True:
                sequence, current_terminal = _merge_worker_events(
                    events_path=events_path,
                    after_sequence=sequence,
                    records=records,
                    order=order,
                    manifest_path=manifest_path,
                    config_hash=config_hash,
                    split=split,
                    config=config,
                )
                terminal_ids.update(current_terminal)
                worker_returncode = process.poll()
                if worker_returncode is not None:
                    break
                time.sleep(0.5)
            sequence, current_terminal = _merge_worker_events(
                events_path=events_path,
                after_sequence=sequence,
                records=records,
                order=order,
                manifest_path=manifest_path,
                config_hash=config_hash,
                split=split,
                config=config,
            )
            terminal_ids.update(current_terminal)
    except Exception as error:
        launch_error = repr(error)

    fatal_event = _latest_worker_fatal_event(events_path, start_sequence)
    try:
        worker_log_tail = log_path.read_text(encoding="utf-8")[-12000:]
    except Exception:
        worker_log_tail = ""

    for record in executable:
        run_id = str(record["run_id"])
        # Terminal events are merged before the subprocess return code is
        # known. Persist it for every record after the worker exits.
        record["worker_returncode"] = worker_returncode
        if run_id in terminal_ids:
            continue
        integrity = audit_run_artifacts(record, config)
        freshness = _worker_artifact_freshness(
            record, artifact_baselines[run_id]
        )
        record["artifact_integrity"] = integrity
        record["worker_artifact_freshness"] = freshness
        record["finished_at"] = _utc_now()
        if integrity.get("complete") and freshness.get("refreshed"):
            record["status"] = "succeeded"
            record["recovered_after_worker_exit"] = True
            record["error"] = None
            _copy_distribution_metadata(record)
        else:
            record["status"] = "failed"
            if fatal_event is not None:
                failure_kind = str(fatal_event.get("event"))
                failure_error = str(fatal_event.get("error") or "unknown worker error")
                record["error"] = "{}: {}".format(failure_kind, failure_error)
                record["traceback"] = str(fatal_event.get("traceback", ""))[-12000:]
                record["worker_failure_event"] = fatal_event
                record["oom"] = "out of memory" in failure_error.lower()
                record["artifact_integrity"] = {
                    "complete": False,
                    "issues": ["worker_{}".format(failure_kind)],
                }
            elif integrity.get("complete") and not freshness.get("refreshed"):
                record["error"] = (
                    "worker exited without terminal event and the complete bundle "
                    "was not fully refreshed by this invocation; returncode={}, "
                    "stale_components={}".format(
                        worker_returncode, freshness.get("stale_components", [])
                    )
                )
            else:
                record["error"] = launch_error or (
                    "worker exited without terminal event; returncode={}, "
                    "issues={}, freshness={}".format(
                        worker_returncode,
                        integrity.get("issues", []),
                        freshness,
                    )
                )
        write_manifest(
            manifest_path,
            [records[item] for item in order],
            config_hash,
            split,
        )

    final_records = [records[str(item["run_id"])] for item in plan]
    for item in final_records:
        if item.get("status") == "succeeded":
            item["artifact_integrity"] = audit_run_artifacts(item, config)
    write_manifest(
        manifest_path, [records[item] for item in order], config_hash, split
    )
    return {
        "planned": len(plan),
        "executed": len(executable),
        "estimated_output_bytes": estimate_bytes,
        "worker_strategy": "persistent_worker",
        "worker_returncode": worker_returncode,
        "worker_plan_path": str(worker_plan_path),
        "worker_events_path": str(events_path),
        "worker_log_path": str(log_path),
        "worker_log_tail": worker_log_tail,
        "worker_fatal_event": fatal_event,
        "succeeded": sum(item.get("status") == "succeeded" for item in final_records),
        "artifact_complete": sum(
            item.get("status") == "succeeded"
            and bool(item.get("artifact_integrity", {}).get("complete"))
            for item in final_records
        ),
        "artifact_incomplete": sum(
            item.get("status") == "succeeded"
            and not bool(item.get("artifact_integrity", {}).get("complete"))
            for item in final_records
        ),
        "failed": sum(item.get("status") == "failed" for item in final_records),
        "newly_succeeded": sum(
            str(item["run_id"]) in {str(row["run_id"]) for row in executable}
            and item.get("status") == "succeeded"
            for item in final_records
        ),
        "newly_failed": sum(
            str(item["run_id"]) in {str(row["run_id"]) for row in executable}
            and item.get("status") == "failed"
            for item in final_records
        ),
    }


def execute_plan(
    plan: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
    config_hash: str,
    split: str,
    config: Mapping[str, Any],
    resume: bool,
    force_rerun: bool,
    dry_run: bool,
    max_runs: Optional[int],
) -> Dict[str, Any]:
    previous = read_manifest(manifest_path, config_hash, split)
    if previous:
        previous_ids = {str(item["run_id"]) for item in previous}
        proposed_ids = {str(item["run_id"]) for item in plan}
        if previous_ids != proposed_ids:
            raise ValueError(
                "resume plan differs from frozen manifest: removed={}, added={}".format(
                    sorted(previous_ids.difference(proposed_ids)),
                    sorted(proposed_ids.difference(previous_ids)),
                )
            )
    records: MutableMapping[str, Dict[str, Any]] = {
        str(item["run_id"]): dict(item) for item in previous
    }
    order = [str(item["run_id"]) for item in previous]
    for proposed in plan:
        run_id = str(proposed["run_id"])
        if run_id not in records:
            records[run_id] = dict(proposed)
            order.append(run_id)

    executable = []
    for proposed in plan:
        current = records[str(proposed["run_id"])]
        if current.get("status") == "succeeded" and resume and not force_rerun:
            integrity = audit_run_artifacts(current, config)
            current["artifact_integrity"] = integrity
            if integrity.get("complete"):
                continue
            current["status"] = "failed"
            current["error"] = "resume integrity audit failed: {}".format(
                integrity.get("issues", [])
            )
        executable.append(current)
    if max_runs is not None:
        executable = executable[:max_runs]

    estimate_bytes = len(plan) * 25 * 1024 * 1024
    if dry_run:
        preview_commands = [
            {
                "run_id": str(record["run_id"]),
                "command": build_inference_command(record, config),
            }
            for record in executable
        ]
        final_records = [records[str(item["run_id"])] for item in plan]
        return {
            "dry_run": True,
            "planned": len(plan),
            "would_execute": len(executable),
            "estimated_output_bytes": estimate_bytes,
            "preview_commands": preview_commands,
            "succeeded": sum(item.get("status") == "succeeded" for item in final_records),
            "failed": sum(item.get("status") == "failed" for item in final_records),
        }

    write_manifest(
        manifest_path, [records[item] for item in order], config_hash, split
    )
    runtime = config.get("runtime", {})
    if executable and runtime.get("model_load_strategy") == "persistent_worker":
        return _execute_persistent_worker(
            executable=executable,
            plan=plan,
            records=records,
            order=order,
            manifest_path=manifest_path,
            config_hash=config_hash,
            split=split,
            config=config,
            estimate_bytes=estimate_bytes,
        )
    max_retries = int(runtime.get("max_retries", 1))
    newly_succeeded = newly_failed = 0
    for record in executable:
        output = Path(record["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        command = build_inference_command(record, config)
        record["command"] = command
        attempts = max_retries + 1
        for attempt in range(attempts):
            record.update(
                {
                    "status": "running",
                    "started_at": _utc_now(),
                    "retry_count": attempt,
                }
            )
            write_manifest(manifest_path, [records[item] for item in order], config_hash, split)
            start = time.monotonic()
            try:
                process = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                stdout = process.stdout
                stderr = process.stderr
                returncode = process.returncode
            except Exception as error:
                stdout = ""
                stderr = traceback.format_exc()
                returncode = None
                record["launch_error"] = repr(error)
            duration = time.monotonic() - start
            log_path = output.with_name("run.log")
            log_path.write_text(
                "COMMAND\n{}\n\nSTDOUT\n{}\n\nSTDERR\n{}\n".format(
                    json.dumps(command), stdout, stderr
                ),
                encoding="utf-8",
            )
            integrity = (
                audit_run_artifacts(record, config)
                if returncode == 0
                else {"complete": False, "issues": ["process_failed"]}
            )
            success = returncode == 0 and bool(integrity.get("complete"))
            record.update(
                {
                    "returncode": returncode,
                    "duration_seconds": round(duration, 3),
                    "finished_at": _utc_now(),
                    "log_path": str(log_path),
                    "artifact_integrity": integrity,
                    "error": None
                    if success
                    else (
                        stderr[-8000:]
                        or "artifact integrity failed: {}".format(
                            integrity.get("issues", [])
                        )
                    ),
                }
            )
            if success:
                record["status"] = "succeeded"
                metadata_path = Path(record["metadata_path"])
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    distribution = metadata.get("distribution", {})
                    for field in (
                        "achieved_kl",
                        "alpha",
                        "basis_checksum",
                        "covariance_checksum",
                    ):
                        if field in distribution:
                            record[field] = distribution[field]
                newly_succeeded += 1
                break
            record["status"] = "failed"
            write_manifest(manifest_path, [records[item] for item in order], config_hash, split)
            if attempt + 1 < attempts:
                if "out of memory" in (stderr or "").lower():
                    record["oom_retry"] = True
                _clear_cuda_cache()
        else:
            newly_failed += 1
        write_manifest(manifest_path, [records[item] for item in order], config_hash, split)
    final_records = [records[str(item["run_id"])] for item in plan]
    final_integrity = {
        str(item["run_id"]): audit_run_artifacts(item, config)
        for item in final_records
        if item.get("status") == "succeeded"
    }
    for item in final_records:
        if str(item["run_id"]) in final_integrity:
            item["artifact_integrity"] = final_integrity[str(item["run_id"])]
    artifact_complete = sum(
        item.get("status") == "succeeded"
        and bool(item.get("artifact_integrity", {}).get("complete"))
        for item in final_records
    )
    write_manifest(manifest_path, [records[item] for item in order], config_hash, split)
    return {
        "planned": len(plan),
        "executed": len(executable),
        "estimated_output_bytes": estimate_bytes,
        "succeeded": sum(item.get("status") == "succeeded" for item in final_records),
        "artifact_complete": artifact_complete,
        "artifact_incomplete": sum(
            item.get("status") == "succeeded"
            and not bool(item.get("artifact_integrity", {}).get("complete"))
            for item in final_records
        ),
        "failed": sum(item.get("status") == "failed" for item in final_records),
        "newly_succeeded": newly_succeeded,
        "newly_failed": newly_failed,
    }


def _status_path(root: Path) -> Path:
    return root / "runtime_status.json"


def load_runtime_status(root: Path, config_hash: str) -> Dict[str, Any]:
    path = _status_path(root)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != config_hash:
            raise ValueError("runtime status belongs to another config")
        return payload
    return {
        "schema_version": 1,
        "config_hash": config_hash,
        "statement": STATEMENT,
        "implementation_complete": False,
        "tests_complete": False,
        "pilot_complete": False,
        "full_complete": False,
        "met3r_complete": False,
        "trajectory_complete": False,
        "report_complete": False,
        "stages": {},
        "blockers": [],
    }


def update_runtime_status(
    root: Path,
    status: Dict[str, Any],
    stage: str,
    payload: Mapping[str, Any],
    *,
    blocker: Optional[Mapping[str, Any]] = None,
) -> None:
    status["updated_at"] = _utc_now()
    status.setdefault("stages", {})[stage] = dict(payload)
    if blocker is not None and blocker not in status.setdefault("blockers", []):
        status["blockers"].append(dict(blocker))
    _atomic_json(_status_path(root), status)


def validate_inputs_stage(root: Path, config: Mapping[str, Any], input_override: Optional[Path]) -> Dict[str, Any]:
    data = config["data"]
    explicit = input_override or (Path(data["input_dir"]) if data.get("input_dir") else None)
    drive = Path(data["drive_input_dir"]) if data.get("drive_input_dir") else None
    directory = resolve_input_directory(explicit, drive_directory=drive)
    output = root / "inputs"
    frozen_path = output / "input_validation.json"

    def scan(destination: Path) -> Dict[str, Any]:
        if directory is None:
            return {
                "schema_version": 1,
                "formal_ready": False,
                "distinct_count": 0,
                "pilot_count": 0,
                "full_count": 0,
                "required_pilot_count": int(data["pilot_count"]),
                "required_full_count": int(data["full_count"]),
                "min_distinct_inputs": int(data["min_distinct_inputs"]),
                "missing_distinct_inputs": int(data["min_distinct_inputs"]),
                "blocker": "no_input_directory",
                "records": [],
                "rejected": [],
            }
        return validate_input_directory(
            directory,
            destination,
            pilot_count=int(data["pilot_count"]),
            full_count=int(data["full_count"]),
            min_distinct_inputs=int(data["min_distinct_inputs"]),
        )

    if not frozen_path.exists():
        payload = scan(output)
        if directory is None:
            _atomic_json(frozen_path, payload)
        return payload
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if not isinstance(frozen, dict):
            raise ValueError("frozen input validation root is not an object")
    except Exception as error:
        return {
            "schema_version": 1,
            "formal_ready": False,
            "input_manifest_changed": True,
            "input_manifest_change": {
                "reason": "frozen_input_manifest_invalid",
                "error": repr(error),
                "frozen_path": str(frozen_path),
                "requires_new_experiment_id": True,
            },
            "records": [],
        }
    with tempfile.TemporaryDirectory(prefix="nile-input-rescan-") as temporary:
        observed = scan(Path(temporary))
    if _canonical(observed) == _canonical(frozen):
        return frozen
    result = dict(frozen)
    frozen_shas = {
        str(item.get("sha256"))
        for item in frozen.get("records", [])
        if isinstance(item, Mapping) and item.get("sha256")
    }
    observed_shas = {
        str(item.get("sha256"))
        for item in observed.get("records", [])
        if isinstance(item, Mapping) and item.get("sha256")
    }
    result["input_manifest_changed"] = True
    result["input_manifest_change"] = {
        "reason": "input_directory_no_longer_matches_frozen_manifest",
        "frozen_path": str(frozen_path),
        "added_sha256": sorted(observed_shas.difference(frozen_shas)),
        "removed_sha256": sorted(frozen_shas.difference(observed_shas)),
        "frozen_record_count": len(frozen.get("records", [])),
        "observed_record_count": len(observed.get("records", [])),
        "requires_new_experiment_id": True,
    }
    return result


def run_preflight_stage(root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "scripts.diagnose_nile_lowrank",
        "--config",
        str(root / "configs" / "resolved_config.json"),
        "--output-dir",
        str(root / "distribution_gates"),
    ]
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = root / "distribution_gates" / "preflight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        log_path,
        "COMMAND\n{}\n\nSTDOUT\n{}\n\nSTDERR\n{}\n".format(
            json.dumps(command), process.stdout or "", process.stderr or ""
        ),
    )
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    output = root / "distribution_gates" / "configuration_gates.json"
    payload = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "passed": False,
        "error": "preflight process returned {}".format(process.returncode),
        "configurations": [],
    }
    payload["returncode"] = process.returncode
    payload["completed"] = process.returncode == 0 and output.exists()
    payload["log_path"] = str(log_path)
    if process.returncode != 0:
        payload["stderr_tail"] = (process.stderr or "")[-8000:]
    return payload


def gated_pilot_configurations(root: Path, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requested = build_pilot_configurations(config)
    path = root / "distribution_gates" / "configuration_gates.json"
    if not path.exists():
        raise ValueError("preflight output is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("diagnostic_plots_complete") is not True:
        raise ValueError(
            "preflight diagnostic plots are missing or incomplete; GPU generation is blocked"
        )
    gates = {item["config_id"]: item for item in payload.get("configurations", [])}
    requested_ids = {item["config_id"] for item in requested}
    if set(gates) != requested_ids:
        raise ValueError(
            "preflight gate coverage mismatch: missing={}, unexpected={}".format(
                sorted(requested_ids.difference(gates)),
                sorted(set(gates).difference(requested_ids)),
            )
        )
    selected = []
    for item in requested:
        gate = gates.get(item["config_id"])
        if (
            gate is None
            or not gate.get("passed", False)
            or not gate.get("eligible_for_generation", False)
        ):
            continue
        merged = dict(item)
        merged["distribution_gate_passed"] = True
        for field in (
            "achieved_kl",
            "alpha",
            "basis_checksum",
            "covariance_checksum",
        ):
            if field in gate:
                merged[field] = gate[field]
        selected.append(merged)
    excluded_unattainable = [
        item
        for item in gates.values()
        if item.get("exclusion_reason") == "unattainable_target_kl"
    ]
    expected_unattainable = config.get("preflight", {}).get(
        "expected_unattainable_count", 3
    )
    if expected_unattainable is not None and len(excluded_unattainable) != int(
        expected_unattainable
    ):
        raise ValueError(
            "formal matrix expected {} unattainable targets, observed {}".format(
                expected_unattainable, len(excluded_unattainable)
            )
        )
    return selected


def audit_test_results_receipt(path: Path) -> Dict[str, Any]:
    """Validate the atomic compileall/pytest receipt emitted by Colab cell 8."""

    path = Path(path)
    base: Dict[str, Any] = {
        "receipt_path": str(path),
        "exists": path.is_file(),
        "verified": False,
        "issues": [],
    }
    if not path.is_file():
        base["issues"] = ["test_results_missing"]
        return base
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        base["issues"] = ["test_results_invalid_json"]
        base["error"] = repr(error)
        return base
    if not isinstance(payload, Mapping):
        base["issues"] = ["test_results_root_not_object"]
        return base
    result = {**dict(payload), **base}
    issues = []
    if payload.get("schema_version") != 1:
        issues.append("test_results_schema_version_mismatch")
    if payload.get("passed") is not True:
        issues.append("tests_passed_not_true")
    if payload.get("tests_complete") is not True:
        issues.append("tests_complete_not_true")
    if payload.get("compileall_returncode") != 0:
        issues.append("compileall_failed_or_missing")
    if payload.get("pytest_returncode") != 0:
        issues.append("pytest_failed_or_missing")
    if not isinstance(payload.get("finished_at"), str) or not payload.get(
        "finished_at"
    ):
        issues.append("test_results_finished_at_missing")
    commands = payload.get("command")
    if not isinstance(commands, list) or len(commands) != 2:
        issues.append("test_results_commands_missing")
    result["issues"] = issues
    result["verified"] = not issues
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file_identity(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "declared_path": str(path),
        "resolved_path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _checkpoint_stable_identity(
    identity: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return fields that indicate content replacement, excluding FUSE metadata."""

    if not isinstance(identity, Mapping):
        return None
    return {
        "declared_path": identity.get("declared_path"),
        "resolved_path": identity.get("resolved_path"),
        "size": identity.get("size"),
    }


def audit_checkpoint_manifest(
    path: Path,
    config: Mapping[str, Any],
    *,
    cache_path: Optional[Path] = None,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify frozen revisions and the exact cached MV-Adapter checkpoint."""

    path = Path(path)
    result: Dict[str, Any] = {
        "manifest_path": str(path),
        "exists": path.is_file(),
        "verified": False,
        "issues": [],
    }
    if not path.is_file():
        result["issues"] = ["checkpoint_manifest_missing"]
        return result
    try:
        manifest_bytes = path.read_bytes()
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as error:
        result["issues"] = ["checkpoint_manifest_invalid_json"]
        result["error"] = repr(error)
        return result
    if not isinstance(payload, Mapping):
        result["issues"] = ["checkpoint_manifest_root_not_object"]
        return result
    manifest_content_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    resolved_config_hash = str(config_hash or _hash(config))
    checkpoint_entry = payload.get("adapter_checkpoint")
    checkpoint_path = None
    checkpoint_identity = None
    if isinstance(checkpoint_entry, Mapping) and checkpoint_entry.get("path") not in (
        None,
        "",
    ):
        checkpoint_path = Path(str(checkpoint_entry["path"])).expanduser()
        checkpoint_identity = _checkpoint_file_identity(checkpoint_path)
    cache_key = {
        "config_hash": resolved_config_hash,
        "manifest_content_sha256": manifest_content_sha256,
        "checkpoint_file": checkpoint_identity,
    }
    if cache_path is not None and checkpoint_identity is not None:
        cache_path = Path(cache_path)
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if (
            isinstance(cached, Mapping)
            and cached.get("schema_version") == 1
            and cached.get("verified") is True
            and cached.get("cache_key") == cache_key
            and isinstance(cached.get("audit"), Mapping)
            and cached["audit"].get("verified") is True
            and not cached["audit"].get("issues")
            and _checkpoint_file_identity(checkpoint_path) == checkpoint_identity
        ):
            cached_audit = dict(cached["audit"])
            cached_audit.update(
                {
                    "cache_hit": True,
                    "cache_path": str(cache_path),
                    "cache_key": cache_key,
                }
            )
            return cached_audit
    issues = []
    if payload.get("schema_version") != 1:
        issues.append("checkpoint_manifest_schema_version_mismatch")
    model = config.get("model", {})
    evaluation = config.get("evaluation", {})
    expected_revisions = {
        "base_model": (
            model.get("base_model"),
            model.get("base_model_revision"),
        ),
        "vae_model": (
            model.get("vae_model"),
            model.get("vae_model_revision"),
        ),
        "adapter_path": (
            model.get("adapter_path"),
            model.get("adapter_revision"),
        ),
        "birefnet_model": (
            model.get("birefnet_model"),
            model.get("birefnet_revision"),
        ),
        "identity_model": (
            evaluation.get("identity_model"),
            evaluation.get("identity_model_revision"),
        ),
    }
    revisions = payload.get("resolved_revisions")
    if not isinstance(revisions, Mapping):
        issues.append("resolved_revisions_missing")
        revisions = {}
    for label, (expected_repo, expected_revision) in expected_revisions.items():
        entry = revisions.get(label)
        if not isinstance(entry, Mapping):
            issues.append("resolved_revision_missing:{}".format(label))
            continue
        if str(entry.get("repo_id")) != str(expected_repo):
            issues.append("resolved_repo_mismatch:{}".format(label))
        if str(entry.get("revision")) != str(expected_revision):
            issues.append("resolved_revision_mismatch:{}".format(label))
    if str(payload.get("met3r_revision")) != str(
        evaluation.get("met3r_revision")
    ):
        issues.append("met3r_revision_mismatch")

    checkpoint = checkpoint_entry
    actual_sha256 = None
    identity_before_hash = checkpoint_identity
    identity_after_hash = checkpoint_identity
    checkpoint_metadata_changed_during_hash = False
    if not isinstance(checkpoint, Mapping):
        issues.append("adapter_checkpoint_manifest_missing")
    else:
        checkpoint_path_value = checkpoint.get("path")
        if checkpoint_path_value in (None, ""):
            issues.append("adapter_checkpoint_path_missing")
        else:
            checkpoint_path = Path(str(checkpoint_path_value)).expanduser()
            if not checkpoint_path.is_file():
                issues.append("adapter_checkpoint_file_missing")
            else:
                expected_name = str(model.get("mv_adapter_checkpoint") or "")
                if expected_name and checkpoint_path.name != expected_name:
                    issues.append("adapter_checkpoint_filename_mismatch")
                try:
                    actual_sha256 = _sha256_file(checkpoint_path)
                except Exception as error:
                    issues.append("adapter_checkpoint_hash_failed")
                    result["hash_error"] = repr(error)
                identity_after_hash = _checkpoint_file_identity(checkpoint_path)
                checkpoint_metadata_changed_during_hash = (
                    identity_after_hash != identity_before_hash
                )
                if _checkpoint_stable_identity(
                    identity_after_hash
                ) != _checkpoint_stable_identity(identity_before_hash):
                    issues.append("adapter_checkpoint_changed_during_hash")
                cache_key["checkpoint_file"] = identity_after_hash
        manifest_sha256 = str(checkpoint.get("sha256") or "").lower()
        config_sha256 = str(model.get("adapter_sha256") or "").lower()
        if manifest_sha256 != config_sha256:
            issues.append("adapter_sha256_config_manifest_mismatch")
        if actual_sha256 is not None and actual_sha256 != manifest_sha256:
            issues.append("adapter_sha256_file_manifest_mismatch")
        if actual_sha256 is not None and actual_sha256 != config_sha256:
            issues.append("adapter_sha256_file_config_mismatch")
    result.update(
        {
            "issues": sorted(set(issues)),
            "verified": not issues,
            "adapter_checkpoint_path": (
                str(checkpoint_path) if checkpoint_path is not None else None
            ),
            "actual_adapter_sha256": actual_sha256,
            "checkpoint_identity_before_hash": identity_before_hash,
            "checkpoint_identity_after_hash": identity_after_hash,
            "checkpoint_metadata_changed_during_hash": (
                checkpoint_metadata_changed_during_hash
            ),
            "manifest_content_sha256": manifest_content_sha256,
            "config_hash": resolved_config_hash,
            "cache_hit": False,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "cache_key": cache_key,
        }
    )
    if result["verified"] and cache_path is not None:
        _atomic_json(
            cache_path,
            {
                "schema_version": 1,
                "verified": True,
                "cached_at": _utc_now(),
                "cache_key": cache_key,
                "audit": result,
            },
        )
    return result


def _formal_blockers(
    config: Mapping[str, Any],
    input_validation: Mapping[str, Any],
    environment: Mapping[str, Any],
    test_results_audit: Optional[Mapping[str, Any]] = None,
    checkpoint_audit: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    blockers = []
    if not isinstance(test_results_audit, Mapping) or not test_results_audit.get(
        "verified", False
    ):
        blockers.append(
            {
                "code": "tests_not_verified",
                "receipt_path": (
                    test_results_audit.get("receipt_path")
                    if isinstance(test_results_audit, Mapping)
                    else None
                ),
                "issues": (
                    list(test_results_audit.get("issues", []))
                    if isinstance(test_results_audit, Mapping)
                    else ["test_results_audit_missing"]
                ),
            }
        )
    if not isinstance(checkpoint_audit, Mapping) or not checkpoint_audit.get(
        "verified", False
    ):
        blockers.append(
            {
                "code": "checkpoint_provenance_not_verified",
                "manifest_path": (
                    checkpoint_audit.get("manifest_path")
                    if isinstance(checkpoint_audit, Mapping)
                    else None
                ),
                "issues": (
                    list(checkpoint_audit.get("issues", []))
                    if isinstance(checkpoint_audit, Mapping)
                    else ["checkpoint_audit_missing"]
                ),
            }
        )
    if input_validation.get("input_manifest_changed", False):
        change = input_validation.get("input_manifest_change", {})
        blockers.append(
            {
                "code": "input_manifest_changed",
                "frozen_path": (
                    change.get("frozen_path")
                    if isinstance(change, Mapping)
                    else None
                ),
                "change": dict(change) if isinstance(change, Mapping) else {},
                "requires_new_experiment_id": True,
            }
        )
    if not input_validation.get("formal_ready", False):
        blockers.append(
            {
                "code": "insufficient_formal_inputs",
                "required": config["data"]["min_distinct_inputs"],
                "available": input_validation.get("distinct_count", 0),
                "missing": input_validation.get("missing_distinct_inputs"),
            }
        )
    cuda = environment.get("python_cuda", {})
    if not cuda.get("cuda_available", False):
        blockers.append(
            {
                "code": "cuda_unavailable_to_python",
                "detail": cuda.get("error"),
            }
        )
    minimum_free_gib = float(
        config.get("runtime", {}).get("min_free_disk_gib", 25.0)
    )
    available_bytes = int(environment.get("disk_free_bytes", 0) or 0)
    if available_bytes < minimum_free_gib * (1024 ** 3):
        blockers.append(
            {
                "code": "insufficient_disk_space",
                "required_free_gib": minimum_free_gib,
                "available_free_gib": round(available_bytes / (1024 ** 3), 3),
            }
        )
    model = config["model"]
    missing_revisions = [
        key
        for key in (
            "base_model_revision",
            "vae_model_revision",
            "adapter_revision",
            "birefnet_revision",
        )
        if model.get(key) in (None, "", "main")
    ]
    if missing_revisions:
        blockers.append(
            {
                "code": "model_revisions_not_immutable",
                "fields": missing_revisions,
            }
        )
    adapter_sha256 = str(model.get("adapter_sha256") or "")
    if len(adapter_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in adapter_sha256
    ):
        blockers.append(
            {
                "code": "adapter_checkpoint_sha256_missing_or_invalid",
                "field": "model.adapter_sha256",
            }
        )
    evaluation = config.get("evaluation", {})
    if config.get("experiment", {}).get("run_met3r", True) and not evaluation.get(
        "met3r_revision"
    ):
        blockers.append(
            {"code": "met3r_revision_not_immutable", "field": "evaluation.met3r_revision"}
        )
    if config.get("experiment", {}).get("run_met3r", True):
        try:
            met3r_available = importlib.util.find_spec("met3r") is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            met3r_available = False
        if not met3r_available:
            blockers.append(
                {
                    "code": "met3r_package_unavailable",
                    "required_revision": evaluation.get("met3r_revision"),
                }
            )
    if not evaluation.get("identity_model_revision"):
        blockers.append(
            {
                "code": "identity_model_revision_not_immutable",
                "field": "evaluation.identity_model_revision",
            }
        )
    return blockers


def run_evaluation_stage(
    root: Path,
    split: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = root / split / "manifest.json"
    if not manifest.exists():
        return {"completed": False, "reason": "manifest_missing", "split": split}
    evaluation = config.get("evaluation", {})
    output_dir = root / "metrics" / split
    command = [
        sys.executable,
        "-m",
        "scripts.eval_nile_lowrank_study",
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--metrics",
        "all" if config["experiment"].get("run_met3r", True) else "lightweight",
        "--identity-model",
        str(evaluation.get("identity_model", "facebook/dinov2-small")),
        "--identity-device",
        str(evaluation.get("met3r_device", "cuda")),
        "--met3r-device",
        str(evaluation.get("met3r_device", "cuda")),
        "--met3r-image-size",
        str(evaluation.get("met3r_image_size", 256)),
        "--met3r-batch-size",
        str(evaluation.get("met3r_batch_size", 1)),
        "--bootstrap-iterations",
        str(config["experiment"].get("bootstrap_iterations", 10000)),
        "--bootstrap-seed",
        str(config["experiment"].get("bootstrap_seed", 20260711)),
        "--plots-dir",
        str(root / "plots" / split),
        "--contact-sheets-dir",
        str(root / "contact_sheets" / split),
    ]
    command.extend(
        ["--angle-bins-deg"]
        + [str(float(value)) for value in evaluation["angle_bins_deg"]]
    )
    _add_optional(
        command,
        "--identity-model-revision",
        evaluation.get("identity_model_revision"),
    )
    _add_optional(command, "--met3r-revision", evaluation.get("met3r_revision"))
    process = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    evaluator_stdout = str(getattr(process, "stdout", "") or "")
    evaluator_stderr = str(getattr(process, "stderr", "") or "")
    evaluator_log = output_dir / "evaluator.log"
    _atomic_text(
        evaluator_log,
        "\n".join(
            [
                "COMMAND " + json.dumps(command, ensure_ascii=False),
                "RETURN_CODE " + str(process.returncode),
                "",
                "=== STDOUT ===",
                evaluator_stdout.rstrip() or "<empty>",
                "",
                "=== STDERR ===",
                evaluator_stderr.rstrip() or "<empty>",
                "",
            ]
        ),
    )
    metrics_path = output_dir / "lowrank_metrics.json"
    metrics_payload: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics_payload = {}
    manifest_records = _read_manifest(manifest)
    expected_samples = sum(item.get("status") == "succeeded" for item in manifest_records)
    samples = metrics_payload.get("samples", [])
    met3r_required = bool(config["experiment"].get("run_met3r", True))
    met3r_complete = bool(
        met3r_required
        and expected_samples > 0
        and len(samples) == expected_samples
        and all(
            item.get("status") == "succeeded"
            and _finite_number(item.get("angle_all_met3r_score"))
            for item in samples
        )
        and bool(metrics_payload.get("met3r_provenance", {}).get("verified"))
    )
    guardrails_complete = bool(
        expected_samples > 0
        and len(samples) == expected_samples
        and all(
            _finite_number(item.get("dino_reference_mean"))
            and int(item.get("mask_view_count", 0)) == len(config["model"]["views_deg"])
            and not item.get("guardrail_error")
            for item in samples
        )
    )
    payload_formal_complete = bool(metrics_payload.get("formal_evaluation_complete"))
    return {
        "completed": process.returncode == 0 and metrics_path.exists(),
        "formal_evaluation_complete": bool(
            process.returncode == 0
            and metrics_path.exists()
            and (met3r_complete if met3r_required else True)
            and guardrails_complete
            and payload_formal_complete
        ),
        "returncode": process.returncode,
        "split": split,
        "metrics_path": str(metrics_path),
        "expected_sample_count": expected_samples,
        "evaluated_sample_count": len(samples),
        "met3r_required": met3r_required,
        "met3r_complete": met3r_complete,
        "guardrails_complete": guardrails_complete,
        "payload_formal_evaluation_complete": payload_formal_complete,
        "plots_dir": str(root / "plots" / split),
        "evaluator_log": str(evaluator_log),
        "stdout_tail": evaluator_stdout[-12000:],
        "stderr_tail": evaluator_stderr[-12000:],
    }


def run_selection_stage(root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    metrics_path = root / "metrics" / "pilot" / "lowrank_metrics.json"
    if not metrics_path.exists():
        return {"completed": False, "reason": "pilot_metrics_missing"}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        not metrics.get("met3r_required", False)
        or metrics.get("met3r_score_direction") != "lower_is_better"
        or not metrics.get("formal_evaluation_complete", False)
        or not metrics.get("met3r_provenance", {}).get("verified", False)
    ):
        return {"completed": False, "reason": "pilot_met3r_missing"}
    result = select_candidates(
        metrics.get("configuration_summaries", []), config.get("selection", {})
    )
    result["study_config_hash"] = _hash(config)
    result["pilot_metrics_sha256"] = hashlib.sha256(
        metrics_path.read_bytes()
    ).hexdigest()
    result["candidate_configuration_hash"] = result["configuration_hash"]
    result["configuration_hash"] = _hash(
        {
            "candidate_configuration_hash": result["candidate_configuration_hash"],
            "study_config_hash": result["study_config_hash"],
            "pilot_metrics_sha256": result["pilot_metrics_sha256"],
        }
    )
    directory = root / "selected_candidates"
    json_path = directory / "selected_candidates.json"
    yaml_path = directory / "selected_candidates.yaml"
    freeze_candidates(result, json_path, yaml_path)
    return {
        "completed": True,
        "configuration_hash": result["configuration_hash"],
        "selected_candidates": str(json_path),
        "selections": result["selections"],
    }


SELECTED_CONFIGURATION_FIELDS = (
    "config_id",
    "method",
    "rank",
    "target_kl",
    "achieved_kl",
    "alpha",
    "rbf_length_scale_deg",
    "basis_checksum",
    "covariance_checksum",
    "distribution_gate_passed",
)


def _selection_value_matches(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        )
    return left == right


def audit_selected_candidates(
    root: Path,
    config: Mapping[str, Any],
    *,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify the complete frozen-selection proof before costly generation."""

    selected_json = root / "selected_candidates" / "selected_candidates.json"
    selected_yaml = root / "selected_candidates" / "selected_candidates.yaml"
    metrics_path = root / "metrics" / "pilot" / "lowrank_metrics.json"
    gate_path = root / "distribution_gates" / "configuration_gates.json"
    issues: List[str] = []

    def load_mapping(path: Path, label: str) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root is not an object")
            return payload
        except Exception as error:
            issues.append("{}_missing_or_invalid".format(label))
            return {}

    selected = load_mapping(selected_json, "selected_candidates_json")
    selected_yaml_payload = load_mapping(
        selected_yaml, "selected_candidates_yaml"
    )
    metrics = load_mapping(metrics_path, "pilot_metrics")
    gates_payload = load_mapping(gate_path, "preflight_gates")
    current_config_hash = str(config_hash or _hash(config))
    actual_metrics_sha256 = (
        hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        if metrics_path.is_file()
        else None
    )

    if selected and selected_yaml_payload:
        if _canonical(selected) != _canonical(selected_yaml_payload):
            issues.append("selected_json_yaml_mismatch")
    if selected.get("study_config_hash") != current_config_hash:
        issues.append("selected_study_config_hash_mismatch")
    if selected.get("pilot_metrics_sha256") != actual_metrics_sha256:
        issues.append("selected_pilot_metrics_sha256_mismatch")

    selected_core = dict(selected)
    for field in (
        "configuration_hash",
        "candidate_configuration_hash",
        "study_config_hash",
        "pilot_metrics_sha256",
    ):
        selected_core.pop(field, None)
    recomputed_candidate_hash = _hash(selected_core) if selected else None
    declared_candidate_hash = selected.get("candidate_configuration_hash")
    if declared_candidate_hash != recomputed_candidate_hash:
        issues.append("selected_candidate_hash_mismatch")

    summaries = metrics.get("configuration_summaries", [])
    if not isinstance(summaries, list):
        summaries = []
        issues.append("pilot_configuration_summaries_invalid")
    try:
        expected_selection = select_candidates(
            summaries, config.get("selection", {})
        )
    except Exception as error:
        expected_selection = {}
        issues.append("current_candidate_selection_failed")
    expected_candidate_hash = expected_selection.get("configuration_hash")
    expected_core = dict(expected_selection)
    expected_core.pop("configuration_hash", None)
    if declared_candidate_hash != expected_candidate_hash:
        issues.append("selected_candidate_not_current_policy_result")
    if selected and _canonical(selected_core) != _canonical(expected_core):
        issues.append("selected_candidate_content_mismatch")

    recomputed_configuration_hash = (
        _hash(
            {
                "candidate_configuration_hash": recomputed_candidate_hash,
                "study_config_hash": current_config_hash,
                "pilot_metrics_sha256": actual_metrics_sha256,
            }
        )
        if recomputed_candidate_hash is not None
        and actual_metrics_sha256 is not None
        else None
    )
    if selected.get("configuration_hash") != recomputed_configuration_hash:
        issues.append("selected_configuration_hash_mismatch")

    try:
        passed_preflight = gated_pilot_configurations(root, config)
    except Exception as error:
        passed_preflight = []
        issues.append("current_preflight_gate_audit_failed")
    preflight_by_id = {
        str(item.get("config_id")): item
        for item in passed_preflight
        if item.get("config_id") is not None
    }
    summary_by_id: Dict[str, List[Mapping[str, Any]]] = {}
    for row in summaries:
        if isinstance(row, Mapping) and row.get("config_id") is not None:
            summary_by_id.setdefault(str(row["config_id"]), []).append(row)

    topology_methods = {
        "camera_rbf": "lowrank_camera_rbf",
        "nested_tree_a": "lowrank_nested_tree_a",
        "nested_tree_ab": "lowrank_nested_tree_ab",
    }
    formal_rank_kl: List[tuple] = []
    selections = selected.get("selections", {})
    if not isinstance(selections, Mapping):
        selections = {}
        issues.append("selected_selections_invalid")
    for topology, expected_method in topology_methods.items():
        selection = selections.get(topology)
        if not isinstance(selection, Mapping):
            issues.append("selected_topology_missing:{}".format(topology))
            continue
        configuration = selection.get("configuration")
        if not isinstance(configuration, Mapping):
            issues.append("selected_configuration_missing:{}".format(topology))
            continue
        config_id = str(configuration.get("config_id"))
        if configuration.get("method") != expected_method:
            issues.append("selected_method_mismatch:{}".format(topology))
        formal = (
            selection.get("status") == "selected"
            and not bool(selection.get("diagnostic_only", False))
        )
        diagnostic = (
            selection.get("status") == "no_eligible_candidate"
            and bool(selection.get("diagnostic_only", False))
        )
        if not formal and not diagnostic:
            issues.append(
                "selected_candidate_status_invalid:{}".format(topology)
            )
        if formal:
            try:
                formal_rank_kl.append(
                    (
                        int(configuration["rank"]),
                        float(configuration["target_kl"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                issues.append(
                    "selected_rank_kl_invalid:{}".format(topology)
                )

        gate = preflight_by_id.get(config_id)
        if gate is None:
            issues.append("selected_not_in_passed_preflight:{}".format(topology))
        summary_rows = summary_by_id.get(config_id, [])
        if len(summary_rows) != 1:
            issues.append(
                "selected_pilot_summary_count_mismatch:{}".format(topology)
            )
            summary = None
        else:
            summary = summary_rows[0]
        for source_name, source in (("preflight", gate), ("pilot", summary)):
            if source is None:
                continue
            if source.get("distribution_gate_passed") is not True:
                issues.append(
                    "selected_{}_gate_not_passed:{}".format(
                        source_name, topology
                    )
                )
            for field in SELECTED_CONFIGURATION_FIELDS:
                if not _selection_value_matches(
                    configuration.get(field), source.get(field)
                ):
                    issues.append(
                        "selected_{}_field_mismatch:{}:{}".format(
                            source_name, topology, field
                        )
                    )

    if len(formal_rank_kl) >= 2 and len(set(formal_rank_kl)) != 1:
        issues.append("selected_equal_rank_kl_mismatch")

    return {
        "ready": not issues,
        "issues": sorted(set(issues)),
        "selected_json": str(selected_json),
        "selected_yaml": str(selected_yaml),
        "pilot_metrics": str(metrics_path),
        "preflight_gates": str(gate_path),
        "study_config_hash": current_config_hash,
        "declared_candidate_configuration_hash": declared_candidate_hash,
        "recomputed_candidate_configuration_hash": recomputed_candidate_hash,
        "expected_candidate_configuration_hash": expected_candidate_hash,
        "declared_configuration_hash": selected.get("configuration_hash"),
        "recomputed_configuration_hash": recomputed_configuration_hash,
        "declared_pilot_metrics_sha256": selected.get(
            "pilot_metrics_sha256"
        ),
        "actual_pilot_metrics_sha256": actual_metrics_sha256,
        "formal_rank_kl": [list(item) for item in formal_rank_kl],
        "preflight_configuration_count": len(
            gates_payload.get("configurations", [])
        ),
        "pilot_summary_count": len(summaries),
    }


def audit_pilot_met3r_prerequisite(
    root: Path, config: Mapping[str, Any]
) -> Dict[str, Any]:
    metrics_path = root / "metrics" / "pilot" / "lowrank_metrics.json"
    manifest_path = root / "pilot" / "manifest.json"
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}
    manifest: List[Dict[str, Any]] = []
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        reasons.append("pilot_metrics_missing_or_invalid")
    try:
        manifest = _read_manifest(manifest_path)
    except Exception:
        reasons.append("pilot_manifest_missing_or_invalid")
    if not metrics.get("met3r_required", False):
        reasons.append("pilot_met3r_not_required_in_artifact")
    if metrics.get("met3r_score_direction") != "lower_is_better":
        reasons.append("pilot_met3r_direction_invalid")
    if not metrics.get("met3r_provenance", {}).get("verified", False):
        reasons.append("pilot_met3r_revision_unverified")
    if not metrics.get("formal_evaluation_complete", False):
        reasons.append("pilot_formal_evaluation_incomplete")
    if not manifest or any(item.get("status") != "succeeded" for item in manifest):
        reasons.append("pilot_generation_incomplete")
    samples = metrics.get("samples", [])
    if len(samples) != len(manifest):
        reasons.append("pilot_evaluated_count_mismatch")
    if any(
        item.get("status") != "succeeded"
        or not _finite_number(item.get("angle_all_met3r_score"))
        for item in samples
    ):
        reasons.append("pilot_met3r_sample_missing")
    if bool(config.get("experiment", {}).get("strict_full_requires_met3r", True)):
        ready = not reasons
    else:
        ready = metrics_path.is_file()
    return {
        "ready": ready,
        "reasons": sorted(set(reasons)),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "planned_count": len(manifest),
        "evaluated_count": len(samples),
    }


def _run_generation_split(
    *,
    root: Path,
    split: str,
    config: Mapping[str, Any],
    config_hash: str,
    environment: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    inputs: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    plan = build_run_plan(
        split=split,
        inputs=inputs,
        seeds=seeds,
        configurations=configurations,
        config_hash=config_hash,
        root=root,
        config=config,
        git_commit=environment["git_commit"],
        code_revision=environment.get("code_revision"),
        gpu=environment["gpu"],
    )
    if split == "trajectory":
        for record in plan:
            record["trajectory_output"] = str(
                Path(record["output"]).with_name("trajectory.npz")
            )
    return execute_plan(
        plan,
        manifest_path=root / split / "manifest.json",
        config_hash=config_hash,
        split=split,
        config=config,
        resume=args.resume,
        force_rerun=args.force_rerun,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )


def _save_figure_atomic(figure: Any, path: Path) -> Path:
    """Save a matplotlib figure without exposing a partial PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    os.replace(temporary, path)
    return path


def _save_pair_view_correlation_heatmaps(
    iid_arrays: Mapping[str, np.ndarray],
    correlated_arrays: Mapping[str, np.ndarray],
    milestones: Sequence[str],
    path: Path,
) -> Path:
    """Render IID and correlated view-correlation matrices at every milestone."""

    import matplotlib.pyplot as plt

    iid = np.asarray(iid_arrays["view_correlation"], dtype=np.float64)
    correlated = np.asarray(
        correlated_arrays["view_correlation"], dtype=np.float64
    )
    if iid.shape != correlated.shape or iid.ndim != 4:
        raise ValueError(
            "paired view correlations must share shape [milestone, B, V, V]"
        )
    if iid.shape[0] != len(milestones) or iid.shape[-1] != iid.shape[-2]:
        raise ValueError("view-correlation shape does not match trajectory milestones")
    if iid.shape[1] <= 0 or iid.shape[-1] <= 1:
        raise ValueError("view-correlation heatmaps require a non-empty batch and views")
    if not np.isfinite(iid).all() or not np.isfinite(correlated).all():
        raise ValueError("view-correlation heatmaps require finite matrices")

    iid_mean = np.mean(iid, axis=1)
    correlated_mean = np.mean(correlated, axis=1)
    count = len(milestones)
    figure, axes = plt.subplots(
        2,
        count,
        figsize=(max(8.0, 2.65 * count), 5.4),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, (role, matrices) in enumerate(
        (("IID control", iid_mean), ("Correlated", correlated_mean))
    ):
        for index, milestone in enumerate(milestones):
            axis = axes[row, index]
            image = axis.imshow(
                matrices[index],
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
                interpolation="nearest",
            )
            axis.set_title(str(milestone))
            axis.set_xlabel("view")
            if index == 0:
                axis.set_ylabel(role + "\nview")
            view_count = matrices.shape[-1]
            axis.set_xticks(range(view_count))
            axis.set_yticks(range(view_count))
            axis.set_xticklabels([str(item + 1) for item in range(view_count)])
            axis.set_yticklabels([str(item + 1) for item in range(view_count)])
    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Pearson correlation",
        shrink=0.82,
    )
    try:
        return _save_figure_atomic(figure, path)
    finally:
        plt.close(figure)


def _save_aggregate_trajectory_plot(
    pairs: Sequence[Mapping[str, Any]],
    method_summaries: Sequence[Mapping[str, Any]],
    path: Path,
) -> Path:
    """Plot per-pair traces and per-method aggregate G_t/Delta_t curves."""

    import matplotlib.pyplot as plt

    if not pairs or not method_summaries:
        raise ValueError("aggregate trajectory plot requires paired method summaries")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for summary in method_summaries:
        method = str(summary["method"])
        progress = np.asarray(summary["target_progress"], dtype=np.float64)
        aggregate_g = np.asarray(summary["mean_g_t"], dtype=np.float64)
        aggregate_delta = np.asarray(summary["mean_delta_t"], dtype=np.float64)
        if (
            progress.ndim != 1
            or progress.shape != aggregate_g.shape
            or progress.shape != aggregate_delta.shape
            or not np.isfinite(progress).all()
            or not np.isfinite(aggregate_g).all()
            or not np.isfinite(aggregate_delta).all()
        ):
            raise ValueError("aggregate trajectory curves must be aligned and finite")
        method_pairs = [item for item in pairs if str(item.get("method")) == method]
        for pair in method_pairs:
            pair_progress = np.asarray(pair["target_progress"], dtype=np.float64)
            axes[0].plot(
                pair_progress,
                np.asarray(pair["mean_g_t"], dtype=np.float64),
                color="0.65",
                linewidth=0.8,
                alpha=0.28,
            )
            axes[1].plot(
                pair_progress,
                np.asarray(pair["mean_delta_t"], dtype=np.float64),
                color="0.65",
                linewidth=0.8,
                alpha=0.28,
            )
        label = "{} (n={})".format(method, len(method_pairs))
        axes[0].plot(progress, aggregate_g, marker="o", linewidth=2.0, label=label)
        axes[1].plot(
            progress, aggregate_delta, marker="o", linewidth=2.0, label=label
        )

    axes[0].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].set(
        xlabel="relative denoising progress",
        ylabel="G_t",
        title="Correlation retention",
    )
    axes[1].set(
        xlabel="relative denoising progress",
        ylabel="Delta_t",
        title="Paired divergence from IID",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    try:
        return _save_figure_atomic(figure, path)
    finally:
        plt.close(figure)


def summarize_trajectory_stage(
    root: Path,
    config_hash: str,
    *,
    washout_max_final_g: float = 0.5,
    amplified_min_final_g: float = 1.5,
) -> Dict[str, Any]:
    """Pair trajectory artifacts and classify correlation retention."""

    from mvadapter.nile.trajectory import load_trajectory_npz, save_paired_delta

    manifest_path = root / "trajectory" / "manifest.json"
    output_path = root / "trajectory" / "trajectory_summary.json"
    csv_path = root / "trajectory" / "trajectory_summary.csv"
    aggregate_plot_path = (
        root / "plots" / "trajectory" / "aggregate_g_delta_curves.png"
    )
    if not manifest_path.exists():
        payload = {
            "schema_version": 2,
            "complete": False,
            "visualization_complete": False,
            "correlation_state": "not_available",
            "pair_count": 0,
            "failures": [{"reason": "manifest_missing"}],
            "pairs": [],
            "method_summaries": [],
            "artifacts": {
                "summary_json": str(output_path),
                "summary_csv": None,
                "aggregate_g_delta_plot": None,
                "pair_view_correlation_heatmaps": [],
            },
        }
        _atomic_json(output_path, payload)
        return payload

    records = read_manifest(manifest_path, config_hash, "trajectory")
    groups: MutableMapping[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        pair_id = record.get("trajectory_pair_id")
        if pair_id:
            key = (record.get("input_sha256"), record.get("seed"), pair_id)
            groups.setdefault(key, []).append(record)

    unpaired_manifest_records = [
        str(record.get("run_id"))
        for record in records
        if not record.get("trajectory_pair_id")
    ]

    pairs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for (input_sha256, seed, pair_id), members in sorted(groups.items(), key=lambda item: repr(item[0])):
        control = next(
            (item for item in members if item.get("trajectory_role") == "iid_control"),
            None,
        )
        correlated = next(
            (item for item in members if item.get("trajectory_role") == "correlated"),
            None,
        )
        identity = {
            "input_sha256": input_sha256,
            "seed": seed,
            "trajectory_pair_id": pair_id,
        }
        if len(members) != 2:
            failures.append(
                {**identity, "reason": "pair_cardinality_invalid", "member_count": len(members)}
            )
            continue
        if control is None or correlated is None:
            failures.append({**identity, "reason": "paired_role_missing"})
            continue
        if control.get("status") != "succeeded" or correlated.get("status") != "succeeded":
            failures.append(
                {
                    **identity,
                    "reason": "paired_run_not_succeeded",
                    "iid_status": control.get("status"),
                    "correlated_status": correlated.get("status"),
                }
            )
            continue
        if control.get("rank") != correlated.get("rank"):
            failures.append(
                {
                    **identity,
                    "reason": "observer_rank_mismatch",
                    "iid_rank": control.get("rank"),
                    "correlated_rank": correlated.get("rank"),
                }
            )
            continue
        if control.get("paired_method") != correlated.get("method"):
            failures.append({**identity, "reason": "paired_method_mismatch"})
            continue
        iid_path = Path(str(control.get("trajectory_output", "")))
        corr_path = Path(str(correlated.get("trajectory_output", "")))
        if not iid_path.is_file() or not corr_path.is_file():
            failures.append(
                {
                    **identity,
                    "reason": "trajectory_npz_missing",
                    "iid_path": str(iid_path),
                    "correlated_path": str(corr_path),
                }
            )
            continue
        try:
            pair_prefix = corr_path.with_name("paired_delta")
            corr_arrays = load_trajectory_npz(corr_path)
            iid_arrays = load_trajectory_npz(iid_path)
            expected_rank = int(correlated["rank"])
            corr_rank = int(np.asarray(corr_arrays.get("basis_rank", -1)).item())
            iid_rank = int(np.asarray(iid_arrays.get("basis_rank", -1)).item())
            if corr_rank != expected_rank or iid_rank != expected_rank:
                raise ValueError(
                    "trajectory artifact ranks ({}, {}) do not match planned rank {}".format(
                        iid_rank, corr_rank, expected_rank
                    )
                )
            if str(np.asarray(corr_arrays.get("basis_checksum", "")).item()) != str(
                np.asarray(iid_arrays.get("basis_checksum", "")).item()
            ):
                raise ValueError("paired trajectories use different observer bases")
            paired_paths = save_paired_delta(
                corr_arrays, iid_arrays, pair_prefix, make_plot=True
            )
            missing_paired_artifacts = [
                key
                for key in ("npz", "csv", "plot")
                if paired_paths.get(key) is None
                or not Path(str(paired_paths[key])).is_file()
            ]
            if missing_paired_artifacts:
                raise RuntimeError(
                    "paired trajectory artifacts missing: {}".format(
                        missing_paired_artifacts
                    )
                )
            with np.load(pair_prefix.with_suffix(".npz"), allow_pickle=False) as archive:
                delta_t = archive["delta_t"].copy()
            mean_g = np.nanmean(corr_arrays["g_t"], axis=1)
            mean_delta = np.nanmean(delta_t, axis=1)
            milestones = [str(item) for item in corr_arrays["milestones"].tolist()]
            target_progress = [float(item) for item in corr_arrays["target_progress"]]
            heatmap_path = _save_pair_view_correlation_heatmaps(
                iid_arrays,
                corr_arrays,
                milestones,
                corr_path.with_name("view_correlation_heatmaps.png"),
            )
            if not heatmap_path.is_file() or heatmap_path.stat().st_size <= 0:
                raise RuntimeError("view-correlation heatmap was not written")
            row = {
                **identity,
                "method": correlated.get("method"),
                "rank": correlated.get("rank"),
                "target_kl": correlated.get("target_kl"),
                "diagnostic_only": bool(correlated.get("diagnostic_only", False)),
                "milestones": milestones,
                "target_progress": target_progress,
                "mean_g_t": [float(item) for item in mean_g],
                "mean_delta_t": [float(item) for item in mean_delta],
                "final_g_t": float(mean_g[-1]) if np.isfinite(mean_g[-1]) else None,
                "final_delta_t": (
                    float(mean_delta[-1]) if np.isfinite(mean_delta[-1]) else None
                ),
                "iid_trajectory": str(iid_path),
                "correlated_trajectory": str(corr_path),
                "paired_delta": {
                    key: str(value) if value is not None else None
                    for key, value in paired_paths.items()
                },
                "visualizations": {
                    "complete": True,
                    "view_correlation_heatmaps": str(heatmap_path),
                    "paired_delta_plot": str(paired_paths["plot"]),
                },
            }
            pairs.append(row)
            for index, milestone in enumerate(milestones):
                csv_rows.append(
                    {
                        **identity,
                        "method": correlated.get("method"),
                        "rank": correlated.get("rank"),
                        "target_kl": correlated.get("target_kl"),
                        "milestone": milestone,
                        "target_progress": target_progress[index],
                        "mean_g_t": float(mean_g[index]),
                        "mean_delta_t": float(mean_delta[index]),
                    }
                )
        except Exception as error:
            failures.append({**identity, "reason": "pairing_failed", "error": repr(error)})

    finite_final_g = [
        float(item["final_g_t"])
        for item in pairs
        if item.get("final_g_t") is not None and math.isfinite(float(item["final_g_t"]))
    ]
    if not finite_final_g:
        correlation_state = "not_available"
    else:
        aggregate_final_g = float(np.mean(finite_final_g))
        if aggregate_final_g < washout_max_final_g:
            correlation_state = "wash_out"
        elif aggregate_final_g > amplified_min_final_g:
            correlation_state = "amplified"
        else:
            correlation_state = "retained"
    method_summaries: List[Dict[str, Any]] = []
    by_method: MutableMapping[str, List[Mapping[str, Any]]] = {}
    for pair in pairs:
        by_method.setdefault(str(pair.get("method")), []).append(pair)
    for method, method_pairs in sorted(by_method.items()):
        milestones = list(method_pairs[0]["milestones"])
        progress = list(method_pairs[0]["target_progress"])
        if any(
            list(item["milestones"]) != milestones
            or list(item["target_progress"]) != progress
            for item in method_pairs
        ):
            failures.append({"method": method, "reason": "method_milestones_mismatch"})
            continue
        mean_g = np.mean(
            np.asarray([item["mean_g_t"] for item in method_pairs], dtype=np.float64),
            axis=0,
        )
        mean_delta = np.mean(
            np.asarray([item["mean_delta_t"] for item in method_pairs], dtype=np.float64),
            axis=0,
        )
        method_summaries.append(
            {
                "method": method,
                "pair_count": len(method_pairs),
                "rank": method_pairs[0].get("rank"),
                "milestones": milestones,
                "target_progress": progress,
                "mean_g_t": [float(item) for item in mean_g],
                "mean_delta_t": [float(item) for item in mean_delta],
                "final_g_t": float(mean_g[-1]),
                "final_delta_t": float(mean_delta[-1]),
            }
        )
    if unpaired_manifest_records:
        failures.append(
            {
                "reason": "manifest_records_without_pair_id",
                "run_ids": unpaired_manifest_records,
            }
        )
    summary_csv: Optional[Path] = None
    if csv_rows:
        temporary = csv_path.with_name(csv_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        os.replace(temporary, csv_path)
        if csv_path.is_file() and csv_path.stat().st_size > 0:
            summary_csv = csv_path

    aggregate_plot: Optional[Path] = None
    if pairs and method_summaries:
        try:
            aggregate_plot = _save_aggregate_trajectory_plot(
                pairs, method_summaries, aggregate_plot_path
            )
            if (
                not aggregate_plot.is_file()
                or aggregate_plot.stat().st_size <= 0
            ):
                raise RuntimeError("aggregate trajectory plot was not written")
        except Exception as error:
            aggregate_plot = None
            failures.append(
                {"reason": "aggregate_trajectory_plot_failed", "error": repr(error)}
            )

    pair_heatmaps = [
        Path(str(item.get("visualizations", {}).get("view_correlation_heatmaps")))
        for item in pairs
        if item.get("visualizations", {}).get("view_correlation_heatmaps")
    ]
    valid_pair_heatmaps = [
        path for path in pair_heatmaps if path.is_file() and path.stat().st_size > 0
    ]
    visualization_complete = bool(pairs) and (
        len(valid_pair_heatmaps) == len(pairs)
        and aggregate_plot is not None
        and aggregate_plot.is_file()
        and aggregate_plot.stat().st_size > 0
    )
    summary_csv_complete = (
        summary_csv is not None
        and summary_csv.is_file()
        and summary_csv.stat().st_size > 0
    )
    payload = {
        "schema_version": 2,
        "complete": (
            bool(groups)
            and len(pairs) == len(groups)
            and not failures
            and visualization_complete
            and summary_csv_complete
        ),
        "visualization_complete": visualization_complete,
        "visualization_audit": {
            "expected_pair_heatmap_count": len(pairs),
            "pair_heatmap_count": len(valid_pair_heatmaps),
            "aggregate_plot_complete": aggregate_plot is not None,
        },
        "correlation_state": correlation_state,
        "classification_thresholds": {
            "washout_max_final_g": washout_max_final_g,
            "amplified_min_final_g": amplified_min_final_g,
        },
        "expected_pair_count": len(groups),
        "pair_count": len(pairs),
        "failure_count": len(failures),
        "aggregate_final_g_t": (
            float(np.mean(finite_final_g)) if finite_final_g else None
        ),
        "failures": failures,
        "pairs": pairs,
        "method_summaries": method_summaries,
        "artifacts": {
            "summary_json": str(output_path),
            "summary_csv": str(summary_csv) if summary_csv is not None else None,
            "aggregate_g_delta_plot": (
                str(aggregate_plot) if aggregate_plot is not None else None
            ),
            "pair_view_correlation_heatmaps": [
                str(path) for path in valid_pair_heatmaps
            ],
        },
    }
    _atomic_json(output_path, payload)
    return payload


def _load_inputs(root: Path, split: str) -> List[Dict[str, Any]]:
    return _read_input_records(root / "inputs" / "input_validation.json", split)


def audit_input_split_isolation(
    root: Path, config: Mapping[str, Any], *, stage: str
) -> Dict[str, Any]:
    """Prove frozen PILOT/FULL membership cannot leak across manifests."""

    if stage not in {"full", "trajectory"}:
        raise ValueError("input split isolation audit is only for full/trajectory")
    reasons: List[str] = []
    frozen_path = root / "inputs" / "input_validation.json"
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        records = frozen.get("records", [])
        if not isinstance(records, list):
            raise ValueError("frozen input records are not a list")
    except Exception as error:
        return {
            "ready": False,
            "stage": stage,
            "reasons": ["frozen_input_manifest_invalid"],
            "error": repr(error),
            "frozen_path": str(frozen_path),
        }

    def split_values(name: str) -> List[str]:
        return [
            str(item.get("sha256"))
            for item in records
            if isinstance(item, Mapping)
            and item.get("split") == name
            and item.get("sha256")
        ]

    pilot_values = split_values("pilot")
    full_values = split_values("full")
    pilot_shas = set(pilot_values)
    full_shas = set(full_values)
    expected_pilot = int(config["data"]["pilot_count"])
    expected_full = int(config["data"]["full_count"])
    if len(pilot_values) != expected_pilot or len(pilot_shas) != expected_pilot:
        reasons.append("frozen_pilot_count_mismatch")
    if len(full_values) != expected_full or len(full_shas) != expected_full:
        reasons.append("frozen_full_count_mismatch")
    frozen_overlap = sorted(pilot_shas.intersection(full_shas))
    if frozen_overlap:
        reasons.append("frozen_pilot_full_overlap")

    manifest_sets: Dict[str, set] = {}
    manifest_counts: Dict[str, Optional[int]] = {}
    for split in ("pilot", "full"):
        manifest_path = root / split / "manifest.json"
        required = split == "pilot" or stage == "trajectory"
        if not manifest_path.is_file():
            manifest_sets[split] = set()
            manifest_counts[split] = None
            if required:
                reasons.append("{}_manifest_missing".format(split))
            continue
        try:
            manifest_records = _read_manifest(manifest_path)
        except Exception as error:
            manifest_sets[split] = set()
            manifest_counts[split] = None
            reasons.append("{}_manifest_invalid".format(split))
            continue
        values = [
            str(item.get("input_sha256"))
            for item in manifest_records
            if item.get("input_sha256")
        ]
        if len(values) != len(manifest_records):
            reasons.append("{}_manifest_input_sha_missing".format(split))
        observed = set(values)
        manifest_sets[split] = observed
        manifest_counts[split] = len(observed)
        expected = pilot_shas if split == "pilot" else full_shas
        if observed != expected:
            reasons.append("{}_manifest_split_mismatch".format(split))

    pilot_manifest = manifest_sets.get("pilot", set())
    full_manifest = manifest_sets.get("full", set())
    if pilot_manifest.intersection(full_shas):
        reasons.append("pilot_manifest_contains_full_inputs")
    if full_manifest.intersection(pilot_shas):
        reasons.append("full_manifest_contains_pilot_inputs")
    if pilot_manifest.intersection(full_manifest):
        reasons.append("pilot_full_manifest_overlap")
    return {
        "ready": not reasons,
        "stage": stage,
        "reasons": sorted(set(reasons)),
        "frozen_path": str(frozen_path),
        "expected_pilot_count": expected_pilot,
        "expected_full_count": expected_full,
        "frozen_pilot_count": len(pilot_shas),
        "frozen_full_count": len(full_shas),
        "pilot_manifest_input_count": manifest_counts.get("pilot"),
        "full_manifest_input_count": manifest_counts.get("full"),
        "frozen_overlap": frozen_overlap,
    }


def _run_report(root: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "scripts.report_nile_lowrank_study",
        "--artifact-root",
        str(root),
    ]
    process = subprocess.run(command, check=False)
    return {
        "completed": process.returncode == 0 and (root / "FINAL_STATUS.json").exists(),
        "returncode": process.returncode,
        "report": str(root / "FULL_EXPERIMENT_REPORT.md"),
        "final_status": str(root / "FINAL_STATUS.json"),
    }


def audit_implementation_files(paths: Sequence[Path]) -> Dict[str, Any]:
    missing: List[str] = []
    invalid: List[Dict[str, str]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(str(path))
            continue
        try:
            if path.suffix == ".py":
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            elif path.suffix == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(notebook.get("cells"), list) or not notebook.get("cells"):
                    raise ValueError("notebook has no cells")
                if int(notebook.get("nbformat", 0)) < 4:
                    raise ValueError("unsupported notebook format")
        except Exception as error:
            invalid.append({"path": str(path), "error": repr(error)})
    return {
        "complete": not missing and not invalid,
        "required_count": len(paths),
        "missing": missing,
        "invalid": invalid,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nile_lowrank_full.yaml")
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    if args.preflight_only:
        args.stage = "preflight"
    requested = load_config(args.config.expanduser().resolve())
    config = resolve_config(
        requested, input_dir=args.input_dir, output_root=args.output_root
    )
    if not args.resume:
        args.resume = bool(config.get("experiment", {}).get("resume", True))
    root = experiment_root(config)
    root.mkdir(parents=True, exist_ok=True)
    config_hash = lock_resolved_config(root, config)
    repo_root = Path(__file__).resolve().parents[1]
    environment = capture_environment(root, repo_root)
    status = load_runtime_status(root, config_hash)
    required_files = (
        repo_root / "mvadapter" / "nile" / "basis.py",
        repo_root / "mvadapter" / "nile" / "covariance.py",
        repo_root / "mvadapter" / "nile" / "lowrank_coupling.py",
        repo_root / "mvadapter" / "nile" / "diagnostics.py",
        repo_root / "mvadapter" / "nile" / "trajectory.py",
        repo_root / "scripts" / "diagnose_nile_lowrank.py",
        repo_root / "scripts" / "nile_lowrank_inference_worker.py",
        repo_root / "scripts" / "run_nile_lowrank_study.py",
        repo_root / "scripts" / "select_nile_lowrank_candidates.py",
        repo_root / "scripts" / "validate_nile_inputs.py",
        repo_root / "scripts" / "eval_nile_lowrank_study.py",
        repo_root / "scripts" / "report_nile_lowrank_study.py",
        repo_root / "notebooks" / "mvadapter_nile_lowrank_full_colab.ipynb",
    )
    implementation_audit = audit_implementation_files(required_files)
    status["implementation_complete"] = bool(implementation_audit["complete"])
    status.setdefault("stages", {})["implementation"] = implementation_audit
    test_results_path = root / "environment" / "test_results.json"
    test_results_audit = audit_test_results_receipt(test_results_path)
    status["tests_complete"] = bool(test_results_audit["verified"])
    status.setdefault("stages", {})["tests"] = test_results_audit
    checkpoint_manifest_path = root / "configs" / "checkpoint_manifest.json"
    checkpoint_audit = audit_checkpoint_manifest(
        checkpoint_manifest_path,
        config,
        cache_path=root / "environment" / "checkpoint_audit_cache.json",
        config_hash=config_hash,
    )
    status.setdefault("stages", {})["checkpoint_provenance"] = checkpoint_audit

    input_validation = validate_inputs_stage(root, config, args.input_dir)
    update_runtime_status(
        root,
        status,
        "inputs",
        {key: value for key, value in input_validation.items() if key != "records"},
    )
    blockers = _formal_blockers(
        config,
        input_validation,
        environment,
        test_results_audit=test_results_audit,
        checkpoint_audit=checkpoint_audit,
    )
    previous_blockers = [
        dict(item)
        for item in status.get("blockers", [])
        if isinstance(item, Mapping)
    ]
    if previous_blockers and previous_blockers != blockers:
        history_entry = {
            "observed_at": _utc_now(),
            "blockers": previous_blockers,
        }
        if history_entry not in status.setdefault("blocker_history", []):
            status["blocker_history"].append(history_entry)
    status["blockers"] = [dict(item) for item in blockers]
    _atomic_json(_status_path(root), status)

    stages = [args.stage] if args.stage != "all" else [
        "preflight",
        "pilot",
        "evaluate_pilot",
        "select",
        "full",
        "trajectory",
        "evaluate_full",
        "report",
    ]
    for stage in stages:
        if stage == "preflight":
            payload = run_preflight_stage(root, config)
            update_runtime_status(root, status, "preflight", payload)
        elif stage == "pilot":
            if blockers:
                update_runtime_status(
                    root, status, "pilot", {"completed": False, "reason": "formal_blockers", "blockers": blockers}
                )
                continue
            try:
                pilot_configurations = gated_pilot_configurations(root, config)
            except Exception as error:
                payload = {
                    "completed": False,
                    "reason": "preflight_gate_prerequisite_failed",
                    "error": repr(error),
                }
                update_runtime_status(root, status, "pilot", payload)
                continue
            payload = _run_generation_split(
                root=root,
                split="pilot",
                config=config,
                config_hash=config_hash,
                environment=environment,
                configurations=pilot_configurations,
                seeds=config["pilot"]["seeds"],
                inputs=_load_inputs(root, "pilot"),
                args=args,
            )
            payload["completed"] = bool(
                payload.get("planned", 0) > 0
                and payload.get("failed", 0) == 0
                and payload.get("succeeded", 0) == payload.get("planned", -1)
                and payload.get("artifact_complete", -1) == payload.get("planned", -2)
            )
            status["pilot_complete"] = bool(payload["completed"] and not args.dry_run)
            update_runtime_status(root, status, "pilot", payload)
        elif stage == "evaluate_pilot":
            payload = run_evaluation_stage(root, "pilot", config)
            update_runtime_status(root, status, "evaluate_pilot", payload)
        elif stage == "select":
            payload = run_selection_stage(root, config)
            update_runtime_status(root, status, "select", payload)
        elif stage == "full":
            selection_path = root / "selected_candidates" / "selected_candidates.json"
            pilot_readiness = audit_pilot_met3r_prerequisite(root, config)
            input_isolation = audit_input_split_isolation(
                root, config, stage="full"
            )
            selection_audit = audit_selected_candidates(
                root, config, config_hash=config_hash
            )
            selection_blocker = (
                {
                    "code": "selected_candidates_audit_failed",
                    "issues": selection_audit["issues"],
                    "selected_candidates": str(selection_path),
                }
                if not selection_audit["ready"]
                else None
            )
            selected_payload: Dict[str, Any] = {}
            if selection_audit["ready"]:
                try:
                    selected_payload = json.loads(selection_path.read_text(encoding="utf-8"))
                except Exception:
                    selected_payload = {}
            if (
                blockers
                or not selection_audit["ready"]
                or not pilot_readiness["ready"]
                or not input_isolation["ready"]
            ):
                payload = {
                    "completed": False,
                    "reason": "full_prerequisites_missing",
                    "blockers": blockers,
                    "selection_exists": selection_path.exists(),
                    "selected_candidates_audit": selection_audit,
                    "pilot_met3r": pilot_readiness,
                    "input_split_isolation": input_isolation,
                }
            else:
                payload = _run_generation_split(
                    root=root,
                    split="full",
                    config=config,
                    config_hash=config_hash,
                    environment=environment,
                    configurations=build_full_configurations(selected_payload),
                    seeds=config["full"]["seeds"],
                    inputs=_load_inputs(root, "full"),
                    args=args,
                )
                payload["completed"] = bool(
                    payload.get("planned", 0) > 0
                    and payload.get("failed", 0) == 0
                    and payload.get("succeeded", 0) == payload.get("planned", -1)
                    and payload.get("artifact_complete", -1)
                    == payload.get("planned", -2)
                )
            status["full_complete"] = bool(payload.get("completed") and not args.dry_run)
            update_runtime_status(
                root, status, "full", payload, blocker=selection_blocker
            )
        elif stage == "trajectory":
            selection_path = root / "selected_candidates" / "selected_candidates.json"
            input_isolation = audit_input_split_isolation(
                root, config, stage="trajectory"
            )
            selection_audit = audit_selected_candidates(
                root, config, config_hash=config_hash
            )
            selection_blocker = (
                {
                    "code": "selected_candidates_audit_failed",
                    "issues": selection_audit["issues"],
                    "selected_candidates": str(selection_path),
                }
                if not selection_audit["ready"]
                else None
            )
            if (
                blockers
                or not selection_audit["ready"]
                or not input_isolation["ready"]
            ):
                payload = {
                    "completed": False,
                    "reason": "trajectory_prerequisites_missing",
                    "blockers": blockers,
                    "selected_candidates_audit": selection_audit,
                    "input_split_isolation": input_isolation,
                }
            else:
                selected = json.loads(selection_path.read_text(encoding="utf-8"))
                configs = build_trajectory_configurations(selected)
                count = int(config["trajectory"].get("input_count", 2))
                payload = _run_generation_split(
                    root=root,
                    split="trajectory",
                    config=config,
                    config_hash=config_hash,
                    environment=environment,
                    configurations=configs,
                    seeds=config["trajectory"]["seeds"],
                    inputs=_load_inputs(root, "full")[:count],
                    args=args,
                )
                generation_complete = (
                    payload.get("planned", 0) > 0
                    and payload.get("failed", 0) == 0
                    and payload.get("succeeded", 0) == payload.get("planned", -1)
                    and payload.get("artifact_complete", -1)
                    == payload.get("planned", -2)
                    and not args.dry_run
                )
                if generation_complete:
                    thresholds = config["trajectory"]
                    summary = summarize_trajectory_stage(
                        root,
                        config_hash,
                        washout_max_final_g=float(
                            thresholds.get("washout_max_final_g", 0.5)
                        ),
                        amplified_min_final_g=float(
                            thresholds.get("amplified_min_final_g", 1.5)
                        ),
                    )
                else:
                    summary = {
                        "complete": False,
                        "correlation_state": "not_available",
                        "reason": "trajectory_generation_incomplete",
                    }
                payload["trajectory_summary"] = summary
                payload["completed"] = bool(generation_complete and summary.get("complete"))
            status["trajectory_complete"] = bool(payload.get("completed"))
            update_runtime_status(
                root, status, "trajectory", payload, blocker=selection_blocker
            )
        elif stage in {"evaluate_full"}:
            payload = run_evaluation_stage(root, "full", config)
            status["met3r_complete"] = bool(payload.get("met3r_complete"))
            update_runtime_status(root, status, "evaluate_full", payload)
        elif stage == "evaluate":
            for split in ("pilot", "full"):
                payload = run_evaluation_stage(root, split, config)
                if split == "full":
                    status["met3r_complete"] = bool(payload.get("met3r_complete"))
                update_runtime_status(root, status, "evaluate_" + split, payload)
        elif stage == "report":
            payload = _run_report(root)
            status["report_complete"] = bool(payload.get("completed"))
            update_runtime_status(root, status, "report", payload)
        else:
            raise AssertionError("unhandled stage: {}".format(stage))

    _atomic_json(_status_path(root), status)
    print(json.dumps({"artifact_root": str(root), "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
