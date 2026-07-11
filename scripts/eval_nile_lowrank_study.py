"""Evaluate low-rank NILE-inspired PILOT/FULL runs and paired statistics.

MEt3R remains the primary multiview metric. Identity and silhouette metrics
are guardrails; lightweight pixel metrics are collapse diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (_finite(item) for item in values) if value is not None]
    return float(np.mean(finite)) if finite else None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    return [dict(item) for item in payload.get("runs", [])]


CANONICAL_RUN_FIELDS = (
    "experiment_id",
    "code_revision",
    "input_image",
    "input_sha256",
    "method",
    "seed",
    "config_id",
    "rank",
    "target_kl",
    "achieved_kl",
    "alpha",
    "rbf_length_scale_deg",
    "basis_checksum",
    "covariance_checksum",
    "distribution_gate_passed",
    "selection_status",
    "diagnostic_only",
)


def reconcile_manifest_samples(
    manifest_rows: Sequence[Mapping[str, Any]],
    evaluated_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Reconcile evaluation rows against every planned manifest record.

    This prevents a missing artifact from disappearing from configuration
    summaries merely because the base evaluator could not discover it.
    """

    evaluated_by_id: Dict[str, List[Mapping[str, Any]]] = {}
    for row in evaluated_rows:
        evaluated_by_id.setdefault(str(row.get("sample_id")), []).append(row)
    reconciled: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    duplicate_ids: List[str] = []
    conflict_ids: List[str] = []
    expected_ids = {str(record.get("run_id")) for record in manifest_rows}
    for record in manifest_rows:
        run_id = str(record.get("run_id"))
        matches = evaluated_by_id.get(run_id, [])
        if len(matches) == 1:
            row = dict(matches[0])
        else:
            row = {
                "sample_id": run_id,
                "source": str(record.get("metadata_path", "")),
                "status": "missing_evaluation" if not matches else "duplicate_evaluation",
            }
            if not matches:
                missing_ids.append(run_id)
            else:
                duplicate_ids.append(run_id)
        row["generation_status"] = record.get("status")
        row["run_id"] = run_id
        config_conflicts = dict(row.get("metadata_config_conflicts") or {})
        for field in CANONICAL_RUN_FIELDS:
            if field not in record:
                continue
            if field in row and row[field] != record[field]:
                config_conflicts[field] = {
                    "manifest": record[field],
                    "evaluation": row[field],
                }
            row[field] = record[field]
        if config_conflicts:
            row["metadata_config_conflicts"] = config_conflicts
            conflict_ids.append(run_id)
        if record.get("status") != "succeeded":
            row["status"] = "generation_{}".format(record.get("status", "unknown"))
        reconciled.append(row)
    extra_ids = sorted(
        run_id for run_id in evaluated_by_id if run_id not in expected_ids
    )
    audit = {
        "manifest_run_count": len(manifest_rows),
        "manifest_succeeded_count": sum(
            row.get("status") == "succeeded" for row in manifest_rows
        ),
        "evaluated_row_count": len(evaluated_rows),
        "missing_evaluation_run_ids": sorted(missing_ids),
        "duplicate_evaluation_run_ids": sorted(duplicate_ids),
        "unexpected_evaluation_run_ids": extra_ids,
        "metadata_conflict_run_ids": sorted(set(conflict_ids)),
    }
    return reconciled, audit


def assess_evaluation_completeness(
    rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    met3r_required: bool,
    identity_required: bool,
) -> Dict[str, Any]:
    expected_views = {
        str(record.get("run_id")): len(record.get("camera_list", []))
        for record in manifest_rows
    }
    issues: List[Dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id", row.get("sample_id")))
        reasons: List[str] = []
        if row.get("generation_status") != "succeeded":
            reasons.append("generation_not_succeeded")
        if row.get("status") != "succeeded":
            reasons.append("evaluation_not_succeeded")
        if row.get("metadata_config_conflicts"):
            reasons.append("metadata_config_conflict")
        if met3r_required and _finite(row.get("angle_all_met3r_score")) is None:
            reasons.append("met3r_missing")
        if identity_required and _finite(row.get("dino_reference_mean")) is None:
            reasons.append("identity_missing")
        view_count = expected_views.get(run_id, 0)
        if view_count <= 0 or int(row.get("num_views", 0)) != view_count:
            reasons.append("view_count_incomplete")
        if int(row.get("mask_view_count", 0)) != view_count:
            reasons.append("mask_count_incomplete")
        if int(row.get("mask_failure_count", 0)) != 0:
            reasons.append("mask_read_failure")
        if row.get("guardrail_error"):
            reasons.append("guardrail_error")
        if reasons:
            issues.append({"run_id": run_id, "reasons": sorted(set(reasons))})
    return {
        "complete": bool(rows) and len(rows) == len(manifest_rows) and not issues,
        "expected_run_count": len(manifest_rows),
        "audited_run_count": len(rows),
        "issue_count": len(issues),
        "issues": issues,
    }


def connected_component_areas(mask: np.ndarray) -> List[int]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    try:
        import cv2

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=4
        )
        return sorted(
            (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)),
            reverse=True,
        )
    except (ImportError, ModuleNotFoundError):
        pass
    visited = np.zeros(mask.shape, dtype=bool)
    areas = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask & ~visited)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        area = 0
        while queue:
            current_y, current_x = queue.popleft()
            area += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_y, next_x = current_y + dy, current_x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        areas.append(area)
    return sorted(areas, reverse=True)


def silhouette_metrics(mask: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    total_pixels = int(mask.size)
    foreground = int(mask.sum())
    areas = connected_component_areas(mask)
    if foreground == 0:
        return {
            "connected_component_count": 0,
            "small_component_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "foreground_area": 0.0,
            "silhouette_compactness": None,
            "boundary_high_curvature_proxy": None,
            "artifact_failure": True,
        }
    small_limit = max(4, int(round(0.01 * foreground)))
    small_area = sum(area for area in areas[1:] if area <= small_limit)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[:-2, 1:-1].astype(np.int8)
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    boundary = mask & (neighbors < 4)
    perimeter = int(boundary.sum())
    high_curvature = boundary & (neighbors <= 2)
    small_ratio = float(small_area / foreground)
    largest_ratio = float(areas[0] / foreground)
    return {
        "connected_component_count": len(areas),
        "small_component_ratio": small_ratio,
        "largest_component_ratio": largest_ratio,
        "foreground_area": float(foreground / total_pixels),
        "silhouette_compactness": float(perimeter * perimeter / foreground),
        "boundary_high_curvature_proxy": float(high_curvature.sum() / max(perimeter, 1)),
        "artifact_failure": bool(
            len(areas) > 3 or small_ratio > 0.02 or largest_ratio < 0.90
        ),
    }


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if "A" in image.getbands():
            array = np.asarray(image.getchannel("A"), dtype=np.uint8)
        else:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array >= 128


def aggregate_mask_metrics(paths: Sequence[Path]) -> Dict[str, Any]:
    rows = []
    failures = []
    for path in paths:
        try:
            rows.append(silhouette_metrics(_load_mask(path)))
        except Exception as error:
            failures.append({"path": str(path), "error": repr(error)})
    areas = [row["foreground_area"] for row in rows]
    mean_area = float(np.mean(areas)) if areas else None
    area_cv = (
        float(np.std(areas, ddof=0) / mean_area)
        if mean_area is not None and mean_area > 0.0
        else None
    )
    return {
        "mask_view_count": len(rows),
        "mask_failure_count": len(failures),
        "connected_component_count_mean": _mean(
            row["connected_component_count"] for row in rows
        ),
        "small_component_ratio": _mean(row["small_component_ratio"] for row in rows),
        "largest_component_ratio": _mean(row["largest_component_ratio"] for row in rows),
        "foreground_area": mean_area,
        "foreground_area_cv": area_cv,
        "silhouette_compactness": _mean(row["silhouette_compactness"] for row in rows),
        "boundary_high_curvature_proxy": _mean(
            row["boundary_high_curvature_proxy"] for row in rows
        ),
        "component_failure_rate": (
            float(np.mean([row["artifact_failure"] for row in rows])) if rows else None
        ),
        "mask_failures": failures,
    }


class DinoIdentityBackend:
    def __init__(
        self, model_name: str, device: str, revision: Optional[str] = None
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as error:
            raise RuntimeError("DINO identity dependencies are unavailable") from error
        self.torch = torch
        self.device = device
        revision_kwargs = {"revision": revision} if revision else {}
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, **revision_kwargs
        )
        self.model = AutoModel.from_pretrained(
            model_name, **revision_kwargs
        ).to(device).eval()

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model(**inputs)
            features = output.last_hidden_state[:, 0]
            features = self.torch.nn.functional.normalize(features, dim=-1)
        return features.detach().cpu().float().numpy()

    def evaluate(self, reference: Path, views: Sequence[Path]) -> Dict[str, Any]:
        embeddings = self.encode([reference, *views])
        similarities = embeddings[1:] @ embeddings[0]
        return {
            "dino_reference_view_scores": [float(item) for item in similarities],
            "dino_reference_front": float(similarities[0]),
            "dino_reference_mean": float(np.mean(similarities)),
        }


def verify_installed_met3r_revision(expected_revision: Optional[str]) -> Dict[str, Any]:
    """Verify the PEP 610 commit for the installed official MEt3R package."""

    if not expected_revision:
        raise RuntimeError(
            "--met3r-revision is required when --metrics=all for a formal run"
        )
    try:
        distribution = importlib_metadata.distribution("met3r")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError("the met3r distribution is not installed") from error
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(
            "installed met3r has no direct_url.json; VCS commit provenance is unavailable"
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError("installed met3r direct_url.json is invalid") from error
    vcs_info = direct_url.get("vcs_info")
    commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
    if not commit_id:
        raise RuntimeError(
            "installed met3r direct_url.json does not record a VCS commit_id"
        )
    expected = str(expected_revision).lower()
    observed = str(commit_id).lower()
    if observed != expected:
        raise RuntimeError(
            "installed met3r commit {} does not match frozen revision {}".format(
                observed, expected
            )
        )
    return {
        "distribution": distribution.metadata.get("Name", "met3r"),
        "version": distribution.version,
        "expected_revision": expected,
        "installed_commit_id": observed,
        "direct_url": direct_url.get("url"),
        "verified": True,
    }

def augment_sample_metrics(
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    identity_backend: Optional[DinoIdentityBackend],
) -> List[Dict[str, Any]]:
    augmented = []
    for sample in sample_rows:
        row = dict(sample)
        source = Path(str(row.get("source", "")))
        try:
            metadata = json.loads(source.read_text(encoding="utf-8"))
            base = source.parent
            reference = Path(metadata["reference_output"])
            if not reference.is_absolute():
                reference = (base / reference).resolve()
            view_paths = [Path(item) for item in metadata.get("view_files", [])]
            view_paths = [item if item.is_absolute() else (base / item).resolve() for item in view_paths]
            mask_paths = [Path(item) for item in metadata.get("mask_files", [])]
            mask_paths = [item if item.is_absolute() else (base / item).resolve() for item in mask_paths]
            if not reference.is_file():
                raise FileNotFoundError("reference image is missing: {}".format(reference))
            missing_views = [str(item) for item in view_paths if not item.is_file()]
            if not view_paths or missing_views:
                raise FileNotFoundError(
                    "view artifacts are incomplete: {}".format(missing_views or "none listed")
                )
            row["reference_exists"] = True
            row["view_artifact_count"] = len(view_paths)
            if identity_backend is not None:
                row.update(identity_backend.evaluate(reference, view_paths))
            else:
                row["identity_status"] = "backend_unavailable"
            if mask_paths:
                row.update(aggregate_mask_metrics(mask_paths))
            else:
                row["mask_status"] = "missing"
        except Exception as error:
            row["guardrail_error"] = repr(error)
        augmented.append(row)
    return augmented


PAIR_FIELDS = (
    "experiment_id",
    "code_revision",
    "input_sha256",
    "input_image",
    "seed",
)


def _pair_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(row.get(field) for field in PAIR_FIELDS)


def _baseline_groups(
    rows: Sequence[Mapping[str, Any]], method: str
) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("method") == method and row.get("status") == "succeeded":
            groups.setdefault(_pair_key(row), []).append(row)
    return groups


def add_paired_deltas(
    rows: Sequence[Mapping[str, Any]], baseline_method: str = "iid_external"
) -> List[Dict[str, Any]]:
    baselines = _baseline_groups(rows, baseline_method)
    result = []
    fields = (
        ("dino_reference_mean", "dino_identity_mean_delta"),
        ("small_component_ratio", "small_component_ratio_delta"),
        ("component_failure_rate", "component_failure_rate_delta"),
        ("foreground_area_cv", "foreground_area_cv_delta"),
        ("angle_all_met3r_score", "met3r_paired_delta"),
    )
    for original in rows:
        row = dict(original)
        matches = baselines.get(_pair_key(row), [])
        baseline = matches[0] if len(matches) == 1 else None
        row["iid_pair_status"] = (
            "paired" if len(matches) == 1 else "missing" if not matches else "ambiguous"
        )
        for field, output in fields:
            value = _finite(row.get(field))
            reference = _finite(baseline.get(field)) if baseline else None
            row[output] = value - reference if value is not None and reference is not None else None
        result.append(row)
    return result


def add_metric_comparison_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    target_methods: Sequence[str],
    metric_field: str = "angle_all_met3r_score",
    output_field: str = "comparison_delta",
) -> List[Dict[str, Any]]:
    """Return strict input/seed paired target-minus-baseline metric rows."""

    baselines = _baseline_groups(rows, baseline_method)
    output = []
    targets = set(target_methods)
    for original in rows:
        if original.get("method") not in targets:
            continue
        row = dict(original)
        matches = baselines.get(_pair_key(row), [])
        baseline = matches[0] if len(matches) == 1 else None
        value = _finite(row.get(metric_field))
        reference = _finite(baseline.get(metric_field)) if baseline else None
        row[output_field] = (
            value - reference
            if value is not None and reference is not None
            else None
        )
        row["comparison_baseline"] = baseline_method
        row["comparison_pair_status"] = (
            "paired" if len(matches) == 1 else "missing" if not matches else "ambiguous"
        )
        output.append(row)
    return output


def configuration_summaries(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    keys = (
        "config_id",
        "method",
        "rank",
        "target_kl",
        "achieved_kl",
        "alpha",
        "rbf_length_scale_deg",
        "basis_checksum",
        "covariance_checksum",
        "selection_status",
        "diagnostic_only",
    )
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        successful = [row for row in members if row.get("status") == "succeeded"]
        met3r = [_finite(row.get("angle_all_met3r_score")) for row in successful]
        met3r = [value for value in met3r if value is not None]
        summary = dict(zip(keys, key))
        summary.update(
            {
                "sample_count": len(members),
                "successful_sample_count": len(successful),
                "output_missing": len(successful) != len(members),
                "generation_failed_count": sum(
                    row.get("generation_status", row.get("status")) != "succeeded"
                    for row in members
                ),
                "distribution_gate_passed": all(
                    bool(row.get("distribution_gate_passed", row.get("preflight_passed", False)))
                    for row in members
                ),
                "met3r_all_pair_mean": float(np.mean(met3r)) if met3r else None,
                "met3r_standard_error": (
                    float(np.std(met3r, ddof=1) / math.sqrt(len(met3r)))
                    if len(met3r) > 1
                    else 0.0 if len(met3r) == 1 else None
                ),
                "met3r_failure_rate": 1.0 - len(met3r) / max(len(members), 1),
                "dino_identity_mean_delta": _mean(row.get("dino_identity_mean_delta") for row in successful),
                "small_component_ratio_delta": _mean(row.get("small_component_ratio_delta") for row in successful),
                "component_failure_rate_delta": _mean(row.get("component_failure_rate_delta") for row in successful),
                "foreground_area_cv_delta": _mean(row.get("foreground_area_cv_delta") for row in successful),
                "r_hf": _mean(row.get("r_hf") for row in successful),
                "collapse_detector_label": (
                    "view_collapse_alert"
                    if any(row.get("collapse_detector_label") == "view_collapse_alert" for row in successful)
                    else "no_collapse_signal"
                ),
            }
        )
        output.append(summary)
    return output


def cluster_bootstrap_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    iterations: int = 10000,
    seed: int = 20260711,
    cluster_field: str = "input_image",
) -> Optional[Tuple[float, float]]:
    groups: MutableMapping[Any, List[float]] = {}
    for row in rows:
        value = _finite(row.get(field))
        if value is not None:
            groups.setdefault(row.get(cluster_field), []).append(value)
    if not groups:
        return None
    cluster_means = np.asarray([np.mean(values) for values in groups.values()], dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(cluster_means, size=(iterations, len(cluster_means)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_bonferroni(p_values: Mapping[str, Optional[float]]) -> Dict[str, Optional[float]]:
    valid = sorted((value, key) for key, value in p_values.items() if value is not None)
    adjusted: Dict[str, Optional[float]] = {key: None for key in p_values}
    running = 0.0
    count = len(valid)
    for index, (value, key) in enumerate(valid):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[key] = running
    return adjusted


def paired_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    delta_field: str = "met3r_paired_delta",
    iterations: int = 10000,
    seed: int = 20260711,
    methods: Optional[Sequence[str]] = None,
    comparison_baseline: str = "iid_external",
) -> List[Dict[str, Any]]:
    if methods is None:
        selected_methods = sorted(
            {
                str(row.get("method"))
                for row in rows
                if row.get("method") not in {None, comparison_baseline}
            }
        )
    else:
        selected_methods = [str(item) for item in methods]
    output: List[Dict[str, Any]] = []
    p_values: Dict[str, Optional[float]] = {}
    for method in selected_methods:
        method_rows = [
            row
            for row in rows
            if row.get("method") == method
            and _finite(row.get(delta_field)) is not None
        ]
        config_ids = sorted(
            {row.get("config_id") for row in method_rows}, key=lambda value: str(value)
        )
        for config_id in config_ids:
            selected = [
                row for row in method_rows if row.get("config_id") == config_id
            ]
            values = np.asarray(
                [float(row[delta_field]) for row in selected], dtype=np.float64
            )
            p_value = None
            if len(values) >= 2 and np.any(values != 0.0):
                try:
                    from scipy.stats import wilcoxon

                    p_value = float(wilcoxon(values).pvalue)
                except Exception:
                    p_value = None
            comparison_id = "{}__vs__{}__{}".format(
                method, comparison_baseline, config_id if config_id is not None else "none"
            )
            p_values[comparison_id] = p_value
            ci = cluster_bootstrap_mean_ci(
                selected, delta_field, iterations=iterations, seed=seed
            )
            cluster_count = len(
                {
                    row.get("input_sha256") or row.get("input_image")
                    for row in selected
                }
            )
            output.append(
                {
                    "comparison_id": comparison_id,
                    "method": method,
                    "config_id": config_id,
                    "comparison_baseline": comparison_baseline,
                    "pair_count": len(values),
                    "object_cluster_count": cluster_count,
                    "mean_delta": float(values.mean()) if len(values) else None,
                    "median_delta": float(np.median(values)) if len(values) else None,
                    "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None,
                    "win_rate": float(np.mean(values < 0.0)) if len(values) else None,
                    "bootstrap_95_ci": list(ci) if ci is not None else None,
                    "wilcoxon_p": p_value,
                    "effect_size_dz": (
                        float(values.mean() / values.std(ddof=1))
                        if len(values) > 1 and values.std(ddof=1) > 0.0
                        else None
                    ),
                }
            )
    corrected = holm_bonferroni(p_values)
    for row in output:
        row["holm_bonferroni_p"] = corrected[row["comparison_id"]]
    return output


def apply_global_holm(rows: Sequence[MutableMapping[str, Any]]) -> None:
    """Apply one correction family across IID and nested-vs-RBF tests."""

    corrected = holm_bonferroni(
        {str(row["comparison_id"]): _finite(row.get("wilcoxon_p")) for row in rows}
    )
    for row in rows:
        row["holm_bonferroni_p"] = corrected[str(row["comparison_id"])]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def resolve_plots_directory(output_dir: Path, requested: Optional[Path]) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    resolved = output_dir.expanduser().resolve()
    if resolved.parent.name == "metrics":
        return resolved.parent.parent / "plots" / resolved.name
    return resolved / "plots"


def resolve_contact_sheets_directory(
    output_dir: Path, requested: Optional[Path]
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    resolved = output_dir.expanduser().resolve()
    if resolved.parent.name == "metrics":
        return resolved.parent.parent / "contact_sheets" / resolved.name
    return resolved / "contact_sheets"


def _row_grid_path(row: Mapping[str, Any]) -> Optional[Path]:
    source = Path(str(row.get("source", "")))
    if not source.is_file():
        return None
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = metadata.get("output")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (source.parent / path).resolve()
    return path if path.is_file() else None


def _labelled_thumbnail(
    path: Optional[Path], label: str, *, width: int = 1200, height: int = 260
) -> Image.Image:
    canvas = Image.new("RGB", (width, height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill="black")
    if path is None:
        draw.text((6, 42), "grid artifact unavailable", fill="red")
        return canvas
    with Image.open(path) as image:
        item = image.convert("RGB")
        item.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = (width - item.width) // 2
        y = 28 + (height - item.height) // 2
        canvas.paste(item, (x, y))
    return canvas


def generate_contact_sheets(
    rows: Sequence[Mapping[str, Any]], directory: Path
) -> Dict[str, Any]:
    """Build paired method sheets and an explicit failure-case gallery."""

    directory.mkdir(parents=True, exist_ok=True)
    groups: MutableMapping[Tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row.get("input_sha256") or row.get("input_image"), row.get("seed"))
        groups.setdefault(key, []).append(row)
    artifacts: List[str] = []
    failures: List[Mapping[str, Any]] = []
    for (identity, seed), members in sorted(groups.items(), key=lambda item: repr(item[0])):
        panels = []
        for row in sorted(
            members,
            key=lambda item: (str(item.get("method")), str(item.get("config_id"))),
        ):
            label = "{} config={} seed={} status={}".format(
                row.get("method"), row.get("config_id"), seed, row.get("status")
            )
            panels.append(_labelled_thumbnail(_row_grid_path(row), label))
            if (
                row.get("status") != "succeeded"
                or bool(row.get("artifact_failure", False))
                or bool(row.get("guardrail_error"))
                or row.get("collapse_detector_label") == "view_collapse_alert"
            ):
                failures.append(row)
        if not panels:
            continue
        sheet = Image.new(
            "RGB", (max(panel.width for panel in panels), sum(panel.height for panel in panels)), "white"
        )
        top = 0
        for panel in panels:
            sheet.paste(panel, (0, top))
            top += panel.height
        digest = hashlib.sha256("{}|{}".format(identity, seed).encode("utf-8")).hexdigest()[:12]
        path = directory / "paired_{}_seed_{}.jpg".format(digest, seed)
        sheet.save(path, quality=90)
        artifacts.append(str(path))

    gallery_rows = failures[:20]
    if gallery_rows:
        panels = [
            _labelled_thumbnail(
                _row_grid_path(row),
                "{} config={} seed={} status={}".format(
                    row.get("method"), row.get("config_id"), row.get("seed"), row.get("status")
                ),
                width=1000,
                height=220,
            )
            for row in gallery_rows
        ]
        gallery = Image.new("RGB", (1000, sum(panel.height for panel in panels)), "white")
        top = 0
        for panel in panels:
            gallery.paste(panel, (0, top))
            top += panel.height
    else:
        gallery = Image.new("RGB", (1000, 120), "white")
        ImageDraw.Draw(gallery).text((12, 45), "No recorded failure rows.", fill="black")
    gallery_path = directory / "failure_gallery.jpg"
    gallery.save(gallery_path, quality=90)
    artifacts.append(str(gallery_path))
    return {
        "complete": bool(groups) and len(artifacts) == len(groups) + 1,
        "directory": str(directory),
        "paired_sheet_count": max(0, len(artifacts) - 1),
        "failure_row_count": len(failures),
        "failure_gallery": str(gallery_path),
        "artifacts": artifacts,
    }


def generate_evaluation_plots(
    plots_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    angle_summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        return {
            "complete": False,
            "plots_dir": str(plots_dir),
            "artifacts": [],
            "error": repr(error),
        }

    artifacts: List[str] = []

    def save(figure: Any, name: str) -> None:
        path = plots_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(str(path))

    figure, axis = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for summary in summaries:
        x = _finite(summary.get("target_kl"))
        y = _finite(summary.get("met3r_all_pair_mean"))
        if x is None or y is None:
            continue
        axis.scatter(x, y, label="{} K={}".format(summary.get("method"), summary.get("rank")))
        plotted = True
    axis.set(xlabel="target joint KL", ylabel="MEt3R all-pair mean", title="MEt3R versus KL/rank")
    if plotted:
        axis.legend(fontsize=7)
    else:
        axis.text(0.5, 0.5, "No finite MEt3R rows", ha="center", va="center")
    save(figure, "pilot_met3r_vs_kl_rank.png")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    iid_values = [
        float(row["met3r_paired_delta"])
        for row in rows
        if _finite(row.get("met3r_paired_delta")) is not None
        and row.get("method") != "iid_external"
    ]
    rbf_rows = add_metric_comparison_deltas(
        rows,
        baseline_method="lowrank_camera_rbf",
        target_methods=("lowrank_nested_tree_a", "lowrank_nested_tree_ab"),
        output_field="met3r_vs_rbf_delta",
    )
    rbf_values = [
        float(row["met3r_vs_rbf_delta"])
        for row in rbf_rows
        if _finite(row.get("met3r_vs_rbf_delta")) is not None
    ]
    for axis, values, title in zip(
        axes,
        (iid_values, rbf_values),
        ("method minus IID", "nested minus selected RBF"),
    ):
        if values:
            axis.boxplot(values, vert=True)
            axis.scatter(np.ones(len(values)), values, alpha=0.45, s=12)
        else:
            axis.text(0.5, 0.5, "No strict pairs", ha="center", va="center")
        axis.axhline(0.0, color="0.5", linewidth=1)
        axis.set(title=title, ylabel="paired MEt3R delta")
    save(figure, "paired_met3r_deltas.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    angle_groups: Dict[Tuple[Any, Any], List[Tuple[float, float]]] = {}
    for row in angle_summaries:
        angle = _finite(row.get("angle_bin_deg"))
        score = _finite(row.get("met3r_score"))
        if angle is not None and score is not None:
            angle_groups.setdefault(
                (row.get("method") or row.get("inference_method"), row.get("config_id")), []
            ).append((angle, score))
    for (method, config_id), values in sorted(angle_groups.items(), key=lambda item: repr(item[0])):
        by_angle: Dict[float, List[float]] = {}
        for angle, score in values:
            by_angle.setdefault(angle, []).append(score)
        x = sorted(by_angle)
        y = [float(np.mean(by_angle[item])) for item in x]
        axis.plot(x, y, marker="o", label="{}:{}".format(method, config_id))
    if angle_groups:
        axis.legend(fontsize=6)
    else:
        axis.text(0.5, 0.5, "No finite angle-bin MEt3R rows", ha="center", va="center")
    axis.set(xlabel="angular separation (deg)", ylabel="MEt3R", title="Per-angle MEt3R")
    save(figure, "per_angle_met3r.png")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels: List[str] = []
    costs: List[float] = []
    for summary in summaries:
        parts = [
            abs(value)
            for value in (
                _finite(summary.get("dino_identity_mean_delta")),
                _finite(summary.get("small_component_ratio_delta")),
                _finite(summary.get("component_failure_rate_delta")),
                _finite(summary.get("foreground_area_cv_delta")),
            )
            if value is not None
        ]
        if parts:
            labels.append("{}:{}".format(summary.get("method"), summary.get("config_id")))
            costs.append(float(sum(parts)))
    if costs:
        axis.bar(np.arange(len(costs)), costs)
        axis.set_xticks(np.arange(len(costs)), labels, rotation=80, fontsize=6)
    else:
        axis.text(0.5, 0.5, "No finite identity/artifact costs", ha="center", va="center")
    axis.set(ylabel="absolute guardrail cost", title="Identity and artifact guardrails")
    save(figure, "artifact_guardrails.png")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    pareto_count = 0
    for summary in summaries:
        score = _finite(summary.get("met3r_all_pair_mean"))
        parts = [
            abs(value)
            for value in (
                _finite(summary.get("dino_identity_mean_delta")),
                _finite(summary.get("small_component_ratio_delta")),
                _finite(summary.get("component_failure_rate_delta")),
                _finite(summary.get("foreground_area_cv_delta")),
            )
            if value is not None
        ]
        if score is None or not parts:
            continue
        cost = float(sum(parts))
        axis.scatter(cost, score)
        axis.annotate(
            "{}:K{}:KL{}".format(
                summary.get("method"), summary.get("rank"), summary.get("target_kl")
            ),
            (cost, score),
            fontsize=6,
        )
        pareto_count += 1
    if not pareto_count:
        axis.text(0.5, 0.5, "No finite Pareto rows", ha="center", va="center")
    axis.set(
        xlabel="identity/artifact cost",
        ylabel="MEt3R all-pair mean (lower is better)",
        title="PILOT Pareto view",
    )
    save(figure, "pilot_pareto.png")
    return {
        "complete": len(artifacts) == 5,
        "plots_dir": str(plots_dir),
        "artifacts": artifacts,
        "error": None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", choices=("lightweight", "all"), default="all")
    parser.add_argument("--identity-model", default="facebook/dinov2-small")
    parser.add_argument("--identity-model-revision", default=None)
    parser.add_argument("--identity-device", default="cuda")
    parser.add_argument("--met3r-revision", default=None)
    parser.add_argument("--met3r-device", default="cuda")
    parser.add_argument("--met3r-image-size", type=int, default=256)
    parser.add_argument("--met3r-batch-size", type=int, default=1)
    parser.add_argument(
        "--angle-bins-deg",
        type=float,
        nargs="+",
        required=True,
        help="Angular separation bins supplied by evaluation.angle_bins_deg.",
    )
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--contact-sheets-dir", type=Path, default=None)
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(
        not math.isfinite(value) or not 0 < value <= 180
        for value in args.angle_bins_deg
    ):
        parser.error("--angle-bins-deg values must be finite and lie in (0, 180]")
    if len(set(args.angle_bins_deg)) != len(args.angle_bins_deg):
        parser.error("--angle-bins-deg values must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = _read_manifest(args.manifest)
    met3r_provenance: Optional[Dict[str, Any]] = None
    provenance_error: Optional[str] = None
    if args.metrics == "all":
        try:
            met3r_provenance = verify_installed_met3r_revision(args.met3r_revision)
        except Exception as error:
            provenance_error = repr(error)
            failure_payload = {
                "schema_version": 1,
                "met3r_required": True,
                "met3r_score_direction": "lower_is_better",
                "angle_bins_deg": list(args.angle_bins_deg),
                "formal_evaluation_complete": False,
                "met3r_provenance": None,
                "met3r_provenance_error": provenance_error,
                "samples": [],
                "configuration_summaries": [],
                "paired_statistics": [],
            }
            _atomic_json(args.output_dir / "lowrank_metrics.json", failure_payload)
            print(provenance_error, file=sys.stderr)
            return 2
    raw_path = args.output_dir / "multiview_metrics.json"
    command = [
        sys.executable,
        "-m",
        "scripts.eval_multiview_consistency",
        "--manifest",
        str(args.manifest),
        "--metrics",
        args.metrics,
        "--angle-bins",
        *[str(value) for value in args.angle_bins_deg],
        "--iid-baseline-method",
        "iid_external",
        "--met3r-device",
        str(args.met3r_device),
        "--met3r-image-size",
        str(args.met3r_image_size),
        "--met3r-batch-size",
        str(args.met3r_batch_size),
        "--output",
        str(raw_path),
    ]
    result = subprocess.run(command, check=False)
    if not raw_path.exists():
        failure_payload = {
            "schema_version": 1,
            "met3r_required": args.metrics == "all",
            "angle_bins_deg": list(args.angle_bins_deg),
            "formal_evaluation_complete": False,
            "base_evaluator_returncode": result.returncode,
            "met3r_provenance": met3r_provenance,
            "error": "multiview evaluator produced no metrics artifact",
            "samples": [],
            "configuration_summaries": [],
            "paired_statistics": [],
        }
        _atomic_json(args.output_dir / "lowrank_metrics.json", failure_payload)
        return 2
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    reconciled, manifest_audit = reconcile_manifest_samples(
        manifest_rows, raw.get("samples", [])
    )
    identity_error: Optional[str] = None
    if args.skip_identity:
        backend = None
    else:
        try:
            backend = DinoIdentityBackend(
                args.identity_model,
                args.identity_device,
                revision=args.identity_model_revision,
            )
        except Exception as error:
            backend = None
            identity_error = repr(error)
    rows = augment_sample_metrics(reconciled, identity_backend=backend)
    rows = add_paired_deltas(rows)
    summaries = configuration_summaries(rows)
    iid_statistics = paired_statistics(
        rows,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    rbf_pairs = add_metric_comparison_deltas(
        rows,
        baseline_method="lowrank_camera_rbf",
        target_methods=("lowrank_nested_tree_a", "lowrank_nested_tree_ab"),
        output_field="met3r_vs_rbf_delta",
    )
    rbf_statistics = paired_statistics(
        rbf_pairs,
        delta_field="met3r_vs_rbf_delta",
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
        methods=("lowrank_nested_tree_a", "lowrank_nested_tree_ab"),
        comparison_baseline="lowrank_camera_rbf",
    )
    comparison_statistics = [*iid_statistics, *rbf_statistics]
    apply_global_holm(comparison_statistics)
    plots = generate_evaluation_plots(
        resolve_plots_directory(args.output_dir, args.plots_dir),
        rows=rows,
        summaries=summaries,
        angle_summaries=raw.get("angle_bin_summaries", []),
    )
    contact_sheets = generate_contact_sheets(
        rows,
        resolve_contact_sheets_directory(
            args.output_dir, args.contact_sheets_dir
        ),
    )
    completeness = assess_evaluation_completeness(
        rows,
        manifest_rows,
        met3r_required=args.metrics == "all",
        identity_required=not args.skip_identity,
    )
    formal_complete = bool(
        result.returncode == 0
        and completeness["complete"]
        and identity_error is None
        and plots.get("complete", False)
        and contact_sheets.get("complete", False)
        and (met3r_provenance is not None if args.metrics == "all" else True)
    )
    payload = {
        "schema_version": 1,
        "met3r_required": args.metrics == "all",
        "met3r_score_direction": "lower_is_better" if args.metrics == "all" else None,
        "met3r_provenance": met3r_provenance,
        "met3r_provenance_error": provenance_error,
        "identity_model": args.identity_model,
        "identity_model_revision": args.identity_model_revision,
        "angle_bins_deg": list(args.angle_bins_deg),
        "identity_error": identity_error,
        "base_evaluator_returncode": result.returncode,
        "manifest_audit": manifest_audit,
        "evaluation_completeness": completeness,
        "formal_evaluation_complete": formal_complete,
        "statement": "NILE-inspired nested Gaussian element topology; strict NILE/SZ is not implemented in this study.",
        "samples": rows,
        "configuration_summaries": summaries,
        "paired_statistics": iid_statistics,
        "rbf_paired_rows": rbf_pairs,
        "rbf_paired_statistics": rbf_statistics,
        "paired_comparison_statistics": comparison_statistics,
        "angle_bin_summaries": raw.get("angle_bin_summaries", []),
        "plots": plots,
        "contact_sheets": contact_sheets,
        "lightweight_role": "collapse_detector_only",
    }
    _atomic_json(args.output_dir / "lowrank_metrics.json", payload)
    _write_csv(args.output_dir / "sample_metrics.csv", rows)
    _write_csv(args.output_dir / "configuration_summaries.csv", summaries)
    _write_csv(args.output_dir / "paired_statistics.csv", comparison_statistics)
    _write_csv(args.output_dir / "rbf_paired_rows.csv", rbf_pairs)
    print(args.output_dir / "lowrank_metrics.json")
    return 0 if formal_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
