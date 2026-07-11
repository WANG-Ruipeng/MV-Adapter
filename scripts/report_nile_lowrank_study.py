"""Build an evidence-bounded report for the NILE low-rank study.

The reporter only summarizes artifacts that already exist.  Missing stages are
reported as blockers; metrics are never imputed and a partial run is never
promoted to a completed FULL experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit


SCIENTIFIC_LABELS = {
    "nested_positive",
    "generic_coupling_only",
    "initial_noise_no_go",
    "full_blocked",
}
SUCCESS_STATUSES = {"success", "succeeded", "complete", "completed", "skipped_complete"}
STRICT_STATEMENT = (
    "NILE-inspired nested Gaussian element topology; strict NILE/SZ is not "
    "implemented in this study."
)


DEFAULT_CANDIDATES: Mapping[str, Tuple[str, ...]] = {
    "config_lock": ("configs/config_lock.json", "config_lock.json"),
    "resolved_config": (
        "configs/resolved_config.json",
        "configs/resolved_config.yaml",
        "configs/resolved_config.yml",
        "resolved_config.json",
        "resolved_config.yaml",
        "resolved_config.yml",
    ),
    "input_validation": ("inputs/input_validation.json", "input_validation.json"),
    "preflight": (
        "distribution_gates/configuration_gates.json",
        "distribution_gates/preflight_summary.json",
        "distribution_gates/preflight.json",
        "distribution_gates/distribution_gates.json",
        "preflight.json",
    ),
    "pilot_metrics": (
        "metrics/pilot/lowrank_metrics.json",
        "pilot/metrics/lowrank_metrics.json",
        "pilot/lowrank_metrics.json",
        "metrics/pilot_lowrank_metrics.json",
    ),
    "selected_candidates": (
        "selected_candidates/selected_candidates.json",
        "selected_candidates.json",
    ),
    "full_metrics": (
        "metrics/full/lowrank_metrics.json",
        "full/metrics/lowrank_metrics.json",
        "full/lowrank_metrics.json",
        "metrics/full_lowrank_metrics.json",
        "metrics/lowrank_metrics.json",
    ),
    "trajectory": (
        "trajectory/trajectory_summary.json",
        "trajectory/summary.json",
        "trajectory/trajectory_metrics.json",
    ),
    "manifest": ("manifest.jsonl", "manifest.json", "manifest.csv"),
    "pilot_manifest": ("pilot/manifest.json", "pilot/manifest.jsonl"),
    "full_manifest": ("full/manifest.json", "full/manifest.jsonl"),
    "trajectory_manifest": ("trajectory/manifest.json", "trajectory/manifest.jsonl"),
    "workflow_status": (
        "runtime_status.json",
        "environment/workflow_status.json",
        "environment/completion_status.json",
        "workflow_status.json",
    ),
    "test_status": (
        "environment/test_results.json",
        "environment/test_status.json",
        "environment/tests_summary.json",
        "test_status.json",
    ),
}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nested_get(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _configuration_hash(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_structured(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required to read a YAML resolved config") from error
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError("unsupported structured artifact: {}".format(path))


def _load_manifest(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)], {}
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(json.loads(line)))
        return rows, {}
    payload = _load_structured(path)
    if isinstance(payload, list):
        return [dict(item) for item in payload], {}
    if not isinstance(payload, Mapping):
        raise ValueError("manifest must be a list or object")
    records = payload.get("runs", payload.get("records", []))
    return [dict(item) for item in records], dict(payload)


def _resolve_artifact(
    root: Path, name: str, explicit: Optional[Path]
) -> Optional[Path]:
    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve() if candidate.exists() else candidate
    for relative in DEFAULT_CANDIDATES[name]:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_artifact(path: Optional[Path], *, manifest: bool = False) -> Tuple[Any, Optional[str]]:
    if path is None or not path.is_file():
        return None, None
    try:
        return (_load_manifest(path) if manifest else _load_structured(path)), None
    except Exception as error:
        return None, "{}: {}".format(type(error).__name__, error)


def _explicit_boolean(payloads: Iterable[Any], key: str) -> Optional[bool]:
    for payload in payloads:
        if isinstance(payload, Mapping) and isinstance(payload.get(key), bool):
            return bool(payload[key])
        completion = payload.get("completion") if isinstance(payload, Mapping) else None
        if isinstance(completion, Mapping) and isinstance(completion.get(key), bool):
            return bool(completion[key])
    return None


def _preflight_passed(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if "diagnostic_plots_complete" in payload and payload.get(
        "diagnostic_plots_complete"
    ) is not True:
        return False
    for key in ("passed", "all_passed", "distribution_gate_passed", "complete"):
        if isinstance(payload.get(key), bool):
            return bool(payload[key])
    records = payload.get(
        "configurations", payload.get("records", payload.get("results"))
    )
    if isinstance(records, list) and records:
        valid_records = [row for row in records if isinstance(row, Mapping)]
        return all(
            bool(row.get("passed", row.get("distribution_gate_passed", False)))
            for row in valid_records
        ) and len(valid_records) == len(records)
    failures = payload.get("failures")
    if isinstance(failures, list):
        return not failures and bool(payload.get("record_count", payload.get("total", 0)))
    return False


def _met3r_complete(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    explicit = _explicit_boolean((payload,), "met3r_complete")
    if explicit is not None:
        return explicit
    if payload.get("met3r_required") is not True:
        return False
    samples = payload.get("samples")
    if isinstance(samples, list) and samples:
        succeeded = [
            row
            for row in samples
            if isinstance(row, Mapping)
            and str(row.get("status", "succeeded")).lower() in SUCCESS_STATUSES
        ]
        if succeeded:
            return all(_score(row) is not None for row in succeeded)
    summaries = payload.get("configuration_summaries")
    return bool(
        isinstance(summaries, list)
        and summaries
        and all(_finite(row.get("met3r_all_pair_mean")) is not None for row in summaries)
    )


def _selection_complete(payload: Any) -> bool:
    if not isinstance(payload, Mapping) or not payload.get("configuration_hash"):
        return False
    selections = payload.get("selections")
    if not isinstance(selections, Mapping):
        return False
    required = ("camera_rbf", "nested_tree_a", "nested_tree_ab")
    for name in required:
        selected = selections.get(name)
        if not isinstance(selected, Mapping):
            return False
        if selected.get("status") not in {"selected", "no_eligible_candidate"}:
            return False
        if not isinstance(selected.get("configuration"), Mapping):
            return False
    return True


def _stage_records(records: Sequence[Mapping[str, Any]], stage: str) -> List[Mapping[str, Any]]:
    stage = stage.lower()
    result = []
    for row in records:
        labels = " ".join(
            str(row.get(field, "")).lower()
            for field in ("stage", "split", "phase", "run_type")
        )
        if stage in labels:
            result.append(row)
    return result


def _run_counts(records: Sequence[Mapping[str, Any]], stage: str) -> Dict[str, int]:
    selected = _stage_records(records, stage)
    succeeded = sum(str(row.get("status", "")).lower() in SUCCESS_STATUSES for row in selected)
    failed = sum(str(row.get("status", "")).lower() in {"failed", "error"} for row in selected)
    blocked = sum(str(row.get("status", "")).lower() == "blocked" for row in selected)
    return {"planned": len(selected), "succeeded": succeeded, "failed": failed, "blocked": blocked}


def _expected_runs(
    config: Any,
    inputs: Any,
    stage: str,
    *,
    preflight: Any = None,
    selected_candidates: Any = None,
) -> Optional[int]:
    if not isinstance(config, Mapping) or not isinstance(inputs, Mapping):
        return None
    section = config.get(stage)
    if not isinstance(section, Mapping):
        return None
    count_key = "required_pilot_count" if stage == "pilot" else "required_full_count"
    input_count = inputs.get(count_key)
    if input_count is None:
        input_count = _nested_get(config, "data", "{}_count".format(stage))
    try:
        input_count = int(input_count)
        seed_count = len(section.get("seeds", []))
        if stage == "pilot":
            gate_rows = (
                preflight.get("configurations", [])
                if isinstance(preflight, Mapping)
                else []
            )
            eligible = [
                row
                for row in gate_rows
                if isinstance(row, Mapping)
                and bool(
                    row.get(
                        "eligible_for_generation",
                        row.get("distribution_gate_passed", row.get("passed", False)),
                    )
                )
            ]
            per_pair = (
                len(eligible)
                if gate_rows
                else int(section.get("expected_configs_per_input_seed", 0))
            )
        else:
            selections = (
                selected_candidates.get("selections", {})
                if isinstance(selected_candidates, Mapping)
                else {}
            )
            if isinstance(selections, Mapping) and selections:
                selectable = 0
                for selection in selections.values():
                    configuration = (
                        selection.get("configuration")
                        if isinstance(selection, Mapping)
                        else None
                    )
                    if isinstance(configuration, Mapping) and configuration.get(
                        "distribution_gate_passed", True
                    ) is not False:
                        selectable += 1
                per_pair = 2 + selectable
            else:
                per_pair = len(section.get("methods", []))
        return input_count * seed_count * per_pair if seed_count and per_pair else None
    except (TypeError, ValueError):
        return None


def _stage_complete(
    records: Sequence[Mapping[str, Any]], stage: str, expected: Optional[int]
) -> bool:
    counts = _run_counts(records, stage)
    if not counts["planned"] or counts["failed"] or counts["blocked"]:
        return False
    if counts["succeeded"] != counts["planned"]:
        return False
    return expected is None or counts["succeeded"] >= expected


def _score(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("angle_all_met3r_score", "met3r_all_pair_mean", "met3r_score"):
        value = _finite(row.get(key))
        if value is not None:
            return value
    return None


def _method_kind(method: Any) -> str:
    name = str(method or "").lower()
    if "iid" in name:
        return "iid"
    if "nested" in name or "tree_a" in name or "tree_ab" in name:
        return "nested"
    if "rbf" in name or "camera" in name:
        return "rbf"
    if "shared_full" in name:
        return "shared_full"
    return "other"


def _pair_key(row: Mapping[str, Any]) -> Optional[Tuple[str, Any]]:
    identity = next(
        (
            row.get(key)
            for key in ("input_hash", "input_image", "input_path", "source")
            if row.get(key) not in (None, "")
        ),
        None,
    )
    seed = row.get("seed")
    return (str(identity), seed) if identity is not None and seed is not None else None


def _paired_evidence(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[str, Any], Dict[str, Tuple[str, float, Mapping[str, Any]]]] = {}
    for row in samples:
        key = _pair_key(row)
        score = _score(row)
        kind = _method_kind(row.get("method"))
        if key is None or score is None or kind not in {
            "iid",
            "rbf",
            "nested",
            "shared_full",
        }:
            continue
        method = str(row.get("method"))
        by_key.setdefault(key, {})[method] = (kind, score, row)
    method_deltas: Dict[str, Dict[str, List[float]]] = {}
    method_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for members in by_key.values():
        iid = next((value for value in members.values() if value[0] == "iid"), None)
        rbf = next((value for value in members.values() if value[0] == "rbf"), None)
        for method, (kind, value, row) in members.items():
            if kind not in {"rbf", "nested", "shared_full"}:
                continue
            entry = method_deltas.setdefault(method, {"vs_iid": [], "vs_rbf": []})
            if iid is not None:
                entry["vs_iid"].append(value - iid[1])
            if kind == "nested" and rbf is not None:
                entry["vs_rbf"].append(value - rbf[1])
            method_rows.setdefault(method, []).append(row)
    return {"deltas": method_deltas, "rows": method_rows}


def _mean_ci(values: Sequence[float]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mean = sum(ordered) / len(ordered)
    # An exact, deterministic cluster bootstrap is not possible without raw
    # object IDs here.  For reporting we use only a supplied CI or, when all
    # paired deltas agree exactly, the mathematically exact point interval.
    ci = [ordered[0], ordered[0]] if ordered[0] == ordered[-1] else None
    return {
        "pair_count": len(ordered),
        "mean_delta": mean,
        "win_rate": sum(value < 0.0 for value in ordered) / len(ordered),
        "bootstrap_95_ci": ci,
    }


def _comparison_baseline(row: Mapping[str, Any], default: Any = None) -> Any:
    """Return the named baseline, including a legacy comparison-id fallback."""

    baseline = row.get("comparison_baseline")
    if baseline not in (None, ""):
        return baseline
    comparison_id = str(row.get("comparison_id") or "")
    if "__vs__" in comparison_id:
        tail = comparison_id.split("__vs__", 1)[1]
        return tail.rsplit("__", 1)[0] if "__" in tail else tail
    return default


def _normalize_comparison_rows(
    rows: Any, *, default_baseline: Any = None
) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for original in rows:
        if not isinstance(original, Mapping):
            continue
        row = dict(original)
        baseline = _comparison_baseline(row, default_baseline)
        if baseline not in (None, "") and row.get("comparison_baseline") in (None, ""):
            row["comparison_baseline"] = baseline
            row["comparison_baseline_inferred"] = True
        normalized.append(row)
    return normalized


def _comparison_statistics(payload: Any) -> Dict[str, Any]:
    """Read the current comparison schema without breaking legacy artifacts.

    The current evaluator emits a globally Holm-corrected union in
    paired_comparison_statistics. When that key exists it is authoritative,
    even when empty: silently reconstructing missing formal statistics from raw
    samples would overstate an incomplete evaluation. Older artifacts are read
    from their separate IID and RBF fields.
    """

    if not isinstance(payload, Mapping):
        return {"source": None, "authoritative": False, "rows": []}
    if "paired_comparison_statistics" in payload:
        return {
            "source": "paired_comparison_statistics",
            "authoritative": True,
            "rows": _normalize_comparison_rows(
                payload.get("paired_comparison_statistics")
            ),
        }
    iid_rows = _normalize_comparison_rows(
        payload.get("paired_statistics"), default_baseline="iid_external"
    )
    has_rbf_field = "rbf_paired_statistics" in payload
    rbf_rows = _normalize_comparison_rows(
        payload.get("rbf_paired_statistics"),
        default_baseline="lowrank_camera_rbf",
    )
    rows: List[Dict[str, Any]] = []
    seen = set()
    for row in [*iid_rows, *rbf_rows]:
        identity = row.get("comparison_id") or json.dumps(
            row, sort_keys=True, ensure_ascii=True, default=str
        )
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(row)
    if has_rbf_field:
        source = "paired_statistics+rbf_paired_statistics"
    elif "paired_statistics" in payload:
        source = "paired_statistics"
    else:
        source = None
    return {
        "source": source,
        "authoritative": bool(has_rbf_field),
        "rows": rows,
    }


def _comparison_rows_for_baseline(
    rows: Sequence[Mapping[str, Any]], baseline_kind: str
) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _method_kind(_comparison_baseline(row)) == baseline_kind
    ]


def _method_topology(method: Any) -> str:
    name = str(method or "").lower()
    if "tree_ab" in name:
        return "nested_tree_ab"
    if "tree_a" in name:
        return "nested_tree_a"
    if "tree_b" in name:
        return "nested_tree_b"
    if "rbf" in name or "camera" in name:
        return "camera_rbf"
    return _method_kind(method)


def _formal_summary_for_method(
    rows: Sequence[Mapping[str, Any]], method: str, baseline_kind: str
) -> Optional[Mapping[str, Any]]:
    exact = [
        row
        for row in rows
        if str(row.get("method")) == method
        and _method_kind(_comparison_baseline(row)) == baseline_kind
    ]
    if len(exact) == 1:
        return exact[0]
    # Legacy artifacts occasionally changed selected_* to lowrank_* between
    # generation and evaluation. A topology match is safe only when unique.
    topology = _method_topology(method)
    compatible = [
        row
        for row in rows
        if _method_topology(row.get("method")) == topology
        and _method_kind(_comparison_baseline(row)) == baseline_kind
    ]
    return compatible[0] if len(compatible) == 1 else None


def _comparison_statistics_complete(bundle: Mapping[str, Any]) -> bool:
    """Validate the required current-schema IID and nested-vs-RBF evidence."""

    if not bundle.get("authoritative"):
        return True
    rows = bundle.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return False
    core = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and (
            _method_kind(_comparison_baseline(row)) == "iid"
            and _method_kind(row.get("method")) in {"rbf", "nested"}
            or _method_kind(_comparison_baseline(row)) == "rbf"
            and _method_kind(row.get("method")) == "nested"
        )
    ]
    required_relationships = {
        (
            _method_kind(row.get("method")),
            _method_kind(_comparison_baseline(row)),
        )
        for row in core
    }
    if not {("rbf", "iid"), ("nested", "iid"), ("nested", "rbf")}.issubset(
        required_relationships
    ):
        return False
    for row in core:
        interval = row.get("bootstrap_95_ci")
        if (
            _finite(row.get("pair_count")) is None
            or float(row["pair_count"]) <= 0
            or _finite(row.get("mean_delta")) is None
            or not isinstance(interval, list)
            or len(interval) != 2
            or any(_finite(value) is None for value in interval)
            or "wilcoxon_p" not in row
            or "holm_bonferroni_p" not in row
        ):
            return False
    return True


def _supported_improvement(summary: Optional[Mapping[str, Any]]) -> bool:
    if not summary or _finite(summary.get("mean_delta")) is None:
        return False
    if float(summary["mean_delta"]) >= 0.0:
        return False
    interval = summary.get("bootstrap_95_ci")
    if isinstance(interval, list) and len(interval) == 2:
        upper = _finite(interval[1])
        if upper is not None and upper < 0.0:
            return True
    corrected_p = _finite(summary.get("holm_bonferroni_p"))
    if corrected_p is not None and corrected_p < 0.05:
        return True
    pair_count = _finite(summary.get("pair_count"))
    win_rate = _finite(summary.get("win_rate"))
    return bool(
        pair_count is not None
        and pair_count >= 10
        and win_rate is not None
        and win_rate >= 0.8
    )


def _guardrails_status(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> Optional[bool]:
    if not rows:
        return None
    fields = (
        ("dino_identity_mean_delta", -float(policy.get("max_dino_drop_abs", 0.02)), "min"),
        ("small_component_ratio_delta", float(policy.get("max_small_component_ratio_increase", 0.02)), "max"),
        ("component_failure_rate_delta", float(policy.get("max_component_failure_rate_increase", 0.10)), "max"),
        ("foreground_area_cv_delta", float(policy.get("max_foreground_area_cv_increase", 0.05)), "max"),
    )
    for field, threshold, direction in fields:
        values = [_finite(row.get(field)) for row in rows]
        if any(value is None for value in values):
            return None
        if direction == "min" and any(float(value) < threshold for value in values):
            return False
        if direction == "max" and any(float(value) > threshold for value in values):
            return False
    r_hf_values = [_finite(row.get("r_hf")) for row in rows]
    if any(value is None for value in r_hf_values):
        return None
    r_hf_min = float(policy.get("rhf_min", 0.5))
    r_hf_max = float(policy.get("rhf_max", 1.5))
    if any(
        not r_hf_min <= float(value) <= r_hf_max
        for value in r_hf_values
    ):
        return False
    collapse_labels = [row.get("collapse_detector_label") for row in rows]
    if any(label in (None, "") for label in collapse_labels):
        return None
    if any(
        str(label) == "view_collapse_alert" for label in collapse_labels
    ) or any(bool(row.get("collapse_alert", False)) for row in rows):
        return False
    return not any(bool(row.get("artifact_failure", False)) for row in rows)


def _trajectory_state(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    for key in ("correlation_state", "classification", "retention_classification"):
        value = payload.get(key)
        if value:
            normalized = str(value).lower().replace("-", "_")
            if "wash" in normalized:
                return "wash_out"
            if "ampl" in normalized:
                return "amplified"
            if "retain" in normalized or "stable" in normalized:
                return "retained"
            return normalized
    return None


def _method_trajectory_evidence(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"source": None, "states": {}, "thresholds": {}}
    schema_version = _finite(payload.get("schema_version"))
    current_schema = (
        schema_version is not None
        and schema_version >= 2
        or "method_summaries" in payload
    )
    if not current_schema:
        return {
            "source": "legacy_aggregate",
            "states": {},
            "thresholds": {},
        }
    thresholds = payload.get("classification_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    washout = _finite(thresholds.get("washout_max_final_g"))
    amplified = _finite(thresholds.get("amplified_min_final_g"))
    states: Dict[str, Dict[str, Any]] = {}
    summaries = payload.get("method_summaries", [])
    if (
        washout is None
        or amplified is None
        or washout >= amplified
        or not isinstance(summaries, list)
    ):
        return {
            "source": "method_summaries",
            "states": states,
            "thresholds": dict(thresholds),
        }
    for row in summaries:
        if not isinstance(row, Mapping) or row.get("method") in (None, ""):
            continue
        final_g = _finite(row.get("final_g_t"))
        if final_g is None:
            continue
        if final_g < washout:
            state = "wash_out"
        elif final_g > amplified:
            state = "amplified"
        else:
            state = "retained"
        states[str(row["method"])] = {
            "state": state,
            "final_g_t": final_g,
            "pair_count": row.get("pair_count"),
            "rank": row.get("rank"),
        }
    return {
        "source": "method_summaries",
        "states": states,
        "thresholds": {
            "washout_max_final_g": washout,
            "amplified_min_final_g": amplified,
        },
    }


def _trajectory_for_method(
    method: str,
    evidence: Mapping[str, Any],
    legacy_state: Optional[str],
) -> Optional[Mapping[str, Any]]:
    states = evidence.get("states", {})
    if isinstance(states, Mapping):
        exact = states.get(method)
        if isinstance(exact, Mapping):
            return exact
        topology = _method_topology(method)
        compatible = [
            row
            for candidate, row in states.items()
            if _method_topology(candidate) == topology
            and isinstance(row, Mapping)
        ]
        if len(compatible) == 1:
            return compatible[0]
    if evidence.get("source") == "legacy_aggregate" and legacy_state is not None:
        return {
            "state": legacy_state,
            "final_g_t": None,
            "pair_count": None,
            "rank": None,
            "legacy_aggregate_fallback": True,
        }
    return None


def _selected_configuration(method: str, selected: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(selected, Mapping):
        return None
    selections = selected.get("selections")
    if not isinstance(selections, Mapping):
        return None
    topology = _method_topology(method)
    row = selections.get(topology)
    configuration = row.get("configuration") if isinstance(row, Mapping) else None
    return configuration if isinstance(configuration, Mapping) else None


def _nested_fairness(method: str, selected: Any) -> Dict[str, Any]:
    nested = _selected_configuration(method, selected)
    rbf = _selected_configuration("camera_rbf", selected)
    if nested is None or rbf is None:
        return {
            "valid": None,
            "reason": "selected_configuration_missing",
            "nested_rank": nested.get("rank") if nested else None,
            "rbf_rank": rbf.get("rank") if rbf else None,
            "nested_target_kl": nested.get("target_kl") if nested else None,
            "rbf_target_kl": rbf.get("target_kl") if rbf else None,
        }
    nested_rank = _finite(nested.get("rank"))
    rbf_rank = _finite(rbf.get("rank"))
    nested_kl = _finite(nested.get("target_kl"))
    rbf_kl = _finite(rbf.get("target_kl"))
    if None in (nested_rank, rbf_rank, nested_kl, rbf_kl):
        return {
            "valid": None,
            "reason": "rank_or_target_kl_missing",
            "nested_rank": nested_rank,
            "rbf_rank": rbf_rank,
            "nested_target_kl": nested_kl,
            "rbf_target_kl": rbf_kl,
        }
    equal_rank = bool(nested_rank == rbf_rank)
    equal_target_kl = math.isclose(
        float(nested_kl), float(rbf_kl), rel_tol=1e-9, abs_tol=1e-12
    )
    valid = bool(equal_rank and equal_target_kl)
    return {
        "valid": valid,
        "reason": "equal_rank_equal_target_kl" if valid else "fairness_mismatch",
        "equal_rank": equal_rank,
        "equal_target_kl": equal_target_kl,
        "nested_rank": nested_rank,
        "rbf_rank": rbf_rank,
        "nested_target_kl": nested_kl,
        "rbf_target_kl": rbf_kl,
    }


def _formal_equal_rank_kl_audit(selected: Any) -> Dict[str, Any]:
    if not isinstance(selected, Mapping) or not isinstance(
        selected.get("selections"), Mapping
    ):
        return {
            "complete": False,
            "formal_comparison_count": 0,
            "records": [],
            "reason": "selected_candidates_missing",
        }
    selections = selected["selections"]
    rbf_selection = selections.get("camera_rbf")
    rbf_formal = bool(
        isinstance(rbf_selection, Mapping)
        and rbf_selection.get("status") == "selected"
        and rbf_selection.get("diagnostic_only") is not True
        and isinstance(rbf_selection.get("configuration"), Mapping)
    )
    records: List[Dict[str, Any]] = []
    for topology in ("nested_tree_a", "nested_tree_ab"):
        selection = selections.get(topology)
        formal = bool(
            isinstance(selection, Mapping)
            and selection.get("status") == "selected"
            and selection.get("diagnostic_only") is not True
        )
        if not formal:
            continue
        method = "selected_{}".format(topology)
        row = dict(_nested_fairness(method, selected))
        row.update(
            {
                "topology": topology,
                "method": method,
                "selection_status": selection.get("status"),
                "diagnostic_only": False,
            }
        )
        if not rbf_formal:
            row.update(
                {
                    "valid": None,
                    "reason": "selected_camera_rbf_not_formal_or_missing",
                }
            )
        records.append(row)
    return {
        "complete": all(row.get("valid") is True for row in records),
        "formal_comparison_count": len(records),
        "records": records,
        "reason": (
            "all_formal_nested_comparisons_equal_rank_equal_target_kl"
            if all(row.get("valid") is True for row in records)
            else "formal_nested_vs_rbf_fairness_invalid"
        ),
    }


def _method_selection_eligible(method: str, selected: Any) -> Optional[bool]:
    if not isinstance(selected, Mapping):
        return None
    selections = selected.get("selections")
    if not isinstance(selections, Mapping):
        return None
    kind = _method_kind(method)
    if kind == "shared_full":
        return True
    if kind == "rbf":
        topology = "camera_rbf"
    elif kind == "nested":
        topology = "nested_tree_ab" if "tree_ab" in method.lower() else "nested_tree_a"
    else:
        return True
    row = selections.get(topology)
    if not isinstance(row, Mapping):
        return None
    return bool(
        row.get("status") == "selected"
        and row.get("diagnostic_only") is not True
    )


def classify_scientific_result(
    *,
    full_complete: bool,
    met3r_complete: bool,
    trajectory_complete: bool,
    full_metrics: Any,
    trajectory: Any,
    selection_policy: Optional[Mapping[str, Any]] = None,
    selected_candidates: Any = None,
) -> Dict[str, Any]:
    """Return a conservative, machine-readable scientific judgment."""

    if not full_complete or not met3r_complete or not trajectory_complete:
        return {
            "label": "full_blocked",
            "rationale": "FULL, required MEt3R, or trajectory evidence is incomplete.",
            "evidence": {},
        }
    samples = full_metrics.get("samples", []) if isinstance(full_metrics, Mapping) else []
    evidence = _paired_evidence(samples if isinstance(samples, list) else [])
    statistics = _comparison_statistics(full_metrics)
    formal_rows = statistics["rows"]
    formal_is_authoritative = bool(statistics["authoritative"])
    policy = dict(selection_policy or {})
    trajectory_state = _trajectory_state(trajectory)
    trajectory_evidence = _method_trajectory_evidence(trajectory)
    summaries: Dict[str, Any] = {}
    safe: Dict[str, Optional[bool]] = {}
    eligible: Dict[str, Optional[bool]] = {}
    fairness: Dict[str, Mapping[str, Any]] = {}
    method_trajectory: Dict[str, Optional[Mapping[str, Any]]] = {}
    for method, values in evidence["deltas"].items():
        raw_iid = _mean_ci(values["vs_iid"])
        raw_rbf = _mean_ci(values["vs_rbf"])
        formal_iid = _formal_summary_for_method(formal_rows, method, "iid")
        formal_rbf = _formal_summary_for_method(formal_rows, method, "rbf")
        summaries[method] = {
            "kind": _method_kind(method),
            "vs_iid": (
                formal_iid
                if formal_iid is not None
                else None if formal_is_authoritative else raw_iid
            ),
            "vs_rbf": (
                formal_rbf
                if formal_rbf is not None
                else None if formal_is_authoritative else raw_rbf
            ),
        }
        safe[method] = _guardrails_status(evidence["rows"].get(method, []), policy)
        eligible[method] = _method_selection_eligible(method, selected_candidates)
        kind = _method_kind(method)
        if kind == "nested":
            fairness[method] = _nested_fairness(method, selected_candidates)
        if kind in {"rbf", "nested"}:
            method_trajectory[method] = _trajectory_for_method(
                method, trajectory_evidence, trajectory_state
            )
    kinds = {row["kind"] for row in summaries.values()}
    required_pairing_present = (
        "rbf" in kinds
        and "nested" in kinds
        and any(
            row["kind"] == "rbf" and row["vs_iid"] is not None
            for row in summaries.values()
        )
        and any(
            row["kind"] == "nested"
            and row["vs_iid"] is not None
            and row["vs_rbf"] is not None
            for row in summaries.values()
        )
    )
    if (
        not summaries
        or not required_pairing_present
        or trajectory_state is None
        or any(value is None for value in safe.values())
        or any(value is None for value in eligible.values())
        or any(
            row.get("valid") is None for row in fairness.values()
        )
        or any(value is None for value in method_trajectory.values())
    ):
        return {
            "label": "full_blocked",
            "rationale": "Required paired samples, guardrails, or trajectory classification are missing.",
            "evidence": {
                "paired": summaries,
                "guardrails_safe": safe,
                "selection_eligible": eligible,
                "trajectory_state": trajectory_state,
                "method_trajectory": method_trajectory,
                "trajectory_evidence_source": trajectory_evidence["source"],
                "equal_rank_kl_fairness": fairness,
                "statistics_source": statistics["source"],
            },
        }
    nested_positive = any(
        row["kind"] == "nested"
        and _supported_improvement(row["vs_iid"])
        and _supported_improvement(row["vs_rbf"])
        and safe.get(method, False)
        and eligible.get(method, False)
        and fairness.get(method, {}).get("valid") is True
        and method_trajectory.get(method, {}).get("state") == "retained"
        for method, row in summaries.items()
    )
    if nested_positive:
        label = "nested_positive"
        rationale = "At least one nested topology improves on paired IID and equal-rank/equal-target-KL RBF while measured FULL guardrails remain within policy and that method's correlation is retained."
    else:
        rbf_positive = any(
            row["kind"] == "rbf"
            and _supported_improvement(row["vs_iid"])
            and safe.get(method, False)
            and eligible.get(method, False)
            and method_trajectory.get(method, {}).get("state") == "retained"
            for method, row in summaries.items()
        )
        if rbf_positive:
            label = "generic_coupling_only"
            rationale = "Camera-RBF coupling improves on IID, but nested topology does not meet the independent-advantage criterion."
        else:
            label = "initial_noise_no_go"
            rationale = "The predefined positive criteria are not met by the complete paired evidence; no improvement is claimed."
    return {
        "label": label,
        "rationale": rationale,
        "evidence": {
            "paired": summaries,
            "guardrails_safe": safe,
            "selection_eligible": eligible,
            "trajectory_state": trajectory_state,
            "method_trajectory": method_trajectory,
            "trajectory_evidence_source": trajectory_evidence["source"],
            "equal_rank_kl_fairness": fairness,
            "statistics_source": statistics["source"],
        },
    }


def _summary_rows(payload: Any) -> List[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("configuration_summaries", payload.get("paired_statistics", []))
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _evaluation_visual_summary(payload: Any) -> Dict[str, Any]:
    plots = payload.get("plots", {}) if isinstance(payload, Mapping) else {}
    contacts = (
        payload.get("contact_sheets", {})
        if isinstance(payload, Mapping)
        else {}
    )
    if not isinstance(plots, Mapping):
        plots = {}
    if not isinstance(contacts, Mapping):
        contacts = {}
    plot_artifacts = plots.get("artifacts", [])
    contact_artifacts = contacts.get("artifacts", [])
    return {
        "plots": {
            "complete": plots.get("complete"),
            "directory": plots.get("plots_dir"),
            "artifacts": (
                list(plot_artifacts)
                if isinstance(plot_artifacts, list)
                else []
            ),
        },
        "contact_sheets": {
            "complete": contacts.get("complete"),
            "directory": contacts.get("directory"),
            "artifacts": (
                list(contact_artifacts)
                if isinstance(contact_artifacts, list)
                else []
            ),
        },
        "paired_sheet_count": contacts.get("paired_sheet_count"),
        "failure_row_count": contacts.get("failure_row_count"),
        "failure_gallery": contacts.get("failure_gallery"),
    }


def _recorded_failure_cases(
    stage_payloads: Mapping[str, Any], *, limit: int = 50
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for stage in ("pilot", "full"):
        payload = stage_payloads.get(stage)
        samples = payload.get("samples", []) if isinstance(payload, Mapping) else []
        if not isinstance(samples, list):
            continue
        for row in samples:
            if not isinstance(row, Mapping):
                continue
            reasons: List[str] = []
            status = str(row.get("status") or "").lower()
            generation_status = str(
                row.get("generation_status") or ""
            ).lower()
            if status in {"failed", "error"}:
                reasons.append("status_{}".format(status))
            if generation_status in {"failed", "error"}:
                reasons.append(
                    "generation_status_{}".format(generation_status)
                )
            if bool(row.get("guardrail_error")):
                reasons.append("guardrail_error")
            if bool(row.get("artifact_failure", False)):
                reasons.append("artifact_failure")
            if row.get("collapse_detector_label") == "view_collapse_alert":
                reasons.append("view_collapse_alert")
            if not reasons:
                continue
            records.append(
                {
                    "stage": stage,
                    "input_sha256": row.get("input_sha256")
                    or row.get("input_hash"),
                    "input_image": row.get("input_image"),
                    "seed": row.get("seed"),
                    "method": row.get("method"),
                    "config_id": row.get("config_id"),
                    "status": row.get("status"),
                    "generation_status": row.get("generation_status"),
                    "reasons": reasons,
                    "guardrail_error": row.get("guardrail_error"),
                    "artifact_failure": bool(
                        row.get("artifact_failure", False)
                    ),
                    "collapse_detector_label": row.get(
                        "collapse_detector_label"
                    ),
                }
            )
    return {
        "total_count": len(records),
        "reported_count": min(len(records), limit),
        "truncated": len(records) > limit,
        "limit": limit,
        "records": records[:limit],
    }


def _markdown_path_link(value: Any) -> Any:
    if value in (None, ""):
        return None
    raw = str(value)
    target = (
        raw.replace("\\", "/")
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )
    label = Path(raw).name or raw
    return "[{}]({})".format(label, target)


def _markdown_path_links(values: Any) -> Any:
    if not isinstance(values, list) or not values:
        return None
    return "<br>".join(
        str(_markdown_path_link(value))
        for value in values
        if value not in (None, "")
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "不可用"
    if isinstance(value, float):
        return "{:.6g}".format(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "数据 artifact 缺失或没有可报告记录；未估算任何指标。"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _build_markdown(report: Mapping[str, Any]) -> str:
    completion = report["completion"]
    counts = report["run_counts"]
    artifacts = report["artifacts"]
    selected = report.get("selected_candidates", {})
    selections = selected.get("selections", {}) if isinstance(selected, Mapping) else {}
    selected_rows = []
    for topology in ("camera_rbf", "nested_tree_a", "nested_tree_ab"):
        row = selections.get(topology, {}) if isinstance(selections, Mapping) else {}
        config = row.get("configuration", {}) if isinstance(row, Mapping) else {}
        selected_rows.append(
            (topology, row.get("status"), row.get("diagnostic_only"), config.get("rank"), config.get("target_kl"), config.get("achieved_kl"), config.get("alpha"), config.get("rbf_length_scale_deg"))
        )
    pilot_rows = [
        (row.get("method"), row.get("rank"), row.get("target_kl"), row.get("alpha"), row.get("met3r_all_pair_mean"), row.get("dino_identity_mean_delta"), row.get("r_hf"))
        for row in report.get("pilot_summaries", [])
    ]
    def statistic_row(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            row.get("method"),
            row.get("comparison_baseline"),
            row.get("config_id"),
            row.get("pair_count"),
            row.get("object_cluster_count"),
            row.get("mean_delta"),
            row.get("median_delta"),
            row.get("std_delta"),
            row.get("win_rate"),
            row.get("bootstrap_95_ci"),
            row.get("wilcoxon_p"),
            row.get("holm_bonferroni_p"),
            row.get("effect_size_dz"),
        )

    iid_statistic_rows = [
        statistic_row(row) for row in report.get("full_iid_statistics", [])
    ]
    rbf_statistic_rows = [
        statistic_row(row)
        for row in report.get("full_nested_vs_rbf_statistics", [])
    ]
    statistics_schema = report.get("full_statistics_schema", {})
    judgment_evidence = report.get("scientific_judgment", {}).get(
        "evidence", {}
    )
    fairness_evidence = judgment_evidence.get(
        "equal_rank_kl_fairness", {}
    )
    if not fairness_evidence:
        fairness_evidence = {
            str(row.get("method") or row.get("topology")): row
            for row in report.get(
                "formal_equal_rank_kl_audit", {}
            ).get("records", [])
            if isinstance(row, Mapping)
        }
    fairness_rows = [
        (
            method,
            row.get("nested_rank"),
            row.get("rbf_rank"),
            row.get("nested_target_kl"),
            row.get("rbf_target_kl"),
            row.get("equal_rank"),
            row.get("equal_target_kl"),
            row.get("valid"),
            row.get("reason"),
        )
        for method, row in fairness_evidence.items()
        if isinstance(row, Mapping)
    ]
    method_trajectory_rows = [
        (
            method,
            row.get("state"),
            row.get("final_g_t"),
            row.get("pair_count"),
            row.get("rank"),
            row.get("legacy_aggregate_fallback", False),
        )
        for method, row in judgment_evidence.get(
            "method_trajectory", {}
        ).items()
        if isinstance(row, Mapping)
    ]
    visual_rows = []
    for stage in ("pilot", "full"):
        visual = report.get("evaluation_visual_artifacts", {}).get(
            stage, {}
        )
        plots = visual.get("plots", {}) if isinstance(visual, Mapping) else {}
        contacts = (
            visual.get("contact_sheets", {})
            if isinstance(visual, Mapping)
            else {}
        )
        visual_rows.append(
            (
                stage,
                plots.get("complete"),
                _markdown_path_link(plots.get("directory")),
                len(plots.get("artifacts", []))
                if isinstance(plots.get("artifacts"), list)
                else 0,
                _markdown_path_links(plots.get("artifacts")),
                contacts.get("complete"),
                _markdown_path_link(contacts.get("directory")),
                visual.get("paired_sheet_count"),
                _markdown_path_links(contacts.get("artifacts")),
                visual.get("failure_row_count"),
                _markdown_path_link(visual.get("failure_gallery")),
            )
        )
    failure_summary = report.get("failure_cases", {})
    failure_rows = [
        (
            row.get("stage"),
            row.get("input_sha256") or row.get("input_image"),
            row.get("seed"),
            row.get("method"),
            row.get("config_id"),
            row.get("status"),
            row.get("generation_status"),
            row.get("reasons"),
            row.get("guardrail_error"),
            row.get("artifact_failure"),
            row.get("collapse_detector_label"),
        )
        for row in failure_summary.get("records", [])
        if isinstance(row, Mapping)
    ]
    gate_rows = [
        (
            row.get("config_id"),
            row.get("method"),
            row.get("rank"),
            row.get("target_kl"),
            row.get("achieved_kl"),
            row.get("alpha"),
            row.get("passed"),
            row.get("covariance_mae"),
        )
        for row in report.get("distribution_gate_summaries", [])
    ]
    input_summary = report.get("input_summary", {})
    blockers = report.get("blockers", [])
    label = report["scientific_judgment"]["label"]
    if label in {"nested_positive", "generic_coupling_only"}:
        next_steps = (
            "- Evaluate strict SZ sample-budget scheduling under the same frozen protocol.\n"
            "- Test scheduler variance-noise coupling without changing the initial-noise result."
        )
    else:
        next_steps = (
            "Not proposed: the current scientific classification does not support "
            "expanding to strict SZ scheduling or scheduler variance noise."
        )
    blocker_text = "\n".join("- `{}`: {}".format(item["code"], item["detail"]) for item in blockers) or "- 无已记录 blocker。"
    return """# NILE Low-Rank Equal-KL Study Report

生成时间：`{generated}`

> {statement}

## Mathematical construction and fairness

For IID latent vectors `Z`, the method projects only onto a deterministic
orthonormal DCT-II basis `B`: `A = ZB`, `R = Z - AB^T`, and
`Z' = R + (LA)B^T`, where `LL^T = K_view`. Each view therefore retains an
exact `N(0, I)` marginal; there is no per-sample standardization. Identity
mixing calibrates `alpha` against the complete joint budget
`KL = K/2 * (trace(K_view) - logdet(K_view) - V)`. Comparisons use equal basis
rank and equal target joint KL, never equal raw correlation strength.

All paired methods share the input, seed, camera angles, model revisions,
scheduler, 30 denoising steps, guidance, resolution, reference-VAE random
stream, and scheduler random stream. The initial-noise stream is separate from
both other random streams.

## 摘要

科学判定：`{label}`。{rationale}

本报告只汇总已落盘 artifact。缺失值保持为“不可用”，没有补造、插值或推断实验指标。`shared_full` 仅是退化联合分布诊断上限，不代表正确的三维一致性。

## 完成状态

{completion_table}

## 阻塞项

{blockers}

## Artifact 清单

{artifact_table}

## 运行计数

{count_table}

## Input and distribution gates

Inputs are content-hashed, perceptual/rotation duplicates are excluded, and
the SHA-256 ordered PILOT and FULL splits are disjoint. An unattainable KL
budget is explicitly excluded before sampling; it is never silently changed.

{input_table}

{gate_table}

## PILOT 候选指标

MEt3R 为主指标且越低越好；DINO/轮廓为 guardrail，`R_HF` 仅作 collapse 诊断。

{pilot_table}

## 冻结候选

FULL 只允许读取冻结配置，不允许在 FULL 数据上重新调参。

{selected_table}

## FULL 配对统计

Delta 定义为方法减去配对 IID；MEt3R delta 为负表示更好。只有 artifact 中实际存在的统计量才显示。

Statistics source: `{statistics_source}`; Holm scope:
`{holm_scope}`. The current schema applies one Holm correction family across
IID and nested-vs-RBF comparisons.

### Methods vs IID

{iid_statistics_table}

### Nested methods vs selected camera RBF

{rbf_statistics_table}

### Equal-rank/equal-target-KL fairness audit

{fairness_table}

### Method-level trajectory evidence

Source: `{trajectory_evidence_source}`.

{method_trajectory_table}

## Trajectory

相关性状态：`{trajectory_state}`。Trajectory observer 是只读诊断，不修改 latent，也不替代 scheduler stochastic noise。

## 科学结论

`{label}`：{rationale}

旧版 scalar Sobol reshape、频谱不匹配 low-pass 与 latent-state averaging callback 仍只作为失败分析保留，未作为正式方法复用。本研究不声称实现 strict NILE/SZ。
## Failure cases and evidence limits

### Evaluator visual artifacts

{visual_artifact_table}

### Explicitly recorded sample failures

Recorded total: {failure_total}; shown: {failure_reported}; truncated:
{failure_truncated}.

{failure_case_table}

The blocker list and failed manifest records above are the complete recorded
failure set. Missing images, MEt3R pairs, identity scores, masks, trajectories,
or statistics are not assigned zero and are not estimated. Artifact masks are
guardrails for fragments, blobs, and tails; they are not standalone 3D quality
metrics. `R_HF > 1` and angle monotonicity are not interpreted as proof of
better geometry.

## Conditional next steps

{next_steps}
""".format(
        generated=report["generated_at_utc"],
        next_steps=next_steps,
        statement=STRICT_STATEMENT,
        label=report["scientific_judgment"]["label"],
        rationale=report["scientific_judgment"]["rationale"],
        completion_table=_markdown_table(("项目", "完成"), [(key, value) for key, value in completion.items()]),
        blockers=blocker_text,
        artifact_table=_markdown_table(("artifact", "路径", "存在", "解析错误"), [(key, value.get("path"), value.get("present"), value.get("parse_error")) for key, value in artifacts.items()]),
        count_table=_markdown_table(("阶段", "计划记录", "成功", "失败", "阻塞", "期望"), [(stage, value["planned"], value["succeeded"], value["failed"], value["blocked"], value.get("expected")) for stage, value in counts.items()]),
        pilot_table=_markdown_table(("method", "rank", "target KL", "alpha", "MEt3R", "identity delta", "R_HF"), pilot_rows),
        selected_table=_markdown_table(("topology", "status", "diagnostic only", "rank", "target KL", "achieved KL", "alpha", "RBF ell"), selected_rows),
        statistics_source=statistics_schema.get("source") or "unavailable",
        holm_scope=statistics_schema.get("holm_scope") or "unavailable",
        iid_statistics_table=_markdown_table(
            (
                "method",
                "baseline",
                "config",
                "pairs",
                "object clusters",
                "mean delta",
                "median delta",
                "std delta",
                "win rate",
                "95% cluster-bootstrap CI",
                "Wilcoxon p",
                "global Holm p",
                "effect size dz",
            ),
            iid_statistic_rows,
        ),
        rbf_statistics_table=_markdown_table(
            (
                "nested method",
                "selected RBF baseline",
                "config",
                "pairs",
                "object clusters",
                "mean delta",
                "median delta",
                "std delta",
                "win rate",
                "95% cluster-bootstrap CI",
                "Wilcoxon p",
                "global Holm p",
                "effect size dz",
            ),
            rbf_statistic_rows,
        ),
        fairness_table=_markdown_table(
            (
                "nested method",
                "nested rank",
                "RBF rank",
                "nested target KL",
                "RBF target KL",
                "equal rank",
                "equal target KL",
                "fair comparison",
                "reason",
            ),
            fairness_rows,
        ),
        trajectory_evidence_source=judgment_evidence.get(
            "trajectory_evidence_source"
        ) or "unavailable",
        method_trajectory_table=_markdown_table(
            (
                "method",
                "state",
                "final G_t",
                "pairs",
                "rank",
                "legacy aggregate fallback",
            ),
            method_trajectory_rows,
        ),
        visual_artifact_table=_markdown_table(
            (
                "stage",
                "plots complete",
                "plots directory",
                "plot count",
                "plot artifacts",
                "contact sheets complete",
                "contact sheets directory",
                "paired sheet count",
                "contact sheet artifacts",
                "failure row count",
                "failure gallery",
            ),
            visual_rows,
        ),
        failure_total=failure_summary.get("total_count", 0),
        failure_reported=failure_summary.get("reported_count", 0),
        failure_truncated=failure_summary.get("truncated", False),
        failure_case_table=_markdown_table(
            (
                "stage",
                "input",
                "seed",
                "method",
                "config",
                "status",
                "generation status",
                "recorded reasons",
                "guardrail error",
                "artifact failure",
                "collapse detector",
            ),
            failure_rows,
        ),
        input_table=_markdown_table(
            ("input field", "value"),
            [(key, value) for key, value in input_summary.items()],
        ),
        gate_table=_markdown_table(
            (
                "config",
                "method",
                "rank",
                "target KL",
                "achieved KL",
                "alpha",
                "passed",
                "covariance MAE",
            ),
            gate_rows,
        ),
        trajectory_state=report["scientific_judgment"].get("evidence", {}).get("trajectory_state") or "不可用",
    )


def _credential_safe_repository(value: Any) -> Optional[str]:
    """Remove URL credentials/query data before rendering a repository."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        host = parsed.hostname
        if parsed.port is not None:
            host = "{}:{}".format(host, parsed.port)
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return None


def _build_reproduce(root: Path, config_path: Optional[Path]) -> str:
    frozen_config = config_path or root / "configs" / "resolved_config.json"
    resolved: Mapping[str, Any] = {}
    if frozen_config.is_file():
        try:
            loaded = _load_structured(frozen_config)
            if isinstance(loaded, Mapping):
                resolved = loaded
        except Exception:
            resolved = {}
    evaluation = resolved.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        evaluation = {}
    met3r_repository = _credential_safe_repository(
        evaluation.get("met3r_repository")
    )
    met3r_revision = evaluation.get("met3r_revision")
    immutable_met3r = bool(
        met3r_repository
        and isinstance(met3r_revision, str)
        and len(met3r_revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in met3r_revision)
    )
    if immutable_met3r:
        met3r_install = "python -m pip install {}".format(
            shlex.quote(
                "git+{}@{}".format(
                    str(met3r_repository).rstrip("/"), met3r_revision
                )
            )
        )
        met3r_display = str(met3r_revision)
    else:
        met3r_install = (
            'echo "BLOCKED: resolved config lacks a safe repository and immutable '
            'evaluation.met3r_revision"; exit 2'
        )
        met3r_display = "<missing immutable revision in resolved config>"
    artifact_root = shlex.quote(str(root))
    config = shlex.quote(str(frozen_config))
    checkpoint_manifest = shlex.quote(
        str(root / "configs" / "checkpoint_manifest.json")
    )
    config_lock = shlex.quote(str(root / "configs" / "config_lock.json"))
    input_hashes = shlex.quote(str(root / "inputs" / "input_validation.json"))
    candidate_file = shlex.quote(
        str(root / "selected_candidates" / "selected_candidates.json")
    )
    return """# Reproduce NILE Low-Rank Study

## Recommended: Colab Run All

Open notebooks/mvadapter_nile_lowrank_full_colab.ipynb in a GPU Colab runtime,
edit only its configuration cell, mount the same Google Drive project, and use
Runtime > Run all. The notebook installs dependencies, resolves immutable model
revisions, writes the checkpoint manifest and input hashes, freezes PILOT
candidates, records tests, resumes every stage, and regenerates this report.

Put HF_TOKEN in Colab Secrets or the process environment only when access
requires it. Never paste a token into this notebook, a command, the resolved
config, checkpoint manifest, report, or logs; the token value must not be
printed.

## CLI exact-resume path

Start from a clean checkout matching environment/git_commit.txt and
environment/worktree.diff. Copy or mount the same artifact root and original
input files. Do not re-resolve model revisions, reorder inputs, regenerate the
candidate selection, or edit the frozen config in place.

Pinned MEt3R revision read from the frozen resolved config:
{met3r_revision}

~~~bash
set -euo pipefail
export ARTIFACT_ROOT={artifact_root}
FROZEN_CONFIG={config}
CONFIG_LOCK={config_lock}
CHECKPOINT_MANIFEST={checkpoint_manifest}
INPUT_HASH_MANIFEST={input_hashes}
CANDIDATE_FILE={candidate_file}

python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
{met3r_install}

for required in "$FROZEN_CONFIG" "$CONFIG_LOCK" "$CHECKPOINT_MANIFEST" "$INPUT_HASH_MANIFEST" "$CANDIDATE_FILE"; do
  test -s "$required" || {{ echo "BLOCKED: missing frozen artifact $required"; exit 2; }}
done

python -m compileall mvadapter scripts
python -m pytest -q
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

destination = Path(os.environ["ARTIFACT_ROOT"]) / "environment" / "test_results.json"
destination.parent.mkdir(parents=True, exist_ok=True)
payload = {{
    "schema_version": 1,
    "passed": True,
    "tests_complete": True,
    "compileall_returncode": 0,
    "pytest_returncode": 0,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "command": [
        ["python", "-m", "compileall", "mvadapter", "scripts"],
        ["python", "-m", "pytest", "-q"],
    ],
}}
temporary = destination.with_name(destination.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
os.replace(temporary, destination)
PY

python -m scripts.run_nile_lowrank_study --config "$FROZEN_CONFIG" --stage all --resume
python -m scripts.report_nile_lowrank_study --artifact-root "$ARTIFACT_ROOT"
~~~

The checkpoint manifest supplies the frozen repository revisions and adapter
SHA-256; the input validation artifact supplies the ordered content hashes; the
candidate file freezes PILOT selection for FULL. The original images must still
match those hashes. MEt3R is mandatory for strict FULL completion.

Missing credentials, CUDA, weights, inputs, MEt3R, or frozen artifacts remain
explicit blockers. Re-running these commands never converts absent evidence
into a completed result.
""".format(
        artifact_root=artifact_root,
        candidate_file=candidate_file,
        checkpoint_manifest=checkpoint_manifest,
        config=config,
        config_lock=config_lock,
        input_hashes=input_hashes,
        met3r_install=met3r_install,
        met3r_revision=met3r_display,
    )


def generate_report(
    artifact_root: Path,
    *,
    output_dir: Optional[Path] = None,
    paths: Optional[Mapping[str, Optional[Path]]] = None,
    implementation_complete: Optional[bool] = None,
    tests_complete: Optional[bool] = None,
) -> Dict[str, Any]:
    root = artifact_root.expanduser().resolve()
    destination = (output_dir or root).expanduser().resolve()
    overrides = dict(paths or {})
    resolved_paths = {
        name: _resolve_artifact(root, name, overrides.get(name))
        for name in DEFAULT_CANDIDATES
    }
    payloads: Dict[str, Any] = {}
    errors: Dict[str, Optional[str]] = {}
    manifest_names = {
        "manifest",
        "pilot_manifest",
        "full_manifest",
        "trajectory_manifest",
    }
    for name, path in resolved_paths.items():
        payloads[name], errors[name] = _load_artifact(
            path, manifest=name in manifest_names
        )
    manifest_records: List[Dict[str, Any]] = []
    manifest_metas: List[Dict[str, Any]] = []
    seen_run_ids = set()
    for name in ("manifest", "pilot_manifest", "full_manifest", "trajectory_manifest"):
        loaded = payloads.get(name)
        if loaded is None:
            continue
        rows, metadata = loaded
        manifest_metas.append(metadata)
        implied_stage = name.removesuffix("_manifest") if name != "manifest" else None
        for original in rows:
            row = dict(original)
            if implied_stage is not None and not any(
                row.get(field) for field in ("stage", "split", "phase", "run_type")
            ):
                row["split"] = implied_stage
            run_id = row.get("run_id")
            stage_identity = implied_stage or next(
                (
                    str(row.get(field))
                    for field in ("stage", "split", "phase", "run_type")
                    if row.get(field)
                ),
                "",
            )
            identity = (
                (stage_identity, str(run_id))
                if run_id is not None
                else (stage_identity, json.dumps(row, sort_keys=True, default=str))
            )
            if identity in seen_run_ids:
                continue
            seen_run_ids.add(identity)
            manifest_records.append(row)
    manifest_is_blocked = any(
        bool(metadata.get("blocked"))
        or str(metadata.get("status", "")).lower() == "blocked"
        for metadata in manifest_metas
    )
    config = payloads["resolved_config"]
    config_lock = payloads["config_lock"]
    expected_config_hash = _configuration_hash(config)
    config_lock_ok = bool(
        isinstance(config_lock, Mapping)
        and expected_config_hash is not None
        and config_lock.get("config_hash") == expected_config_hash
    )
    inputs = payloads["input_validation"]
    preflight_ok = _preflight_passed(payloads["preflight"])
    selection_ok = _selection_complete(payloads["selected_candidates"])
    pilot_expected = _expected_runs(
        config, inputs, "pilot", preflight=payloads["preflight"]
    )
    full_expected = _expected_runs(
        config,
        inputs,
        "full",
        selected_candidates=payloads["selected_candidates"],
    )
    pilot_counts = _run_counts(manifest_records, "pilot")
    full_counts = _run_counts(manifest_records, "full")
    pilot_counts["expected"] = pilot_expected
    full_counts["expected"] = full_expected
    pilot_runs_ok = _stage_complete(manifest_records, "pilot", pilot_expected)
    full_runs_ok = _stage_complete(manifest_records, "full", full_expected)
    formal_ready = bool(isinstance(inputs, Mapping) and inputs.get("formal_ready") is True)
    pilot_met3r = _met3r_complete(payloads["pilot_metrics"])
    full_met3r = _met3r_complete(payloads["full_metrics"])
    comparison_statistics = _comparison_statistics(payloads["full_metrics"])
    comparison_statistics_ok = _comparison_statistics_complete(
        comparison_statistics
    )
    formal_fairness_audit = _formal_equal_rank_kl_audit(
        payloads["selected_candidates"]
    )
    formal_fairness_ok = bool(
        not comparison_statistics["authoritative"]
        or formal_fairness_audit["complete"]
    )
    trajectory_explicit = _explicit_boolean((payloads["trajectory"],), "trajectory_complete")
    trajectory_ok = bool(
        trajectory_explicit
        if trajectory_explicit is not None
        else isinstance(payloads["trajectory"], Mapping)
        and payloads["trajectory"].get("complete") is True
        and _trajectory_state(payloads["trajectory"]) is not None
    )
    workflow = payloads["workflow_status"]
    test_status = payloads["test_status"]
    if implementation_complete is None:
        implementation_complete = _explicit_boolean((workflow,), "implementation_complete")
    if tests_complete is None:
        tests_complete = _explicit_boolean((test_status, workflow), "tests_complete")
        if tests_complete is None and isinstance(test_status, Mapping):
            tests_complete = bool(
                test_status.get("passed")
                and test_status.get("compileall_returncode") == 0
                and test_status.get("pytest_returncode") == 0
            )
    implementation_complete = bool(implementation_complete)
    tests_complete = bool(tests_complete)
    workflow_blockers = (
        workflow.get("blockers", []) if isinstance(workflow, Mapping) else []
    )
    workflow_blockers = (
        [item for item in workflow_blockers if isinstance(item, Mapping)]
        if isinstance(workflow_blockers, list)
        else []
    )
    pilot_status = _explicit_boolean((workflow,), "pilot_complete")
    full_status = _explicit_boolean((workflow,), "full_complete")
    met3r_status = _explicit_boolean((workflow,), "met3r_complete")
    pilot_complete = bool(
        preflight_ok
        and config_lock_ok
        and pilot_runs_ok
        and pilot_met3r
        and selection_ok
        and pilot_status is not False
        and not manifest_is_blocked
        and not workflow_blockers
    )
    full_complete = bool(
        formal_ready
        and config_lock_ok
        and full_runs_ok
        and payloads["full_metrics"] is not None
        and full_met3r
        and comparison_statistics_ok
        and formal_fairness_ok
        and full_status is not False
        and not manifest_is_blocked
        and not workflow_blockers
    )
    met3r_complete = bool(
        pilot_met3r and full_met3r and met3r_status is not False
    )
    policy = _nested_get(config, "selection", default={}) if isinstance(config, Mapping) else {}
    judgment = classify_scientific_result(
        full_complete=full_complete,
        met3r_complete=met3r_complete,
        trajectory_complete=trajectory_ok,
        full_metrics=payloads["full_metrics"],
        trajectory=payloads["trajectory"],
        selection_policy=policy if isinstance(policy, Mapping) else {},
        selected_candidates=payloads["selected_candidates"],
    )
    blockers: List[Dict[str, str]] = []

    def block(code: str, detail: str) -> None:
        if code not in {item["code"] for item in blockers}:
            blockers.append({"code": code, "detail": detail})

    for item in workflow_blockers:
        code = str(item.get("code", "runtime_blocker"))
        detail = item.get("detail")
        if detail is None:
            detail = json.dumps(dict(item), ensure_ascii=False, sort_keys=True)
        block(code, str(detail))
    for name, error in errors.items():
        if error:
            block("{}_parse_error".format(name), error)
    if config is None:
        block("resolved_config_missing", "No readable resolved configuration was found.")
    if not config_lock_ok:
        block(
            "config_lock_missing_or_mismatched",
            "config_lock.json is missing or does not match the resolved configuration hash.",
        )
    if inputs is None:
        block("input_validation_missing", "Input validation artifact is missing.")
    elif not formal_ready:
        missing = inputs.get("missing_distinct_inputs") if isinstance(inputs, Mapping) else None
        block("formal_inputs_not_ready", "Formal input requirements are not met; missing distinct inputs: {}.".format(_fmt(missing)))
    if not preflight_ok:
        block("distribution_preflight_incomplete", "Distribution gates are missing or did not all pass.")
    if not manifest_records:
        block("manifest_missing", "No readable run manifest records were found.")
    if manifest_is_blocked:
        block("manifest_blocked", "The manifest records the workflow as blocked.")
    if not pilot_runs_ok:
        block("pilot_runs_incomplete", "PILOT manifest runs are absent, failed, blocked, or below the expected count.")
    if not pilot_met3r:
        block("pilot_met3r_incomplete", "PILOT MEt3R evidence is missing or incomplete.")
    if not selection_ok:
        block("candidate_selection_incomplete", "Three topology candidates were not deterministically frozen.")
    if not full_runs_ok:
        block("full_runs_incomplete", "FULL manifest runs are absent, failed, blocked, or below the expected count.")
    if payloads["full_metrics"] is None:
        block("full_metrics_missing", "FULL metrics artifact is missing.")
    if not full_met3r:
        block("full_met3r_incomplete", "FULL MEt3R evidence is missing or incomplete.")
    if not comparison_statistics_ok:
        block(
            "paired_comparison_statistics_incomplete",
            "Current-schema IID and nested-vs-RBF comparisons, cluster-bootstrap CI, or Holm fields are incomplete.",
        )
    if not formal_fairness_ok:
        block(
            "equal_rank_kl_fairness_mismatch",
            "A formal selected nested-vs-selected-RBF comparison is missing or does not use equal rank and equal target joint KL.",
        )
    if not trajectory_ok:
        block("trajectory_incomplete", "Trajectory summary/classification is missing or incomplete.")
    if not implementation_complete:
        block("implementation_unverified", "Implementation completion was not explicitly verified.")
    if not tests_complete:
        block("tests_unverified", "Passing test completion was not explicitly recorded.")
    completion = {
        "implementation_complete": implementation_complete,
        "tests_complete": tests_complete,
        "pilot_complete": pilot_complete,
        "full_complete": full_complete,
        "met3r_complete": met3r_complete,
        "trajectory_complete": trajectory_ok,
        "report_complete": True,
    }
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(root),
        "statement": STRICT_STATEMENT,
        "completion": completion,
        "overall_complete": all(completion.values()),
        "blockers": blockers,
        "artifacts": {
            name: {
                "path": str(resolved_paths[name]) if resolved_paths[name] is not None else None,
                "present": bool(resolved_paths[name] is not None and resolved_paths[name].is_file()),
                "parse_error": errors[name],
            }
            for name in DEFAULT_CANDIDATES
        },
        "input_summary": {
            key: inputs.get(key)
            for key in ("distinct_count", "pilot_count", "full_count", "missing_distinct_inputs", "formal_ready")
            if isinstance(inputs, Mapping) and key in inputs
        },
        "distribution_preflight_passed": preflight_ok,
        "distribution_gate_summaries": (
            [
                row
                for row in payloads["preflight"].get("configurations", [])
                if isinstance(row, Mapping)
            ]
            if isinstance(payloads["preflight"], Mapping)
            and isinstance(
                payloads["preflight"].get("configurations", []), list
            )
            else []
        ),
        "run_counts": {"pilot": pilot_counts, "full": full_counts},
        "selected_candidates": payloads["selected_candidates"] or {},
        "pilot_summaries": _summary_rows(payloads["pilot_metrics"]),
        "evaluation_visual_artifacts": {
            "pilot": _evaluation_visual_summary(
                payloads["pilot_metrics"]
            ),
            "full": _evaluation_visual_summary(
                payloads["full_metrics"]
            ),
        },
        "failure_cases": _recorded_failure_cases(
            {
                "pilot": payloads["pilot_metrics"],
                "full": payloads["full_metrics"],
            }
        ),
        "full_statistics_schema": {
            "source": comparison_statistics["source"],
            "authoritative": comparison_statistics["authoritative"],
            "complete": comparison_statistics_ok,
            "row_count": len(comparison_statistics["rows"]),
            "holm_scope": (
                "global_iid_and_nested_vs_rbf"
                if comparison_statistics["source"]
                == "paired_comparison_statistics"
                else "artifact_defined"
            ),
        },
        "full_statistics": comparison_statistics["rows"],
        "full_iid_statistics": _comparison_rows_for_baseline(
            comparison_statistics["rows"], "iid"
        ),
        "full_nested_vs_rbf_statistics": [
            row
            for row in _comparison_rows_for_baseline(
                comparison_statistics["rows"], "rbf"
            )
            if _method_kind(row.get("method")) == "nested"
        ],
        "formal_equal_rank_kl_audit": formal_fairness_audit,
        "trajectory_summary": payloads["trajectory"] or {},
        "scientific_judgment": judgment,
    }
    if judgment["label"] not in SCIENTIFIC_LABELS:
        raise AssertionError("invalid scientific label")
    _atomic_write(
        destination / "FULL_EXPERIMENT_REPORT.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write(destination / "FULL_EXPERIMENT_REPORT.md", _build_markdown(report))
    final_status = {
        "schema_version": 1,
        "generated_at_utc": report["generated_at_utc"],
        "artifact_root": str(root),
        **completion,
        "completion": completion,
        "overall_complete": report["overall_complete"],
        "blockers": blockers,
        "scientific_judgment": judgment,
        "report_paths": {
            "markdown": str(destination / "FULL_EXPERIMENT_REPORT.md"),
            "json": str(destination / "FULL_EXPERIMENT_REPORT.json"),
            "reproduce": str(destination / "REPRODUCE.md"),
        },
    }
    _atomic_write(
        destination / "REPRODUCE.md",
        _build_reproduce(root, resolved_paths["resolved_config"]),
    )
    # FINAL_STATUS is written last: its report_complete=true assertion therefore
    # means every report/reproduction artifact above was written successfully.
    _atomic_write(
        destination / "FINAL_STATUS.json",
        json.dumps(final_status, indent=2, ensure_ascii=False) + "\n",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    for name in DEFAULT_CANDIDATES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=None)
    parser.add_argument("--implementation-complete", action="store_true", default=None)
    parser.add_argument("--tests-complete", action="store_true", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in DEFAULT_CANDIDATES
        if getattr(args, name) is not None
    }
    report = generate_report(
        args.artifact_root,
        output_dir=args.output_dir,
        paths=overrides,
        implementation_complete=args.implementation_complete,
        tests_complete=args.tests_complete,
    )
    print(json.dumps({
        "completion": report["completion"],
        "scientific_judgment": report["scientific_judgment"],
        "blocker_count": len(report["blockers"]),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
