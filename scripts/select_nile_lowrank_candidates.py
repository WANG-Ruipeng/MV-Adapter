"""Freeze one equal-KL PILOT candidate per low-rank covariance topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


TOPOLOGY_METHODS = {
    "camera_rbf": "lowrank_camera_rbf",
    "nested_tree_a": "lowrank_nested_tree_a",
    "nested_tree_ab": "lowrank_nested_tree_ab",
}


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _eligibility_reasons(
    row: Mapping[str, Any], policy: Mapping[str, Any]
) -> List[str]:
    reasons = []
    if not bool(row.get("distribution_gate_passed", False)):
        reasons.append("distribution_gate_failed")
    if bool(row.get("output_missing", False)):
        reasons.append("output_missing")
    met3r = _finite(row.get("met3r_all_pair_mean"))
    if met3r is None:
        reasons.append("met3r_missing")
    failure_rate = _finite(row.get("met3r_failure_rate"))
    if failure_rate is None or failure_rate > float(
        policy.get("max_met3r_failure_rate", 0.10)
    ):
        reasons.append("met3r_failure_rate")

    checks = (
        ("dino_identity_mean_delta", -float(policy.get("max_dino_drop_abs", 0.02)), "identity_drop"),
        (
            "small_component_ratio_delta",
            float(policy.get("max_small_component_ratio_increase", 0.02)),
            "small_component_ratio",
        ),
        (
            "component_failure_rate_delta",
            float(policy.get("max_component_failure_rate_increase", 0.10)),
            "component_failure_rate",
        ),
        (
            "foreground_area_cv_delta",
            float(policy.get("max_foreground_area_cv_increase", 0.05)),
            "foreground_area_cv",
        ),
    )
    for field, threshold, reason in checks:
        value = _finite(row.get(field))
        if value is None:
            reasons.append(field + "_missing")
        elif field == "dino_identity_mean_delta":
            if value < threshold:
                reasons.append(reason)
        elif value > threshold:
            reasons.append(reason)

    r_hf = _finite(row.get("r_hf"))
    if r_hf is None or not (
        float(policy.get("rhf_min", 0.5))
        <= r_hf
        <= float(policy.get("rhf_max", 1.5))
    ):
        reasons.append("r_hf_out_of_range")
    if row.get("collapse_detector_label") == "view_collapse_alert" or bool(
        row.get("collapse_alert", False)
    ):
        reasons.append("collapse_alert")
    return sorted(set(reasons))


def _safe_sort_key(row: Mapping[str, Any]) -> tuple:
    """Return the deterministic scientific tie-break key.

    Keep explicit ``None`` handling here: a valid zero KL or zero length scale
    must never be treated as a missing/infinite value.
    """

    target_kl = _finite(row.get("target_kl"))
    length_scale = _finite(row.get("rbf_length_scale_deg"))
    return (
        target_kl if target_kl is not None else math.inf,
        int(row.get("rank") or 10**9),
        length_scale if length_scale is not None else math.inf,
        str(row.get("config_id") or ""),
    )


def select_candidates(
    summaries: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> Dict[str, Any]:
    selections: Dict[str, Any] = {}
    annotated = []
    for row in summaries:
        copy = dict(row)
        copy["eligibility_reasons"] = _eligibility_reasons(copy, policy)
        copy["eligible"] = not copy["eligibility_reasons"]
        annotated.append(copy)
    annotated.sort(
        key=lambda row: (
            str(row.get("method") or ""),
            *_safe_sort_key(row),
        )
    )

    for topology, method in TOPOLOGY_METHODS.items():
        rows = [row for row in annotated if row.get("method") == method]
        eligible = [row for row in rows if row["eligible"]]
        if eligible:
            eligible.sort(
                key=lambda row: (
                    _finite(row.get("met3r_all_pair_mean")),
                    *_safe_sort_key(row),
                )
            )
            best = eligible[0]
            standard_error = _finite(best.get("met3r_standard_error")) or 0.0
            best_score = _finite(best.get("met3r_all_pair_mean"))
            if best_score is None:
                raise ValueError("eligible candidate is missing a finite MEt3R score")
            near_best = [
                row
                for row in eligible
                if _finite(row.get("met3r_all_pair_mean")) is not None
                and float(_finite(row.get("met3r_all_pair_mean")))
                <= best_score + standard_error
            ]
            chosen = sorted(near_best, key=_safe_sort_key)[0]
            status = "selected"
            diagnostic_only = False
        elif rows:
            chosen = sorted(rows, key=_safe_sort_key)[0]
            status = "no_eligible_candidate"
            diagnostic_only = True
        else:
            selections[topology] = {
                "topology": topology,
                "method": method,
                "status": "missing_pilot_rows",
                "diagnostic_only": True,
                "configuration": None,
            }
            continue
        configuration_fields = (
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
        configuration = {field: chosen.get(field) for field in configuration_fields}
        selections[topology] = {
            "topology": topology,
            "method": method,
            "status": status,
            "diagnostic_only": diagnostic_only,
            "configuration": configuration,
            "pilot_metrics": {
                "met3r_all_pair_mean": chosen.get("met3r_all_pair_mean"),
                "met3r_standard_error": chosen.get("met3r_standard_error"),
                "dino_identity_mean_delta": chosen.get("dino_identity_mean_delta"),
                "small_component_ratio_delta": chosen.get("small_component_ratio_delta"),
                "component_failure_rate_delta": chosen.get("component_failure_rate_delta"),
                "foreground_area_cv_delta": chosen.get("foreground_area_cv_delta"),
                "r_hf": chosen.get("r_hf"),
            },
            "eligibility_reasons": chosen.get("eligibility_reasons", []),
        }

    frozen_core = {
        "schema_version": 1,
        "statement": (
            "NILE-inspired nested Gaussian element topology; strict NILE/SZ "
            "is not implemented in this study."
        ),
        "selection_policy": dict(policy),
        "selections": selections,
        "annotated_candidate_count": len(annotated),
        "eligible_candidate_count": sum(row["eligible"] for row in annotated),
        "candidates": annotated,
    }
    frozen_core["configuration_hash"] = hashlib.sha256(
        _canonical(frozen_core).encode("utf-8")
    ).hexdigest()
    return frozen_core


def freeze_candidates(
    result: Mapping[str, Any], output_json: Path, output_yaml: Path
) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output_json.exists():
        existing = json.loads(output_json.read_text(encoding="utf-8"))
        if existing.get("configuration_hash") != result.get("configuration_hash"):
            raise ValueError(
                "selected candidates are already frozen with another configuration hash"
            )
        if _canonical(existing) != _canonical(result):
            raise ValueError(
                "selected candidate artifact content changed despite the same hash"
            )
    else:
        _atomic_write(output_json, text)
    # JSON is valid YAML 1.2 and is the dependency-free fallback representation.
    if output_yaml.exists():
        existing_yaml = json.loads(output_yaml.read_text(encoding="utf-8"))
        if _canonical(existing_yaml) != _canonical(result):
            raise ValueError("selected candidate YAML does not match frozen JSON")
    else:
        _atomic_write(output_yaml, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    summaries = payload.get("configuration_summaries", payload.get("rows", []))
    if not isinstance(summaries, list):
        raise ValueError("metrics must contain configuration_summaries or rows")
    policy = (
        json.loads(args.policy.read_text(encoding="utf-8"))
        if args.policy is not None
        else {}
    )
    result = select_candidates(summaries, policy)
    freeze_candidates(result, args.output_json, args.output_yaml)
    print(json.dumps(result["selections"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
