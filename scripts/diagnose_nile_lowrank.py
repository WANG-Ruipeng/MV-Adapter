"""CPU preflight for the equal-joint-KL low-rank NILE study.

The command calibrates every requested covariance before allocating a latent
ensemble. Unattainable KL budgets are explicitly excluded and do not make the
preflight process fail; a distribution, basis, covariance, KL, or eigenvalue
gate failure does. No diffusion model is loaded by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

from mvadapter.nile.basis import build_dct2_basis
from mvadapter.nile.covariance import (
    calibrate_alpha_for_target_kl,
    covariance_metadata,
    periodic_camera_rbf_covariance,
    tree_a_covariance,
    tree_ab_covariance,
)
from mvadapter.nile.diagnostics import (
    DEFAULT_DISTRIBUTION_THRESHOLDS,
    DEFAULT_LOWRANK_GATE_THRESHOLDS,
    diagnose_latents,
    diagnose_lowrank_latents,
    evaluate_distribution_gates,
    evaluate_lowrank_distribution_gates,
    project_basis_coefficients,
)
from mvadapter.nile.lowrank_coupling import (
    correlate_orthonormal_subspace,
    make_shared_full_latents,
)


STATEMENT = (
    "NILE-inspired nested Gaussian element topology; strict NILE/SZ is not "
    "implemented in this study."
)
LOWRANK_METHODS = (
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
BASELINE_METHODS = ("iid_external", "shared_full")
DEFAULT_COEFFICIENT_MIN_OBSERVATIONS = 8192


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _configuration_id(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:20]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read a YAML study config")
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("study config must contain a mapping")
    return payload


def build_pilot_configurations(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Build the configured PILOT matrix and IDs used by the study runner."""

    pilot = config["pilot"]
    configurations: List[Dict[str, Any]] = [
        {
            "method": "iid_external",
            "rank": None,
            "target_kl": 0.0,
            "rbf_length_scale_deg": None,
        },
        {
            "method": "shared_full",
            "rank": None,
            "target_kl": None,
            "rbf_length_scale_deg": None,
        },
    ]
    for rank in pilot["ranks"]:
        for target_kl in pilot["target_kls"]:
            for ell_deg in pilot["rbf_length_scales_deg"]:
                configurations.append(
                    {
                        "method": "lowrank_camera_rbf",
                        "rank": int(rank),
                        "target_kl": float(target_kl),
                        "rbf_length_scale_deg": float(ell_deg),
                    }
                )
            for method in (
                "lowrank_nested_tree_a",
                "lowrank_nested_tree_ab",
            ):
                configurations.append(
                    {
                        "method": method,
                        "rank": int(rank),
                        "target_kl": float(target_kl),
                        "rbf_length_scale_deg": None,
                    }
                )
    for item in configurations:
        item["config_id"] = _configuration_id(item)
    expected = int(pilot.get("expected_configs_per_input_seed", 18))
    if len(configurations) != expected:
        raise ValueError(
            "PILOT preflight built {} configurations but config expected {}".format(
                len(configurations), expected
            )
        )
    return configurations


def _thresholds(
    config: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    preflight = config.get("preflight", {})
    distribution = dict(DEFAULT_DISTRIBUTION_THRESHOLDS)
    aliases = {
        "mean_abs_max": "max_abs_mean",
        "std_min": "min_std",
        "std_max": "max_std",
        "lag_abs_max": "max_abs_lag_autocorrelation",
        "radial_psd_max": "max_radial_psd_deviation",
        "axis_stripe_max": "max_axis_stripe_score",
        "covariance_mae_max": "max_cross_view_covariance_mae",
    }
    for source, target in aliases.items():
        if source in preflight:
            distribution[target] = float(preflight[source])

    lowrank = dict(DEFAULT_LOWRANK_GATE_THRESHOLDS)
    lowrank_aliases = {
        "basis_orthonormality_max": "max_basis_orthonormality_error",
        "covariance_mae_max": "max_basis_coefficient_covariance_mae",
        "kl_relative_error_max": "max_joint_kl_relative_error",
        "min_eigenvalue": "min_covariance_eigenvalue",
    }
    for source, target in lowrank_aliases.items():
        if source in preflight:
            lowrank[target] = float(preflight[source])
    return distribution, lowrank


def _covariance_checksum(matrix: torch.Tensor) -> str:
    canonical = matrix.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = "nile-view-covariance-v1|{}|{}|".format(*canonical.shape).encode(
        "ascii"
    )
    raw = bytes(canonical.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + raw).hexdigest()


def _target_covariance(
    method: str,
    azimuths_deg: Sequence[float],
    ell_deg: Optional[float],
) -> torch.Tensor:
    if method == "lowrank_camera_rbf":
        if ell_deg is None:
            raise ValueError("camera RBF requires rbf_length_scale_deg")
        return periodic_camera_rbf_covariance(
            azimuths_deg, ell_deg=float(ell_deg), dtype=torch.float64
        )
    if method == "lowrank_nested_tree_a":
        return tree_a_covariance(azimuths_deg, dtype=torch.float64)
    if method == "lowrank_nested_tree_ab":
        return tree_ab_covariance(azimuths_deg, dtype=torch.float64)
    raise ValueError("unsupported low-rank method: {}".format(method))


def _failed_check_names(gates: Mapping[str, Any]) -> List[str]:
    return sorted(
        name
        for name, check in gates.get("checks", {}).items()
        if not bool(check.get("passed", False))
    )


def _coefficient_ensemble(
    covariance: torch.Tensor,
    *,
    required_total: int,
    existing_count: int,
    seed: int,
) -> Optional[torch.Tensor]:
    additional_count = max(0, int(required_total) - int(existing_count))
    if additional_count == 0:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    iid = torch.randn(
        (additional_count, covariance.shape[0]),
        generator=generator,
        dtype=torch.float64,
    )
    factor = torch.linalg.cholesky(covariance.to(device="cpu", dtype=torch.float64))
    return iid.matmul(factor.mT)


def _basis_projection_consistency(
    iid_latents: torch.Tensor,
    output_latents: torch.Tensor,
    basis: torch.Tensor,
    covariance: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> float:
    source = project_basis_coefficients(
        iid_latents,
        basis,
        batch_size=batch_size,
        num_views=num_views,
    )
    observed = project_basis_coefficients(
        output_latents,
        basis,
        batch_size=batch_size,
        num_views=num_views,
    )
    factor = torch.linalg.cholesky(covariance.to(dtype=torch.float64))
    expected = torch.einsum("vw,bwk->bvk", factor, source)
    return float((observed - expected).abs().max().item())


def _common_record(configuration: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "config_id": configuration["config_id"],
        "method": configuration["method"],
        "rank": configuration.get("rank"),
        "target_kl": configuration.get("target_kl"),
        "rbf_length_scale_deg": configuration.get("rbf_length_scale_deg"),
        "statement": STATEMENT,
        "per_sample_standardization": False,
    }


@torch.no_grad()
def run_configuration_preflight(
    configuration: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    basis_cache: Optional[
        MutableMapping[int, Tuple[torch.Tensor, Dict[str, Any]]]
    ] = None,
    device: Any = "cpu",
    coefficient_min_observations: int = DEFAULT_COEFFICIENT_MIN_OBSERVATIONS,
) -> Dict[str, Any]:
    """Run one baseline or calibrated low-rank preflight configuration."""

    method = str(configuration["method"])
    if method not in BASELINE_METHODS + LOWRANK_METHODS:
        raise ValueError("unsupported preflight method: {}".format(method))
    record = _common_record(configuration)
    model = config["model"]
    preflight = config["preflight"]
    batch_size = int(preflight["batch_size"])
    channels = int(preflight["channels"])
    height = int(preflight["latent_height"])
    width = int(preflight["latent_width"])
    seeds = [int(value) for value in preflight["seeds"]]
    azimuths_deg = [float(value) for value in model["views_deg"]]
    num_views = len(azimuths_deg)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for preflight but is unavailable")
    if batch_size <= 0 or channels <= 0 or height <= 1 or width <= 1:
        raise ValueError("preflight batch/channels must be positive and H/W > 1")
    if len(seeds) < 2:
        raise ValueError("preflight requires multiple IID batches/seeds")
    if coefficient_min_observations < 2:
        raise ValueError("coefficient_min_observations must be at least two")
    distribution_limits, lowrank_limits = _thresholds(config)

    target = None
    calibration = None
    if method in LOWRANK_METHODS:
        # KL feasibility intentionally precedes basis construction and latent
        # allocation. An impossible row never reaches random sampling.
        rank = int(configuration["rank"])
        target = _target_covariance(
            method, azimuths_deg, configuration.get("rbf_length_scale_deg")
        )
        calibration = calibrate_alpha_for_target_kl(
            target,
            rank,
            float(configuration["target_kl"]),
            relative_tolerance=1e-8,
            max_iterations=80,
        )
        record.update(
            {
                "calibration_status": calibration["status"],
                "target_kl": calibration["target_kl"],
                "achieved_kl": calibration["achieved_kl"],
                "kl_relative_error": calibration["relative_error"],
                "alpha": calibration["alpha"],
                "target_covariance_metadata": covariance_metadata(
                    target,
                    azimuths_deg=azimuths_deg,
                    ell_deg=(
                        configuration.get("rbf_length_scale_deg")
                        if method == "lowrank_camera_rbf"
                        else None
                    ),
                    topology=method,
                ),
                "effective_covariance_metadata": calibration["json_metadata"],
                "covariance_checksum": _covariance_checksum(
                    calibration["covariance"]
                ),
            }
        )
        if calibration["status"] != "calibrated":
            record.update(
                {
                    "status": "excluded",
                    "passed": False,
                    "distribution_gate_passed": False,
                    "eligible_for_generation": False,
                    "excluded": True,
                    "exclusion_reason": "unattainable_target_kl",
                    "sampling_performed": False,
                    "failed_checks": [],
                }
            )
            return record

    basis = None
    basis_metadata = None
    if method in LOWRANK_METHODS:
        rank = int(configuration["rank"])
        if basis_cache is None:
            basis_cache = {}
        if rank not in basis_cache:
            built_basis, built_metadata = build_dct2_basis(
                channels,
                height,
                width,
                rank,
                exclude_dc=True,
                device=resolved_device,
                dtype=torch.float32,
                return_metadata=True,
            )
            basis_cache[rank] = (built_basis, built_metadata)
        basis, basis_metadata = basis_cache[rank]
        record["basis_checksum"] = basis_metadata["basis_checksum"]
        record["basis_metadata"] = basis_metadata

    output_batches = []
    projection_errors: List[float] = []
    shape = (batch_size * num_views, channels, height, width)
    for seed in seeds:
        generator = torch.Generator(device=resolved_device).manual_seed(seed)
        iid_latents = torch.randn(
            shape,
            generator=generator,
            device=resolved_device,
            dtype=torch.float32,
        )
        if method == "iid_external":
            output = iid_latents
        elif method == "shared_full":
            output = make_shared_full_latents(iid_latents, num_views)
        else:
            assert basis is not None and calibration is not None
            output = correlate_orthonormal_subspace(
                iid_latents,
                basis,
                calibration["covariance"],
                num_views,
            )
            projection_errors.append(
                _basis_projection_consistency(
                    iid_latents,
                    output,
                    basis,
                    calibration["covariance"],
                    batch_size=batch_size,
                    num_views=num_views,
                )
            )
        output_batches.append(output.detach().to(device="cpu"))

    ensemble = torch.cat(output_batches, dim=0)
    ensemble_batch_size = batch_size * len(seeds)
    if method == "iid_external":
        full_target = torch.eye(num_views, dtype=torch.float64)
        report = diagnose_latents(
            ensemble,
            batch_size=ensemble_batch_size,
            num_views=num_views,
            target_covariance=full_target,
        )
        gates = evaluate_distribution_gates(
            report,
            thresholds=distribution_limits,
            require_covariance_target=True,
        )
        record.update(
            {
                "alpha": 0.0,
                "achieved_kl": 0.0,
                "kl_relative_error": 0.0,
                "covariance_checksum": _covariance_checksum(full_target),
                "diagnostic_only": False,
            }
        )
    elif method == "shared_full":
        full_target = torch.ones((num_views, num_views), dtype=torch.float64)
        report = diagnose_latents(
            ensemble,
            batch_size=ensemble_batch_size,
            num_views=num_views,
            target_covariance=full_target,
        )
        gates = evaluate_distribution_gates(
            report,
            thresholds=distribution_limits,
            require_covariance_target=True,
        )
        record.update(
            {
                "alpha": None,
                "achieved_kl": None,
                "kl_relative_error": None,
                "covariance_checksum": _covariance_checksum(full_target),
                "diagnostic_only": True,
                "degenerate_joint_distribution": True,
                "joint_kl_finite": False,
                "interpretation": "diagnostic_upper_bound_not_3d_consistency",
            }
        )
    else:
        assert basis is not None and calibration is not None
        spatial_observations = ensemble_batch_size * int(configuration["rank"])
        supplemental = _coefficient_ensemble(
            calibration["covariance"],
            required_total=coefficient_min_observations,
            existing_count=spatial_observations,
            seed=20260711,
        )
        report = diagnose_lowrank_latents(
            ensemble,
            batch_size=ensemble_batch_size,
            num_views=num_views,
            basis=basis.to(device="cpu"),
            coefficient_target_covariance=calibration["covariance"],
            full_space_target_covariance=torch.eye(num_views, dtype=torch.float64),
            additional_coefficient_samples=supplemental,
            target_kl=calibration["target_kl"],
            achieved_kl=calibration["achieved_kl"],
            alpha=calibration["alpha"],
        )
        maximum_projection_error = max(projection_errors, default=0.0)
        report["basis_coefficient_projection_consistency"] = {
            "max_abs_error": maximum_projection_error,
            "batch_count": len(seeds),
            "limit": 1e-4,
            "passed": maximum_projection_error < 1e-4,
        }
        gates = evaluate_lowrank_distribution_gates(
            report,
            distribution_thresholds=distribution_limits,
            lowrank_thresholds=lowrank_limits,
            require_finite_kl=True,
        )
        gates["checks"]["basis_coefficient_projection_consistency"] = dict(
            report["basis_coefficient_projection_consistency"]
        )
        gates["passed"] = bool(gates["passed"]) and bool(
            report["basis_coefficient_projection_consistency"]["passed"]
        )
        record["diagnostic_only"] = False

    passed = bool(gates["passed"])
    record.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "distribution_gate_passed": passed,
            "eligible_for_generation": passed,
            "excluded": not passed,
            "exclusion_reason": None if passed else "distribution_gate_failed",
            "sampling_performed": True,
            "ensemble": {
                "batch_size_per_seed": batch_size,
                "seeds": seeds,
                "batch_count": len(seeds),
                "ensemble_batch_size": ensemble_batch_size,
                "num_views": num_views,
                "channels": channels,
                "height": height,
                "width": width,
                "coefficient_min_observations": coefficient_min_observations,
            },
            "report": report,
            "gates": gates,
            "failed_checks": _failed_check_names(gates),
        }
    )
    return record


def summarize_configuration_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize exclusions without treating unattainable KL as gate failure."""

    unattainable = [
        item
        for item in records
        if item.get("exclusion_reason") == "unattainable_target_kl"
    ]
    attempted = [item for item in records if item.get("sampling_performed", False)]
    passed = [item for item in attempted if bool(item.get("passed", False))]
    # Only a mathematically unattainable requested KL is a neutral exclusion.
    # Exceptions and any other unsampled/failed rows must fail the preflight.
    failed = [
        item
        for item in records
        if not bool(item.get("passed", False))
        and item.get("exclusion_reason") != "unattainable_target_kl"
    ]
    return {
        "passed": len(failed) == 0,
        "all_attempted_gates_passed": len(failed) == 0,
        "requested_configuration_count": len(records),
        "attempted_configuration_count": len(attempted),
        "passed_configuration_count": len(passed),
        "failed_configuration_count": len(failed),
        "excluded_unattainable_count": len(unattainable),
        "eligible_configuration_count": len(passed),
        "eligible_count": len(passed),
        "excluded_count": len(records) - len(passed),
        "failed_config_ids": [item["config_id"] for item in failed],
        "unattainable_config_ids": [item["config_id"] for item in unattainable],
    }


def _matrix_from_record(
    record: Mapping[str, Any]
) -> Tuple[List[List[float]], List[List[float]]]:
    report = record.get("report", {})
    coefficient = report.get("basis_coefficient_covariance")
    if isinstance(coefficient, Mapping):
        return coefficient["empirical"], coefficient["target"]
    empirical = report.get("cross_view_covariance", [])
    if record.get("method") == "shared_full":
        target = [[1.0 for _ in range(len(empirical))] for _ in empirical]
    else:
        target = [
            [1.0 if row == column else 0.0 for column in range(len(empirical))]
            for row in range(len(empirical))
        ]
    return empirical, target


def _colour(
    value: float, minimum: float = -1.0, maximum: float = 1.0
) -> Tuple[int, int, int]:
    fraction = max(
        0.0, min(1.0, (float(value) - minimum) / (maximum - minimum))
    )
    if fraction < 0.5:
        blend = fraction * 2.0
        return (int(30 + 225 * blend), int(80 + 175 * blend), 255)
    blend = (fraction - 0.5) * 2.0
    return (255, int(255 - 205 * blend), int(255 - 225 * blend))


def _write_covariance_plot(
    record: Mapping[str, Any], path: Path
) -> Optional[str]:
    empirical, target = _matrix_from_record(record)
    if not empirical or Image is None or ImageDraw is None:
        return None
    size = len(empirical)
    cell, margin, gap = 28, 34, 28
    canvas = Image.new(
        "RGB",
        (margin * 2 + size * cell * 2 + gap, margin * 2 + size * cell + 22),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), "empirical", fill="black")
    draw.text((margin + size * cell + gap, 8), "target", fill="black")
    for panel, matrix in enumerate((empirical, target)):
        left = margin + panel * (size * cell + gap)
        for row in range(size):
            for column in range(size):
                x0 = left + column * cell
                y0 = margin + row * cell
                draw.rectangle(
                    (x0, y0, x0 + cell - 1, y0 + cell - 1),
                    fill=_colour(matrix[row][column]),
                    outline=(220, 220, 220),
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _write_covariance_eigenvalue_spectra(
    records: Sequence[Mapping[str, Any]], path: Path
) -> Optional[str]:
    """Plot the effective six-view covariance spectrum for every low-rank row."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError):
        return None

    series = []
    for record in records:
        if record.get("method") not in LOWRANK_METHODS:
            continue
        metadata = record.get("effective_covariance_metadata", {})
        eigenvalues = metadata.get("eigenvalues") if isinstance(metadata, Mapping) else None
        if not isinstance(eigenvalues, list) or not eigenvalues:
            continue
        try:
            values = sorted(float(value) for value in eigenvalues)
        except (TypeError, ValueError):
            continue
        label = "{} K={} KL={}".format(
            str(record.get("method", "")).replace("lowrank_", ""),
            record.get("rank"),
            record.get("target_kl"),
        )
        if record.get("rbf_length_scale_deg") is not None:
            label += " ell={}".format(record.get("rbf_length_scale_deg"))
        if record.get("exclusion_reason") == "unattainable_target_kl":
            label += " (unattainable cap)"
        series.append((label, values))
    if not series:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    for label, values in series:
        axis.plot(range(1, len(values) + 1), values, marker="o", alpha=0.72, label=label)
    axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="IID eigenvalue")
    axis.set(
        xlabel="sorted eigenvalue index",
        ylabel="effective covariance eigenvalue",
        title="Effective view-covariance eigenvalue spectra",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=6, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def _write_gate_matrix(
    records: Sequence[Mapping[str, Any]], path: Path
) -> Optional[str]:
    if Image is None or ImageDraw is None:
        return None
    check_names = sorted(
        {
            name
            for record in records
            for name in record.get("gates", {}).get("checks", {})
        }
    )
    cell_w, cell_h, left, top = 22, 18, 250, 100
    canvas = Image.new(
        "RGB",
        (
            left + max(1, len(check_names)) * cell_w + 20,
            top + len(records) * cell_h + 20,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(check_names):
        draw.text((left + column * cell_w, 8), str(column + 1), fill="black")
    for row, record in enumerate(records):
        y = top + row * cell_h
        label = "{} {}".format(record["config_id"][:8], record["method"])
        draw.text((5, y), label[:39], fill="black")
        checks = record.get("gates", {}).get("checks", {})
        for column, name in enumerate(check_names):
            if record.get("exclusion_reason") == "unattainable_target_kl":
                colour = (160, 160, 160)
            elif name not in checks:
                colour = (225, 225, 225)
            elif checks[name].get("passed"):
                colour = (65, 170, 90)
            else:
                colour = (210, 65, 65)
            x = left + column * cell_w
            draw.rectangle(
                (x, y, x + cell_w - 2, y + cell_h - 2), fill=colour
            )
    for index, name in enumerate(check_names):
        y = 28 + index * 10
        if y > top - 12:
            break
        draw.text((5, y), "{} {}".format(index + 1, name)[:42], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _write_alpha_kl_plot(
    records: Sequence[Mapping[str, Any]], path: Path
) -> Optional[str]:
    if Image is None or ImageDraw is None:
        return None
    lowrank = [item for item in records if item.get("method") in LOWRANK_METHODS]
    width, height = 720, 420
    left, right, top, bottom = 60, 20, 30, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="black",
        width=2,
    )
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.text((width // 2 - 45, height - 25), "calibrated alpha", fill="black")
    draw.text((5, 8), "joint KL (nats)", fill="black")
    maximum_kl = max(
        [float(item.get("target_kl") or 0.0) for item in lowrank]
        + [float(item.get("achieved_kl") or 0.0) for item in lowrank]
        + [1.0]
    )
    method_colours = {
        "lowrank_camera_rbf": (45, 105, 200),
        "lowrank_nested_tree_a": (225, 130, 25),
        "lowrank_nested_tree_ab": (125, 70, 175),
    }
    for item in lowrank:
        alpha = float(item.get("alpha") or 0.0)
        achieved = float(item.get("achieved_kl") or 0.0)
        x = left + alpha * (width - left - right)
        y = height - bottom - achieved / maximum_kl * (height - top - bottom)
        colour = method_colours[item["method"]]
        radius = 5 if item.get("sampling_performed") else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
        if item.get("exclusion_reason") == "unattainable_target_kl":
            draw.line((x - 6, y - 6, x + 6, y + 6), fill=(180, 0, 0), width=2)
            draw.line((x - 6, y + 6, x + 6, y - 6), fill=(180, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "config_id",
        "method",
        "rank",
        "target_kl",
        "achieved_kl",
        "kl_relative_error",
        "alpha",
        "rbf_length_scale_deg",
        "status",
        "passed",
        "eligible_for_generation",
        "exclusion_reason",
        "basis_checksum",
        "covariance_checksum",
        "mean_abs",
        "std",
        "skewness",
        "excess_kurtosis",
        "lag_x_abs",
        "lag_y_abs",
        "radial_psd_deviation",
        "axis_stripe_score",
        "basis_covariance_mae",
        "basis_orthonormality_error",
        "min_eigenvalue",
        "condition_number",
        "failed_checks",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            report = record.get("report", {})
            global_stats = report.get("global", {})
            lag_values = report.get("lag_autocorrelation", {}).get("values", {})
            coefficient = report.get("basis_coefficient_covariance", {})
            spectrum = report.get("coefficient_covariance_spectrum", {})
            effective = record.get("effective_covariance_metadata", {})
            writer.writerow(
                {
                    "config_id": record["config_id"],
                    "method": record["method"],
                    "rank": record.get("rank"),
                    "target_kl": record.get("target_kl"),
                    "achieved_kl": record.get("achieved_kl"),
                    "kl_relative_error": record.get("kl_relative_error"),
                    "alpha": record.get("alpha"),
                    "rbf_length_scale_deg": record.get("rbf_length_scale_deg"),
                    "status": record.get("status"),
                    "passed": record.get("passed"),
                    "eligible_for_generation": record.get(
                        "eligible_for_generation"
                    ),
                    "exclusion_reason": record.get("exclusion_reason"),
                    "basis_checksum": record.get("basis_checksum"),
                    "covariance_checksum": record.get("covariance_checksum"),
                    "mean_abs": (
                        abs(float(global_stats.get("mean", 0.0)))
                        if global_stats
                        else None
                    ),
                    "std": global_stats.get("std"),
                    "skewness": global_stats.get("skewness"),
                    "excess_kurtosis": global_stats.get("excess_kurtosis"),
                    "lag_x_abs": (
                        abs(float(lag_values.get("0,1", 0.0)))
                        if lag_values
                        else None
                    ),
                    "lag_y_abs": (
                        abs(float(lag_values.get("1,0", 0.0)))
                        if lag_values
                        else None
                    ),
                    "radial_psd_deviation": report.get(
                        "radial_psd_deviation"
                    ),
                    "axis_stripe_score": report.get(
                        "axis_stripe_score", {}
                    ).get("max"),
                    "basis_covariance_mae": coefficient.get("error", {}).get(
                        "offdiag_mae"
                    ),
                    "basis_orthonormality_error": report.get("basis", {}).get(
                        "orthonormality_max_error"
                    ),
                    "min_eigenvalue": spectrum.get(
                        "min_eigenvalue", effective.get("min_eigenvalue")
                    ),
                    "condition_number": spectrum.get(
                        "condition_number", effective.get("condition_number")
                    ),
                    "failed_checks": ";".join(record.get("failed_checks", [])),
                }
            )


def write_preflight_artifacts(
    output_dir: Path,
    payload: MutableMapping[str, Any],
    *,
    plots: bool = True,
) -> None:
    """Write JSON, CSV, and deterministic diagnostic images."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = payload["configurations"]
    plot_paths: Dict[str, Any] = {}
    if plots:
        diagnostic_dir = output_dir / "diagnostics"
        covariance_plots = {}
        for record in records:
            if not record.get("sampling_performed"):
                continue
            path = diagnostic_dir / (record["config_id"] + "_covariance.png")
            if _write_covariance_plot(record, path) is not None:
                covariance_plots[record["config_id"]] = str(
                    path.relative_to(output_dir).as_posix()
                )
        gate_path = diagnostic_dir / "configuration_gate_matrix.png"
        alpha_path = diagnostic_dir / "alpha_vs_achieved_kl.png"
        eigenvalue_path = diagnostic_dir / "covariance_eigenvalue_spectra.png"
        if _write_gate_matrix(records, gate_path) is not None:
            plot_paths["gate_matrix"] = gate_path.relative_to(output_dir).as_posix()
        if _write_alpha_kl_plot(records, alpha_path) is not None:
            plot_paths["alpha_vs_achieved_kl"] = str(
                alpha_path.relative_to(output_dir).as_posix()
            )
        if _write_covariance_eigenvalue_spectra(records, eigenvalue_path) is not None:
            plot_paths["covariance_eigenvalue_spectra"] = str(
                eigenvalue_path.relative_to(output_dir).as_posix()
            )
        plot_paths["covariance"] = covariance_plots
    expected_covariance_ids = sorted(
        str(record["config_id"])
        for record in records
        if record.get("sampling_performed")
    )
    observed_covariance_ids = sorted(plot_paths.get("covariance", {}))
    required_summary_plots = (
        "gate_matrix",
        "alpha_vs_achieved_kl",
        "covariance_eigenvalue_spectra",
    )
    missing_summary_plots = [
        name for name in required_summary_plots if name not in plot_paths
    ]
    missing_covariance_ids = sorted(
        set(expected_covariance_ids).difference(observed_covariance_ids)
    )
    payload["diagnostic_plot_audit"] = {
        "requested": bool(plots),
        "complete": bool(
            plots and not missing_summary_plots and not missing_covariance_ids
        ),
        "required_summary_plots": list(required_summary_plots),
        "missing_summary_plots": missing_summary_plots,
        "expected_covariance_plot_count": len(expected_covariance_ids),
        "observed_covariance_plot_count": len(observed_covariance_ids),
        "missing_covariance_config_ids": missing_covariance_ids,
    }
    payload["diagnostic_plots_complete"] = bool(
        payload["diagnostic_plot_audit"]["complete"]
    )
    payload["diagnostic_plots"] = plot_paths
    _write_csv(output_dir / "configuration_gates.csv", records)
    _atomic_json(output_dir / "configuration_gates.json", payload)
    _atomic_json(
        output_dir / "preflight_summary.json",
        {key: value for key, value in payload.items() if key != "configurations"},
    )


def run_study_preflight(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    device: Any = "cpu",
    coefficient_min_observations: int = DEFAULT_COEFFICIENT_MIN_OBSERVATIONS,
    plots: bool = True,
) -> Dict[str, Any]:
    """Run and persist every configuration in the requested preflight matrix."""

    configurations = build_pilot_configurations(config)
    basis_cache: Dict[int, Tuple[torch.Tensor, Dict[str, Any]]] = {}
    records = []
    for index, configuration in enumerate(configurations, start=1):
        print(
            "[preflight {}/{}] {} {}".format(
                index,
                len(configurations),
                configuration["method"],
                configuration["config_id"],
            ),
            flush=True,
        )
        try:
            record = run_configuration_preflight(
                configuration,
                config,
                basis_cache=basis_cache,
                device=device,
                coefficient_min_observations=coefficient_min_observations,
            )
        except Exception as error:
            record = _common_record(configuration)
            record.update(
                {
                    "status": "failed",
                    "passed": False,
                    "distribution_gate_passed": False,
                    "eligible_for_generation": False,
                    "excluded": True,
                    "exclusion_reason": "preflight_exception",
                    "sampling_performed": False,
                    "failed_checks": ["preflight_exception"],
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )
        records.append(record)
    summary = summarize_configuration_records(records)
    distribution_limits, lowrank_limits = _thresholds(config)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "statement": STATEMENT,
        **summary,
        "passed": summary["passed"],
        "device": str(torch.device(device)),
        "thresholds": {
            "distribution": distribution_limits,
            "lowrank": lowrank_limits,
        },
        "coefficient_min_observations": int(coefficient_min_observations),
        "configurations": records,
    }
    write_preflight_artifacts(Path(output_dir), payload, plots=plots)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--coefficient-min-observations",
        type=int,
        default=None,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    config = _load_config(args.config.expanduser().resolve())
    coefficient_min_observations = (
        int(args.coefficient_min_observations)
        if args.coefficient_min_observations is not None
        else int(
            config.get("preflight", {}).get(
                "coefficient_min_observations",
                DEFAULT_COEFFICIENT_MIN_OBSERVATIONS,
            )
        )
    )
    if coefficient_min_observations < 2:
        parser.error("--coefficient-min-observations must be at least two")
    payload = run_study_preflight(
        config,
        args.output_dir.expanduser().resolve(),
        device=args.device,
        coefficient_min_observations=coefficient_min_observations,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                key: payload.get(key)
                for key in (
                    "passed",
                    "requested_configuration_count",
                    "attempted_configuration_count",
                    "passed_configuration_count",
                    "failed_configuration_count",
                    "excluded_unattainable_count",
                    "eligible_configuration_count",
                    "diagnostic_plots_complete",
                )
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] and (
        args.no_plots or payload.get("diagnostic_plots_complete", False)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
