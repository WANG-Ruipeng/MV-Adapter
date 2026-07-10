"""Preflight distribution gates for the formal multi-view latent samplers.

This command intentionally runs before loading MV-Adapter or SDXL.  It rejects
samplers that do not preserve each view's white-Gaussian marginal or that miss
their declared cross-view covariance.  Legacy Sobol/low-pass/callback methods
are not accepted by this formal gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

from mvadapter.nile.diagnostics import diagnose_latents, evaluate_distribution_gates
from mvadapter.nile.nested_elements import (
    make_nested_tree_latents,
    nested_tree_spatial_covariance_target,
)
from mvadapter.nile.spectral_gaussian import (
    camera_rbf_spatial_covariance_target,
    global_spatial_covariance_target,
    make_camera_rbf_correlated_latents,
    make_spectral_global_correlated_latents,
)


FORMAL_METHODS = (
    "iid_default",
    "iid_external",
    "shared_full",
    "spectral_global_corr",
    "camera_rbf_corr",
    "nested_tree_a",
    "nested_tree_ab",
)
STRENGTH_INDEPENDENT_METHODS = {
    "iid_default",
    "iid_external",
    "shared_full",
}
DEFAULT_STRENGTHS = (0.15, 0.30, 0.45, 0.60)


def run_preflight(
    method: str,
    *,
    view_angles: Sequence[float],
    seed: int,
    max_correlation: float,
    frequency_scale: float,
    camera_length_scale: float,
    batch_size: int = 16,
    channels: int = 4,
    height: int = 96,
    width: int = 96,
    device: Any = "cpu",
    thresholds: Optional[Dict[str, float]] = None,
    output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one formal sampler's ensemble distribution gate.

    The ensemble is intentionally larger than a generation batch: hard
    moment, PSD, lag, stripe, and covariance gates are statistical tests and
    are too noisy on the CLI's single generated sample. This function does
    not load any diffusion model and is therefore safe to call before model
    construction.
    """

    if method not in FORMAL_METHODS:
        raise ValueError(
            "unsupported formal method {!r}; legacy methods cannot pass this gate".format(
                method
            )
        )
    if batch_size <= 0 or channels <= 0 or height <= 1 or width <= 1:
        raise ValueError(
            "batch size/channels must be positive and spatial dimensions > 1"
        )
    if not view_angles:
        raise ValueError("at least one view angle is required")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    resolved_device = torch.device(device)
    record: Dict[str, Any] = {
        "method": method,
        "seed": int(seed),
        "max_correlation": float(max_correlation),
    }
    try:
        latents = _draw_latents(
            method,
            batch_size=batch_size,
            num_views=len(view_angles),
            channels=channels,
            height=height,
            width=width,
            view_angles=view_angles,
            seed=seed,
            device=resolved_device,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            camera_length_scale=camera_length_scale,
        )
        target = _target_covariance(
            method,
            num_views=len(view_angles),
            height=height,
            width=width,
            view_angles=view_angles,
            device=resolved_device,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            camera_length_scale=camera_length_scale,
        )
        report = diagnose_latents(
            latents,
            batch_size=batch_size,
            num_views=len(view_angles),
            target_covariance=target,
        )
        gates = evaluate_distribution_gates(
            report,
            thresholds=thresholds,
            require_covariance_target=True,
        )
        record.update(
            {"passed": bool(gates["passed"]), "report": report, "gates": gates}
        )
    except Exception as error:
        record.update(
            {
                "passed": False,
                "error": "{}: {}".format(type(error).__name__, error),
            }
        )

    payload: Dict[str, Any] = {
        "schema_version": 2,
        "passed": bool(record["passed"]),
        "config": {
            "method": method,
            "seed": int(seed),
            "max_correlation": float(max_correlation),
            "batch_size": int(batch_size),
            "num_views": len(view_angles),
            "channels": int(channels),
            "height": int(height),
            "width": int(width),
            "azimuth_deg": [float(value) for value in view_angles],
            "frequency_scale": float(frequency_scale),
            "camera_length_scale": float(camera_length_scale),
            "device": str(resolved_device),
        },
        "record": record,
    }
    if output is not None:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return payload


def _draw_latents(
    method: str,
    *,
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    view_angles: Sequence[float],
    seed: int,
    device: torch.device,
    max_correlation: float,
    frequency_scale: float,
    camera_length_scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch_size * num_views, channels, height, width)

    if method in {"iid_default", "iid_external"}:
        # Both baselines have the same IID target law at this distribution
        # gate. This draw does not establish pipeline-level equivalence; the
        # generation grid retains the real default-vs-external comparison.
        return torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    if method == "shared_full":
        shared = torch.randn(
            (batch_size, channels, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return shared[:, None].expand(
            batch_size, num_views, channels, height, width
        ).reshape(shape)
    if method == "spectral_global_corr":
        return make_spectral_global_correlated_latents(
            batch_size,
            num_views,
            channels,
            height,
            width,
            device=device,
            dtype=torch.float32,
            generator=generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
        )
    if method == "camera_rbf_corr":
        return make_camera_rbf_correlated_latents(
            batch_size,
            num_views,
            channels,
            height,
            width,
            view_angles,
            device=device,
            dtype=torch.float32,
            generator=generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            length_scale=camera_length_scale,
        )
    if method in {"nested_tree_a", "nested_tree_ab"}:
        return make_nested_tree_latents(
            batch_size,
            num_views,
            channels,
            height,
            width,
            view_angles,
            device=device,
            dtype=torch.float32,
            generator=generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            tree_mode="a" if method == "nested_tree_a" else "ab",
        )
    raise ValueError(
        "unsupported formal method {!r}; legacy methods cannot pass this gate".format(
            method
        )
    )


def _target_covariance(
    method: str,
    *,
    num_views: int,
    height: int,
    width: int,
    view_angles: Sequence[float],
    device: torch.device,
    max_correlation: float,
    frequency_scale: float,
    camera_length_scale: float,
) -> torch.Tensor:
    if method in {"iid_default", "iid_external"}:
        return torch.eye(num_views, device=device, dtype=torch.float64)
    if method == "shared_full":
        return torch.ones((num_views, num_views), device=device, dtype=torch.float64)
    if method == "spectral_global_corr":
        return global_spatial_covariance_target(
            num_views,
            height,
            width,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            device=device,
            dtype=torch.float64,
        )
    if method == "camera_rbf_corr":
        return camera_rbf_spatial_covariance_target(
            view_angles,
            height,
            width,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            length_scale=camera_length_scale,
            device=device,
            dtype=torch.float64,
        )
    if method in {"nested_tree_a", "nested_tree_ab"}:
        return nested_tree_spatial_covariance_target(
            view_angles,
            height,
            width,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            tree_mode="a" if method == "nested_tree_a" else "ab",
            device=device,
            dtype=torch.float64,
        )
    raise ValueError("unsupported formal method: {}".format(method))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hard white-Gaussian/covariance gates before MV-Adapter inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--methods", nargs="+", choices=FORMAL_METHODS, default=FORMAL_METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument(
        "--azimuth-deg",
        type=float,
        nargs="+",
        default=[0, 45, 90, 180, 270, 315],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--frequency-scale", type=float, default=0.12)
    parser.add_argument("--camera-length-scale", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-abs-mean", type=float, default=None)
    parser.add_argument("--min-std", type=float, default=None)
    parser.add_argument("--max-std", type=float, default=None)
    parser.add_argument("--max-abs-lag-autocorrelation", type=float, default=None)
    parser.add_argument("--max-radial-psd-deviation", type=float, default=None)
    parser.add_argument("--max-axis-stripe-score", type=float, default=None)
    parser.add_argument("--max-cross-view-covariance-mae", type=float, default=None)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.channels <= 0 or args.height <= 1 or args.width <= 1:
        parser.error("batch size/channels must be positive and spatial dimensions > 1")
    if len(args.azimuth_deg) <= 0:
        parser.error("at least one azimuth is required")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must contain non-negative integers")
    if not args.strengths or any(not 0.0 <= value < 1.0 for value in args.strengths):
        parser.error("--strengths must contain values in [0, 1)")
    if (
        not math.isfinite(args.frequency_scale)
        or not math.isfinite(args.camera_length_scale)
        or args.frequency_scale <= 0.0
        or args.camera_length_scale <= 0.0
    ):
        parser.error("frequency and camera length scales must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")


def _threshold_overrides(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    mapping = {
        "max_abs_mean": args.max_abs_mean,
        "min_std": args.min_std,
        "max_std": args.max_std,
        "max_abs_lag_autocorrelation": args.max_abs_lag_autocorrelation,
        "max_radial_psd_deviation": args.max_radial_psd_deviation,
        "max_axis_stripe_score": args.max_axis_stripe_score,
        "max_cross_view_covariance_mae": args.max_cross_view_covariance_mae,
    }
    overrides = {key: value for key, value in mapping.items() if value is not None}
    return overrides or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    methods = list(dict.fromkeys(args.methods))
    seeds = list(dict.fromkeys(args.seeds))
    strengths = list(dict.fromkeys(args.strengths))
    thresholds = _threshold_overrides(args)

    records = []
    for method in methods:
        method_strengths = [0.0] if method in STRENGTH_INDEPENDENT_METHODS else strengths
        for seed in seeds:
            for strength in method_strengths:
                single = run_preflight(
                    method,
                    view_angles=args.azimuth_deg,
                    seed=seed,
                    max_correlation=strength,
                    frequency_scale=args.frequency_scale,
                    camera_length_scale=args.camera_length_scale,
                    batch_size=args.batch_size,
                    channels=args.channels,
                    height=args.height,
                    width=args.width,
                    device=args.device,
                    thresholds=thresholds,
                )
                records.append(single["record"])

    payload = {
        "schema_version": 2,
        "passed": all(bool(record.get("passed")) for record in records),
        "config": {
            "methods": methods,
            "seeds": seeds,
            "strengths": strengths,
            "batch_size": args.batch_size,
            "num_views": len(args.azimuth_deg),
            "channels": args.channels,
            "height": args.height,
            "width": args.width,
            "azimuth_deg": list(args.azimuth_deg),
            "frequency_scale": args.frequency_scale,
            "camera_length_scale": args.camera_length_scale,
            "device": str(torch.device(args.device)),
        },
        "records": records,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print("Diagnostics: {}".format(output))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
