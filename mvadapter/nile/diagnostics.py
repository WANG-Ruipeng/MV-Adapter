"""Pre-inference diagnostics for generated multi-view latent fields.

These checks intentionally operate on unstandardised sampler output.  A
sampler that only passes after per-sample mean/variance repair has not
preserved the diffusion model's IID Gaussian input law.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch


Lag = Tuple[int, int]
DEFAULT_DISTRIBUTION_THRESHOLDS = {
    "max_abs_mean": 0.01,
    "min_std": 0.99,
    "max_std": 1.01,
    "max_abs_lag_autocorrelation": 0.02,
    "max_radial_psd_deviation": 0.05,
    "max_axis_stripe_score": 0.15,
    "max_cross_view_covariance_mae": 0.03,
}

DEFAULT_LOWRANK_GATE_THRESHOLDS = {
    "max_basis_orthonormality_error": 1e-6,
    "max_basis_coefficient_covariance_mae": 0.03,
    "max_joint_kl_relative_error": 1e-5,
    "min_covariance_eigenvalue": 1e-8,
}


def _validate_latents(latents: torch.Tensor) -> torch.Tensor:
    if not isinstance(latents, torch.Tensor):
        raise TypeError("latents must be a torch.Tensor")
    if latents.ndim != 4 or latents.numel() == 0:
        raise ValueError("latents must have non-empty [B*V, C, H, W] shape")
    if not latents.is_floating_point():
        raise TypeError("latents must have a floating-point dtype")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("latents must contain only finite values")
    return latents


def _reshape_views(
    latents: torch.Tensor, batch_size: int, num_views: int
) -> torch.Tensor:
    latents = _validate_latents(latents)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_views, bool) or not isinstance(num_views, int) or num_views <= 0:
        raise ValueError("num_views must be a positive integer")
    if latents.shape[0] != batch_size * num_views:
        raise ValueError(
            "latents first dimension must equal batch_size * num_views"
        )
    return latents.reshape(
        batch_size,
        num_views,
        latents.shape[1],
        latents.shape[2],
        latents.shape[3],
    )


def moment_statistics(values: torch.Tensor) -> Dict[str, float]:
    """Return mean, population std, skewness, and Gaussian-style kurtosis."""

    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        raise ValueError("values must be a non-empty torch.Tensor")
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")
    work = values.detach().to(dtype=torch.float64).reshape(-1)
    mean = work.mean()
    centered = work - mean
    variance = centered.square().mean()
    std = variance.sqrt()
    if float(std.item()) == 0.0:
        skewness = torch.tensor(float("nan"), device=work.device, dtype=work.dtype)
        kurtosis = torch.tensor(float("nan"), device=work.device, dtype=work.dtype)
    else:
        standardized = centered / std
        skewness = standardized.pow(3).mean()
        kurtosis = standardized.pow(4).mean()
    return {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "skewness": float(skewness.item()),
        "kurtosis": float(kurtosis.item()),
        "excess_kurtosis": float((kurtosis - 3.0).item()),
    }


def per_view_moment_statistics(
    latents: torch.Tensor, batch_size: int, num_views: int
) -> Sequence[Dict[str, float]]:
    """Return moment statistics pooled over batch/channel/space per view."""

    views = _reshape_views(latents, batch_size, num_views)
    return [moment_statistics(views[:, index]) for index in range(num_views)]


def _lagged_pair(values: torch.Tensor, dy: int, dx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = values.shape[-2:]
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("lag magnitude must be smaller than the spatial dimensions")
    if dy >= 0:
        ay, by = slice(0, height - dy), slice(dy, height)
    else:
        ay, by = slice(-dy, height), slice(0, height + dy)
    if dx >= 0:
        ax, bx = slice(0, width - dx), slice(dx, width)
    else:
        ax, bx = slice(-dx, width), slice(0, width + dx)
    return values[..., ay, ax], values[..., by, bx]


def lag_autocorrelations(
    latents: torch.Tensor,
    lags: Sequence[Lag] = ((0, 1), (1, 0), (1, 1)),
) -> Dict[str, Any]:
    """Measure non-wrapping Pearson autocorrelation at requested spatial lags."""

    latents = _validate_latents(latents).detach().to(dtype=torch.float64)
    if len(lags) == 0:
        raise ValueError("lags must contain at least one non-zero lag")
    values = {}
    for lag in lags:
        if len(lag) != 2:
            raise ValueError("each lag must be a (dy, dx) pair")
        dy, dx = int(lag[0]), int(lag[1])
        if dy == 0 and dx == 0:
            raise ValueError("zero lag is not a useful autocorrelation diagnostic")
        first, second = _lagged_pair(latents, dy, dx)
        first = first.reshape(-1)
        second = second.reshape(-1)
        first = first - first.mean()
        second = second - second.mean()
        denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
        correlation = first.mul(second).mean() / denominator.clamp_min(1e-15)
        values["{},{}".format(dy, dx)] = float(correlation.item())
    return {
        "values": values,
        "max_abs": max(abs(value) for value in values.values()),
    }


def per_view_lag_autocorrelations(
    latents: torch.Tensor,
    batch_size: int,
    num_views: int,
    *,
    lags: Sequence[Lag] = ((0, 1), (1, 0), (1, 1)),
) -> Dict[str, Any]:
    """Measure spatial lag correlations separately for every camera view."""

    views = _reshape_views(latents, batch_size, num_views)
    reports = [
        lag_autocorrelations(views[:, view_index], lags=lags)
        for view_index in range(num_views)
    ]
    return {
        "values": reports,
        "mean_max_abs": sum(report["max_abs"] for report in reports)
        / float(len(reports)),
        "max_abs": max(report["max_abs"] for report in reports),
    }


def radial_power_spectrum(
    latents: torch.Tensor,
    *,
    num_bins: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Return the channel/sample-averaged full-FFT radial power spectrum."""

    latents = _validate_latents(latents)
    height, width = latents.shape[-2:]
    if num_bins is None:
        num_bins = max(8, min(height, width) // 2)
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 1:
        raise ValueError("num_bins must be an integer greater than one")

    work = latents.detach().to(dtype=torch.float32)
    power = torch.fft.fft2(work, norm="ortho").abs().square().mean(dim=(0, 1))
    fy = torch.fft.fftfreq(height, device=work.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=work.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    max_radius = float(radius.max().item())
    edges = torch.linspace(
        0.0,
        max_radius + torch.finfo(torch.float32).eps,
        num_bins + 1,
        device=work.device,
        dtype=torch.float32,
    )
    indices = torch.bucketize(radius.reshape(-1), edges[1:-1], right=False)
    counts = torch.zeros(num_bins, device=work.device, dtype=torch.float32)
    sums = torch.zeros_like(counts)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    sums.scatter_add_(0, indices, power.reshape(-1))
    profile = sums / counts.clamp_min(1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"radii": centers, "power": profile, "counts": counts}


def radial_psd_deviation(
    latents: torch.Tensor,
    *,
    reference: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]] = None,
    num_bins: Optional[int] = None,
) -> float:
    """Return weighted relative radial-PSD shape error versus IID/reference.

    Both profiles are divided by their count-weighted mean power.  This makes
    the metric diagnose spectral colour while standard deviation remains a
    separate hard gate.
    """

    observed = radial_power_spectrum(latents, num_bins=num_bins)
    if reference is None:
        reference_power = torch.ones_like(observed["power"])
        reference_counts = observed["counts"]
    elif isinstance(reference, torch.Tensor):
        reference_report = radial_power_spectrum(reference, num_bins=observed["power"].numel())
        reference_power = reference_report["power"].to(observed["power"].device)
        reference_counts = reference_report["counts"].to(observed["counts"].device)
    else:
        if "power" not in reference:
            raise ValueError("reference PSD mapping must contain 'power'")
        reference_power = reference["power"].to(
            device=observed["power"].device, dtype=observed["power"].dtype
        )
        reference_counts = reference.get("counts", observed["counts"]).to(
            device=observed["counts"].device, dtype=observed["counts"].dtype
        )
    if reference_power.shape != observed["power"].shape:
        raise ValueError("reference and observed radial PSD profiles must have equal shape")

    valid = (observed["counts"] > 0) & (reference_counts > 0)
    counts = observed["counts"][valid]
    observed_power = observed["power"][valid]
    reference_power = reference_power[valid]
    observed_mean = (observed_power * counts).sum() / counts.sum()
    reference_mean = (reference_power * counts).sum() / counts.sum()
    observed_shape = observed_power / observed_mean.clamp_min(1e-15)
    reference_shape = reference_power / reference_mean.clamp_min(1e-15)
    relative = (observed_shape - reference_shape).abs() / reference_shape.abs().clamp_min(1e-8)
    return float(((relative * counts).sum() / counts.sum()).item())


def coarse_radial_psd_deviation(
    latents: torch.Tensor,
    *,
    reference: Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]] = None,
    num_bands: int = 4,
) -> Dict[str, Any]:
    """Return worst spectral-shape error over a few well-sampled radial bands.

    The legacy count-weighted scalar can hide a damaged low-frequency band
    because high-radius annuli contain many more Fourier cells.  Four coarse
    bands retain radial locality while pooling enough coefficients that a
    legal IID diagnostic ensemble is not rejected by ordinary sampling noise.
    """

    if isinstance(num_bands, bool) or not isinstance(num_bands, int) or num_bands <= 1:
        raise ValueError("num_bands must be an integer greater than one")
    observed = radial_power_spectrum(latents, num_bins=num_bands)
    if reference is None:
        reference_power = torch.ones_like(observed["power"])
        reference_counts = observed["counts"]
    elif isinstance(reference, torch.Tensor):
        if reference.shape[-2:] != latents.shape[-2:]:
            raise ValueError("reference must have the same spatial shape as latents")
        reference_report = radial_power_spectrum(reference, num_bins=num_bands)
        reference_power = reference_report["power"].to(observed["power"].device)
        reference_counts = reference_report["counts"].to(observed["counts"].device)
    else:
        if "power" not in reference:
            raise ValueError("reference PSD mapping must contain 'power'")
        reference_power = reference["power"].to(
            device=observed["power"].device, dtype=observed["power"].dtype
        )
        reference_counts = reference.get("counts", observed["counts"]).to(
            device=observed["counts"].device, dtype=observed["counts"].dtype
        )
    if reference_power.shape != observed["power"].shape:
        raise ValueError("reference and observed coarse PSD profiles must have equal shape")

    valid = (observed["counts"] > 0) & (reference_counts > 0)
    counts = observed["counts"][valid]
    observed_power = observed["power"][valid]
    reference_power = reference_power[valid]
    observed_mean = (observed_power * counts).sum() / counts.sum()
    reference_mean = (reference_power * counts).sum() / counts.sum()
    observed_shape = observed_power / observed_mean.clamp_min(1e-15)
    reference_shape = reference_power / reference_mean.clamp_min(1e-15)
    relative = (
        (observed_shape - reference_shape).abs()
        / reference_shape.abs().clamp_min(1e-8)
    )
    values = [float(value) for value in relative.cpu().tolist()]
    return {
        "values": values,
        "mean": float(((relative * counts).sum() / counts.sum()).item()),
        "max": max(values),
        "radii": [
            float(value)
            for value in observed["radii"][valid].cpu().tolist()
        ],
        "counts": [
            int(value)
            for value in observed["counts"][valid].cpu().tolist()
        ],
    }


def per_view_radial_psd_deviation(
    latents: torch.Tensor,
    batch_size: int,
    num_views: int,
    *,
    reference_latents: Optional[torch.Tensor] = None,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure radial-PSD shape error independently for every camera view."""

    views = _reshape_views(latents, batch_size, num_views)
    reference_views = None
    if reference_latents is not None:
        reference_views = _reshape_views(reference_latents, batch_size, num_views)
        if reference_views.shape[-2:] != views.shape[-2:]:
            raise ValueError("reference_latents must have the same spatial shape")
    values = []
    for view_index in range(num_views):
        reference = None if reference_views is None else reference_views[:, view_index]
        values.append(
            radial_psd_deviation(
                views[:, view_index], reference=reference, num_bins=num_bins
            )
        )
    return {
        "values": values,
        "mean": sum(values) / float(len(values)),
        "max": max(values),
    }


def spectral_axis_stripe_score(
    latents: torch.Tensor,
    *,
    axis_tolerance: float = 0.08,
    minimum_radius: float = 0.05,
) -> float:
    """Detect excess Fourier energy near horizontal or vertical axes.

    A horizontal/vertical spatial stripe produces Fourier energy on the
    corresponding frequency axis. The score compares the observed axial
    energy fraction with the fraction of cells in the same angular wedges.
    Isotropic white noise is near zero; one means a 100% relative excess.
    """

    latents = _validate_latents(latents)
    axis_tolerance = float(axis_tolerance)
    minimum_radius = float(minimum_radius)
    if not math.isfinite(axis_tolerance) or not 0.0 < axis_tolerance < 1.0:
        raise ValueError("axis_tolerance must lie strictly between zero and one")
    if not math.isfinite(minimum_radius) or minimum_radius < 0.0:
        raise ValueError("minimum_radius must be finite and non-negative")

    work = latents.detach().to(dtype=torch.float32)
    height, width = work.shape[-2:]
    power = torch.fft.fft2(work, norm="ortho").abs().square().mean(dim=(0, 1))
    fy = torch.fft.fftfreq(height, device=work.device, dtype=torch.float32)[:, None]
    fx = torch.fft.fftfreq(width, device=work.device, dtype=torch.float32)[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    valid = radius >= minimum_radius
    axial = valid & (
        (fx.abs() <= axis_tolerance * radius)
        | (fy.abs() <= axis_tolerance * radius)
    )
    valid_count = int(valid.sum().item())
    axial_count = int(axial.sum().item())
    if valid_count == 0 or axial_count == 0:
        raise ValueError("spatial dimensions are too small for the stripe diagnostic")
    expected_fraction = float(axial_count) / float(valid_count)
    valid_power = power[valid].sum()
    observed_fraction = power[axial].sum() / valid_power.clamp_min(1e-15)
    return abs(float(observed_fraction.item()) - expected_fraction) / expected_fraction


def per_view_axis_stripe_scores(
    latents: torch.Tensor,
    batch_size: int,
    num_views: int,
    *,
    axis_tolerance: float = 0.08,
    minimum_radius: float = 0.05,
) -> Dict[str, Any]:
    """Return per-view and worst-case axis-stripe scores."""

    views = _reshape_views(latents, batch_size, num_views)
    values = [
        spectral_axis_stripe_score(
            views[:, view_index],
            axis_tolerance=axis_tolerance,
            minimum_radius=minimum_radius,
        )
        for view_index in range(num_views)
    ]
    return {
        "values": values,
        "mean": sum(values) / float(len(values)),
        "max": max(values),
    }


def cross_view_radial_frequency_correlation(
    latents: torch.Tensor,
    batch_size: int,
    num_views: int,
    *,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Estimate complex cross-view covariance/correlation in radial FFT bands.

    Each matrix is estimated over batch, channel, and all Fourier coefficients
    in one radial band. The real part is reported because all target view
    covariances in this experiment are real and phase-neutral.
    """

    views = _reshape_views(latents, batch_size, num_views).detach().to(
        dtype=torch.float32
    )
    height, width = views.shape[-2:]
    if num_bins is None:
        num_bins = max(8, min(height, width) // 2)
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 1:
        raise ValueError("num_bins must be an integer greater than one")

    coefficients = torch.fft.fft2(views, norm="ortho")
    fy = torch.fft.fftfreq(height, device=views.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=views.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    max_radius = float(radius.max().item())
    edges = torch.linspace(
        0.0,
        max_radius + torch.finfo(torch.float32).eps,
        num_bins + 1,
        device=views.device,
        dtype=torch.float32,
    )
    band_ids = torch.bucketize(radius.reshape(-1), edges[1:-1], right=False)
    flattened = coefficients.reshape(
        batch_size, num_views, views.shape[2], height * width
    )
    covariance_matrices = []
    correlation_matrices = []
    radii = []
    counts = []
    for band_index in range(num_bins):
        mask = band_ids == band_index
        frequency_count = int(mask.sum().item())
        if frequency_count == 0:
            continue
        samples = flattened[..., mask].permute(1, 0, 2, 3).reshape(num_views, -1)
        covariance = samples.matmul(samples.conj().mT).real / float(samples.shape[1])
        diagonal = torch.diagonal(covariance).clamp_min(1e-15)
        denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :])
        correlation = covariance / denominator
        covariance_matrices.append(
            [[float(value) for value in row] for row in covariance.cpu().tolist()]
        )
        correlation_matrices.append(
            [[float(value) for value in row] for row in correlation.cpu().tolist()]
        )
        radii.append(float((0.5 * (edges[band_index] + edges[band_index + 1])).item()))
        counts.append(frequency_count)
    return {
        "radii": radii,
        "frequency_counts": counts,
        "covariance": covariance_matrices,
        "correlation": correlation_matrices,
    }


def empirical_cross_view_covariance(
    latents: torch.Tensor,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    """Estimate same-coordinate view covariance over batch/channel/space."""

    views = _reshape_views(latents, batch_size, num_views).detach().to(dtype=torch.float64)
    flattened = views.permute(1, 0, 2, 3, 4).reshape(num_views, -1)
    flattened = flattened - flattened.mean(dim=1, keepdim=True)
    return flattened.matmul(flattened.mT) / float(flattened.shape[1])


def empirical_view_covariance(samples: torch.Tensor) -> torch.Tensor:
    """Estimate a population covariance from ``[observations, views]``.

    This helper is shared by spatial projections and supplemental IID
    coefficient ensembles.  It only subtracts the ensemble mean; it never
    standardises an observation or repairs its variance.
    """

    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("samples must have shape [observations>=2, views>=1]")
    if not samples.is_floating_point():
        raise TypeError("samples must have a floating-point dtype")
    if not bool(torch.isfinite(samples).all()):
        raise ValueError("samples must contain only finite values")
    work = samples.detach().to(dtype=torch.float64)
    centered = work - work.mean(dim=0, keepdim=True)
    return centered.mT.matmul(centered) / float(centered.shape[0])


def project_basis_coefficients(
    latents: torch.Tensor,
    basis: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    """Project latent fields onto an orthonormal basis as ``[B,V,K]``."""

    views = _reshape_views(latents, batch_size, num_views)
    if not isinstance(basis, torch.Tensor):
        raise TypeError("basis must be a torch.Tensor")
    ambient_dimension = int(views.shape[2] * views.shape[3] * views.shape[4])
    if basis.ndim != 2 or basis.shape[0] != ambient_dimension or basis.shape[1] < 1:
        raise ValueError(
            "basis must have shape [{}, K] with K > 0".format(ambient_dimension)
        )
    if not basis.is_floating_point() or not bool(torch.isfinite(basis).all()):
        raise ValueError("basis must be a finite floating-point matrix")
    basis64 = basis.detach().to(device="cpu", dtype=torch.float64)
    gram = basis64.mT.matmul(basis64)
    identity = torch.eye(basis64.shape[1], dtype=torch.float64)
    error = float((gram - identity).abs().max().item())
    if error >= 1e-6:
        raise ValueError(
            "basis must be orthonormal with max error < 1e-6; got {:.3e}".format(
                error
            )
        )
    flat = views.reshape(batch_size, num_views, ambient_dimension).to(
        dtype=torch.float64
    )
    return flat.matmul(basis64.to(device=flat.device))


def empirical_basis_coefficient_covariance(
    latents: torch.Tensor,
    basis: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    additional_samples: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Estimate cross-view covariance over independent basis coefficients.

    Every ``(batch, basis-column)`` pair is one observation.  Optional
    ``additional_samples`` must already have shape ``[N,V]`` and is useful for
    a large, explicitly reported coefficient-space Monte Carlo ensemble.
    """

    coefficients = project_basis_coefficients(
        latents,
        basis,
        batch_size=batch_size,
        num_views=num_views,
    )
    samples = coefficients.permute(0, 2, 1).reshape(-1, num_views)
    if additional_samples is not None:
        if (
            not isinstance(additional_samples, torch.Tensor)
            or additional_samples.ndim != 2
            or additional_samples.shape[1] != num_views
        ):
            raise ValueError("additional_samples must have shape [N, num_views]")
        if not additional_samples.is_floating_point() or not bool(
            torch.isfinite(additional_samples).all()
        ):
            raise ValueError("additional_samples must be finite floating point")
        samples = torch.cat(
            (samples, additional_samples.to(device=samples.device, dtype=samples.dtype)),
            dim=0,
        )
    return empirical_view_covariance(samples)


def diagnose_lowrank_latents(
    latents: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    basis: torch.Tensor,
    coefficient_target_covariance: torch.Tensor,
    reference_latents: Optional[torch.Tensor] = None,
    full_space_target_covariance: Optional[torch.Tensor] = None,
    additional_coefficient_samples: Optional[torch.Tensor] = None,
    target_kl: Optional[float] = None,
    achieved_kl: Optional[float] = None,
    alpha: Optional[float] = None,
    lags: Sequence[Lag] = ((0, 1), (1, 0), (1, 1)),
    radial_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the ordinary marginal report plus low-rank joint-law checks."""

    if full_space_target_covariance is None:
        full_space_target_covariance = torch.eye(
            num_views, dtype=torch.float64, device=latents.device
        )
    report = diagnose_latents(
        latents,
        batch_size=batch_size,
        num_views=num_views,
        reference_latents=reference_latents,
        target_covariance=full_space_target_covariance,
        lags=lags,
        radial_bins=radial_bins,
    )
    coefficients = project_basis_coefficients(
        latents,
        basis,
        batch_size=batch_size,
        num_views=num_views,
    )
    spatial_samples = coefficients.permute(0, 2, 1).reshape(-1, num_views)
    all_samples = spatial_samples
    supplemental_count = 0
    if additional_coefficient_samples is not None:
        if (
            not isinstance(additional_coefficient_samples, torch.Tensor)
            or additional_coefficient_samples.ndim != 2
            or additional_coefficient_samples.shape[1] != num_views
        ):
            raise ValueError(
                "additional_coefficient_samples must have shape [N, num_views]"
            )
        if not additional_coefficient_samples.is_floating_point() or not bool(
            torch.isfinite(additional_coefficient_samples).all()
        ):
            raise ValueError(
                "additional_coefficient_samples must be finite floating point"
            )
        supplemental_count = int(additional_coefficient_samples.shape[0])
        all_samples = torch.cat(
            (
                spatial_samples,
                additional_coefficient_samples.to(
                    device=spatial_samples.device, dtype=spatial_samples.dtype
                ),
            ),
            dim=0,
        )
    empirical = empirical_view_covariance(all_samples)
    target = coefficient_target_covariance.detach().to(
        device=empirical.device, dtype=torch.float64
    )
    covariance_error = cross_view_covariance_error(empirical, target)

    basis64 = basis.detach().to(device="cpu", dtype=torch.float64)
    identity = torch.eye(basis64.shape[1], dtype=torch.float64)
    orthonormality_error = float(
        (basis64.mT.matmul(basis64) - identity).abs().max().item()
    )
    eigenvalues = torch.linalg.eigvalsh(target)
    minimum_eigenvalue = float(eigenvalues.min().item())
    maximum_eigenvalue = float(eigenvalues.max().item())
    condition_number = (
        maximum_eigenvalue / minimum_eigenvalue
        if minimum_eigenvalue > 0.0
        else None
    )
    if target_kl is None or achieved_kl is None:
        kl_relative_error = None
    elif float(target_kl) == 0.0:
        kl_relative_error = abs(float(achieved_kl))
    else:
        kl_relative_error = abs(float(achieved_kl) - float(target_kl)) / abs(
            float(target_kl)
        )

    report["basis"] = {
        "ambient_dimension": int(basis.shape[0]),
        "rank": int(basis.shape[1]),
        "orthonormality_max_error": orthonormality_error,
    }
    report["basis_coefficient_covariance"] = {
        "empirical": [
            [float(value) for value in row] for row in empirical.cpu().tolist()
        ],
        "target": [
            [float(value) for value in row] for row in target.cpu().tolist()
        ],
        "error": covariance_error,
        "observation_count": int(all_samples.shape[0]),
        "spatial_projection_observation_count": int(spatial_samples.shape[0]),
        "supplemental_observation_count": supplemental_count,
        "per_sample_standardization": False,
    }
    report["joint_kl"] = {
        "target": None if target_kl is None else float(target_kl),
        "achieved": None if achieved_kl is None else float(achieved_kl),
        "relative_error": kl_relative_error,
        "alpha": None if alpha is None else float(alpha),
    }
    report["coefficient_covariance_spectrum"] = {
        "eigenvalues": [float(value) for value in eigenvalues.cpu().tolist()],
        "min_eigenvalue": minimum_eigenvalue,
        "max_eigenvalue": maximum_eigenvalue,
        "condition_number": condition_number,
    }
    return report


def cross_view_covariance_error(
    empirical: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Return all-entry and off-diagonal target-covariance errors."""

    if not isinstance(empirical, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("empirical and target must be torch.Tensor instances")
    if empirical.ndim != 2 or empirical.shape[0] != empirical.shape[1]:
        raise ValueError("empirical covariance must be square")
    if target.shape != empirical.shape:
        raise ValueError("target covariance must match empirical covariance shape")
    difference = empirical.to(dtype=torch.float64) - target.to(
        device=empirical.device, dtype=torch.float64
    )
    off_diagonal = ~torch.eye(
        difference.shape[0], device=difference.device, dtype=torch.bool
    )
    if bool(off_diagonal.any()):
        off_diagonal_error = difference[off_diagonal].abs()
        off_diagonal_mae = float(off_diagonal_error.mean().item())
        off_diagonal_max = float(off_diagonal_error.max().item())
    else:
        off_diagonal_mae = 0.0
        off_diagonal_max = 0.0
    return {
        "mae": float(difference.abs().mean().item()),
        "max_abs": float(difference.abs().max().item()),
        "offdiag_mae": off_diagonal_mae,
        "offdiag_max_abs": off_diagonal_max,
    }


def diagnose_latents(
    latents: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    reference_latents: Optional[torch.Tensor] = None,
    target_covariance: Optional[torch.Tensor] = None,
    lags: Sequence[Lag] = ((0, 1), (1, 0), (1, 1)),
    radial_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-serialisable distribution report for one sampler."""

    views = _reshape_views(latents, batch_size, num_views)
    if reference_latents is not None:
        _validate_latents(reference_latents)
        if reference_latents.shape[-2:] != latents.shape[-2:]:
            raise ValueError("reference_latents must have the same spatial shape")
    moments = moment_statistics(latents)
    per_view = per_view_moment_statistics(latents, batch_size, num_views)
    lag_report = lag_autocorrelations(latents, lags=lags)
    per_view_lag_report = per_view_lag_autocorrelations(
        latents, batch_size, num_views, lags=lags
    )
    psd = radial_power_spectrum(latents, num_bins=radial_bins)
    psd_deviation = radial_psd_deviation(
        latents, reference=reference_latents, num_bins=psd["power"].numel()
    )
    coarse_psd_deviation = coarse_radial_psd_deviation(
        latents, reference=reference_latents
    )
    per_view_psd_deviation = per_view_radial_psd_deviation(
        latents,
        batch_size,
        num_views,
        reference_latents=reference_latents,
        num_bins=psd["power"].numel(),
    )
    stripe_score = spectral_axis_stripe_score(latents)
    per_view_stripe_score = per_view_axis_stripe_scores(
        latents, batch_size, num_views
    )
    empirical_covariance = empirical_cross_view_covariance(
        latents, batch_size, num_views
    )
    frequency_correlation = cross_view_radial_frequency_correlation(
        latents,
        batch_size,
        num_views,
        num_bins=psd["power"].numel(),
    )
    covariance_error = None
    if target_covariance is not None:
        covariance_error = cross_view_covariance_error(
            empirical_covariance, target_covariance
        )

    return {
        "shape": list(latents.shape),
        "batch_size": batch_size,
        "num_views": num_views,
        "global": moments,
        "per_view": list(per_view),
        "lag_autocorrelation": lag_report,
        "per_view_lag_autocorrelation": per_view_lag_report,
        "radial_psd": {
            "radii": [float(value) for value in psd["radii"].cpu().tolist()],
            "power": [float(value) for value in psd["power"].cpu().tolist()],
            "counts": [int(value) for value in psd["counts"].cpu().tolist()],
        },
        "radial_psd_deviation": psd_deviation,
        "coarse_radial_psd_deviation": coarse_psd_deviation,
        "per_view_radial_psd_deviation": per_view_psd_deviation,
        "axis_stripe_score": {
            "pooled": stripe_score,
            "per_view": per_view_stripe_score["values"],
            "per_view_mean": per_view_stripe_score["mean"],
            "max": max(stripe_score, per_view_stripe_score["max"]),
        },
        "cross_view_frequency": frequency_correlation,
        "cross_view_covariance": [
            [float(value) for value in row]
            for row in empirical_covariance.cpu().tolist()
        ],
        "cross_view_covariance_error": covariance_error,
    }


def _finite_thresholds(overrides: Optional[Mapping[str, float]]) -> Dict[str, float]:
    thresholds = dict(DEFAULT_DISTRIBUTION_THRESHOLDS)
    if overrides is not None:
        unknown = set(overrides) - set(thresholds)
        if unknown:
            raise ValueError("unknown distribution thresholds: {}".format(sorted(unknown)))
        thresholds.update({key: float(value) for key, value in overrides.items()})
    if any(not math.isfinite(value) for value in thresholds.values()):
        raise ValueError("all distribution thresholds must be finite")
    if thresholds["min_std"] > thresholds["max_std"]:
        raise ValueError("min_std must not exceed max_std")
    return thresholds


def evaluate_distribution_gates(
    report: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    require_covariance_target: bool = True,
) -> Dict[str, Any]:
    """Apply the prompt's hard pre-inference thresholds to a diagnostic report."""

    limits = _finite_thresholds(thresholds)
    global_stats = report["global"]
    per_view_stats = list(report.get("per_view", ()))
    global_mean_value = abs(float(global_stats["mean"]))
    per_view_mean_value = max(
        (abs(float(stats["mean"])) for stats in per_view_stats),
        default=global_mean_value,
    )
    mean_value = max(global_mean_value, per_view_mean_value)
    global_std_value = float(global_stats["std"])
    per_view_std_min = min(
        (float(stats["std"]) for stats in per_view_stats),
        default=global_std_value,
    )
    per_view_std_max = max(
        (float(stats["std"]) for stats in per_view_stats),
        default=global_std_value,
    )
    pooled_lag_value = float(report["lag_autocorrelation"]["max_abs"])
    per_view_lag_value = float(
        report.get("per_view_lag_autocorrelation", {}).get(
            "max_abs", pooled_lag_value
        )
    )
    lag_value = max(pooled_lag_value, per_view_lag_value)
    psd_value = float(report["radial_psd_deviation"])
    per_view_psd_value = float(report["per_view_radial_psd_deviation"]["max"])
    coarse_psd_value = float(
        report.get("coarse_radial_psd_deviation", {}).get("max", psd_value)
    )
    psd_gate_value = max(psd_value, per_view_psd_value, coarse_psd_value)
    stripe_value = float(report["axis_stripe_score"]["max"])
    higher_moments = [
        float(global_stats["skewness"]),
        float(global_stats["excess_kurtosis"]),
    ]
    for stats in per_view_stats:
        higher_moments.extend(
            (float(stats["skewness"]), float(stats["excess_kurtosis"]))
        )
    checks = {
        "mean": {
            "value": mean_value,
            "pooled_value": global_mean_value,
            "per_view_max": per_view_mean_value,
            "limit": limits["max_abs_mean"],
            "passed": mean_value < limits["max_abs_mean"],
        },
        "std": {
            "value": global_std_value,
            "pooled_value": global_std_value,
            "per_view_min": per_view_std_min,
            "per_view_max": per_view_std_max,
            "minimum": limits["min_std"],
            "maximum": limits["max_std"],
            "passed": limits["min_std"]
            <= min(global_std_value, per_view_std_min)
            and max(global_std_value, per_view_std_max)
            <= limits["max_std"],
        },
        "lag_autocorrelation": {
            "value": lag_value,
            "pooled_value": pooled_lag_value,
            "per_view_max": per_view_lag_value,
            "limit": limits["max_abs_lag_autocorrelation"],
            "passed": lag_value < limits["max_abs_lag_autocorrelation"],
        },
        "radial_psd": {
            "value": psd_gate_value,
            "pooled_value": psd_value,
            "per_view_max": per_view_psd_value,
            "coarse_band_max": coarse_psd_value,
            "limit": limits["max_radial_psd_deviation"],
            "passed": psd_gate_value < limits["max_radial_psd_deviation"],
        },
        "axis_stripes": {
            "value": stripe_value,
            "limit": limits["max_axis_stripe_score"],
            "passed": stripe_value < limits["max_axis_stripe_score"],
        },
        "finite_higher_moments": {
            "values": higher_moments,
            "passed": all(math.isfinite(value) for value in higher_moments),
        },
    }
    covariance_error = report.get("cross_view_covariance_error")
    if covariance_error is None:
        checks["cross_view_covariance"] = {
            "value": None,
            "limit": limits["max_cross_view_covariance_mae"],
            "passed": not require_covariance_target,
            "evaluated": False,
        }
    else:
        covariance_value = float(
            covariance_error.get("offdiag_mae", covariance_error["mae"])
        )
        checks["cross_view_covariance"] = {
            "value": covariance_value,
            "all_entry_mae": float(covariance_error["mae"]),
            "offdiag_mae": covariance_value,
            "limit": limits["max_cross_view_covariance_mae"],
            "passed": covariance_value < limits["max_cross_view_covariance_mae"],
            "evaluated": True,
        }
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "thresholds": limits,
    }


def _finite_lowrank_thresholds(
    overrides: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    thresholds = dict(DEFAULT_LOWRANK_GATE_THRESHOLDS)
    if overrides is not None:
        unknown = set(overrides) - set(thresholds)
        if unknown:
            raise ValueError(
                "unknown low-rank thresholds: {}".format(sorted(unknown))
            )
        thresholds.update({key: float(value) for key, value in overrides.items()})
    if any(not math.isfinite(value) for value in thresholds.values()):
        raise ValueError("all low-rank thresholds must be finite")
    if thresholds["min_covariance_eigenvalue"] < 0.0:
        raise ValueError("min_covariance_eigenvalue must be non-negative")
    return thresholds


def evaluate_lowrank_distribution_gates(
    report: Mapping[str, Any],
    *,
    distribution_thresholds: Optional[Mapping[str, float]] = None,
    lowrank_thresholds: Optional[Mapping[str, float]] = None,
    require_finite_kl: bool = True,
) -> Dict[str, Any]:
    """Apply marginal, basis, coefficient-covariance, KL, and eigen gates."""

    base = evaluate_distribution_gates(
        report,
        thresholds=distribution_thresholds,
        require_covariance_target=True,
    )
    limits = _finite_lowrank_thresholds(lowrank_thresholds)
    basis_report = report.get("basis")
    coefficient_report = report.get("basis_coefficient_covariance")
    spectrum = report.get("coefficient_covariance_spectrum")
    joint_kl = report.get("joint_kl")
    if not isinstance(basis_report, Mapping):
        raise ValueError("report is missing basis diagnostics")
    if not isinstance(coefficient_report, Mapping):
        raise ValueError("report is missing basis coefficient diagnostics")
    if not isinstance(spectrum, Mapping):
        raise ValueError("report is missing coefficient covariance spectrum")
    if not isinstance(joint_kl, Mapping):
        raise ValueError("report is missing joint KL diagnostics")

    basis_error = float(basis_report["orthonormality_max_error"])
    covariance_error = coefficient_report.get("error")
    if not isinstance(covariance_error, Mapping):
        raise ValueError("basis coefficient report is missing covariance error")
    covariance_mae = float(
        covariance_error.get("offdiag_mae", covariance_error["mae"])
    )
    minimum_eigenvalue = float(spectrum["min_eigenvalue"])
    condition_number = spectrum.get("condition_number")
    kl_relative_error = joint_kl.get("relative_error")
    kl_evaluated = kl_relative_error is not None
    kl_passed = (
        float(kl_relative_error) < limits["max_joint_kl_relative_error"]
        if kl_evaluated
        else not require_finite_kl
    )
    higher_moments = [
        float(report["global"]["skewness"]),
        float(report["global"]["excess_kurtosis"]),
    ]
    for item in report.get("per_view", ()):
        higher_moments.extend(
            (float(item["skewness"]), float(item["excess_kurtosis"]))
        )

    checks = dict(base["checks"])
    checks.update(
        {
            "finite_higher_moments": {
                "values": higher_moments,
                "passed": all(math.isfinite(value) for value in higher_moments),
            },
            "basis_orthonormality": {
                "value": basis_error,
                "limit": limits["max_basis_orthonormality_error"],
                "passed": basis_error < limits["max_basis_orthonormality_error"],
            },
            "basis_coefficient_covariance": {
                "value": covariance_mae,
                "all_entry_mae": float(covariance_error["mae"]),
                "offdiag_mae": covariance_mae,
                "observation_count": int(
                    coefficient_report.get("observation_count", 0)
                ),
                "limit": limits["max_basis_coefficient_covariance_mae"],
                "passed": covariance_mae
                < limits["max_basis_coefficient_covariance_mae"],
            },
            "joint_kl": {
                "value": (
                    None if kl_relative_error is None else float(kl_relative_error)
                ),
                "target": joint_kl.get("target"),
                "achieved": joint_kl.get("achieved"),
                "alpha": joint_kl.get("alpha"),
                "limit": limits["max_joint_kl_relative_error"],
                "evaluated": kl_evaluated,
                "passed": kl_passed,
            },
            "minimum_eigenvalue": {
                "value": minimum_eigenvalue,
                "minimum": limits["min_covariance_eigenvalue"],
                "passed": minimum_eigenvalue
                > limits["min_covariance_eigenvalue"],
            },
            "covariance_condition_number": {
                "value": condition_number,
                "passed": condition_number is not None
                and math.isfinite(float(condition_number))
                and float(condition_number) >= 1.0,
            },
        }
    )
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
        "thresholds": {**base["thresholds"], **limits},
    }


def assert_distribution_gates(
    report: Mapping[str, Any],
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    require_covariance_target: bool = True,
) -> Dict[str, Any]:
    """Return gate details or raise ``RuntimeError`` when any check fails."""

    result = evaluate_distribution_gates(
        report,
        thresholds=thresholds,
        require_covariance_target=require_covariance_target,
    )
    if not result["passed"]:
        failed = [
            name for name, check in result["checks"].items() if not check["passed"]
        ]
        raise RuntimeError("latent distribution gates failed: {}".format(", ".join(failed)))
    return result


__all__ = [
    "DEFAULT_DISTRIBUTION_THRESHOLDS",
    "DEFAULT_LOWRANK_GATE_THRESHOLDS",
    "assert_distribution_gates",
    "coarse_radial_psd_deviation",
    "cross_view_covariance_error",
    "cross_view_radial_frequency_correlation",
    "diagnose_latents",
    "diagnose_lowrank_latents",
    "empirical_basis_coefficient_covariance",
    "empirical_cross_view_covariance",
    "empirical_view_covariance",
    "evaluate_distribution_gates",
    "evaluate_lowrank_distribution_gates",
    "lag_autocorrelations",
    "moment_statistics",
    "per_view_axis_stripe_scores",
    "per_view_lag_autocorrelations",
    "per_view_moment_statistics",
    "per_view_radial_psd_deviation",
    "project_basis_coefficients",
    "radial_power_spectrum",
    "radial_psd_deviation",
    "spectral_axis_stripe_score",
]
