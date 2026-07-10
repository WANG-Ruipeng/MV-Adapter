"""Tensor operations shared by the NILE samplers and callbacks."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _validate_float_tensor(
    x: torch.Tensor,
    name: str,
    *,
    ndim: Optional[int] = None,
) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {x.dtype}")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"{name} must be non-empty")


def _validate_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be a finite positive number, got {eps}")
    return eps


def _stats_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Choose a stable accumulation dtype while retaining float64 inputs."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def standardize_like(
    x: torch.Tensor,
    ref: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match the per-sample mean and standard deviation of ``x`` to ``ref``.

    Statistics are accumulated in at least float32 and use population variance.
    Population variance avoids ``NaN`` for degenerate one-element samples.  The
    result retains ``x``'s device and dtype.
    """

    _validate_float_tensor(x, "x")
    _validate_float_tensor(ref, "ref")
    eps = _validate_eps(eps)

    if x.ndim < 2:
        raise ValueError(f"x and ref must include batch and feature dimensions, got {x.ndim}D")
    if x.shape != ref.shape:
        raise ValueError(
            f"x and ref must have the same shape, got {tuple(x.shape)} and {tuple(ref.shape)}"
        )
    if x.device != ref.device:
        raise ValueError(f"x and ref must be on the same device, got {x.device} and {ref.device}")

    work_dtype = _stats_dtype(x, ref)
    x_work = x.to(dtype=work_dtype)
    ref_work = ref.to(dtype=work_dtype)
    dims = tuple(range(1, x.ndim))

    x_mean = x_work.mean(dim=dims, keepdim=True)
    x_std = x_work.var(dim=dims, keepdim=True, unbiased=False).sqrt().clamp_min(eps)
    ref_mean = ref_work.mean(dim=dims, keepdim=True)
    ref_std = ref_work.var(dim=dims, keepdim=True, unbiased=False).sqrt().clamp_min(eps)

    result = (x_work - x_mean) / x_std * ref_std + ref_mean
    return result.to(dtype=x.dtype)


def standardize_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize every batch item to approximately zero mean and unit std."""

    _validate_float_tensor(x, "x")
    eps = _validate_eps(eps)
    if x.ndim < 2:
        raise ValueError(f"x must include batch and feature dimensions, got {x.ndim}D")

    work_dtype = _stats_dtype(x)
    x_work = x.to(dtype=work_dtype)
    dims = tuple(range(1, x.ndim))
    mean = x_work.mean(dim=dims, keepdim=True)
    std = x_work.var(dim=dims, keepdim=True, unbiased=False).sqrt().clamp_min(eps)
    return ((x_work - mean) / std).to(dtype=x.dtype)


def gaussian_blur_latent(
    x: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 2.5,
) -> torch.Tensor:
    """Apply a depthwise separable Gaussian blur to ``[B, C, H, W]`` latents.

    Reflection padding is used when the spatial dimension is large enough.  For
    tiny latent maps (where PyTorch reflection padding is undefined), the
    function falls back to replication padding.  This also permits kernels that
    are larger than a latent dimension, which is useful in small unit tests and
    low-resolution experiments.
    """

    _validate_float_tensor(x, "x", ndim=4)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError(f"kernel_size must be an integer, got {type(kernel_size).__name__}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be a finite positive number, got {sigma}")
    if any(size <= 0 for size in x.shape):
        raise ValueError(f"x must have non-zero B, C, H, and W dimensions, got {tuple(x.shape)}")
    if kernel_size == 1:
        return x

    radius = kernel_size // 2
    work_dtype = _stats_dtype(x)
    x_work = x.to(dtype=work_dtype)
    grid = torch.arange(kernel_size, device=x.device, dtype=work_dtype) - radius
    kernel_1d = torch.exp(-(grid.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = x.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, kernel_size).expand(channels, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)

    pad_x_mode = "reflect" if radius < x.shape[-1] else "replicate"
    x_pad = F.pad(x_work, (radius, radius, 0, 0), mode=pad_x_mode)
    x_blur = F.conv2d(x_pad, kernel_x, groups=channels)

    pad_y_mode = "reflect" if radius < x.shape[-2] else "replicate"
    x_pad = F.pad(x_blur, (0, 0, radius, radius), mode=pad_y_mode)
    x_blur = F.conv2d(x_pad, kernel_y, groups=channels)
    return x_blur.to(dtype=x.dtype)


def low_high_split(
    x: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a latent into low-frequency and residual high-frequency parts."""

    low = gaussian_blur_latent(x, kernel_size=kernel_size, sigma=sigma)
    high = x - low
    return low, high


__all__ = [
    "gaussian_blur_latent",
    "low_high_split",
    "standardize_like",
    "standardize_unit",
]
