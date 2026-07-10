"""Distribution-preserving Gaussian coupling in orthogonal Fourier bands.

All samplers first draw a normal IID field in ``[B, V, C, H, W]`` order.  A
zero correlation strength returns that field directly, without an FFT round
trip and without consuming any additional random numbers.  This makes the
zero-strength path exactly equivalent to manually supplied IID float32
latents for a fixed generator state.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch

from .covariance import periodic_camera_rbf_covariance


DeviceLike = Union[str, torch.device]
GeneratorLike = Optional[torch.Generator]
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return value


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("{} must be finite and lie in [0, 1]".format(name))
    return value


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("{} must be finite and positive".format(name))
    return value


def _validate_runtime(
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    device: DeviceLike,
    dtype: torch.dtype,
) -> Tuple[int, int, int, int, int, torch.device, torch.dtype]:
    values = (
        _positive_int(batch_size, "batch_size"),
        _positive_int(num_views, "num_views"),
        _positive_int(channels, "channels"),
        _positive_int(height, "height"),
        _positive_int(width, "width"),
    )
    device = torch.device(device)
    if dtype not in _FLOAT_DTYPES:
        raise TypeError("dtype must be a floating-point torch dtype")
    return values + (device, dtype)


def _resolve_generator(
    device: torch.device,
    *,
    seed: Optional[int],
    generator: GeneratorLike,
) -> torch.Generator:
    if seed is not None and generator is not None:
        raise ValueError("pass either seed or generator, not both")
    if generator is not None:
        generator_device = torch.device(generator.device)
        if generator_device.type != device.type:
            raise ValueError(
                "generator device {} does not match sampling device {}".format(
                    generator_device, device
                )
            )
        if generator_device.type == "cuda":
            generator_index = generator_device.index
            device_index = device.index
            if generator_index is not None and device_index is not None:
                if generator_index != device_index:
                    raise ValueError(
                        "generator CUDA index does not match sampling device index"
                    )
        return generator
    if seed is None:
        seed = 0
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return torch.Generator(device=device).manual_seed(seed)


def radial_frequency_grid(
    height: int,
    width: int,
    *,
    device: DeviceLike,
    dtype: torch.dtype = torch.float32,
    onesided: bool = True,
) -> torch.Tensor:
    """Return normalized radial frequencies in cycles per spatial sample."""

    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    device = torch.device(device)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("frequency-grid dtype must be float32 or float64")
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype)
    if onesided:
        fx = torch.fft.rfftfreq(width, device=device, dtype=dtype)
    else:
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype)
    return torch.sqrt(fy[:, None].square() + fx[None, :].square())


def spectral_correlation_profile(
    height: int,
    width: int,
    *,
    max_correlation: float,
    frequency_scale: float,
    device: DeviceLike,
    dtype: torch.dtype = torch.float32,
    onesided: bool = True,
) -> torch.Tensor:
    """Build ``c(f)=c_max exp(-0.5 (|f|/scale)^2)``."""

    max_correlation = _probability(max_correlation, "max_correlation")
    frequency_scale = _positive_float(frequency_scale, "frequency_scale")
    radius = radial_frequency_grid(
        height, width, device=device, dtype=dtype, onesided=onesided
    )
    return max_correlation * torch.exp(
        -0.5 * (radius / frequency_scale).square()
    )


def _draw_local(
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    # Float32 is intentional: FFTs are numerically stable and this is the
    # canonical external-IID equivalence path used by the inference CLI.
    return torch.randn(
        batch_size * num_views,
        channels,
        height,
        width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, num_views, channels, height, width)


def _flatten_and_cast(latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return latents.reshape(
        latents.shape[0] * latents.shape[1],
        latents.shape[2],
        latents.shape[3],
        latents.shape[4],
    ).to(dtype=dtype)


@torch.no_grad()
def make_spectral_global_correlated_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    *,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    generator: GeneratorLike = None,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
) -> torch.Tensor:
    """Generate white per-view latents with a shared low-frequency component."""

    (
        batch_size,
        num_views,
        channels,
        height,
        width,
        device,
        dtype,
    ) = _validate_runtime(
        batch_size, num_views, channels, height, width, device, dtype
    )
    max_correlation = _probability(max_correlation, "max_correlation")
    frequency_scale = _positive_float(frequency_scale, "frequency_scale")
    generator = _resolve_generator(device, seed=seed, generator=generator)

    local = _draw_local(
        batch_size,
        num_views,
        channels,
        height,
        width,
        generator=generator,
        device=device,
    )
    if max_correlation == 0.0:
        return _flatten_and_cast(local, dtype)

    # Draw local first so the zero-strength path is exactly manual IID.
    shared = torch.randn(
        batch_size,
        1,
        channels,
        height,
        width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    local_fft = torch.fft.rfft2(local, norm="ortho")
    shared_fft = torch.fft.rfft2(shared, norm="ortho")
    correlation = spectral_correlation_profile(
        height,
        width,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        device=device,
        dtype=torch.float32,
        onesided=True,
    )[None, None, None, :, :]
    mixed_fft = correlation.sqrt() * shared_fft + (1.0 - correlation).sqrt() * local_fft
    latents = torch.fft.irfft2(mixed_fft, s=(height, width), norm="ortho")
    return _flatten_and_cast(latents, dtype)


def _validate_view_angles(
    view_angles: Union[torch.Tensor, Sequence[float]],
    num_views: int,
    device: torch.device,
) -> torch.Tensor:
    angles = torch.as_tensor(view_angles, device=device, dtype=torch.float64)
    if angles.ndim != 1 or angles.numel() != num_views:
        raise ValueError("view_angles must contain exactly num_views values")
    if not bool(torch.isfinite(angles).all()):
        raise ValueError("view_angles must contain only finite values")
    return angles


@torch.no_grad()
def make_camera_rbf_correlated_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    view_angles: Union[torch.Tensor, Sequence[float]],
    *,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    generator: GeneratorLike = None,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    length_scale: float = 0.8,
) -> torch.Tensor:
    """Generate white latents with frequency- and camera-dependent covariance.

    At every frequency the view covariance is
    ``(1-lambda(f)) I + lambda(f) K_camera``.  A symmetric square root is used
    instead of a sample-dependent normalization.
    """

    (
        batch_size,
        num_views,
        channels,
        height,
        width,
        device,
        dtype,
    ) = _validate_runtime(
        batch_size, num_views, channels, height, width, device, dtype
    )
    max_correlation = _probability(max_correlation, "max_correlation")
    frequency_scale = _positive_float(frequency_scale, "frequency_scale")
    length_scale = _positive_float(length_scale, "length_scale")
    angles = _validate_view_angles(view_angles, num_views, device)
    generator = _resolve_generator(device, seed=seed, generator=generator)
    local = _draw_local(
        batch_size,
        num_views,
        channels,
        height,
        width,
        generator=generator,
        device=device,
    )
    if max_correlation == 0.0:
        return _flatten_and_cast(local, dtype)

    camera_covariance = periodic_camera_rbf_covariance(
        angles,
        length_scale=length_scale,
        device=device,
        dtype=torch.float64,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(camera_covariance)
    if float(eigenvalues.min().item()) < -1e-8:
        raise ValueError("periodic camera covariance is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0).to(dtype=torch.float32)
    eigenvectors = eigenvectors.to(dtype=torch.float32)

    correlation = spectral_correlation_profile(
        height,
        width,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        device=device,
        dtype=torch.float32,
        onesided=True,
    )
    # Eigenvalues of (1-lambda) I + lambda K share K's eigenvectors.
    mixed_eigenvalues = (
        1.0
        - correlation[None, :, :]
        + correlation[None, :, :] * eigenvalues[:, None, None]
    ).clamp_min(0.0)

    local_fft = torch.fft.rfft2(local, norm="ortho")
    # rfft2 promotes the real IID field to a complex coefficient tensor.
    # PyTorch's einsum requires both operands to have the same scalar dtype;
    # keeping the eigensystem real here therefore raises at runtime.
    complex_eigenvectors = eigenvectors.to(dtype=local_fft.dtype)
    eigen_coefficients = torch.einsum(
        "kv,bvchw->bkchw", complex_eigenvectors.mT, local_fft
    )
    eigen_coefficients = eigen_coefficients * mixed_eigenvalues.sqrt()[
        None, :, None, :, :
    ]
    mixed_fft = torch.einsum(
        "vk,bkchw->bvchw", complex_eigenvectors, eigen_coefficients
    )
    latents = torch.fft.irfft2(mixed_fft, s=(height, width), norm="ortho")
    return _flatten_and_cast(latents, dtype)


def _mean_full_spectral_profile(
    height: int,
    width: int,
    max_correlation: float,
    frequency_scale: float,
    *,
    device: DeviceLike = "cpu",
) -> float:
    profile = spectral_correlation_profile(
        height,
        width,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        device=device,
        dtype=torch.float64,
        onesided=False,
    )
    return float(profile.mean().item())


def global_spatial_covariance_target(
    num_views: int,
    height: int,
    width: int,
    *,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the expected same-pixel view covariance after inverse FFT."""

    num_views = _positive_int(num_views, "num_views")
    mean_correlation = _mean_full_spectral_profile(
        height, width, max_correlation, frequency_scale, device=device
    )
    identity = torch.eye(num_views, device=device, dtype=dtype)
    return (1.0 - mean_correlation) * identity + mean_correlation * torch.ones_like(
        identity
    )


def camera_rbf_spatial_covariance_target(
    view_angles: Union[torch.Tensor, Sequence[float]],
    height: int,
    width: int,
    *,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    length_scale: float = 0.8,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the expected same-pixel camera-RBF covariance in spatial space."""

    angles = torch.as_tensor(view_angles, device=device, dtype=torch.float64)
    if angles.ndim != 1 or angles.numel() == 0:
        raise ValueError("view_angles must be a non-empty one-dimensional sequence")
    mean_correlation = _mean_full_spectral_profile(
        height, width, max_correlation, frequency_scale, device=device
    )
    camera = periodic_camera_rbf_covariance(
        angles, length_scale=length_scale, device=device, dtype=dtype
    )
    identity = torch.eye(angles.numel(), device=device, dtype=dtype)
    return (1.0 - mean_correlation) * identity + mean_correlation * camera


__all__ = [
    "camera_rbf_spatial_covariance_target",
    "global_spatial_covariance_target",
    "make_camera_rbf_correlated_latents",
    "make_spectral_global_correlated_latents",
    "radial_frequency_grid",
    "spectral_correlation_profile",
]
