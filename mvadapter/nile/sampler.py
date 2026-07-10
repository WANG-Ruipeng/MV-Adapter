"""Initial latent samplers for NILE and its experimental baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Tuple, Union

import torch

from .ops import gaussian_blur_latent, low_high_split, standardize_unit
from .sequence import SobolBackend, inverse_normal_cdf


SamplerMode = Literal[
    "iid",
    "shared",
    "lowpass_shared",
    "flat_sobol",
    "nile_v",
    "nile_vtp",
]
DeviceLike = Union[str, torch.device]
_VALID_MODES = {
    "iid",
    "shared",
    "lowpass_shared",
    "flat_sobol",
    "nile_v",
    "nile_vtp",
}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1], got {value}")
    return value


def _validate_blur(kernel_size: int, sigma: float) -> Tuple[int, float]:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError(f"blur_kernel must be an integer, got {type(kernel_size).__name__}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"blur_kernel must be a positive odd integer, got {kernel_size}")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"blur_sigma must be a finite positive number, got {sigma}")
    return kernel_size, sigma


def _validate_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"dtype must be a floating-point torch dtype, got {dtype}")
    return dtype


def _validate_shape_and_runtime(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[int, int, int, int, int, torch.device, torch.dtype, int]:
    batch_size = _positive_int(batch_size, "batch_size")
    num_views = _positive_int(num_views, "num_views")
    channels = _positive_int(channels, "channels")
    latent_h = _positive_int(latent_h, "latent_h")
    latent_w = _positive_int(latent_w, "latent_w")
    device = torch.device(device)
    dtype = _validate_dtype(dtype)
    seed = _validate_seed(seed)
    return batch_size, num_views, channels, latent_h, latent_w, device, dtype, seed


@dataclass
class NILEConfig:
    mode: SamplerMode = "nile_v"
    seed: int = 0
    rho_geo: float = 0.65
    blur_kernel: int = 11
    blur_sigma: float = 2.5
    patch_size: int = 8
    qmc_scramble: bool = True
    qmc_dim: int = 4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "NILEConfig":
        if not isinstance(self.mode, str) or self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown NILE sampler mode {self.mode!r}; expected one of {sorted(_VALID_MODES)}"
            )
        self.seed = _validate_seed(self.seed)
        self.rho_geo = _validate_probability(self.rho_geo, "rho_geo")
        self.blur_kernel, self.blur_sigma = _validate_blur(
            self.blur_kernel, self.blur_sigma
        )
        self.patch_size = _positive_int(self.patch_size, "patch_size")
        if not isinstance(self.qmc_scramble, bool):
            raise TypeError(
                f"qmc_scramble must be bool, got {type(self.qmc_scramble).__name__}"
            )
        self.qmc_dim = _positive_int(self.qmc_dim, "qmc_dim")
        if self.qmc_dim > 21201:
            raise ValueError(f"qmc_dim must not exceed SobolEngine's limit of 21201")
        return self


def make_iid_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    (
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        seed,
    ) = _validate_shape_and_runtime(
        batch_size, num_views, channels, latent_h, latent_w, device, dtype, seed
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        batch_size * num_views,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def make_shared_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    (
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        seed,
    ) = _validate_shape_and_runtime(
        batch_size, num_views, channels, latent_h, latent_w, device, dtype, seed
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    parent = torch.randn(
        batch_size,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    shared = parent[:, None].repeat(1, num_views, 1, 1, 1)
    return shared.reshape(batch_size * num_views, channels, latent_h, latent_w)


def make_flat_sobol_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int,
    scramble: bool = True,
) -> torch.Tensor:
    """Generate a flat Sobol baseline without NILE element hierarchy."""

    (
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        seed,
    ) = _validate_shape_and_runtime(
        batch_size, num_views, channels, latent_h, latent_w, device, dtype, seed
    )
    if not isinstance(scramble, bool):
        raise TypeError(f"scramble must be bool, got {type(scramble).__name__}")
    total = batch_size * num_views * channels * latent_h * latent_w
    backend = SobolBackend(dim=1, scramble=scramble, seed=seed)
    uniform = backend.draw(total, device=device, dtype=torch.float32).reshape(-1)
    latents = inverse_normal_cdf(uniform).to(device=device, dtype=dtype)
    latents = latents.reshape(batch_size * num_views, channels, latent_h, latent_w)
    return standardize_unit(latents)


def make_lowpass_shared_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: int,
    rho_geo: float = 0.65,
    blur_kernel: int = 11,
    blur_sigma: float = 2.5,
) -> torch.Tensor:
    """Low-frequency shared parent plus IID high-frequency view children.

    This follows the experiment specification exactly. Consequently,
    rho_geo=0 selects standardized high-pass child noise rather than an IID
    full-spectrum latent; the separate iid mode is the proper IID baseline.
    """

    (
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        seed,
    ) = _validate_shape_and_runtime(
        batch_size, num_views, channels, latent_h, latent_w, device, dtype, seed
    )
    rho_geo = _validate_probability(rho_geo, "rho_geo")
    blur_kernel, blur_sigma = _validate_blur(blur_kernel, blur_sigma)
    generator = torch.Generator(device=device).manual_seed(seed)

    parent = torch.randn(
        batch_size,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    local = torch.randn(
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    parent_low = gaussian_blur_latent(parent, kernel_size=blur_kernel, sigma=blur_sigma)
    parent_low = standardize_unit(parent_low)[:, None]

    local_flat = local.reshape(batch_size * num_views, channels, latent_h, latent_w)
    _, local_high = low_high_split(
        local_flat, kernel_size=blur_kernel, sigma=blur_sigma
    )
    local_high = standardize_unit(local_high).reshape(
        batch_size, num_views, channels, latent_h, latent_w
    )

    local_weight = math.sqrt(max(0.0, 1.0 - rho_geo * rho_geo))
    latents = rho_geo * parent_low + local_weight * local_high
    latents = latents.reshape(batch_size * num_views, channels, latent_h, latent_w)
    return standardize_unit(latents)


def make_nile_v_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    cfg: NILEConfig,
) -> torch.Tensor:
    """NILE-V prototype with a shared low band and Sobol view-local children.

    As in the specified prototype, rho_geo=0 is Sobol high-pass noise, not an
    IID full-spectrum latent. This is deliberately kept distinct from a future
    strict hierarchical NILE/SZ implementation.
    """

    if not isinstance(cfg, NILEConfig):
        raise TypeError(f"cfg must be NILEConfig, got {type(cfg).__name__}")
    cfg.validate()
    (
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        seed,
    ) = _validate_shape_and_runtime(
        batch_size,
        num_views,
        channels,
        latent_h,
        latent_w,
        device,
        dtype,
        cfg.seed,
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    parent = torch.randn(
        batch_size,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    parent_low = gaussian_blur_latent(
        parent, kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma
    )
    parent_low = standardize_unit(parent_low)[:, None]

    total = batch_size * num_views * channels * latent_h * latent_w
    # qmc_dim is reserved for the hierarchical SZ implementation.  The prompt's
    # prototype intentionally uses a one-dimensional stream as a flat child bank.
    backend = SobolBackend(dim=1, scramble=cfg.qmc_scramble, seed=seed + 17)
    uniform = backend.draw(total, device=device, dtype=torch.float32).reshape(-1)
    local = inverse_normal_cdf(uniform).to(device=device, dtype=dtype)
    local = local.reshape(batch_size * num_views, channels, latent_h, latent_w)

    _, local_high = low_high_split(
        local, kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma
    )
    local_high = standardize_unit(local_high).reshape(
        batch_size, num_views, channels, latent_h, latent_w
    )

    local_weight = math.sqrt(max(0.0, 1.0 - cfg.rho_geo * cfg.rho_geo))
    latents = cfg.rho_geo * parent_low + local_weight * local_high
    latents = latents.reshape(batch_size * num_views, channels, latent_h, latent_w)
    return standardize_unit(latents)


def make_initial_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    latent_h: int,
    latent_w: int,
    device: DeviceLike,
    dtype: torch.dtype,
    cfg: NILEConfig,
) -> torch.Tensor:
    """Dispatch to the initial-latent sampler selected by ``cfg.mode``."""

    if not isinstance(cfg, NILEConfig):
        raise TypeError(f"cfg must be NILEConfig, got {type(cfg).__name__}")
    cfg.validate()

    if cfg.mode == "iid":
        return make_iid_latents(
            batch_size, num_views, channels, latent_h, latent_w, device, dtype, cfg.seed
        )
    if cfg.mode == "shared":
        return make_shared_latents(
            batch_size, num_views, channels, latent_h, latent_w, device, dtype, cfg.seed
        )
    if cfg.mode == "flat_sobol":
        return make_flat_sobol_latents(
            batch_size,
            num_views,
            channels,
            latent_h,
            latent_w,
            device,
            dtype,
            cfg.seed,
            scramble=cfg.qmc_scramble,
        )
    if cfg.mode == "lowpass_shared":
        return make_lowpass_shared_latents(
            batch_size=batch_size,
            num_views=num_views,
            channels=channels,
            latent_h=latent_h,
            latent_w=latent_w,
            device=device,
            dtype=dtype,
            seed=cfg.seed,
            rho_geo=cfg.rho_geo,
            blur_kernel=cfg.blur_kernel,
            blur_sigma=cfg.blur_sigma,
        )
    if cfg.mode in ("nile_v", "nile_vtp"):
        return make_nile_v_latents(
            batch_size=batch_size,
            num_views=num_views,
            channels=channels,
            latent_h=latent_h,
            latent_w=latent_w,
            device=device,
            dtype=dtype,
            cfg=cfg,
        )
    # cfg.validate normally makes this unreachable, but retaining the explicit
    # error keeps dispatch robust if a custom config mutates during a call.
    raise ValueError(f"Unknown NILE sampler mode: {cfg.mode}")


__all__ = [
    "NILEConfig",
    "SamplerMode",
    "make_flat_sobol_latents",
    "make_iid_latents",
    "make_initial_latents",
    "make_lowpass_shared_latents",
    "make_nile_v_latents",
    "make_shared_latents",
]
