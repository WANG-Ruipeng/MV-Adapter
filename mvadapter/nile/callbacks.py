"""Denoising-step latent coupling callbacks for NILE-ViewTime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Union

import torch

from .morton import patch_morton_order
from .ops import low_high_split, standardize_like


CallbackMode = Literal["none", "nile_vt", "nile_vtp"]
DeviceLike = Union[str, torch.device]
_VALID_MODES = {"none", "nile_vt", "nile_vtp"}
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1], got {value}")
    return value


def _validate_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"dtype must be a floating-point torch dtype, got {dtype}")
    return dtype


@dataclass
class NILECallbackConfig:
    mode: CallbackMode = "nile_vt"
    num_views: int = 6
    batch_size: int = 1

    rho_start: float = 0.45
    rho_end: float = 0.0
    active_ratio: float = 0.6

    blur_kernel: int = 9
    blur_sigma: float = 2.0

    patch_size: int = 8
    zindex_strength: float = 0.25

    preserve_marginal: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "NILECallbackConfig":
        if not isinstance(self.mode, str) or self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown NILE callback mode {self.mode!r}; expected one of {sorted(_VALID_MODES)}"
            )
        self.num_views = _positive_int(self.num_views, "num_views")
        self.batch_size = _positive_int(self.batch_size, "batch_size")
        self.rho_start = _validate_probability(self.rho_start, "rho_start")
        self.rho_end = _validate_probability(self.rho_end, "rho_end")
        self.active_ratio = _validate_probability(self.active_ratio, "active_ratio")
        if self.active_ratio <= 0.0:
            raise ValueError("active_ratio must be greater than zero")

        if isinstance(self.blur_kernel, bool) or not isinstance(self.blur_kernel, int):
            raise TypeError(
                f"blur_kernel must be an integer, got {type(self.blur_kernel).__name__}"
            )
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError(
                f"blur_kernel must be a positive odd integer, got {self.blur_kernel}"
            )
        self.blur_sigma = float(self.blur_sigma)
        if not math.isfinite(self.blur_sigma) or self.blur_sigma <= 0.0:
            raise ValueError(
                f"blur_sigma must be a finite positive number, got {self.blur_sigma}"
            )

        self.patch_size = _positive_int(self.patch_size, "patch_size")
        self.zindex_strength = _validate_probability(
            self.zindex_strength, "zindex_strength"
        )
        if not isinstance(self.preserve_marginal, bool):
            raise TypeError(
                "preserve_marginal must be bool, got "
                f"{type(self.preserve_marginal).__name__}"
            )
        return self


def linear_rho(step: int, total_steps: int, cfg: NILECallbackConfig) -> float:
    """Linearly decay coupling over the configured active denoising prefix."""

    if not isinstance(cfg, NILECallbackConfig):
        raise TypeError(f"cfg must be NILECallbackConfig, got {type(cfg).__name__}")
    cfg.validate()
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be an integer, got {type(step).__name__}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    total_steps = _positive_int(total_steps, "total_steps")

    active_steps = max(int(total_steps * cfg.active_ratio), 1)
    if step >= active_steps:
        return 0.0
    alpha = step / active_steps
    return cfg.rho_start * (1.0 - alpha) + cfg.rho_end * alpha


def build_patch_rho_map(
    h: int,
    w: int,
    patch_size: int,
    base_rho: float,
    zindex_strength: float,
    device: DeviceLike,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a Morton-ordered patch coupling map of shape ``[1, 1, 1, H, W]``.

    Bottom and right edge patches are retained even when the latent dimensions
    are not divisible by ``patch_size``.
    """

    h = _positive_int(h, "h")
    w = _positive_int(w, "w")
    patch_size = _positive_int(patch_size, "patch_size")
    base_rho = _validate_probability(base_rho, "base_rho")
    zindex_strength = _validate_probability(zindex_strength, "zindex_strength")
    device = torch.device(device)
    dtype = _validate_float_dtype(dtype)

    coords = patch_morton_order(h, w, patch_size, device=device)
    count = coords.shape[0]
    if count <= 1:
        return torch.full((1, 1, 1, h, w), base_rho, device=device, dtype=dtype)

    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    idx = torch.arange(count, device=device, dtype=work_dtype)
    phase = idx / (count - 1)
    modulation = 1.0 + zindex_strength * torch.sin(2.0 * torch.pi * phase)
    patch_values = (base_rho * modulation).clamp(0.0, 1.0)

    patch_h = (h + patch_size - 1) // patch_size
    patch_w = (w + patch_size - 1) // patch_size
    patch_map = torch.empty((patch_h, patch_w), device=device, dtype=work_dtype)
    patch_map[coords[:, 0], coords[:, 1]] = patch_values
    rho_map = patch_map.repeat_interleave(patch_size, dim=0).repeat_interleave(
        patch_size, dim=1
    )
    rho_map = rho_map[:h, :w].to(dtype=dtype)
    return rho_map[None, None, None]


class NILEViewTimeCallback:
    """Training-free low-frequency coupling for MV-Adapter denoising latents.

    The callback accepts and returns the Diffusers ``callback_on_step_end``
    dictionary.  Its expected latent shape is ``[B * V, C, H, W]``.
    """

    def __init__(self, cfg: NILECallbackConfig):
        if not isinstance(cfg, NILECallbackConfig):
            raise TypeError(f"cfg must be NILECallbackConfig, got {type(cfg).__name__}")
        cfg.validate()
        self.cfg = cfg

    def __call__(
        self,
        pipe,
        step: int,
        timestep: int,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self.cfg.validate()
        if self.cfg.mode == "none":
            return callback_kwargs
        if not isinstance(callback_kwargs, dict):
            raise TypeError(
                f"callback_kwargs must be a dict, got {type(callback_kwargs).__name__}"
            )
        if "latents" not in callback_kwargs:
            raise KeyError("callback_kwargs must contain a 'latents' tensor")

        latents = callback_kwargs["latents"]
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"latents must be a torch.Tensor, got {type(latents).__name__}")
        if not latents.is_floating_point():
            raise TypeError(f"latents must have a floating-point dtype, got {latents.dtype}")
        if latents.ndim != 4:
            raise ValueError(
                f"latents must have shape [B * V, C, H, W], got {tuple(latents.shape)}"
            )
        if any(size <= 0 for size in latents.shape):
            raise ValueError(f"latents must have no empty dimensions, got {tuple(latents.shape)}")

        bvc, channels, height, width = latents.shape
        expected_bvc = self.cfg.batch_size * self.cfg.num_views
        if bvc != expected_bvc:
            raise ValueError(
                "latent batch must equal batch_size * num_views; got "
                f"{bvc}, expected {self.cfg.batch_size} * {self.cfg.num_views} = {expected_bvc}"
            )

        total_steps = getattr(pipe, "_num_timesteps", None)
        if total_steps is None:
            total_steps = 50
        if isinstance(total_steps, torch.Tensor):
            if total_steps.numel() != 1:
                raise ValueError("pipe._num_timesteps must be a scalar")
            total_steps = int(total_steps.item())
        elif not isinstance(total_steps, int):
            try:
                total_steps = int(total_steps)
            except (TypeError, ValueError) as error:
                raise TypeError("pipe._num_timesteps must be an integer") from error

        rho = linear_rho(step, total_steps, self.cfg)
        if rho <= 1e-6:
            return callback_kwargs

        ref = latents
        batch = self.cfg.batch_size
        views = self.cfg.num_views
        z = latents.reshape(batch, views, channels, height, width)
        z_flat = z.reshape(batch * views, channels, height, width)
        low, high = low_high_split(
            z_flat,
            kernel_size=self.cfg.blur_kernel,
            sigma=self.cfg.blur_sigma,
        )
        low = low.reshape(batch, views, channels, height, width)
        high = high.reshape(batch, views, channels, height, width)
        parent_low = low.mean(dim=1, keepdim=True)

        if self.cfg.mode == "nile_vtp":
            rho_map = build_patch_rho_map(
                h=height,
                w=width,
                patch_size=self.cfg.patch_size,
                base_rho=rho,
                zindex_strength=self.cfg.zindex_strength,
                device=latents.device,
                dtype=latents.dtype,
            )
            low_new = (1.0 - rho_map) * low + rho_map * parent_low
        else:
            low_new = (1.0 - rho) * low + rho * parent_low

        z_new = (low_new + high).reshape(bvc, channels, height, width)
        if self.cfg.preserve_marginal:
            z_new = standardize_like(z_new, ref)
        callback_kwargs["latents"] = z_new.to(device=latents.device, dtype=latents.dtype)
        return callback_kwargs


__all__ = [
    "CallbackMode",
    "NILECallbackConfig",
    "NILEViewTimeCallback",
    "build_patch_rho_map",
    "linear_rho",
]
