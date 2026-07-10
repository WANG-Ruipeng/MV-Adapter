"""Low-discrepancy sequence backends for NILE."""

from __future__ import annotations

import math
from typing import Union

import torch


DeviceLike = Union[str, torch.device]
_SOBOL_MAX_DIMENSION = 21201
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


def _validate_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _FLOAT_DTYPES:
        raise TypeError(f"dtype must be a floating-point torch dtype, got {dtype}")
    return dtype


def inverse_normal_cdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Transform uniform samples in ``[0, 1]`` into standard Gaussian samples."""

    if not isinstance(u, torch.Tensor):
        raise TypeError(f"u must be a torch.Tensor, got {type(u).__name__}")
    if not u.is_floating_point():
        raise TypeError(f"u must have a floating-point dtype, got {u.dtype}")
    eps = float(eps)
    if not math.isfinite(eps) or not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be finite and lie in (0, 0.5), got {eps}")

    # erfinv is not implemented for every low-precision CPU dtype.  Computing in
    # float32 is both more portable and more accurate near the clipped tails.
    work_dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
    u_work = u.to(dtype=work_dtype).clamp(eps, 1.0 - eps)
    result = math.sqrt(2.0) * torch.erfinv(2.0 * u_work - 1.0)
    return result.to(dtype=u.dtype)


class SobolBackend:
    """Scrambled Sobol prototype backend with a stable ``draw`` interface.

    This is intentionally kept distinct from the future strict SZ backend; flat
    Sobol is an experimental baseline rather than the final NILE/SZ method.
    """

    def __init__(self, dim: int, scramble: bool = True, seed: int = 0):
        dim = _positive_int(dim, "dim")
        if dim > _SOBOL_MAX_DIMENSION:
            raise ValueError(
                f"dim exceeds PyTorch SobolEngine's maximum of {_SOBOL_MAX_DIMENSION}: {dim}"
            )
        if not isinstance(scramble, bool):
            raise TypeError(f"scramble must be bool, got {type(scramble).__name__}")
        seed = _validate_seed(seed)

        self.dim = dim
        self.scramble = scramble
        self.seed = seed
        self.engine = torch.quasirandom.SobolEngine(
            dimension=dim,
            scramble=scramble,
            seed=seed,
        )

    def draw(
        self,
        n: int,
        device: DeviceLike,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        dtype = _validate_float_dtype(dtype)
        device = torch.device(device)
        return self.engine.draw(n).to(device=device, dtype=dtype)

    def reset(self) -> "SobolBackend":
        """Reset the sequence to its first point and return ``self``."""

        self.engine.reset()
        return self


class SZBackend:
    """Placeholder interface for strict NILE/SZ binary generator matrices."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = _positive_int(dim, "dim")
        self.seed = _validate_seed(seed)
        raise NotImplementedError(
            "SZ binary generator matrices are not available yet. "
            "Use SobolBackend for the prototype baseline."
        )


__all__ = ["inverse_normal_cdf", "SobolBackend", "SZBackend"]
