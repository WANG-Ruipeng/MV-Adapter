"""Morton (Z-order) helpers used by NILE's patch hierarchy."""

from __future__ import annotations

from typing import Union

import torch


DeviceLike = Union[str, torch.device]
_MAX_16BIT = 0xFFFF


def _validate_integer_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")

    value = value.to(dtype=torch.int64)
    if value.numel() > 0:
        min_value = int(value.min().item())
        max_value = int(value.max().item())
        if min_value < 0 or max_value > _MAX_16BIT:
            raise ValueError(
                f"{name} must contain unsigned 16-bit values in [0, {_MAX_16BIT}], "
                f"got range [{min_value}, {max_value}]"
            )
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def part1by1(n: torch.Tensor) -> torch.Tensor:
    """Interleave the bits of unsigned 16-bit integers with zero bits.

    The returned tensor is always ``torch.int64`` and stays on the input device.
    """

    n = _validate_integer_tensor(n, "n")
    n = n & 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return 2D Morton codes, broadcasting ``x`` and ``y`` as needed."""

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must both be torch.Tensor instances")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    try:
        x, y = torch.broadcast_tensors(x, y)
    except RuntimeError as error:
        raise ValueError(
            f"x and y must be broadcastable, got shapes {tuple(x.shape)} and {tuple(y.shape)}"
        ) from error
    return part1by1(x) | (part1by1(y) << 1)


def patch_morton_order(
    h: int,
    w: int,
    patch_size: int,
    device: DeviceLike,
) -> torch.Tensor:
    """Return ``[patch_y, patch_x]`` coordinates sorted by Morton order.

    Partial patches along the bottom and right edges are included.  Consequently
    the number of coordinates is ``ceil(h / patch_size) * ceil(w / patch_size)``.
    """

    h = _positive_int(h, "h")
    w = _positive_int(w, "w")
    patch_size = _positive_int(patch_size, "patch_size")
    device = torch.device(device)

    patch_h = (h + patch_size - 1) // patch_size
    patch_w = (w + patch_size - 1) // patch_size
    if patch_h > _MAX_16BIT + 1 or patch_w > _MAX_16BIT + 1:
        raise ValueError(
            "patch grid dimensions must each fit in 16 bits, got "
            f"{patch_h}x{patch_w}"
        )

    ys, xs = torch.meshgrid(
        torch.arange(patch_h, device=device, dtype=torch.long),
        torch.arange(patch_w, device=device, dtype=torch.long),
        indexing="ij",
    )
    flat_y = ys.reshape(-1)
    flat_x = xs.reshape(-1)
    codes = morton2d(flat_x, flat_y)
    order = torch.argsort(codes)
    coords = torch.stack((flat_y, flat_x), dim=-1)
    return coords[order]


__all__ = ["morton2d", "part1by1", "patch_morton_order"]
