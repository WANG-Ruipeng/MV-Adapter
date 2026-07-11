"""Deterministic orthonormal low-frequency bases for NILE experiments.

The public builder constructs real two-dimensional DCT-II modes in float64,
validates them before casting, and places every spatial mode in exactly one
latent channel.  Columns are ordered by spatial frequency and distributed
round-robin across channels.  The pure spatial DC mode is excluded by default
so that the coupled subspace does not begin with global colour/background
offsets.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Tuple, Union

import torch


DeviceLike = Union[str, torch.device]
BasisMetadata = Dict[str, Any]
_OUTPUT_DTYPES = (torch.float32, torch.float64)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_output_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in _OUTPUT_DTYPES:
        raise TypeError(
            "DCT basis dtype must be torch.float32 or torch.float64; "
            f"got {dtype}"
        )
    return dtype


def _dct_ii_vectors(length: int) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix ``[frequency, position]``."""

    positions = torch.arange(length, dtype=torch.float64)
    frequencies = torch.arange(length, dtype=torch.float64)[:, None]
    vectors = torch.cos(math.pi * (positions[None, :] + 0.5) * frequencies / length)
    scales = torch.full((length,), math.sqrt(2.0 / length), dtype=torch.float64)
    scales[0] = math.sqrt(1.0 / length)
    return vectors * scales[:, None]


def ordered_dct2_modes(
    height: int,
    width: int,
    *,
    exclude_dc: bool = True,
) -> List[Tuple[int, int]]:
    """List spatial DCT-II modes in deterministic low-to-high order.

    Ties in ``u**2 + v**2`` are resolved by ``u`` and then ``v``.  This
    explicit tie-break makes the basis independent of set/dict ordering and
    Python implementation details.
    """

    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    if not isinstance(exclude_dc, bool):
        raise TypeError("exclude_dc must be a bool")

    modes = [
        (u, v)
        for u in range(height)
        for v in range(width)
        if not (exclude_dc and u == 0 and v == 0)
    ]
    modes.sort(key=lambda mode: (mode[0] * mode[0] + mode[1] * mode[1], mode[0], mode[1]))
    return modes


def basis_checksum(basis: torch.Tensor) -> str:
    """Return a SHA-256 checksum of a basis' shape, dtype, and tensor bytes."""

    if not isinstance(basis, torch.Tensor):
        raise TypeError(f"basis must be a torch.Tensor, got {type(basis).__name__}")
    if basis.ndim != 2 or basis.numel() == 0:
        raise ValueError(f"basis must be a non-empty matrix, got {tuple(basis.shape)}")
    if basis.dtype not in _OUTPUT_DTYPES:
        raise TypeError("basis checksum supports torch.float32 and torch.float64")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("basis contains non-finite values")

    canonical = basis.detach().to(device="cpu").contiguous()
    header = (
        f"nile-dct2-basis-v1|{canonical.shape[0]}|{canonical.shape[1]}|"
        f"{str(canonical.dtype)}|"
    ).encode("ascii")
    # ``tolist`` avoids a NumPy dependency while retaining the exact bytes of
    # the returned tensor (including float32 rounding).
    raw = bytes(canonical.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + raw).hexdigest()


def build_dct2_basis(
    channels: int,
    height: int,
    width: int,
    rank: int,
    *,
    exclude_dc: bool = True,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.float32,
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, BasisMetadata]]:
    """Build a channel-balanced orthonormal 2D DCT-II basis.

    Args:
        channels: Number of latent channels.
        height: Latent spatial height.
        width: Latent spatial width.
        rank: Number of basis columns.
        exclude_dc: Exclude spatial mode ``(0, 0)`` in every channel.
        device: Device of the returned basis.
        dtype: ``torch.float32`` (default) or ``torch.float64``.
        return_metadata: Return ``(basis, metadata)`` when true.

    The construction and validation always happen in float64 on CPU.  The
    requested cast/device transfer occurs only after the strict orthonormality
    checks pass.
    """

    channels = _positive_int(channels, "channels")
    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    rank = _positive_int(rank, "rank")
    if not isinstance(exclude_dc, bool):
        raise TypeError("exclude_dc must be a bool")
    dtype = _validate_output_dtype(dtype)
    device = torch.device(device)

    spatial_modes = ordered_dct2_modes(height, width, exclude_dc=exclude_dc)
    maximum_rank = channels * len(spatial_modes)
    if rank > maximum_rank:
        raise ValueError(
            f"rank {rank} exceeds the available rank {maximum_rank} for "
            f"channels={channels}, height={height}, width={width}, "
            f"exclude_dc={exclude_dc}"
        )

    dct_y = _dct_ii_vectors(height)
    dct_x = _dct_ii_vectors(width)
    ambient_dimension = channels * height * width
    basis64 = torch.zeros((ambient_dimension, rank), dtype=torch.float64)
    column_records: List[Dict[str, Union[int, float]]] = []

    # Visit all channels for one spatial mode before advancing in frequency.
    # Thus column i uses channel i % channels, including when rank truncates a
    # mode part-way through its channel cycle.
    column = 0
    for u, v in spatial_modes:
        spatial_vector = torch.outer(dct_y[u], dct_x[v]).reshape(-1)
        frequency_squared = u * u + v * v
        for channel in range(channels):
            if column == rank:
                break
            start = channel * height * width
            basis64[start : start + height * width, column] = spatial_vector
            column_records.append(
                {
                    "index": column,
                    "channel": channel,
                    "u": u,
                    "v": v,
                    "spatial_frequency_squared": frequency_squared,
                }
            )
            column += 1
        if column == rank:
            break

    gram64 = basis64.mT @ basis64
    identity64 = torch.eye(rank, dtype=torch.float64)
    orthonormality_error = float((gram64 - identity64).abs().max().item())
    column_norm_max_error = float(
        (torch.linalg.vector_norm(basis64, dim=0) - 1.0).abs().max().item()
    )
    if orthonormality_error >= 1e-6:
        raise RuntimeError(
            "constructed DCT-II basis failed orthonormality validation: "
            f"max error={orthonormality_error:.3e}"
        )
    if column_norm_max_error >= 1e-6:
        raise RuntimeError(
            "constructed DCT-II basis failed column norm validation: "
            f"max error={column_norm_max_error:.3e}"
        )

    basis = basis64.to(device=device, dtype=dtype)
    # Also validate the required float32 deliverable after its cast.  DCT-II
    # roundoff is comfortably below the same 1e-6 contract.
    output_gram = basis.detach().to(device="cpu", dtype=torch.float64).mT @ basis.detach().to(
        device="cpu", dtype=torch.float64
    )
    output_orthonormality_error = float((output_gram - identity64).abs().max().item())
    if output_orthonormality_error >= 1e-6:
        raise RuntimeError(
            "cast DCT-II basis failed orthonormality validation: "
            f"max error={output_orthonormality_error:.3e}"
        )

    metadata: BasisMetadata = {
        "basis_type": "real_2d_dct_ii_orthonormal",
        "channels": channels,
        "height": height,
        "width": width,
        "ambient_dimension": ambient_dimension,
        "rank": rank,
        "maximum_rank": maximum_rank,
        "exclude_dc": exclude_dc,
        "columns": column_records,
        "basis_checksum": basis_checksum(basis),
        "basis_checksum_algorithm": "sha256",
        "orthonormality_error": orthonormality_error,
        "output_orthonormality_error": output_orthonormality_error,
        "column_norm_max_error": column_norm_max_error,
        "construction_dtype": "float64",
        "output_dtype": str(dtype).replace("torch.", "", 1),
        "device": str(device),
    }
    if return_metadata:
        return basis, metadata
    return basis


# A concise alias for callers that prefer a noun-like factory name.
make_dct2_basis = build_dct2_basis


__all__ = [
    "BasisMetadata",
    "basis_checksum",
    "build_dct2_basis",
    "make_dct2_basis",
    "ordered_dct2_modes",
]
