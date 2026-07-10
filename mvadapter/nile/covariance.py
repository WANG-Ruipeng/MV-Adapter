"""View-covariance builders for distribution-preserving NILE experiments.

The functions in this module only construct small covariance matrices.  They
never touch latent values and never repair a distribution by standardising a
sample.  In particular, every public covariance builder returns a matrix with
unit diagonal unless the caller explicitly asks for numerical ``jitter``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float], Sequence[int]]
DeviceLike = Union[str, torch.device]


def _floating_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("covariance dtype must be torch.float32 or torch.float64")
    return dtype


def _as_angles(
    angles: TensorLike,
    *,
    device: Optional[DeviceLike],
    dtype: torch.dtype,
) -> torch.Tensor:
    dtype = _floating_dtype(dtype)
    tensor = torch.as_tensor(angles, device=device, dtype=dtype)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("angles must be a non-empty one-dimensional sequence")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("angles must contain only finite values")
    return tensor


def _as_slots(
    view_slots: Union[torch.Tensor, Sequence[int]],
    *,
    device: Optional[DeviceLike],
) -> torch.Tensor:
    slots = torch.as_tensor(view_slots, device=device)
    if slots.ndim != 1 or slots.numel() == 0:
        raise ValueError("view_slots must be a non-empty one-dimensional sequence")
    if slots.dtype == torch.bool or slots.is_floating_point() or slots.is_complex():
        raise TypeError("view_slots must contain integers")
    slots = slots.to(dtype=torch.long)
    if bool(((slots < 0) | (slots > 7)).any()):
        raise ValueError("view_slots must lie in the eight-slot range [0, 7]")
    if torch.unique(slots).numel() != slots.numel():
        raise ValueError("view_slots must be unique")
    return slots


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return value


def periodic_camera_rbf_covariance(
    angles: TensorLike,
    length_scale: float,
    jitter: float = 0.0,
    *,
    period: float = 360.0,
    device: Optional[DeviceLike] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the periodic RBF covariance for camera azimuths.

    ``angles`` and ``period`` use the same unit (degrees by default).  The
    kernel is ``exp(-2 sin(delta / 2)^2 / length_scale^2)`` after converting a
    full period to ``2*pi``.  ``length_scale`` is therefore dimensionless and
    expressed in radians on the unit circle.
    """

    length_scale = float(length_scale)
    period = float(period)
    jitter = _nonnegative_finite(jitter, "jitter")
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")

    theta = _as_angles(angles, device=device, dtype=dtype)
    delta = (theta[:, None] - theta[None, :]) * (2.0 * math.pi / period)
    covariance = torch.exp(
        -2.0 * torch.sin(0.5 * delta).square() / (length_scale * length_scale)
    )
    if jitter:
        covariance = covariance + jitter * torch.eye(
            theta.numel(), device=theta.device, dtype=theta.dtype
        )
    return covariance


def _validate_level_weights(
    level_weights: Sequence[float],
) -> Tuple[float, float, float, float]:
    if len(level_weights) != 4:
        raise ValueError(
            "level_weights must contain root, coarse, pair, and leaf weights"
        )
    weights = tuple(_nonnegative_finite(value, "level weight") for value in level_weights)
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("level_weights must sum to one")
    return weights  # type: ignore[return-value]


def _tree_group_ids(slots: torch.Tensor, tree: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if tree == "a":
        shifted = slots
    elif tree == "b":
        # This produces pairs [7,0], [1,2], [3,4], [5,6] and coarse
        # intervals [7,0,1,2], [3,4,5,6].
        shifted = torch.remainder(slots + 1, 8)
    else:
        raise ValueError("tree must be 'a' or 'b'")
    return torch.div(shifted, 4, rounding_mode="floor"), torch.div(
        shifted, 2, rounding_mode="floor"
    )


def single_tree_covariance(
    view_slots: Union[torch.Tensor, Sequence[int]],
    level_weights: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
    *,
    tree: str = "a",
    device: Optional[DeviceLike] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a unit-diagonal covariance for one eight-slot dyadic tree."""

    slots = _as_slots(view_slots, device=device)
    root_weight, coarse_weight, pair_weight, leaf_weight = _validate_level_weights(
        level_weights
    )
    dtype = _floating_dtype(dtype)
    coarse_ids, pair_ids = _tree_group_ids(slots, tree)
    same_coarse = coarse_ids[:, None].eq(coarse_ids[None, :]).to(dtype=dtype)
    same_pair = pair_ids[:, None].eq(pair_ids[None, :]).to(dtype=dtype)
    identity = torch.eye(slots.numel(), device=slots.device, dtype=dtype)
    covariance = (
        root_weight * torch.ones_like(identity)
        + coarse_weight * same_coarse
        + pair_weight * same_pair
        + leaf_weight * identity
    )
    return 0.5 * (covariance + covariance.mT)


def staggered_two_tree_covariance(
    view_slots: Union[torch.Tensor, Sequence[int]],
    level_weights: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
    *,
    tree_a_weight: float = 0.5,
    device: Optional[DeviceLike] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a convex combination of the normal and one-slot-shifted trees."""

    tree_a_weight = float(tree_a_weight)
    if not math.isfinite(tree_a_weight) or not 0.0 <= tree_a_weight <= 1.0:
        raise ValueError("tree_a_weight must be finite and lie in [0, 1]")
    tree_a = single_tree_covariance(
        view_slots, level_weights, tree="a", device=device, dtype=dtype
    )
    tree_b = single_tree_covariance(
        view_slots, level_weights, tree="b", device=device, dtype=dtype
    )
    covariance = tree_a_weight * tree_a + (1.0 - tree_a_weight) * tree_b
    return 0.5 * (covariance + covariance.mT)


def minimum_eigenvalue(matrix: torch.Tensor) -> float:
    """Return the smallest eigenvalue of a finite real symmetric matrix."""

    matrix = validate_covariance_matrix(matrix, require_unit_diagonal=False)
    eigenvalues = torch.linalg.eigvalsh(matrix.to(dtype=torch.float64))
    return float(eigenvalues.min().item())


def is_positive_semidefinite(matrix: torch.Tensor, atol: float = 1e-8) -> bool:
    """Check positive semidefiniteness with an absolute eigenvalue tolerance."""

    atol = _nonnegative_finite(atol, "atol")
    try:
        return minimum_eigenvalue(matrix) >= -atol
    except (TypeError, ValueError, RuntimeError):
        return False


def validate_covariance_matrix(
    matrix: torch.Tensor,
    *,
    symmetry_atol: float = 1e-7,
    require_unit_diagonal: bool = True,
    psd_atol: Optional[float] = None,
) -> torch.Tensor:
    """Validate shape, finiteness, symmetry, diagonal, and optionally PSD."""

    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.numel() == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not matrix.is_floating_point():
        raise TypeError("matrix must have a floating-point dtype")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")
    symmetry_atol = _nonnegative_finite(symmetry_atol, "symmetry_atol")
    if not torch.allclose(matrix, matrix.mT, rtol=0.0, atol=symmetry_atol):
        raise ValueError("matrix must be symmetric")
    if require_unit_diagonal:
        ones = torch.ones(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        if not torch.allclose(torch.diagonal(matrix), ones, rtol=0.0, atol=symmetry_atol):
            raise ValueError("covariance matrix must have unit diagonal")
    if psd_atol is not None:
        psd_atol = _nonnegative_finite(psd_atol, "psd_atol")
        eigenvalues = torch.linalg.eigvalsh(matrix.to(dtype=torch.float64))
        if float(eigenvalues.min().item()) < -psd_atol:
            raise ValueError("matrix is not positive semidefinite")
    return matrix


def stable_cholesky(
    matrix: torch.Tensor,
    *,
    jitter: float = 1e-8,
    max_tries: int = 6,
) -> torch.Tensor:
    """Compute a Cholesky factor, increasing diagonal jitter when necessary.

    The input is not modified.  The returned factor corresponds to the first
    successful ``matrix + used_jitter * I`` attempt.  Positive-definite inputs
    are tried without jitter first.
    """

    matrix = validate_covariance_matrix(matrix, require_unit_diagonal=False)
    jitter = _nonnegative_finite(jitter, "jitter")
    if isinstance(max_tries, bool) or not isinstance(max_tries, int) or max_tries < 1:
        raise ValueError("max_tries must be a positive integer")
    identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    last_error = None
    for attempt in range(max_tries):
        added = 0.0 if attempt == 0 else jitter * (10.0 ** (attempt - 1))
        try:
            return torch.linalg.cholesky(matrix + added * identity)
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        "Cholesky factorization failed after {} attempts".format(max_tries)
    ) from last_error


__all__ = [
    "is_positive_semidefinite",
    "minimum_eigenvalue",
    "periodic_camera_rbf_covariance",
    "single_tree_covariance",
    "stable_cholesky",
    "staggered_two_tree_covariance",
    "validate_covariance_matrix",
]
