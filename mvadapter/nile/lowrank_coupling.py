"""Distribution-preserving coupling in an orthonormal latent subspace.

For IID latents ``Z`` and an orthonormal basis ``B``, this module decomposes
``Z`` into basis coefficients and an orthogonal residual.  Only the
coefficients are correlated across views.  No sample-dependent centring,
scaling, clipping, or other post-hoc normalization is performed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch


CouplingMetadata = Dict[str, Any]
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_FACTOR_METHODS = ("auto", "cholesky", "symmetric_eigh")
SUPPORTED_COUPLING_METHODS = (
    "iid_external",
    "shared_full",
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)


def _dtype_name(dtype: torch.dtype) -> str:
    # ``str.removeprefix`` is Python 3.9+, while this package supports 3.8.
    return str(dtype).replace("torch.", "", 1)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_latents(
    iid_latents: torch.Tensor,
    num_views: int,
) -> Tuple[int, int, int, int, int]:
    if not isinstance(iid_latents, torch.Tensor):
        raise TypeError(
            f"iid_latents must be a torch.Tensor, got {type(iid_latents).__name__}"
        )
    if iid_latents.ndim != 4 or iid_latents.numel() == 0:
        raise ValueError(
            "iid_latents must have non-empty shape [B*V,C,H,W], got "
            f"{tuple(iid_latents.shape)}"
        )
    if iid_latents.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"iid_latents must be floating point, got {iid_latents.dtype}")
    if not bool(torch.isfinite(iid_latents).all()):
        raise ValueError("iid_latents contains non-finite values")
    num_views = _positive_int(num_views, "num_views")
    if iid_latents.shape[0] % num_views != 0:
        raise ValueError(
            f"leading latent dimension {iid_latents.shape[0]} is not divisible by "
            f"num_views={num_views}"
        )
    batch_size = iid_latents.shape[0] // num_views
    channels, height, width = iid_latents.shape[1:]
    ambient_dimension = channels * height * width
    return batch_size, channels, height, width, ambient_dimension


def _validate_basis(basis: torch.Tensor, ambient_dimension: int) -> Tuple[int, float]:
    if not isinstance(basis, torch.Tensor):
        raise TypeError(f"basis must be a torch.Tensor, got {type(basis).__name__}")
    if basis.ndim != 2 or basis.shape[0] != ambient_dimension or basis.shape[1] == 0:
        raise ValueError(
            f"basis must have shape [{ambient_dimension},K] with K>0, got "
            f"{tuple(basis.shape)}"
        )
    if not basis.is_floating_point():
        raise TypeError(f"basis must be floating point, got {basis.dtype}")
    if basis.shape[1] > ambient_dimension:
        raise ValueError("basis rank cannot exceed its ambient dimension")
    if not bool(torch.isfinite(basis).all()):
        raise ValueError("basis contains non-finite values")

    basis64 = basis.detach().to(device="cpu", dtype=torch.float64)
    gram = basis64.mT @ basis64
    error = float(
        (gram - torch.eye(basis.shape[1], dtype=torch.float64)).abs().max().item()
    )
    if error >= 1e-6:
        raise ValueError(
            "basis must be orthonormal within max error < 1e-6; "
            f"got {error:.3e}"
        )
    return basis.shape[1], error


def _validate_target_covariance(
    view_covariance: torch.Tensor,
    num_views: int,
) -> Tuple[torch.Tensor, float, float, list]:
    if not isinstance(view_covariance, torch.Tensor):
        raise TypeError(
            "view_covariance must be a torch.Tensor, got "
            f"{type(view_covariance).__name__}"
        )
    if view_covariance.ndim != 2 or tuple(view_covariance.shape) != (
        num_views,
        num_views,
    ):
        raise ValueError(
            f"view_covariance must have shape [{num_views},{num_views}], got "
            f"{tuple(view_covariance.shape)}"
        )
    if not view_covariance.is_floating_point():
        raise TypeError("view_covariance must be floating point")
    if not bool(torch.isfinite(view_covariance).all()):
        raise ValueError("view_covariance contains non-finite values")

    covariance64 = view_covariance.detach().to(device="cpu", dtype=torch.float64)
    symmetry_error = float((covariance64 - covariance64.mT).abs().max().item())
    if symmetry_error > 1e-7:
        raise ValueError(
            "view_covariance must be symmetric; "
            f"max asymmetry={symmetry_error:.3e}"
        )
    diagonal_error = float((covariance64.diagonal() - 1.0).abs().max().item())
    if diagonal_error > 1e-6:
        raise ValueError(
            "view_covariance must have unit diagonal; "
            f"max diagonal error={diagonal_error:.3e}"
        )
    covariance64 = 0.5 * (covariance64 + covariance64.mT)
    eigenvalues = torch.linalg.eigvalsh(covariance64)
    minimum = float(eigenvalues.min().item())
    if minimum < -1e-8:
        raise ValueError(
            "view_covariance must be positive semidefinite; "
            f"minimum eigenvalue={minimum:.3e}"
        )
    return covariance64, symmetry_error, diagonal_error, [
        float(value) for value in eigenvalues.tolist()
    ]


def _validate_alpha(alpha: Optional[float]) -> Optional[float]:
    if alpha is None:
        return None
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0,1] or be None, got {alpha}")
    return alpha


def _covariance_statistics(covariance64: torch.Tensor, rank: int) -> Dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(covariance64)
    minimum = float(eigenvalues.min().item())
    maximum = float(eigenvalues.max().item())
    positive = bool((eigenvalues > 0.0).all())
    if positive:
        logdet = float(torch.log(eigenvalues).sum().item())
        condition_number: Optional[float] = maximum / minimum
        joint_kl: Optional[float] = 0.5 * rank * (
            float(torch.trace(covariance64).item()) - logdet - covariance64.shape[0]
        )
    else:
        logdet = None
        condition_number = None
        joint_kl = None
    return {
        "covariance_eigenvalues": [float(value) for value in eigenvalues.tolist()],
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "condition_number": condition_number,
        "logdet": logdet,
        "joint_kl_nats": joint_kl,
        "joint_kl_finite": positive,
    }


def _factor_covariance(
    covariance: torch.Tensor,
    factor_method: str,
) -> Tuple[torch.Tensor, str]:
    if factor_method not in _FACTOR_METHODS:
        raise ValueError(
            f"factor_method must be one of {_FACTOR_METHODS}, got {factor_method!r}"
        )

    if factor_method in ("auto", "cholesky"):
        factor, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
        if int(info.max().item()) == 0:
            return factor, "cholesky"
        if factor_method == "cholesky":
            raise ValueError("effective view covariance is not positive definite")

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    tolerance = 1e-6 if covariance.dtype == torch.float32 else 1e-10
    if float(eigenvalues.min().item()) < -tolerance:
        raise ValueError("effective view covariance is not positive semidefinite")
    eigenvalues = eigenvalues.clamp_min(0.0)
    factor = (eigenvectors * eigenvalues.sqrt()[None, :]) @ eigenvectors.mT
    return factor, "symmetric_eigh"


def _base_metadata(
    iid_latents: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    rank: Optional[int],
    alpha: Optional[float],
    computation_dtype: Optional[torch.dtype],
    covariance_factor_method: str,
    identity_passthrough: bool,
) -> CouplingMetadata:
    return {
        "batch_size": batch_size,
        "num_views": num_views,
        "channels": channels,
        "height": height,
        "width": width,
        "ambient_dimension": channels * height * width,
        "rank": rank,
        "alpha": alpha,
        "input_dtype": _dtype_name(iid_latents.dtype),
        "output_dtype": _dtype_name(iid_latents.dtype),
        "computation_dtype": (
            _dtype_name(computation_dtype) if computation_dtype is not None else None
        ),
        "device": str(iid_latents.device),
        "covariance_factor_method": covariance_factor_method,
        "identity_passthrough": identity_passthrough,
        "per_sample_standardization": False,
    }


@torch.no_grad()
def correlate_orthonormal_subspace(
    iid_latents: torch.Tensor,
    basis: torch.Tensor,
    view_covariance: torch.Tensor,
    num_views: int,
    *,
    alpha: Optional[float] = None,
    factor_method: str = "auto",
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CouplingMetadata]]:
    """Correlate basis coefficients across views while preserving marginals.

    ``iid_latents`` has shape ``[B*V,C,H,W]`` and ``basis`` has shape
    ``[C*H*W,K]``.  When ``alpha`` is a float, ``view_covariance`` is treated
    as a target and the effective covariance is
    ``(1-alpha) * I + alpha * view_covariance``.  When ``alpha`` is ``None``,
    the supplied covariance is already effective and is used as-is.

    ``alpha == 0`` and an exactly identity effective covariance return the
    original tensor object without projection/reconstruction.  This makes the
    external-IID control bit-exact and preserves its ``data_ptr``.
    """

    batch_size, channels, height, width, ambient_dimension = _validate_latents(
        iid_latents, num_views
    )
    num_views = int(num_views)
    rank, basis_error = _validate_basis(basis, ambient_dimension)
    target64, symmetry_error, diagonal_error, target_eigenvalues = (
        _validate_target_covariance(view_covariance, num_views)
    )
    alpha = _validate_alpha(alpha)
    if factor_method not in _FACTOR_METHODS:
        raise ValueError(
            f"factor_method must be one of {_FACTOR_METHODS}, got {factor_method!r}"
        )

    identity64 = torch.eye(num_views, dtype=torch.float64)
    effective64 = target64 if alpha is None else (1.0 - alpha) * identity64 + alpha * target64
    is_identity = bool(torch.equal(effective64, identity64))
    covariance_stats = _covariance_statistics(effective64, rank)

    if is_identity:
        metadata = _base_metadata(
            iid_latents,
            batch_size=batch_size,
            num_views=num_views,
            channels=channels,
            height=height,
            width=width,
            rank=rank,
            alpha=alpha,
            computation_dtype=None,
            covariance_factor_method="none_identity_passthrough",
            identity_passthrough=True,
        )
        metadata.update(
            {
                "method": "lowrank_orthonormal_subspace",
                "basis_orthonormality_error": basis_error,
                "target_covariance_symmetry_error": symmetry_error,
                "target_covariance_diagonal_error": diagonal_error,
                "target_covariance_eigenvalues": target_eigenvalues,
                "effective_view_covariance": effective64.tolist(),
                **covariance_stats,
            }
        )
        if return_metadata:
            return iid_latents, metadata
        return iid_latents

    work_dtype = torch.float64 if iid_latents.dtype == torch.float64 else torch.float32
    work_basis = basis.to(device=iid_latents.device, dtype=work_dtype)
    effective = effective64.to(device=iid_latents.device, dtype=work_dtype)
    factor, actual_factor_method = _factor_covariance(effective, factor_method)

    flat = iid_latents.reshape(batch_size, num_views, ambient_dimension).to(
        dtype=work_dtype
    )
    coefficients = flat @ work_basis
    residual = flat - coefficients @ work_basis.mT
    correlated_coefficients = torch.einsum("vw,bwk->bvk", factor, coefficients)
    output_flat = residual + correlated_coefficients @ work_basis.mT
    output = output_flat.reshape_as(iid_latents).to(dtype=iid_latents.dtype)

    metadata = _base_metadata(
        iid_latents,
        batch_size=batch_size,
        num_views=num_views,
        channels=channels,
        height=height,
        width=width,
        rank=rank,
        alpha=alpha,
        computation_dtype=work_dtype,
        covariance_factor_method=actual_factor_method,
        identity_passthrough=False,
    )
    metadata.update(
        {
            "method": "lowrank_orthonormal_subspace",
            "basis_orthonormality_error": basis_error,
            "target_covariance_symmetry_error": symmetry_error,
            "target_covariance_diagonal_error": diagonal_error,
            "target_covariance_eigenvalues": target_eigenvalues,
            "effective_view_covariance": effective64.tolist(),
            **covariance_stats,
        }
    )
    if return_metadata:
        return output, metadata
    return output


@torch.no_grad()
def make_shared_full_latents(
    iid_latents: torch.Tensor,
    num_views: int,
    *,
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CouplingMetadata]]:
    """Use one complete IID field for every view within each batch item.

    This is a deliberately degenerate diagnostic upper bound, not a proposed
    geometry method.  Its joint covariance is singular and has no finite
    Gaussian KL relative to the IID joint distribution.
    """

    batch_size, channels, height, width, _ = _validate_latents(iid_latents, num_views)
    num_views = int(num_views)
    views = iid_latents.reshape(batch_size, num_views, channels, height, width)
    output = views[:, :1].expand(-1, num_views, -1, -1, -1).reshape_as(iid_latents).clone()
    metadata = _base_metadata(
        iid_latents,
        batch_size=batch_size,
        num_views=num_views,
        channels=channels,
        height=height,
        width=width,
        rank=None,
        alpha=None,
        computation_dtype=None,
        covariance_factor_method="none_shared_full",
        identity_passthrough=False,
    )
    metadata.update(
        {
            "method": "shared_full",
            "degenerate_joint_distribution": True,
            "joint_kl_nats": None,
            "joint_kl_finite": False,
            "interpretation": "diagnostic_upper_bound_not_3d_consistency",
        }
    )
    if return_metadata:
        return output, metadata
    return output


@torch.no_grad()
def apply_latent_coupling(
    iid_latents: torch.Tensor,
    method: str,
    num_views: int,
    *,
    basis: Optional[torch.Tensor] = None,
    view_covariance: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    factor_method: str = "auto",
    return_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CouplingMetadata]]:
    """Dispatch the formal IID, shared-full, and low-rank method names."""

    if method not in SUPPORTED_COUPLING_METHODS:
        raise ValueError(
            f"method must be one of {SUPPORTED_COUPLING_METHODS}, got {method!r}"
        )
    if method == "iid_external":
        batch_size, channels, height, width, _ = _validate_latents(
            iid_latents, num_views
        )
        metadata = _base_metadata(
            iid_latents,
            batch_size=batch_size,
            num_views=int(num_views),
            channels=channels,
            height=height,
            width=width,
            rank=None,
            alpha=0.0,
            computation_dtype=None,
            covariance_factor_method="none_identity_passthrough",
            identity_passthrough=True,
        )
        metadata["method"] = "iid_external"
        if return_metadata:
            return iid_latents, metadata
        return iid_latents
    if method == "shared_full":
        return make_shared_full_latents(
            iid_latents, num_views, return_metadata=return_metadata
        )
    if basis is None or view_covariance is None:
        raise ValueError(f"{method} requires basis and view_covariance")
    result = correlate_orthonormal_subspace(
        iid_latents,
        basis,
        view_covariance,
        num_views,
        alpha=alpha,
        factor_method=factor_method,
        return_metadata=return_metadata,
    )
    if return_metadata:
        output, metadata = result
        metadata["method"] = method
        return output, metadata
    return result


__all__ = [
    "CouplingMetadata",
    "SUPPORTED_COUPLING_METHODS",
    "apply_latent_coupling",
    "correlate_orthonormal_subspace",
    "make_shared_full_latents",
]
