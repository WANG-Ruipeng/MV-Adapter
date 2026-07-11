"""View-covariance builders for distribution-preserving NILE experiments.

The functions in this module only construct small covariance matrices.  They
never touch latent values and never repair a distribution by standardising a
sample.  In particular, every public covariance builder returns a matrix with
unit diagonal unless the caller explicitly asks for numerical ``jitter``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, Sequence[float], Sequence[int]]
DeviceLike = Union[str, torch.device]


# The low-rank equal-KL study uses a little shared variance and keeps half of
# every coefficient view-local.  ``single_tree_covariance`` retains its older
# default below for backwards compatibility; all azimuth-based study builders
# use this constant.
DEFAULT_LOWRANK_TREE_WEIGHTS: Tuple[float, float, float, float] = (
    0.05,
    0.15,
    0.30,
    0.50,
)


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
    length_scale: Optional[float] = None,
    jitter: float = 0.0,
    *,
    ell_deg: Optional[float] = None,
    period: float = 360.0,
    device: Optional[DeviceLike] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the periodic RBF covariance for camera azimuths.

    ``angles`` and ``period`` use the same unit (degrees by default).  The
    kernel is ``exp(-2 sin(delta / 2)^2 / ell^2)`` after converting a full
    period to ``2*pi``.  New low-rank-study callers should pass ``ell_deg``;
    it is converted to the corresponding angular scale in radians.  The old
    ``length_scale`` argument remains supported and retains its historical
    radians-on-the-unit-circle meaning.  Passing both is an error.
    """

    period = float(period)
    jitter = _nonnegative_finite(jitter, "jitter")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    if length_scale is not None and ell_deg is not None:
        raise ValueError("pass either length_scale or ell_deg, not both")
    if ell_deg is not None:
        ell_deg = float(ell_deg)
        if not math.isfinite(ell_deg) or ell_deg <= 0.0:
            raise ValueError("ell_deg must be finite and positive")
        length_scale = ell_deg * (2.0 * math.pi / period)
    elif length_scale is None:
        raise ValueError("one of length_scale or ell_deg must be provided")
    else:
        length_scale = float(length_scale)
        if not math.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive")

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


def azimuths_to_slots(
    azimuths_deg: TensorLike,
    *,
    device: Optional[DeviceLike] = None,
) -> torch.Tensor:
    """Map real camera azimuths to the fixed eight-slot dyadic circle.

    The mapping is exactly ``round((azimuth mod 360) / 45) mod 8``.  It
    preserves input ordering and deliberately does not infer slots from view
    indices.  Two nearby cameras may map to the same slot; the azimuth-based
    tree builders handle that case by selecting the same leaf twice.
    """

    angles = _as_angles(azimuths_deg, device=device, dtype=torch.float64)
    scaled = torch.remainder(angles, 360.0) / 45.0
    return torch.remainder(torch.round(scaled).to(dtype=torch.long), 8)


def tree_covariance_from_azimuths(
    azimuths_deg: TensorLike,
    level_weights: Sequence[float] = DEFAULT_LOWRANK_TREE_WEIGHTS,
    *,
    tree: str = "a",
    tree_a_weight: float = 0.5,
    device: Optional[DeviceLike] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build Tree A, Tree B, or Tree AB covariance in actual view order.

    ``tree='ab'`` is the requested convex combination of the ordinary tree
    and its one-slot cyclic shift.  Building the full eight-slot covariance
    first and then selecting slots makes the topology independent of the
    number and order of requested views.  Repeated mapped slots yield a PSD
    target; identity mixing with ``alpha < 1`` makes it strictly PD.
    """

    slots = azimuths_to_slots(azimuths_deg, device=device)
    full_slots = torch.arange(8, device=slots.device, dtype=torch.long)
    normalized_tree = str(tree).strip().lower().replace("tree_", "")
    if normalized_tree in ("a", "b"):
        full = single_tree_covariance(
            full_slots,
            level_weights,
            tree=normalized_tree,
            device=slots.device,
            dtype=dtype,
        )
    elif normalized_tree == "ab":
        full = staggered_two_tree_covariance(
            full_slots,
            level_weights,
            tree_a_weight=tree_a_weight,
            device=slots.device,
            dtype=dtype,
        )
    else:
        raise ValueError("tree must be 'a', 'b', or 'ab'")
    selected = full.index_select(0, slots).index_select(1, slots)
    return 0.5 * (selected + selected.mT)


def tree_a_covariance(
    azimuths_deg: TensorLike,
    level_weights: Sequence[float] = DEFAULT_LOWRANK_TREE_WEIGHTS,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience wrapper for the formal low-rank Tree A target."""

    return tree_covariance_from_azimuths(
        azimuths_deg, level_weights, tree="a", **kwargs
    )


def tree_b_covariance(
    azimuths_deg: TensorLike,
    level_weights: Sequence[float] = DEFAULT_LOWRANK_TREE_WEIGHTS,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience wrapper for the one-slot-shifted Tree B target."""

    return tree_covariance_from_azimuths(
        azimuths_deg, level_weights, tree="b", **kwargs
    )


def tree_ab_covariance(
    azimuths_deg: TensorLike,
    level_weights: Sequence[float] = DEFAULT_LOWRANK_TREE_WEIGHTS,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience wrapper for ``0.5 * Tree A + 0.5 * Tree B``."""

    return tree_covariance_from_azimuths(
        azimuths_deg, level_weights, tree="ab", **kwargs
    )


def mix_covariance_with_identity(
    target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return ``(1-alpha) I + alpha target`` without altering the diagonal.

    A PSD unit-diagonal target becomes strictly positive definite whenever
    ``alpha < 1``.  At ``alpha == 0`` this function constructs and returns the
    identity directly, avoiding needless arithmetic.
    """

    target = validate_covariance_matrix(target, psd_atol=1e-10)
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    identity = torch.eye(target.shape[0], device=target.device, dtype=target.dtype)
    if alpha == 0.0:
        return identity
    mixed = (1.0 - alpha) * identity + alpha * target
    return 0.5 * (mixed + mixed.mT)


def identity_mix_covariance(target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Alias for :func:`mix_covariance_with_identity`."""

    return mix_covariance_with_identity(target, alpha)


def _positive_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    return rank


def joint_gaussian_kl(covariance: torch.Tensor, rank: int) -> float:
    """Compute the complete joint Gaussian KL from IID in float64.

    For ``rank`` independent low-rank coefficients and ``V`` views this is
    ``rank / 2 * (trace(K) - logdet(K) - V)``.  Cholesky log-determinants are
    used for numerical stability.  Singular targets (for example
    ``shared_full``) have no finite joint KL and raise ``ValueError``.
    """

    rank = _positive_rank(rank)
    covariance = validate_covariance_matrix(
        covariance, require_unit_diagonal=False
    ).to(dtype=torch.float64)
    try:
        factor = torch.linalg.cholesky(covariance)
    except RuntimeError as error:
        raise ValueError(
            "joint KL is finite only for a positive-definite covariance"
        ) from error
    logdet = 2.0 * torch.log(torch.diagonal(factor)).sum()
    value = 0.5 * float(rank) * (
        torch.trace(covariance) - logdet - covariance.shape[0]
    )
    result = float(value.item())
    if result < 0.0 and result >= -1e-10:
        return 0.0
    if result < 0.0:
        raise RuntimeError("computed a negative Gaussian KL")
    return result


def joint_kl_divergence(covariance: torch.Tensor, rank: int) -> float:
    """Alias for :func:`joint_gaussian_kl`."""

    return joint_gaussian_kl(covariance, rank)


def _periodic_distance_deg(first: float, second: float) -> float:
    difference = abs((first - second) % 360.0)
    return min(difference, 360.0 - difference)


def covariance_metadata(
    covariance: torch.Tensor,
    *,
    azimuths_deg: Optional[TensorLike] = None,
    ell_deg: Optional[float] = None,
    topology: Optional[str] = None,
) -> Dict[str, Any]:
    """Return JSON-safe spectral, energy, and camera-relation metadata."""

    covariance = validate_covariance_matrix(
        covariance, require_unit_diagonal=False
    )
    matrix = covariance.to(dtype=torch.float64)
    eigenvalues_tensor = torch.linalg.eigvalsh(matrix)
    eigenvalues = [float(value) for value in eigenvalues_tensor.detach().cpu().tolist()]
    minimum = min(eigenvalues)
    maximum = max(eigenvalues)
    sign, logabsdet = torch.linalg.slogdet(matrix)
    determinant_sign = float(sign.item())
    logdet = float(logabsdet.item()) if determinant_sign > 0.0 else None
    condition_number = maximum / minimum if minimum > 0.0 else None

    nonnegative = eigenvalues_tensor.clamp_min(0.0)
    eigenvalue_sum = float(nonnegative.sum().item())
    if eigenvalue_sum > 0.0:
        probabilities = nonnegative / eigenvalue_sum
        positive = probabilities > 0.0
        entropy = -torch.sum(probabilities[positive] * torch.log(probabilities[positive]))
        effective_rank = float(torch.exp(entropy).item())
    else:
        effective_rank = 0.0

    diagonal = torch.diag(torch.diagonal(matrix))
    off_diagonal = matrix - diagonal
    off_diagonal_energy = float(off_diagonal.square().sum().item())
    total_energy = float(matrix.square().sum().item())
    metadata: Dict[str, Any] = {
        "size": int(matrix.shape[0]),
        "topology": topology,
        "eigenvalues": eigenvalues,
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "determinant_sign": determinant_sign,
        "logdet": logdet,
        "condition_number": condition_number,
        "condition_number_is_infinite": condition_number is None,
        "effective_rank": effective_rank,
        "off_diagonal_energy": off_diagonal_energy,
        "off_diagonal_frobenius_norm": math.sqrt(off_diagonal_energy),
        "off_diagonal_energy_fraction": (
            off_diagonal_energy / total_energy if total_energy > 0.0 else 0.0
        ),
        "diagonal_min": float(torch.diagonal(matrix).min().item()),
        "diagonal_max": float(torch.diagonal(matrix).max().item()),
    }

    if ell_deg is not None:
        ell_deg = float(ell_deg)
        if not math.isfinite(ell_deg) or ell_deg <= 0.0:
            raise ValueError("ell_deg must be finite and positive")
        metadata["ell_deg"] = ell_deg

    if azimuths_deg is not None:
        angles_tensor = _as_angles(
            azimuths_deg, device=matrix.device, dtype=torch.float64
        )
        if angles_tensor.numel() != matrix.shape[0]:
            raise ValueError("azimuths_deg length must match covariance size")
        angles = [float(value) for value in angles_tensor.detach().cpu().tolist()]
        normalized = [value % 360.0 for value in angles]
        metadata["azimuths_deg"] = angles
        metadata["mapped_slots"] = [
            int(value)
            for value in azimuths_to_slots(angles_tensor).detach().cpu().tolist()
        ]

        pair_relations = []
        for first in range(len(angles)):
            for second in range(first + 1, len(angles)):
                distance = _periodic_distance_deg(
                    normalized[first], normalized[second]
                )
                relation: Dict[str, Any] = {
                    "first_index": first,
                    "second_index": second,
                    "periodic_distance_deg": distance,
                    "covariance": float(matrix[first, second].item()),
                }
                if ell_deg is not None:
                    ell_radians = ell_deg * math.pi / 180.0
                    distance_radians = distance * math.pi / 180.0
                    relation["theoretical_correlation"] = math.exp(
                        -2.0
                        * math.sin(0.5 * distance_radians) ** 2
                        / (ell_radians * ell_radians)
                    )
                pair_relations.append(relation)
        pair_relations.sort(
            key=lambda item: (
                item["periodic_distance_deg"],
                item["first_index"],
                item["second_index"],
            )
        )
        metadata["pairwise_relations"] = pair_relations

        ordered = sorted(range(len(normalized)), key=lambda index: (normalized[index], index))
        adjacent_relations = []
        seen_pairs = set()
        if len(ordered) > 1:
            for position, first in enumerate(ordered):
                second = ordered[(position + 1) % len(ordered)]
                pair_key = tuple(sorted((first, second)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                distance = _periodic_distance_deg(
                    normalized[first], normalized[second]
                )
                relation = {
                    "first_index": first,
                    "second_index": second,
                    "first_azimuth_deg": angles[first],
                    "second_azimuth_deg": angles[second],
                    "periodic_distance_deg": distance,
                    "covariance": float(matrix[first, second].item()),
                }
                if ell_deg is not None:
                    ell_radians = ell_deg * math.pi / 180.0
                    distance_radians = distance * math.pi / 180.0
                    relation["theoretical_correlation"] = math.exp(
                        -2.0
                        * math.sin(0.5 * distance_radians) ** 2
                        / (ell_radians * ell_radians)
                    )
                adjacent_relations.append(relation)
        metadata["adjacent_relations"] = adjacent_relations
        # Singular-key alias retained for callers that used the wording in the
        # experiment plan before the schema was finalized.
        metadata["adjacent_relation"] = adjacent_relations

    return metadata


def calibrate_alpha_for_target_kl(
    target: torch.Tensor,
    rank: int,
    target_kl: float,
    *,
    eps: float = 1e-8,
    relative_tolerance: float = 1e-8,
    max_iterations: int = 80,
) -> Dict[str, Any]:
    """Calibrate identity mixing to a requested joint KL budget.

    The returned mapping always contains ``status``, ``target_kl``,
    ``achieved_kl``, ``relative_error``, ``alpha``, and ``covariance``.
    ``status='unattainable'`` leaves the requested KL untouched and reports
    the maximum achievable value at ``alpha=1-eps``.  ``json_metadata`` is a
    tensor-free branch suitable for manifests and resolved configurations.
    """

    rank = _positive_rank(rank)
    target_kl = float(target_kl)
    if not math.isfinite(target_kl) or target_kl < 0.0:
        raise ValueError("target_kl must be finite and non-negative")
    eps = float(eps)
    if not math.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must be finite and lie in (0, 1)")
    relative_tolerance = float(relative_tolerance)
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations != 80
    ):
        raise ValueError("max_iterations must be exactly 80 for reproducibility")

    target = validate_covariance_matrix(target, psd_atol=1e-10).to(
        dtype=torch.float64
    )
    maximum_alpha = 1.0 - eps

    def relative_error(achieved: float) -> float:
        if target_kl == 0.0:
            return abs(achieved)
        return abs(achieved - target_kl) / target_kl

    def finalize(
        status: str,
        alpha: float,
        covariance: torch.Tensor,
        achieved: float,
        iterations: int,
    ) -> Dict[str, Any]:
        metadata = covariance_metadata(covariance, topology="identity_mixed")
        metadata.update(
            {
                "status": status,
                "rank": rank,
                "target_kl": target_kl,
                "achieved_kl": achieved,
                "relative_error": relative_error(achieved),
                "alpha": alpha,
                "alpha_upper_bound": maximum_alpha,
                "iterations": iterations,
            }
        )
        return {
            "status": status,
            "target_kl": target_kl,
            "achieved_kl": achieved,
            "relative_error": metadata["relative_error"],
            "alpha": alpha,
            "covariance": covariance,
            "iterations": iterations,
            "rank": rank,
            "metadata": metadata,
            "json_metadata": dict(metadata),
        }

    identity = torch.eye(target.shape[0], device=target.device, dtype=torch.float64)
    if target_kl == 0.0:
        return finalize("calibrated", 0.0, identity, 0.0, 0)

    upper_covariance = mix_covariance_with_identity(target, maximum_alpha)
    upper_kl = joint_gaussian_kl(upper_covariance, rank)
    if target_kl > upper_kl and relative_error(upper_kl) > relative_tolerance:
        return finalize(
            "unattainable", maximum_alpha, upper_covariance, upper_kl, 0
        )

    lower_alpha = 0.0
    upper_alpha = maximum_alpha
    best_alpha = upper_alpha
    best_covariance = upper_covariance
    best_kl = upper_kl
    best_error = relative_error(upper_kl)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        alpha = 0.5 * (lower_alpha + upper_alpha)
        covariance = mix_covariance_with_identity(target, alpha)
        achieved = joint_gaussian_kl(covariance, rank)
        error = relative_error(achieved)
        if error < best_error:
            best_alpha = alpha
            best_covariance = covariance
            best_kl = achieved
            best_error = error
        if error < relative_tolerance:
            break
        if achieved < target_kl:
            lower_alpha = alpha
        else:
            upper_alpha = alpha

    status = "calibrated" if best_error < 1e-5 else "unattainable"
    return finalize(status, best_alpha, best_covariance, best_kl, iterations)


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
    "DEFAULT_LOWRANK_TREE_WEIGHTS",
    "azimuths_to_slots",
    "calibrate_alpha_for_target_kl",
    "covariance_metadata",
    "identity_mix_covariance",
    "is_positive_semidefinite",
    "joint_gaussian_kl",
    "joint_kl_divergence",
    "minimum_eigenvalue",
    "mix_covariance_with_identity",
    "periodic_camera_rbf_covariance",
    "single_tree_covariance",
    "stable_cholesky",
    "staggered_two_tree_covariance",
    "tree_a_covariance",
    "tree_ab_covariance",
    "tree_b_covariance",
    "tree_covariance_from_azimuths",
    "validate_covariance_matrix",
]
