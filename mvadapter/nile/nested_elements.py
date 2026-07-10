"""Distribution-preserving nested Gaussian view-element bank.

The eight dyadic slots are fixed at 45-degree increments.  Tree A starts at
slot zero; Tree B is shifted by one slot so that the circular seam ``315 -> 0``
receives the same treatment as every other neighbouring pair.  Morton or
Sobol values are deliberately not used as latent coefficients here.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from .covariance import single_tree_covariance, staggered_two_tree_covariance
from .spectral_gaussian import (
    DeviceLike,
    GeneratorLike,
    _draw_local,
    _flatten_and_cast,
    _positive_float,
    _probability,
    _resolve_generator,
    _validate_runtime,
    radial_frequency_grid,
)


AnglesLike = Union[torch.Tensor, Sequence[float]]
SlotsLike = Union[torch.Tensor, Sequence[int]]
_DEFAULT_LOW_WEIGHTS = (0.10, 0.20, 0.30, 0.40)


def angles_to_dyadic_slots(
    view_angles: AnglesLike,
    *,
    period: float = 360.0,
    num_slots: int = 8,
    tolerance: float = 1e-4,
    device: Optional[DeviceLike] = None,
) -> torch.Tensor:
    """Map camera angles to unique nearest slots on an eight-way circle."""

    period = float(period)
    tolerance = float(tolerance)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    if isinstance(num_slots, bool) or not isinstance(num_slots, int) or num_slots <= 0:
        raise ValueError("num_slots must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    angles = torch.as_tensor(view_angles, device=device, dtype=torch.float64)
    if angles.ndim != 1 or angles.numel() == 0:
        raise ValueError("view_angles must be a non-empty one-dimensional sequence")
    if not bool(torch.isfinite(angles).all()):
        raise ValueError("view_angles must contain only finite values")

    slot_width = period / float(num_slots)
    normalized = torch.remainder(angles, period)
    nearest_unwrapped = torch.round(normalized / slot_width)
    slots = torch.remainder(nearest_unwrapped.to(dtype=torch.long), num_slots)
    snapped = slots.to(dtype=torch.float64) * slot_width
    circular_error = torch.remainder(normalized - snapped + 0.5 * period, period) - 0.5 * period
    if bool((circular_error.abs() > tolerance).any()):
        raise ValueError(
            "every view angle must lie on a {}-degree dyadic slot".format(slot_width)
        )
    if torch.unique(slots).numel() != slots.numel():
        raise ValueError("view angles must map to unique dyadic slots")
    return slots


def tree_ancestor_ids(view_slots: SlotsLike, tree: str = "a") -> Dict[str, torch.Tensor]:
    """Return root, coarse, pair, and leaf element IDs for selected slots."""

    slots = torch.as_tensor(view_slots)
    if slots.ndim != 1 or slots.numel() == 0:
        raise ValueError("view_slots must be a non-empty one-dimensional sequence")
    if slots.dtype == torch.bool or slots.is_floating_point() or slots.is_complex():
        raise TypeError("view_slots must contain integers")
    slots = slots.to(dtype=torch.long)
    if bool(((slots < 0) | (slots > 7)).any()):
        raise ValueError("view_slots must lie in [0, 7]")
    if tree == "a":
        shifted = slots
    elif tree == "b":
        shifted = torch.remainder(slots + 1, 8)
    else:
        raise ValueError("tree must be 'a' or 'b'")
    return {
        "root": torch.zeros_like(slots),
        "coarse": torch.div(shifted, 4, rounding_mode="floor"),
        "pair": torch.div(shifted, 2, rounding_mode="floor"),
        "leaf": slots.clone(),
    }


def element_seed_key(base_seed: int, tree: str, level: str, group_id: int) -> int:
    """Create a stable 63-bit seed key for a named Gaussian element.

    The main sampler uses these keys for every named tree element.  Its caller
    derives ``base_seed`` from the generator state *before* the canonical local
    IID draw, so elements remain stable under view reordering/subsetting while
    a deliberately advanced generator still selects a different element bank.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    if tree not in ("a", "b"):
        raise ValueError("tree must be 'a' or 'b'")
    level_codes = {"root": 1, "coarse": 2, "pair": 3, "leaf": 4}
    if level not in level_codes:
        raise ValueError("level must be root, coarse, pair, or leaf")
    if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id < 0:
        raise ValueError("group_id must be a non-negative integer")

    # SplitMix64-style integer avalanche, with an explicit final 63-bit mask
    # because torch.Generator.manual_seed accepts signed-64-compatible seeds.
    value = (
        int(base_seed)
        + (0x9E3779B97F4A7C15 if tree == "a" else 0xD1B54A32D192ED03)
        + level_codes[level] * 0x94D049BB133111EB
        + int(group_id) * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value = value ^ (value >> 31)
    return int(value & 0x7FFFFFFFFFFFFFFF)


def _generator_state_seed_key(generator: torch.Generator) -> int:
    """Hash the complete current RNG state into a stable 63-bit bank key."""

    state = generator.get_state().detach().cpu().contiguous()
    digest = hashlib.blake2b(
        bytes(state.tolist()),
        digest_size=8,
        person=b"NILE-elements",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _validate_low_weights(
    low_frequency_weights: Sequence[float],
) -> Tuple[float, float, float, float]:
    if len(low_frequency_weights) != 4:
        raise ValueError(
            "low_frequency_weights must contain root, coarse, pair, and leaf"
        )
    weights = tuple(float(value) for value in low_frequency_weights)
    if any((not math.isfinite(value) or value < 0.0) for value in weights):
        raise ValueError("low_frequency_weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("low_frequency_weights must sum to one")
    return weights  # type: ignore[return-value]


def _dc_level_weights(
    view_slots: torch.Tensor,
    tree_mode: str,
    max_correlation: float,
    low_frequency_weights: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Allocate the requested DC shared-variance budget across tree levels.

    max_correlation names the total non-leaf variance, not the largest
    correlation observed in the selected view subset. Consequently Tree A
    and the staggered A/B construction use identical level maps, and selecting
    a principal subset of the eight slots cannot change their meaning.
    """

    root, coarse, pair, _leaf = _validate_low_weights(low_frequency_weights)
    if tree_mode not in ("a", "ab"):
        raise ValueError("tree_mode must be 'a' or 'ab'")
    if view_slots.ndim != 1 or view_slots.numel() == 0:
        raise ValueError("view_slots must be a non-empty one-dimensional sequence")
    if max_correlation == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    shared_template = root + coarse + pair
    if shared_template <= 0.0:
        raise ValueError(
            "positive max_correlation requires a positive root/coarse/pair weight"
        )
    scale = max_correlation / shared_template
    return (
        scale * root,
        scale * coarse,
        scale * pair,
        1.0 - max_correlation,
    )


def frequency_dependent_level_weights(
    height: int,
    width: int,
    view_slots: SlotsLike,
    *,
    tree_mode: str = "a",
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    low_frequency_weights: Sequence[float] = _DEFAULT_LOW_WEIGHTS,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.float32,
    onesided: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return frequency maps whose four level weights sum to one everywhere."""

    max_correlation = _probability(max_correlation, "max_correlation")
    frequency_scale = _positive_float(frequency_scale, "frequency_scale")
    slots = torch.as_tensor(view_slots, device=device, dtype=torch.long)
    if slots.ndim != 1 or slots.numel() == 0:
        raise ValueError("view_slots must be a non-empty one-dimensional sequence")
    root, coarse, pair, unused_leaf = _dc_level_weights(
        slots, tree_mode, max_correlation, low_frequency_weights
    )
    radius = radial_frequency_grid(
        height, width, device=device, dtype=dtype, onesided=onesided
    )
    envelope = torch.exp(-0.5 * (radius / frequency_scale).square())
    root_map = root * envelope
    coarse_map = coarse * envelope
    pair_map = pair * envelope
    leaf_map = 1.0 - root_map - coarse_map - pair_map
    if float(leaf_map.min().item()) < -1e-6:
        raise RuntimeError("frequency-dependent tree weights became negative")
    return {
        "root": root_map,
        "coarse": coarse_map,
        "pair": pair_map,
        "leaf": leaf_map.clamp_min(0.0),
    }


def _draw_keyed_element(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    *,
    base_seed: int,
    tree: str,
    level: str,
    group_id: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        element_seed_key(base_seed, tree, level, group_id)
    )
    return torch.randn(
        batch_size,
        channels,
        height,
        width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


@torch.no_grad()
def make_nested_tree_latents(
    batch_size: int,
    num_views: int,
    channels: int,
    height: int,
    width: int,
    view_angles: AnglesLike,
    *,
    device: DeviceLike,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    generator: GeneratorLike = None,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    tree_mode: str = "a",
    low_frequency_weights: Sequence[float] = _DEFAULT_LOW_WEIGHTS,
) -> torch.Tensor:
    """Generate white latents from nested Gaussian interval elements.

    ``tree_mode='a'`` uses one dyadic tree.  ``tree_mode='ab'`` averages two
    independent staggered trees, eliminating the circular 315/0 seam.  At
    every Fourier coefficient the element variances sum exactly to one.
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
    if tree_mode not in ("a", "ab"):
        raise ValueError("tree_mode must be 'a' or 'ab'")
    slots = angles_to_dyadic_slots(view_angles, device=device)
    if slots.numel() != num_views:
        raise ValueError("view_angles must contain exactly num_views values")
    _validate_low_weights(low_frequency_weights)
    generator = _resolve_generator(device, seed=seed, generator=generator)

    # The state is captured before the canonical IID draw. Hashing the full
    # state (rather than generator.initial_seed()) makes deliberately advanced
    # streams select a different static element bank.
    element_base_seed = _generator_state_seed_key(generator)
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

    level_weights = frequency_dependent_level_weights(
        height,
        width,
        slots,
        tree_mode=tree_mode,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        low_frequency_weights=low_frequency_weights,
        device=device,
        dtype=torch.float32,
        onesided=True,
    )
    trees = ("a",) if tree_mode == "a" else ("a", "b")
    tree_weight = 1.0 / float(len(trees))

    # The main generator has now advanced by exactly one canonical local IID
    # draw. Positive-strength latents use independently keyed leaf fields so
    # the field associated with a physical slot does not depend on traversal
    # order or on which other slots were requested.
    element_ffts: Dict[Tuple[str, str, int], torch.Tensor] = {}
    for slot in sorted(set(int(value) for value in slots.tolist())):
        leaf = _draw_keyed_element(
            batch_size,
            channels,
            height,
            width,
            base_seed=element_base_seed,
            tree="a",
            level="leaf",
            group_id=slot,
            device=device,
        )
        element_ffts[("a", "leaf", slot)] = torch.fft.rfft2(leaf, norm="ortho")

    ancestor_ids_by_tree: Dict[str, Dict[str, torch.Tensor]] = {}
    for tree in trees:
        ancestor_ids = tree_ancestor_ids(slots, tree=tree)
        ancestor_ids_by_tree[tree] = ancestor_ids
        for level in ("root", "coarse", "pair"):
            for group_id in sorted(
                set(int(value) for value in ancestor_ids[level].tolist())
            ):
                field = _draw_keyed_element(
                    batch_size,
                    channels,
                    height,
                    width,
                    base_seed=element_base_seed,
                    tree=tree,
                    level=level,
                    group_id=group_id,
                    device=device,
                )
                element_ffts[(tree, level, group_id)] = torch.fft.rfft2(
                    field, norm="ortho"
                )

    # Transform and mix each requested view independently. Besides making the
    # element addressing explicit, this avoids batch-shape-dependent FFT
    # roundoff from defeating the promised bitwise subset/order stability.
    latent_views = []
    for view_index, slot_value in enumerate(slots.tolist()):
        slot = int(slot_value)
        mixed_fft = level_weights["leaf"].sqrt()[None, None, :, :] * element_ffts[
            ("a", "leaf", slot)
        ]
        for tree in trees:
            ancestor_ids = ancestor_ids_by_tree[tree]
            for level in ("root", "coarse", "pair"):
                group_id = int(ancestor_ids[level][view_index].item())
                coefficient = (tree_weight * level_weights[level]).sqrt()[
                    None, None, :, :
                ]
                mixed_fft = mixed_fft + coefficient * element_ffts[
                    (tree, level, group_id)
                ]
        latent_views.append(
            torch.fft.irfft2(mixed_fft, s=(height, width), norm="ortho")
        )

    latents = torch.stack(latent_views, dim=1)
    return _flatten_and_cast(latents, dtype)


def nested_tree_spatial_covariance_target(
    view_angles: AnglesLike,
    height: int,
    width: int,
    *,
    max_correlation: float = 0.45,
    frequency_scale: float = 0.12,
    tree_mode: str = "a",
    low_frequency_weights: Sequence[float] = _DEFAULT_LOW_WEIGHTS,
    device: DeviceLike = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the expected same-pixel covariance of nested-tree latents."""

    slots = angles_to_dyadic_slots(view_angles, device=device)
    maps = frequency_dependent_level_weights(
        height,
        width,
        slots,
        tree_mode=tree_mode,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        low_frequency_weights=low_frequency_weights,
        device=device,
        dtype=torch.float64,
        onesided=False,
    )
    weights = tuple(float(maps[level].mean().item()) for level in ("root", "coarse", "pair", "leaf"))
    # Numerical averaging keeps the sum within roundoff; make it exact before
    # handing it to the strict covariance validator.
    weights = (weights[0], weights[1], weights[2], 1.0 - sum(weights[:3]))
    if tree_mode == "a":
        return single_tree_covariance(
            slots, weights, tree="a", device=device, dtype=dtype
        )
    if tree_mode == "ab":
        return staggered_two_tree_covariance(
            slots, weights, device=device, dtype=dtype
        )
    raise ValueError("tree_mode must be 'a' or 'ab'")


__all__ = [
    "angles_to_dyadic_slots",
    "element_seed_key",
    "frequency_dependent_level_weights",
    "make_nested_tree_latents",
    "nested_tree_spatial_covariance_target",
    "tree_ancestor_ids",
]
