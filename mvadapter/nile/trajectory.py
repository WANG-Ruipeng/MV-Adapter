"""Read-only denoising trajectory diagnostics for low-rank view coupling.

The observer in this module is deliberately passive: it projects snapshots of
the denoising state onto a supplied orthonormal basis, but returns the exact
``callback_kwargs`` mapping (and therefore the exact latent tensor object) it
received.  It must never be used to inject scheduler noise or to alter the
denoising state.

``callback_on_step_end`` runs after a scheduler step, so a real initial sample
cannot be inferred by the callback.  Call :meth:`TrajectoryObserver.record_initial`
before inference, or pass ``initial_latents`` to the constructor.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


DEFAULT_TRAJECTORY_MILESTONES: Tuple[float, ...] = (
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    1.0,
)
TRAJECTORY_SCHEMA_VERSION = "nile_trajectory_v1"


PathLike = Union[str, Path]
TrajectorySource = Union[
    "TrajectoryObserver", Mapping[str, np.ndarray], str, Path
]


def _validate_milestones(milestones: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in milestones)
    if not values:
        raise ValueError("milestones must not be empty")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("milestones must be finite values in [0, 1]")
    if tuple(sorted(set(values))) != values:
        raise ValueError("milestones must be strictly increasing and unique")
    if not math.isclose(values[0], 0.0, abs_tol=1e-12):
        raise ValueError("milestones must start at 0.0 (initial)")
    if not math.isclose(values[-1], 1.0, abs_tol=1e-12):
        raise ValueError("milestones must end at 1.0 (final)")
    return values


def milestone_label(progress: float) -> str:
    """Return a stable human-readable label for a relative milestone."""

    progress = float(progress)
    if math.isclose(progress, 0.0, abs_tol=1e-12):
        return "initial"
    if math.isclose(progress, 1.0, abs_tol=1e-12):
        return "final"
    return f"{progress * 100:g}%"


def milestone_step_indices(
    total_steps: int,
    milestones: Sequence[float] = DEFAULT_TRAJECTORY_MILESTONES,
) -> Dict[float, int]:
    """Map progress targets to zero-based post-scheduler callback indices.

    A target is assigned to the nearest completed-step fraction, with ties
    rounded upward.  The initial state is represented by ``-1`` because it is
    recorded before the first scheduler step.
    """

    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    targets = _validate_milestones(milestones)
    result: Dict[float, int] = {}
    for target in targets:
        if target == 0.0:
            result[target] = -1
            continue
        completed_steps = int(math.floor(target * total_steps + 0.5))
        completed_steps = min(total_steps, max(1, completed_steps))
        result[target] = completed_steps - 1
    return result


def _validate_basis(basis: torch.Tensor) -> torch.Tensor:
    if not isinstance(basis, torch.Tensor):
        raise TypeError(f"basis must be a torch.Tensor, got {type(basis).__name__}")
    if basis.ndim != 2 or basis.shape[0] <= 0 or basis.shape[1] <= 0:
        raise ValueError(
            f"basis must have non-empty shape [D, K], got {tuple(basis.shape)}"
        )
    if not basis.is_floating_point():
        raise TypeError(f"basis must be floating point, got {basis.dtype}")
    if not bool(torch.isfinite(basis).all().item()):
        raise ValueError("basis must contain only finite values")
    return basis.detach().clone()


def project_basis_coefficients(
    latents: torch.Tensor,
    basis: torch.Tensor,
    *,
    num_views: int,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Project ``[B*V,C,H,W]`` latents into coefficients ``[B,V,K]``.

    This function performs no in-place operations and never attaches the
    diagnostic projection to an autograd graph.
    """

    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latents must be a torch.Tensor, got {type(latents).__name__}")
    if latents.ndim != 4 or any(size <= 0 for size in latents.shape):
        raise ValueError(
            "latents must have non-empty shape [B * V, C, H, W], got "
            f"{tuple(latents.shape)}"
        )
    if not latents.is_floating_point():
        raise TypeError(f"latents must be floating point, got {latents.dtype}")
    if isinstance(num_views, bool) or not isinstance(num_views, int):
        raise TypeError("num_views must be an integer")
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    bvc, channels, height, width = latents.shape
    if bvc % num_views != 0:
        raise ValueError(
            f"latent batch {bvc} is not divisible by num_views={num_views}"
        )
    inferred_batch = bvc // num_views
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer or None")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if inferred_batch != batch_size:
            raise ValueError(
                f"inferred batch size is {inferred_batch}, expected {batch_size}"
            )

    basis = _validate_basis(basis)
    latent_dimension = channels * height * width
    if basis.shape[0] != latent_dimension:
        raise ValueError(
            f"basis dimension D={basis.shape[0]} does not match latent D={latent_dimension}"
        )
    work_dtype = (
        torch.float64
        if latents.dtype == torch.float64 or basis.dtype == torch.float64
        else torch.float32
    )
    with torch.no_grad():
        flat = latents.detach().reshape(inferred_batch, num_views, latent_dimension)
        coefficients = torch.matmul(
            flat.to(dtype=work_dtype),
            basis.to(device=latents.device, dtype=work_dtype),
        )
    return coefficients


def coefficient_view_correlation(
    coefficients: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-batch Pearson view correlation matrices ``[B,V,V]``."""

    if not isinstance(coefficients, torch.Tensor) or coefficients.ndim != 3:
        raise ValueError("coefficients must be a torch.Tensor with shape [B, V, K]")
    if not coefficients.is_floating_point():
        raise TypeError("coefficients must be floating point")
    if coefficients.shape[-1] < 2:
        raise ValueError("at least two basis coefficients are needed for correlation")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")

    with torch.no_grad():
        centered = coefficients.detach() - coefficients.detach().mean(
            dim=-1, keepdim=True
        )
        lengths = torch.linalg.vector_norm(centered, dim=-1)
        denominator = lengths.unsqueeze(-1) * lengths.unsqueeze(-2)
        gram = torch.matmul(centered, centered.transpose(-1, -2))
        correlation = gram / denominator.clamp_min(eps)
        views = coefficients.shape[1]
        eye = torch.eye(views, device=coefficients.device, dtype=coefficients.dtype)
        correlation = correlation * (1.0 - eye) + eye
        correlation = correlation.clamp(-1.0, 1.0)
    return correlation


def _timestep_value(timestep: Any) -> float:
    if timestep is None:
        return float("nan")
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() != 1:
            raise ValueError("timestep tensor must contain exactly one value")
        return float(timestep.detach().cpu().item())
    try:
        value = float(timestep)
    except (TypeError, ValueError) as error:
        raise TypeError("timestep must be scalar-like") from error
    return value


@dataclass(frozen=True)
class TrajectorySnapshot:
    """A compact, CPU-resident diagnostic snapshot."""

    milestone: str
    target_progress: float
    actual_progress: float
    step: int
    timestep: float
    basis_coefficients: np.ndarray
    view_correlation: np.ndarray
    offdiag_frobenius: np.ndarray
    per_view_coefficient_norm: np.ndarray
    g_t: np.ndarray


class TrajectoryObserver:
    """Passive Diffusers ``callback_on_step_end`` trajectory observer.

    Parameters
    ----------
    basis:
        Orthonormal basis with shape ``[C*H*W, K]``.  The observer does not
        alter or re-normalize it.
    num_views:
        Number of views packed into the leading latent dimension.
    total_steps:
        Optional known scheduler step count.  Otherwise ``pipe._num_timesteps``
        is read on the first callback.
    initial_latents:
        Optional true pre-scheduler state.  Supplying it is equivalent to
        calling :meth:`record_initial` immediately after construction.
    """

    tensor_inputs = ["latents"]

    def __init__(
        self,
        basis: torch.Tensor,
        *,
        num_views: int,
        batch_size: Optional[int] = None,
        total_steps: Optional[int] = None,
        milestones: Sequence[float] = DEFAULT_TRAJECTORY_MILESTONES,
        initial_latents: Optional[torch.Tensor] = None,
        initial_timestep: Any = None,
        eps: float = 1e-12,
    ) -> None:
        self.basis = _validate_basis(basis)
        if isinstance(num_views, bool) or not isinstance(num_views, int):
            raise TypeError("num_views must be an integer")
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        if batch_size is not None and (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if total_steps is not None and (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer or None")
        eps = float(eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be finite and positive")

        self.num_views = num_views
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.milestones = _validate_milestones(milestones)
        self.eps = eps
        self._snapshots: list[TrajectorySnapshot] = []
        self._captured: set[float] = set()
        self._initial_offdiag: Optional[np.ndarray] = None
        self._step_schedule: Optional[Dict[float, int]] = None
        if initial_latents is not None:
            self.record_initial(initial_latents, timestep=initial_timestep)

    @property
    def snapshots(self) -> Tuple[TrajectorySnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def captured_milestones(self) -> Tuple[str, ...]:
        return tuple(snapshot.milestone for snapshot in self._snapshots)

    def _resolve_total_steps(self, pipe: Any) -> int:
        candidate = self.total_steps
        if candidate is None:
            candidate = getattr(pipe, "_num_timesteps", None)
        if isinstance(candidate, torch.Tensor):
            if candidate.numel() != 1:
                raise ValueError("pipe._num_timesteps must be scalar")
            candidate = int(candidate.detach().cpu().item())
        elif candidate is not None:
            try:
                candidate = int(candidate)
            except (TypeError, ValueError) as error:
                raise TypeError("total scheduler steps must be an integer") from error
        if candidate is None or candidate <= 0:
            raise ValueError(
                "total_steps is required (pass it explicitly or expose pipe._num_timesteps)"
            )
        if self.total_steps is None:
            self.total_steps = candidate
        elif candidate != self.total_steps:
            raise ValueError(
                f"scheduler step count changed from {self.total_steps} to {candidate}"
            )
        if self._step_schedule is None:
            self._step_schedule = milestone_step_indices(
                self.total_steps, self.milestones
            )
        return self.total_steps

    def _measure(
        self,
        latents: torch.Tensor,
        *,
        target_progress: float,
        actual_progress: float,
        step: int,
        timestep: Any,
    ) -> TrajectorySnapshot:
        coefficients = project_basis_coefficients(
            latents,
            self.basis,
            num_views=self.num_views,
            batch_size=self.batch_size,
        )
        if self.batch_size is None:
            self.batch_size = int(coefficients.shape[0])
        correlation = coefficient_view_correlation(coefficients, eps=self.eps)
        eye = torch.eye(
            self.num_views,
            device=correlation.device,
            dtype=correlation.dtype,
        )
        offdiag = torch.linalg.matrix_norm(correlation - eye, ord="fro")
        coefficient_norm = torch.linalg.vector_norm(coefficients, dim=-1)

        coefficients_np = (
            coefficients.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        )
        correlation_np = (
            correlation.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        )
        offdiag_np = (
            offdiag.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
        )
        norm_np = (
            coefficient_norm.detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
            .copy()
        )
        if self._initial_offdiag is None:
            g_t = np.full_like(offdiag_np, np.nan, dtype=np.float64)
        else:
            g_t = np.divide(
                offdiag_np,
                self._initial_offdiag,
                out=np.full_like(offdiag_np, np.nan, dtype=np.float64),
                where=np.abs(self._initial_offdiag) > self.eps,
            )
        return TrajectorySnapshot(
            milestone=milestone_label(target_progress),
            target_progress=float(target_progress),
            actual_progress=float(actual_progress),
            step=int(step),
            timestep=_timestep_value(timestep),
            basis_coefficients=coefficients_np,
            view_correlation=correlation_np,
            offdiag_frobenius=offdiag_np,
            per_view_coefficient_norm=norm_np,
            g_t=g_t,
        )

    def record_initial(self, latents: torch.Tensor, *, timestep: Any = None) -> None:
        """Record the true pre-scheduler latent state exactly once."""

        if 0.0 in self._captured:
            raise RuntimeError("initial trajectory state has already been recorded")
        snapshot = self._measure(
            latents,
            target_progress=0.0,
            actual_progress=0.0,
            step=-1,
            timestep=timestep,
        )
        self._initial_offdiag = snapshot.offdiag_frobenius.copy()
        initial_g = np.divide(
            snapshot.offdiag_frobenius,
            self._initial_offdiag,
            out=np.full_like(snapshot.offdiag_frobenius, np.nan),
            where=np.abs(self._initial_offdiag) > self.eps,
        )
        snapshot = TrajectorySnapshot(
            **{**snapshot.__dict__, "g_t": initial_g}
        )
        self._snapshots.append(snapshot)
        self._captured.add(0.0)

    def __call__(
        self,
        pipe: Any,
        step: int,
        timestep: Any,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Observe a scheduler step and return its inputs without modification."""

        if not isinstance(callback_kwargs, dict):
            raise TypeError("callback_kwargs must be a dict")
        if "latents" not in callback_kwargs:
            raise KeyError("callback_kwargs must contain 'latents'")
        latents = callback_kwargs["latents"]
        if not isinstance(latents, torch.Tensor):
            raise TypeError("callback_kwargs['latents'] must be a torch.Tensor")
        if 0.0 not in self._captured:
            raise RuntimeError(
                "record_initial must be called with the pre-scheduler latents before callbacks"
            )
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")

        total_steps = self._resolve_total_steps(pipe)
        if step >= total_steps:
            raise ValueError(f"step {step} is outside total_steps={total_steps}")
        assert self._step_schedule is not None
        due = [
            target
            for target in self.milestones
            if target > 0.0
            and target not in self._captured
            and step >= self._step_schedule[target]
        ]
        for target in due:
            self._snapshots.append(
                self._measure(
                    latents,
                    target_progress=target,
                    actual_progress=(step + 1) / total_steps,
                    step=step,
                    timestep=timestep,
                )
            )
            self._captured.add(target)

        # The identity of both objects is intentional and is part of the
        # mutation-free contract tested by the integration regression.
        return callback_kwargs

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Return the complete trajectory using an ``allow_pickle=False`` schema."""

        if not self._snapshots:
            raise RuntimeError("no trajectory snapshots have been recorded")
        basis_cpu = (
            self.basis.detach().to(device="cpu", dtype=torch.float64).contiguous().numpy()
        )
        checksum_hasher = hashlib.sha256()
        checksum_hasher.update(str(tuple(basis_cpu.shape)).encode("ascii"))
        checksum_hasher.update(basis_cpu.tobytes(order="C"))
        return {
            "schema_version": np.asarray(TRAJECTORY_SCHEMA_VERSION),
            "milestones": np.asarray(
                [snapshot.milestone for snapshot in self._snapshots], dtype="U32"
            ),
            "target_progress": np.asarray(
                [snapshot.target_progress for snapshot in self._snapshots],
                dtype=np.float64,
            ),
            "actual_progress": np.asarray(
                [snapshot.actual_progress for snapshot in self._snapshots],
                dtype=np.float64,
            ),
            "steps": np.asarray(
                [snapshot.step for snapshot in self._snapshots], dtype=np.int64
            ),
            "timesteps": np.asarray(
                [snapshot.timestep for snapshot in self._snapshots], dtype=np.float64
            ),
            "basis_coefficients": np.stack(
                [snapshot.basis_coefficients for snapshot in self._snapshots]
            ),
            "view_correlation": np.stack(
                [snapshot.view_correlation for snapshot in self._snapshots]
            ),
            "offdiag_frobenius": np.stack(
                [snapshot.offdiag_frobenius for snapshot in self._snapshots]
            ),
            "per_view_coefficient_norm": np.stack(
                [snapshot.per_view_coefficient_norm for snapshot in self._snapshots]
            ),
            "g_t": np.stack([snapshot.g_t for snapshot in self._snapshots]),
            "num_views": np.asarray(self.num_views, dtype=np.int64),
            "batch_size": np.asarray(self.batch_size, dtype=np.int64),
            "basis_rank": np.asarray(self.basis.shape[1], dtype=np.int64),
            "latent_dimension": np.asarray(self.basis.shape[0], dtype=np.int64),
            "basis_checksum": np.asarray(checksum_hasher.hexdigest()),
        }

    def save(self, output_prefix: PathLike, *, make_plot: bool = True) -> Dict[str, Optional[Path]]:
        """Atomically save NPZ/CSV diagnostics and, when available, a PNG plot."""

        prefix = _normalise_prefix(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_arrays()
        npz_path = prefix.with_suffix(".npz")
        csv_path = prefix.with_suffix(".csv")
        _atomic_save_npz(npz_path, arrays)
        _atomic_save_trajectory_csv(csv_path, arrays)
        plot_path: Optional[Path] = None
        if make_plot:
            plot_path = _save_trajectory_plot(prefix.with_suffix(".png"), arrays)
        return {"npz": npz_path, "csv": csv_path, "plot": plot_path}


def _normalise_prefix(path: PathLike) -> Path:
    result = Path(path)
    if result.suffix.lower() in {".npz", ".csv", ".png"}:
        result = result.with_suffix("")
    return result


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_save_trajectory_csv(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fields = [
        "milestone",
        "target_progress",
        "actual_progress",
        "step",
        "timestep",
        "batch",
        "offdiag_frobenius",
        "g_t",
        "per_view_coefficient_norm",
        "basis_coefficients",
        "view_correlation",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        count, batch_size = arrays["offdiag_frobenius"].shape
        for index in range(count):
            for batch in range(batch_size):
                g_value = float(arrays["g_t"][index, batch])
                writer.writerow(
                    {
                        "milestone": str(arrays["milestones"][index]),
                        "target_progress": float(arrays["target_progress"][index]),
                        "actual_progress": float(arrays["actual_progress"][index]),
                        "step": int(arrays["steps"][index]),
                        "timestep": float(arrays["timesteps"][index]),
                        "batch": batch,
                        "offdiag_frobenius": float(
                            arrays["offdiag_frobenius"][index, batch]
                        ),
                        "g_t": g_value if math.isfinite(g_value) else "",
                        "per_view_coefficient_norm": json.dumps(
                            arrays["per_view_coefficient_norm"][index, batch].tolist(),
                            separators=(",", ":"),
                        ),
                        "basis_coefficients": json.dumps(
                            arrays["basis_coefficients"][index, batch].tolist(),
                            separators=(",", ":"),
                        ),
                        "view_correlation": json.dumps(
                            arrays["view_correlation"][index, batch].tolist(),
                            separators=(",", ":"),
                        ),
                    }
                )
    temporary.replace(path)


def _save_trajectory_plot(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError):
        return None

    x = arrays["target_progress"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(x, np.nanmean(arrays["g_t"], axis=1), marker="o")
    axes[0].axhline(1.0, color="0.6", linewidth=1.0)
    axes[0].set(xlabel="relative progress", ylabel="G_t", title="Correlation retention")
    axes[1].plot(
        x,
        np.nanmean(arrays["offdiag_frobenius"], axis=1),
        marker="o",
    )
    axes[1].set(
        xlabel="relative progress",
        ylabel="off-diagonal Frobenius norm",
        title="View correlation",
    )
    figure.tight_layout()
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    temporary.replace(path)
    return path


_REQUIRED_NPZ_KEYS = {
    "schema_version",
    "milestones",
    "target_progress",
    "actual_progress",
    "steps",
    "timesteps",
    "basis_coefficients",
    "view_correlation",
    "offdiag_frobenius",
    "per_view_coefficient_norm",
    "g_t",
}


def load_trajectory_npz(path: PathLike) -> Dict[str, np.ndarray]:
    """Load and minimally validate a trajectory NPZ without pickle support."""

    with np.load(Path(path), allow_pickle=False) as archive:
        missing = _REQUIRED_NPZ_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"trajectory NPZ is missing keys: {sorted(missing)}")
        arrays = {key: archive[key].copy() for key in archive.files}
    if str(arrays["schema_version"].item()) != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported trajectory schema {arrays['schema_version'].item()!r}"
        )
    return arrays


def _trajectory_arrays(source: TrajectorySource) -> Dict[str, np.ndarray]:
    if isinstance(source, TrajectoryObserver):
        return source.to_arrays()
    if isinstance(source, (str, Path)):
        return load_trajectory_npz(source)
    if isinstance(source, Mapping):
        missing = {"milestones", "target_progress", "basis_coefficients"}.difference(
            source
        )
        if missing:
            raise ValueError(f"trajectory mapping is missing keys: {sorted(missing)}")
        return {key: np.asarray(value) for key, value in source.items()}
    raise TypeError(f"unsupported trajectory source {type(source).__name__}")


def compute_paired_delta(
    correlated: TrajectorySource,
    iid: TrajectorySource,
    *,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Compute IID-paired ``Delta_t`` for matching input/seed trajectories.

    ``Delta_t = ||A_corr - A_iid|| / ||A_iid||`` is reported separately for
    each batch item.  Pairing is strict: milestone labels, target progress and
    coefficient tensor shapes must match exactly.
    """

    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    corr = _trajectory_arrays(correlated)
    base = _trajectory_arrays(iid)
    corr_labels = np.asarray(corr["milestones"]).astype("U32")
    base_labels = np.asarray(base["milestones"]).astype("U32")
    if not np.array_equal(corr_labels, base_labels):
        raise ValueError("paired trajectories must have identical milestones")
    corr_progress = np.asarray(corr["target_progress"], dtype=np.float64)
    base_progress = np.asarray(base["target_progress"], dtype=np.float64)
    if not np.array_equal(corr_progress, base_progress):
        raise ValueError("paired trajectories must have identical target progress")
    corr_coeff = np.asarray(corr["basis_coefficients"], dtype=np.float64)
    base_coeff = np.asarray(base["basis_coefficients"], dtype=np.float64)
    if corr_coeff.shape != base_coeff.shape or corr_coeff.ndim != 4:
        raise ValueError(
            "paired basis coefficients must share shape [milestone, B, V, K]"
        )
    numerator = np.linalg.norm(corr_coeff - base_coeff, axis=(-2, -1))
    denominator = np.linalg.norm(base_coeff, axis=(-2, -1))
    delta = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > eps,
    )
    return {
        "schema_version": np.asarray("nile_paired_delta_v1"),
        "milestones": corr_labels,
        "target_progress": corr_progress,
        "delta_t": delta,
        "numerator_norm": numerator,
        "iid_denominator_norm": denominator,
    }


def save_paired_delta(
    correlated: TrajectorySource,
    iid: TrajectorySource,
    output_prefix: PathLike,
    *,
    make_plot: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Optional[Path]]:
    """Save paired ``Delta_t`` as NPZ/CSV and optionally a PNG curve."""

    arrays = compute_paired_delta(correlated, iid, eps=eps)
    prefix = _normalise_prefix(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    csv_path = prefix.with_suffix(".csv")
    _atomic_save_npz(npz_path, arrays)
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "milestone",
                "target_progress",
                "batch",
                "delta_t",
                "numerator_norm",
                "iid_denominator_norm",
            ],
        )
        writer.writeheader()
        for index, label in enumerate(arrays["milestones"]):
            for batch in range(arrays["delta_t"].shape[1]):
                writer.writerow(
                    {
                        "milestone": str(label),
                        "target_progress": float(arrays["target_progress"][index]),
                        "batch": batch,
                        "delta_t": float(arrays["delta_t"][index, batch]),
                        "numerator_norm": float(
                            arrays["numerator_norm"][index, batch]
                        ),
                        "iid_denominator_norm": float(
                            arrays["iid_denominator_norm"][index, batch]
                        ),
                    }
                )
    temporary.replace(csv_path)

    plot_path: Optional[Path] = None
    if make_plot:
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError):
            pass
        else:
            plot_path = prefix.with_suffix(".png")
            figure, axis = plt.subplots(figsize=(5, 3.5))
            axis.plot(
                arrays["target_progress"],
                np.nanmean(arrays["delta_t"], axis=1),
                marker="o",
            )
            axis.set(
                xlabel="relative progress",
                ylabel="Delta_t",
                title="IID-paired latent divergence",
            )
            figure.tight_layout()
            temporary_plot = plot_path.with_name(
                plot_path.stem + ".tmp" + plot_path.suffix
            )
            figure.savefig(temporary_plot, dpi=150)
            plt.close(figure)
            temporary_plot.replace(plot_path)
    return {"npz": npz_path, "csv": csv_path, "plot": plot_path}


# A descriptive alias makes the class easy to discover without changing the
# package-level exports while the wider workflow integration is developed.
NILETrajectoryObserver = TrajectoryObserver


__all__ = [
    "DEFAULT_TRAJECTORY_MILESTONES",
    "NILETrajectoryObserver",
    "TRAJECTORY_SCHEMA_VERSION",
    "TrajectoryObserver",
    "TrajectorySnapshot",
    "coefficient_view_correlation",
    "compute_paired_delta",
    "load_trajectory_npz",
    "milestone_label",
    "milestone_step_indices",
    "project_basis_coefficients",
    "save_paired_delta",
]
