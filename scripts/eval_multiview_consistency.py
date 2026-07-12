"""Evaluate camera response and collapse diagnostics in multiview outputs.

The default ``lightweight`` metrics depend only on NumPy and Pillow. They are
strictly collapse detectors and distribution guardrails, not multiview-consistency
scores and not a replacement for a geometry-aware metric. MEt3R is available as
an explicit optional backend and
is imported only when requested.  The official default ``cosine`` MEt3R output is
the distance ``1 - cosine`` in approximately ``[0, 2]``: ``met3r_score`` is lower
when consistency is better.  This is the opposite of the ``MEt3R upward arrow``
shorthand sometimes used in experiment notes.

Accepted sample layouts
-----------------------
* an inference metadata JSON file;
* a directory containing per-view images;
* a horizontal grid image plus ``--num-views``/``--azimuth-deg``;
* a manifest emitted by :mod:`scripts.run_nile_grid`.

Examples
--------
Evaluate a completed grid manifest with dependency-free proxy metrics::

    python -m scripts.eval_multiview_consistency \
        --manifest outputs/nile_grid/manifest.jsonl \
        --output outputs/nile_grid/metrics.json

Evaluate one horizontal six-view grid::

    python -m scripts.eval_multiview_consistency \
        --input output.png --num-views 6 \
        --azimuth-deg 0 45 90 180 270 315 --output metrics.csv

Run the official MEt3R package as well (large CUDA dependencies required)::

    python -m scripts.eval_multiview_consistency \
        --manifest outputs/nile_grid/manifest.jsonl --metrics all \
        --output outputs/nile_grid/metrics.json
"""

from __future__ import annotations

import argparse
import csv
import functools
import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_AZIMUTHS = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]
DEFAULT_ANGLE_BINS = [45.0, 90.0, 135.0, 180.0]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
MET3R_DIRECTIONS = {
    "cosine": "lower_is_better",
    "lpips": "lower_is_better",
    "rmse": "lower_is_better",
    "mse": "lower_is_better",
    "psnr": "higher_is_better",
    "ssim": "higher_is_better",
}
REPORT_FIELDS = (
    "experiment_id",
    "code_revision",
    "input_image",
    "input_sha256",
    "method",
    "inference_method",
    "seed",
    "config_id",
    "rank",
    "target_kl",
    "achieved_kl",
    "alpha",
    "rbf_length_scale_deg",
    "basis_checksum",
    "covariance_checksum",
    "distribution_gate_passed",
    "selection_status",
    "diagnostic_only",
    "metadata_config_conflicts",
    "max_correlation",
    "frequency_scale",
    "camera_length_scale",
    "nile_mode",
    "nile_callback",
    "rho_geo",
    "rho_start",
    "active_ratio",
)
AGGREGATE_GROUP_FIELDS = (
    "experiment_id",
    "code_revision",
    "method",
    "inference_method",
    "config_id",
    "rank",
    "target_kl",
    "achieved_kl",
    "alpha",
    "rbf_length_scale_deg",
    "basis_checksum",
    "covariance_checksum",
    "distribution_gate_passed",
    "selection_status",
    "diagnostic_only",
    "max_correlation",
    "frequency_scale",
    "camera_length_scale",
    "nile_mode",
    "nile_callback",
    "rho_geo",
)

IMMUTABLE_MANIFEST_FIELDS = (
    "run_id",
    "experiment_id",
    "code_revision",
    "input_image",
    "input_sha256",
    "method",
    "seed",
    "config_id",
    "rank",
    "target_kl",
    "rbf_length_scale_deg",
    "distribution_gate_passed",
    "selection_status",
    "diagnostic_only",
)

FEATUP_TORCH_HUB_REPOSITORY = "mhamilton723/FeatUp"


def install_featup_torch_hub_trust(torch_module: Any) -> str:
    """Trust only the official FeatUp torch.hub repository non-interactively.

    PyTorch 2.11 checks its local torch.hub trust list by default and prompts for
    confirmation when the repository is absent. The evaluator runs in a
    non-interactive subprocess, so that prompt otherwise fails with EOF.
    """

    current_load = torch_module.hub.load
    if getattr(current_load, "_mvadapter_featup_trust", False):
        return "already_installed"

    @functools.wraps(current_load)
    def trusted_load(*args: Any, **kwargs: Any) -> Any:
        repository = args[0] if args else kwargs.get("repo_or_dir")
        repository_name = str(repository).split(":", 1)[0]
        if repository_name == FEATUP_TORCH_HUB_REPOSITORY:
            kwargs["trust_repo"] = True
        return current_load(*args, **kwargs)

    setattr(trusted_load, "_mvadapter_featup_trust", True)
    torch_module.hub.load = trusted_load
    return "trust_repo_true_for_mhamilton723_FeatUp"


@dataclass
class Sample:
    sample_id: str
    source: Path
    grid_path: Optional[Path] = None
    view_paths: List[Path] = field(default_factory=list)
    azimuth_deg: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _warn(message: str) -> None:
    print("warning: " + message, file=sys.stderr)


def _flatten(values: Optional[Sequence[Sequence[str]]]) -> List[str]:
    if not values:
        return []
    return [item for group in values for item in group]


def _stable_id(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(value).stem).strip("-._") or "sample"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return "{}-{}".format(stem, digest)


def _resolve_path(value: Any, base: Path) -> Optional[Path]:
    if value is None or value == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "[{\"" or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _read_records(path: Path, list_key: str = "runs") -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                {key: _jsonish(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get(list_key, payload.get("samples", []))
        if not isinstance(payload, list):
            raise ValueError("{} does not contain a record list.".format(path))
        return [dict(record) for record in payload]
    if suffix not in {".jsonl", ".ndjson"}:
        raise ValueError("Expected a .json, .jsonl/.ndjson, or .csv file: {}".format(path))
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(dict(json.loads(line)))
                except Exception as error:
                    raise ValueError(
                        "Invalid JSON on line {} of {}: {}".format(line_number, path, error)
                    ) from error
    return records


def _expand_specs(specifications: Sequence[str], recursive: bool) -> List[Path]:
    matches: List[Path] = []
    for specification in specifications:
        expanded = os.path.expandvars(os.path.expanduser(specification))
        if glob.has_magic(expanded):
            current = [Path(item) for item in glob.glob(expanded, recursive=recursive)]
        else:
            path = Path(expanded)
            current = [path] if path.exists() else []
        if not current:
            _warn("no input matched {!r}".format(specification))
        matches.extend(item.resolve() for item in current)
    unique = []
    seen = set()
    for path in matches:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _view_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(?:view|frame)[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return (int(match.group(1)) if match else sys.maxsize, path.name.lower())


def _view_files(directory: Path, recursive: bool = False) -> List[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [
        path.resolve()
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_reference")
    ]
    return sorted(files, key=_view_sort_key)


def _metadata_angles(metadata: Mapping[str, Any]) -> List[float]:
    candidates = [
        metadata.get("azimuth_deg"),
        metadata.get("azimuths"),
        metadata.get("view_azimuths"),
    ]
    config = metadata.get("config")
    if isinstance(config, Mapping):
        candidates.append(config.get("azimuth_deg"))
    for candidate in candidates:
        candidate = _jsonish(candidate)
        if isinstance(candidate, (list, tuple)):
            try:
                return [float(item) for item in candidate]
            except (TypeError, ValueError):
                continue
    return []


def _normalize_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten the stable experiment fields used for grouping and reporting."""

    normalized = dict(metadata)
    nile = metadata.get("nile")
    if not isinstance(nile, Mapping):
        nile = {}
    distribution = metadata.get("distribution")
    if not isinstance(distribution, Mapping):
        distribution = {}
    aliases = {
        "nile_mode": ("nile_mode", "mode"),
        "nile_callback": ("nile_callback", "callback"),
        "rho_geo": ("rho_geo",),
        "rho_start": ("rho_start",),
        "rho_end": ("rho_end",),
        "active_ratio": ("active_ratio",),
    }
    for destination, candidates in aliases.items():
        if destination in normalized:
            continue
        for candidate in candidates:
            if candidate in metadata:
                normalized[destination] = metadata[candidate]
                break
            if candidate in nile:
                normalized[destination] = nile[candidate]
                break
    distribution_aliases = {
        "inference_method": ("inference_method", "method"),
        "max_correlation": ("max_correlation",),
        "frequency_scale": ("frequency_scale",),
        "camera_length_scale": ("camera_length_scale",),
        "config_id": ("config_id",),
        "rank": ("rank", "basis_rank"),
        "target_kl": ("target_kl", "target_joint_kl"),
        "achieved_kl": ("achieved_kl",),
        "alpha": ("alpha",),
        "rbf_length_scale_deg": ("rbf_length_scale_deg",),
        "basis_checksum": ("basis_checksum",),
        "covariance_checksum": ("covariance_checksum",),
        "distribution_gate_passed": ("distribution_gate_passed",),
    }
    for destination, candidates in distribution_aliases.items():
        if destination in normalized:
            continue
        for candidate in candidates:
            if candidate in distribution:
                normalized[destination] = distribution[candidate]
                break
    input_value = normalized.get("input", normalized.get("input_path"))
    if isinstance(input_value, Mapping):
        input_value = input_value.get("image", input_value.get("path"))
    if input_value is not None:
        normalized["input_image"] = str(input_value)
    if "method" not in normalized and "nile_mode" in normalized:
        callback = normalized.get("nile_callback", "none")
        normalized["method"] = (
            normalized["nile_mode"]
            if callback in {None, "none"}
            else str(callback)
        )
    return normalized


def _paths_from_view_entries(entries: Any, base: Path) -> List[Path]:
    entries = _jsonish(entries)
    if not isinstance(entries, (list, tuple)):
        return []
    paths: List[Path] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            value = entry.get("path", entry.get("file", entry.get("image")))
        else:
            value = entry
        path = _resolve_path(value, base)
        if path is not None:
            paths.append(path)
    return paths


def _sample_from_metadata(path: Path, inherited: Optional[Mapping[str, Any]] = None) -> Sample:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must contain an object: {}".format(path))
    inherited_normalized = _normalize_metadata(dict(inherited or {}))
    metadata_normalized = _normalize_metadata(metadata)
    combined: Dict[str, Any] = dict(inherited_normalized)
    combined.update(metadata_normalized)
    conflicts: Dict[str, Dict[str, Any]] = {}
    for field in IMMUTABLE_MANIFEST_FIELDS:
        if field not in inherited_normalized:
            continue
        if field in metadata_normalized and metadata_normalized[field] != inherited_normalized[field]:
            conflicts[field] = {
                "manifest": inherited_normalized[field],
                "metadata": metadata_normalized[field],
            }
        # The manifest is the frozen experiment plan and therefore canonical.
        combined[field] = inherited_normalized[field]
    if conflicts:
        combined["metadata_config_conflicts"] = conflicts
    base = path.parent

    view_paths: List[Path] = []
    for key in ("view_files", "views", "view_paths", "images"):
        view_paths = _paths_from_view_entries(metadata.get(key), base)
        if view_paths:
            break

    views_dir = _resolve_path(
        metadata.get("views_dir", (inherited or {}).get("views_dir")), base
    )
    if not view_paths and views_dir is not None and views_dir.is_dir():
        view_paths = _view_files(views_dir)

    output_value = metadata.get(
        "output", metadata.get("grid_path", (inherited or {}).get("output"))
    )
    grid_path = _resolve_path(output_value, base)
    if grid_path is not None and not grid_path.is_file():
        grid_path = None
    sample_id = str(
        metadata.get(
            "run_id",
            (inherited or {}).get("run_id", _stable_id(str(path.resolve()))),
        )
    )
    return Sample(
        sample_id=sample_id,
        source=path.resolve(),
        grid_path=grid_path,
        view_paths=view_paths,
        azimuth_deg=_metadata_angles(combined),
        metadata=combined,
    )


def _sample_from_path(
    path: Path,
    inherited: Optional[Mapping[str, Any]],
    recursive: bool,
) -> List[Sample]:
    inherited = dict(inherited or {})
    if path.is_dir():
        direct_files = _view_files(path)
        looks_like_view_dir = path.name.endswith("_views") or any(
            re.search(r"(?:view|frame)[_-]?\d+", item.stem, re.IGNORECASE)
            for item in direct_files
        )
        metadata_files = sorted(path.rglob("*_metadata.json") if recursive else path.glob("*_metadata.json"))
        if metadata_files and not looks_like_view_dir:
            return [_sample_from_metadata(item, inherited) for item in metadata_files]
        if not direct_files:
            raise ValueError("No view images or metadata JSON found in {}".format(path))
        return [
            Sample(
                sample_id=str(inherited.get("run_id", _stable_id(str(path.resolve())))),
                source=path.resolve(),
                view_paths=direct_files,
                azimuth_deg=_metadata_angles(inherited),
                metadata=inherited,
            )
        ]

    if path.suffix.lower() == ".json":
        return [_sample_from_metadata(path, inherited)]
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported sample input: {}".format(path))

    sibling_metadata = path.with_name(path.stem + "_metadata.json")
    if sibling_metadata.is_file():
        sample = _sample_from_metadata(sibling_metadata, inherited)
        if sample.grid_path is None:
            sample.grid_path = path.resolve()
        return [sample]
    sibling_views = path.with_name(path.stem + "_views")
    view_paths = _view_files(sibling_views) if sibling_views.is_dir() else []
    return [
        Sample(
            sample_id=str(inherited.get("run_id", _stable_id(str(path.resolve())))),
            source=path.resolve(),
            grid_path=path.resolve(),
            view_paths=view_paths,
            azimuth_deg=_metadata_angles(inherited),
            metadata=inherited,
        )
    ]


def _samples_from_manifests(paths: Sequence[Path], recursive: bool) -> List[Sample]:
    samples: List[Sample] = []
    for manifest_path in paths:
        records = _read_records(manifest_path, list_key="runs")
        base = manifest_path.parent
        for record in records:
            if str(record.get("status", "succeeded")) not in {"succeeded", "skipped"}:
                continue
            metadata_path = _resolve_path(record.get("metadata_path"), base)
            output_path = _resolve_path(record.get("output"), base)
            views_dir = _resolve_path(record.get("views_dir"), base)
            try:
                if metadata_path is not None and metadata_path.is_file():
                    samples.append(_sample_from_metadata(metadata_path, record))
                elif views_dir is not None and views_dir.is_dir():
                    samples.extend(_sample_from_path(views_dir, record, recursive=False))
                elif output_path is not None and output_path.is_file():
                    samples.extend(_sample_from_path(output_path, record, recursive=recursive))
                else:
                    _warn(
                        "manifest run {} has no existing output artifact".format(
                            record.get("run_id", "<unknown>")
                        )
                    )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                _warn("could not load manifest run {}: {}".format(record.get("run_id"), error))
    return samples


def _deduplicate_samples(samples: Sequence[Sample]) -> List[Sample]:
    result = []
    seen = set()
    for sample in samples:
        key = sample.sample_id
        if key in seen:
            _warn("duplicate sample id {!r}; keeping the first occurrence".format(key))
            continue
        seen.add(key)
        result.append(sample)
    return result


def _default_angles(num_views: int) -> List[float]:
    if num_views == len(DEFAULT_AZIMUTHS):
        return list(DEFAULT_AZIMUTHS)
    return [360.0 * index / num_views for index in range(num_views)]


def _load_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image_size > 0:
            image = image.resize((image_size, image_size), RESAMPLE_LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _resize_array(image: np.ndarray, image_size: int) -> np.ndarray:
    if image_size <= 0 or image.shape[:2] == (image_size, image_size):
        return image
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    resized = Image.fromarray(image_u8).resize(
        (image_size, image_size), RESAMPLE_LANCZOS
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _load_views(
    sample: Sample,
    num_views_override: Optional[int],
    azimuth_override: Optional[Sequence[float]],
    image_size: int,
) -> Tuple[List[np.ndarray], List[float], List[str]]:
    paths = [path for path in sample.view_paths if path.is_file()]
    labels: List[str] = []
    if paths:
        views = [_load_rgb(path, image_size) for path in paths]
        labels = [str(path) for path in paths]
    else:
        if sample.grid_path is None or not sample.grid_path.is_file():
            raise ValueError("No existing per-view files or grid for {}".format(sample.source))
        with Image.open(sample.grid_path) as grid:
            grid = grid.convert("RGB")
            inferred = len(sample.azimuth_deg) or (len(azimuth_override) if azimuth_override else 0)
            num_views = num_views_override or int(sample.metadata.get("num_views", 0) or 0) or inferred
            if num_views <= 0:
                raise ValueError(
                    "Cannot split grid {}; provide --num-views or --azimuth-deg.".format(
                        sample.grid_path
                    )
                )
            if grid.width % num_views != 0:
                raise ValueError(
                    "Grid width {} is not divisible by {} views: {}".format(
                        grid.width, num_views, sample.grid_path
                    )
                )
            width = grid.width // num_views
            views = []
            for index in range(num_views):
                view = grid.crop((index * width, 0, (index + 1) * width, grid.height))
                if image_size > 0:
                    view = view.resize((image_size, image_size), RESAMPLE_LANCZOS)
                views.append(np.asarray(view, dtype=np.float32) / 255.0)
                labels.append("{}#view{}".format(sample.grid_path, index))

    angles = (
        [float(item) for item in azimuth_override]
        if azimuth_override is not None
        else list(sample.azimuth_deg)
    )
    if not angles:
        angles = _default_angles(len(views))
    if len(angles) != len(views):
        raise ValueError(
            "Found {} views but {} azimuths for {}.".format(
                len(views), len(angles), sample.source
            )
        )
    if num_views_override is not None and len(views) != num_views_override:
        raise ValueError(
            "Found {} views, not the requested --num-views {}: {}".format(
                len(views), num_views_override, sample.source
            )
        )
    return views, angles, labels


def _angular_distance(first: float, second: float) -> float:
    difference = abs((first - second) % 360.0)
    return min(difference, 360.0 - difference)


def _pair_groups(
    angles: Sequence[float], adjacent_wrap: bool, opposite_tolerance: float
) -> Dict[str, List[Tuple[int, int]]]:
    if len(angles) < 2:
        raise ValueError("At least two views are required for pair evaluation.")
    order = sorted(range(len(angles)), key=lambda index: angles[index] % 360.0)
    adjacent = [(order[index], order[index + 1]) for index in range(len(order) - 1)]
    if adjacent_wrap and len(order) > 2:
        adjacent.append((order[-1], order[0]))
    adjacent = list(dict.fromkeys(tuple(sorted(pair)) for pair in adjacent))

    opposite = []
    for first in range(len(angles)):
        for second in range(first + 1, len(angles)):
            if abs(_angular_distance(angles[first], angles[second]) - 180.0) <= opposite_tolerance:
                opposite.append((first, second))
    return {"adjacent": adjacent, "opposite": opposite}


def _angle_bin_token(angle: float) -> str:
    rounded = round(float(angle))
    if math.isclose(float(angle), rounded, abs_tol=1e-9):
        return str(int(rounded))
    return format(float(angle), ".8g").replace("-", "m").replace(".", "p")


def _angle_bin_pairs(
    angles: Sequence[float],
    angle_bins: Sequence[float],
    tolerance: float,
) -> Dict[Tuple[int, int], float]:
    """Assign every matching unordered pair to its nearest real angle bin."""

    assignments: Dict[Tuple[int, int], float] = {}
    for first in range(len(angles)):
        for second in range(first + 1, len(angles)):
            distance = _angular_distance(angles[first], angles[second])
            nearest = min(angle_bins, key=lambda value: abs(distance - value))
            if abs(distance - nearest) <= tolerance:
                assignments[(first, second)] = float(nearest)
    return assignments


def _blur(image: np.ndarray, radius: float) -> np.ndarray:
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    filtered = Image.fromarray(image_u8).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(filtered, dtype=np.float32) / 255.0


def _cosine(first: np.ndarray, second: np.ndarray, center: bool = False) -> float:
    x = first.astype(np.float64, copy=False).reshape(-1)
    y = second.astype(np.float64, copy=False).reshape(-1)
    if center:
        x = x - x.mean()
        y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.dot(x, y) / denominator)


def _global_ssim_proxy(first: np.ndarray, second: np.ndarray) -> float:
    # A single-window SSIM diagnostic.  The explicit "proxy" name avoids
    # confusing it with the standard local-window implementation.
    x = first.mean(axis=2).astype(np.float64)
    y = second.mean(axis=2).astype(np.float64)
    mean_x, mean_y = float(x.mean()), float(y.mean())
    var_x, var_y = float(x.var()), float(y.var())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    denominator = (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(x, y) else 0.0
    return ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / denominator


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def _histogram_intersection(first: np.ndarray, second: np.ndarray, bins: int = 32) -> float:
    similarities = []
    for channel in range(3):
        hist_first, _ = np.histogram(first[..., channel], bins=bins, range=(0.0, 1.0))
        hist_second, _ = np.histogram(second[..., channel], bins=bins, range=(0.0, 1.0))
        hist_first = hist_first.astype(np.float64) / max(float(hist_first.sum()), 1.0)
        hist_second = hist_second.astype(np.float64) / max(float(hist_second.sum()), 1.0)
        similarities.append(float(np.minimum(hist_first, hist_second).sum()))
    return float(np.mean(similarities))


def _lightweight_metrics(
    first: np.ndarray,
    second: np.ndarray,
    blur_radius: float,
    image_size: int,
) -> Dict[str, float]:
    first = _resize_array(first, image_size)
    second = _resize_array(second, image_size)
    if first.shape != second.shape:
        raise ValueError(
            "Pair images have different shapes {} and {}; set --image-size to a "
            "positive value.".format(first.shape, second.shape)
        )
    low_first = _blur(first, blur_radius)
    low_second = _blur(second, blur_radius)
    high_first = first - low_first
    high_second = second - low_second
    difference = low_first - low_second
    return {
        "lowfreq_l1_similarity": float(np.clip(1.0 - np.mean(np.abs(difference)), 0.0, 1.0)),
        "lowfreq_rmse": float(np.sqrt(np.mean(difference ** 2))),
        "lowfreq_centered_cosine": _cosine(low_first, low_second, center=True),
        "global_ssim_proxy": float(_global_ssim_proxy(low_first, low_second)),
        "edge_cosine": _cosine(_edge_magnitude(low_first), _edge_magnitude(low_second)),
        "color_hist_intersection": _histogram_intersection(first, second),
        "highfreq_l1_distance": float(np.mean(np.abs(high_first - high_second))),
        "highfreq_cosine": _cosine(high_first, high_second),
    }


def _prepare_met3r_numpy_batch(
    image_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    image_size: int,
) -> np.ndarray:
    """Return official MEt3R input layout ``[B, 2, 3, H, W]`` in ``[-1, 1]``."""

    prepared = [
        (
            _resize_array(first, image_size),
            _resize_array(second, image_size),
        )
        for first, second in image_pairs
    ]
    shapes = {image.shape for pair in prepared for image in pair}
    if len(shapes) != 1:
        raise ValueError(
            "MEt3R inputs have different shapes {}; set --met3r-image-size "
            "to a positive value.".format(sorted(shapes))
        )
    array = np.stack([np.stack(pair, axis=0) for pair in prepared], axis=0)
    if array.ndim != 5 or array.shape[1] != 2 or array.shape[-1] != 3:
        raise ValueError(
            "MEt3R expects RGB pairs with shape [B, 2, H, W, 3], got {}.".format(
                array.shape
            )
        )
    array = array.transpose(0, 1, 4, 2, 3)
    return np.ascontiguousarray(array * 2.0 - 1.0, dtype=np.float32)


class Met3rBackend:
    """Small, lazy adapter around the optional official ``met3r`` package."""

    INSTALL_GUIDANCE = (
        "MEt3R is optional and is not installed with MV-Adapter. Install it in a "
        "compatible CUDA environment with:\n"
        "  pip install git+https://github.com/mohammadasim98/met3r\n"
        "Its official setup currently requires PyTorch >=2.1, CUDA, PyTorch3D and "
        "FeatUp (tested upstream with Python 3.10/CUDA 11.8). Alternatively rerun "
        "this evaluator with --metrics lightweight."
    )

    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            from met3r import MEt3R
        except Exception as error:
            raise RuntimeError("{}\nOriginal import error: {}".format(self.INSTALL_GUIDANCE, error)) from error

        self.torch = torch
        install_featup_torch_hub_trust(torch)
        self.device = args.met3r_device
        self.input_size = args.met3r_image_size
        self.distance = args.met3r_distance
        self.score_direction = MET3R_DIRECTIONS[args.met3r_distance]
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "--met3r-device={} requested, but CUDA is unavailable. {}".format(
                    self.device, self.INSTALL_GUIDANCE
                )
            )
        try:
            self.metric = MEt3R(
                img_size=None if args.met3r_image_size == 0 else args.met3r_image_size,
                use_norm=not args.met3r_no_norm,
                backbone=args.met3r_backbone,
                feature_backbone=args.met3r_feature_backbone,
                feature_backbone_weights=args.met3r_feature_backbone_weights,
                upsampler=args.met3r_upsampler,
                distance=args.met3r_distance,
                freeze=True,
            ).to(self.device)
            self.metric.eval()
        except Exception as error:
            raise RuntimeError("Could not initialize MEt3R: {}\n{}".format(error, self.INSTALL_GUIDANCE)) from error
        self.batch_size = args.met3r_batch_size

    def evaluate(self, image_pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
        results: List[float] = []
        torch = self.torch
        for start in range(0, len(image_pairs), self.batch_size):
            chunk = image_pairs[start : start + self.batch_size]
            try:
                array = _prepare_met3r_numpy_batch(chunk, self.input_size)
            except ValueError as error:
                raise RuntimeError(str(error)) from error
            tensor = torch.from_numpy(array).to(
                device=self.device, dtype=torch.float32
            )
            try:
                with torch.inference_mode():
                    score, *_ = self.metric(
                        images=tensor,
                        return_overlap_mask=False,
                        return_score_map=False,
                        return_projections=False,
                    )
            except Exception as error:
                raise RuntimeError("MEt3R evaluation failed: {}".format(error)) from error
            score = score.detach().float()
            if score.ndim == 0:
                if len(chunk) != 1:
                    raise RuntimeError(
                        "MEt3R returned one scalar for a batch of {} pairs; use "
                        "--met3r-batch-size 1 with this package version.".format(len(chunk))
                    )
                values = [float(score.cpu().item())]
            elif score.shape[0] == len(chunk):
                values = score.reshape(len(chunk), -1).mean(dim=1).cpu().tolist()
            elif score.numel() == len(chunk):
                values = score.reshape(-1).cpu().tolist()
            else:
                raise RuntimeError(
                    "MEt3R returned shape {} for a batch of {} pairs.".format(
                        tuple(score.shape), len(chunk)
                    )
                )
            if len(values) != len(chunk):
                raise RuntimeError(
                    "MEt3R returned {} scores for a batch of {} pairs.".format(
                        len(values), len(chunk)
                    )
                )
            results.extend(float(item) for item in values)
        return results


def _mean_or_none(values: Iterable[Any]) -> Optional[float]:
    finite = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(np.mean(finite)) if finite else None


PAIR_NON_METRIC_FIELDS = {
    "sample_id",
    "source",
    "pair_group",
    "first_index",
    "second_index",
    "first_path",
    "second_path",
    "first_azimuth_deg",
    "second_azimuth_deg",
    "angular_distance_deg",
    "angle_bin_deg",
    *REPORT_FIELDS,
}


def _numeric_metric_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in PAIR_NON_METRIC_FIELDS
            and not isinstance(value, bool)
            and isinstance(value, (int, float, np.number))
        }
    )


def _summarize_pair_rows(pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    metric_names = _numeric_metric_names(pair_rows)
    for group in ("adjacent", "opposite"):
        rows = [row for row in pair_rows if row.get("pair_group") == group]
        summary["{}_pair_count".format(group)] = len(rows)
        for metric_name in metric_names:
            summary["{}_{}".format(group, metric_name)] = _mean_or_none(
                row.get(metric_name) for row in rows
            )
    return summary


def _summarize_angle_rows(
    angle_rows: Sequence[Mapping[str, Any]],
    angle_bins: Sequence[float],
) -> Dict[str, Any]:
    """Summarize camera response at each requested real angular separation."""

    summary: Dict[str, Any] = {"angle_all_pair_count": len(angle_rows)}
    metric_names = _numeric_metric_names(angle_rows)
    for metric_name in metric_names:
        summary["angle_all_{}".format(metric_name)] = _mean_or_none(
            row.get(metric_name) for row in angle_rows
        )
    for angle_bin in angle_bins:
        token = _angle_bin_token(angle_bin)
        rows = [
            row
            for row in angle_rows
            if row.get("angle_bin_deg") is not None
            and math.isclose(float(row["angle_bin_deg"]), angle_bin, abs_tol=1e-9)
        ]
        prefix = "angle_{}".format(token)
        summary["{}_pair_count".format(prefix)] = len(rows)
        for metric_name in metric_names:
            summary["{}_{}".format(prefix, metric_name)] = _mean_or_none(
                row.get(metric_name) for row in rows
            )
    summary["camera_response_monotonic"] = _camera_response_monotonic(
        summary, angle_bins
    )
    return summary


def _camera_response_monotonic(
    summary: Mapping[str, Any],
    angle_bins: Sequence[float],
) -> str:
    """Check the lightweight S(near) > ... > S(far) collapse diagnostic."""

    values = []
    for angle_bin in sorted(angle_bins):
        key = "angle_{}_lowfreq_l1_similarity".format(
            _angle_bin_token(angle_bin)
        )
        value = _mean_or_none([summary.get(key)])
        if value is None:
            return "not_available"
        values.append(value)
    if len(values) < 2:
        return "not_available"
    return (
        "passed"
        if all(first > second for first, second in zip(values, values[1:]))
        else "failed"
    )


def _collapse_detector_label(
    angle_rows: Sequence[Mapping[str, Any]],
    angle_bins: Sequence[float],
    threshold: float,
) -> str:
    """Flag near-identical responses at every covered angle; never claim geometry."""

    similarities = []
    for angle_bin in angle_bins:
        rows = [
            row
            for row in angle_rows
            if row.get("angle_bin_deg") is not None
            and math.isclose(float(row["angle_bin_deg"]), angle_bin, abs_tol=1e-9)
        ]
        value = _mean_or_none(row.get("lowfreq_l1_similarity") for row in rows)
        if value is None:
            return "not_available" if not similarities else "incomplete_angle_coverage"
        similarities.append(value)
    return (
        "view_collapse_alert"
        if similarities and all(value >= threshold for value in similarities)
        else "no_collapse_signal"
    )


def _aggregate_samples(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group_keys = list(AGGREGATE_GROUP_FIELDS)
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for sample in samples:
        key = tuple(sample.get(name) for name in group_keys)
        groups.setdefault(key, []).append(sample)

    aggregates = []
    excluded = {
        "sample_id",
        "source",
        "status",
        "error",
        "seed",
        "num_views",
        *group_keys,
    }
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        aggregate: Dict[str, Any] = dict(zip(group_keys, key))
        aggregate["sample_count"] = len(rows)
        aggregate["successful_sample_count"] = sum(row.get("status") == "succeeded" for row in rows)
        metric_names = sorted(
            {
                name
                for row in rows
                for name, value in row.items()
                if name not in excluded and isinstance(value, (int, float, np.number))
            }
        )
        for name in metric_names:
            aggregate[name] = _mean_or_none(row.get(name) for row in rows)
        r_hf = _mean_or_none(row.get("r_hf") for row in rows)
        if r_hf is not None:
            aggregate["r_hf"] = r_hf
            aggregate["r_hf_status"] = _r_hf_status(r_hf)
        elif any(row.get("r_hf_status") == "missing_iid_reference" for row in rows):
            aggregate["r_hf_status"] = "missing_iid_reference"
        elif any(row.get("r_hf_status") == "invalid_iid_reference" for row in rows):
            aggregate["r_hf_status"] = "invalid_iid_reference"
        else:
            aggregate["r_hf_status"] = "not_available"
        aggregate["r_hf_reference_method"] = next(
            (
                row.get("r_hf_reference_method")
                for row in rows
                if row.get("r_hf_reference_method") is not None
            ),
            None,
        )
        aggregates.append(aggregate)
    return aggregates


def _build_sample_angle_bin_summaries(
    angle_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    group_keys = [
        "sample_id",
        "input_image",
        "seed",
        *AGGREGATE_GROUP_FIELDS,
        "angle_bin_deg",
    ]
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in angle_rows:
        if row.get("angle_bin_deg") is None:
            continue
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        summary: Dict[str, Any] = dict(zip(group_keys, key))
        summary["pair_count"] = len(rows)
        for metric_name in _numeric_metric_names(rows):
            summary[metric_name] = _mean_or_none(
                row.get(metric_name) for row in rows
            )
        summaries.append(summary)
    return summaries


def _build_angle_bin_summaries(
    sample_angle_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate per-sample angle summaries after paired IID normalization."""

    group_keys = [*AGGREGATE_GROUP_FIELDS, "angle_bin_deg"]
    groups: MutableMapping[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in sample_angle_rows:
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)

    summaries: List[Dict[str, Any]] = []
    excluded = {
        "sample_id",
        "input_image",
        "seed",
        "pair_count",
        *group_keys,
    }
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        summary: Dict[str, Any] = dict(zip(group_keys, key))
        summary["sample_count"] = len(rows)
        summary["pair_count"] = sum(int(row.get("pair_count", 0)) for row in rows)
        metric_names = sorted(
            {
                name
                for row in rows
                for name, value in row.items()
                if name not in excluded
                and not isinstance(value, bool)
                and isinstance(value, (int, float, np.number))
            }
        )
        for metric_name in metric_names:
            summary[metric_name] = _mean_or_none(
                row.get(metric_name) for row in rows
            )
        r_hf = _mean_or_none(row.get("r_hf") for row in rows)
        if r_hf is not None:
            summary["r_hf"] = r_hf
            summary["r_hf_status"] = _r_hf_status(r_hf)
        elif any(row.get("r_hf_status") == "missing_iid_reference" for row in rows):
            summary["r_hf_status"] = "missing_iid_reference"
        elif any(row.get("r_hf_status") == "invalid_iid_reference" for row in rows):
            summary["r_hf_status"] = "invalid_iid_reference"
        else:
            summary["r_hf_status"] = "not_available"
        summary["r_hf_reference_method"] = next(
            (
                row.get("r_hf_reference_method")
                for row in rows
                if row.get("r_hf_reference_method") is not None
            ),
            None,
        )
        summaries.append(summary)
    return summaries


def _effective_method(row: Mapping[str, Any]) -> str:
    return str(row.get("inference_method") or row.get("method") or "")


def _r_hf_status(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "not_available"
    if value > 0.75:
        return "healthy"
    if value >= 0.5:
        return "visual_check"
    if value >= 0.2:
        return "overcoupling_alert"
    return "likely_view_collapse"


def _annotate_relative_high_frequency(
    rows: Sequence[MutableMapping[str, Any]],
    *,
    iid_method: str,
    metric_name: str,
    match_fields: Sequence[str] = (),
) -> None:
    """Add R_HF = method high-frequency distance / IID reference distance."""

    if not any(_mean_or_none([row.get(metric_name)]) is not None for row in rows):
        for row in rows:
            row["r_hf"] = None
            row["r_hf_status"] = "not_available"
            row["r_hf_reference_method"] = iid_method
            row["r_hf_reference_highfreq_l1_distance"] = None
        return

    references: MutableMapping[Tuple[Any, ...], List[float]] = {}
    for row in rows:
        if _effective_method(row) != iid_method:
            continue
        value = _mean_or_none([row.get(metric_name)])
        if value is not None:
            key = tuple(row.get(field) for field in match_fields)
            references.setdefault(key, []).append(value)

    reference_means = {
        key: _mean_or_none(values) for key, values in references.items()
    }
    for row in rows:
        key = tuple(row.get(field) for field in match_fields)
        reference = reference_means.get(key)
        numerator = _mean_or_none([row.get(metric_name)])
        if reference is None:
            row["r_hf"] = None
            row["r_hf_status"] = "missing_iid_reference"
        elif reference <= 1e-12:
            row["r_hf"] = None
            row["r_hf_status"] = "invalid_iid_reference"
        elif numerator is None:
            row["r_hf"] = None
            row["r_hf_status"] = "not_available"
        else:
            row["r_hf"] = numerator / reference
            row["r_hf_status"] = _r_hf_status(row["r_hf"])
        row["r_hf_reference_method"] = iid_method
        row["r_hf_reference_highfreq_l1_distance"] = reference


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(value) for key, value in row.items()})
    os.replace(str(temporary), str(path))


def _write_results(
    path: Path,
    sample_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    angle_pair_rows: Sequence[Mapping[str, Any]],
    aggregates: Sequence[Mapping[str, Any]],
    angle_bin_summaries: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        temporary = path.with_name(path.name + ".tmp")
        payload = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "metric_notice": (
                "Lightweight pixel metrics are collapse detectors and distribution "
                "guardrails only; they are not multiview-consistency scores and "
                "not geometry-aware substitutes for MEt3R. "
                + (
                    str(settings["met3r"]["interpretation"])
                    if settings.get("met3r")
                    else ""
                )
            ).strip(),
            "settings": dict(settings),
            "samples": list(sample_rows),
            "pairs": list(pair_rows),
            "angle_pairs": list(angle_pair_rows),
            "aggregates": list(aggregates),
            "angle_bin_summaries": list(angle_bin_summaries),
        }
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(str(temporary), str(path))
        return [path]
    if path.suffix.lower() == ".csv":
        pair_path = path.with_name(path.stem + "_pairs.csv")
        angle_pair_path = path.with_name(path.stem + "_angle_pairs.csv")
        summary_path = path.with_name(path.stem + "_summary.csv")
        angle_bin_path = path.with_name(path.stem + "_angle_bins.csv")
        _write_csv(path, sample_rows)
        _write_csv(pair_path, pair_rows)
        _write_csv(angle_pair_path, angle_pair_rows)
        _write_csv(summary_path, aggregates)
        _write_csv(angle_bin_path, angle_bin_summaries)
        return [path, pair_path, angle_pair_path, summary_path, angle_bin_path]
    raise ValueError("--output must end in .json or .csv.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate angle-binned camera response, collapse guardrails, and "
            "optional MEt3R on multiview outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "MEt3R install (optional): pip install "
            "git+https://github.com/mohammadasim98/met3r"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH_OR_GLOB",
        help="Grid image, per-view directory, metadata JSON, or glob.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Grid-run manifest (.json/.jsonl/.csv); repeat to combine experiments.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directories recursively for metadata.")
    parser.add_argument("--num-views", type=int, default=None, help="Required only for grids without metadata.")
    parser.add_argument("--azimuth-deg", type=float, nargs="+", default=None, help="Override metadata azimuths.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help=(
            "Square resolution for lightweight metrics; 0 keeps source size. "
            "This option never changes MEt3R inputs."
        ),
    )
    parser.add_argument("--blur-radius", type=float, default=4.0, help="Low/high-frequency split radius in evaluation pixels.")
    parser.add_argument("--opposite-tolerance", type=float, default=5.0, help="Allowed deviation from 180 degrees.")
    parser.add_argument(
        "--angle-bins",
        type=float,
        nargs="+",
        default=DEFAULT_ANGLE_BINS,
        help="Real angular separations used for camera-response summaries.",
    )
    parser.add_argument(
        "--angle-bin-tolerance",
        type=float,
        default=5.0,
        help="Maximum absolute angular error when assigning a pair to a bin.",
    )
    parser.add_argument(
        "--iid-baseline-method",
        default="iid_default",
        help="Method used as the denominator of R_HF.",
    )
    parser.add_argument(
        "--collapse-similarity-threshold",
        type=float,
        default=0.98,
        help=(
            "Lightweight alert threshold: all available angle-bin low-frequency "
            "similarities at or above this value indicate possible view collapse."
        ),
    )
    parser.set_defaults(adjacent_wrap=True)
    parser.add_argument("--no-adjacent-wrap", dest="adjacent_wrap", action="store_false", help="Do not compare the last sorted azimuth with the first.")
    parser.add_argument("--metrics", choices=["lightweight", "met3r", "all"], default="lightweight")

    parser.add_argument("--met3r-device", default="cuda")
    parser.add_argument("--met3r-image-size", type=int, default=256, help="MEt3R internal size; 0 means dynamic input resolution.")
    parser.add_argument("--met3r-batch-size", type=int, default=1)
    parser.add_argument("--met3r-backbone", choices=["mast3r", "dust3r", "raft"], default="mast3r")
    parser.add_argument("--met3r-feature-backbone", choices=["dino16", "dinov2", "maskclip", "vit", "clip", "resnet50"], default="dino16")
    parser.add_argument("--met3r-feature-backbone-weights", default="mhamilton723/FeatUp")
    parser.add_argument("--met3r-upsampler", choices=["featup", "nearest", "bilinear", "bicubic"], default="featup")
    parser.add_argument(
        "--met3r-distance",
        choices=["cosine", "lpips", "rmse", "psnr", "mse", "ssim"],
        default="cosine",
        help=(
            "Official MEt3R comparison. cosine (1-cosine), lpips, rmse and mse "
            "are lower-is-better; psnr and ssim are higher-is-better."
        ),
    )
    parser.add_argument("--met3r-no-norm", action="store_true")

    parser.add_argument("--output", type=Path, required=True, help="Result .json or .csv path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate/discover samples without loading images.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first invalid sample.")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not _flatten(args.input) and not args.manifest:
        parser.error("Provide at least one --input or --manifest.")
    if args.num_views is not None and args.num_views < 2:
        parser.error("--num-views must be at least 2.")
    if args.azimuth_deg is not None and args.num_views is not None and len(args.azimuth_deg) != args.num_views:
        parser.error("--azimuth-deg length must equal --num-views.")
    if args.image_size < 0 or args.met3r_image_size < 0:
        parser.error("Image sizes cannot be negative.")
    if args.blur_radius <= 0:
        parser.error("--blur-radius must be positive.")
    if not 0 <= args.opposite_tolerance <= 180:
        parser.error("--opposite-tolerance must lie in [0, 180].")
    if not args.angle_bins:
        parser.error("--angle-bins requires at least one value.")
    if any(
        not math.isfinite(value) or not 0 < value <= 180
        for value in args.angle_bins
    ):
        parser.error("--angle-bins values must be finite and lie in (0, 180].")
    args.angle_bins = sorted(set(float(value) for value in args.angle_bins))
    if not math.isfinite(args.angle_bin_tolerance) or not 0 <= args.angle_bin_tolerance <= 180:
        parser.error("--angle-bin-tolerance must lie in [0, 180].")
    if not args.iid_baseline_method.strip():
        parser.error("--iid-baseline-method cannot be empty.")
    if not 0 <= args.collapse_similarity_threshold <= 1:
        parser.error("--collapse-similarity-threshold must lie in [0, 1].")
    if args.met3r_batch_size <= 0:
        parser.error("--met3r-batch-size must be positive.")
    if args.output.suffix.lower() not in {".json", ".csv"}:
        parser.error("--output must end in .json or .csv.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    samples: List[Sample] = []
    try:
        manifest_paths = [path.expanduser().resolve() for path in args.manifest]
        samples.extend(_samples_from_manifests(manifest_paths, args.recursive))
        for path in _expand_specs(_flatten(args.input), args.recursive):
            samples.extend(_sample_from_path(path, inherited=None, recursive=args.recursive))
        samples = _deduplicate_samples(samples)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if not samples:
        parser.error("No evaluable completed samples were discovered.")

    print("Discovered {} samples.".format(len(samples)))
    for sample in samples:
        layout = "{} per-view files".format(len(sample.view_paths)) if sample.view_paths else "horizontal grid"
        print("  {}: {} ({})".format(sample.sample_id, sample.source, layout))
    if args.dry_run:
        return 0

    met3r: Optional[Met3rBackend] = None
    if args.metrics in {"met3r", "all"}:
        try:
            met3r = Met3rBackend(args)
        except RuntimeError as error:
            parser.error(str(error))

    sample_rows: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    all_angle_pair_rows: List[Dict[str, Any]] = []
    failures = 0
    for index, sample in enumerate(samples, 1):
        print("[{}/{}] evaluating {}".format(index, len(samples), sample.sample_id), flush=True)
        base_row: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source": str(sample.source),
            "status": "running",
        }
        for key in REPORT_FIELDS:
            if key in sample.metadata:
                base_row[key] = sample.metadata[key]
        try:
            views, angles, labels = _load_views(
                sample,
                num_views_override=args.num_views,
                azimuth_override=args.azimuth_deg,
                image_size=0,
            )
            groups = _pair_groups(angles, args.adjacent_wrap, args.opposite_tolerance)
            angle_assignments = _angle_bin_pairs(
                angles, args.angle_bins, args.angle_bin_tolerance
            )
            required_pairs = set(angle_assignments)
            for pairs in groups.values():
                required_pairs.update(pairs)

            evaluated_rows: Dict[Tuple[int, int], Dict[str, Any]] = {}
            evaluated_images: List[Tuple[np.ndarray, np.ndarray]] = []
            evaluated_keys: List[Tuple[int, int]] = []
            for first, second in sorted(required_pairs):
                row: Dict[str, Any] = {
                    "sample_id": sample.sample_id,
                    "source": str(sample.source),
                    "first_index": first,
                    "second_index": second,
                    "first_path": labels[first],
                    "second_path": labels[second],
                    "first_azimuth_deg": angles[first],
                    "second_azimuth_deg": angles[second],
                    "angular_distance_deg": _angular_distance(
                        angles[first], angles[second]
                    ),
                    "angle_bin_deg": angle_assignments.get((first, second)),
                }
                for key in REPORT_FIELDS:
                    if key in base_row:
                        row[key] = base_row[key]
                if args.metrics in {"lightweight", "all"}:
                    row.update(
                        _lightweight_metrics(
                            views[first],
                            views[second],
                            args.blur_radius,
                            args.image_size,
                        )
                    )
                evaluated_rows[(first, second)] = row
                evaluated_keys.append((first, second))
                evaluated_images.append((views[first], views[second]))

            if met3r is not None and evaluated_images:
                scores = met3r.evaluate(evaluated_images)
                for pair_key, score in zip(evaluated_keys, scores):
                    row = evaluated_rows[pair_key]
                    row["met3r_score"] = score
                    row["met3r_score_direction"] = met3r.score_direction

            pair_rows: List[Dict[str, Any]] = []
            for group_name, pairs in groups.items():
                for pair_key in pairs:
                    row = dict(evaluated_rows[pair_key])
                    row["pair_group"] = group_name
                    pair_rows.append(row)
            angle_rows: List[Dict[str, Any]] = []
            for pair_key, angle_bin in sorted(
                angle_assignments.items(), key=lambda item: (item[1], item[0])
            ):
                row = dict(evaluated_rows[pair_key])
                row["pair_group"] = "angle_{}".format(
                    _angle_bin_token(angle_bin)
                )
                angle_rows.append(row)

            base_row.update(
                {
                    "status": "succeeded",
                    "num_views": len(views),
                    "azimuth_deg": angles,
                    **_summarize_pair_rows(pair_rows),
                    **_summarize_angle_rows(angle_rows, args.angle_bins),
                    "collapse_detector_label": _collapse_detector_label(
                        angle_rows,
                        args.angle_bins,
                        args.collapse_similarity_threshold,
                    ),
                }
            )
            if met3r is not None:
                base_row["met3r_score_direction"] = met3r.score_direction
            sample_rows.append(base_row)
            all_pair_rows.extend(pair_rows)
            all_angle_pair_rows.extend(angle_rows)
            if not angle_rows:
                _warn(
                    "sample {} has no pairs in angle bins {} +/- {} degrees".format(
                        sample.sample_id,
                        args.angle_bins,
                        args.angle_bin_tolerance,
                    )
                )
            if not groups["opposite"]:
                _warn(
                    "sample {} has no pairs within {} degrees of opposite".format(
                        sample.sample_id, args.opposite_tolerance
                    )
                )
        except Exception as error:
            failures += 1
            base_row.update({"status": "failed", "error": str(error)})
            sample_rows.append(base_row)
            print("sample {} failed: {}".format(sample.sample_id, error), file=sys.stderr)
            if args.fail_fast:
                break

    _annotate_relative_high_frequency(
        sample_rows,
        iid_method=args.iid_baseline_method,
        metric_name="angle_all_highfreq_l1_distance",
        match_fields=("experiment_id", "code_revision", "input_image", "seed"),
    )
    aggregates = _aggregate_samples(sample_rows)
    for aggregate in aggregates:
        aggregate["camera_response_monotonic"] = _camera_response_monotonic(
            aggregate, args.angle_bins
        )
    sample_angle_bin_summaries = _build_sample_angle_bin_summaries(
        all_angle_pair_rows
    )
    _annotate_relative_high_frequency(
        sample_angle_bin_summaries,
        iid_method=args.iid_baseline_method,
        metric_name="highfreq_l1_distance",
        match_fields=(
            "experiment_id",
            "code_revision",
            "input_image",
            "seed",
            "angle_bin_deg",
        ),
    )
    angle_bin_summaries = _build_angle_bin_summaries(
        sample_angle_bin_summaries
    )
    settings = {
        "metrics": args.metrics,
        "lightweight_role": "collapse_detector_only",
        "image_size": args.image_size,
        "blur_radius": args.blur_radius,
        "opposite_tolerance": args.opposite_tolerance,
        "adjacent_wrap": args.adjacent_wrap,
        "angle_bins_deg": list(args.angle_bins),
        "angle_bin_tolerance_deg": args.angle_bin_tolerance,
        "collapse_similarity_threshold": args.collapse_similarity_threshold,
        "camera_response_monotonic": (
            "lightweight collapse diagnostic only: "
            "S(45) > S(90) > S(135) > S(180) for the configured bins"
        ),
        "r_hf": {
            "definition": (
                "method highfreq_l1_distance / IID highfreq_l1_distance, "
                "paired by experiment, code revision, input, seed, and angle bin "
                "before aggregation"
            ),
            "iid_baseline_method": args.iid_baseline_method,
            "healthy": "> 0.75",
            "visual_check": "0.5 <= R_HF <= 0.75",
            "overcoupling_alert": "0.2 <= R_HF < 0.5",
            "likely_view_collapse": "< 0.2",
        },
        "met3r": (
            {
                "device": args.met3r_device,
                "image_size": args.met3r_image_size,
                "backbone": args.met3r_backbone,
                "feature_backbone": args.met3r_feature_backbone,
                "upsampler": args.met3r_upsampler,
                "distance": args.met3r_distance,
                "score_direction": met3r.score_direction,
                "interpretation": (
                    "met3r_score uses {} and is {}. For the default cosine "
                    "distance (1-cosine, approximately [0, 2]), lower scores mean "
                    "better multiview consistency."
                ).format(args.met3r_distance, met3r.score_direction),
            }
            if met3r is not None
            else None
        ),
    }
    output_path = args.output.expanduser().resolve()
    try:
        written = _write_results(
            output_path,
            sample_rows,
            all_pair_rows,
            all_angle_pair_rows,
            aggregates,
            angle_bin_summaries,
            settings,
        )
    except (OSError, ValueError) as error:
        parser.error("Could not write results: {}".format(error))
    print("Wrote: " + ", ".join(str(path) for path in written))
    if failures:
        print("{} samples failed; successful samples were still written.".format(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
