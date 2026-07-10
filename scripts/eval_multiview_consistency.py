"""Evaluate adjacent and opposite pairs in multiview image outputs.

The default ``lightweight`` metrics depend only on NumPy and Pillow.  They are
image-space diagnostics, not a replacement for a geometry-aware metric: they are
useful for quickly detecting low-frequency drift and high-frequency collapse while
an experiment is running.  MEt3R is available as an explicit optional backend and
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
    combined: Dict[str, Any] = dict(inherited or {})
    combined.update(metadata)
    combined = _normalize_metadata(combined)
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


def _summarize_pair_rows(pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    metric_names = sorted(
        {
            key
            for row in pair_rows
            for key, value in row.items()
            if key
            not in {
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
            }
            and isinstance(value, (int, float, np.number))
        }
    )
    for group in ("adjacent", "opposite"):
        rows = [row for row in pair_rows if row.get("pair_group") == group]
        summary["{}_pair_count".format(group)] = len(rows)
        for metric_name in metric_names:
            summary["{}_{}".format(group, metric_name)] = _mean_or_none(
                row.get(metric_name) for row in rows
            )
    return summary


def _aggregate_samples(samples: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    group_keys = ["method", "nile_mode", "nile_callback", "rho_geo"]
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
        aggregates.append(aggregate)
    return aggregates


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
    aggregates: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> List[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        temporary = path.with_name(path.name + ".tmp")
        payload = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "metric_notice": (
                "Lightweight metrics are unregistered image-space diagnostics; "
                "they are not geometry-aware substitutes for MEt3R. "
                + (
                    str(settings["met3r"]["interpretation"])
                    if settings.get("met3r")
                    else ""
                )
            ).strip(),
            "settings": dict(settings),
            "samples": list(sample_rows),
            "pairs": list(pair_rows),
            "aggregates": list(aggregates),
        }
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(str(temporary), str(path))
        return [path]
    if path.suffix.lower() == ".csv":
        pair_path = path.with_name(path.stem + "_pairs.csv")
        summary_path = path.with_name(path.stem + "_summary.csv")
        _write_csv(path, sample_rows)
        _write_csv(pair_path, pair_rows)
        _write_csv(summary_path, aggregates)
        return [path, pair_path, summary_path]
    raise ValueError("--output must end in .json or .csv.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate adjacent/opposite consistency of multiview outputs.",
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
    failures = 0
    for index, sample in enumerate(samples, 1):
        print("[{}/{}] evaluating {}".format(index, len(samples), sample.sample_id), flush=True)
        base_row: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "source": str(sample.source),
            "status": "running",
        }
        for key in ("method", "nile_mode", "nile_callback", "seed", "rho_geo", "rho_start", "active_ratio"):
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
            pair_rows: List[Dict[str, Any]] = []
            pair_images: List[Tuple[np.ndarray, np.ndarray]] = []
            for group_name, pairs in groups.items():
                for first, second in pairs:
                    row: Dict[str, Any] = {
                        "sample_id": sample.sample_id,
                        "source": str(sample.source),
                        "pair_group": group_name,
                        "first_index": first,
                        "second_index": second,
                        "first_path": labels[first],
                        "second_path": labels[second],
                        "first_azimuth_deg": angles[first],
                        "second_azimuth_deg": angles[second],
                        "angular_distance_deg": _angular_distance(angles[first], angles[second]),
                    }
                    if args.metrics in {"lightweight", "all"}:
                        row.update(
                            _lightweight_metrics(
                                views[first],
                                views[second],
                                args.blur_radius,
                                args.image_size,
                            )
                        )
                    pair_rows.append(row)
                    pair_images.append((views[first], views[second]))

            if met3r is not None and pair_images:
                scores = met3r.evaluate(pair_images)
                for row, score in zip(pair_rows, scores):
                    row["met3r_score"] = score
                    row["met3r_score_direction"] = met3r.score_direction

            base_row.update(
                {
                    "status": "succeeded",
                    "num_views": len(views),
                    "azimuth_deg": angles,
                    **_summarize_pair_rows(pair_rows),
                }
            )
            if met3r is not None:
                base_row["met3r_score_direction"] = met3r.score_direction
            sample_rows.append(base_row)
            all_pair_rows.extend(pair_rows)
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

    aggregates = _aggregate_samples(sample_rows)
    settings = {
        "metrics": args.metrics,
        "image_size": args.image_size,
        "blur_radius": args.blur_radius,
        "opposite_tolerance": args.opposite_tolerance,
        "adjacent_wrap": args.adjacent_wrap,
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
        written = _write_results(output_path, sample_rows, all_pair_rows, aggregates, settings)
    except (OSError, ValueError) as error:
        parser.error("Could not write results: {}".format(error))
    print("Wrote: " + ", ".join(str(path) for path in written))
    if failures:
        print("{} samples failed; successful samples were still written.".format(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
