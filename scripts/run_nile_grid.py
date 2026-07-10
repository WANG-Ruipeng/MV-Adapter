"""Run a reproducible grid of MV-Adapter/NILE image-to-multiview jobs.

The script intentionally launches the inference entry point in a fresh process for
each configuration.  That is slower than keeping a pipeline resident, but it makes
individual jobs independently reproducible and makes interrupted experiments easy
to resume from the manifest.

Examples
--------
Run the distribution-preserving ablation on every PNG in ``inputs``::

    python -m scripts.run_nile_grid \
        --input "inputs/*.png" \
        --seeds 0 1 2 --strengths 0.15 0.30 0.45 0.60

Preview commands without launching inference::

    python -m scripts.run_nile_grid --input inputs/chair.png --dry-run

Resume an interrupted experiment::

    python -m scripts.run_nile_grid --input "inputs/*.png" --resume
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
import shlex
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_AZIMUTHS = [0, 45, 90, 180, 270, 315]
FORMAL_METHODS = [
    "iid_default",
    "iid_external",
    "shared_full",
    "spectral_global_corr",
    "camera_rbf_corr",
    "nested_tree_a",
    "nested_tree_ab",
]
DEFAULT_METHODS = [
    *FORMAL_METHODS,
]
DEFAULT_STRENGTHS = [0.15, 0.30, 0.45, 0.60]
METHOD_ALIASES: Mapping[str, Tuple[str, str]] = {
    # Formal distribution-preserving matrix.
    "iid_default": ("iid_default", "none"),
    "iid_external": ("iid_external", "none"),
    "shared_full": ("shared_full", "none"),
    "spectral_global_corr": ("spectral_global_corr", "none"),
    "camera_rbf_corr": ("camera_rbf_corr", "none"),
    "nested_tree_a": ("nested_tree_a", "none"),
    "nested_tree_ab": ("nested_tree_ab", "none"),
    # Legacy failure-analysis aliases. They remain runnable but are excluded
    # from DEFAULT_METHODS and must never be presented as formal NILE results.
    "iid": ("iid", "none"),
    "shared": ("shared", "none"),
    "lowpass_shared": ("lowpass_shared", "none"),
    "flat_sobol": ("flat_sobol", "none"),
    "nile_v": ("nile_v", "none"),
    "nile_vt": ("nile_v", "nile_vt"),
    "nile_vtp": ("nile_vtp", "nile_vtp"),
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "item"


def _float_token(value: float) -> str:
    text = format(value, ".8g")
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _flatten(values: Optional[Sequence[Sequence[str]]]) -> List[str]:
    if not values:
        return []
    return [item for group in values for item in group]


def _unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        marker = value if isinstance(value, (str, int, float, tuple)) else repr(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _expand_inputs(
    specifications: Sequence[str], recursive: bool, extensions: Sequence[str]
) -> List[Path]:
    """Expand files, directories, and shell-independent glob expressions."""

    allowed = {
        extension.lower() if extension.startswith(".") else "." + extension.lower()
        for extension in extensions
    }
    discovered: List[Path] = []
    unmatched: List[str] = []

    for specification in specifications:
        expanded = os.path.expandvars(os.path.expanduser(specification))
        matches: List[Path]
        if glob.has_magic(expanded):
            matches = [Path(item) for item in glob.glob(expanded, recursive=recursive)]
        else:
            path = Path(expanded)
            matches = [path] if path.exists() else []

        if not matches:
            unmatched.append(specification)
            continue

        for match in matches:
            if match.is_dir():
                iterator = match.rglob("*") if recursive else match.iterdir()
                discovered.extend(
                    item
                    for item in iterator
                    if item.is_file() and item.suffix.lower() in allowed
                )
            elif match.is_file() and match.suffix.lower() in allowed:
                discovered.append(match)

    unique: List[Path] = []
    seen = set()
    for path in sorted(discovered, key=lambda item: str(item).lower()):
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)

    if unmatched:
        print(
            "warning: no input matched: " + ", ".join(repr(item) for item in unmatched),
            file=sys.stderr,
        )
    if not unique:
        raise ValueError("No supported input images were found.")
    return unique


def _parse_method(specification: str) -> Tuple[str, str, str]:
    """Return ``(label, sampler mode, callback mode)`` for a method spec."""

    if specification in METHOD_ALIASES:
        mode, callback = METHOD_ALIASES[specification]
        return specification, mode, callback

    label = ""
    value = specification
    if "=" in value:
        label, value = value.split("=", 1)
    if ":" in value:
        mode, callback = value.split(":", 1)
    else:
        mode, callback = value, "none"

    valid_modes = {mode for mode, _ in METHOD_ALIASES.values()}
    valid_callbacks = {"none", "nile_vt", "nile_vtp"}
    if mode not in valid_modes:
        raise ValueError(
            "Unknown sampler mode {!r}; choose one of {}.".format(
                mode, ", ".join(sorted(valid_modes))
            )
        )
    if callback not in valid_callbacks:
        raise ValueError(
            "Unknown callback mode {!r}; choose one of {}.".format(
                callback, ", ".join(sorted(valid_callbacks))
            )
        )
    if callback == "nile_vtp" and mode != "nile_vtp":
        raise ValueError("The nile_vtp callback requires the nile_vtp sampler mode.")
    if mode in FORMAL_METHODS and callback != "none":
        raise ValueError("Formal distribution-preserving methods do not allow callbacks.")
    label = _slug(label or (mode if callback == "none" else "{}_{}".format(mode, callback)))
    return label, mode, callback


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _run_id(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()[:20]


def _detect_code_revision(repo_root: Path) -> str:
    """Return a resume-safe revision including tracked and untracked sources."""

    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(repo_root), "diff", "HEAD", "--binary"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"

    if status:
        digest = hashlib.sha256()
        digest.update(tracked_diff)
        untracked = sorted(
            entry[3:]
            for entry in status.split(b"\0")
            if entry.startswith(b"?? ") and entry[3:]
        )
        for relative_bytes in untracked:
            relative = Path(os.fsdecode(relative_bytes))
            candidate = repo_root / relative
            digest.update(b"\0untracked\0")
            digest.update(relative_bytes)
            digest.update(b"\0")
            try:
                candidate.relative_to(repo_root)
            except ValueError:
                raise ValueError(
                    "git reported an untracked path outside the repository: {}".format(
                        relative
                    )
                )
            try:
                metadata = candidate.lstat()
            except OSError as error:
                raise ValueError(
                    "could not fingerprint untracked path {}: {}".format(
                        relative, error
                    )
                )
            digest.update(str(metadata.st_mode).encode("ascii"))
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(candidate)
                digest.update(b"\0symlink\0")
                digest.update(os.fsencode(target))
                raise ValueError(
                    "automatic code revision refuses untracked symlink {}; "
                    "commit/remove it or pass --code-revision explicitly".format(
                        relative
                    )
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    "automatic code revision refuses non-regular untracked path {}; "
                    "commit/remove it or pass --code-revision explicitly".format(
                        relative
                    )
                )
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        revision = "{}+dirty.{}".format(revision, digest.hexdigest()[:12])
    return revision or "unknown"


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("runs", [])
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list or contain a 'runs' list.")
        return [dict(item) for item in payload]
    if suffix not in {".jsonl", ".ndjson"}:
        raise ValueError("Manifest extension must be .json, .jsonl/.ndjson, or .csv.")
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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_manifest(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "runs": list(records),
        }
        experiment_ids = _unique(
            str(record["experiment_id"])
            for record in records
            if record.get("experiment_id")
        )
        code_revisions = _unique(
            str(record["code_revision"])
            for record in records
            if record.get("code_revision")
        )
        if len(experiment_ids) == 1:
            payload["experiment_id"] = experiment_ids[0]
        if len(code_revisions) == 1:
            payload["code_revision"] = code_revisions[0]
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    elif suffix in {".jsonl", ".ndjson"}:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    elif suffix == ".csv":
        preferred = [
            "run_id",
            "experiment_id",
            "code_revision",
            "status",
            "input",
            "method",
            "inference_method",
            "nile_mode",
            "nile_callback",
            "seed",
            "max_correlation",
            "frequency_scale",
            "camera_length_scale",
            "rho_geo",
            "output",
            "metadata_path",
            "views_dir",
            "returncode",
            "elapsed_seconds",
            "started_at",
            "finished_at",
            "error",
            "command",
        ]
        fieldnames = list(preferred)
        extras = sorted({key for record in records for key in record if key not in fieldnames})
        fieldnames.extend(extras)
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({key: _csv_value(value) for key, value in record.items()})
    else:
        raise ValueError("Manifest extension must be .json, .jsonl/.ndjson, or .csv.")
    os.replace(str(temporary), str(path))


def _display_command(command: Sequence[str]) -> str:
    # shlex.join was added in Python 3.8, the minimum supported project version.
    return shlex.join(list(command))


def _add_optional(command: List[str], name: str, value: Optional[Any]) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable NILE ablation grid over image inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        nargs="+",
        required=True,
        metavar="PATH_OR_GLOB",
        help="Input image, directory, or glob. Repeat the option to add groups.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directories/globs.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(IMAGE_EXTENSIONS),
        help="Image extensions accepted when expanding directories.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        metavar="METHOD",
        help=(
            "Formal aliases or legacy label=mode:callback specs. Formal defaults: "
            + ", ".join(FORMAL_METHODS)
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=None,
        help="max_correlation sweep for correlated formal methods.",
    )
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=None,
        help="Legacy alias for --strengths/rho_geo sweeps.",
    )
    parser.add_argument(
        "--baseline-rho-once",
        dest="repeat_baseline_rhos",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repeat-baseline-strengths",
        "--repeat-baseline-rhos",
        dest="repeat_baseline_rhos",
        action="store_true",
        help="Intentionally repeat strength-independent baselines at every strength.",
    )
    parser.set_defaults(repeat_baseline_rhos=False)
    parser.add_argument("--text", default="high quality, detailed object")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--azimuth-deg", type=int, nargs="+", default=DEFAULT_AZIMUTHS)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--reference-conditioning-scale", type=float, default=None)
    parser.add_argument("--lora-scale", type=float, default=None)
    parser.add_argument("--remove-bg", action="store_true")

    parser.add_argument(
        "--rho-start",
        "--rho-starts",
        dest="rho_starts",
        type=float,
        nargs="+",
        default=[0.45],
        help="rho_start values to sweep for callback-enabled methods.",
    )
    parser.add_argument("--rho-end", type=float, default=0.0)
    parser.add_argument(
        "--active-ratio",
        "--active-ratios",
        dest="active_ratios",
        type=float,
        nargs="+",
        default=[0.6],
        help="Active callback ratios to sweep for callback-enabled methods.",
    )
    parser.add_argument("--blur-kernel", type=int, default=11)
    parser.add_argument("--blur-sigma", type=float, default=2.5)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--frequency-scale", type=float, default=0.12)
    parser.add_argument("--camera-length-scale", type=float, default=0.8)

    parser.add_argument(
        "--experiment-id",
        default=None,
        help=(
            "Experiment namespace stored in every record and run id. An ad-hoc "
            "namespace derived from the code revision is used when omitted."
        ),
    )
    parser.add_argument(
        "--code-revision",
        default=None,
        help=(
            "Exact source revision stored in every record and run id. Defaults to "
            "git HEAD plus a tracked/untracked worktree fingerprint."
        ),
    )
    parser.add_argument(
        "--base-model",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base-model identifier passed to inference and recorded in the manifest.",
    )
    parser.add_argument(
        "--base-model-revision",
        default=None,
        help="Requested base-model revision recorded for experiment provenance.",
    )
    parser.add_argument(
        "--vae-model",
        default="madebyollin/sdxl-vae-fp16-fix",
        help="VAE identifier passed to inference and recorded in the manifest.",
    )
    parser.add_argument(
        "--vae-model-revision",
        default=None,
        help="Requested VAE revision recorded for experiment provenance.",
    )
    parser.add_argument("--unet-model", default=None)
    parser.add_argument("--unet-model-revision", default=None)
    parser.add_argument("--lora-model", default=None)
    parser.add_argument("--lora-model-revision", default=None)
    parser.add_argument("--adapter-path", default="huanngzh/mv-adapter")
    parser.add_argument(
        "--adapter-revision",
        default=None,
        help="Requested adapter repository/checkpoint revision recorded in the manifest.",
    )
    parser.add_argument(
        "--adapter-sha256",
        default=None,
        help="SHA-256 of the resolved adapter weight recorded in the run configuration.",
    )
    parser.add_argument("--birefnet-model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--birefnet-revision", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--inference-arg",
        action="append",
        default=[],
        help=(
            "Append one raw token to the inference command; repeat for multiple tokens "
            "(for example --inference-arg=--foo --inference-arg=bar)."
        ),
    )

    parser.add_argument("--output-root", type=Path, default=Path("outputs/nile_grid"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run manifest (.json, .jsonl/.ndjson, or .csv). Defaults inside output-root.",
    )
    parser.add_argument(
        "--module",
        default="scripts.inference_i2mv_sdxl_nile",
        help="Python inference module launched for every run.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child jobs.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Working directory for inference subprocesses.",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Per-job timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Skip completed artifacts and continue the grid.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write planned records only.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed child process.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.experiment_id).strip():
        raise ValueError("--experiment-id must not be empty.")
    if not str(args.code_revision).strip():
        raise ValueError("--code-revision must not be empty.")
    if args.adapter_sha256 is not None:
        args.adapter_sha256 = str(args.adapter_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", args.adapter_sha256):
            raise ValueError("--adapter-sha256 must be exactly 64 hexadecimal characters.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if args.strengths is not None and args.rhos is not None:
        raise ValueError("Use either --strengths or legacy --rhos, not both.")
    using_legacy_rhos = args.strengths is None and args.rhos is not None
    strengths = args.strengths if args.strengths is not None else args.rhos
    args.strengths = list(DEFAULT_STRENGTHS if strengths is None else strengths)
    # Keep this attribute populated for callers that consumed the legacy parser.
    args.rhos = args.strengths
    if not args.strengths:
        raise ValueError("At least one correlation strength is required.")
    for value in args.strengths:
        valid = 0.0 <= value <= 1.0 if using_legacy_rhos else 0.0 <= value < 1.0
        if not valid:
            raise ValueError(
                "correlation values are outside the allowed interval, got {}.".format(
                    value
                )
            )
    if not 0.0 <= args.rho_end <= 1.0:
        raise ValueError("--rho-end must lie in [0, 1].")
    for value in args.rho_starts:
        if not 0.0 <= value <= 1.0:
            raise ValueError("--rho-start values must lie in [0, 1].")
    for value in args.active_ratios:
        if not 0.0 < value <= 1.0:
            raise ValueError("--active-ratio values must lie in (0, 1].")
    if args.blur_kernel < 1 or args.blur_kernel % 2 == 0:
        raise ValueError("--blur-kernel must be a positive odd integer.")
    if args.blur_sigma <= 0:
        raise ValueError("--blur-sigma must be positive.")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if not math.isfinite(args.frequency_scale) or args.frequency_scale <= 0:
        raise ValueError("--frequency-scale must be positive.")
    if not math.isfinite(args.camera_length_scale) or args.camera_length_scale <= 0:
        raise ValueError("--camera-length-scale must be positive.")
    if not args.azimuth_deg:
        raise ValueError("At least one azimuth is required.")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive.")


def _configuration(
    args: argparse.Namespace,
    image: Path,
    method: Tuple[str, str, str],
    seed: int,
    strength: float,
    rho_start: float,
    active_ratio: float,
) -> Dict[str, Any]:
    label, mode, callback = method
    return {
        "experiment_id": args.experiment_id,
        "code_revision": args.code_revision,
        "input": str(image),
        "method": label,
        "inference_method": mode,
        # Retained for consumers of the v1 manifest schema.
        "nile_mode": mode,
        "nile_callback": callback,
        "seed": seed,
        "max_correlation": strength,
        "frequency_scale": args.frequency_scale,
        "camera_length_scale": args.camera_length_scale,
        "rho_geo": strength,
        "rho_start": rho_start,
        "rho_end": args.rho_end,
        "active_ratio": active_ratio,
        "blur_kernel": args.blur_kernel,
        "blur_sigma": args.blur_sigma,
        "patch_size": args.patch_size,
        "text": args.text,
        "negative_prompt": args.negative_prompt,
        "azimuth_deg": list(args.azimuth_deg),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "reference_conditioning_scale": args.reference_conditioning_scale,
        "lora_scale": args.lora_scale,
        "remove_bg": bool(args.remove_bg),
        "base_model": args.base_model,
        "base_model_revision": args.base_model_revision,
        "vae_model": args.vae_model,
        "vae_model_revision": args.vae_model_revision,
        "unet_model": args.unet_model,
        "unet_model_revision": args.unet_model_revision,
        "lora_model": args.lora_model,
        "lora_model_revision": args.lora_model_revision,
        "adapter_path": args.adapter_path,
        "adapter_revision": args.adapter_revision,
        "adapter_sha256": args.adapter_sha256,
        "birefnet_model": args.birefnet_model,
        "birefnet_revision": args.birefnet_revision,
        "scheduler": args.scheduler,
        "device": args.device,
        "module": args.module,
        "inference_args": list(args.inference_arg),
    }


def _build_command(args: argparse.Namespace, config: Mapping[str, Any], output: Path) -> List[str]:
    command = [
        str(args.python),
        "-m",
        str(args.module),
        "--image",
        str(config["input"]),
        "--text",
        str(config["text"]),
        "--seed",
        str(config["seed"]),
        "--num_inference_steps",
        str(config["num_inference_steps"]),
        "--guidance_scale",
        str(config["guidance_scale"]),
        "--azimuth_deg",
        *[str(value) for value in config["azimuth_deg"]],
        "--method",
        str(config["inference_method"]),
        "--nile_callback",
        str(config["nile_callback"]),
        "--max_correlation",
        str(config["max_correlation"]),
        "--frequency_scale",
        str(config["frequency_scale"]),
        "--camera_length_scale",
        str(config["camera_length_scale"]),
        "--rho_geo",
        str(config["rho_geo"]),
        "--rho_start",
        str(config["rho_start"]),
        "--rho_end",
        str(config["rho_end"]),
        "--active_ratio",
        str(config["active_ratio"]),
        "--blur_kernel",
        str(config["blur_kernel"]),
        "--blur_sigma",
        str(config["blur_sigma"]),
        "--patch_size",
        str(config["patch_size"]),
        "--output",
        str(output),
    ]
    _add_optional(command, "--negative_prompt", config.get("negative_prompt"))
    _add_optional(command, "--reference_conditioning_scale", config.get("reference_conditioning_scale"))
    _add_optional(command, "--lora_scale", config.get("lora_scale"))
    _add_optional(command, "--base_model", config.get("base_model"))
    _add_optional(
        command, "--base_model_revision", config.get("base_model_revision")
    )
    _add_optional(command, "--vae_model", config.get("vae_model"))
    _add_optional(command, "--vae_model_revision", config.get("vae_model_revision"))
    _add_optional(command, "--unet_model", config.get("unet_model"))
    _add_optional(
        command, "--unet_model_revision", config.get("unet_model_revision")
    )
    _add_optional(command, "--lora_model", config.get("lora_model"))
    _add_optional(
        command, "--lora_model_revision", config.get("lora_model_revision")
    )
    _add_optional(command, "--adapter_path", config.get("adapter_path"))
    _add_optional(command, "--adapter_revision", config.get("adapter_revision"))
    _add_optional(command, "--birefnet_model", config.get("birefnet_model"))
    _add_optional(
        command, "--birefnet_revision", config.get("birefnet_revision")
    )
    _add_optional(command, "--scheduler", config.get("scheduler"))
    _add_optional(command, "--device", config.get("device"))
    if config.get("remove_bg"):
        command.append("--remove_bg")
    command.extend(str(item) for item in config.get("inference_args", []))
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        parser.error("--repo-root is not a directory: {}".format(repo_root))
    try:
        args.code_revision = (
            str(args.code_revision).strip()
            if args.code_revision is not None
            else _detect_code_revision(repo_root)
        )
    except ValueError as error:
        parser.error(str(error))
    args.experiment_id = (
        str(args.experiment_id).strip()
        if args.experiment_id is not None
        else "adhoc-{}".format(_slug(args.code_revision)[:40])
    )
    try:
        _validate_args(args)
        images = _expand_inputs(_flatten(args.input), args.recursive, args.extensions)
        methods = _unique(_parse_method(item) for item in args.methods)
        args.seeds = _unique(args.seeds)
        args.strengths = _unique(args.strengths)
        args.rhos = args.strengths
        args.rho_starts = _unique(args.rho_starts)
        args.active_ratios = _unique(args.active_ratios)
    except ValueError as error:
        parser.error(str(error))

    output_root = args.output_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_root / "manifest.jsonl"
    )
    try:
        previous = _read_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error("Could not read manifest {}: {}".format(manifest_path, error))

    foreign_records = [
        record
        for record in previous
        if record.get("experiment_id") != args.experiment_id
        or record.get("code_revision") != args.code_revision
    ]
    if foreign_records:
        parser.error(
            "Manifest {} contains {} record(s) from a different experiment or "
            "code revision; use a new isolated manifest path.".format(
                manifest_path, len(foreign_records)
            )
        )

    records_by_id: MutableMapping[str, Dict[str, Any]] = {
        str(record.get("run_id")): record
        for record in previous
        if record.get("run_id")
    }
    ordered_ids = _unique(
        str(record["run_id"]) for record in previous if record.get("run_id")
    )

    planned: List[Tuple[Dict[str, Any], Path, List[str]]] = []
    for image in images:
        image_digest = hashlib.sha1(str(image).encode("utf-8")).hexdigest()[:8]
        image_label = "{}-{}".format(_slug(image.stem), image_digest)
        for method in methods:
            label, mode, callback = method
            method_strengths = list(args.strengths)
            strength_independent = {
                "iid_default",
                "iid_external",
                "shared_full",
                "iid",
                "shared",
                "flat_sobol",
            }
            if not args.repeat_baseline_rhos and mode in strength_independent:
                # Correlation strength is not a parameter of these baselines;
                # record zero rather than a misleading member of the sweep.
                method_strengths = [0.0]
            method_rho_starts = args.rho_starts if callback != "none" else args.rho_starts[:1]
            method_active_ratios = (
                args.active_ratios if callback != "none" else args.active_ratios[:1]
            )
            for seed in args.seeds:
                for strength in method_strengths:
                    for rho_start in method_rho_starts:
                        for active_ratio in method_active_ratios:
                            config = _configuration(
                                args,
                                image,
                                method,
                                seed,
                                strength,
                                rho_start,
                                active_ratio,
                            )
                            run_id = _run_id(config)
                            output = (
                                output_root
                                / image_label
                                / _slug(label)
                                / "seed_{:06d}".format(seed)
                                / "strength_{}__{}.png".format(
                                    _float_token(strength), run_id
                                )
                            )
                            command = _build_command(args, config, output)
                            stem = output.with_suffix("")
                            record = dict(config)
                            record.update(
                                {
                                    "schema_version": 2,
                                    "run_id": run_id,
                                    "output": str(output),
                                    "metadata_path": str(stem) + "_metadata.json",
                                    "views_dir": str(stem) + "_views",
                                    "reference_output": str(stem) + "_reference.png",
                                    "command": command,
                                    "status": "planned",
                                    "created_at": records_by_id.get(run_id, {}).get(
                                        "created_at", _utc_now()
                                    ),
                                }
                            )
                            planned.append((record, output, command))

    total = len(planned)
    print(
        "Planned {} jobs from {} inputs, {} methods, {} seeds.".format(
            total, len(images), len(methods), len(args.seeds)
        )
    )
    failures = 0
    completed = 0
    skipped = 0

    for index, (record, output, command) in enumerate(planned, 1):
        run_id = str(record["run_id"])
        old_record = records_by_id.get(run_id, {})
        if run_id not in ordered_ids:
            ordered_ids.append(run_id)

        print("[{}/{}] {}".format(index, total, _display_command(command)), flush=True)
        artifact_exists = output.is_file() and output.stat().st_size > 0
        metadata_path = Path(str(record["metadata_path"]))
        metadata_exists = metadata_path.is_file() and metadata_path.stat().st_size > 0
        previously_complete = old_record.get("status") in {"succeeded", "skipped"}
        if args.resume and artifact_exists and (previously_complete or metadata_exists):
            record.update(old_record)
            record.update(
                {
                    "status": "skipped",
                    "skip_reason": (
                        "completed manifest record and output exist"
                        if previously_complete
                        else "output and inference metadata already exist"
                    ),
                    "last_checked_at": _utc_now(),
                }
            )
            records_by_id[run_id] = record
            skipped += 1
            _write_manifest(manifest_path, [records_by_id[item] for item in ordered_ids])
            continue

        if args.dry_run:
            # Preserve a prior successful record when merely previewing the same grid.
            if previously_complete and artifact_exists:
                records_by_id[run_id] = old_record
            else:
                record.update({"status": "dry_run", "finished_at": _utc_now()})
                records_by_id[run_id] = record
            _write_manifest(manifest_path, [records_by_id[item] for item in ordered_ids])
            continue

        output.parent.mkdir(parents=True, exist_ok=True)
        record.update({"status": "running", "started_at": _utc_now()})
        records_by_id[run_id] = record
        _write_manifest(manifest_path, [records_by_id[item] for item in ordered_ids])

        start = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=str(repo_root),
                check=False,
                timeout=args.timeout,
            )
            returncode = result.returncode
            error_message = (
                None
                if returncode == 0
                else "Inference exited with return code {}".format(returncode)
            )
        except subprocess.TimeoutExpired as error:
            returncode = None
            error_message = "Timed out after {} seconds".format(error.timeout)
        except OSError as error:
            returncode = None
            error_message = "Could not launch inference: {}".format(error)

        elapsed = time.monotonic() - start
        success = returncode == 0 and output.is_file() and output.stat().st_size > 0
        if returncode == 0 and not success:
            error_message = "Inference exited successfully but did not create a non-empty output."
        record.update(
            {
                "status": "succeeded" if success else "failed",
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
                "finished_at": _utc_now(),
                "error": error_message,
            }
        )
        records_by_id[run_id] = record
        _write_manifest(manifest_path, [records_by_id[item] for item in ordered_ids])

        if success:
            completed += 1
        else:
            failures += 1
            print(
                "job failed (run_id={}, returncode={}): {}".format(
                    run_id, returncode, error_message or "see inference output above"
                ),
                file=sys.stderr,
            )
            if args.fail_fast:
                break

    print(
        "Finished: {} succeeded, {} skipped, {} failed. Manifest: {}".format(
            completed, skipped, failures, manifest_path
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
