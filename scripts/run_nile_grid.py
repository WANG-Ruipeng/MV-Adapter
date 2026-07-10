"""Run a reproducible grid of MV-Adapter/NILE image-to-multiview jobs.

The script intentionally launches the inference entry point in a fresh process for
each configuration.  That is slower than keeping a pipeline resident, but it makes
individual jobs independently reproducible and makes interrupted experiments easy
to resume from the manifest.

Examples
--------
Run the minimum ablation on every PNG in ``inputs``::

    python -m scripts.run_nile_grid \
        --input "inputs/*.png" \
        --methods iid shared lowpass_shared flat_sobol nile_v \
        --seeds 0 1 2 3 4 --rhos 0.0 0.25 0.5 0.65 0.8 1.0

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
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_AZIMUTHS = [0, 45, 90, 180, 270, 315]
DEFAULT_METHODS = [
    "iid",
    "shared",
    "lowpass_shared",
    "flat_sobol",
    "nile_v",
    "nile_vt",
    "nile_vtp",
]
METHOD_ALIASES: Mapping[str, Tuple[str, str]] = {
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

    valid_modes = {"iid", "shared", "lowpass_shared", "flat_sobol", "nile_v", "nile_vtp"}
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
    label = _slug(label or (mode if callback == "none" else "{}_{}".format(mode, callback)))
    return label, mode, callback


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _run_id(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(configuration).encode("utf-8")).hexdigest()[:20]


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
            "status",
            "input",
            "method",
            "nile_mode",
            "nile_callback",
            "seed",
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
            "Aliases or custom label=mode:callback specs. Built-ins: "
            + ", ".join(DEFAULT_METHODS)
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.65, 0.8, 1.0],
        help="rho_geo sweep values.",
    )
    parser.add_argument(
        "--baseline-rho-once",
        dest="repeat_baseline_rhos",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repeat-baseline-rhos",
        dest="repeat_baseline_rhos",
        action="store_true",
        help="Intentionally repeat rho-independent baselines for every rho.",
    )
    parser.set_defaults(repeat_baseline_rhos=False)
    parser.add_argument("--text", default="high quality, detailed object")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--azimuth-deg", type=int, nargs="+", default=DEFAULT_AZIMUTHS)
    parser.add_argument("--num-inference-steps", type=int, default=50)
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

    parser.add_argument("--base-model", default=None, help="Override inference-script default.")
    parser.add_argument("--vae-model", default=None, help="Override inference-script default.")
    parser.add_argument("--unet-model", default=None)
    parser.add_argument("--lora-model", default=None)
    parser.add_argument("--adapter-path", default=None)
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
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if not args.rhos:
        raise ValueError("At least one rho value is required.")
    for name in ("rhos",):
        for value in getattr(args, name):
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} values must lie in [0, 1], got {}.".format(name, value))
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
    if not args.azimuth_deg:
        raise ValueError("At least one azimuth is required.")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive.")


def _configuration(
    args: argparse.Namespace,
    image: Path,
    method: Tuple[str, str, str],
    seed: int,
    rho: float,
    rho_start: float,
    active_ratio: float,
) -> Dict[str, Any]:
    label, mode, callback = method
    return {
        "input": str(image),
        "method": label,
        "nile_mode": mode,
        "nile_callback": callback,
        "seed": seed,
        "rho_geo": rho,
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
        "vae_model": args.vae_model,
        "unet_model": args.unet_model,
        "lora_model": args.lora_model,
        "adapter_path": args.adapter_path,
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
        "--nile_mode",
        str(config["nile_mode"]),
        "--nile_callback",
        str(config["nile_callback"]),
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
    _add_optional(command, "--vae_model", config.get("vae_model"))
    _add_optional(command, "--unet_model", config.get("unet_model"))
    _add_optional(command, "--lora_model", config.get("lora_model"))
    _add_optional(command, "--adapter_path", config.get("adapter_path"))
    _add_optional(command, "--scheduler", config.get("scheduler"))
    _add_optional(command, "--device", config.get("device"))
    if config.get("remove_bg"):
        command.append("--remove_bg")
    command.extend(str(item) for item in config.get("inference_args", []))
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        images = _expand_inputs(_flatten(args.input), args.recursive, args.extensions)
        methods = _unique(_parse_method(item) for item in args.methods)
        args.seeds = _unique(args.seeds)
        args.rhos = _unique(args.rhos)
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
    repo_root = args.repo_root.expanduser().resolve()
    if not repo_root.is_dir():
        parser.error("--repo-root is not a directory: {}".format(repo_root))

    try:
        previous = _read_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error("Could not read manifest {}: {}".format(manifest_path, error))

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
            method_rhos = list(args.rhos)
            if not args.repeat_baseline_rhos and mode in {"iid", "shared", "flat_sobol"}:
                method_rhos = [method_rhos[0]]
            method_rho_starts = args.rho_starts if callback != "none" else args.rho_starts[:1]
            method_active_ratios = (
                args.active_ratios if callback != "none" else args.active_ratios[:1]
            )
            for seed in args.seeds:
                for rho in method_rhos:
                    for rho_start in method_rho_starts:
                        for active_ratio in method_active_ratios:
                            config = _configuration(
                                args,
                                image,
                                method,
                                seed,
                                rho,
                                rho_start,
                                active_ratio,
                            )
                            run_id = _run_id(config)
                            output = (
                                output_root
                                / image_label
                                / _slug(label)
                                / "seed_{:06d}".format(seed)
                                / "rho_{}__{}.png".format(_float_token(rho), run_id)
                            )
                            command = _build_command(args, config, output)
                            stem = output.with_suffix("")
                            record = dict(config)
                            record.update(
                                {
                                    "schema_version": 1,
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
