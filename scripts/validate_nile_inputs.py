"""Validate and deterministically split NILE low-rank study inputs.

Formal samples are content-addressed. Exact duplicate content is never counted
twice, and PILOT/FULL membership is a stable SHA-256 ordering rather than a
manual selection. This module does not download or synthesize inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_PILOT_COUNT = 5
DEFAULT_FULL_COUNT = 20
DEFAULT_MIN_DISTINCT_INPUTS = 25


@dataclass(frozen=True)
class InputRecord:
    path: str
    sha256: str
    width: int
    height: int
    mode: str
    perceptual_key: str
    split: str = "unused"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(image: Image.Image, size: int = 16) -> str:
    grayscale = ImageOps.grayscale(image).resize(
        (size + 1, size), Image.Resampling.LANCZOS
    )
    flattened = getattr(grayscale, "get_flattened_data", None)
    pixels = list(flattened() if flattened is not None else grayscale.getdata())
    bits = []
    for row in range(size):
        offset = row * (size + 1)
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1]
            for column in range(size)
        )
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return "{:0{}x}".format(value, size * size // 4)


def rotation_invariant_perceptual_key(image: Image.Image) -> str:
    """Return a conservative key that catches exact rotations and recoloring.

    Grayscale dHash is intentionally a guardrail, not a general near-duplicate
    classifier. A collision is reported and excluded from formal counts so that
    rotations or simple color changes cannot be used to pad the dataset.
    """

    normalized = ImageOps.exif_transpose(image).convert("RGB")
    hashes = []
    rotated = normalized
    for _ in range(4):
        hashes.append(_difference_hash(rotated))
        rotated = rotated.transpose(Image.Transpose.ROTATE_90)
    return min(hashes)


def discover_images(directory: Path, recursive: bool = True) -> List[Path]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        return []
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_input_directory(
    explicit: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
    drive_directory: Optional[Path] = None,
) -> Optional[Path]:
    candidates: List[Optional[Path]] = [explicit]
    environment = os.environ.get("NILE_INPUT_DIR")
    candidates.append(Path(environment) if environment else None)
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    candidates.append(root / "inputs" / "formal")
    candidates.append(drive_directory)
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_dir():
            return candidate.expanduser().resolve()
    return None


def inspect_inputs(paths: Sequence[Path]) -> Tuple[List[InputRecord], List[Dict[str, str]]]:
    by_sha: Dict[str, InputRecord] = {}
    by_perceptual: Dict[str, InputRecord] = {}
    rejected: List[Dict[str, str]] = []
    for path in paths:
        try:
            digest = sha256_file(path)
            if digest in by_sha:
                rejected.append(
                    {
                        "path": str(path),
                        "reason": "duplicate_sha256",
                        "duplicate_of": by_sha[digest].path,
                    }
                )
                continue
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                perceptual = rotation_invariant_perceptual_key(image)
            if perceptual in by_perceptual:
                rejected.append(
                    {
                        "path": str(path),
                        "reason": "perceptual_or_rotation_duplicate",
                        "duplicate_of": by_perceptual[perceptual].path,
                    }
                )
                continue
            record = InputRecord(
                path=str(path),
                sha256=digest,
                width=width,
                height=height,
                mode=mode,
                perceptual_key=perceptual,
            )
            by_sha[digest] = record
            by_perceptual[perceptual] = record
        except Exception as error:  # corrupt/unsupported files are explicit records
            rejected.append(
                {"path": str(path), "reason": "unreadable", "error": repr(error)}
            )
    return sorted(by_sha.values(), key=lambda item: item.sha256), rejected


def stable_split(
    records: Sequence[InputRecord], pilot_count: int, full_count: int
) -> List[InputRecord]:
    if pilot_count < 0 or full_count < 0:
        raise ValueError("pilot_count and full_count must be non-negative")
    ordered = sorted(records, key=lambda item: item.sha256)
    result = []
    for index, record in enumerate(ordered):
        if index < pilot_count:
            split = "pilot"
        elif index < pilot_count + full_count:
            split = "full"
        else:
            split = "unused"
        result.append(InputRecord(**{**asdict(record), "split": split}))
    return result


def write_manifest(path: Path, records: Sequence[InputRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(InputRecord.__dataclass_fields__)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    os.replace(temporary, path)


def write_contact_sheet(
    path: Path,
    records: Sequence[InputRecord],
    *,
    cell_size: int = 192,
    columns: int = 5,
) -> None:
    if not records:
        return
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + 24)), "white")
    for index, record in enumerate(records):
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((cell_size - 8, cell_size - 8), Image.Resampling.LANCZOS)
            x = (index % columns) * cell_size + (cell_size - image.width) // 2
            y = (index // columns) * (cell_size + 24) + (cell_size - image.height) // 2
            sheet.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def validate_input_directory(
    input_directory: Path,
    output_directory: Path,
    *,
    pilot_count: int = DEFAULT_PILOT_COUNT,
    full_count: int = DEFAULT_FULL_COUNT,
    min_distinct_inputs: int = DEFAULT_MIN_DISTINCT_INPUTS,
) -> Dict[str, object]:
    paths = discover_images(input_directory)
    unique, rejected = inspect_inputs(paths)
    split = stable_split(unique, pilot_count, full_count)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_manifest(output_directory / "input_manifest.csv", split)
    write_contact_sheet(output_directory / "input_contact_sheet.jpg", split)
    actual_pilot = sum(item.split == "pilot" for item in split)
    actual_full = sum(item.split == "full" for item in split)
    payload: Dict[str, object] = {
        "schema_version": 1,
        "input_directory": str(input_directory.resolve()),
        "discovered_count": len(paths),
        "distinct_count": len(split),
        "pilot_count": actual_pilot,
        "full_count": actual_full,
        "required_pilot_count": pilot_count,
        "required_full_count": full_count,
        "min_distinct_inputs": min_distinct_inputs,
        "missing_distinct_inputs": max(0, min_distinct_inputs - len(split)),
        "formal_ready": (
            len(split) >= min_distinct_inputs
            and actual_pilot == pilot_count
            and actual_full == full_count
        ),
        "records": [asdict(item) for item in split],
        "rejected": rejected,
        "policy": {
            "ordering": "sha256_ascending",
            "pilot_full_disjoint": True,
            "synthetic_or_downloaded_inputs": False,
            "perceptual_rotation_duplicates_excluded": True,
        },
    }
    _atomic_write_text(
        output_directory / "input_validation.json",
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--drive-input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-count", type=int, default=DEFAULT_PILOT_COUNT)
    parser.add_argument("--full-count", type=int, default=DEFAULT_FULL_COUNT)
    parser.add_argument(
        "--min-distinct-inputs", type=int, default=DEFAULT_MIN_DISTINCT_INPUTS
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    directory = resolve_input_directory(
        args.input_dir, drive_directory=args.drive_input_dir
    )
    if directory is None:
        payload = {
            "schema_version": 1,
            "formal_ready": False,
            "distinct_count": 0,
            "missing_distinct_inputs": args.min_distinct_inputs,
            "blocker": "no_input_directory",
        }
        _atomic_write_text(
            args.output_dir / "input_validation.json",
            json.dumps(payload, indent=2) + "\n",
        )
        print(json.dumps(payload, indent=2))
        return 2 if args.strict else 0
    payload = validate_input_directory(
        directory,
        args.output_dir,
        pilot_count=args.pilot_count,
        full_count=args.full_count,
        min_distinct_inputs=args.min_distinct_inputs,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))
    return 2 if args.strict and not payload["formal_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
