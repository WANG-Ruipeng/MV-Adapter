"""Persistent GPU worker for a prepared NILE low-rank inference plan.

The worker loads MV-Adapter and BiRefNet once, executes records sequentially,
and emits durable JSONL lifecycle events. It deliberately owns no manifest:
the study runner remains the source of truth and can resume after a worker
crash by auditing completed artifact bundles and scheduling the remainder.

Plan schema::

    {
      "schema_version": 1,
      "resolved_config": {...},
      "records": [{... "distribution_gate_passed": true}, ...]
    }

For compatibility with a runner manifest, ``runs`` is accepted as an alias
for ``records`` and ``config`` as an alias for ``resolved_config``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import traceback
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


LOWRANK_METHODS = (
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
FORMAL_METHODS = ("iid_external", "shared_full") + LOWRANK_METHODS


def install_timm_layers_compatibility() -> str:
    """Expose the modern timm.layers path for legacy timm releases.

    The pinned official MEt3R dependency installs timm 0.4.12, while the
    frozen BiRefNet remote module imports the same symbols from timm.layers.
    Legacy timm exposes them as timm.models.layers. Alias only that missing
    module path and preserve every other import error.
    """

    try:
        importlib.import_module("timm.layers")
    except ModuleNotFoundError as error:
        if error.name != "timm.layers":
            raise
        legacy_layers = importlib.import_module("timm.models.layers")
        timm_module = importlib.import_module("timm")
        sys.modules["timm.layers"] = legacy_layers
        setattr(timm_module, "layers", legacy_layers)
        return "legacy_timm.models.layers_alias"
    return "native_timm.layers"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("{} must be a JSON object".format(name))
    return value


@dataclass(frozen=True)
class WorkerPlan:
    resolved_config: Dict[str, Any]
    records: Tuple[Dict[str, Any], ...]
    source: str


def load_worker_plan(path: Path) -> WorkerPlan:
    """Load and strictly validate a worker plan before any model is loaded."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "plan")
    config_value = root.get("resolved_config", root.get("config"))
    config = dict(_require_mapping(config_value, "resolved_config"))
    model = _require_mapping(config.get("model"), "resolved_config.model")
    views = model.get("views_deg")
    if not isinstance(views, list) or len(views) == 0:
        raise ValueError("resolved_config.model.views_deg must be a non-empty list")

    records_value = root.get("records", root.get("runs"))
    if not isinstance(records_value, list) or not records_value:
        raise ValueError("plan.records must be a non-empty list")
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    required_fields = (
        "run_id",
        "input_path",
        "seed",
        "method",
        "output",
        "metadata_path",
        "views_dir",
        "mask_dir",
    )
    for index, raw_record in enumerate(records_value):
        record = dict(_require_mapping(raw_record, "records[{}]".format(index)))
        missing = [field for field in required_fields if record.get(field) in (None, "")]
        if missing:
            raise ValueError(
                "records[{}] is missing required fields: {}".format(index, missing)
            )
        run_id = str(record["run_id"])
        if run_id in seen_ids:
            raise ValueError("duplicate run_id in worker plan: {}".format(run_id))
        seen_ids.add(run_id)
        if record.get("distribution_gate_passed") is not True:
            raise ValueError(
                "run {} rejected: distribution_gate_passed must be true".format(
                    run_id
                )
            )
        method = str(record["method"])
        if method not in FORMAL_METHODS:
            raise ValueError("run {} has unsupported method {}".format(run_id, method))
        if method in LOWRANK_METHODS:
            if record.get("rank") is None or record.get("target_kl") is None:
                raise ValueError(
                    "run {} low-rank record requires rank and target_kl".format(
                        run_id
                    )
                )
            rank = int(record["rank"])
            target_kl = float(record["target_kl"])
            if rank <= 0 or not math.isfinite(target_kl) or target_kl < 0.0:
                raise ValueError(
                    "run {} has invalid rank or target_kl".format(run_id)
                )
            if method == "lowrank_camera_rbf":
                ell_deg = float(record.get("rbf_length_scale_deg", 0.0))
                if not math.isfinite(ell_deg) or ell_deg <= 0.0:
                    raise ValueError(
                        "run {} camera RBF requires positive length scale".format(
                            run_id
                        )
                    )
        cameras = record.get("camera_list")
        if cameras is not None and [float(item) for item in cameras] != [
            float(item) for item in views
        ]:
            raise ValueError("run {} camera_list differs from resolved config".format(run_id))
        records.append(record)
    return WorkerPlan(config, tuple(records), str(path))


class JsonlEventWriter:
    """Append one flushed and fsynced machine-readable JSON object per event."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self._sequence = max(
                        self._sequence, int(json.loads(line).get("sequence", 0))
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    # Preserve an append-only forensic log even if an earlier
                    # process crashed after a partial/nonconforming line.
                    continue
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def emit(self, event: str, **payload: Any) -> Dict[str, Any]:
        self._sequence += 1
        record = {
            "schema_version": 1,
            "sequence": self._sequence,
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "event": str(event),
            **payload,
        }
        self._handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def __enter__(self) -> "JsonlEventWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _artifact_path(value: Any, base: Path) -> Optional[Path]:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value_matches(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        )
    return left == right


def audit_artifact_bundle(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate grid, metadata, views, masks, reference, and trajectory."""

    issues: List[str] = []
    output = Path(str(record["output"])).expanduser()
    metadata_path = Path(str(record["metadata_path"])).expanduser()
    expected_views = len(config.get("model", {}).get("views_deg", []))
    if not output.is_file() or output.stat().st_size <= 0:
        issues.append("grid_missing_or_empty")
    metadata: Dict[str, Any] = {}
    if not metadata_path.is_file() or metadata_path.stat().st_size <= 0:
        issues.append("metadata_missing_or_empty")
    else:
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("metadata root is not an object")
            metadata = loaded
        except Exception as error:
            issues.append("metadata_invalid:{!r}".format(error))
    base = metadata_path.parent
    model = config.get("model", {})
    metadata_output = _artifact_path(metadata.get("output"), base)
    if metadata_output is None or metadata_output.resolve() != output.resolve():
        issues.append("output_metadata_mismatch")
    if metadata.get("config_id") != record.get("config_id"):
        issues.append("config_id_metadata_mismatch")
    try:
        if int(metadata.get("seed")) != int(record.get("seed")):
            issues.append("seed_metadata_mismatch")
    except (TypeError, ValueError):
        issues.append("seed_metadata_mismatch")
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, Mapping):
        input_metadata = {}
        issues.append("input_metadata_missing")
    expected_input = Path(str(record.get("input_path", ""))).expanduser()
    observed_input = _artifact_path(input_metadata.get("image"), base)
    if (
        observed_input is None
        or observed_input.resolve() != expected_input.resolve()
    ):
        issues.append("input_path_metadata_mismatch")
    expected_input_sha = str(record.get("input_sha256") or "").lower()
    if expected_input_sha:
        if str(input_metadata.get("sha256") or "").lower() != expected_input_sha:
            issues.append("input_sha256_metadata_mismatch")
        if not expected_input.is_file():
            issues.append("input_file_missing_during_audit")
        elif _sha256_file(expected_input).lower() != expected_input_sha:
            issues.append("input_sha256_file_mismatch")
    inference = metadata.get("inference")
    if not isinstance(inference, Mapping):
        inference = {}
        issues.append("inference_metadata_missing")
    if inference.get("num_inference_steps") != int(
        model.get("num_inference_steps", 30)
    ):
        issues.append("steps_metadata_mismatch")
    if not _metadata_value_matches(
        inference.get("guidance_scale"), model.get("guidance_scale", 3.0)
    ):
        issues.append("guidance_scale_metadata_mismatch")
    models_metadata = metadata.get("models")
    if not isinstance(models_metadata, Mapping):
        models_metadata = {}
        issues.append("models_metadata_missing")
    if models_metadata.get("scheduler") != model.get("scheduler"):
        issues.append("scheduler_metadata_mismatch")
    for field in (
        "base_model_revision",
        "vae_model_revision",
        "unet_model_revision",
        "lora_model_revision",
        "adapter_revision",
        "birefnet_revision",
    ):
        if models_metadata.get(field) != model.get(field):
            issues.append("{}_metadata_mismatch".format(field))
    if models_metadata.get("mv_adapter_checkpoint") != model.get(
        "mv_adapter_checkpoint"
    ):
        issues.append("mv_adapter_checkpoint_metadata_mismatch")
    view_paths = [_artifact_path(item, base) for item in metadata.get("view_files", [])]
    mask_paths = [_artifact_path(item, base) for item in metadata.get("mask_files", [])]
    reference = _artifact_path(metadata.get("reference_output"), base)
    if expected_views <= 0 or len(view_paths) != expected_views:
        issues.append("view_count_mismatch")
    if expected_views <= 0 or len(mask_paths) != expected_views:
        issues.append("mask_count_mismatch")
    if any(
        path is None or not path.is_file() or path.stat().st_size <= 0
        for path in view_paths
    ):
        issues.append("view_artifact_missing_or_empty")
    if any(
        path is None or not path.is_file() or path.stat().st_size <= 0
        for path in mask_paths
    ):
        issues.append("mask_artifact_missing_or_empty")
    if reference is None or not reference.is_file() or reference.stat().st_size <= 0:
        issues.append("reference_missing_or_empty")
    metadata_angles = metadata.get("azimuth_deg")
    expected_angles = [float(item) for item in config.get("model", {}).get("views_deg", [])]
    if metadata_angles is None or [float(item) for item in metadata_angles] != expected_angles:
        issues.append("camera_list_mismatch")
    distribution = metadata.get("distribution")
    if not isinstance(distribution, Mapping):
        distribution = {}
        issues.append("distribution_metadata_missing")
    if distribution.get("method") != record.get("method"):
        issues.append("method_metadata_mismatch")
    if record.get("method") in LOWRANK_METHODS:
        if int(distribution.get("basis_rank", -1)) != int(record.get("rank", -2)):
            issues.append("rank_metadata_mismatch")
        for field in ("achieved_kl", "alpha", "basis_checksum", "covariance_checksum"):
            if distribution.get(field) in (None, ""):
                issues.append("{}_missing".format(field))
        if not _metadata_value_matches(
            distribution.get("target_joint_kl"), record.get("target_kl")
        ):
            issues.append("target_kl_metadata_mismatch")
    trajectory = record.get("trajectory_output")
    if trajectory:
        trajectory_path = Path(str(trajectory)).expanduser()
        if not trajectory_path.is_file() or trajectory_path.stat().st_size <= 0:
            issues.append("trajectory_missing_or_empty")
    return {
        "complete": not issues,
        "issues": sorted(set(issues)),
        "grid": str(output),
        "metadata": str(metadata_path),
        "expected_view_count": expected_views,
        "view_count": len(view_paths),
        "mask_count": len(mask_paths),
    }


class DefaultInferenceBackend:
    """Lazy adapter over the existing inference module's public helpers."""

    def __init__(self) -> None:
        self.module = importlib.import_module("scripts.inference_i2mv_sdxl_nile")
        self.torch = self.module.torch

    def prepare_pipeline(
        self, model: Mapping[str, Any], *, num_views: int
    ) -> Any:
        device = str(model.get("device", "cuda"))
        dtype = self.torch.float16 if device.startswith("cuda") else self.torch.float32
        return self.module.prepare_pipeline(
            base_model=model["base_model"],
            vae_model=model.get("vae_model"),
            unet_model=model.get("unet_model"),
            lora_model=model.get("lora_model"),
            adapter_path=model["adapter_path"],
            mv_adapter_checkpoint=model.get(
                "mv_adapter_checkpoint", "mvadapter_i2mv_sdxl.safetensors"
            ),
            scheduler=model.get("scheduler"),
            num_views=num_views,
            device=device,
            dtype=dtype,
            base_model_revision=model.get("base_model_revision"),
            vae_model_revision=model.get("vae_model_revision"),
            unet_model_revision=model.get("unet_model_revision"),
            lora_model_revision=model.get("lora_model_revision"),
            adapter_revision=model.get("adapter_revision"),
        )

    def prepare_segmentation(
        self, model: Mapping[str, Any]
    ) -> Tuple[Optional[Any], Any]:
        device = str(model.get("device", "cuda"))
        install_timm_layers_compatibility()
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if model.get("birefnet_revision") is not None:
            kwargs["revision"] = model["birefnet_revision"]
        network = self.module.AutoModelForImageSegmentation.from_pretrained(
            model.get("birefnet_model", "ZhengPeng7/BiRefNet"), **kwargs
        )
        network.to(device)
        network.eval()
        transform = self.module.transforms.Compose(
            [
                self.module.transforms.Resize((1024, 1024)),
                self.module.transforms.ToTensor(),
                self.module.transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        def segment(image: Any) -> Any:
            return self.module.remove_bg(image, network, transform, device)

        def mask(image: Any) -> Any:
            return segment(image).getchannel("A")

        return segment if model.get("remove_background", True) else None, mask

    def run_pipeline(self, **kwargs: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        return self.module.run_pipeline(**kwargs)

    def save_outputs(
        self, images: Any, reference_image: Any, args: Namespace, mask_fn: Any
    ) -> Tuple[Path, Path, Optional[Path], Path]:
        return self.module._save_outputs(
            images, reference_image, args, mask_fn=mask_fn
        )

    def is_oom(self, error: BaseException) -> bool:
        out_of_memory = getattr(self.torch.cuda, "OutOfMemoryError", ())
        return (
            bool(out_of_memory) and isinstance(error, out_of_memory)
        ) or "out of memory" in str(error).lower()

    def clear_cuda_cache(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _save_namespace(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> Namespace:
    """Create the namespace consumed by the existing ``_save_outputs``."""

    model = config["model"]
    method = str(record["method"])
    trajectory = record.get("trajectory_output")
    observer_rank = record.get("rank") if trajectory else None
    basis_rank = (
        int(record["rank"])
        if method in LOWRANK_METHODS or observer_rank is not None
        else 8
    )
    target_kl = float(record["target_kl"]) if method in LOWRANK_METHODS else 1.0
    ell_deg = (
        float(record["rbf_length_scale_deg"])
        if record.get("rbf_length_scale_deg") is not None
        else 90.0
    )
    return Namespace(
        azimuth_deg=[int(item) for item in model["views_deg"]],
        output=str(record["output"]),
        views_dir=str(record["views_dir"]),
        mask_dir=str(record["mask_dir"]),
        save_views=True,
        save_masks=True,
        seed=int(record["seed"]),
        resolved_method=method,
        method=method,
        nile_mode=None,
        nile_callback="none",
        image=str(record["input_path"]),
        text=str(record.get("text", config.get("prompt", "high quality, detailed object"))),
        max_correlation=float(record.get("max_correlation", 0.45)),
        frequency_scale=float(record.get("frequency_scale", 0.12)),
        camera_length_scale=float(record.get("camera_length_scale", 0.8)),
        basis_rank=basis_rank,
        target_joint_kl=target_kl,
        rbf_length_scale_deg=ell_deg,
        rho_geo=0.65,
        rho_start=0.45,
        rho_end=0.0,
        active_ratio=0.6,
        blur_kernel=11,
        blur_sigma=2.5,
        patch_size=8,
        qmc_scramble=True,
        qmc_dim=4,
        callback_blur_kernel=9,
        callback_blur_sigma=2.0,
        zindex_strength=0.25,
        preserve_marginal=True,
        base_model=model["base_model"],
        base_model_revision=model.get("base_model_revision"),
        vae_model=model.get("vae_model"),
        vae_model_revision=model.get("vae_model_revision"),
        unet_model=model.get("unet_model"),
        unet_model_revision=model.get("unet_model_revision"),
        lora_model=model.get("lora_model"),
        lora_model_revision=model.get("lora_model_revision"),
        adapter_path=model["adapter_path"],
        adapter_revision=model.get("adapter_revision"),
        mv_adapter_checkpoint=model.get(
            "mv_adapter_checkpoint", "mvadapter_i2mv_sdxl.safetensors"
        ),
        birefnet_model=model.get("birefnet_model", "ZhengPeng7/BiRefNet"),
        birefnet_revision=model.get("birefnet_revision"),
        scheduler=model.get("scheduler"),
        num_inference_steps=int(model["num_inference_steps"]),
        guidance_scale=float(model["guidance_scale"]),
        negative_prompt=str(
            config.get(
                "negative_prompt",
                "watermark, ugly, deformed, noisy, blurry, low contrast",
            )
        ),
        reference_conditioning_scale=float(
            config.get("reference_conditioning_scale", 1.0)
        ),
        lora_scale=float(config.get("lora_scale", 1.0)),
        preflight_summary={
            "applicable": True,
            "passed": True,
            "config_id": record.get("config_id"),
            "distribution_gate_passed": True,
        },
        run_metadata={},
        config_id=record.get("config_id"),
        input_sha256=record.get("input_sha256"),
        trajectory_output=(str(trajectory) if trajectory else None),
        trajectory_milestones=list(
            config.get("trajectory", {}).get(
                "milestones", (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
            )
        ),
    )


def _pipeline_kwargs(
    pipe: Any,
    args: Namespace,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    remove_bg_fn: Optional[Any],
) -> Dict[str, Any]:
    model = config["model"]
    return {
        "pipe": pipe,
        "num_views": len(args.azimuth_deg),
        "text": args.text,
        "image": args.image,
        "height": int(model.get("height", 768)),
        "width": int(model.get("width", 768)),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "remove_bg_fn": remove_bg_fn,
        "reference_conditioning_scale": args.reference_conditioning_scale,
        "negative_prompt": args.negative_prompt,
        "lora_scale": args.lora_scale,
        "device": str(model.get("device", "cuda")),
        "azimuth_deg": args.azimuth_deg,
        "method": args.resolved_method,
        "nile_mode": None,
        "nile_callback": "none",
        "max_correlation": args.max_correlation,
        "frequency_scale": args.frequency_scale,
        "camera_length_scale": args.camera_length_scale,
        "basis_rank": args.basis_rank,
        "target_joint_kl": args.target_joint_kl,
        "rbf_length_scale_deg": args.rbf_length_scale_deg,
        "trajectory_output": args.trajectory_output,
        "trajectory_milestones": args.trajectory_milestones,
        "return_run_metadata": True,
    }


class ArtifactIntegrityError(RuntimeError):
    def __init__(self, integrity: Mapping[str, Any]):
        self.integrity = dict(integrity)
        super().__init__(
            "artifact integrity failed: {}".format(self.integrity.get("issues", []))
        )


def validate_runtime_contract(plan: WorkerPlan) -> None:
    """Reject paths/settings that cannot be represented by existing helpers."""

    model = plan.resolved_config["model"]
    if int(model.get("height", 768)) != 768 or int(model.get("width", 768)) != 768:
        raise ValueError("persistent worker currently requires 768x768 inference")
    outputs = set()
    for record in plan.records:
        run_id = str(record["run_id"])
        input_path = Path(str(record["input_path"])).expanduser()
        if not input_path.is_file():
            raise ValueError("run {} input is missing: {}".format(run_id, input_path))
        output = Path(str(record["output"])).expanduser().resolve()
        if str(output) in outputs:
            raise ValueError("worker plan contains duplicate output path: {}".format(output))
        outputs.add(str(output))
        expected_metadata = output.with_name(output.stem + "_metadata.json")
        supplied_metadata = Path(str(record["metadata_path"])).expanduser().resolve()
        if supplied_metadata != expected_metadata:
            raise ValueError(
                "run {} metadata_path must be {} for _save_outputs compatibility".format(
                    run_id, expected_metadata
                )
            )


class PersistentInferenceWorker:
    """Execute one validated plan while retaining both model bundles in memory."""

    def __init__(
        self,
        plan: WorkerPlan,
        events: JsonlEventWriter,
        *,
        backend: Optional[Any] = None,
    ) -> None:
        self.plan = plan
        self.events = events
        self.backend = backend
        self.pipe = None
        self.remove_bg_fn = None
        self.mask_fn = None
        self._models_loaded = False

    def _load_models_once(self) -> None:
        if self._models_loaded:
            return
        if self.backend is None:
            self.backend = DefaultInferenceBackend()
        model = self.plan.resolved_config["model"]
        num_views = len(model["views_deg"])
        started = monotonic()
        self.events.emit(
            "model_load_started",
            num_views=num_views,
            device=model.get("device", "cuda"),
        )
        self.pipe = self.backend.prepare_pipeline(model, num_views=num_views)
        self.remove_bg_fn, self.mask_fn = self.backend.prepare_segmentation(model)
        if self.mask_fn is None:
            raise RuntimeError("BiRefNet mask function was not constructed")
        self._models_loaded = True
        self.events.emit(
            "model_load_succeeded",
            duration_seconds=round(monotonic() - started, 6),
            pipeline_load_count=1,
            segmentation_load_count=1,
        )

    def _run_attempt(
        self, record: Mapping[str, Any], args: Namespace
    ) -> Dict[str, Any]:
        pipeline_kwargs = _pipeline_kwargs(
            self.pipe,
            args,
            record,
            self.plan.resolved_config,
            self.remove_bg_fn,
        )
        images, reference_image, run_metadata = self.backend.run_pipeline(
            **pipeline_kwargs
        )
        args.run_metadata = dict(run_metadata)
        self.backend.save_outputs(images, reference_image, args, self.mask_fn)
        integrity = audit_artifact_bundle(record, self.plan.resolved_config)
        if not integrity["complete"]:
            raise ArtifactIntegrityError(integrity)
        return integrity

    def _run_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        run_id = str(record["run_id"])
        args = _save_namespace(record, self.plan.resolved_config)
        started = monotonic()
        self.events.emit(
            "run_started",
            run_id=run_id,
            method=record["method"],
            seed=int(record["seed"]),
            config_id=record.get("config_id"),
        )
        for attempt in (1, 2):
            attempt_started = monotonic()
            try:
                integrity = self._run_attempt(record, args)
                result = {
                    "run_id": run_id,
                    "status": "succeeded",
                    "attempts": attempt,
                    "duration_seconds": round(monotonic() - started, 6),
                    "artifact_integrity": integrity,
                }
                self.events.emit(
                    "run_succeeded",
                    run_id=run_id,
                    attempt=attempt,
                    attempt_duration_seconds=round(
                        monotonic() - attempt_started, 6
                    ),
                    duration_seconds=result["duration_seconds"],
                    artifact_integrity=integrity,
                )
                return result
            except Exception as error:
                rendered_traceback = traceback.format_exc()
                is_oom = bool(self.backend.is_oom(error))
                if is_oom and attempt == 1:
                    self.events.emit(
                        "run_oom_retry",
                        run_id=run_id,
                        attempt=attempt,
                        retry_attempt=2,
                        parameters_unchanged=True,
                        error_type=type(error).__name__,
                        error=str(error),
                        traceback=rendered_traceback,
                    )
                    self.backend.clear_cuda_cache()
                    continue
                integrity = (
                    error.integrity
                    if isinstance(error, ArtifactIntegrityError)
                    else None
                )
                result = {
                    "run_id": run_id,
                    "status": "failed",
                    "attempts": attempt,
                    "duration_seconds": round(monotonic() - started, 6),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": rendered_traceback,
                    "artifact_integrity": integrity,
                }
                self.events.emit(
                    "run_failed",
                    run_id=run_id,
                    attempt=attempt,
                    oom=is_oom,
                    duration_seconds=result["duration_seconds"],
                    error_type=result["error_type"],
                    error=result["error"],
                    traceback=rendered_traceback,
                    artifact_integrity=integrity,
                )
                return result
        raise AssertionError("unreachable retry loop")

    def run(self) -> Dict[str, Any]:
        validate_runtime_contract(self.plan)
        worker_started = monotonic()
        self.events.emit(
            "worker_started",
            plan=self.plan.source,
            planned=len(self.plan.records),
        )
        try:
            self._load_models_once()
        except Exception as error:
            rendered_traceback = traceback.format_exc()
            self.events.emit(
                "model_load_failed",
                error_type=type(error).__name__,
                error=str(error),
                traceback=rendered_traceback,
            )
            summary = {
                "planned": len(self.plan.records),
                "succeeded": 0,
                "failed": len(self.plan.records),
                "fatal": True,
                "error": str(error),
                "records": [],
            }
            self.events.emit(
                "worker_finished",
                **{key: value for key, value in summary.items() if key != "records"},
                duration_seconds=round(monotonic() - worker_started, 6),
            )
            return summary

        results = [self._run_record(record) for record in self.plan.records]
        summary = {
            "planned": len(results),
            "succeeded": sum(item["status"] == "succeeded" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "fatal": False,
            "records": results,
        }
        self.events.emit(
            "worker_finished",
            planned=summary["planned"],
            succeeded=summary["succeeded"],
            failed=summary["failed"],
            fatal=False,
            duration_seconds=round(monotonic() - worker_started, 6),
        )
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    return parser


def main(
    argv: Optional[Sequence[str]] = None, *, backend: Optional[Any] = None
) -> int:
    args = build_parser().parse_args(argv)
    with JsonlEventWriter(args.events) as events:
        events.emit("worker_invoked", plan=str(args.plan), events=str(args.events))
        try:
            plan = load_worker_plan(args.plan)
            validate_runtime_contract(plan)
        except Exception as error:
            events.emit(
                "plan_rejected",
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            return 2
        worker = PersistentInferenceWorker(plan, events, backend=backend)
        summary = worker.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
