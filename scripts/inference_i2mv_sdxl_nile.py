"""MV-Adapter I2MV SDXL inference with view-correlated Gaussian latents.

The formal methods preserve each view's white-Gaussian marginal and only
change cross-view covariance. Frozen Sobol, frequency-mismatched, and latent
projection prototypes remain available solely for legacy failure analysis.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mvadapter.nile.callbacks import NILECallbackConfig, NILEViewTimeCallback
from mvadapter.nile.basis import build_dct2_basis
from mvadapter.nile.covariance import (
    calibrate_alpha_for_target_kl,
    covariance_metadata,
    periodic_camera_rbf_covariance,
    tree_a_covariance,
    tree_ab_covariance,
)
from mvadapter.nile.lowrank_coupling import apply_latent_coupling
from mvadapter.nile.nested_elements import make_nested_tree_latents
from mvadapter.nile.sampler import NILEConfig, make_initial_latents
from mvadapter.nile.spectral_gaussian import (
    make_camera_rbf_correlated_latents,
    make_spectral_global_correlated_latents,
)
from mvadapter.nile.trajectory import (
    DEFAULT_TRAJECTORY_MILESTONES,
    TrajectoryObserver,
)
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (
    MVAdapterI2MVSDXLPipeline,
)
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils import get_orthogonal_camera

try:
    from scripts.diagnose_nile_latents import run_preflight
except ModuleNotFoundError as error:
    # Support direct ``python scripts/inference_i2mv_sdxl_nile.py`` execution
    # as well as the preferred ``python -m scripts...`` entry point.
    if error.name != "scripts":
        raise
    from diagnose_nile_latents import run_preflight


FORMAL_METHODS = (
    "iid_default",
    "iid_external",
    "shared_full",
    "spectral_global_corr",
    "camera_rbf_corr",
    "nested_tree_a",
    "nested_tree_ab",
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
LOWRANK_METHODS = (
    "lowrank_camera_rbf",
    "lowrank_nested_tree_a",
    "lowrank_nested_tree_ab",
)
LEGACY_NILE_MODES = (
    "iid",
    "shared",
    "lowpass_shared",
    "flat_sobol",
    "nile_v",
    "nile_vtp",
)
ALL_METHODS = FORMAL_METHODS + LEGACY_NILE_MODES
NILE_CALLBACK_MODES = ("none", "nile_vt", "nile_vtp")
PREFLIGHT_BATCH_SIZE = 16
PREFLIGHT_CHANNELS = 4
PREFLIGHT_HEIGHT = 96
PREFLIGHT_WIDTH = 96


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    mv_adapter_checkpoint,
    scheduler,
    num_views,
    device,
    dtype,
    base_model_revision=None,
    vae_model_revision=None,
    unet_model_revision=None,
    lora_model_revision=None,
    adapter_revision=None,
):
    if scheduler not in (None, "ddpm", "lcm"):
        raise ValueError(
            "scheduler must be one of None, 'ddpm', or 'lcm'; got {!r}".format(
                scheduler
            )
        )
    if not isinstance(mv_adapter_checkpoint, str) or not mv_adapter_checkpoint.strip():
        raise ValueError("mv_adapter_checkpoint must be a non-empty filename")

    # Load VAE and U-Net overrides if provided.
    pipe_kwargs = {}
    if vae_model is not None:
        vae_kwargs = {}
        if vae_model_revision is not None:
            vae_kwargs["revision"] = vae_model_revision
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model, **vae_kwargs)
    if unet_model is not None:
        unet_kwargs = {}
        if unet_model_revision is not None:
            unet_kwargs["revision"] = unet_model_revision
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
            unet_model, **unet_kwargs
        )

    pipe: MVAdapterI2MVSDXLPipeline
    if base_model_revision is not None:
        pipe_kwargs["revision"] = base_model_revision
    pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

    scheduler_class = None
    if scheduler == "ddpm":
        scheduler_class = DDPMScheduler
    elif scheduler == "lcm":
        scheduler_class = LCMScheduler

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=scheduler_class,
    )
    pipe.init_custom_adapter(num_views=num_views)
    adapter_kwargs = {"weight_name": mv_adapter_checkpoint}
    if adapter_revision is not None:
        adapter_kwargs["revision"] = adapter_revision
    pipe.load_custom_adapter(adapter_path, **adapter_kwargs)

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        lora_kwargs = {"weight_name": name_}
        if lora_model_revision is not None:
            lora_kwargs["revision"] = lora_model_revision
        pipe.load_lora_weights(model_, **lora_kwargs)

    # VAE slicing reduces peak memory without changing the generated samples.
    pipe.enable_vae_slicing()
    return pipe


def remove_bg(image, net, transform, device):
    image_size = image.size
    model_dtype = next(net.parameters()).dtype
    input_images = (
        transform(image.convert("RGB"))
        .unsqueeze(0)
        .to(device=device, dtype=model_dtype)
    )
    with torch.inference_mode():
        preds = net(input_images)[-1].sigmoid().float().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image


def preprocess_image(image: Image.Image, height, width):
    image = np.array(image)
    alpha = image[..., 3] > 0
    h, w = alpha.shape

    # Crop to the non-transparent object bounds.
    y, x = np.where(alpha)
    y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, h)
    x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, w)
    image_center = image[y0:y1, x0:x1]

    # Resize the longer side to 90% of the target canvas.
    h, w, _ = image_center.shape
    if h > w:
        w = int(w * (height * 0.9) / h)
        h = int(height * 0.9)
    else:
        h = int(h * (width * 0.9) / w)
        w = int(width * 0.9)
    image_center = np.array(Image.fromarray(image_center).resize((w, h)))

    start_h = (height - h) // 2
    start_w = (width - w) // 2
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[start_h : start_h + h, start_w : start_w + w] = image_center
    image = image.astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _effective_seed(seed: int) -> int:
    # NILE owns all initial-noise generation. Resolving the legacy -1 sentinel
    # to zero keeps the experiment reproducible and is the convention used in
    # the NILE design specification.
    return 0 if seed == -1 else seed


def _resolve_method(method, nile_mode):
    if method is not None:
        return method
    if nile_mode is not None:
        return nile_mode
    return "iid_default"


def _reference_vae_seed(seed: int) -> int:
    """Use a deterministic random stream disjoint from the NILE latent stream."""

    return (_effective_seed(seed) + 1_000_003) % (2**63 - 1)


def _scheduler_seed(seed: int) -> int:
    """Keep stochastic scheduler noise independent of latent construction."""

    return (_effective_seed(seed) + 2_000_033) % (2**63 - 1)


def _preflight_output_path(output: str) -> Path:
    output_path = Path(output).expanduser()
    return output_path.with_name(f"{output_path.stem}_preflight.json")


def _covariance_checksum(covariance: torch.Tensor) -> str:
    """Hash the exact float64 covariance used for one inference run."""

    canonical = covariance.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = (
        "nile-view-covariance-v1|{}|{}|".format(
            canonical.shape[0], canonical.shape[1]
        )
    ).encode("ascii")
    raw = bytes(canonical.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + raw).hexdigest()


def _lowrank_target_covariance(method, azimuth_deg, rbf_length_scale_deg):
    if method == "lowrank_camera_rbf":
        return periodic_camera_rbf_covariance(
            azimuth_deg,
            ell_deg=rbf_length_scale_deg,
            dtype=torch.float64,
        )
    if method == "lowrank_nested_tree_a":
        return tree_a_covariance(azimuth_deg, dtype=torch.float64)
    if method == "lowrank_nested_tree_ab":
        return tree_ab_covariance(azimuth_deg, dtype=torch.float64)
    raise ValueError(f"unsupported low-rank method: {method}")


def _prepare_lowrank_components(
    *,
    method,
    azimuth_deg,
    channels,
    latent_h,
    latent_w,
    basis_rank,
    target_joint_kl,
    rbf_length_scale_deg,
    basis_device="cpu",
):
    """Build and calibrate a low-rank coupling without touching model state."""

    basis, basis_info = build_dct2_basis(
        channels,
        latent_h,
        latent_w,
        basis_rank,
        device=basis_device,
        dtype=torch.float32,
        return_metadata=True,
    )
    target = _lowrank_target_covariance(
        method, azimuth_deg, rbf_length_scale_deg
    )
    calibration = calibrate_alpha_for_target_kl(
        target,
        basis_rank,
        target_joint_kl,
    )
    topology = {
        "lowrank_camera_rbf": "periodic_camera_rbf",
        "lowrank_nested_tree_a": "nested_tree_a",
        "lowrank_nested_tree_ab": "nested_tree_ab",
    }[method]
    target_info = covariance_metadata(
        target,
        azimuths_deg=azimuth_deg,
        ell_deg=(
            rbf_length_scale_deg if method == "lowrank_camera_rbf" else None
        ),
        topology=topology,
    )
    effective = calibration["covariance"]
    effective_info = covariance_metadata(
        effective,
        azimuths_deg=azimuth_deg,
        topology="identity_mixed_{}".format(topology),
    )
    metadata = {
        "method": method,
        "basis_rank": int(basis_rank),
        "target_joint_kl": float(target_joint_kl),
        "achieved_kl": float(calibration["achieved_kl"]),
        "kl_relative_error": float(calibration["relative_error"]),
        "alpha": float(calibration["alpha"]),
        "calibration_status": calibration["status"],
        "rbf_length_scale_deg": (
            float(rbf_length_scale_deg)
            if method == "lowrank_camera_rbf"
            else None
        ),
        "basis_checksum": basis_info["basis_checksum"],
        "target_covariance_checksum": _covariance_checksum(target),
        "covariance_checksum": _covariance_checksum(effective),
        "basis": basis_info,
        "target_covariance": target_info,
        "effective_covariance": effective_info,
        "calibration": calibration["json_metadata"],
        "topology_statement": "NILE-inspired nested Gaussian element topology",
        "strict_nile_sz_implemented": False,
    }
    return basis, target, calibration, metadata


def _run_lowrank_preflight(args, report_path: Path):
    """Fail fast on an invalid or unattainable low-rank CLI configuration."""

    _, _, calibration, construction = _prepare_lowrank_components(
        method=args.resolved_method,
        azimuth_deg=args.azimuth_deg,
        channels=PREFLIGHT_CHANNELS,
        latent_h=PREFLIGHT_HEIGHT,
        latent_w=PREFLIGHT_WIDTH,
        basis_rank=args.basis_rank,
        target_joint_kl=args.target_joint_kl,
        rbf_length_scale_deg=args.rbf_length_scale_deg,
    )
    checks = {
        "calibration_attainable": calibration["status"] == "calibrated",
        "kl_relative_error": calibration["relative_error"] < 1e-5,
        "basis_orthonormality": (
            construction["basis"]["output_orthonormality_error"] < 1e-6
        ),
        "effective_covariance_positive_definite": (
            construction["effective_covariance"]["min_eigenvalue"] > 0.0
        ),
    }
    passed = all(checks.values())
    payload = {
        "schema_version": "nile_lowrank_cli_preflight_v1",
        "passed": passed,
        "config": {
            "method": args.resolved_method,
            "azimuth_deg": list(args.azimuth_deg),
            "basis_rank": args.basis_rank,
            "target_joint_kl": args.target_joint_kl,
            "rbf_length_scale_deg": args.rbf_length_scale_deg,
        },
        "checks": checks,
        "construction": construction,
        "scope": (
            "deterministic construction gate; ensemble distribution gates are "
            "recorded by scripts.diagnose_nile_lowrank for formal studies"
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if not passed:
        failed = ", ".join(name for name, value in checks.items() if not value)
        raise RuntimeError(
            "formal low-rank preflight failed for {}: {} (requested KL {}, "
            "maximum/calibrated KL {}). Report: {}".format(
                args.resolved_method,
                failed,
                args.target_joint_kl,
                calibration["achieved_kl"],
                report_path,
            )
        )
    print(f"Preflight: {report_path}")
    return {
        "applicable": True,
        "passed": True,
        "report": str(report_path.resolve()),
        "schema_version": payload["schema_version"],
        "config": payload["config"],
        "checks": checks,
        "construction": construction,
    }


def _preflight_summary(payload, report_path: Path):
    """Keep generation metadata compact while retaining the full sidecar report."""

    record = payload["record"]
    report = record.get("report", {})
    gates = record.get("gates", {})
    summary = {
        "applicable": True,
        "passed": bool(payload["passed"]),
        "report": str(report_path.resolve()),
        "schema_version": payload.get("schema_version"),
        "config": payload.get("config"),
        "checks": gates.get("checks"),
    }
    if "error" in record:
        summary["error"] = record["error"]
    if report:
        summary["metrics"] = {
            "global": report.get("global"),
            "max_abs_lag_autocorrelation": report.get(
                "lag_autocorrelation", {}
            ).get("max_abs"),
            "max_radial_psd_deviation": report.get(
                "per_view_radial_psd_deviation", {}
            ).get("max"),
            "max_axis_stripe_score": report.get("axis_stripe_score", {}).get(
                "max"
            ),
            "cross_view_covariance_error": report.get(
                "cross_view_covariance_error"
            ),
        }
    return summary


def _run_required_preflight(args):
    """Gate every formal CLI run before any diffusion weights are loaded."""

    if args.resolved_method not in FORMAL_METHODS:
        return {
            "applicable": False,
            "passed": None,
            "reason": "legacy_failure_analysis_method",
        }

    report_path = _preflight_output_path(args.output)
    if args.resolved_method in LOWRANK_METHODS:
        return _run_lowrank_preflight(args, report_path)
    payload = run_preflight(
        args.resolved_method,
        view_angles=args.azimuth_deg,
        seed=_effective_seed(args.seed),
        max_correlation=args.max_correlation,
        frequency_scale=args.frequency_scale,
        camera_length_scale=args.camera_length_scale,
        batch_size=PREFLIGHT_BATCH_SIZE,
        channels=PREFLIGHT_CHANNELS,
        height=PREFLIGHT_HEIGHT,
        width=PREFLIGHT_WIDTH,
        device=args.device,
        output=report_path,
    )
    summary = _preflight_summary(payload, report_path)
    if not payload["passed"]:
        record = payload["record"]
        if "error" in record:
            failure = record["error"]
        else:
            failure = ", ".join(
                name
                for name, check in record["gates"]["checks"].items()
                if not check["passed"]
            )
        raise RuntimeError(
            "formal latent distribution preflight failed for {}: {}. "
            "Report: {}".format(args.resolved_method, failure, report_path)
        )
    print(f"Preflight: {report_path}")
    return summary


def _validated_trajectory_milestones(values):
    milestones = tuple(float(value) for value in values)
    if not milestones:
        raise ValueError("trajectory_milestones must not be empty")
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in milestones
    ):
        raise ValueError("trajectory_milestones must be finite values in [0, 1]")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError(
            "trajectory_milestones must be strictly increasing and unique"
        )
    if not math.isclose(milestones[0], 0.0, abs_tol=1e-12):
        raise ValueError("trajectory_milestones must start at 0.0")
    if not math.isclose(milestones[-1], 1.0, abs_tol=1e-12):
        raise ValueError("trajectory_milestones must end at 1.0")
    return milestones


def _validate_nile_configuration(
    *,
    num_views,
    height,
    width,
    vae_scale_factor,
    method,
    nile_callback,
    rho_geo,
    rho_start,
    rho_end,
    active_ratio,
    blur_kernel,
    blur_sigma,
    patch_size,
    qmc_dim,
    callback_blur_kernel,
    callback_blur_sigma,
    zindex_strength,
    max_correlation,
    frequency_scale,
    camera_length_scale,
    basis_rank,
    target_joint_kl,
    rbf_length_scale_deg,
    trajectory_output,
    trajectory_milestones,
):
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % vae_scale_factor or width % vae_scale_factor:
        raise ValueError(
            "height and width must be divisible by pipe.vae_scale_factor "
            f"({vae_scale_factor})"
        )
    if method not in ALL_METHODS:
        raise ValueError(f"unsupported sampler method: {method}")
    if nile_callback not in NILE_CALLBACK_MODES:
        raise ValueError(f"unsupported NILE callback mode: {nile_callback}")
    if method in FORMAL_METHODS and nile_callback != "none":
        raise ValueError(
            "formal distribution-preserving methods prohibit legacy latent callbacks"
        )
    if trajectory_output is not None and nile_callback != "none":
        raise ValueError(
            "trajectory observation cannot be combined with a latent-mutating callback"
        )
    if method in FORMAL_METHODS:
        valid_correlation = (
            math.isfinite(max_correlation) and 0.0 <= max_correlation < 1.0
        )
        correlation_interval = "[0, 1)"
    else:
        # Legacy methods ignore this new field; accepting one preserves old
        # --rhos 1.0 grid commands while rho_geo remains independently checked.
        valid_correlation = (
            math.isfinite(max_correlation) and 0.0 <= max_correlation <= 1.0
        )
        correlation_interval = "[0, 1]"
    if not valid_correlation:
        raise ValueError(
            f"max_correlation must be in {correlation_interval}, got {max_correlation}"
        )
    if not math.isfinite(frequency_scale) or frequency_scale <= 0.0:
        raise ValueError(f"frequency_scale must be positive, got {frequency_scale}")
    if not math.isfinite(camera_length_scale) or camera_length_scale <= 0.0:
        raise ValueError(
            f"camera_length_scale must be positive, got {camera_length_scale}"
        )

    for name, value in (
        ("rho_geo", rho_geo),
        ("rho_start", rho_start),
        ("rho_end", rho_end),
        ("zindex_strength", zindex_strength),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if not math.isfinite(active_ratio) or not 0.0 < active_ratio <= 1.0:
        raise ValueError(f"active_ratio must be in (0, 1], got {active_ratio}")

    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    if isinstance(basis_rank, bool) or not isinstance(basis_rank, int):
        raise ValueError("basis_rank must be a positive integer")
    maximum_basis_rank = 4 * (latent_h * latent_w - 1)
    if basis_rank <= 0 or basis_rank > maximum_basis_rank:
        raise ValueError(
            f"basis_rank must be in [1, {maximum_basis_rank}], got {basis_rank}"
        )
    if trajectory_output is not None and basis_rank < 2:
        raise ValueError("trajectory observation requires basis_rank >= 2")
    _validated_trajectory_milestones(trajectory_milestones)
    if not math.isfinite(target_joint_kl) or target_joint_kl < 0.0:
        raise ValueError(
            f"target_joint_kl must be finite and non-negative, got {target_joint_kl}"
        )
    if (
        not math.isfinite(rbf_length_scale_deg)
        or rbf_length_scale_deg <= 0.0
    ):
        raise ValueError(
            "rbf_length_scale_deg must be finite and positive, got "
            f"{rbf_length_scale_deg}"
        )
    for name, kernel in (
        ("blur_kernel", blur_kernel),
        ("callback_blur_kernel", callback_blur_kernel),
    ):
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer, got {kernel}")
        if kernel // 2 >= min(latent_h, latent_w):
            raise ValueError(
                f"{name}={kernel} is too large for latent size "
                f"{latent_h}x{latent_w}"
            )
    if not math.isfinite(blur_sigma) or blur_sigma <= 0.0:
        raise ValueError(f"blur_sigma must be positive, got {blur_sigma}")
    if not math.isfinite(callback_blur_sigma) or callback_blur_sigma <= 0.0:
        raise ValueError(
            f"callback_blur_sigma must be positive, got {callback_blur_sigma}"
        )
    if patch_size <= 0 or patch_size > min(latent_h, latent_w):
        raise ValueError(
            f"patch_size must be in [1, {min(latent_h, latent_w)}], "
            f"got {patch_size}"
        )
    if qmc_dim <= 0 or qmc_dim > 21201:
        raise ValueError(f"qmc_dim must be in [1, 21201], got {qmc_dim}")


def run_pipeline(
    pipe,
    num_views,
    text,
    image,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed,
    remove_bg_fn=None,
    reference_conditioning_scale=1.0,
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    lora_scale=1.0,
    device="cuda",
    azimuth_deg=None,
    method=None,
    nile_mode=None,
    nile_callback="none",
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
    max_correlation=0.45,
    frequency_scale=0.12,
    camera_length_scale=0.8,
    basis_rank=8,
    target_joint_kl=1.0,
    rbf_length_scale_deg=90.0,
    trajectory_output=None,
    trajectory_milestones=DEFAULT_TRAJECTORY_MILESTONES,
    return_run_metadata=False,
):
    if azimuth_deg is None:
        azimuth_deg = [0, 45, 90, 180, 270, 315]
    if len(azimuth_deg) != num_views:
        raise ValueError(
            f"num_views={num_views} does not match {len(azimuth_deg)} azimuths"
        )
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    selected_method = _resolve_method(method, nile_mode)
    trajectory_milestones = _validated_trajectory_milestones(
        trajectory_milestones
    )
    vae_scale_factor = int(pipe.vae_scale_factor)
    _validate_nile_configuration(
        num_views=num_views,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
        method=selected_method,
        nile_callback=nile_callback,
        rho_geo=rho_geo,
        rho_start=rho_start,
        rho_end=rho_end,
        active_ratio=active_ratio,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
        patch_size=patch_size,
        qmc_dim=qmc_dim,
        callback_blur_kernel=callback_blur_kernel,
        callback_blur_sigma=callback_blur_sigma,
        zindex_strength=zindex_strength,
        max_correlation=max_correlation,
        frequency_scale=frequency_scale,
        camera_length_scale=camera_length_scale,
        basis_rank=basis_rank,
        target_joint_kl=target_joint_kl,
        rbf_length_scale_deg=rbf_length_scale_deg,
        trajectory_output=trajectory_output,
        trajectory_milestones=trajectory_milestones,
    )

    # Prepare cameras and per-view Plucker controls exactly as in the original
    # I2MV SDXL inference path.
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views,
        distance=[1.8] * num_views,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in azimuth_deg],
        device=device,
    )
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, width
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    # Preserve the original background-removal and RGBA preprocessing behavior.
    reference_image = Image.open(image) if isinstance(image, str) else image
    if remove_bg_fn is not None:
        reference_image = remove_bg_fn(reference_image)
        reference_image = preprocess_image(reference_image, height, width)
    elif reference_image.mode == "RGBA":
        reference_image = preprocess_image(reference_image, height, width)

    batch_size = 1  # This CLI accepts one prompt and one reference image per run.
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor
    channels = int(pipe.unet.config.in_channels)
    execution_device = pipe._execution_device
    latent_dtype = pipe.unet.dtype

    expected_shape = (batch_size * num_views, channels, latent_h, latent_w)
    latent_generator = torch.Generator(device=execution_device).manual_seed(
        _effective_seed(seed)
    )
    reference_generator = torch.Generator(device=execution_device).manual_seed(
        _reference_vae_seed(seed)
    )
    scheduler_generator = torch.Generator(device=execution_device).manual_seed(
        _scheduler_seed(seed)
    )

    latents = None
    trajectory_basis = None
    construction_metadata = {
        "method": selected_method,
        "basis_rank": None,
        "target_joint_kl": None,
        "achieved_kl": None,
        "alpha": None,
        "basis_checksum": None,
        "covariance_checksum": None,
        "per_sample_standardization": False,
    }
    if selected_method == "iid_default":
        # Deliberately omit external latents. The pipeline consumes exactly the
        # same latent_generator stream that iid_external consumes below.
        construction_metadata.update(
            {
                "identity_passthrough": True,
                "external_latents": False,
                "interpretation": "pipeline_native_iid_control",
            }
        )
    elif selected_method == "iid_external":
        iid_latents = torch.randn(
            expected_shape,
            generator=latent_generator,
            device=execution_device,
            dtype=latent_dtype,
        )
        latents, coupling_info = apply_latent_coupling(
            iid_latents,
            "iid_external",
            num_views,
            return_metadata=True,
        )
        construction_metadata.update(coupling_info)
    elif selected_method == "shared_full":
        # Draw the same canonical full IID tensor as iid_external. Coupling
        # then reuses its first view, so all methods consume an identical
        # initial-noise stream even though this diagnostic is degenerate.
        iid_latents = torch.randn(
            expected_shape,
            generator=latent_generator,
            device=execution_device,
            dtype=latent_dtype,
        )
        latents, coupling_info = apply_latent_coupling(
            iid_latents,
            "shared_full",
            num_views,
            return_metadata=True,
        )
        construction_metadata.update(coupling_info)
    elif selected_method in LOWRANK_METHODS:
        iid_latents = torch.randn(
            expected_shape,
            generator=latent_generator,
            device=execution_device,
            dtype=latent_dtype,
        )
        basis, target_covariance, calibration, lowrank_info = (
            _prepare_lowrank_components(
                method=selected_method,
                azimuth_deg=azimuth_deg,
                channels=channels,
                latent_h=latent_h,
                latent_w=latent_w,
                basis_rank=basis_rank,
                target_joint_kl=target_joint_kl,
                rbf_length_scale_deg=rbf_length_scale_deg,
                basis_device=execution_device,
            )
        )
        if calibration["status"] != "calibrated":
            raise RuntimeError(
                "target joint KL {} is unattainable for method={} rank={}; "
                "maximum calibrated KL is {}".format(
                    target_joint_kl,
                    selected_method,
                    basis_rank,
                    calibration["achieved_kl"],
                )
            )
        latents, coupling_info = apply_latent_coupling(
            iid_latents,
            selected_method,
            num_views,
            basis=basis,
            view_covariance=target_covariance,
            alpha=calibration["alpha"],
            return_metadata=True,
        )
        lowrank_info["coupling"] = coupling_info
        lowrank_info["alpha_zero_exact_iid_passthrough"] = bool(
            calibration["alpha"] == 0.0
            and latents is iid_latents
            and latents.data_ptr() == iid_latents.data_ptr()
        )
        construction_metadata = lowrank_info
        trajectory_basis = basis
    elif selected_method == "spectral_global_corr":
        latents = make_spectral_global_correlated_latents(
            batch_size,
            num_views,
            channels,
            latent_h,
            latent_w,
            device=execution_device,
            dtype=latent_dtype,
            generator=latent_generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
        )
    elif selected_method == "camera_rbf_corr":
        latents = make_camera_rbf_correlated_latents(
            batch_size,
            num_views,
            channels,
            latent_h,
            latent_w,
            azimuth_deg,
            device=execution_device,
            dtype=latent_dtype,
            generator=latent_generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            length_scale=camera_length_scale,
        )
    elif selected_method in {"nested_tree_a", "nested_tree_ab"}:
        latents = make_nested_tree_latents(
            batch_size,
            num_views,
            channels,
            latent_h,
            latent_w,
            azimuth_deg,
            device=execution_device,
            dtype=latent_dtype,
            generator=latent_generator,
            max_correlation=max_correlation,
            frequency_scale=frequency_scale,
            tree_mode="a" if selected_method == "nested_tree_a" else "ab",
        )
    else:
        # Frozen v0 failure-analysis path. It is intentionally excluded from
        # the formal default matrix and distribution gate.
        legacy_cfg = NILEConfig(
            mode=selected_method,
            seed=_effective_seed(seed),
            rho_geo=rho_geo,
            blur_kernel=blur_kernel,
            blur_sigma=blur_sigma,
            patch_size=patch_size,
            qmc_scramble=qmc_scramble,
            qmc_dim=qmc_dim,
        )
        latents = make_initial_latents(
            batch_size=batch_size,
            num_views=num_views,
            channels=channels,
            latent_h=latent_h,
            latent_w=latent_w,
            device=execution_device,
            dtype=latent_dtype,
            cfg=legacy_cfg,
        )

    if latents is not None and tuple(latents.shape) != expected_shape:
        raise ValueError(
            "sampler returned an invalid latent shape: "
            f"expected {expected_shape}, got {tuple(latents.shape)}"
        )

    pipeline_kwargs = {
        "prompt": text,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": num_views,
        "control_image": control_images,
        "control_conditioning_scale": 1.0,
        "reference_image": reference_image,
        "reference_conditioning_scale": reference_conditioning_scale,
        "negative_prompt": negative_prompt,
        "cross_attention_kwargs": {"scale": lora_scale},
        "generator": latent_generator,
        "reference_generator": reference_generator,
        "scheduler_generator": scheduler_generator,
    }
    if latents is not None:
        pipeline_kwargs["latents"] = latents

    observer = None
    if trajectory_output is not None:
        if trajectory_basis is None:
            trajectory_basis, trajectory_basis_info = build_dct2_basis(
                channels,
                latent_h,
                latent_w,
                basis_rank,
                device=execution_device,
                dtype=torch.float32,
                return_metadata=True,
            )
            construction_metadata["trajectory_basis_checksum"] = (
                trajectory_basis_info["basis_checksum"]
            )
        observer = TrajectoryObserver(
            trajectory_basis,
            num_views=num_views,
            batch_size=batch_size,
            total_steps=num_inference_steps,
            milestones=trajectory_milestones,
        )
        pipeline_kwargs["callback_on_step_end"] = observer
        pipeline_kwargs["callback_on_step_end_tensor_inputs"] = observer.tensor_inputs
    elif nile_callback != "none":
        callback_cfg = NILECallbackConfig(
            mode=nile_callback,
            num_views=num_views,
            batch_size=batch_size,
            rho_start=rho_start,
            rho_end=rho_end,
            active_ratio=active_ratio,
            blur_kernel=callback_blur_kernel,
            blur_sigma=callback_blur_sigma,
            patch_size=patch_size,
            zindex_strength=zindex_strength,
            preserve_marginal=preserve_marginal,
        )
        pipeline_kwargs["callback_on_step_end"] = NILEViewTimeCallback(callback_cfg)
        pipeline_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    images = pipe(**pipeline_kwargs).images
    trajectory_files = None
    if observer is not None:
        saved = observer.save(trajectory_output)
        trajectory_files = {
            name: str(path.resolve()) if path is not None else None
            for name, path in saved.items()
        }
    run_metadata = {
        "distribution": construction_metadata,
        "trajectory": {
            "enabled": observer is not None,
            "read_only": True,
            "recorded_after_prepare_latents": observer is not None,
            "milestones": (
                list(trajectory_milestones) if observer is not None else None
            ),
            "files": trajectory_files,
        },
    }
    if return_run_metadata:
        return images, reference_image, run_metadata
    return images, reference_image


def _save_outputs(images, reference_image, args, mask_fn=None):
    if len(images) != len(args.azimuth_deg):
        raise ValueError(
            f"pipeline returned {len(images)} images for "
            f"{len(args.azimuth_deg)} requested azimuths"
        )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    make_image_grid(images, rows=1).save(output_path)
    reference_path = output_path.with_name(
        f"{output_path.stem}_reference{output_path.suffix}"
    )
    reference_image.save(reference_path)

    view_files = []
    views_dir = None
    if args.save_views:
        views_dir = (
            Path(args.views_dir).expanduser()
            if args.views_dir is not None
            else output_path.with_name(f"{output_path.stem}_views")
        )
        views_dir.mkdir(parents=True, exist_ok=True)
        for index, (view_image, azimuth) in enumerate(zip(images, args.azimuth_deg)):
            view_path = views_dir / f"view_{index:03d}_azimuth_{azimuth:+04d}.png"
            view_image.save(view_path)
            view_files.append(str(view_path.resolve()))

    mask_files = []
    mask_dir = None
    if getattr(args, "save_masks", False):
        if mask_fn is None:
            raise RuntimeError(
                "--save_masks requires the configured foreground segmentation model"
            )
        mask_dir = (
            Path(args.mask_dir).expanduser()
            if args.mask_dir is not None
            else output_path.with_name(f"{output_path.stem}_masks")
        )
        mask_dir.mkdir(parents=True, exist_ok=True)
        for index, (view_image, azimuth) in enumerate(zip(images, args.azimuth_deg)):
            mask = mask_fn(view_image.copy())
            if mask.mode != "L":
                mask = mask.convert("L")
            mask_path = mask_dir / f"view_{index:03d}_azimuth_{azimuth:+04d}.png"
            mask.save(mask_path)
            mask_files.append(str(mask_path.resolve()))

    input_path = Path(args.image).expanduser().resolve()
    input_hasher = hashlib.sha256()
    with input_path.open("rb") as input_handle:
        for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
            input_hasher.update(chunk)
    actual_input_sha256 = input_hasher.hexdigest()
    declared_input_sha256 = getattr(args, "input_sha256", None)
    if (
        declared_input_sha256 is not None
        and str(declared_input_sha256).lower() != actual_input_sha256
    ):
        raise ValueError(
            "input_sha256 does not match the current input file: {}".format(
                input_path
            )
        )

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata = {
        "config_id": getattr(args, "config_id", None),
        "output": str(output_path.resolve()),
        "reference_output": str(reference_path.resolve()),
        "views_dir": str(views_dir.resolve()) if views_dir is not None else None,
        "view_files": view_files,
        "mask_dir": str(mask_dir.resolve()) if mask_dir is not None else None,
        "mask_files": mask_files,
        "azimuth_deg": list(args.azimuth_deg),
        "num_views": len(args.azimuth_deg),
        "seed": args.seed,
        "effective_seed": _effective_seed(args.seed),
        "reference_vae_seed": _reference_vae_seed(args.seed),
        "method": args.resolved_method,
        "max_correlation": args.max_correlation,
        "frequency_scale": args.frequency_scale,
        "camera_length_scale": args.camera_length_scale,
        "basis_rank": args.basis_rank,
        "target_joint_kl": args.target_joint_kl,
        "rbf_length_scale_deg": args.rbf_length_scale_deg,
        "mode": args.resolved_method,
        "callback": args.nile_callback,
        "rho_geo": args.rho_geo,
        "rho_start": args.rho_start,
        "rho_end": args.rho_end,
        "preflight": getattr(
            args,
            "preflight_summary",
            {"applicable": False, "passed": None, "reason": "not_recorded"},
        ),
        "input": {
            "image": str(input_path),
            "sha256": actual_input_sha256,
            "text": args.text,
        },
        "models": {
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
            "mv_adapter_checkpoint": args.mv_adapter_checkpoint,
            "birefnet_model": args.birefnet_model,
            "birefnet_revision": args.birefnet_revision,
            "scheduler": args.scheduler,
        },
        "inference": {
            "height": 768,
            "width": 768,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "reference_conditioning_scale": args.reference_conditioning_scale,
            "lora_scale": args.lora_scale,
        },
        "distribution": {
            "method": args.resolved_method,
            "formal_method": args.resolved_method in FORMAL_METHODS,
            "external_latents": args.resolved_method != "iid_default",
            "max_correlation": args.max_correlation,
            "frequency_scale": args.frequency_scale,
            "camera_length_scale": args.camera_length_scale,
            "basis_rank": args.basis_rank,
            "target_joint_kl": args.target_joint_kl,
            "rbf_length_scale_deg": args.rbf_length_scale_deg,
            "latent_generator_seed": _effective_seed(args.seed),
            "reference_generator_seed": _reference_vae_seed(args.seed),
            "reference_generator_is_independent": True,
            "scheduler_generator_seed": _scheduler_seed(args.seed),
            "scheduler_generator_is_independent": True,
            "callback_allowed": args.resolved_method not in FORMAL_METHODS,
            "per_sample_standardization": False,
        },
        "nile": {
            "mode": args.nile_mode,
            "callback": args.nile_callback,
            "rho_geo": args.rho_geo,
            "rho_start": args.rho_start,
            "rho_end": args.rho_end,
            "active_ratio": args.active_ratio,
            "blur_kernel": args.blur_kernel,
            "blur_sigma": args.blur_sigma,
            "patch_size": args.patch_size,
            "qmc_scramble": args.qmc_scramble,
            "qmc_dim": args.qmc_dim,
            "effective_qmc_dim": (
                1
                if args.resolved_method in {"flat_sobol", "nile_v", "nile_vtp"}
                else None
            ),
            "effective_qmc_scramble": (
                args.qmc_scramble
                if args.resolved_method in {"flat_sobol", "nile_v", "nile_vtp"}
                else None
            ),
            "qmc_dim_status": "reserved_for_strict_sz",
            "callback_blur_kernel": args.callback_blur_kernel,
            "callback_blur_sigma": args.callback_blur_sigma,
            "zindex_strength": args.zindex_strength,
            "preserve_marginal": args.preserve_marginal,
            "sequence_backend": (
                "sobol_prototype"
                if args.resolved_method in {"flat_sobol", "nile_v", "nile_vtp"}
                else "pseudorandom"
            ),
            "strict_sz_implemented": False,
            "rho_zero_note": (
                "For lowpass_shared/nile_v/nile_vtp, the prompt-defined "
                "formula yields standardized local high-pass noise at "
                "rho_geo=0; use the explicit iid mode as the IID baseline."
                if args.resolved_method in {"lowpass_shared", "nile_v", "nile_vtp"}
                else None
            ),
        },
    }
    run_metadata = getattr(args, "run_metadata", {})
    metadata["distribution"].update(run_metadata.get("distribution", {}))
    metadata["trajectory"] = run_metadata.get(
        "trajectory",
        {
            "enabled": False,
            "read_only": True,
            "recorded_after_prepare_latents": False,
            "files": None,
        },
    )
    metadata["foreground_masks"] = {
        "enabled": bool(getattr(args, "save_masks", False)),
        "backend": args.birefnet_model if mask_dir is not None else None,
        "backend_revision": args.birefnet_revision if mask_dir is not None else None,
        "directory": str(mask_dir.resolve()) if mask_dir is not None else None,
        "files": mask_files,
    }
    metadata_temporary = metadata_path.with_name(metadata_path.name + ".tmp")
    with metadata_temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(metadata_temporary, metadata_path)

    return output_path, reference_path, views_dir, metadata_path


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run MV-Adapter I2MV SDXL with NILE initial-latent and optional "
            "view-time trajectory coupling."
        )
    )

    # Models
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--base_model_revision", type=str, default=None)
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--vae_model_revision", type=str, default=None)
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--unet_model_revision", type=str, default=None)
    parser.add_argument("--scheduler", choices=("ddpm", "lcm"), default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--lora_model_revision", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")
    parser.add_argument("--adapter_revision", type=str, default=None)
    parser.add_argument(
        "--mv_adapter_checkpoint",
        type=str,
        default="mvadapter_i2mv_sdxl.safetensors",
    )
    parser.add_argument("--birefnet_model", type=str, default="ZhengPeng7/BiRefNet")
    parser.add_argument("--birefnet_revision", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Inference
    parser.add_argument(
        "--num_views",
        type=int,
        default=6,
        help="Deprecated compatibility option; azimuth_deg determines view count.",
    )
    parser.add_argument(
        "--azimuth_deg", type=int, nargs="+", default=[0, 45, 90, 180, 270, 315]
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, default="high quality")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--reference_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="watermark, ugly, deformed, noisy, blurry, low contrast",
    )
    parser.add_argument("--output", type=str, default="output.png")

    # Formal sampler selection. The old --nile_mode entry point remains
    # available exclusively for reproducing the frozen failure-analysis paths.
    parser.add_argument("--method", choices=ALL_METHODS, default=None)
    parser.add_argument(
        "--nile_mode",
        choices=LEGACY_NILE_MODES,
        default=None,
        help="Legacy v0 sampler selector; excluded from the formal matrix.",
    )
    parser.add_argument("--max_correlation", type=float, default=0.45)
    parser.add_argument("--frequency_scale", type=float, default=0.12)
    parser.add_argument("--camera_length_scale", type=float, default=0.8)
    parser.add_argument(
        "--basis_rank",
        type=int,
        default=8,
        help="Rank of the deterministic orthonormal DCT-II coupling subspace.",
    )
    parser.add_argument(
        "--target_joint_kl",
        type=float,
        default=1.0,
        help="Requested complete joint Gaussian KL from IID, in nats.",
    )
    parser.add_argument(
        "--rbf_length_scale_deg",
        type=float,
        default=90.0,
        help="Periodic camera-RBF length scale in degrees.",
    )
    parser.add_argument("--rho_geo", type=float, default=0.65)
    parser.add_argument("--blur_kernel", type=int, default=11)
    parser.add_argument("--blur_sigma", type=float, default=2.5)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument(
        "--qmc_dim",
        type=int,
        default=4,
        help=(
            "Reserved for the future hierarchical SZ backend; the current "
            "Sobol prototype uses effective dimension 1."
        ),
    )
    parser.add_argument(
        "--qmc_scramble", dest="qmc_scramble", action="store_true"
    )
    parser.add_argument(
        "--no_qmc_scramble",
        "--no-qmc-scramble",
        dest="qmc_scramble",
        action="store_false",
    )
    parser.set_defaults(qmc_scramble=True)

    # NILE denoising callback
    parser.add_argument(
        "--nile_callback", choices=NILE_CALLBACK_MODES, default="none"
    )
    parser.add_argument("--rho_start", type=float, default=0.45)
    parser.add_argument("--rho_end", type=float, default=0.0)
    parser.add_argument("--active_ratio", type=float, default=0.6)
    parser.add_argument("--callback_blur_kernel", type=int, default=9)
    parser.add_argument("--callback_blur_sigma", type=float, default=2.0)
    parser.add_argument("--zindex_strength", type=float, default=0.25)
    parser.add_argument(
        "--preserve_marginal", dest="preserve_marginal", action="store_true"
    )
    parser.add_argument(
        "--no_preserve_marginal",
        "--no-preserve-marginal",
        dest="preserve_marginal",
        action="store_false",
    )
    parser.set_defaults(preserve_marginal=True)

    # Input/output extras
    parser.add_argument("--remove_bg", action="store_true", help="Remove background")
    parser.add_argument(
        "--trajectory_output",
        type=str,
        default=None,
        help="Optional NPZ/prefix for read-only denoising trajectory diagnostics.",
    )
    parser.add_argument(
        "--trajectory_milestones",
        nargs="+",
        type=float,
        default=list(DEFAULT_TRAJECTORY_MILESTONES),
    )
    parser.add_argument("--config_id", type=str, default=None)
    parser.add_argument("--input_sha256", type=str, default=None)
    parser.add_argument(
        "--views_dir",
        type=str,
        default=None,
        help="Directory for individual views (default: <output_stem>_views).",
    )
    parser.add_argument("--save_views", dest="save_views", action="store_true")
    parser.add_argument(
        "--no_save_views",
        "--no-save-views",
        dest="save_views",
        action="store_false",
    )
    parser.set_defaults(save_views=True)
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory for per-view foreground masks (default: <output_stem>_masks).",
    )
    parser.add_argument("--save_masks", dest="save_masks", action="store_true")
    parser.add_argument(
        "--no_save_masks",
        "--no-save-masks",
        dest="save_masks",
        action="store_false",
    )
    parser.set_defaults(save_masks=False)
    return parser


def _validate_cli_args(parser, args):
    if args.method is not None and args.nile_mode is not None:
        parser.error("use either --method or legacy --nile_mode, not both")
    args.resolved_method = _resolve_method(args.method, args.nile_mode)
    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        parser.error(f"input image does not exist: {image_path}")
    if not args.azimuth_deg:
        parser.error("--azimuth_deg requires at least one angle")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive")
    if args.seed < -1:
        parser.error("--seed must be -1 or a non-negative integer")
    if not Path(args.output).suffix:
        parser.error("--output must include an image extension, for example .png")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {args.device}")

    try:
        # SDXL I2MV uses 768x768 and a VAE scale factor of eight. The same
        # checks run again against the loaded pipeline before allocating noise.
        _validate_nile_configuration(
            num_views=len(args.azimuth_deg),
            height=768,
            width=768,
            vae_scale_factor=8,
            method=args.resolved_method,
            nile_callback=args.nile_callback,
            rho_geo=args.rho_geo,
            rho_start=args.rho_start,
            rho_end=args.rho_end,
            active_ratio=args.active_ratio,
            blur_kernel=args.blur_kernel,
            blur_sigma=args.blur_sigma,
            patch_size=args.patch_size,
            qmc_dim=args.qmc_dim,
            callback_blur_kernel=args.callback_blur_kernel,
            callback_blur_sigma=args.callback_blur_sigma,
            zindex_strength=args.zindex_strength,
            max_correlation=args.max_correlation,
            frequency_scale=args.frequency_scale,
            camera_length_scale=args.camera_length_scale,
            basis_rank=args.basis_rank,
            target_joint_kl=args.target_joint_kl,
            rbf_length_scale_deg=args.rbf_length_scale_deg,
            trajectory_output=args.trajectory_output,
            trajectory_milestones=args.trajectory_milestones,
        )
    except ValueError as error:
        parser.error(str(error))


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(parser, args)

    num_views = len(args.azimuth_deg)
    args.preflight_summary = _run_required_preflight(args)
    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        mv_adapter_checkpoint=args.mv_adapter_checkpoint,
        scheduler=args.scheduler,
        num_views=num_views,
        device=args.device,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        base_model_revision=args.base_model_revision,
        vae_model_revision=args.vae_model_revision,
        unet_model_revision=args.unet_model_revision,
        lora_model_revision=args.lora_model_revision,
        adapter_revision=args.adapter_revision,
    )

    if args.remove_bg or args.save_masks:
        birefnet_kwargs = {"trust_remote_code": True}
        if args.birefnet_revision is not None:
            birefnet_kwargs["revision"] = args.birefnet_revision
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            args.birefnet_model, **birefnet_kwargs
        )
        birefnet.to(args.device)
        birefnet.eval()
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        segmentation_fn = lambda x: remove_bg(
            x, birefnet, transform_image, args.device
        )
        foreground_mask_fn = lambda x: segmentation_fn(x).getchannel("A")
        remove_bg_fn = segmentation_fn if args.remove_bg else None
    else:
        remove_bg_fn = None
        foreground_mask_fn = None

    images, reference_image, args.run_metadata = run_pipeline(
        pipe,
        num_views=num_views,
        text=args.text,
        image=args.image,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        lora_scale=args.lora_scale,
        reference_conditioning_scale=args.reference_conditioning_scale,
        negative_prompt=args.negative_prompt,
        device=args.device,
        remove_bg_fn=remove_bg_fn,
        azimuth_deg=args.azimuth_deg,
        method=args.resolved_method,
        nile_mode=args.nile_mode,
        nile_callback=args.nile_callback,
        rho_geo=args.rho_geo,
        rho_start=args.rho_start,
        rho_end=args.rho_end,
        active_ratio=args.active_ratio,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        patch_size=args.patch_size,
        qmc_scramble=args.qmc_scramble,
        qmc_dim=args.qmc_dim,
        callback_blur_kernel=args.callback_blur_kernel,
        callback_blur_sigma=args.callback_blur_sigma,
        zindex_strength=args.zindex_strength,
        preserve_marginal=args.preserve_marginal,
        max_correlation=args.max_correlation,
        frequency_scale=args.frequency_scale,
        camera_length_scale=args.camera_length_scale,
        basis_rank=args.basis_rank,
        target_joint_kl=args.target_joint_kl,
        rbf_length_scale_deg=args.rbf_length_scale_deg,
        trajectory_output=args.trajectory_output,
        trajectory_milestones=args.trajectory_milestones,
        return_run_metadata=True,
    )
    output_path, reference_path, views_dir, metadata_path = _save_outputs(
        images, reference_image, args, mask_fn=foreground_mask_fn
    )

    print(f"Grid: {output_path}")
    print(f"Reference: {reference_path}")
    if views_dir is not None:
        print(f"Views: {views_dir}")
    if args.save_masks:
        print(
            "Masks: {}".format(
                Path(args.mask_dir).expanduser()
                if args.mask_dir is not None
                else output_path.with_name(f"{output_path.stem}_masks")
            )
        )
    if args.trajectory_output is not None:
        print(f"Trajectory: {args.trajectory_output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
