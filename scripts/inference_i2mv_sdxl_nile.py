"""MV-Adapter I2MV SDXL inference with the NILE-ViewTime prototype.

The low-discrepancy backend in this implementation is scrambled Sobol. It is
an experiment scaffold, not the strict hierarchical NILE/SZ backend, which is
kept as an explicit unimplemented interface in mvadapter.nile.sequence.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mvadapter.nile.callbacks import NILECallbackConfig, NILEViewTimeCallback
from mvadapter.nile.sampler import NILEConfig, make_initial_latents
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import (
    MVAdapterI2MVSDXLPipeline,
)
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from mvadapter.utils.mesh_utils import get_orthogonal_camera


NILE_MODES = (
    "iid",
    "shared",
    "lowpass_shared",
    "flat_sobol",
    "nile_v",
    "nile_vtp",
)
NILE_CALLBACK_MODES = ("none", "nile_vt", "nile_vtp")


def prepare_pipeline(
    base_model,
    vae_model,
    unet_model,
    lora_model,
    adapter_path,
    scheduler,
    num_views,
    device,
    dtype,
):
    # Load VAE and U-Net overrides if provided.
    pipe_kwargs = {}
    if vae_model is not None:
        pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(vae_model)
    if unet_model is not None:
        pipe_kwargs["unet"] = UNet2DConditionModel.from_pretrained(unet_model)

    pipe: MVAdapterI2MVSDXLPipeline
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
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_i2mv_sdxl.safetensors"
    )

    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    if lora_model is not None:
        model_, name_ = lora_model.rsplit("/", 1)
        pipe.load_lora_weights(model_, weight_name=name_)

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


def _reference_vae_seed(seed: int) -> int:
    """Use a deterministic random stream disjoint from the NILE latent stream."""

    return (_effective_seed(seed) + 1_000_003) % (2**63 - 1)


def _validate_nile_configuration(
    *,
    num_views,
    height,
    width,
    vae_scale_factor,
    nile_mode,
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
    if nile_mode not in NILE_MODES:
        raise ValueError(f"unsupported NILE mode: {nile_mode}")
    if nile_callback not in NILE_CALLBACK_MODES:
        raise ValueError(f"unsupported NILE callback mode: {nile_callback}")

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
    nile_mode="iid",
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
):
    if azimuth_deg is None:
        azimuth_deg = [0, 45, 90, 180, 270, 315]
    if len(azimuth_deg) != num_views:
        raise ValueError(
            f"num_views={num_views} does not match {len(azimuth_deg)} azimuths"
        )
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")

    vae_scale_factor = int(pipe.vae_scale_factor)
    _validate_nile_configuration(
        num_views=num_views,
        height=height,
        width=width,
        vae_scale_factor=vae_scale_factor,
        nile_mode=nile_mode,
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

    nile_cfg = NILEConfig(
        mode=nile_mode,
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
        cfg=nile_cfg,
    )
    expected_shape = (batch_size * num_views, channels, latent_h, latent_w)
    if tuple(latents.shape) != expected_shape:
        raise ValueError(
            "NILE sampler returned an invalid latent shape: "
            f"expected {expected_shape}, got {tuple(latents.shape)}"
        )

    # Custom initial latents remain owned by NILE. The pipeline generator is
    # still needed to make sampling the reference-image VAE posterior
    # deterministic and identical across sampler baselines.
    pipeline_generator = torch.Generator(device=execution_device).manual_seed(
        _reference_vae_seed(seed)
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
        "latents": latents,
        "generator": pipeline_generator,
    }

    if nile_callback != "none":
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
    return images, reference_image


def _save_outputs(images, reference_image, args):
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

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata = {
        "output": str(output_path.resolve()),
        "reference_output": str(reference_path.resolve()),
        "views_dir": str(views_dir.resolve()) if views_dir is not None else None,
        "view_files": view_files,
        "azimuth_deg": list(args.azimuth_deg),
        "num_views": len(args.azimuth_deg),
        "seed": args.seed,
        "effective_seed": _effective_seed(args.seed),
        "reference_vae_seed": _reference_vae_seed(args.seed),
        "mode": args.nile_mode,
        "callback": args.nile_callback,
        "rho_geo": args.rho_geo,
        "rho_start": args.rho_start,
        "rho_end": args.rho_end,
        "input": {
            "image": str(Path(args.image).expanduser().resolve()),
            "text": args.text,
        },
        "models": {
            "base_model": args.base_model,
            "vae_model": args.vae_model,
            "unet_model": args.unet_model,
            "lora_model": args.lora_model,
            "adapter_path": args.adapter_path,
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
                if args.nile_mode in {"flat_sobol", "nile_v", "nile_vtp"}
                else None
            ),
            "effective_qmc_scramble": (
                args.qmc_scramble
                if args.nile_mode in {"flat_sobol", "nile_v", "nile_vtp"}
                else None
            ),
            "qmc_dim_status": "reserved_for_strict_sz",
            "callback_blur_kernel": args.callback_blur_kernel,
            "callback_blur_sigma": args.callback_blur_sigma,
            "zindex_strength": args.zindex_strength,
            "preserve_marginal": args.preserve_marginal,
            "sequence_backend": (
                "sobol_prototype"
                if args.nile_mode in {"flat_sobol", "nile_v", "nile_vtp"}
                else "pseudorandom"
            ),
            "strict_sz_implemented": False,
            "rho_zero_note": (
                "For lowpass_shared/nile_v/nile_vtp, the prompt-defined "
                "formula yields standardized local high-pass noise at "
                "rho_geo=0; use the explicit iid mode as the IID baseline."
                if args.nile_mode in {"lowpass_shared", "nile_v", "nile_vtp"}
                else None
            ),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

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
    parser.add_argument(
        "--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix"
    )
    parser.add_argument("--unet_model", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default="huanngzh/mv-adapter")

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

    # NILE initial sampler
    parser.add_argument("--nile_mode", choices=NILE_MODES, default="iid")
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
    return parser


def _validate_cli_args(parser, args):
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
            nile_mode=args.nile_mode,
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
        )
    except ValueError as error:
        parser.error(str(error))


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _validate_cli_args(parser, args)

    num_views = len(args.azimuth_deg)
    pipe = prepare_pipeline(
        base_model=args.base_model,
        vae_model=args.vae_model,
        unet_model=args.unet_model,
        lora_model=args.lora_model,
        adapter_path=args.adapter_path,
        scheduler=args.scheduler,
        num_views=num_views,
        device=args.device,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    )

    if args.remove_bg:
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(args.device)
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        remove_bg_fn = lambda x: remove_bg(
            x, birefnet, transform_image, args.device
        )
    else:
        remove_bg_fn = None

    images, reference_image = run_pipeline(
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
    )
    output_path, reference_path, views_dir, metadata_path = _save_outputs(
        images, reference_image, args
    )

    print(f"Grid: {output_path}")
    print(f"Reference: {reference_path}")
    if views_dir is not None:
        print(f"Views: {views_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
