import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from scripts.nile_lowrank_inference_worker import (
    JsonlEventWriter,
    PersistentInferenceWorker,
    audit_artifact_bundle,
    build_parser,
    install_timm_layers_compatibility,
    load_worker_plan,
    main,
)


class FakeOOM(RuntimeError):
    pass


class MockBackend:
    def __init__(
        self,
        *,
        oom_seed=None,
        fail_seed=None,
        missing_masks_seed=None,
    ):
        self.oom_seed = oom_seed
        self.fail_seed = fail_seed
        self.missing_masks_seed = missing_masks_seed
        self.pipeline_loads = 0
        self.segmentation_loads = 0
        self.run_calls = []
        self.save_calls = 0
        self.clear_calls = 0
        self._oom_raised = False

    def prepare_pipeline(self, model, *, num_views):
        self.pipeline_loads += 1
        return {"pipeline": True, "num_views": num_views}

    def prepare_segmentation(self, model):
        self.segmentation_loads += 1
        return (lambda image: image), (lambda image: image)

    def run_pipeline(self, **kwargs):
        snapshot = {
            key: kwargs[key]
            for key in (
                "seed",
                "method",
                "basis_rank",
                "target_joint_kl",
                "rbf_length_scale_deg",
                "num_inference_steps",
                "guidance_scale",
                "azimuth_deg",
            )
        }
        self.run_calls.append(snapshot)
        if kwargs["seed"] == self.oom_seed and not self._oom_raised:
            self._oom_raised = True
            raise FakeOOM("CUDA out of memory in mock")
        if kwargs["seed"] == self.fail_seed:
            raise RuntimeError("deliberate non-OOM failure")
        distribution = {"method": kwargs["method"]}
        if kwargs["method"].startswith("lowrank_"):
            distribution.update(
                {
                    "basis_rank": kwargs["basis_rank"],
                    "target_joint_kl": kwargs["target_joint_kl"],
                    "achieved_kl": kwargs["target_joint_kl"],
                    "alpha": 0.2,
                    "basis_checksum": "basis",
                    "covariance_checksum": "covariance",
                }
            )
        metadata = {"distribution": distribution}
        images = [object() for _ in kwargs["azimuth_deg"]]
        return images, object(), metadata

    def save_outputs(self, images, reference_image, args, mask_fn):
        self.save_calls += 1
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"grid")
        reference = output.with_name(output.stem + "_reference" + output.suffix)
        reference.write_bytes(b"reference")
        views_dir = Path(args.views_dir)
        masks_dir = Path(args.mask_dir)
        views_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        view_files = []
        mask_files = []
        for index, _ in enumerate(args.azimuth_deg):
            view = views_dir / "view_{:03d}.png".format(index)
            view.write_bytes(b"view")
            view_files.append(str(view.resolve()))
            if args.seed != self.missing_masks_seed or index == 0:
                mask = masks_dir / "mask_{:03d}.png".format(index)
                mask.write_bytes(b"mask")
                mask_files.append(str(mask.resolve()))
        if args.trajectory_output:
            trajectory = Path(args.trajectory_output)
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            trajectory.write_bytes(b"trajectory")
        metadata_path = output.with_name(output.stem + "_metadata.json")
        metadata = {
            "config_id": args.config_id,
            "output": str(output.resolve()),
            "reference_output": str(reference.resolve()),
            "view_files": view_files,
            "mask_files": mask_files,
            "azimuth_deg": list(args.azimuth_deg),
            "seed": args.seed,
            "input": {
                "image": str(Path(args.image).resolve()),
                "sha256": args.input_sha256,
            },
            "models": {
                "base_model_revision": args.base_model_revision,
                "vae_model_revision": args.vae_model_revision,
                "unet_model_revision": args.unet_model_revision,
                "lora_model_revision": args.lora_model_revision,
                "adapter_revision": args.adapter_revision,
                "birefnet_revision": args.birefnet_revision,
                "scheduler": args.scheduler,
                "mv_adapter_checkpoint": args.mv_adapter_checkpoint,
            },
            "inference": {
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
            },
            "distribution": args.run_metadata["distribution"],
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return output, reference, views_dir, metadata_path

    def is_oom(self, error):
        return isinstance(error, FakeOOM)

    def clear_cuda_cache(self):
        self.clear_calls += 1


def _config():
    return {
        "prompt": "high quality, detailed object",
        "model": {
            "base_model": "base",
            "vae_model": "vae",
            "adapter_path": "adapter",
            "mv_adapter_checkpoint": "mvadapter_i2mv_sdxl.safetensors",
            "birefnet_model": "birefnet",
            "device": "cuda",
            "height": 768,
            "width": 768,
            "views_deg": [0, 90],
            "num_inference_steps": 30,
            "guidance_scale": 3.0,
            "remove_background": True,
        },
    }


def _record(root, run_id, seed, method="iid_external"):
    input_path = root / (run_id + "_input.png")
    input_path.write_bytes(b"input")
    output = root / run_id / "grid.png"
    record = {
        "run_id": run_id,
        "config_id": "config-" + run_id,
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed": seed,
        "method": method,
        "distribution_gate_passed": True,
        "camera_list": [0, 90],
        "output": str(output),
        "metadata_path": str(output.with_name("grid_metadata.json")),
        "views_dir": str(output.with_name("grid_views")),
        "mask_dir": str(output.with_name("grid_masks")),
    }
    if method.startswith("lowrank_"):
        record.update(
            {"rank": 8, "target_kl": 1.0, "rbf_length_scale_deg": 90.0}
        )
    else:
        record.update(
            {"rank": None, "target_kl": 0.0, "rbf_length_scale_deg": None}
        )
    return record


def _write_plan(root, records):
    path = root / "worker_plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "resolved_config": _config(),
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def _events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class PersistentWorkerContractTests(unittest.TestCase):
    def test_legacy_timm_layers_are_aliased_for_birefnet(self):
        missing = ModuleNotFoundError("No module named 'timm.layers'")
        missing.name = "timm.layers"
        legacy_layers = ModuleType("timm.models.layers")
        legacy_layers.DropPath = object()
        legacy_layers.to_2tuple = object()
        legacy_layers.trunc_normal_ = object()
        timm_module = SimpleNamespace()

        def fake_import(name):
            if name == "timm.layers":
                raise missing
            if name == "timm.models.layers":
                return legacy_layers
            if name == "timm":
                return timm_module
            raise AssertionError("unexpected import: " + name)

        previous = sys.modules.pop("timm.layers", None)
        try:
            with patch(
                "scripts.nile_lowrank_inference_worker.importlib.import_module",
                side_effect=fake_import,
            ):
                mode = install_timm_layers_compatibility()
            self.assertEqual(mode, "legacy_timm.models.layers_alias")
            self.assertIs(sys.modules["timm.layers"], legacy_layers)
            self.assertIs(timm_module.layers, legacy_layers)
            imported = {}
            exec(
                "from timm.layers import DropPath, to_2tuple, trunc_normal_",
                imported,
            )
            self.assertIs(imported["DropPath"], legacy_layers.DropPath)
            self.assertIs(imported["to_2tuple"], legacy_layers.to_2tuple)
            self.assertIs(
                imported["trunc_normal_"], legacy_layers.trunc_normal_
            )
        finally:
            sys.modules.pop("timm.layers", None)
            if previous is not None:
                sys.modules["timm.layers"] = previous

    def test_two_runs_reuse_pipeline_and_segmentation_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = _write_plan(
                root,
                [_record(root, "first", 1), _record(root, "second", 2, "shared_full")],
            )
            plan = load_worker_plan(plan_path)
            event_path = root / "events.jsonl"
            backend = MockBackend()
            with JsonlEventWriter(event_path) as writer:
                result = PersistentInferenceWorker(
                    plan, writer, backend=backend
                ).run()
            self.assertEqual(result["succeeded"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(backend.pipeline_loads, 1)
            self.assertEqual(backend.segmentation_loads, 1)
            self.assertEqual(len(backend.run_calls), 2)
            names = [item["event"] for item in _events(event_path)]
            self.assertEqual(names.count("model_load_succeeded"), 1)
            self.assertEqual(names.count("run_succeeded"), 2)
            self.assertEqual(names[-1], "worker_finished")

    def test_saved_metadata_uses_plan_not_inference_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record(root, "iid", 1)
            plan_path = _write_plan(root, [record])
            event_path = root / "events.jsonl"
            backend = MockBackend()
            with JsonlEventWriter(event_path) as writer:
                result = PersistentInferenceWorker(
                    load_worker_plan(plan_path), writer, backend=backend
                ).run()

            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(backend.run_calls[0]["basis_rank"], 8)
            self.assertEqual(backend.run_calls[0]["target_joint_kl"], 1.0)
            self.assertEqual(
                backend.run_calls[0]["rbf_length_scale_deg"], 90.0
            )
            metadata = json.loads(
                Path(record["metadata_path"]).read_text(encoding="utf-8")
            )
            distribution = metadata["distribution"]
            self.assertIsNone(distribution["basis_rank"])
            self.assertEqual(distribution["target_joint_kl"], 0.0)
            self.assertIsNone(distribution["rbf_length_scale_deg"])

    def test_oom_retries_once_without_parameter_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record(root, "lowrank", 7, "lowrank_camera_rbf")
            plan_path = _write_plan(root, [record])
            event_path = root / "events.jsonl"
            backend = MockBackend(oom_seed=7)
            with JsonlEventWriter(event_path) as writer:
                result = PersistentInferenceWorker(
                    load_worker_plan(plan_path), writer, backend=backend
                ).run()
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(len(backend.run_calls), 2)
            self.assertEqual(backend.run_calls[0], backend.run_calls[1])
            self.assertEqual(backend.clear_calls, 1)
            retry = next(
                item for item in _events(event_path) if item["event"] == "run_oom_retry"
            )
            self.assertTrue(retry["parameters_unchanged"])
            self.assertIn("Traceback", retry["traceback"])

    def test_non_oom_failure_records_traceback_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = _write_plan(
                root, [_record(root, "fails", 3), _record(root, "passes", 4)]
            )
            event_path = root / "events.jsonl"
            backend = MockBackend(fail_seed=3)
            with JsonlEventWriter(event_path) as writer:
                result = PersistentInferenceWorker(
                    load_worker_plan(plan_path), writer, backend=backend
                ).run()
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["succeeded"], 1)
            self.assertEqual(len(backend.run_calls), 2)
            failure = next(
                item for item in _events(event_path) if item["event"] == "run_failed"
            )
            self.assertEqual(failure["error_type"], "RuntimeError")
            self.assertIn("Traceback", failure["traceback"])

    def test_incomplete_masks_turn_successful_inference_into_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record(root, "bad_masks", 9)
            plan_path = _write_plan(root, [record])
            event_path = root / "events.jsonl"
            backend = MockBackend(missing_masks_seed=9)
            with JsonlEventWriter(event_path) as writer:
                result = PersistentInferenceWorker(
                    load_worker_plan(plan_path), writer, backend=backend
                ).run()
            self.assertEqual(result["failed"], 1)
            integrity = result["records"][0]["artifact_integrity"]
            self.assertIn("mask_count_mismatch", integrity["issues"])
            self.assertFalse(audit_artifact_bundle(record, _config())["complete"])

    def test_plan_rejects_any_non_true_distribution_gate_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = _record(root, "rejected", 1)
            record["distribution_gate_passed"] = False
            plan_path = _write_plan(root, [record])
            with self.assertRaisesRegex(ValueError, "distribution_gate_passed"):
                load_worker_plan(plan_path)

            events = root / "events.jsonl"
            backend = MockBackend()
            code = main(
                ["--plan", str(plan_path), "--events", str(events)],
                backend=backend,
            )
            self.assertEqual(code, 2)
            self.assertEqual(backend.pipeline_loads, 0)
            self.assertEqual(_events(events)[-1]["event"], "plan_rejected")

    def test_cli_contract_requires_plan_and_events(self):
        parsed = build_parser().parse_args(["--plan", "p.json", "--events", "e.jsonl"])
        self.assertEqual(parsed.plan, Path("p.json"))
        self.assertEqual(parsed.events, Path("e.jsonl"))

    def test_event_sequence_continues_when_log_is_reopened(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with JsonlEventWriter(path) as writer:
                writer.emit("first")
            with JsonlEventWriter(path) as writer:
                writer.emit("second")
            self.assertEqual([item["sequence"] for item in _events(path)], [1, 2])


if __name__ == "__main__":
    unittest.main()
