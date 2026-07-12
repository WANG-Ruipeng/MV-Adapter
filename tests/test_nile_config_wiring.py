import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.eval_nile_lowrank_study import main as evaluation_main
from scripts.run_nile_lowrank_study import (
    load_config,
    resolve_config,
    run_evaluation_stage,
    validate_formal_protocol_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = REPOSITORY_ROOT / "configs" / "nile_lowrank_full.yaml"


class FormalProtocolConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(FORMAL_CONFIG)

    def test_checked_in_formal_config_passes_validation(self):
        validate_formal_protocol_config(self.config)
        resolved = resolve_config(self.config)
        self.assertEqual(
            resolved["evaluation"]["angle_bins_deg"],
            self.config["evaluation"]["angle_bins_deg"],
        )

    def test_silent_protocol_overrides_are_rejected(self):
        mutations = {
            "pilot methods": lambda config: config["pilot"].__setitem__(
                "methods", list(reversed(config["pilot"]["methods"]))
            ),
            "full methods": lambda config: config["full"].__setitem__(
                "methods", config["full"]["methods"][:-1]
            ),
            "scheduler": lambda config: config["model"].__setitem__(
                "scheduler", "ddpm"
            ),
            "load strategy": lambda config: config["runtime"].__setitem__(
                "model_load_strategy", "per_run"
            ),
            "retry count": lambda config: config["runtime"].__setitem__(
                "max_retries", 2
            ),
            "oom policy": lambda config: config["runtime"].__setitem__(
                "retry_oom_once", False
            ),
            "unknown runtime policy": lambda config: config["runtime"].__setitem__(
                "unimplemented_policy", True
            ),
            "invalid angle bins": lambda config: config["evaluation"].__setitem__(
                "angle_bins_deg", [45.0, 45.0]
            ),
            "full sweep": lambda config: config["full"].__setitem__(
                "allow_sweep", True
            ),
            "trajectory disabled": lambda config: config["trajectory"].__setitem__(
                "enabled", False
            ),
            "distribution gate bypass": lambda config: config["selection"].__setitem__(
                "require_distribution_gate", False
            ),
            "one standard error bypass": lambda config: config["selection"].__setitem__(
                "one_standard_error_tie_break", False
            ),
            "tie break order": lambda config: config["selection"].__setitem__(
                "tie_break_order", ["rank", "target_kl"]
            ),
            "identity backend": lambda config: config["evaluation"].__setitem__(
                "identity_backend", "clip"
            ),
            "mask backend": lambda config: config["evaluation"].__setitem__(
                "mask_backend", "none"
            ),
            "pair rows disabled": lambda config: config["evaluation"].__setitem__(
                "save_pair_rows", False
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(self.config)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_formal_protocol_config(changed)


class EvaluationAngleBinWiringTests(unittest.TestCase):
    def test_runner_passes_yaml_angle_bins_to_study_evaluator(self):
        config = load_config(FORMAL_CONFIG)
        config["evaluation"]["angle_bins_deg"] = [30.0, 75.0, 150.0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "pilot" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"runs": []}), encoding="utf-8")
            with mock.patch(
                "scripts.run_nile_lowrank_study.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=1,
                    stdout="evaluator stdout",
                    stderr="precise evaluator failure",
                ),
            ) as mocked_run:
                result = run_evaluation_stage(root, "pilot", config)
                evaluator_log = (
                    root / "metrics" / "pilot" / "evaluator.log"
                ).read_text(encoding="utf-8")
        command = mocked_run.call_args.args[0]
        start = command.index("--angle-bins-deg") + 1
        self.assertEqual(command[start : start + 3], ["30.0", "75.0", "150.0"])
        self.assertTrue(mocked_run.call_args.kwargs["capture_output"])
        self.assertTrue(mocked_run.call_args.kwargs["text"])
        self.assertIn("RETURN_CODE 1", evaluator_log)
        self.assertIn("evaluator stdout", evaluator_log)
        self.assertIn("precise evaluator failure", evaluator_log)
        self.assertEqual(result["stderr_tail"], "precise evaluator failure")
        self.assertEqual(
            result["evaluator_log"],
            str(root / "metrics" / "pilot" / "evaluator.log"),
        )

    def test_study_evaluator_passes_cli_angle_bins_to_base_evaluator(self):
        observed_commands = []

        def fake_run(command, check=False):
            observed_commands.append(command)
            raw_path = Path(command[command.index("--output") + 1])
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps({"samples": [], "angle_bin_summaries": []}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            output_dir = root / "metrics"
            manifest.write_text(json.dumps({"runs": []}), encoding="utf-8")
            with mock.patch(
                "scripts.eval_nile_lowrank_study.subprocess.run",
                side_effect=fake_run,
            ), mock.patch(
                "scripts.eval_nile_lowrank_study.generate_evaluation_plots",
                return_value={"complete": True, "artifacts": []},
            ), mock.patch(
                "scripts.eval_nile_lowrank_study.generate_contact_sheets",
                return_value={"complete": True, "artifacts": []},
            ):
                return_code = evaluation_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--output-dir",
                        str(output_dir),
                        "--metrics",
                        "lightweight",
                        "--skip-identity",
                        "--angle-bins-deg",
                        "30",
                        "75",
                        "150",
                    ]
                )
            payload = json.loads(
                (output_dir / "lowrank_metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(return_code, 1)
        self.assertEqual(len(observed_commands), 1)
        command = observed_commands[0]
        start = command.index("--angle-bins") + 1
        self.assertEqual(command[start : start + 3], ["30.0", "75.0", "150.0"])
        self.assertEqual(payload["angle_bins_deg"], [30.0, 75.0, 150.0])


if __name__ == "__main__":
    unittest.main()
