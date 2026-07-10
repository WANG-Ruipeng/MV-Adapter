"""Static/stdlib contract tests for the formal NILE inference entry points."""

import ast
import unittest
from pathlib import Path

from scripts import run_nile_grid


REPO_ROOT = Path(__file__).resolve().parents[1]


class NileGridContractTests(unittest.TestCase):
    def test_formal_default_matrix(self):
        self.assertEqual(
            run_nile_grid.DEFAULT_METHODS,
            [
                "iid_default",
                "iid_external",
                "shared_full",
                "spectral_global_corr",
                "camera_rbf_corr",
                "nested_tree_a",
                "nested_tree_ab",
            ],
        )

    def test_default_strengths_and_quick_run_size(self):
        parser = run_nile_grid.build_parser()
        args = parser.parse_args(["--input", "placeholder.png"])
        run_nile_grid._validate_args(args)
        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertEqual(args.num_inference_steps, 30)
        self.assertEqual(args.strengths, [0.15, 0.30, 0.45, 0.60])

    def test_formal_method_rejects_legacy_callback(self):
        with self.assertRaisesRegex(ValueError, "do not allow callbacks"):
            run_nile_grid._parse_method("bad=nested_tree_ab:nile_vt")


class NileInferenceStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = REPO_ROOT / "scripts" / "inference_i2mv_sdxl_nile.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_pipeline_uses_independent_reference_and_scheduler_generators(self):
        self.assertIn('"generator": latent_generator', self.source)
        self.assertIn('"reference_generator": reference_generator', self.source)
        self.assertIn('"scheduler_generator": scheduler_generator', self.source)
        self.assertIn('"scheduler_generator_is_independent": True', self.source)

    def test_iid_default_omits_external_latents(self):
        self.assertIn('selected_method == "iid_default"', self.source)
        self.assertIn('if latents is not None:', self.source)
        self.assertIn('pipeline_kwargs["latents"] = latents', self.source)

    def test_formal_cli_parameters_exist(self):
        option_strings = {
            constant.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for constant in node.args
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        }
        self.assertTrue(
            {
                "--method",
                "--max_correlation",
                "--frequency_scale",
                "--camera_length_scale",
                "--base_model_revision",
                "--vae_model_revision",
                "--unet_model_revision",
                "--lora_model_revision",
                "--adapter_revision",
                "--birefnet_model",
                "--birefnet_revision",
            }
            <= option_strings
        )

    def test_model_revisions_reach_loaders_and_metadata(self):
        expected_loader_assignments = {
            'vae_kwargs["revision"] = vae_model_revision',
            'unet_kwargs["revision"] = unet_model_revision',
            'pipe_kwargs["revision"] = base_model_revision',
            'lora_kwargs["revision"] = lora_model_revision',
            'adapter_kwargs["revision"] = adapter_revision',
        }
        for assignment in expected_loader_assignments:
            self.assertIn(assignment, self.source)

        for field in (
            "base_model_revision",
            "vae_model_revision",
            "unet_model_revision",
            "lora_model_revision",
            "adapter_revision",
            "birefnet_revision",
        ):
            self.assertIn(f'"{field}": args.{field}', self.source)

        self.assertIn(
            'birefnet_kwargs["revision"] = args.birefnet_revision',
            self.source,
        )

    def test_required_preflight_runs_before_model_loading(self):
        main = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        preflight = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "_run_required_preflight"
        )
        load_pipeline = next(
            node
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "prepare_pipeline"
        )
        self.assertLess(preflight.lineno, load_pipeline.lineno)

    def test_formal_preflight_has_no_public_bypass_flag(self):
        option_strings = {
            constant.value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for constant in node.args
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        }
        self.assertFalse(
            any("preflight" in option.lower() or "distribution_gate" in option.lower()
                for option in option_strings)
        )
        self.assertIn('"preflight": getattr(', self.source)

    def test_custom_generators_are_appended_to_pipeline_parameters(self):
        pipeline_source = (
            REPO_ROOT
            / "mvadapter"
            / "pipelines"
            / "pipeline_mvadapter_i2mv_sdxl.py"
        ).read_text(encoding="utf-8")
        pipeline_tree = ast.parse(pipeline_source)
        call_method = next(
            node
            for node in ast.walk(pipeline_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__call__"
        )
        argument_names = [argument.arg for argument in call_method.args.args]
        self.assertEqual(argument_names[-2:], ["reference_generator", "scheduler_generator"])
        self.assertGreater(
            argument_names.index("reference_generator"),
            argument_names.index("reference_conditioning_scale"),
        )
        self.assertGreater(
            argument_names.index("scheduler_generator"),
            argument_names.index("reference_generator"),
        )

        self.assertIn(
            "scheduler_generator if scheduler_generator is not None else generator",
            pipeline_source,
        )


class NileDiagnosticsStaticContractTests(unittest.TestCase):
    def test_preflight_uses_all_hard_gates(self):
        source = (REPO_ROOT / "scripts" / "diagnose_nile_latents.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("diagnose_latents", source)
        self.assertIn("evaluate_distribution_gates", source)
        self.assertIn("target_covariance=target", source)
        self.assertIn('return 0 if payload["passed"] else 1', source)

    def test_reusable_ensemble_preflight_does_not_claim_pipeline_equivalence(self):
        source = (REPO_ROOT / "scripts" / "diagnose_nile_latents.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        run_preflight = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_preflight"
        )
        defaults = run_preflight.args.kw_defaults
        keyword_names = [argument.arg for argument in run_preflight.args.kwonlyargs]
        batch_default = defaults[keyword_names.index("batch_size")]
        self.assertIsInstance(batch_default, ast.Constant)
        self.assertEqual(batch_default.value, 16)
        self.assertNotIn("iid_default_equivalence", source)


if __name__ == "__main__":
    unittest.main()
