"""Contract checks for the formal low-rank Colab notebook."""

import ast
import json
from pathlib import Path
import unittest


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "mvadapter_nile_lowrank_full_colab.ipynb"
)


class NileLowrankNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.payload["cells"]
        cls.sources = ["".join(cell.get("source", [])) for cell in cls.cells]
        cls.joined = "\n".join(cls.sources)

    def test_exact_eighteen_clean_code_cells_in_required_order(self):
        self.assertEqual(len(self.cells), 18)
        for index, (cell, source) in enumerate(
            zip(self.cells, self.sources), 1
        ):
            self.assertEqual(cell["cell_type"], "code")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            self.assertTrue(source.startswith("# {}.".format(index)))
            ast.parse(source)

    def test_run_all_configuration_and_single_resumable_parent(self):
        for literal in (
            "RUN_ALL = True",
            "RESUME = True",
            "RUN_MET3R = True",
            "FULL_MODE = True",
            "--config",
            "str(FROZEN_CONFIG)",
            "--output-root",
            "str(OUTPUT_ROOT)",
            "--input-dir",
            "str(INPUT_DIR)",
            'command.append("--resume")',
        ):
            self.assertIn(literal, self.joined)
        self.assertNotIn("experiment-id", self.joined.lower())

    def test_actual_hashed_runner_root_is_used_for_every_artifact(self):
        freeze_cell = self.sources[5]
        self.assertIn(
            "RUNNER_ARTIFACT_ROOT = experiment_root(resolved_for_runner)",
            freeze_cell,
        )
        self.assertIn("load_config(FROZEN_CONFIG)", freeze_cell)
        self.assertIn("output_root=OUTPUT_ROOT", freeze_cell)
        for source in self.sources[6:]:
            self.assertNotIn('OUTPUT_ROOT / "', source)
        self.assertIn(
            'input_validation_dir = RUNNER_ARTIFACT_ROOT / "inputs"',
            self.sources[6],
        )
        self.assertIn(
            'RUNNER_ARTIFACT_ROOT / "pilot" / "manifest.json"',
            self.sources[9],
        )
        self.assertIn(
            'RUNNER_ARTIFACT_ROOT / "FULL_EXPERIMENT_REPORT.md"',
            self.sources[16],
        )
        self.assertIn(
            "figure_path.relative_to(RUNNER_ARTIFACT_ROOT)",
            self.sources[17],
        )

    def test_drive_safe_git_and_secret_handling(self):
        self.assertIn('drive.mount("/content/drive"', self.joined)
        self.assertIn('["git", "status", "--porcelain"]', self.joined)
        self.assertIn(
            "preserving it exactly and skipping update", self.joined
        )
        self.assertIn('userdata.get("HF_TOKEN")', self.joined)
        self.assertIn("from getpass import getpass", self.sources[5])
        self.assertIn("HfHubHTTPError", self.sources[5])
        self.assertIn("status_code not in (401, 403)", self.sources[5])
        self.assertIn("accept any gated", self.sources[5])
        self.assertNotIn("print(HF_TOKEN", self.sources[5])
        self.assertNotRegex(self.joined, r"hf_[A-Za-z0-9]{20,}")

    def test_remote_branch_has_formal_experiment_preflight(self):
        source = self.sources[2]
        for required_path in (
            "configs/nile_lowrank_full.yaml",
            "scripts/run_nile_lowrank_study.py",
            "scripts/nile_lowrank_inference_worker.py",
            "scripts/eval_nile_lowrank_study.py",
        ):
            self.assertIn(required_path, source)
        self.assertIn("missing_experiment_files", source)
        self.assertIn("Commit and push the local changes", source)
        self.assertIn("GITHUB_REPO", source)
        self.assertIn("GIT_BRANCH", source)
        self.assertIn("then restart Run All", source)

    def test_dependencies_and_all_immutable_weight_revisions(self):
        self.assertIn("requirements-colab.txt", self.joined)
        self.assertIn(
            "git+https://github.com/mohammadasim98/met3r@", self.joined
        )
        self.assertIn(
            "ee0e1752898559e1a3e85e2e151d3edeb9b55f73",
            self.joined,
        )
        self.assertIn("model_info(", self.joined)
        self.assertIn("snapshot_download(", self.joined)
        self.assertIn("hf_hub_download(", self.joined)
        self.assertIn("adapter_sha256", self.joined)
        self.assertIn("identity_model_revision", self.sources[5])
        self.assertIn(
            'resolved_config["evaluation"]["met3r_revision"]',
            self.sources[5],
        )
        for field in (
            "base_model_revision",
            "vae_model_revision",
            "adapter_revision",
            "birefnet_revision",
        ):
            self.assertIn(field, self.sources[5])

    def test_hugging_face_downloads_share_the_hub_cache(self):
        self.assertIn(
            'os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")',
            self.sources[0],
        )
        self.assertEqual(
            self.sources[5].count(
                'cache_dir=str(HF_CACHE_DIR / "hub"),'
            ),
            2,
        )
        self.assertNotIn(
            "cache_dir=str(HF_CACHE_DIR),",
            self.sources[5],
        )

    def test_resume_reuses_frozen_revisions_instead_of_main(self):
        source = self.sources[5]
        self.assertIn(
            "creating_frozen_config = not FROZEN_CONFIG.exists()", source
        )
        self.assertIn("if creating_frozen_config:", source)
        self.assertIn(
            "Resume: reusing every previously frozen Hugging Face revision.",
            source,
        )
        self.assertIn(
            "Existing frozen config lacks immutable model revisions.", source
        )
        self.assertIn(
            'resolved_config["evaluation"].get("met3r_revision")', source
        )

    def test_environment_and_checkpoint_evidence_are_copied_atomically(self):
        self.assertIn("PARENT_ENVIRONMENT_FILE", self.sources[4])
        self.assertIn(
            "os.replace(parent_environment_temporary, PARENT_ENVIRONMENT_FILE)",
            self.sources[4],
        )
        self.assertIn(
            'RUNNER_ARTIFACT_ROOT / "configs" / "checkpoint_manifest.json"',
            self.sources[5],
        )
        self.assertIn(
            'RUNNER_ARTIFACT_ROOT / "environment" / "notebook_environment.json"',
            self.sources[5],
        )
        self.assertIn("atomic_copy_file(", self.sources[7])

    def test_manifest_schema_uses_runs_key(self):
        self.assertIn('pilot_manifest.get("runs", [])', self.sources[9])
        self.assertIn('full_manifest.get("runs", [])', self.sources[13])
        self.assertNotIn('pilot_manifest.get("records"', self.joined)
        self.assertNotIn('full_manifest.get("records"', self.joined)

    def test_test_results_schema_is_atomic(self):
        for literal in (
            '"passed"',
            '"compileall_returncode"',
            '"pytest_returncode"',
            '"finished_at"',
            '"command"',
            '"test_results.json"',
            "os.replace(temporary, destination)",
        ):
            self.assertIn(literal, self.sources[7])
        self.assertNotIn(
            'raise RuntimeError("Compile or test gate failed',
            self.sources[7],
        )
        self.assertIn(
            "PILOT/FULL will be blocked", self.sources[7]
        )

    def test_pilot_dry_run_displays_estimate_without_manifest_mutation(self):
        source = self.sources[9]
        for literal in (
            'run_stage(\n    "pilot", dry_run=True, capture_output=True',
            '"planned": pilot_preview.get("planned")',
            '"would_execute": pilot_preview.get("would_execute")',
            '"estimated_output_bytes": pilot_preview.get("estimated_output_bytes")',
            "pilot_manifest_after != pilot_manifest_before",
            'run_stage("pilot")',
        ):
            self.assertIn(literal, source)

    def test_required_stage_order_and_no_shell_magics(self):
        expected_markers = [
            "Configuration",
            "Mount Google Drive",
            "Clone or safely update",
            "Install requirements-colab",
            "Record git commit",
            "Resolve immutable Hugging Face",
            "Validate distinct inputs",
            "Run compileall",
            "CPU preflight",
            "PILOT generation",
            "official MEt3R",
            "Select one candidate",
            "Display the frozen",
            "FULL matrix",
            "trajectory observers",
            "paired statistics",
            "evidence-bounded report",
            "Display core tables",
        ]
        for source, marker in zip(self.sources, expected_markers):
            self.assertIn(marker, source)
        self.assertNotIn("shell=True", self.joined)
        self.assertFalse(
            any(
                line.lstrip().startswith(("!", "%pip", "%cd"))
                for line in self.joined.splitlines()
            )
        )

    def test_real_diagnostic_figure_paths_and_final_status_display(self):
        self.assertIn(
            'diagnostics_dir / "configuration_gate_matrix.png"',
            self.sources[17],
        )
        self.assertIn(
            'diagnostics_dir / "alpha_vs_achieved_kl.png"',
            self.sources[17],
        )
        self.assertIn("A100 40GB", self.joined)
        self.assertIn("L4", self.joined)
        self.assertIn("FINAL_STATUS.json", self.sources[-1])


if __name__ == "__main__":
    unittest.main()
