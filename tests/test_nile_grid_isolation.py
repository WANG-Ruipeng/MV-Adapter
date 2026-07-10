"""Regression tests for resume/manifest experiment isolation."""

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts import run_nile_grid


class TestGridManifestIsolation(unittest.TestCase):
    def test_auto_revision_fingerprints_untracked_source_contents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            tracked = root / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=NILE Test",
                    "-c",
                    "user.email=nile@example.invalid",
                    "commit",
                    "-qm",
                    "base",
                ],
                check=True,
            )
            clean_revision = run_nile_grid._detect_code_revision(root)

            untracked = root / "new_sampler.py"
            untracked.write_text("VALUE = 2\n", encoding="utf-8")
            first_dirty = run_nile_grid._detect_code_revision(root)
            untracked.write_text("VALUE = 3\n", encoding="utf-8")
            second_dirty = run_nile_grid._detect_code_revision(root)

            self.assertNotIn("+dirty.", clean_revision)
            self.assertIn("+dirty.", first_dirty)
            self.assertNotEqual(first_dirty, second_dirty)

    def test_auto_revision_refuses_untracked_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            tracked = root / "tracked.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.py"], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=NILE Test",
                    "-c",
                    "user.email=nile@example.invalid",
                    "commit",
                    "-qm",
                    "base",
                ],
                check=True,
            )
            link = root / "external_sampler.py"
            try:
                link.symlink_to(tracked)
            except OSError as error:
                self.skipTest("symlinks are unavailable: {}".format(error))

            with self.assertRaisesRegex(ValueError, "untracked symlink"):
                run_nile_grid._detect_code_revision(root)

    def test_foreign_experiment_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "input.png"
            input_path.write_bytes(b"dry-run placeholder")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "old-experiment",
                        "code_revision": "old-revision",
                        "runs": [
                            {
                                "run_id": "old-run",
                                "experiment_id": "old-experiment",
                                "code_revision": "old-revision",
                                "status": "succeeded",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "--input",
                str(input_path),
                "--methods",
                "iid_default",
                "--seeds",
                "0",
                "--experiment-id",
                "new-experiment",
                "--code-revision",
                "new-revision",
                "--dry-run",
                "--output-root",
                str(root / "outputs"),
                "--manifest",
                str(manifest_path),
            ]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    run_nile_grid.main(argv)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
