"""Regression tests for the read-only denoising trajectory observer."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mvadapter.nile.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryObserver,
    compute_paired_delta,
    load_trajectory_npz,
    milestone_step_indices,
)


class _FakePipeline:
    def __init__(self, total_steps: int):
        self._num_timesteps = total_steps


def _basis(dimension: int = 12, rank: int = 4) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(9182)
    matrix = torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(matrix, mode="reduced").Q.to(torch.float32)


def _latents() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(2468)
    return torch.randn(6, 1, 3, 4, generator=generator)


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class TestTrajectoryObserver(unittest.TestCase):
    def test_callback_is_strictly_mutation_free(self):
        initial = _latents()
        observer = TrajectoryObserver(
            _basis(), num_views=3, batch_size=2, total_steps=10
        )
        initial_copy = initial.clone()
        initial_version = initial._version
        observer.record_initial(initial, timestep=999)
        self.assertTrue(torch.equal(initial, initial_copy))
        self.assertEqual(initial._version, initial_version)

        current = initial * 0.9
        current_copy = current.clone()
        current_version = current._version
        callback_kwargs = {"latents": current, "sentinel": torch.tensor(7)}
        result = observer(_FakePipeline(10), 0, torch.tensor(900), callback_kwargs)

        self.assertIs(result, callback_kwargs)
        self.assertIs(result["latents"], current)
        self.assertTrue(torch.equal(current, current_copy))
        self.assertEqual(current._version, current_version)

    def test_default_milestone_selection(self):
        expected = {0.0: -1, 0.10: 1, 0.25: 4, 0.50: 9, 0.75: 14, 1.0: 19}
        self.assertEqual(milestone_step_indices(20), expected)

        initial = _latents()
        observer = TrajectoryObserver(_basis(), num_views=3, total_steps=20)
        observer.record_initial(initial, timestep=1000)
        current = initial
        pipeline = _FakePipeline(20)
        for step in range(20):
            current = current * 0.99 + (step + 1) * 1e-4
            observer(pipeline, step, 999 - step, {"latents": current})

        self.assertEqual(
            observer.captured_milestones,
            ("initial", "10%", "25%", "50%", "75%", "final"),
        )
        self.assertEqual(
            tuple(snapshot.step for snapshot in observer.snapshots),
            (-1, 1, 4, 9, 14, 19),
        )
        self.assertTrue(
            np.allclose(observer.snapshots[0].g_t, np.ones(2), atol=0.0, rtol=0.0)
        )

    def test_saved_npz_schema_and_csv(self):
        initial = _latents()
        observer = TrajectoryObserver(_basis(), num_views=3, total_steps=4)
        observer.record_initial(initial, timestep=1000)
        pipeline = _FakePipeline(4)
        current = initial
        for step in range(4):
            current = current * 0.95 + 0.01
            observer(pipeline, step, 900 - step, {"latents": current})

        with tempfile.TemporaryDirectory() as directory:
            paths = observer.save(Path(directory) / "trajectory", make_plot=False)
            self.assertTrue(paths["npz"].is_file())
            self.assertTrue(paths["csv"].is_file())
            self.assertIsNone(paths["plot"])

            arrays = load_trajectory_npz(paths["npz"])
            self.assertEqual(arrays["schema_version"].item(), TRAJECTORY_SCHEMA_VERSION)
            self.assertEqual(
                arrays["milestones"].tolist(),
                ["initial", "10%", "25%", "50%", "75%", "final"],
            )
            self.assertEqual(arrays["basis_coefficients"].shape, (6, 2, 3, 4))
            self.assertEqual(arrays["view_correlation"].shape, (6, 2, 3, 3))
            self.assertEqual(arrays["offdiag_frobenius"].shape, (6, 2))
            self.assertEqual(
                arrays["per_view_coefficient_norm"].shape, (6, 2, 3)
            )
            self.assertEqual(arrays["g_t"].shape, (6, 2))
            self.assertEqual(arrays["timesteps"].shape, (6,))

            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]["milestone"], "initial")

    def test_observer_on_off_output_hash_is_identical(self):
        initial = _latents()

        def simulate(enabled: bool) -> str:
            value = initial.clone()
            observer = None
            if enabled:
                observer = TrajectoryObserver(_basis(), num_views=3, total_steps=10)
                observer.record_initial(value, timestep=1000)
            pipeline = _FakePipeline(10)
            for step in range(10):
                # A deterministic stand-in for scheduler.step.  The observer
                # must not change this numerical trajectory or consume RNG.
                value = value * 0.973 + torch.sin(
                    torch.tensor(float(step), dtype=value.dtype)
                ) * 1e-3
                if observer is not None:
                    callback_kwargs = {"latents": value}
                    result = observer(pipeline, step, 999 - step, callback_kwargs)
                    value = result["latents"]
            return _tensor_hash(value)

        self.assertEqual(simulate(False), simulate(True))

    def test_paired_delta_uses_matching_basis_coefficients(self):
        initial = _latents()
        iid = TrajectoryObserver(_basis(), num_views=3, total_steps=1)
        correlated = TrajectoryObserver(_basis(), num_views=3, total_steps=1)
        iid.record_initial(initial)
        correlated.record_initial(initial * 2.0)
        iid(_FakePipeline(1), 0, 0, {"latents": initial})
        correlated(_FakePipeline(1), 0, 0, {"latents": initial * 2.0})

        paired = compute_paired_delta(correlated, iid)
        # With a one-step scheduler all five post-initial milestones refer to
        # the same final state, but remain separately labelled in the schema.
        self.assertEqual(paired["delta_t"].shape, (6, 2))
        self.assertTrue(np.allclose(paired["delta_t"], 1.0, atol=1e-7))

    def test_callback_requires_true_initial_state(self):
        observer = TrajectoryObserver(_basis(), num_views=3, total_steps=1)
        with self.assertRaisesRegex(RuntimeError, "record_initial"):
            observer(_FakePipeline(1), 0, 0, {"latents": _latents()})


if __name__ == "__main__":
    unittest.main()
