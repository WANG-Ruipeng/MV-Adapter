import json
import unittest

import torch

from mvadapter.nile.basis import build_dct2_basis
from mvadapter.nile.lowrank_coupling import (
    SUPPORTED_COUPLING_METHODS,
    apply_latent_coupling,
    correlate_orthonormal_subspace,
    make_shared_full_latents,
)


def _coefficient_covariance(
    latents: torch.Tensor,
    basis: torch.Tensor,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    flat = latents.reshape(batch_size, num_views, -1).to(torch.float64)
    coefficients = flat @ basis.to(torch.float64)
    samples = coefficients.permute(0, 2, 1).reshape(-1, num_views)
    samples = samples - samples.mean(dim=0, keepdim=True)
    return samples.mT @ samples / (samples.shape[0] - 1)


class TestLowRankCoupling(unittest.TestCase):
    def setUp(self):
        self.num_views = 4
        self.channels = 2
        self.height = 6
        self.width = 8
        self.rank = 8
        self.basis = build_dct2_basis(
            self.channels, self.height, self.width, self.rank
        )
        self.target = torch.tensor(
            [
                [1.0, 0.55, 0.20, 0.10],
                [0.55, 1.0, 0.35, 0.20],
                [0.20, 0.35, 1.0, 0.45],
                [0.10, 0.20, 0.45, 1.0],
            ],
            dtype=torch.float64,
        )
        self.assertGreater(float(torch.linalg.eigvalsh(self.target).min()), 0.0)

    def _latents(self, batch_size=3, dtype=torch.float32, seed=123):
        return torch.randn(
            batch_size * self.num_views,
            self.channels,
            self.height,
            self.width,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
        )

    def test_alpha_zero_is_exact_same_object_passthrough(self):
        iid = self._latents()
        output, metadata = correlate_orthonormal_subspace(
            iid,
            self.basis,
            self.target,
            self.num_views,
            alpha=0.0,
            return_metadata=True,
        )
        self.assertIs(output, iid)
        self.assertEqual(output.data_ptr(), iid.data_ptr())
        self.assertTrue(torch.equal(output, iid))
        self.assertTrue(metadata["identity_passthrough"])
        self.assertEqual(
            metadata["covariance_factor_method"], "none_identity_passthrough"
        )
        json.dumps(metadata, allow_nan=False)

    def test_identity_covariance_is_exact_same_object_passthrough(self):
        iid = self._latents()
        output = correlate_orthonormal_subspace(
            iid,
            self.basis,
            torch.eye(self.num_views, dtype=torch.float64),
            self.num_views,
        )
        self.assertIs(output, iid)
        self.assertEqual(output.data_ptr(), iid.data_ptr())

    def test_batch_greater_than_one_reproducible_and_metadata_complete(self):
        iid = self._latents(batch_size=5)
        first, metadata = correlate_orthonormal_subspace(
            iid,
            self.basis,
            self.target,
            self.num_views,
            alpha=0.65,
            return_metadata=True,
        )
        second = correlate_orthonormal_subspace(
            iid, self.basis, self.target, self.num_views, alpha=0.65
        )
        self.assertEqual(tuple(first.shape), tuple(iid.shape))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(metadata["batch_size"], 5)
        self.assertEqual(metadata["num_views"], self.num_views)
        self.assertEqual(metadata["input_dtype"], "float32")
        self.assertEqual(metadata["output_dtype"], "float32")
        self.assertIn(metadata["covariance_factor_method"], ("cholesky", "symmetric_eigh"))
        self.assertFalse(metadata["per_sample_standardization"])
        json.dumps(metadata, allow_nan=False)

    def test_empirical_marginals_and_coefficient_covariance(self):
        batch_size = 4096
        iid = self._latents(batch_size=batch_size, seed=81)
        alpha = 0.60
        output = correlate_orthonormal_subspace(
            iid,
            self.basis,
            self.target,
            self.num_views,
            alpha=alpha,
        )
        effective = (1.0 - alpha) * torch.eye(
            self.num_views, dtype=torch.float64
        ) + alpha * self.target
        empirical = _coefficient_covariance(
            output, self.basis, batch_size, self.num_views
        )
        self.assertLess(float((empirical - effective).abs().max()), 0.035)

        views = output.reshape(batch_size, self.num_views, -1).to(torch.float64)
        means = views.mean(dim=(0, 2))
        stds = views.permute(1, 0, 2).reshape(self.num_views, -1).std(
            dim=1, unbiased=False
        )
        self.assertLess(float(means.abs().max()), 0.01)
        self.assertLess(float((stds - 1.0).abs().max()), 0.01)

    def test_full_space_nonzero_lag_remains_near_zero(self):
        batch_size = 2048
        iid = self._latents(batch_size=batch_size, seed=92)
        output = correlate_orthonormal_subspace(
            iid,
            self.basis,
            self.target,
            self.num_views,
            alpha=0.8,
        ).reshape(
            batch_size,
            self.num_views,
            self.channels,
            self.height,
            self.width,
        )
        lag_x = float((output[..., :-1] * output[..., 1:]).mean())
        lag_y = float((output[..., :-1, :] * output[..., 1:, :]).mean())
        self.assertLess(abs(lag_x), 0.01)
        self.assertLess(abs(lag_y), 0.01)

    def test_no_per_sample_standardization(self):
        iid = self._latents(batch_size=2, seed=111)
        shaped = iid.reshape(
            2,
            self.num_views,
            self.channels,
            self.height,
            self.width,
        )
        shaped[0, 0].add_(3.0)
        output = correlate_orthonormal_subspace(
            iid,
            self.basis,
            self.target,
            self.num_views,
            alpha=0.7,
        ).reshape_as(shaped)
        # The DC mode is not in the basis, so the deliberately unusual sample
        # mean lives in the residual and must not be centred away.
        self.assertGreater(float(output[0, 0].mean()), 2.5)
        self.assertGreater(float(output[0, 0].mean()), float(output[0, 1].mean()) + 2.0)

    def test_original_dtype_and_device_are_preserved(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                iid = self._latents(batch_size=2, dtype=dtype)
                output = correlate_orthonormal_subspace(
                    iid,
                    self.basis,
                    self.target,
                    self.num_views,
                    alpha=0.5,
                )
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.device, iid.device)
                self.assertTrue(bool(torch.isfinite(output).all()))

        if torch.cuda.is_available():
            iid = self._latents(batch_size=2).cuda()
            output = correlate_orthonormal_subspace(
                iid,
                self.basis,
                self.target,
                self.num_views,
                alpha=0.5,
            )
            self.assertEqual(output.device.type, "cuda")

    def test_shared_full_respects_batch_boundaries(self):
        iid = self._latents(batch_size=3)
        output, metadata = make_shared_full_latents(
            iid, self.num_views, return_metadata=True
        )
        views = output.reshape(
            3,
            self.num_views,
            self.channels,
            self.height,
            self.width,
        )
        for batch in range(3):
            for view in range(1, self.num_views):
                self.assertTrue(torch.equal(views[batch, 0], views[batch, view]))
        self.assertFalse(torch.equal(views[0, 0], views[1, 0]))
        self.assertTrue(metadata["degenerate_joint_distribution"])
        self.assertIsNone(metadata["joint_kl_nats"])
        json.dumps(metadata, allow_nan=False)

    def test_dispatcher_supports_all_formal_method_names(self):
        self.assertEqual(
            set(SUPPORTED_COUPLING_METHODS),
            {
                "iid_external",
                "shared_full",
                "lowrank_camera_rbf",
                "lowrank_nested_tree_a",
                "lowrank_nested_tree_ab",
            },
        )
        iid = self._latents()
        external = apply_latent_coupling(iid, "iid_external", self.num_views)
        self.assertIs(external, iid)
        for method in (
            "lowrank_camera_rbf",
            "lowrank_nested_tree_a",
            "lowrank_nested_tree_ab",
        ):
            with self.subTest(method=method):
                output, metadata = apply_latent_coupling(
                    iid,
                    method,
                    self.num_views,
                    basis=self.basis,
                    view_covariance=self.target,
                    alpha=0.4,
                    return_metadata=True,
                )
                self.assertEqual(tuple(output.shape), tuple(iid.shape))
                self.assertEqual(metadata["method"], method)

    def test_rejects_nonorthonormal_basis_and_bad_covariance(self):
        iid = self._latents()
        bad_basis = self.basis.clone()
        bad_basis[:, 0] *= 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            correlate_orthonormal_subspace(
                iid, bad_basis, self.target, self.num_views
            )

        bad_covariance = self.target.clone()
        bad_covariance[0, 0] = 0.9
        with self.assertRaisesRegex(ValueError, "unit diagonal"):
            correlate_orthonormal_subspace(
                iid, self.basis, bad_covariance, self.num_views
            )


if __name__ == "__main__":
    unittest.main()
