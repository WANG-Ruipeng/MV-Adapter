"""Focused CPU tests for the dependency-light NILE core."""

import unittest

import torch

from mvadapter.nile import (
    NILECallbackConfig,
    NILEConfig,
    NILEViewTimeCallback,
    SobolBackend,
    build_patch_rho_map,
    gaussian_blur_latent,
    inverse_normal_cdf,
    linear_rho,
    low_high_split,
    make_initial_latents,
    morton2d,
    patch_morton_order,
    standardize_like,
    standardize_unit,
)


class TestNILEOps(unittest.TestCase):
    def test_standardization_is_per_sample(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        ref = torch.randn_like(x) * 3.0 + 2.0

        unit = standardize_unit(x)
        dims = (1, 2, 3)
        self.assertTrue(torch.allclose(unit.mean(dims), torch.zeros(2), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                unit.var(dims, unbiased=False), torch.ones(2), atol=1e-5
            )
        )

        matched = standardize_like(x, ref)
        self.assertTrue(
            torch.allclose(matched.mean(dims), ref.mean(dims), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                matched.var(dims, unbiased=False),
                ref.var(dims, unbiased=False),
                atol=1e-4,
            )
        )

    def test_large_kernel_handles_tiny_latent(self):
        x = torch.randn(1, 2, 2, 3)
        blurred = gaussian_blur_latent(x, kernel_size=11, sigma=2.5)
        low, high = low_high_split(x, kernel_size=11, sigma=2.5)
        self.assertEqual(blurred.shape, x.shape)
        self.assertEqual(blurred.dtype, x.dtype)
        self.assertTrue(torch.isfinite(blurred).all())
        self.assertTrue(torch.allclose(low + high, x, atol=1e-6))

    def test_invalid_blur_parameters_raise(self):
        x = torch.randn(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            gaussian_blur_latent(x, kernel_size=4)
        with self.assertRaises(ValueError):
            gaussian_blur_latent(x, sigma=0.0)


class TestMortonAndSequence(unittest.TestCase):
    def test_morton_codes_and_partial_edge_patches(self):
        x = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(morton2d(x, y), torch.tensor([0, 1, 2, 3])))

        coords = patch_morton_order(5, 3, patch_size=2, device="cpu")
        self.assertEqual(tuple(coords.shape), (6, 2))
        self.assertEqual(
            {tuple(coord) for coord in coords.tolist()},
            {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)},
        )
        codes = morton2d(coords[:, 1], coords[:, 0])
        self.assertTrue(bool(torch.all(codes[1:] >= codes[:-1])))

    def test_sobol_is_reproducible_and_resettable(self):
        first = SobolBackend(dim=3, scramble=True, seed=17)
        second = SobolBackend(dim=3, scramble=True, seed=17)
        a = first.draw(8, device="cpu")
        b = second.draw(8, device="cpu")
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(first.reset().draw(8, device="cpu"), a))

    def test_inverse_normal_cdf_clips_endpoints(self):
        z = inverse_normal_cdf(torch.tensor([0.0, 0.5, 1.0]))
        self.assertTrue(torch.isfinite(z).all())
        self.assertAlmostEqual(float(z[1]), 0.0, places=6)
        self.assertAlmostEqual(float(z[0]), -float(z[2]), places=4)


class TestNILESamplers(unittest.TestCase):
    def _make(self, mode, seed=23):
        return make_initial_latents(
            batch_size=1,
            num_views=3,
            channels=2,
            latent_h=5,
            latent_w=7,
            device="cpu",
            dtype=torch.float32,
            cfg=NILEConfig(
                mode=mode,
                seed=seed,
                blur_kernel=11,
                blur_sigma=2.0,
                patch_size=4,
            ),
        )

    def test_all_modes_have_expected_shape_and_are_reproducible(self):
        modes = (
            "iid",
            "shared",
            "lowpass_shared",
            "flat_sobol",
            "nile_v",
            "nile_vtp",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                a = self._make(mode)
                b = self._make(mode)
                self.assertEqual(tuple(a.shape), (3, 2, 5, 7))
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.isfinite(a).all())

    def test_shared_mode_reuses_each_batch_parent(self):
        shared = self._make("shared").reshape(1, 3, 2, 5, 7)
        self.assertTrue(torch.equal(shared[:, 0], shared[:, 1]))
        self.assertTrue(torch.equal(shared[:, 1], shared[:, 2]))

    def test_flat_sobol_scramble_setting_is_effective(self):
        common = dict(
            batch_size=1,
            num_views=3,
            channels=2,
            latent_h=5,
            latent_w=7,
            device="cpu",
            dtype=torch.float32,
        )
        scrambled = make_initial_latents(
            **common,
            cfg=NILEConfig(mode="flat_sobol", seed=23, qmc_scramble=True),
        )
        unscrambled = make_initial_latents(
            **common,
            cfg=NILEConfig(mode="flat_sobol", seed=23, qmc_scramble=False),
        )
        self.assertFalse(torch.equal(scrambled, unscrambled))

    def test_invalid_correlation_is_rejected(self):
        with self.assertRaises(ValueError):
            NILEConfig(rho_geo=1.01)
        with self.assertRaises(ValueError):
            NILEConfig(rho_geo=-0.01)


class TestNILECallback(unittest.TestCase):
    class _Pipe:
        _num_timesteps = 10

    def test_schedule_and_partial_patch_map(self):
        cfg = NILECallbackConfig(
            num_views=2,
            rho_start=0.4,
            rho_end=0.0,
            active_ratio=0.6,
        )
        self.assertAlmostEqual(linear_rho(0, 10, cfg), 0.4)
        self.assertAlmostEqual(linear_rho(3, 10, cfg), 0.2)
        self.assertEqual(linear_rho(6, 10, cfg), 0.0)

        rho_map = build_patch_rho_map(
            h=5,
            w=7,
            patch_size=4,
            base_rho=0.4,
            zindex_strength=0.25,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(tuple(rho_map.shape), (1, 1, 1, 5, 7))
        self.assertTrue(torch.isfinite(rho_map).all())
        self.assertTrue(bool(torch.all((rho_map >= 0.0) & (rho_map <= 1.0))))

    def test_callback_preserves_shape_dtype_and_marginal(self):
        cfg = NILECallbackConfig(
            mode="nile_vtp",
            num_views=2,
            batch_size=1,
            rho_start=0.4,
            active_ratio=0.6,
            blur_kernel=11,
            blur_sigma=2.0,
            patch_size=4,
            preserve_marginal=True,
        )
        callback = NILEViewTimeCallback(cfg)
        original = torch.randn(2, 2, 5, 7)
        result = callback(
            self._Pipe(),
            step=0,
            timestep=999,
            callback_kwargs={"latents": original},
        )["latents"]
        dims = (1, 2, 3)
        self.assertEqual(result.shape, original.shape)
        self.assertEqual(result.dtype, original.dtype)
        self.assertTrue(
            torch.allclose(result.mean(dims), original.mean(dims), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                result.var(dims, unbiased=False),
                original.var(dims, unbiased=False),
                atol=1e-4,
            )
        )

    def test_invalid_callback_batch_is_rejected(self):
        callback = NILEViewTimeCallback(
            NILECallbackConfig(mode="nile_vt", num_views=3, batch_size=1)
        )
        with self.assertRaises(ValueError):
            callback(
                self._Pipe(),
                step=0,
                timestep=999,
                callback_kwargs={"latents": torch.randn(2, 1, 4, 4)},
            )


if __name__ == "__main__":
    unittest.main()
