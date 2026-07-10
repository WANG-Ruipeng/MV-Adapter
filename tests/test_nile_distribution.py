"""CPU tests for distribution-preserving Gaussian view coupling."""

import unittest
from unittest.mock import patch

import torch

import mvadapter.nile.nested_elements as nested_elements_module
from mvadapter.nile.covariance import (
    is_positive_semidefinite,
    periodic_camera_rbf_covariance,
    single_tree_covariance,
    stable_cholesky,
    staggered_two_tree_covariance,
    validate_covariance_matrix,
)
from mvadapter.nile.diagnostics import (
    coarse_radial_psd_deviation,
    cross_view_covariance_error,
    cross_view_radial_frequency_correlation,
    diagnose_latents,
    empirical_cross_view_covariance,
    evaluate_distribution_gates,
    lag_autocorrelations,
    radial_psd_deviation,
    spectral_axis_stripe_score,
)
from mvadapter.nile.nested_elements import (
    angles_to_dyadic_slots,
    element_seed_key,
    frequency_dependent_level_weights,
    make_nested_tree_latents,
    nested_tree_spatial_covariance_target,
    tree_ancestor_ids,
)
from mvadapter.nile.spectral_gaussian import (
    camera_rbf_spatial_covariance_target,
    global_spatial_covariance_target,
    make_camera_rbf_correlated_latents,
    make_spectral_global_correlated_latents,
)


ANGLES = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]


class TestCovarianceBuilders(unittest.TestCase):
    def test_periodic_camera_covariance_is_psd_and_angle_aware(self):
        covariance = periodic_camera_rbf_covariance(ANGLES, length_scale=0.8)
        self.assertEqual(tuple(covariance.shape), (6, 6))
        self.assertTrue(torch.allclose(torch.diagonal(covariance), torch.ones(6, dtype=torch.float64)))
        self.assertGreater(float(covariance[0, 1]), float(covariance[0, 3]))
        self.assertTrue(is_positive_semidefinite(covariance))
        validate_covariance_matrix(covariance, psd_atol=1e-10)

        factor = stable_cholesky(covariance)
        self.assertTrue(
            torch.allclose(factor.matmul(factor.mT), covariance, atol=1e-9, rtol=1e-9)
        )

    def test_single_and_staggered_tree_covariances(self):
        slots = torch.arange(8)
        tree_a = single_tree_covariance(slots, tree="a")
        tree_ab = staggered_two_tree_covariance(slots)
        self.assertTrue(is_positive_semidefinite(tree_a))
        self.assertTrue(is_positive_semidefinite(tree_ab))
        self.assertAlmostEqual(float(tree_a[0, 1]), 0.60, places=12)
        self.assertAlmostEqual(float(tree_a[7, 0]), 0.10, places=12)

        # Staggering removes the unique circular seam: 7/0 now has the same
        # covariance as the equivalent coarse boundary 3/4. The two-level
        # hierarchy is intentionally not fully rotation invariant.
        adjacent = [float(tree_ab[index, (index + 1) % 8]) for index in range(8)]
        self.assertAlmostEqual(adjacent[7], adjacent[3], places=12)
        self.assertGreater(adjacent[7], float(tree_a[7, 0]))
        self.assertEqual(sorted(set(round(value, 12) for value in adjacent)), [0.35, 0.45])
        stable_cholesky(tree_a)
        stable_cholesky(tree_ab)


class TestNestedElementMetadata(unittest.TestCase):
    def test_six_real_views_map_to_eight_slots(self):
        slots = angles_to_dyadic_slots(ANGLES)
        self.assertTrue(torch.equal(slots, torch.tensor([0, 1, 2, 4, 6, 7])))
        tree_a = tree_ancestor_ids(slots, tree="a")
        tree_b = tree_ancestor_ids(slots, tree="b")
        self.assertEqual(tree_a["pair"].tolist(), [0, 0, 1, 2, 3, 3])
        self.assertEqual(tree_b["pair"].tolist(), [0, 1, 1, 2, 3, 0])

    def test_static_element_seed_keys_are_stable_and_distinct(self):
        first = element_seed_key(17, "a", "pair", 2)
        self.assertEqual(first, element_seed_key(17, "a", "pair", 2))
        self.assertNotEqual(first, element_seed_key(17, "b", "pair", 2))
        self.assertNotEqual(first, element_seed_key(17, "a", "pair", 3))

    def test_frequency_weights_sum_to_one(self):
        slots = angles_to_dyadic_slots(ANGLES)
        for mode in ("a", "ab"):
            weights = frequency_dependent_level_weights(
                24,
                32,
                slots,
                tree_mode=mode,
                max_correlation=0.45,
                device="cpu",
            )
            total = sum(weights.values())
            self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))
            self.assertTrue(bool((weights["leaf"] >= 0.0).all()))
            self.assertGreater(float(weights["leaf"][-1, -1]), float(weights["leaf"][0, 0]))

    def test_dc_shared_budget_and_tree_maps_have_fixed_semantics(self):
        slots = angles_to_dyadic_slots(ANGLES)
        maps = {}
        for mode in ("a", "ab"):
            maps[mode] = frequency_dependent_level_weights(
                16,
                16,
                slots,
                tree_mode=mode,
                max_correlation=0.60,
                device="cpu",
            )
            dc = tuple(
                float(maps[mode][level][0, 0])
                for level in ("root", "coarse", "pair", "leaf")
            )
            for actual, expected in zip(dc, (0.10, 0.20, 0.30, 0.40)):
                self.assertAlmostEqual(actual, expected, places=6)
        for level in ("root", "coarse", "pair", "leaf"):
            self.assertTrue(torch.equal(maps["a"][level], maps["ab"][level]))

    def test_selected_view_targets_are_principal_submatrices(self):
        full_angles = [45.0 * index for index in range(8)]
        selected = [0, 1, 2, 4, 6, 7]
        for mode in ("a", "ab"):
            with self.subTest(mode=mode):
                full = nested_tree_spatial_covariance_target(
                    full_angles,
                    16,
                    16,
                    max_correlation=0.60,
                    tree_mode=mode,
                )
                subset = nested_tree_spatial_covariance_target(
                    ANGLES,
                    16,
                    16,
                    max_correlation=0.60,
                    tree_mode=mode,
                )
                indices = torch.tensor(selected)
                expected = full.index_select(0, indices).index_select(1, indices)
                self.assertTrue(torch.allclose(subset, expected, atol=1e-12, rtol=0.0))


class TestZeroStrengthIIDEquivalence(unittest.TestCase):
    shape = (2, 6, 3, 9, 11)
    seed = 1234

    def _manual_iid(self):
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        return torch.randn(
            self.shape[0] * self.shape[1],
            *self.shape[2:],
            generator=generator,
            dtype=torch.float32,
        )

    def test_all_zero_strength_paths_are_bit_exact_manual_iid(self):
        expected = self._manual_iid()
        common = dict(
            batch_size=self.shape[0],
            num_views=self.shape[1],
            channels=self.shape[2],
            height=self.shape[3],
            width=self.shape[4],
            device="cpu",
            dtype=torch.float32,
            seed=self.seed,
            max_correlation=0.0,
        )
        global_latents = make_spectral_global_correlated_latents(**common)
        camera_latents = make_camera_rbf_correlated_latents(
            **common, view_angles=ANGLES
        )
        tree_a = make_nested_tree_latents(
            **common, view_angles=ANGLES, tree_mode="a"
        )
        tree_ab = make_nested_tree_latents(
            **common, view_angles=ANGLES, tree_mode="ab"
        )
        for name, latents in (
            ("global", global_latents),
            ("camera", camera_latents),
            ("tree_a", tree_a),
            ("tree_ab", tree_ab),
        ):
            with self.subTest(name=name):
                self.assertTrue(torch.equal(latents, expected))

    def test_zero_strength_consumes_only_the_local_iid_draw(self):
        expected_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        torch.randn(
            self.shape[0] * self.shape[1],
            *self.shape[2:],
            generator=expected_generator,
            dtype=torch.float32,
        )
        expected_next = torch.randn(7, generator=expected_generator)

        actual_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        make_nested_tree_latents(
            self.shape[0],
            self.shape[1],
            self.shape[2],
            self.shape[3],
            self.shape[4],
            ANGLES,
            device="cpu",
            dtype=torch.float32,
            generator=actual_generator,
            max_correlation=0.0,
            tree_mode="ab",
        )
        actual_next = torch.randn(7, generator=actual_generator)
        self.assertTrue(torch.equal(actual_next, expected_next))


class TestStaticNestedElementBank(unittest.TestCase):
    batch_size = 2
    channels = 2
    height = 7
    width = 9
    seed = 2468
    full_angles = [45.0 * index for index in range(8)]

    def _sample(self, angles, mode, *, seed=None, generator=None):
        return make_nested_tree_latents(
            self.batch_size,
            len(angles),
            self.channels,
            self.height,
            self.width,
            angles,
            device="cpu",
            dtype=torch.float32,
            seed=seed,
            generator=generator,
            max_correlation=0.45,
            frequency_scale=0.12,
            tree_mode=mode,
        ).reshape(
            self.batch_size,
            len(angles),
            self.channels,
            self.height,
            self.width,
        )

    def test_positive_strength_is_bitwise_stable_under_view_reordering(self):
        permutation = [5, 0, 7, 2, 1, 6, 3, 4]
        permuted_angles = [self.full_angles[index] for index in permutation]
        indices = torch.tensor(permutation)
        for mode in ("a", "ab"):
            with self.subTest(mode=mode):
                full = self._sample(self.full_angles, mode, seed=self.seed)
                permuted = self._sample(permuted_angles, mode, seed=self.seed)
                self.assertTrue(
                    torch.equal(permuted, full.index_select(1, indices))
                )

    def test_positive_strength_is_bitwise_stable_for_slot_subsets(self):
        selected = [0, 2, 5, 7]
        subset_angles = [self.full_angles[index] for index in selected]
        indices = torch.tensor(selected)
        for mode in ("a", "ab"):
            with self.subTest(mode=mode):
                full = self._sample(self.full_angles, mode, seed=self.seed)
                subset = self._sample(subset_angles, mode, seed=self.seed)
                self.assertTrue(torch.equal(subset, full.index_select(1, indices)))

    def test_tree_a_named_elements_are_identical_in_a_and_ab_modes(self):
        original_draw = nested_elements_module._draw_keyed_element

        def capture(mode):
            records = {}

            def recording_draw(*args, **kwargs):
                value = original_draw(*args, **kwargs)
                records[
                    (kwargs["tree"], kwargs["level"], kwargs["group_id"])
                ] = value.clone()
                return value

            with patch.object(
                nested_elements_module,
                "_draw_keyed_element",
                side_effect=recording_draw,
            ):
                self._sample(self.full_angles, mode, seed=self.seed)
            return records

        tree_a = capture("a")
        tree_ab = capture("ab")
        self.assertTrue(tree_a)
        for key, value in tree_a.items():
            with self.subTest(element=key):
                self.assertEqual(key[0], "a")
                self.assertTrue(torch.equal(value, tree_ab[key]))

    def test_positive_strength_consumes_only_the_local_iid_draw(self):
        num_views = len(self.full_angles)
        expected_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        torch.randn(
            self.batch_size * num_views,
            self.channels,
            self.height,
            self.width,
            generator=expected_generator,
            dtype=torch.float32,
        )

        actual_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._sample(self.full_angles, "ab", generator=actual_generator)
        self.assertTrue(
            torch.equal(actual_generator.get_state(), expected_generator.get_state())
        )

    def test_element_bank_uses_the_complete_pre_draw_rng_state(self):
        def sample_after_advance(count):
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            torch.randn(count, generator=generator)
            return self._sample(self.full_angles, "ab", generator=generator)

        first = sample_after_advance(11)
        repeated = sample_after_advance(11)
        differently_advanced = sample_after_advance(12)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, differently_advanced))


class TestDistributionPreservation(unittest.TestCase):
    def test_camera_sampler_runs_on_small_odd_spatial_shape(self):
        latents = make_camera_rbf_correlated_latents(
            1,
            3,
            2,
            5,
            7,
            [0.0, 45.0, 180.0],
            device="cpu",
            dtype=torch.float32,
            seed=13,
            max_correlation=0.45,
        )
        self.assertEqual(tuple(latents.shape), (3, 2, 5, 7))
        self.assertTrue(bool(torch.isfinite(latents).all()))

    def test_iid_diagnostic_ensemble_passes_worst_case_gates(self):
        batch_size, num_views, channels, height, width = 16, 6, 4, 32, 32
        generator = torch.Generator(device="cpu").manual_seed(101)
        latents = torch.randn(
            batch_size * num_views,
            channels,
            height,
            width,
            generator=generator,
        )
        report = diagnose_latents(
            latents,
            batch_size=batch_size,
            num_views=num_views,
            target_covariance=torch.eye(num_views, dtype=torch.float64),
        )
        gates = evaluate_distribution_gates(report)
        self.assertTrue(gates["passed"], gates)
        self.assertIn("per_view_max", gates["checks"]["mean"])
        self.assertIn("per_view_max", gates["checks"]["lag_autocorrelation"])
        self.assertIn("coarse_band_max", gates["checks"]["radial_psd"])
        self.assertIn(
            "offdiag_mae", report["cross_view_covariance_error"]
        )

    def test_gate_uses_worst_view_mean_and_off_diagonal_covariance_error(self):
        batch_size, num_views = 16, 6
        generator = torch.Generator(device="cpu").manual_seed(102)
        views = torch.randn(
            batch_size,
            num_views,
            4,
            32,
            32,
            generator=generator,
        )
        views[:, 0] += 0.025
        report = diagnose_latents(
            views.reshape(batch_size * num_views, 4, 32, 32),
            batch_size=batch_size,
            num_views=num_views,
            target_covariance=torch.eye(num_views, dtype=torch.float64),
        )
        gates = evaluate_distribution_gates(report)
        self.assertLess(abs(report["global"]["mean"]), 0.01)
        self.assertFalse(gates["checks"]["mean"]["passed"])

        diagonal_only = cross_view_covariance_error(
            1.5 * torch.eye(3), torch.eye(3)
        )
        self.assertGreater(diagonal_only["mae"], 0.03)
        self.assertEqual(diagonal_only["offdiag_mae"], 0.0)
        report["cross_view_covariance_error"] = diagonal_only
        covariance_gate = evaluate_distribution_gates(report)["checks"][
            "cross_view_covariance"
        ]
        self.assertTrue(covariance_gate["passed"])
        self.assertEqual(covariance_gate["value"], 0.0)

    def test_coarse_psd_worst_band_detects_low_frequency_colour(self):
        generator = torch.Generator(device="cpu").manual_seed(103)
        latents = torch.randn(16 * 6, 4, 32, 32, generator=generator)
        x = torch.arange(32, dtype=torch.float32)
        wave = torch.cos(2.0 * torch.pi * x / 32.0)[None, None, None, :]
        coloured = latents + 0.35 * wave
        report = coarse_radial_psd_deviation(coloured)
        self.assertGreater(report["max"], 0.05)

    def test_global_sampler_passes_hard_distribution_gates(self):
        batch_size, num_views, channels, height, width = 16, 6, 4, 32, 32
        latents = make_spectral_global_correlated_latents(
            batch_size,
            num_views,
            channels,
            height,
            width,
            device="cpu",
            dtype=torch.float32,
            seed=29,
            max_correlation=0.45,
            frequency_scale=0.12,
        )
        target = global_spatial_covariance_target(
            num_views,
            height,
            width,
            max_correlation=0.45,
            frequency_scale=0.12,
        )
        report = diagnose_latents(
            latents,
            batch_size=batch_size,
            num_views=num_views,
            target_covariance=target,
        )
        gates = evaluate_distribution_gates(report)
        self.assertTrue(gates["passed"], gates)
        frequency = report["cross_view_frequency"]
        self.assertEqual(len(frequency["correlation"]), len(frequency["radii"]))
        self.assertEqual(
            tuple(torch.tensor(frequency["correlation"]).shape[1:]),
            (num_views, num_views),
        )
        low_correlation = frequency["correlation"][0][0][1]
        high_correlation = frequency["correlation"][-1][0][1]
        self.assertGreater(low_correlation, high_correlation)

    def test_camera_sampler_tracks_target_covariance(self):
        batch_size, num_views, channels, height, width = 12, 6, 4, 24, 24
        latents = make_camera_rbf_correlated_latents(
            batch_size,
            num_views,
            channels,
            height,
            width,
            ANGLES,
            device="cpu",
            dtype=torch.float32,
            seed=31,
            max_correlation=0.45,
            frequency_scale=0.12,
            length_scale=0.8,
        )
        target = camera_rbf_spatial_covariance_target(
            ANGLES,
            height,
            width,
            max_correlation=0.45,
            frequency_scale=0.12,
            length_scale=0.8,
        )
        empirical = empirical_cross_view_covariance(latents, batch_size, num_views)
        self.assertLess(float((empirical - target).abs().mean()), 0.03)
        self.assertGreater(float(empirical[0, 1]), float(empirical[0, 3]))
        self.assertLess(abs(float(latents.std(unbiased=False)) - 1.0), 0.015)
        self.assertLess(lag_autocorrelations(latents)["max_abs"], 0.025)
        self.assertLess(radial_psd_deviation(latents), 0.06)

    def test_both_nested_tree_modes_track_their_targets(self):
        batch_size, num_views, channels, height, width = 12, 6, 4, 24, 24
        for index, mode in enumerate(("a", "ab")):
            with self.subTest(mode=mode):
                latents = make_nested_tree_latents(
                    batch_size,
                    num_views,
                    channels,
                    height,
                    width,
                    ANGLES,
                    device="cpu",
                    dtype=torch.float32,
                    seed=40 + index,
                    max_correlation=0.45,
                    frequency_scale=0.12,
                    tree_mode=mode,
                )
                target = nested_tree_spatial_covariance_target(
                    ANGLES,
                    height,
                    width,
                    max_correlation=0.45,
                    frequency_scale=0.12,
                    tree_mode=mode,
                )
                empirical = empirical_cross_view_covariance(
                    latents, batch_size, num_views
                )
                self.assertLess(float((empirical - target).abs().mean()), 0.03)
                self.assertLess(abs(float(latents.std(unbiased=False)) - 1.0), 0.015)
                self.assertLess(radial_psd_deviation(latents), 0.06)

    def test_hard_gate_rejects_spatially_repeated_stripes(self):
        generator = torch.Generator(device="cpu").manual_seed(7)
        stripe = torch.randn(12, 2, 32, 1, generator=generator).expand(-1, -1, -1, 32)
        target = torch.eye(6, dtype=torch.float64)
        report = diagnose_latents(
            stripe,
            batch_size=2,
            num_views=6,
            target_covariance=target,
        )
        gates = evaluate_distribution_gates(report)
        self.assertFalse(gates["passed"])
        self.assertFalse(gates["checks"]["lag_autocorrelation"]["passed"])
        self.assertFalse(gates["checks"]["radial_psd"]["passed"])
        self.assertFalse(gates["checks"]["axis_stripes"]["passed"])
        self.assertGreater(spectral_axis_stripe_score(stripe), 1.0)


if __name__ == "__main__":
    unittest.main()
