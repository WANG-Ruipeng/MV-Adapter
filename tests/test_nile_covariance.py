"""Regression tests for low-rank equal-KL view covariance construction."""

import json
import math
import unittest

import torch

from mvadapter.nile.covariance import (
    DEFAULT_LOWRANK_TREE_WEIGHTS,
    azimuths_to_slots,
    calibrate_alpha_for_target_kl,
    covariance_metadata,
    joint_gaussian_kl,
    mix_covariance_with_identity,
    periodic_camera_rbf_covariance,
    tree_a_covariance,
    tree_ab_covariance,
    tree_b_covariance,
    tree_covariance_from_azimuths,
    validate_covariance_matrix,
)


FORMAL_AZIMUTHS = [0.0, 45.0, 90.0, 180.0, 270.0, 315.0]
ALL_SLOTS_AZIMUTHS = [float(value) for value in range(0, 360, 45)]


class TestPeriodicCameraRBF(unittest.TestCase):
    def test_ell_deg_matches_periodic_kernel_and_wraparound(self):
        angles = [90.0, 315.0, 0.0, 180.0, 45.0]
        covariance = periodic_camera_rbf_covariance(angles, ell_deg=45.0)
        validate_covariance_matrix(covariance, psd_atol=1e-10)
        self.assertTrue(
            torch.equal(torch.diagonal(covariance), torch.ones(5, dtype=torch.float64))
        )

        ell = math.pi / 4.0
        expected_45 = math.exp(
            -2.0 * math.sin(math.pi / 8.0) ** 2 / (ell * ell)
        )
        self.assertAlmostEqual(float(covariance[1, 2]), expected_45, places=12)
        self.assertAlmostEqual(float(covariance[2, 4]), expected_45, places=12)
        self.assertGreater(float(covariance[1, 2]), float(covariance[2, 3]))

    def test_legacy_radian_length_scale_remains_equivalent(self):
        ell_deg = 90.0
        by_degrees = periodic_camera_rbf_covariance(
            FORMAL_AZIMUTHS, ell_deg=ell_deg
        )
        by_radians = periodic_camera_rbf_covariance(
            FORMAL_AZIMUTHS, length_scale=math.pi / 2.0
        )
        self.assertTrue(torch.allclose(by_degrees, by_radians, atol=1e-14, rtol=0.0))

    def test_length_scale_modes_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            periodic_camera_rbf_covariance(
                FORMAL_AZIMUTHS, length_scale=1.0, ell_deg=45.0
            )


class TestNestedTreeCovariance(unittest.TestCase):
    def test_real_azimuths_map_to_fixed_slots_in_input_order(self):
        slots = azimuths_to_slots([315.0, -45.0, 0.0, 91.0, 181.0])
        self.assertEqual(slots.tolist(), [7, 7, 0, 2, 4])

    def test_tree_a_uses_study_default_group_membership_weights(self):
        self.assertEqual(DEFAULT_LOWRANK_TREE_WEIGHTS, (0.05, 0.15, 0.30, 0.50))
        covariance = tree_a_covariance(ALL_SLOTS_AZIMUTHS)
        validate_covariance_matrix(covariance, psd_atol=1e-12)
        self.assertGreater(float(torch.linalg.eigvalsh(covariance).min()), 0.0)
        self.assertAlmostEqual(float(covariance[0, 1]), 0.50, places=12)
        self.assertAlmostEqual(float(covariance[0, 2]), 0.20, places=12)
        self.assertAlmostEqual(float(covariance[0, 4]), 0.05, places=12)

    def test_tree_b_shift_and_ab_average_include_wraparound_pair(self):
        tree_a = tree_a_covariance(ALL_SLOTS_AZIMUTHS)
        tree_b = tree_b_covariance(ALL_SLOTS_AZIMUTHS)
        tree_ab = tree_ab_covariance(ALL_SLOTS_AZIMUTHS)
        self.assertAlmostEqual(float(tree_a[7, 0]), 0.05, places=12)
        self.assertAlmostEqual(float(tree_b[7, 0]), 0.50, places=12)
        self.assertAlmostEqual(float(tree_b[1, 2]), 0.50, places=12)
        self.assertTrue(
            torch.allclose(tree_ab, 0.5 * tree_a + 0.5 * tree_b, atol=0.0, rtol=0.0)
        )
        self.assertAlmostEqual(float(tree_ab[7, 0]), 0.275, places=12)

    def test_actual_order_is_not_replaced_by_view_index_order(self):
        shuffled = [180.0, 0.0, 315.0, 45.0]
        covariance = tree_covariance_from_azimuths(shuffled, tree="b")
        self.assertAlmostEqual(float(covariance[1, 2]), 0.50, places=12)
        self.assertAlmostEqual(float(covariance[0, 1]), 0.05, places=12)

    def test_repeated_mapped_slot_is_psd_then_identity_mix_is_spd(self):
        target = tree_a_covariance([0.0, 1.0, 90.0])
        self.assertAlmostEqual(float(target[0, 1]), 1.0, places=12)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(target).min()), -1e-12)
        mixed = mix_covariance_with_identity(target, 0.7)
        self.assertGreater(float(torch.linalg.eigvalsh(mixed).min()), 0.0)


class TestIdentityMixingAndKL(unittest.TestCase):
    def test_identity_mixing_formula_and_zero_endpoint(self):
        target = tree_ab_covariance(FORMAL_AZIMUTHS)
        identity = torch.eye(len(FORMAL_AZIMUTHS), dtype=torch.float64)
        self.assertTrue(torch.equal(mix_covariance_with_identity(target, 0.0), identity))
        mixed = mix_covariance_with_identity(target, 0.25)
        self.assertTrue(
            torch.allclose(mixed, 0.75 * identity + 0.25 * target, atol=0.0, rtol=0.0)
        )
        validate_covariance_matrix(mixed, psd_atol=0.0)

    def test_complete_joint_kl_formula(self):
        correlation = 0.4
        covariance = torch.tensor(
            [[1.0, correlation], [correlation, 1.0]], dtype=torch.float64
        )
        rank = 8
        expected = -0.5 * rank * math.log(1.0 - correlation * correlation)
        self.assertAlmostEqual(joint_gaussian_kl(covariance, rank), expected, places=12)
        with self.assertRaises(ValueError):
            joint_gaussian_kl(torch.ones((2, 2), dtype=torch.float64), rank)

    def test_bisection_calibrates_pilot_budgets(self):
        target = tree_ab_covariance(ALL_SLOTS_AZIMUTHS)
        for requested in (1.0, 5.0):
            result = calibrate_alpha_for_target_kl(target, rank=16, target_kl=requested)
            self.assertEqual(result["status"], "calibrated")
            self.assertLess(result["relative_error"], 1e-5)
            self.assertGreaterEqual(result["alpha"], 0.0)
            self.assertLess(result["alpha"], 1.0)
            self.assertLessEqual(result["iterations"], 80)
            self.assertAlmostEqual(
                joint_gaussian_kl(result["covariance"], 16),
                result["achieved_kl"],
                places=12,
            )
            json.dumps(result["json_metadata"], allow_nan=False)

    def test_unattainable_target_is_reported_without_rewriting_request(self):
        target = tree_a_covariance(ALL_SLOTS_AZIMUTHS)
        requested = 1_000_000.0
        result = calibrate_alpha_for_target_kl(target, rank=8, target_kl=requested)
        self.assertEqual(result["status"], "unattainable")
        self.assertEqual(result["target_kl"], requested)
        self.assertLess(result["achieved_kl"], requested)
        self.assertEqual(result["alpha"], 1.0 - 1e-8)


class TestCovarianceMetadata(unittest.TestCase):
    def test_metadata_is_json_safe_and_records_camera_relations(self):
        covariance = periodic_camera_rbf_covariance(
            FORMAL_AZIMUTHS, ell_deg=45.0
        )
        metadata = covariance_metadata(
            covariance,
            azimuths_deg=FORMAL_AZIMUTHS,
            ell_deg=45.0,
            topology="periodic_camera_rbf",
        )
        json.dumps(metadata, allow_nan=False)
        self.assertEqual(len(metadata["eigenvalues"]), len(FORMAL_AZIMUTHS))
        self.assertGreater(metadata["condition_number"], 1.0)
        self.assertGreater(metadata["effective_rank"], 1.0)
        self.assertGreater(metadata["off_diagonal_energy"], 0.0)
        wrap = [
            relation
            for relation in metadata["adjacent_relations"]
            if {relation["first_index"], relation["second_index"]} == {0, 5}
        ]
        self.assertEqual(len(wrap), 1)
        self.assertEqual(wrap[0]["periodic_distance_deg"], 45.0)
        self.assertAlmostEqual(
            wrap[0]["covariance"], wrap[0]["theoretical_correlation"], places=12
        )


if __name__ == "__main__":
    unittest.main()
