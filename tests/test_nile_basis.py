import json
import unittest

import torch

from mvadapter.nile.basis import (
    basis_checksum,
    build_dct2_basis,
    ordered_dct2_modes,
)


class TestDCT2Basis(unittest.TestCase):
    def test_shape_orthonormality_and_metadata(self):
        for rank in (8, 16):
            with self.subTest(rank=rank):
                basis, metadata = build_dct2_basis(
                    4, 8, 10, rank, return_metadata=True
                )
                self.assertEqual(tuple(basis.shape), (4 * 8 * 10, rank))
                self.assertEqual(basis.dtype, torch.float32)
                gram = basis.to(torch.float64).mT @ basis.to(torch.float64)
                error = float(
                    (gram - torch.eye(rank, dtype=torch.float64)).abs().max()
                )
                self.assertLess(error, 1e-6)
                self.assertLess(metadata["orthonormality_error"], 1e-6)
                self.assertLess(metadata["output_orthonormality_error"], 1e-6)
                self.assertEqual(metadata["rank"], rank)
                self.assertEqual(len(metadata["columns"]), rank)
                json.dumps(metadata, allow_nan=False)

    def test_checksum_is_deterministic_and_describes_returned_tensor(self):
        first, first_metadata = build_dct2_basis(
            4, 8, 8, 16, return_metadata=True
        )
        second, second_metadata = build_dct2_basis(
            4, 8, 8, 16, return_metadata=True
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_metadata["basis_checksum"], second_metadata["basis_checksum"])
        self.assertEqual(first_metadata["basis_checksum"], basis_checksum(first))
        self.assertEqual(len(first_metadata["basis_checksum"]), 64)

        different = build_dct2_basis(4, 8, 8, 15)
        self.assertNotEqual(basis_checksum(first), basis_checksum(different))

    def test_dc_is_excluded_from_every_channel_by_default(self):
        channels, height, width, rank = 4, 7, 9, 16
        basis, metadata = build_dct2_basis(
            channels, height, width, rank, return_metadata=True
        )
        self.assertTrue(metadata["exclude_dc"])
        self.assertNotIn(
            (0, 0),
            [(column["u"], column["v"]) for column in metadata["columns"]],
        )
        columns = basis.reshape(channels, height * width, rank)
        self.assertLess(float(columns.sum(dim=1).abs().max()), 1e-6)

        with_dc, with_dc_metadata = build_dct2_basis(
            channels,
            height,
            width,
            channels,
            exclude_dc=False,
            return_metadata=True,
        )
        self.assertEqual(
            [(entry["u"], entry["v"]) for entry in with_dc_metadata["columns"]],
            [(0, 0)] * channels,
        )
        self.assertGreater(
            float(with_dc.reshape(channels, height * width, channels).sum(dim=1).abs().max()),
            1.0,
        )

    def test_frequency_order_and_channel_round_robin(self):
        channels, rank = 4, 19
        _, metadata = build_dct2_basis(
            channels, 8, 8, rank, return_metadata=True
        )
        records = metadata["columns"]
        self.assertEqual(
            [record["channel"] for record in records],
            [index % channels for index in range(rank)],
        )
        frequencies = [record["spatial_frequency_squared"] for record in records]
        self.assertEqual(frequencies, sorted(frequencies))

        counts = [
            sum(record["channel"] == channel for record in records)
            for channel in range(channels)
        ]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_mode_tie_break_and_invalid_rank(self):
        modes = ordered_dct2_modes(4, 5)
        self.assertEqual(modes[:5], [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)])
        with self.assertRaisesRegex(ValueError, "exceeds the available rank"):
            build_dct2_basis(2, 2, 2, 7)
        with self.assertRaises(TypeError):
            build_dct2_basis(2, 4, 4, 4, dtype=torch.float16)


if __name__ == "__main__":
    unittest.main()
