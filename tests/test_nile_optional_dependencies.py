"""Optional dependency boundaries used by image-to-multiview inference."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


MESH_UTILS = (
    Path(__file__).resolve().parents[1]
    / "mvadapter"
    / "utils"
    / "mesh_utils"
)


class OptionalMeshDependencyTests(unittest.TestCase):
    def test_camera_helpers_do_not_import_nvdiffrast(self):
        for filename in ("camera.py", "utils.py"):
            source = (MESH_UTILS / filename).read_text(encoding="utf-8")
            self.assertNotIn("import nvdiffrast", source)

        init_source = (MESH_UTILS / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("except ModuleNotFoundError as error:", init_source)
        self.assertIn('{"nvdiffrast", "nvdiffrast.torch"}', init_source)

        package_name = "_nile_camera_without_optional_renderers"
        package = types.ModuleType(package_name)
        package.__path__ = [str(MESH_UTILS)]
        utility = types.ModuleType(package_name + ".utils")
        utility.LIST_TYPE = object
        sys.modules[package_name] = package
        sys.modules[package_name + ".utils"] = utility
        try:
            spec = importlib.util.spec_from_file_location(
                package_name + ".camera", MESH_UTILS / "camera.py"
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cameras = module.get_orthogonal_camera(
                elevation_deg=[0.0, 0.0],
                distance=[1.8, 1.8],
                left=-1.0,
                right=1.0,
                bottom=-1.0,
                top=1.0,
                azimuth_deg=[0.0, 180.0],
                device="cpu",
            )
        finally:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)

        self.assertEqual(tuple(cameras.mvp_mtx.shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(cameras.mvp_mtx).all())


if __name__ == "__main__":
    unittest.main()
