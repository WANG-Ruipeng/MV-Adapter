from .camera import (
    Camera,
    get_c2w,
    get_camera,
    get_orthogonal_camera,
    get_orthogonal_projection_matrix,
    get_projection_matrix,
)

# Camera-only image-to-multiview inference does not rasterize meshes. Keep the
# nvdiffrast-backed texture/render stack optional so importing camera helpers
# does not require compiling an unrelated CUDA extension in Colab.
try:
    from .mesh import TexturedMesh, load_mesh, replace_mesh_texture_and_save
    from .projection import CameraProjection, CameraProjectionOutput
    from .render import (
        DepthControlNetNormalization,
        DepthNormalizationStrategy,
        NVDiffRastContextWrapper,
        RenderOutput,
        SimpleNormalization,
        Zero123PlusPlusNormalization,
        render,
    )
    from .smart_paint import SmartPainter
except ModuleNotFoundError as error:
    if error.name not in {"nvdiffrast", "nvdiffrast.torch"}:
        raise
