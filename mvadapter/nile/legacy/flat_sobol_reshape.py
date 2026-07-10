"""Legacy scalar-wise Sobol reshape methods that produced stripe artifacts."""

from ..sampler import make_flat_sobol_latents, make_nile_v_latents

__all__ = ["make_flat_sobol_latents", "make_nile_v_latents"]
