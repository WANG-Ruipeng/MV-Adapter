"""Frozen v0 samplers and state-projection callbacks for failure analysis.

These APIs reproduce the prototype that generated structured stripes in the
first quick run.  They are intentionally excluded from the distribution-
preserving experiment defaults and must not be presented as strict NILE/SZ.
"""

from .flat_sobol_reshape import make_flat_sobol_latents, make_nile_v_latents
from .latent_projection_callbacks import (
    NILECallbackConfig,
    NILEViewTimeCallback,
    MortonModulatedStateProjection,
    ViewLowpassStateProjection,
)
from .lowpass_shared_mismatched import make_lowpass_shared_latents

__all__ = [
    "MortonModulatedStateProjection",
    "NILECallbackConfig",
    "NILEViewTimeCallback",
    "ViewLowpassStateProjection",
    "make_flat_sobol_latents",
    "make_lowpass_shared_latents",
    "make_nile_v_latents",
]
