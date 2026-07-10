"""Legacy post-scheduler latent projections retained for failure analysis."""

from ..callbacks import NILECallbackConfig, NILEViewTimeCallback

# Descriptive aliases make clear that these callbacks project semantic latent
# state; they do not implement stochastic NILE-Time variance-noise sampling.
ViewLowpassStateProjection = NILEViewTimeCallback
MortonModulatedStateProjection = NILEViewTimeCallback

__all__ = [
    "MortonModulatedStateProjection",
    "NILECallbackConfig",
    "NILEViewTimeCallback",
    "ViewLowpassStateProjection",
]
