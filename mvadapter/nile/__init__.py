"""Distribution-preserving view-noise experiments and frozen v0 utilities.

The covariance/spectral/nested/diagnostic APIs are the formal experiment
surface. The older Sobol, mismatched low-pass, and state-projection APIs stay
exported for backward compatibility and failure reproduction only.
"""

from .covariance import (
    is_positive_semidefinite,
    minimum_eigenvalue,
    periodic_camera_rbf_covariance,
    single_tree_covariance,
    stable_cholesky,
    staggered_two_tree_covariance,
    validate_covariance_matrix,
)
from .diagnostics import (
    DEFAULT_DISTRIBUTION_THRESHOLDS,
    assert_distribution_gates,
    coarse_radial_psd_deviation,
    cross_view_covariance_error,
    cross_view_radial_frequency_correlation,
    diagnose_latents,
    empirical_cross_view_covariance,
    evaluate_distribution_gates,
    lag_autocorrelations,
    moment_statistics,
    per_view_axis_stripe_scores,
    per_view_lag_autocorrelations,
    per_view_moment_statistics,
    per_view_radial_psd_deviation,
    radial_power_spectrum,
    radial_psd_deviation,
    spectral_axis_stripe_score,
)
from .nested_elements import (
    angles_to_dyadic_slots,
    element_seed_key,
    frequency_dependent_level_weights,
    make_nested_tree_latents,
    nested_tree_spatial_covariance_target,
    tree_ancestor_ids,
)
from .spectral_gaussian import (
    camera_rbf_spatial_covariance_target,
    global_spatial_covariance_target,
    make_camera_rbf_correlated_latents,
    make_spectral_global_correlated_latents,
    radial_frequency_grid,
    spectral_correlation_profile,
)

from .callbacks import (
    CallbackMode,
    NILECallbackConfig,
    NILEViewTimeCallback,
    build_patch_rho_map,
    linear_rho,
)
from .morton import morton2d, part1by1, patch_morton_order
from .ops import gaussian_blur_latent, low_high_split, standardize_like, standardize_unit
from .sampler import (
    NILEConfig,
    SamplerMode,
    make_flat_sobol_latents,
    make_iid_latents,
    make_initial_latents,
    make_lowpass_shared_latents,
    make_nile_v_latents,
    make_shared_latents,
)
from .sequence import SZBackend, SobolBackend, inverse_normal_cdf


__all__ = [
    "CallbackMode",
    "DEFAULT_DISTRIBUTION_THRESHOLDS",
    "NILECallbackConfig",
    "NILEConfig",
    "NILEViewTimeCallback",
    "SZBackend",
    "SamplerMode",
    "SobolBackend",
    "angles_to_dyadic_slots",
    "assert_distribution_gates",
    "build_patch_rho_map",
    "camera_rbf_spatial_covariance_target",
    "coarse_radial_psd_deviation",
    "cross_view_covariance_error",
    "cross_view_radial_frequency_correlation",
    "diagnose_latents",
    "element_seed_key",
    "empirical_cross_view_covariance",
    "evaluate_distribution_gates",
    "frequency_dependent_level_weights",
    "gaussian_blur_latent",
    "inverse_normal_cdf",
    "is_positive_semidefinite",
    "lag_autocorrelations",
    "linear_rho",
    "low_high_split",
    "make_flat_sobol_latents",
    "make_iid_latents",
    "make_initial_latents",
    "make_lowpass_shared_latents",
    "make_camera_rbf_correlated_latents",
    "make_nested_tree_latents",
    "make_nile_v_latents",
    "make_shared_latents",
    "make_spectral_global_correlated_latents",
    "minimum_eigenvalue",
    "moment_statistics",
    "morton2d",
    "part1by1",
    "patch_morton_order",
    "nested_tree_spatial_covariance_target",
    "per_view_axis_stripe_scores",
    "per_view_lag_autocorrelations",
    "per_view_moment_statistics",
    "per_view_radial_psd_deviation",
    "periodic_camera_rbf_covariance",
    "radial_frequency_grid",
    "radial_power_spectrum",
    "radial_psd_deviation",
    "single_tree_covariance",
    "spectral_axis_stripe_score",
    "spectral_correlation_profile",
    "stable_cholesky",
    "staggered_two_tree_covariance",
    "standardize_like",
    "standardize_unit",
    "tree_ancestor_ids",
    "validate_covariance_matrix",
    "global_spatial_covariance_target",
]
