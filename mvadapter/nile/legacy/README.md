# Frozen failure-analysis implementations

This directory preserves the first-generation NILE-inspired prototypes solely
for reproducibility and failure analysis. They are not part of the formal
low-rank equal-KL study.

The archived designs include flat Sobol scalar reshape, mismatched shared
low-pass construction, and denoising-state projection callbacks. They can
produce stripes, colored latent spectra, view collapse, and shared semantic
artifacts. Do not use them as formal methods and do not describe them as a
strict NILE/SZ implementation.

The formal experiment surface is implemented by `basis.py`, `covariance.py`,
`lowrank_coupling.py`, `diagnostics.py`, and `trajectory.py` in the parent
package. See `docs/NILE_V0_FAILURE_ANALYSIS.md` for the empirical history.
