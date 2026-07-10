# NILE prototype v0 failure analysis

The first MV-Adapter I2MV SDXL quick run is preserved in
`notebooks/mvadapter_nile_experiments_colab.ipynb`, including its execution
logs, seven visual grids, metadata paths, and lightweight metric table.

## Run configuration

- Host: MV-Adapter I2MV SDXL
- GPU: NVIDIA L4 (22 GiB)
- Input: juvenile emperor penguin
- Views: 0, 45, 90, 180, 270, 315 degrees
- Seed: 0
- Denoising steps: 30
- Prototype strength: 0.65
- Executed code commit: `253ac4fcea7de5f396371124af597e6cc957bfae`
- Methods: IID, shared, low-pass shared, flat Sobol, NILE-V, NILE-VT, NILE-VTP

All seven subprocesses completed after the BiRefNet dtype fix.

## Recorded lightweight diagnostics

| Method | Adjacent low-frequency similarity | Opposite low-frequency similarity | Adjacent high-frequency distance |
| --- | ---: | ---: | ---: |
| IID | 0.928062 | 0.915892 | 0.047229 |
| Shared | 0.910965 | 0.912199 | 0.047143 |
| Low-pass Shared | 0.979821 | 0.980612 | 0.041580 |
| Flat Sobol | 0.996585 | 0.994636 | 0.003818 |
| NILE-V | 0.995647 | 0.994762 | 0.004293 |
| NILE-VT | 0.998409 | 0.998027 | 0.002579 |
| NILE-VTP | 0.998379 | 0.997994 | 0.002596 |

MEt3R was not run in this quick experiment.

## Visual interpretation

- IID and fully shared Gaussian noise produced recognizable penguin views.
- The mismatched low/high-frequency sampler produced colored noise blobs.
- Scalar-wise flat Sobol reshape produced regular vertical stripes.
- NILE-V inherited colored stripe artifacts from the scalar Sobol child field.
- The VT/VTP state projections made the same invalid structure nearly identical
  across views.

The near-perfect pixel similarities therefore measure a shared out-of-
distribution artifact, not valid 3D consistency.  This result freezes the v0
methods as failure-analysis baselines.  The replacement experiment must preserve
the per-view white-Gaussian law and only modify cross-view covariance.
