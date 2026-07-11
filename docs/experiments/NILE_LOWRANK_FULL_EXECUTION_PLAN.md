# NILE-Inspired Low-Rank Equal-KL Study Execution Plan

## Scope and scientific question

This study tests whether weak cross-view Gaussian coupling restricted to a
small orthonormal low-frequency latent subspace can influence MV-Adapter's
shared geometry hypothesis without changing any single-view white-Gaussian
marginal. It is explicitly **NILE-inspired nested Gaussian element topology;
strict NILE/SZ is not implemented in this study.**

The completed distribution-preserving QUICK_SMOKE is the starting point, not
formal evidence: it ran one demo input, one seed, and two strengths. It proved
the external-latent path and distribution gates, while showing that full-field
nested coupling can create tails, blobs, and limb-like artifacts.

## Repository state at audit

- Source revision at audit: `f508f153f2ece2167d99413426bff06ba0169332`.
- The worktree is intentionally dirty because the user updated `prompt.txt`
  and the executed distribution-preserving notebook. Those files must not be
  reverted or overwritten.
- Existing formal modules provide full-field spectral, camera-RBF, and nested
  Gaussian samplers plus distribution diagnostics.
- Existing inference separates initial-latent, reference-VAE, and stochastic
  scheduler generators and preserves the native pipeline latent scaling.
- Existing grid manifests are resume-safe and the evaluator supports real
  angle bins, lightweight collapse guardrails, and optional MEt3R.
- Legacy Sobol reshape, mismatched low-pass, and latent-state projection
  callbacks remain frozen under `mvadapter/nile/legacy/`.

## Implemented worktree scope

The items below are present in the current worktree. They still require the
final repository-wide test receipt and a formal GPU execution before the study
can be called complete.

### Core math

- `mvadapter/nile/basis.py`: deterministic channel-balanced 2D DCT-II basis,
  pure spatial DC exclusion, checksums, and orthonormality metadata.
- `mvadapter/nile/covariance.py`: actual-angle periodic RBF, Tree A/B/AB
  group-membership targets, identity mixing, stable full Gaussian joint KL,
  equal-KL calibration, and covariance diagnostics.
- `mvadapter/nile/lowrank_coupling.py`: projection, view-axis correlation,
  residual reconstruction, exact identity passthrough, and JSON-safe metadata.
- `mvadapter/nile/diagnostics.py`: basis-coefficient covariance and low-rank
  equal-KL gates in addition to existing full-space white-noise gates.

### Observation and inference

- `mvadapter/nile/trajectory.py`: mutation-free step-end observer, milestone
  snapshots, `G_t`, paired `Delta_t`, NPZ/CSV, and optional plots.
- `scripts/inference_i2mv_sdxl_nile.py`: new `lowrank_camera_rbf`,
  `lowrank_nested_tree_a`, and `lowrank_nested_tree_ab` methods; complete
  construction metadata; optional trajectory artifacts; no legacy averaging.

### Reproducible workflow

- `scripts/validate_nile_inputs.py`: content hashes, duplicate rejection,
  deterministic disjoint PILOT/FULL split, CSV manifest, and contact sheet.
- `scripts/run_nile_lowrank_study.py`: locked resolved config, staged execution,
  distribution preflight, resumable atomic manifest, dry-run estimates, retry
  bookkeeping, frozen selection consumption, test/checkpoint provenance gates,
  input-manifest locking, and explicit blockers.
- `scripts/select_nile_lowrank_candidates.py`: paired eligibility gates and
  deterministic per-topology MEt3R selection with lower-KL/lower-rank tie-breaks.
- `scripts/eval_nile_lowrank_study.py`: MEt3R orchestration, identity/artifact
  guardrails, paired/cluster-bootstrap statistics, and trajectory summaries.
- `scripts/report_nile_lowrank_study.py`: Markdown/JSON report, plots/contact
  sheets where data exist, `FINAL_STATUS.json`, and `REPRODUCE.md`.
- `configs/nile_lowrank_full.yaml`: complete PILOT/FULL/evaluation policy.
- `notebooks/mvadapter_nile_lowrank_full_colab.ipynb`: one Run-All path using a
  single Drive artifact root, frozen checkpoints, an atomic test receipt, and
  resume manifests.

## Experiment matrix

PILOT uses five unique inputs and two seeds. Per input/seed it requests 18
configurations: two baselines, eight camera-RBF configurations
(`rank={8,16}`, `KL={1,5}`, `ell={45,90}`), four Tree A configurations, and
four Tree AB configurations. The requested matrix therefore contains 180 run
slots. The executed CPU preflight described below found three structurally
unattainable equal-KL configurations. They are excluded before generation, so
the current gate-eligible matrix contains 15 configurations per input/seed and
the maximum executable PILOT size is 150 runs, not 180.

After PILOT MEt3R and guardrails, one immutable candidate per topology is
selected or explicitly marked `no_eligible_candidate`/`diagnostic_only`.

FULL uses twenty different inputs and three seeds. It schedules IID external,
shared-full diagnostic, and the three frozen topology candidates only.
Expected FULL size when all topology candidates exist: 300 runs. FULL data are
never used for tuning.

## Fairness controls

All paired runs share the same input, seed, model revisions, scheduler, thirty
steps, guidance, view list, preprocessing, resolution, reference posterior
stream, and scheduler stream. Proposed topologies compare the same basis rank
and target joint KL, never the same raw correlation strength. `shared_full` is
degenerate and excluded from finite-KL method claims.

## Formal provenance and split-isolation contracts

Formal generation is gated by evidence artifacts rather than by the presence
of files in a cache alone.

- Colab cell 8 atomically writes `environment/test_results.json`. The runner
  verifies its schema, `passed=true`, `tests_complete=true`, zero return codes
  for both `compileall` and pytest, a completion timestamp, and both recorded
  commands. A missing, malformed, or failed receipt produces the
  `tests_not_verified` blocker.
- `configs/checkpoint_manifest.json` freezes the base model, VAE, MV-Adapter,
  BiRefNet, DINOv2, and MEt3R revisions plus the exact adapter checkpoint path
  and SHA-256. The runner recomputes the checkpoint hash and compares the
  manifest, resolved config, repository IDs, immutable revisions, filename,
  and bytes. A verified audit may be reused only through
  `environment/checkpoint_audit_cache.json`, whose key includes the resolved
  config hash, manifest-content SHA-256, and checkpoint path/size/mtime/ctime;
  any manifest, config, or file-identity change forces a fresh hash.
- The first input validation atomically freezes `inputs/input_validation.json`
  and `input_manifest.csv` after exact, perceptual, and rotation-duplicate
  rejection and SHA-256 ordering. Every resume rescans into a temporary
  directory. Added, removed, or modified inputs set `input_manifest_changed`,
  leave the frozen manifest untouched, block formal execution, and require a
  new experiment ID.
- Before FULL, the split-isolation audit checks the frozen 5/20 counts and
  uniqueness, requires the PILOT and FULL manifests to contain exactly their
  assigned SHA-256 sets, and rejects frozen overlap, cross-split leakage, or
  overlap between the run manifests. Trajectory repeats the audit before using
  its configured subset of the FULL split.

## Risks and blockers

- Formal execution requires at least 25 distinct, non-augmented inputs. The
  repository audit found no `inputs/formal` directory and no `NILE_INPUT_DIR`.
- The local Python 3.14 environment has no installed PyTorch. CPU math tests can
  use the existing temporary CPU test environment; SDXL execution needs a
  supported CUDA environment such as the Colab notebook.
- The local RTX 5080 has 16 GiB VRAM and model weights are not present in the
  repository. SDXL six-view generation may require Colab L4/A100 or offload.
- Strict FULL completion requires a working MEt3R installation and model cache.
- Missing credentials, model access, disk, CUDA, inputs, or MEt3R are recorded
  as blockers rather than silently converting a smoke run into FULL evidence.

## Executed local CPU preflight

The distribution preflight was executed on CPU and is preserved under
`outputs/nile_lowrank_kl_full/local-audit-f508f153/distribution_gates/`.
The summary was generated at `2026-07-10T18:20:26Z` (July 11 local time) with
three seeds, batch size 32 per seed, six views, and `4 x 96 x 96` latents. This
is mathematical/distribution evidence only; it is not SDXL, PILOT, or FULL
image-generation evidence.

- Requested configurations: 18.
- Attempted configurations: 15.
- Passed attempted configurations: 15/15.
- Gate failures: 0.
- Excluded before sampling because target KL was unattainable: 3.
- Worst absolute per-view mean: `0.0006972` (limit `0.01`).
- Per-view standard-deviation range: `[0.9996875, 1.0005516]` (required
  `[0.99, 1.01]`).
- Worst nonzero-lag autocorrelation: `0.0012981` (limit `0.02`).
- Worst radial-PSD deviation: `0.0042903` (limit `0.05`).
- Worst stripe proxy: `0.0049098` (limit `0.15`).
- Worst full-space covariance MAE: `0.0011035` (limit `0.03`).
- Worst basis-coefficient covariance MAE: `0.0113214` (limit `0.03`).
- Worst basis orthonormality error: `1.4194e-8` (limit `1e-6`).
- Worst attainable-KL relative error: `9.951e-9` (limit `1e-5`).
- Smallest covariance eigenvalue: `0.2896412` (required `>1e-8`).

The machine-readable evidence is in `configuration_gates.json` and
`preflight_summary.json`; the adjacent CSV and diagnostic directory contain
the gate matrix, alpha/KL curve, and covariance plots. An unattainable target
is an explicit configuration exclusion, not a distribution-gate failure.

The final formal plotting contract additionally requires
`covariance_eigenvalue_spectra.png`, covering every low-rank effective
six-view covariance, together with the gate matrix, alpha/KL plot, and all
per-configuration covariance heatmaps. GPU scheduling requires
`diagnostic_plots_complete=true`; the cited `local-audit-f508f153` directory is
a numerical preflight snapshot and is not, by directory existence alone,
authorization to start formal GPU generation.

## Unattainable equal-KL configurations

With the formal six-view camera list and the configured alpha cap
`alpha=0.99999999`, the following requested `KL=5` configurations cannot
reach five nats:

| Method | Rank | Target KL | KL at alpha cap | Disposition |
| --- | ---: | ---: | ---: | --- |
| `lowrank_nested_tree_a` | 8 | 5.0 | 2.7762403 | excluded |
| `lowrank_nested_tree_ab` | 8 | 5.0 | 1.9273333 | excluded |
| `lowrank_nested_tree_ab` | 16 | 5.0 | 3.8546665 | excluded |

The runner requires exactly these three formal exclusions and refuses a gate
artifact with a different unattainable count. It does not lower the requested
KL, reinterpret the achieved KL as equal-KL evidence, or send these records to
the GPU worker. Tree A rank 16 at KL 5 and the remaining KL/rank combinations
are gate eligible.

## Persistent GPU worker and resume contract

`runtime.model_load_strategy` is frozen to `persistent_worker`. The runner
writes a validated worker plan, and the worker rejects any record that is not
explicitly distribution-gate eligible before loading models. One worker keeps
the MV-Adapter SDXL pipeline and BiRefNet bundle resident and processes runs
sequentially rather than reloading both bundles for every record.

The resolved config includes a code revision that fingerprints HEAD plus
tracked and untracked worktree sources. That revision participates in the
config hash, experiment directory, and run IDs, so a source change cannot be
silently resumed as the old experiment. The config lock rejects mismatched
hashes. Each runner invocation also captures git status, commit, binary diff,
`pip freeze`, `nvidia-smi`, and structured Python/CUDA/disk metadata under the
experiment's `environment/` directory.

Lifecycle events are appended to `worker_events.jsonl`, flushed, and fsynced.
The runner polls those events and atomically rewrites the authoritative split
manifest after every terminal run event. Each apparent success is audited for
the grid, metadata, all six individual views, all six masks, reference image,
camera list, method/rank/KL construction metadata, and trajectory file when
requested. An OOM clears the CUDA cache and retries once with unchanged
resolution, steps, method, seed, and other parameters. If the worker exits
without a terminal event, the runner accepts a recovered success only when
the full artifact bundle passes the same audit; incomplete records remain
failed and are eligible for `--resume`.

Before each worker launch the runner fingerprints every expected artifact
component. Recovery after a missing terminal event therefore requires both a
complete bundle and evidence that the current invocation refreshed the grid,
metadata, reference, complete view directory, complete mask directory, and
trajectory when applicable. A stale pre-existing complete bundle, or a
partially refreshed views/masks directory, cannot be promoted to success.
Worker return codes are persisted for terminal and recovered records.

Per-run `grid_metadata.json` is written through a temporary file, flushed,
fsynced, and atomically replaced. Runner manifests, runtime status, config
locks, candidate artifacts, reports, and final status use atomic replacement
as well, so interruption cannot make a partial JSON document authoritative.

This worker contract is implemented and covered by fixture tests, but it has
not been exercised against the real SDXL stack on the current local machine
because Python cannot access CUDA and the model revisions/checkpoint hashes
are not yet frozen.

## Frozen candidate and FULL precheck

Candidate selection is not trusted merely because
`selected_candidates.json` exists. Immediately before FULL and trajectory,
the runner requires JSON/YAML content equality, recomputes the selection and
wrapper hashes, checks the current study-config hash and PILOT-metrics
SHA-256, and reruns the current deterministic selection policy. Every frozen
configuration, including a legitimate `no_eligible_candidate` marked
`diagnostic_only`, must match exactly one PILOT summary and one passed
preflight configuration and preserve method, rank, target/achieved KL, alpha,
RBF scale, and basis/covariance checksums. Diagnostic-only rows may continue
for failure analysis but are excluded from positive topology claims.

The three formal topology candidates must also share one identical
`(rank, target_kl)` pair. If independent PILOT scoring selects unequal rank or
KL, FULL is blocked with `selected_equal_rank_kl_mismatch`; the runner does not
silently compare unequal candidates. FULL additionally requires verified
PILOT MEt3R provenance/completeness and the split-isolation proof above.

## Frozen inference and trajectory contract

The formal adapter checkpoint is
`mvadapter_i2mv_sdxl.safetensors`; its immutable adapter revision and actual
file SHA-256 are frozen by the notebook and re-audited by the runner. The
formal config requires `model.scheduler: null`, meaning the repository-native
scheduler path; declarative DDPM/LCM overrides are rejected rather than
silently changing the protocol. Per-run metadata must match the frozen
checkpoint filename, scheduler value, model revisions, input path/hash, seed,
steps, guidance, cameras, method, rank/KL/alpha, and basis/covariance
checksums before an artifact is accepted.

Trajectory uses the frozen milestones `[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]`.
The true initial latent is recorded immediately after `prepare_latents` and
before the first scheduler step; the observer remains read-only. Each selected
method is paired with an IID control using the same rank, input, and seed, and
summary validation rejects rank/checksum/pair mismatches before producing
per-pair view-correlation heatmaps and aggregate `G_t`/`Delta_t` curves.

## Current execution boundary and formal blockers

The captured local environment at
`outputs/nile_lowrank_kl_full/local-audit-f508f153/environment/` records
Windows 11, Python 3.14.2, an RTX 5080 with 16,303 MiB VRAM, driver 591.86,
and driver CUDA 13.1. Hardware discovery is not equivalent to a usable study
runtime: the system Python has no PyTorch and records
`cuda_runtime_available_to_python=false`.

The formal runner will currently block PILOT/FULL for these independent
reasons:

1. The local audit root does not contain a verified atomic test receipt or
   checkpoint-provenance manifest/cache audit.
2. No configured formal input directory or `NILE_INPUT_DIR` is available, so
   there are zero verified distinct inputs versus the required 25.
3. CUDA is unavailable to the active Python environment.
4. The base-model, VAE, adapter, and BiRefNet revisions are still null rather
   than immutable revisions, and the adapter checkpoint SHA-256 is unset.
5. MEt3R is pinned in the config to commit
   `ee0e1752898559e1a3e85e2e151d3edeb9b55f73`, but its package is not
   installed in the local environment.
6. The DINOv2 identity-model revision is not yet resolved to an immutable
   revision.

Disk is not a current local blocker: the captured E: drive had 920.595 GiB
free, above the configured 25 GiB minimum. Credentials/model access and the
actual remote caches remain unverified until the Colab setup resolves and
freezes them.

Consequently, PILOT, FULL, MEt3R, and real-inference trajectory completion must
remain false until their formal prerequisites and evidence exist. A local
`--stage all` invocation still proceeds through the report stage and
atomically emits an evidence-bounded blocked report plus `FINAL_STATUS.json`.
In that artifact, `report_complete=true` means the report/reproduction files
were written successfully; it does not imply `pilot_complete` or
`full_complete`, which remain false under these blockers.

## Dry-run, Run-All, and report evidence links

Runner `--dry-run` is a non-generation preview: it returns planned counts,
estimated bytes, and commands without launching the worker or mutating the
split manifest, and it never sets PILOT/FULL complete. The notebook's default
`RUN_ALL=True` path performs this preview immediately before the real resumable
PILOT stage, verifies that the manifest bytes did not change, then executes the
same plan without `--dry-run`. When prerequisites are missing, Run-All skips
blocked GPU generation but continues to the blocked report instead of
fabricating success.

Evaluation writes plots under `plots/{pilot,full}` and paired method contact
sheets plus a failure gallery under `contact_sheets/{pilot,full}`. The JSON
report records directories, completeness, counts, and artifact paths. The
Markdown report renders clickable relative links for plot files, paired
sheets, and `failure_gallery.jpg`, and separately lists explicitly recorded
generation, guardrail, artifact, and collapse failures. Missing metrics or
images are never converted to zero-valued successes. `FINAL_STATUS.json` is
written last, after the Markdown/JSON report and `REPRODUCE.md`.

## Acceptance criteria

1. DCT columns are deterministic and orthonormal to max error `<1e-6`.
2. Every mixed covariance is symmetric, unit-diagonal, and has minimum
   eigenvalue `>1e-8`; achieved KL relative error is `<1e-5` when attainable.
3. Identity/alpha-zero coupling is bit-exact passthrough and no method performs
   per-sample standardization.
4. Ensemble latent gates satisfy the existing marginal, autocorrelation, PSD,
   stripe, covariance, basis, and KL thresholds before GPU scheduling.
5. Observer-on/off inference is bit-identical and callback state is immutable.
6. Input hashes are unique and PILOT/FULL splits are deterministic/disjoint.
7. Manifest/config/input locking, provenance receipts, fresh-worker recovery,
   frozen-selection integrity, strict-MEt3R FULL refusal, and fixture-based
   reporting have regression tests.
8. `compileall` and all task tests pass. The pre-existing notebook-cleanliness
   failure caused by the user's executed
   `notebooks/mvadapter_distribution_preserving_colab.ipynb` is recorded
   separately; it is neither overwritten nor allowed to mask task-test
   success.
9. The final status distinguishes implementation, tests, PILOT, FULL, MEt3R,
   and report completion without overstating blocked work.
