# Changelog

## 0.3.0 (2026-10-20)

Fix release addressing the 2026-10-16 adversarial review. Three intentional
behaviour changes mean results for a given seed shift relative to v0.2.2
(all well within the validated errorbars on continuous, well-conditioned
data): gaussianise now average-ranks ties, CV folds now follow the seed,
and dead padding components are dropped before the bootstrap (which moves
estimates at Monte-Carlo noise level).

### Estimator correctness

- `preprocess='gaussianise'` breaks ties by average rank (the scipy
  `rankdata` convention) instead of positionally: a tied block used to be
  mapped to a monotone probit ramp in row order, fabricating MI whenever
  row order correlates with another variable (a sorted column next to a
  constant column returned 1.06 nats; the truth, and the new result, is 0).
  A RuntimeWarning now names columns with more than 1% tied values. Ranks
  and quantiles are computed at float64, where float32 integer ranks round
  above 2^24 samples. `return_fit` with gaussianise warns that the model
  lives in rank-probit space (the rank transform has no parametric
  inverse).
- `cov == chol @ chol^T` is now an invariant: `_chol` returns the repaired
  covariance and every caller writes it back. The Monte-Carlo stage used to
  sample the joint under the repaired factor while scoring marginals
  refactorised from the raw covariance — two different mixtures whenever a
  replicate needed repair. `FittedGMM` factorises once at construction, so
  `covariances_`, `sample` and `score_samples` agree; the repair warnings
  now report correct counts and the cumulative bump (the exhaustion message
  claimed "1e4 x mean variance"; the ceiling is ~1.11e3 x the largest
  absolute diagonal entry), and the non-finite-reset warning is no longer
  swallowed when escalation also runs.
- The legacy chunked M-step gets the v0.2.1 reg_covar diagonal clamp and
  float64 cross-chunk accumulators, matching the GEMM path: float32
  rounding of the shifted moments could push a collapsed component's
  variance below the floor, or negative.
- Sequential selection: the patience counter resets on improvement and the
  best-scoring k is returned — with patience=2 and improvements at k=1 and
  k=3, the old counter arithmetic returned k=2, discarding the better
  model; exhaustion returned max_components regardless of scores; an
  all-NaN metric crashed with KeyError. `patience` is now a parameter of
  `estimate` and `estimate_categorical`.
- The seed now controls the CV folds, as the docstring always claimed
  (they were hardcoded to seed 42); forwarded with a fixed offset so the
  fold stream stays decoupled from the init and bootstrap streams.
- An empty validation fold scores -inf instead of a spurious 0.0 that beat
  every real fit in the fold argmax; degenerate CV settings (n_folds < 2,
  n_folds > N, a category smaller than n_folds) are rejected up front with
  a clear ValueError, as are typo'd quantity/protocol/metric_method
  values, wrong column counts (previously bare asserts that vanish under
  python -O), bad n_chunk, a multi-column X passed to
  `estimate_categorical` (it was silently flattened), and a typo'd
  GMM_MI_TORCH_EM_KERNEL (it silently selected the legacy kernels).
- `estimate` and `estimate_categorical` share one selection->bootstrap
  pipeline (they had drifted): the categorical path no longer silently
  drops metric_method under protocol='sequential' (it raises, as estimate
  does), exposes `threshold_components` and `patience` instead of
  hardcoding them, and gains the N > 2^24 bootstrap fallback.
- `n_bootstrap=1` returns std=None instead of NaN (one replicate has no
  sample deviation).

### Memory and precision

- Dead padding components are sliced away after selection: the batched
  protocol pads to K=max_components, and the dead components used to ride
  through the bootstrap, refit and MC stages (~6.7x wasted work at
  k_best=3, max_components=20) while shrinking the auto-sized replicate
  batches by the same factor.
- `val_ll` scores replicates in `em_chunk` batches under the same
  halve-and-retry CUDA out-of-memory guard as `em_fit` (it used to
  evaluate all replicates in one unguarded allocation that could OOM an
  estimate whose EM stage had already fitted), and `_init_params` runs its
  replicate chunks under the same guard.
- `em_fit` and `val_ll` pin float32 matmuls to full precision at entry, so
  direct kernel calls no longer run under ambient TF32; the pin is
  process-global (not thread-local), now documented.

### Packaging, examples, docs

- New `plot` extra (`pip install ".[plot]"`): `gmm_mi_torch.plotting`
  imports numpy, matplotlib and scipy, none of which the base install
  guarantees.
- `examples/mi_grid_multigpu.py` (and the README copy): workers index into
  the inherited CUDA_VISIBLE_DEVICES mask instead of writing absolute GPU
  indices that escape a scheduler's allocation (e.g. Slurm), and the
  script exits cleanly on a machine with no visible GPU.
- Doc fixes: the em_chunk docstring documents the GEMM byte model (not
  just the legacy one); "Piras et al. 2022" -> 2023; README shell quoting
  for zsh; validation/README.md and gb10_report.md corrections; the three
  stale validation bench scripts import the package instead of the
  pre-package flat module they were silently timing.

## 0.2.2 (2026-10-16)

Documentation release; no code changes.

- README: optional fresh-environment steps, references for the GEMM
  formulation, self-contained multi-GPU example (also shipped as
  `examples/mi_grid_multigpu.py`; must be run as a script, not pasted into
  an interactive session).
- Benchmarks refreshed at v0.2.1: the README micro-benchmark re-measured
  (CUDA fp32 selection 0.90 -> 0.21 s on the GB10), and real-data beta
  numbers added (75-pair MI grid, N = 30,000: 53 min on 16 CPU cores vs
  7.5 min fp32 / 5.1 min fp64 on one A100).
- `docs/validation_summary.pdf` regenerated with a section covering
  v0.1.1 - v0.2.1 and the beta-test validation.

## 0.2.1 (2026-08-31)

- Fix an abort ("covariance factorisation failed even after diagonal
  regularisation") on float32 fits of zero-inflated variable pairs, found in
  beta testing. Two causes: float32 moment accumulation could round a
  collapsed component's variance below the `reg_covar` floor (the M-step now
  restores the floor on the diagonal), and the repair bump was scaled by the
  mean diagonal entry, which is negative for exactly the components that
  need repairing (now the largest absolute entry). Non-finite covariances
  are additionally reset to identity instead of blocking the repair loop.

## 0.2.0 (2026-08-29)

- The EM inner loop evaluates Gaussian log-densities and M-step moments as
  matrix products against per-sample quadratic features. Results are
  unchanged (float64 agreement with the previous kernels at reduction-order
  precision); wall time per EM iteration drops by roughly 3x on GPU and 2x
  on CPU. The previous kernels remain available for this release via
  `GMM_MI_TORCH_EM_KERNEL=legacy`.
- Inputs are standardised internally (per variable; the KL quantities share
  one map between the columns). Estimates are invariant, fitted models are
  returned in the original units, and `reg_covar` now acts relative to each
  variable's variance. This removes float32 factorisation failures on
  unstandardised, strongly correlated data.
- A covariance whose factorisation fails is repaired per component with
  escalating diagonal regularisation (and a RuntimeWarning) instead of
  aborting the fit.
- float32 matmul precision is pinned to 'highest' for the duration of each
  call, so TF32 settings made by other libraries do not affect results.
- `gmm_mi_torch.__version__` added.

## 0.1.1 (2026-08-21)

- `em_chunk` and `mc_chunk` default to `'auto'`: on CUDA each stage's
  replicate batch is sized from the free device memory (measured peak-memory
  model), with a floor of 32; on CPU the previous default of 8 is kept. A
  CUDA out-of-memory error halves the chunk and retries. Results are
  chunk-invariant.

## 0.1.0 (2026-07-09)

- Initial release: pure-torch port of GMM-MI with batched selection
  protocols, vectorised bootstrap, analytic marginals, MI / conditional MI /
  total correlation / marginal KL / categorical MI, chunked EM for large N,
  and plotting helpers.
