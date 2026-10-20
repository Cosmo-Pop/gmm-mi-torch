# Changelog

## 0.4.0 (2026-10-20)

The design pass for the remaining 2026-10-20 review findings, built under
report-before-code: each change below was proposed in
scoping/design-report-v0.4.0-2026-10-20.md, ruled on by an independent
reviewer (scoping/design-review-v0.4.0-2026-10-20.md, all amendments
adopted), and landed with the regression tests the ruling asked for.
Intentional behaviour changes: gaussianise ties are randomised by default,
the Monte-Carlo streams are reseeded per replicate, float32 runs on
large-offset or spike-bearing data now return the float64 answer (they
were wrong before), and AIC/BIC values shift by up to 2*N*threshold_fit
(same selections in all tests).

- Float32 is now safe on collapsed components: the GEMM kernels' uncentred
  moments leave a rounding residue ~ u*P*|x|^2 that swamped a
  reg_covar-floored spike away from the origin (zero-inflated MI biased up
  to +60% at float32 under nothing but repair warnings; float64 kernel
  parity broken below reg_covar ~1e-13). em_fit/val_ll now detect such
  replicates on the parameters entering every E-step and finish them on
  the centred legacy kernels, per replicate and chunk-invariantly, with
  one aggregated RuntimeWarning. Measured on the GB10 at float32 defaults:
  clean data trips nothing (+3% wall for the detector); 30%-zero-inflated
  N=20,000 pays +6% wall and returns 0.966+-0.005 where the undetected
  kernels gave 1.007+-0.064. The detector is deliberately blind to
  near-collinear components with O(1) variances, where the float32 GEMM
  kernels are accurate and the legacy kernels are the worse choice; that
  class is pinned by a must-not-reroute test and documented in the kernel
  section.
- gaussianise breaks ties with seeded random ranks by default
  (tie_rank='random'; 'average' remains available): the v0.3.0 average
  ranks mapped a tied block to one probit atom and fabricated ~0.24 nats
  between INDEPENDENT 50%-zero-inflated columns. Random tie-ranks leave
  the MI estimand exactly unchanged (the original values are recoverable
  from the tie-broken ones) and restore ~0 there. Measured caveat, stated
  in the tie warning: on DEPENDENT atoms (a shared zero mask, truth 0.949
  nats) the randomised transform is biased low (~0.74) while
  preprocess='none' lands at ~0.94 — prefer 'none' for heavily tied data.
  The tie warning also now ignores two-sample blocks (float32-native data
  acquires harmless birthday pairs at large N).
- The entry pipeline (finiteness check, rank transform, standardisation)
  runs at float64 BEFORE the cast to the working dtype: float64 data with
  offset/sigma >~ 1e6 (MJDs, timestamps) used to be quantised at float32
  before the centring could protect it — MI silently destroyed at offset
  1e8, a hard abort at timestamp scale. float32 now tracks float64 at
  every offset. Integer/half inputs are promoted; fp16/bf16 working dtypes
  are rejected at entry.
- return_fit: the covariance is factorised (and repaired) in the fit's
  standardised units and back-transformed together with its Cholesky
  factor at float64. Strain-scale (1e-21) data no longer comes back with
  identity covariances, mixed column scales no longer inflate a repaired
  small column ~100x, and 1e20-scale float32 fits no longer overflow to
  identity. FittedGMM attributes are float64 numpy.
- Reproducibility: each bootstrap replicate draws its MC samples from its
  own seeded stream, so results are independent of mc_chunk and of
  out-of-memory retries by construction (and the GB10 measures the
  per-replicate draws slightly faster than the batched one). Fixed-seed
  results are bitwise reproducible on CPU including with 'auto' chunks;
  on CUDA, em_chunk changes reduction order (documented). |seed| <= 2**62
  and n_bootstrap <= 2**20 are enforced so the stream offsets stay valid.
- The precision pin also suspends ambient torch.autocast (a bf16 region
  used to change both the estimate and the selected k); EM warns once per
  call when replicates exhaust max_iter unconverged; AIC/BIC score the
  refit parameters exactly (em_fit's log-likelihood lags its returned
  parameters by one M-step — now documented); val_ll accumulates at
  float64 across sample chunks like the EM kernels;
  torch.set_default_device('cuda') no longer breaks the CV masks.
- plotting.find_contours: the 1-sigma ellipse constant becomes chi2(2 dof)
  at 0.6827 (2.2958); the inherited 0.77 was the inverted quantile, whose
  ellipse encloses 32% of the mass (upstream GMM-MI has the same constant;
  default contour_levels=[2] unaffected). The reg_covar docstring states
  the KL pooled-variance exception.

## 0.3.1 (2026-10-20)

Mechanical fix release addressing the no-design-decision findings of the
2026-10-20 adversarial review of v0.3.0 (the larger items — the GEMM
kernels' origin-moment cancellation, the gaussianise tie policy on
zero-inflated data, the dtype-cast/standardisation ordering, FittedGMM's
factorisation units, MC chunk reproducibility semantics and the memory
models — are held for design decisions). Estimates for a given seed are
unchanged on valid input; the one reporting change is that `return_k` now
counts the live components of the returned model.

### Entry validation and input handling

- Non-finite input (NaN/inf) is rejected with a ValueError at both entry
  points. With `preprocess='gaussianise'`, NaN rows used to rank silently
  as the largest values and fabricate MI on independent data (0.33 nats
  measured where the truth is 0, no warning); on the 'none' path a single
  NaN poisoned every replicate through the sample-weight product, defeated
  the EM stop test (exactly max_iter 'non-finite covariance reset to
  identity' warnings per stage) and died in a path-dependent far-away
  IndexError. Non-finite category labels are rejected too: torch.unique
  made every NaN its own category, surfacing as a misleading n_folds
  error. `select_batched` now raises the sequential path's informative
  RuntimeError when no validation metric is finite, instead of an
  IndexError from the parsimony mask.
- Integer arguments (seed, n_bootstrap, mc_samples, max_components,
  n_folds, n_inits, max_iter, fixed_components_number) are coerced with
  operator.index at entry: numpy integer seeds — what `rng.integers`
  yields in seed sweeps — crashed `_cv_masks` with 'TypeError: an integer
  is required', and a float mc_samples/n_bootstrap crashed only after the
  selection and bootstrap EM were already paid for. Floors are enforced:
  max_iter >= 1 (0 silently returned the random init, every bootstrap
  replicate identical and the errorbar pure MC noise), reg_covar > 0
  (0 let collapsed components through to garbage estimates or
  factorisation failures), mc_samples/max_components/n_inits/
  fixed_components_number >= 1, n_bootstrap >= 0, threshold_components
  >= 0 (negative made the parsimony mask all-False -> IndexError).
- The protocol/metric_method compatibility rules run before the fixed-k
  shortcut (a typo'd metric_method ran silently when
  fixed_components_number was set, contradicting the 0.3.0 'raise at
  entry' claim), and the public `select_batched`/`select_sequential`
  validate fold feasibility as the estimators do (n_folds > N silently
  returned k=1 with every metric -inf).
- pandas DataFrames (and any `__array__` container) are accepted for X,
  as the 'array-like' docstring promised; read-only arrays are copied so
  torch does not warn. String/object category labels are factorised
  through numpy's unique (categories in sorted label order) instead of
  dying in torch.as_tensor with a raw TypeError, and the
  smallest-category fold error names the offending label.

### Precision pin, autograd, reporting

- `_full_precision` speaks torch >= 2.9's per-backend fp32_precision API.
  Once user code had set any of those knobs — the documented way to
  enable TF32 since 2.9 — the legacy getter raised RuntimeError ('mix of
  the legacy and new APIs') and every public entry point crashed, even on
  CPU. The pin now saves, sets to 'ieee' and restores the generic,
  cuda.matmul and mkldnn.matmul settings exactly (the legacy-setter
  restore used to rewrite the mkldnn backend for a caller who had only
  enabled CUDA TF32), falling back to the legacy get/set pair on older
  torch. A CUDA-gated test asserts a TF32-sensitive GEMM result inside
  em_fit under both APIs — the old test asserted a flag string and was
  structurally blind to both defects.
- `estimate` and `estimate_categorical` run under torch.no_grad() and
  detach the input after as_tensor: an input carrying requires_grad
  (model latents taken without .detach() — the primary use case) built
  and retained the EM autograd graph of every iteration (measured 5-30x
  peak CUDA memory, growing with iteration count) and return_fit=True
  then crashed in FittedGMM on .numpy(). em_fit and val_ll are unchanged:
  differentiating a log-likelihood with respect to the parameters remains
  possible.
- `return_k` reports the live component count of the model that is
  actually bootstrapped, MC-integrated and returned: the 0.3.0 dead-
  component slice could silently return a smaller model under the
  selected/requested k (fixed_components_number=6 returning a 5-component
  fit). A RuntimeWarning fires when a requested fixed_components_number
  loses components during the CV fit.
- The gaussianise tie warning and the return_fit warning attribute to the
  caller's frame instead of the decorator line inside estimator.py, so
  Python's once-per-location filtering distinguishes call sites and
  module-keyed warning filters match the caller.

### Packaging, tests, validation scripts

- `[test]` extra: numpy >= 2.0 (np.trapezoid is 2.0-only; three
  categorical tests failed on numpy 1.x) and scipy (two rank-probit
  oracle tests import scipy.stats, previously only in the [plot] extra),
  so the documented `pip install -e ".[test]" && pytest` passes in a
  fresh environment.
- The 8 bench scripts under validation/ crashed on 0.3.0's `_chol` tuple
  return before any measurement (only bench.py had been updated); all
  call sites unpack the tuple now and validation/README.md records the
  broken window.
- README: the test-count sentence no longer double-counts the
  offline-gated test and names both gated tests; the dependency sentence
  matches the [test] extra.
- test_mc_chunk_auto_within_mc_noise compared mc_chunk=8 against CPU
  'auto' (= 8): bitwise the same run, so it could never fail on the
  documented default pytest invocation. It now compares mc_chunk=3
  against 'auto', which differ on every leg of the test matrix.

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
