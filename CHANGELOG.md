# Changelog

For each release: what changed for a user, and whether results at a fixed
seed change. The record of how each release was verified is in this file
as it stood at the tag `v0.11.2` (`git show v0.11.2:CHANGELOG.md`).

## 0.11.2 (2026-11-13)

Documentation only. The API pages move to `docs/api/`; links to the old
paths (`docs/estimator.md` and so on) no longer resolve. `estimate`'s
docstring is shorter: the full lists of errors and warnings are in
`docs/estimate_notes.md`.

## 0.11.1 (2026-11-13)

Docstrings only. Thirteen small private helpers have short docstrings.

## 0.11.0 (2026-11-13)

Documentation only. An API reference, one page per module, is generated
from the docstrings by `docs/make_api_docs.py` (new `docs` extra).

## 0.10.2 (2026-11-13)

No behaviour change. The README points to `docs/estimate_notes.md`, and two
seed offsets are named constants (`_BOOTSTRAP_SEED_OFFSET`,
`_CATEGORY_SEED_STRIDE`) with the same values.

## 0.10.1 (2026-11-13)

Tests only. Seven tests that patch a setting to reach a code path also
check that the patched path ran.

## 0.10.0 (2026-11-11)

No change to results, defaults, random streams or the text of errors and
warnings.

- The package is split into modules: `constants`, `precision`, `chunking`,
  `gaussian`, `em`, `selection` and `estimator`. Every public name is
  importable from where it was. Private (underscore) names must be imported
  from the module that defines them, and assigning one on
  `gmm_mi_torch.estimator` at run time (for example `_EM_KERNEL`) no longer
  has any effect; the `GMM_MI_TORCH_EM_KERNEL` environment variable is
  unchanged.
- A warning attributed to a frame inside the package names that frame's
  module (for example `gmm_mi_torch.em`, not `gmm_mi_torch.estimator`),
  which affects only a warnings filter keyed on the module name.
- Long comment blocks move to `docs/design_notes.md` and `estimate`'s Notes
  to `docs/estimate_notes.md`; every function, class and public method has
  numpydoc Parameters and Returns sections (a few dunder methods and inner
  wrappers excepted).

## 0.9.0 (2026-11-11)

No behaviour change; internal layout only. The private helpers
`_select_and_fit(X, selection, fit)` and
`_cv_fit_for_k(X, K, n_folds, n_inits, fit, init_chunk=None)` take their
settings as dicts; code that called them directly must pass the dicts.

## 0.8.0 (2026-11-09)

No behaviour change; internal names and layout only. `EMResult`,
`ValLLResult` and `CategoryFit` are named tuples; `em_fit` and `val_ll`
still return plain tuples and tensors.

## 0.7.2 (2026-10-25)

Fixed-seed results are unchanged on valid input.

- A malformed `mc_chunk` raises at entry, before any fitting; it used to
  raise only after component selection.
- `estimate` documents every RuntimeWarning it can issue, with how often
  each is issued.

## 0.7.1 (2026-10-24)

Documentation only.

- Docstrings gain Raises and Warns sections; `docs/design_notes.md` is new;
  the README gains a Known limitations section.
- Corrections to earlier documentation: 'TC' needs at least two columns;
  `scale` also sets the 'minmax' initialisation's standard deviation; and on
  CUDA results depend on the batch size at float64 as well as float32
  (between batch sizes the fitted means moved by 1e-14 to 1e-12 at float64
  and by up to 7e-3 at float32, in measured cases). A CPU run, not "float64
  or CPU", is therefore the bit-reproducible setting, with one float32
  exception: `em_chunk=1` with a one-component model (3e-8 in a measured
  MI).

## 0.7.0 (2026-10-24)

Fixed-seed results change for N > 2^24 only, on both devices; everything
else is bitwise 0.6.1 on CPU.

- Above torch.multinomial's cap of 2^24 categories, the bootstrap resample
  draws its indices as floor(u N) from float64 uniforms. The 32-bit integer
  draw it replaces, reduced modulo N, drew the first 2^32 mod N indices more
  often than the rest (+2.4% on the first 95% of rows at N=1e8). The new
  draw is uniform to 2.2e-8 at N=1e8 and costs about 1.6 times the old one
  (about 50 s on CPU at N=1e8 with 50 replicates).
- For N <= 2^24 the CUDA multinomial draw is kept as it is, with its
  float32-resolution scatter of the per-sample weights (README, Known
  limitations).

## 0.6.1 (2026-10-23)

Fixed-seed results on CPU are unchanged, except for fits with a component
whose variance was clamped to the `reg_covar` floor. On CUDA, automatically
sized batches change only when memory is tight or a per-process memory
fraction is set, and results change with them only through the order of
floating-point summation.

- `n_chunk`, `threshold_fit`, `patience` and `em_chunk` are checked at
  entry, and a non-integer `em_chunk` or `mc_chunk` raises instead of being
  truncated silently (until 0.7.2 `mc_chunk` is checked only after
  component selection). `device` accepts only 'cpu' and 'cuda[:i]'. Missing values among
  string labels raise a readable error.
- A non-finite metric at one k no longer breaks selection: the batched
  protocol's 'valid' metric ranks the finite k only, 'aic' and 'bic' ignore
  a non-finite criterion, and the sequential protocol skips that k with a
  warning.
- Automatic batch sizing: the minimum of 32 replicates is removed, and
  batches respect `torch.cuda.set_per_process_memory_fraction`.
- A float32 fit of duplicated points with `reg_covar` ~1e-10 completes with
  repair warnings instead of aborting.
- Limitations newly documented: the CUDA multinomial's scatter of the
  per-sample weights, the 2^32 per-replicate seed space on CPU, and that on
  CUDA at float32 the batch size can change the selected k.

## 0.6.0 (2026-10-23)

Fixed-seed results change at float32 whenever there is a bootstrap, and on
CUDA with the default initialisation. Two further cases change only through
the order of floating-point summation: with `n_chunk`, direct calls of
`em_fit` or `val_ll` with real-valued weights; and, at float32, the
validation scores of replicates finished on the legacy kernels. Single-pass
float64 results on CPU are bitwise 0.5.0.

- The bootstrap's probability row is float64 whatever the working dtype. At
  float32, torch's CPU multinomial weighted samples unevenly by position in
  the array (a 16-sigma departure from uniform at N=1e6), a bias on
  index-ordered data that the errorbar could not show. The float32 resample
  is now bitwise the float64 one on CPU and CUDA, and uniform on CPU (for
  the CUDA draw see 0.6.1).
- The default and k-means initialisations stream with `n_chunk`. In 0.5.0
  they ran over the full N whatever `n_chunk` was, and could exhaust memory
  before EM started. When not even one replicate fits, a RuntimeError names
  `n_chunk` and `init_type='minmax'`.
- With `n_chunk`, the EM and scoring stages no longer copy the
  (replicates, N) weights: `em_fit` went from 18 to 1.06 times its memory
  model (float32, N=4e5, 32 replicates, `n_chunk=4096`).
- On torch >= 2.9 a float32 precision setting that the caller had left
  inherited is restored as inherited; it used to come back fixed at its
  resolved value. One case cannot be told apart: a setting made explicitly
  equal to its parent's comes back inherited.
- `metric_method='aic'`/`'bic'` on CPU no longer fails under
  `torch.set_default_device('cuda')`.

## 0.5.0 (2026-10-21)

Fixed-seed results change with the default initialisation
(`init_type='random_responsibilities'`), by the scatter between
initialisations, and for N > 2^24. Everything else is bitwise 0.4.0 on CPU;
on CUDA at float32 the M-step also changes through the order of
floating-point summation.

- Lower peak memory in the initialisation, the CV masks, the M-step and the
  bootstrap counts: for example 183 MiB to 3 MiB per CV mask at N=2e5, and
  an end-to-end peak of 1744 MiB reduced to 498 MiB at a 1 GiB budget, with
  the same answer. The memory models behind the 'auto' batch sizes were
  re-measured.
- The default initialisation draws each replicate from its own generator, so
  the draw does not depend on `em_chunk` or on an out-of-memory retry.
- A training fold with fewer than d+1 samples is a ValueError (N = 2-3 used
  to return MI of 5-6 nats silently), and below 100 samples a RuntimeWarning
  names the finite-sample bias. `device` accepts 'cuda:1' and torch.device
  objects. `FittedGMM.score_samples` accepts 1-D input.
- Plotting: repeated mass levels and coarse grids no longer crash the
  contours, and a constant column gives a readable error.

## 0.4.0 (2026-10-20)

Fixed-seed results change: gaussianise breaks ties at random by default, the
Monte-Carlo samples are seeded per replicate, float32 runs on data with
large offsets or point masses now agree with float64, and AIC/BIC values
shift by up to 2 N `threshold_fit`.

- float32 with collapsed components. A component collapsed to the
  `reg_covar` floor away from the origin made the default kernels inaccurate
  at float32: the MI of zero-inflated data was biased by up to +60%. Such
  replicates are now detected and finished on the legacy kernels, with one
  RuntimeWarning. The warning is routine at float32 during selection: on
  ordinary multimodal data 3 of 120 replicates were affected, at about 10%
  more wall time, with MI 0.9065 +- 0.0051 against 0.9052 +- 0.0052 at
  float64.
- gaussianise breaks ties with seeded random ranks (`tie_rank='random'`;
  'average' remains available). Average ranks gave about 0.24 nats between
  independent columns that were each 50% zeros. With random ranks, columns
  that share their zeros are biased low (0.74 nats where the truth is 0.949;
  `preprocess='none'` gives about 0.94), so 'none' is preferred for heavily
  tied data.
- The input is checked, rank-transformed and standardised at float64 before
  the cast to the working dtype. float64 data with offset/sigma above about
  1e6 (MJDs, timestamps) used to be quantised at float32 before it was
  centred: the MI was lost at an offset of 1e8, and the fit aborted at
  timestamp scale. Integer and half-precision inputs are promoted; float16
  and bfloat16 working dtypes are rejected at entry.
- `return_fit`: the covariance is factorised in standardised units and
  transformed back at float64, so data at a scale of 1e-21, or of 1e20 at
  float32, no longer return identity covariances. `FittedGMM`'s attributes are float64 numpy
  arrays.
- Each bootstrap replicate draws its Monte-Carlo samples from its own seeded
  stream, so the samples do not depend on `mc_chunk` or on an out-of-memory
  retry. |seed| <= 2^62 and `n_bootstrap` <= 2^20 are enforced.
- An enclosing `torch.autocast` region is suspended during a call; EM warns
  once per call when replicates reach `max_iter`;
  `torch.set_default_device('cuda')` no longer breaks the CV masks.
- `plotting.find_contours`: the 1-sigma ellipse constant is the chi-squared
  (2 degrees of freedom) value at 0.6827, 2.2958. The inherited 0.77
  enclosed 32% of the mass. The default `contour_levels=[2]` is unaffected.

## 0.3.1 (2026-10-20)

Fixed-seed estimates are unchanged on valid input. `return_k` now counts the
live components of the returned model.

- NaN or infinite input is rejected with a ValueError. Under gaussianise,
  NaN rows used to rank as the largest values and gave 0.33 nats between
  independent columns. Non-finite category labels are rejected too.
- Integer arguments accept numpy integers, and out-of-range values
  (`max_iter` < 1, `reg_covar` <= 0, and so on) are rejected at entry.
- pandas DataFrames are accepted for X, and string category labels for Y.
- The package works with torch >= 2.9's float32-precision settings; once
  user code had set them, every entry point raised.
- `estimate` and `estimate_categorical` run without autograd and detach
  their input. An input with `requires_grad` used to retain the graph of
  every EM iteration (5-30 times the peak CUDA memory).
- A RuntimeWarning is issued when a requested `fixed_components_number`
  loses components during the CV fit.
- An unknown `metric_method` raises also when `fixed_components_number` is
  set, and `select_batched` and `select_sequential` reject `n_folds` > N.
- The `test` extra requires numpy >= 2.0 and scipy.

## 0.3.0 (2026-10-20)

Fixed-seed results change relative to 0.2.2: gaussianise ranks ties by their
average, the CV folds follow the seed, and components with zero weight are
dropped before the bootstrap.

- gaussianise ranks ties by their average instead of by row order, which had
  given 1.06 nats between a sorted column and a constant one (the truth is
  0). A RuntimeWarning names columns with more than 1% tied values, and
  `return_fit` with gaussianise warns that the model describes the
  transformed data.
- When a covariance needed repair, the Monte-Carlo stage sampled from one
  mixture and scored marginals from another; it now uses the repaired one
  throughout.
- Sequential selection: the patience counter resets on improvement and the
  best-scoring k is returned. `patience` is an argument of both estimators.
- The seed controls the CV folds; they were fixed at seed 42.
- Invalid settings are rejected at entry with a ValueError: `n_folds` < 2 or
  > N, an unknown `quantity`, `protocol` or `metric_method` (the last, when
  `fixed_components_number` is set, only from 0.3.1), the wrong number of
  columns, a multi-column X passed to `estimate_categorical` (it was
  flattened silently), an unknown `GMM_MI_TORCH_EM_KERNEL`.
- `estimate_categorical` shares `estimate`'s selection and bootstrap code:
  it gains `threshold_components`, `patience` and the bootstrap for
  N > 2^24, and `metric_method` under `protocol='sequential'` raises, as in
  `estimate`, instead of being dropped silently.
- `n_bootstrap=1` returns std=None instead of NaN.
- Components with zero weight are dropped after selection; they cost about
  6.7 times the necessary work at k=3 with `max_components=20`.
- `val_ll` and the initialisation run in `em_chunk` batches with the
  out-of-memory retry.
- New `plot` extra. The multi-GPU example indexes into the inherited
  `CUDA_VISIBLE_DEVICES` mask.

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
