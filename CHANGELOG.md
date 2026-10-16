# Changelog

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
