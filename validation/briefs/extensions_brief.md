# GB10 task: torch_gmm_mi extensions

Files: torch_gmm_mi.py (current, with chunk plumbing), test_torch_gmm_mi.py, conftest.py.

- Clone https://github.com/dpiras/GMM-MI and port each feature from its source, not from this brief's description: `gmm_mi/utils/initializations.py` (init types), `gmm_mi/gmm.py` (KL), `gmm_mi/mi.py` (categorical path), `gmm_mi/cross_validation.py` (AIC/BIC definitions).
- Fix the peak-memory lever from the previous report: chunk `_init_params`'s initial M-step over replicates using the same `chunk` mechanism as `em_fit`.
- Add marginal Gaussianisation preprocessing: `preprocess='none'|'gaussianise'` in `estimate()`, rank-probit per variable (`norm.ppf(rank/(N+1))` equivalent in torch), applied before fitting; MI, cMI, and TC are invariant under per-variable diffeomorphisms so no correction term.
- Port init types 'minmax', 'kmeans', 'randomized_kmeans' as batched-torch equivalents (k-means may run on a subsample per replicate), exposed as `init_type` in both selection functions with 'random_responsibilities' remaining the default.
- Add `fixed_components_number` to `estimate()`: skip selection entirely and fit that k directly.
- Add `metric_method='valid'|'aic'|'bic'` to `select_batched`, matching Davide's AIC/BIC definitions and keeping the parsimony-threshold rule for 'valid'.
- Add KL divergence between the two 1D marginals (forward and reverse), porting the semantics of `estimate_KL_MC` onto the existing marginal-masking machinery, with bootstrap errorbars via the same batched path.
- Add continuous–categorical MI, porting the `fit_categorical`/`estimate_categorical` semantics: per-category GMM fits of the continuous variable, MI assembled from category weights and per-category densities, batched over categories where shapes allow.
- Extend the test suite in the existing sigma-band style, one test per feature: KL vs analytic Gaussian KL; categorical MI vs an analytic construction; Gaussianisation on the warped-marginal case (exp/cubic/tanh warps of a correlated Gaussian, target = sample-covariance TC of the pre-warp data); AIC and BIC each selecting a sensible k on 3-component mixture data; fixed-k reproducing the batched result at the same k.
- Do not change the parsimony rule, the existing estimator integrands, or existing test tolerances.
- Validate the full suite on CPU fp64 (parity) and CUDA fp32 with full-batch `em_chunk` (production config).
- Deliverables: diff, per-config test results, timing/memory deltas from the previous report, updated report.
