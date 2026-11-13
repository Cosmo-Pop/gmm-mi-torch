# Notes on `estimate`

The Notes section of the docstring of `gmm_mi_torch.estimate` up to
v0.9.0, moved here verbatim in 0.10.0 so that `help(estimate)` stays
short (the docstring keeps a pointer); the last section, the full lists
of its errors and warnings, moved here from the docstring in 0.11.2.
Where `estimate_categorical` says "as in `estimate`", these notes apply
to it too.

## Selection

*Selection.* The batched protocol fits every candidate k at once, so
no k's evaluation depends on another's: it is robust to initialisation
noise and the fast option on GPU. The sequential protocol is cheaper
on CPU when it stops early, but a single bad initialisation near a
plateau can gate off better models at higher k.

## Standardisation and regularisation

*Standardisation and regularisation.* The data are standardised
internally before fitting: each variable is affinely mapped to zero
mean and unit variance (the KL quantities use a single map shared by
both columns, under which KL is invariant). All estimated quantities
are exactly invariant under these maps, and the `return_fit` model is
transformed back (only through this affine map: with
preprocess='gaussianise' the model stays in rank-probit space), so the
standardisation is unobservable except through `reg_covar` (a
scale-free floor), the `scale` argument of the minmax/randomized
k-means initialisers (in standardised units), and the absence of
float32 failures on unstandardised data. For the KL quantities the
shared map makes the `reg_covar` floor relative to the two columns'
POOLED variance: with strongly mismatched marginal scales, use a
smaller reg_covar so the narrow marginal is not over-regularised.
Smaller values of reg_covar perturb the estimate less but risk failed
Cholesky factorisations; a component whose factorisation still fails
receives escalating diagonal regularisation automatically (with a
RuntimeWarning) instead of aborting the fit.

The standardisation (and the rank transform) run at float64 BEFORE the
cast to the working dtype, so float32 runs are safe on data with large
offsets (MJDs, timestamps): the cast sees already-centred values. This
costs a one-off float64 copy of X at entry. Input that is itself
float32 cannot be un-quantised; pass float64 (or raw) data where
offsets are large relative to the spread. The entry transient is that
float64 copy plus the standardisation's reductions and the finiteness
mask (about two to three times the float64 size of X, more under
gaussianise), on the input's device for tensor input.

## Preprocessing

*Preprocessing.* 'gaussianise' maps each variable to
Phi^{-1}(rank / (N + 1)). MI, cMI and TC are invariant under
per-variable diffeomorphisms, so this changes only the difficulty of
the density estimation: heavy tails and monotone nonlinearities that
would require many components (or defeat the fit entirely) are
removed. It is rejected (ValueError) for the KL quantities, which are
not invariant under marginal transforms. With return_fit=True the
returned model describes the rank-probit-transformed data, not the
original units (the rank transform has no parametric inverse); a
RuntimeWarning says so. tie_rank='random' breaks each column's ties
with a seeded random ordering, drawn independently per column from
seed + 40_000: the MI estimand is exactly unchanged — the original
values are recoverable from the tie-broken ones — and the transformed
marginals are exactly Gaussian, but the GMM estimate is biased low when
ties are shared between columns (joint zero-inflation).
tie_rank='average' gives a tied block a single probit atom, which can
fabricate MI between independent heavily tied columns. A RuntimeWarning
fires when more than 1% of a column's samples sit in tied blocks of
three or more, because no tie convention makes the GMM fit of heavily
tied data reliable — prefer preprocess='none' there.

## Initialisation

*Initialisation.* The default and the k-means inits run their
statistics over the full N per replicate unless `n_chunk` is set, in
which case they stream over sample chunks like the EM stage; without
n_chunk at very large N they bind on memory before EM (a readable
error on CUDA) — set n_chunk or use 'minmax', which allocates nothing
of that size.

## Chunking and memory

*Chunking and memory.* On CUDA, 'auto' sizes each stage's batch to
fill half of the free device memory from the measured peak-memory
models (per replicate, in units of N_eff x itemsize with
N_eff = min(N, n_chunk): max(K+6, 2K+3) plus a small block term for the
default GEMM kernels, (4d+4)K + 4 for the legacy kernels, and
(4d+3)K + 12 for the initialisation stage of the default and k-means
inits, over the full N or the streamed init chunk, which gets its own
batch; mc_samples x itemsize x max((3d+2)K + d + 7, d^2 + 4d + 6) for
mc_chunk), capped at the stage's replicate count, with no replicate
floor, and capped by torch.cuda.set_per_process_memory_fraction when
one is set. A CUDA out-of-memory error during EM halves the chunk and
retries, with a RuntimeWarning. With `n_chunk` the per-replicate
weights are read one sample chunk at a time, so the only N-sized
tensors are the caller's own (the (n_bootstrap, N) resample counts are
held once per stage; the CV masks are views of one (n_folds, N) pair).
The initialisation streams over n_chunk rounded to whole 2^14-sample
draw blocks, at least one (so with n_chunk < 2^14 the init chunk
exceeds n_chunk by less than one block). Streamed results agree with
the single-pass path to floating-point reduction order (ulp-level at
float64, not bitwise).

## Reproducibility

*Reproducibility.* Fixed-seed results are bitwise reproducible on CPU,
including with 'auto' chunk sizing: at float64 no chunk size changes
them; at float32 they are also bitwise except em_chunk=1 with a
one-component model, where the M-step's single-row tail product takes
a different BLAS path (3e-8 in a measured MI).
Each bootstrap replicate draws its MC samples from its own seeded
stream, so the samples are independent of `mc_chunk` and of the
out-of-memory retry on every device. On CUDA, `em_chunk` and `mc_chunk`
change the matmul reduction order at either dtype, so same-seed runs
agree to floating-point noise unless both are pinned to integers and no
out-of-memory retry occurs (the retry halves the batch whatever the
pinning; 'auto' tracks the free memory). At float64 the differences
stay at reduction-order level (1e-14 to 1e-12 in the fitted means in
two measured cases); at float32 the blocked M-step folds the batch's
replicates into one GEMM whose summation order follows the batch size,
and on knife-edge data that can move the selected k (measured: k=5 vs
6, MI 0.4741 vs 0.4745 against an errorbar of 0.008). A CPU run is the
bit-reproducible setting (with the float32 exception above); on CUDA,
float64 is far less sensitive.

## Errors and warnings

The full lists behind the docstring's Raises and Warns sections.

**ValueError**, raised before any fitting:

- Unknown options: `quantity`, `protocol`, `preprocess`, `tie_rank`,
  `metric_method` (or any but 'valid' under protocol='sequential'),
  `init_type`, or a string `em_chunk` or `mc_chunk` other than 'auto'.
- Precision and device: a `dtype` other than float32/float64; a
  `device` that is not 'cpu'/'cuda[:i]', is unavailable or is out of
  range.
- Data: `X` not 2-D, with the wrong number of columns, with NaN/inf
  values, or not representable at `dtype` after standardisation;
  preprocess='gaussianise' with a KL quantity.
- Folds: n_folds < 2, more folds than samples, or a smallest training
  fold of fewer than n_variables + 1 samples.
- Counts and thresholds: n_bootstrap < 0 or > 2^20; |seed| > 2^62;
  mc_samples, max_components, n_inits, max_iter,
  fixed_components_number, patience, n_chunk, em_chunk or mc_chunk
  below 1; reg_covar or threshold_fit not > 0; threshold_components
  not >= 0.

**TypeError**: a non-integer count, seed, em_chunk or mc_chunk (numpy
integers are accepted), or a non-numeric threshold_fit. A non-numeric
reg_covar or threshold_components fails with Python's own comparison
TypeError. A non-numeric `scale` fails inside the initialisation with
float()'s own error (ValueError for a string, TypeError otherwise).

**RuntimeError**: on CUDA, when by the memory model not even one
replicate of the first (initialisation or EM) stage fits (set
`n_chunk`); when no candidate k yields a finite validation metric
(degenerate folds or data); when a covariance cannot be factorised even
after escalating regularisation (raise `reg_covar`, or use float64); or
when a CUDA out-of-memory error persists at one replicate per batch.

**RuntimeWarning**:

- Once per call: fewer than 100 samples (a finite-sample bias the
  errorbar does not include); under preprocess='gaussianise', columns
  with more than 1% of their samples in tied blocks of three or more;
  return_fit=True with preprocess='gaussianise' (the model lives in
  rank-probit space); components of a requested
  fixed_components_number that died in the CV fit.
- Once per skipped k: protocol='sequential' met a non-finite
  validation metric (k skipped without consuming patience).
- Once per em_fit call (each CV fit, the aic/bic and return_fit
  refits, the bootstrap), counting replicates: replicates that
  exhausted `max_iter`; replicates finished on the centred legacy
  kernels because a component collapsed beyond the GEMM kernels'
  precision (expected for surplus components during selection at
  float32). Once per validation-scoring call, likewise, for
  replicates scored with the legacy kernels.
- Once per halving: a CUDA out-of-memory error halved a stage's
  replicate batch and retried.
- Once per covariance factorisation that needs it, which can repeat
  across EM iterations and stages, counting components: extra diagonal
  regularisation; an all-zero or a non-finite covariance replaced
  by the identity.
