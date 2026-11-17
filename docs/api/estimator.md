<!-- markdownlint-disable -->

<a href="../../gmm_mi_torch/estimator.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `estimator.py`

Pure-torch port of GMM-MI with vectorised bootstrap and TC.

One weighted EM, batched over a leading replicate dimension, fits every
initialisation x CV fold, every candidate k and every bootstrap
realisation: per-sample weights stand in for GMM-MI's loops and index
resampling, and every marginal is obtained analytically from the joint
fit. `estimate`, `estimate_categorical` and `FittedGMM` are the public
entry points; `em_fit`, `val_ll`, `select_batched`, `select_sequential`
and the Gaussian helpers are the kernels they are built from (in
`gmm_mi_torch.em`, `.selection` and `.gaussian`, re-exported here). The
environment variable GMM_MI_TORCH_EM_KERNEL ('gemm', the default, or
'legacy') selects the EM kernels at import; any other value raises
ValueError. Design notes, memory models and the measurements behind the
constants: [docs/design_notes.md](../design_notes.md).

**On this page:** [`estimate`](#function-estimate) · [`estimate_categorical`](#function-estimate_categorical) · [`FittedGMM`](#class-fittedgmm) · [`CategoryFit`](#class-categoryfit)

---

<a href="../../gmm_mi_torch/estimator.py#L489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FittedGMM`

```python
FittedGMM(w, mu, cov, chol=None)
```

The GMM selected by `estimate`, refitted on the full data.

Mirrors GMM-MI's fitted-model access for plotting and inspection, with
sklearn-style attributes: `weights_` (K,), `means_` (K, d) and
`covariances_` (K, d, d) as float64 numpy arrays on the CPU (dead
padding components dropped), plus `n_components`. Obtained via
`estimate(..., return_fit=True)`, which passes a Cholesky factor
computed (and, when needed, repaired) in the fit's standardised units,
where reg_covar and the repair thresholds are meaningful. Direct
construction without `chol` factorises whatever units the caller
supplies.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `w`, `mu`, `cov` | `torch.Tensor` | - | Weights (K,), means (K, d) and covariances (K, d, d). |
| `chol` | `torch.Tensor of shape (K, d, d), optional` | `None` | Lower Cholesky factor of `cov`; computed (and repaired if needed) from `cov` when omitted. |

<a href="../../gmm_mi_torch/estimator.py#L525"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample`

```python
sample(n_samples, seed=0)
```

Draw `n_samples` from the mixture.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | `int` | - | Number of samples. |
| `seed` | `int` | `0` | Seed of the sampling generator. |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `numpy.ndarray of shape (n_samples, d)` | The samples, in the units of the fitted model. |

<a href="../../gmm_mi_torch/estimator.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `score_samples`

```python
score_samples(X)
```

Mixture log-density at X.

A 1-D X is n points of a 1-D model, or one point of a d-D model
(sklearn rejects 1-D input); any other shape raises a ValueError naming
the expected one.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `array-like of shape (n, d) or (n,)` | Points, in the units of the fitted model. |

#### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `numpy.ndarray of shape (n,)` | Log-density at each point. |

---

<a href="../../gmm_mi_torch/estimator.py#L619"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `estimate`

```python
estimate(
    X,
    quantity='MI',
    protocol='batched',
    n_bootstrap=50,
    mc_samples=100_000,
    max_components=20,
    n_folds=2,
    n_inits=3,
    reg_covar=1e-6,
    threshold_fit=1e-5,
    max_iter=500,
    seed=0,
    return_k=False,
    return_fit=False,
    device='cpu',
    dtype=torch.float64,
    em_chunk='auto',
    mc_chunk='auto',
    preprocess='none',
    tie_rank='random',
    init_type='random_responsibilities',
    scale=None,
    fixed_components_number=None,
    metric_method='valid',
    threshold_components=1e-5,
    patience=1,
    n_chunk=None,
)
```

Estimate an information-theoretic quantity with bootstrap errorbars.

One joint GMM is fitted to all columns of `X` (component count chosen by
cross-validation); every marginal needed by the requested quantity is then
obtained analytically from that single fit, so marginals are exactly
consistent with the joint. The bootstrap is a single batched EM:
multinomial-count sample weights over `n_bootstrap` replicates, each
warm-started from the CV-selected parameters.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `array-like of shape (n_samples, n_variables)` | - | The data. Two columns for 'MI' and the KL quantities; at least three for 'cMI' (variables 0 and 1 are X and Y, the rest condition); at least two for 'TC'. |
| `quantity` | `{'MI', 'cMI', 'TC', 'KL_forward', 'KL_reverse'}` | `'MI'` | 'MI': I(X;Y); 'cMI': I(X;Y\|Z), Z the columns beyond the first two; 'TC': total correlation KL[p(x) \|\| prod_i p(x_i)], zero iff the variables are mutually independent; 'KL_forward': KL[p(x0) \|\| p(x1)] between the two 1D marginals; 'KL_reverse': the opposite order. |
| `protocol` | `{'batched', 'sequential'}` | `'batched'` | Component selection. 'batched' fits k = 1..max_components at once and picks the smallest k within `threshold_components` of the best validation metric; 'sequential' is GMM-MI's original protocol (increment k, stop early via threshold and patience). See Notes. |
| `n_bootstrap` | `int` | `50` | Bootstrap realisations for the errorbar, batched per `em_chunk`. 0 skips the bootstrap (a single fit on the full data). |
| `mc_samples` | `int` | `100_000` | Monte Carlo samples per bootstrap replicate for the integral; more reduce the MC error, at linear cost in time. |
| `max_components` | `int` | `20` | Largest number of components considered. The batched protocol fits them all, so keep this to a plausible ceiling for the data. |
| `n_folds` | `int` | `2` | Cross-validation folds; each should stay representative of the data. |
| `n_inits` | `int` | `3` | Initialisations per fold; more reduce the chance of a local optimum, at proportional cost (batched away on GPU). |
| `reg_covar` | `float` | `1e-6` | Constant added to every covariance diagonal to prevent singularities, relative to each variable's variance (the data are standardised; see Notes). 1e-6 is safe at float32, where GMM-MI's 1e-15 would regularise nothing. |
| `threshold_fit` | `float` | `1e-5` | EM convergence threshold on the change in mean per-sample log-likelihood (sklearn's `tol`). |
| `max_iter` | `int` | `500` | Maximum EM iterations per fit. |
| `seed` | `int` | `0` | Seed controlling initialisations, CV splits, bootstrap resampling, MC integration and gaussianise's tie-breaking. Bitwise reproducible on CPU; on CUDA see Notes, "Reproducibility". |
| `return_k` | `bool` | `False` | If True, also return the selected number of components. |
| `return_fit` | `bool` | `False` | If True, also return the selected model as a `FittedGMM` (refitted on the full data, warm-started like the bootstrap replicates from the CV-selected parameters), e.g. for plotting; see Notes for its units. |
| `device` | `str or torch.device` | `'cpu'` | Torch device on which to fit and integrate: 'cpu', or 'cuda' / 'cuda:i' / a torch.device of those types (other backends are rejected at entry). |
| `dtype` | `torch.dtype` | `torch.float64` | Working precision. float32 is validated and much faster on GPUs with limited fp64 throughput; float64 is the reference precision. |
| `em_chunk` | `int or 'auto'` | `'auto'` | EM replicates (CV fits or bootstrap realisations) per batch; the initialisation and validation-LL stages batch the same way. Larger is faster on GPU at higher peak memory. 'auto' is 8 on CPU and fills half the free device memory on CUDA (see Notes). |
| `mc_chunk` | `int or 'auto'` | `'auto'` | Bootstrap replicates whose MC integration is evaluated per batch; 'auto' as for `em_chunk`. The MC samples do not depend on it; results are bitwise invariant on CPU, and on CUDA change only through the order of floating-point summation. |
| `preprocess` | `{'none', 'gaussianise'}` | `'none'` | 'gaussianise' applies a per-variable rank-probit transform before fitting, which leaves MI, cMI and TC unchanged but can make the density much easier to fit. Not for heavily tied data, and rejected for the KL quantities (see Notes). |
| `tie_rank` | `{'random', 'average'}` | `'random'` | How gaussianise ranks tied values: 'random' breaks each column's ties with a seeded random ordering; 'average' is the deterministic scipy rankdata convention. See Notes. |
| `init_type` | `str` | `'random_responsibilities'` | GMM initialisation scheme (torch ports of GMM-MI's initialisations): 'random_responsibilities', 'minmax', 'kmeans' or 'randomized_kmeans'. 'random_responsibilities' is cheap and adequate for easy fits; the k-means variants are more robust on well-separated multimodal data. See Notes for their memory at large N. |
| `scale` | `float, optional` | `None` | Mean-perturbation scale for 'randomized_kmeans' (GMM-MI's default 1 when None), or the initial isotropic standard deviation for 'minmax' (GMM-MI's n-ball rule when None); in standardised units. |
| `fixed_components_number` | `int, optional` | `None` | If set, skip component selection entirely and fit this many components (best CV init/fold retained, as GMM-MI's fixed_components). Use when k is already known. |
| `metric_method` | `{'valid', 'aic', 'bic'}` | `'valid'` | Selection metric for the batched protocol. 'valid' is the validation log-likelihood with the parsimony-threshold rule; 'aic'/'bic' refit each k's best CV parameters on the full data and minimise the criterion, as in GMM-MI. |
| `threshold_components` | `float` | `1e-5` | Parsimony threshold for the batched protocol's 'valid' metric (and the improvement threshold for the sequential protocol): metric gains smaller than this do not justify a larger k. |
| `patience` | `int` | `1` | Sequential protocol only: consecutive non-improving k values tolerated before the search stops (as in `select_sequential`). |
| `n_chunk` | `int, optional` | `None` | If set, the EM and validation-LL kernels stream over sample chunks of this size, bounding their peak memory by em_chunk \* n_chunk rather than N, and the default/k-means initialisation streams too. Results agree with the single pass up to the order of floating-point summation. See Notes. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `mean` | `float` | Mean of the estimated quantity over the bootstrap distribution (or the single-fit value when n_bootstrap=0), in nats. |
| `std` | `float or None` | Bootstrap standard deviation (None when n_bootstrap < 2: one replicate has no sample deviation). |
| `k` | `int` | Selected number of components; only if `return_k` is True. |
| `fit` | `FittedGMM` | The selected model refitted on the full data; only if `return_fit` is True (appended after `k` when both flags are set). |

### Raises

- `ValueError`: An unknown option or an invalid combination of options, an out-of-range count or threshold, an unsupported dtype or device, data of the wrong shape or with NaN/inf values, or folds too small for the data; raised before any fitting. The full list: [docs/estimate_notes.md](../estimate_notes.md), "Errors and warnings".
- `TypeError`: A non-integer count, seed, em_chunk or mc_chunk (numpy integers are accepted), or a non-numeric threshold or reg_covar.
- `RuntimeError`: On CUDA, when not even one replicate fits in memory (set `n_chunk`) or an out-of-memory error persists at one replicate per batch; when no candidate k yields a finite validation metric; or when a covariance cannot be factorised (raise `reg_covar`, or use float64).

### Warns

- `RuntimeWarning`: For a result that may need a look: fewer than 100 samples, tied data or return_fit under gaussianise, components that died in a fixed-k fit, a skipped k in the sequential protocol, replicates that hit `max_iter` or finished on the legacy kernels, a CUDA out-of-memory retry, or a covariance given extra regularisation or replaced. The full list, with how often each is issued: [docs/estimate_notes.md](../estimate_notes.md), "Errors and warnings".

### Notes

On selection, standardisation, preprocessing, initialisation, memory,
reproducibility, and the full lists of errors and warnings:
[docs/estimate_notes.md](../estimate_notes.md) in the repository,
https://github.com/Cosmo-Pop/gmm-mi-torch/blob/main/docs/estimate_notes.md

---

<a href="../../gmm_mi_torch/estimator.py#L940"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoryFit`

```python
CategoryFit(
    w: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    chol: torch.Tensor,
)
```

Bootstrap fits of one category in `estimate_categorical` (a tuple).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `w`, `mu`, `cov` | `torch.Tensor` | Fitted parameters over the B bootstrap replicates, shapes (B, K), (B, K, 1), (B, K, 1, 1); `cov` as repaired by `_chol`. |
| `chol` | `torch.Tensor` | Cholesky factors of `cov`, shape (B, K, 1, 1). |

---

<a href="../../gmm_mi_torch/estimator.py#L1016"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `estimate_categorical`

```python
estimate_categorical(
    X,
    Y,
    protocol='batched',
    n_bootstrap=50,
    mc_samples=100_000,
    max_components=10,
    n_folds=2,
    n_inits=3,
    reg_covar=1e-6,
    threshold_fit=1e-5,
    max_iter=500,
    seed=0,
    return_k=False,
    device='cpu',
    dtype=torch.float64,
    em_chunk='auto',
    mc_chunk='auto',
    init_type='random_responsibilities',
    scale=None,
    fixed_components_number=None,
    metric_method='valid',
    threshold_components=1e-5,
    patience=1,
    category_weights='empirical',
    n_chunk=None,
)
```

Estimate MI between a continuous variable and a categorical variable.

Port of GMM-MI's fit_categorical/estimate_categorical semantics
(Appendix B of Piras et al. 2023): a 1D GMM p(x|c) is fitted per
category, each with full component selection, and

    MI = sum_c pi_c E_{x~p(x|c)}[ log p(x|c) - log sum_c' pi_c' p(x|c') ]

The bootstrap is stratified: each category's data are resampled with
multinomial weights and refit warm-started from that category's
CV-selected parameters, batched over replicates. pi is held fixed across
replicates, so the errorbar covers the density fits but not the
multinomial sampling uncertainty of the category proportions themselves
(negligible except at small N with extreme imbalance).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `array-like of shape (n_samples,) or (n_samples, 1)` | - | The continuous variable. |
| `Y` | `array-like of shape (n_samples,)` | - | The categorical variable; categories are its unique values. Numeric labels are used as-is; string/object labels are factorised through numpy's unique (categories in sorted label order). Non-finite (NaN/inf) and missing (None/NaN) labels are rejected — torch.unique would otherwise make every NaN its own category. |
| `protocol` | `{'batched', 'sequential'}` | `'batched'` | As in `estimate`, applied per category. |
| `n_bootstrap`, `mc_samples` | `int, optional` | `50`, `100_000` | As in `estimate`, applied per category. |
| `max_components` | `int` | `10` | Largest number of components considered per category's 1D fit. |
| `n_folds`, `n_inits` | `int, optional` | `2`, `3` | As in `estimate`, applied per category. |
| `reg_covar` | `float` | `1e-6` | As in `estimate`; X is standardised once over all categories, so the floor is relative to the variance of X as a whole. |
| `threshold_fit` | `float` | `1e-5` | As in `estimate`. |
| `max_iter`, `seed` | `int, optional` | `500`, `0` | As in `estimate`. |
| `return_k` | `bool` | `False` | If True, also return the selected component count per category. |
| `device` | `str or torch.device` | `'cpu'` | As in `estimate`. |
| `dtype` | `torch.dtype` | `torch.float64` | As in `estimate`. |
| `em_chunk`, `mc_chunk` | `int or 'auto', optional` | `'auto'`, `'auto'` | As in `estimate`. |
| `init_type` | `str` | `'random_responsibilities'` | As in `estimate`, applied per category. |
| `scale` | `float, optional` | `None` | As in `estimate`, applied per category. |
| `fixed_components_number` | `int, optional` | `None` | As in `estimate`, applied per category. |
| `metric_method` | `{'valid', 'aic', 'bic'}` | `'valid'` | As in `estimate`, applied per category. |
| `threshold_components` | `float` | `1e-5` | As in `estimate`, applied per category. |
| `patience` | `int` | `1` | As in `estimate`, applied per category. |
| `category_weights` | `{'empirical', 'uniform'}` | `'empirical'` | 'empirical' sets pi_c = N_c / N from the observed category counts — the correct weighting for MI of the observed joint (X, Y). 'uniform' sets pi_c = 1/C, reproducing the reference code, whose Appendix B formula assumes equiprobable categories. |
| `n_chunk` | `int, optional` | `None` | As in `estimate`, applied per category. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `mean` | `float` | Bootstrap-mean MI in nats. |
| `std` | `float or None` | Bootstrap standard deviation (None when n_bootstrap < 2: one replicate has no sample deviation). |
| `k` | `list of int` | Selected component count per category; only if `return_k` is True. |

### Raises

- `ValueError`: Before any fitting, for: `X` not 1-D or a single column, with NaN/inf values, or not representable at `dtype` after standardisation; `Y` not 1-D, of a different length from `X`, with NaN/inf or missing (None/NaN) labels, or mixing label types that cannot be ordered; a smallest category too small for `n_folds` (as `estimate` checks X, with d = 1); an unknown `category_weights`; and those of `estimate`'s argument errors that apply to this signature (full list: [docs/estimate_notes.md](../estimate_notes.md), "Errors and warnings").
- `TypeError`: As in `estimate`.
- `RuntimeError`: As in `estimate`, for any category's fit.

### Warns

- `RuntimeWarning`: As in `estimate`, for each category's selection, bootstrap and Monte-Carlo stages; the small-sample warning checks the smallest category, and the gaussianise and return_fit warnings do not arise.

---

_This file was generated by `docs/make_api_docs.py` from the docstrings; edit the docstrings, not this file._
