<!-- markdownlint-disable -->

<a href="../gmm_mi_torch/selection.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `selection.py`

Component selection: the cross-validation masks and the sequential and
batched protocols.

---

<a href="../gmm_mi_torch/selection.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `select_sequential`

```python
select_sequential(
    X,
    max_components=20,
    n_folds=2,
    n_inits=3,
    threshold_components=1e-05,
    patience=1,
    reg_covar=1e-06,
    threshold_fit=1e-05,
    max_iter=500,
    seed=0,
    em_chunk=8,
    init_type='random_responsibilities',
    scale=None,
    n_chunk=None,
    init_chunk=None,
)
```

Select the number of components by GMM-MI's sequential protocol.

Components are added one at a time; the search stops when the validation
metric fails to improve by `threshold_components` for `patience`
consecutive values of k (the counter resets whenever k improves), and
the best-scoring k seen so far is selected — also when the search
reaches `max_components` without stopping. Cheaper than the batched
protocol when it stops early, but the stopping decision is made under
initialisation noise: one poor fit at a plateau can gate off better
models at higher k.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `torch.Tensor of shape (n_samples, n_variables)` | - | The data, already on the target device/dtype. |
| `max_components` | `int, default=20` | `20` | Largest k tried if the early stop never triggers. |
| `n_folds`, `n_inits` | `int, optional` | `2`, `3` | As in `estimate`. |
| `threshold_components` | `float, default=1e-5` | `1e-05` | Minimum metric improvement that counts as progress. |
| `patience` | `int, default=1` | `1` | Number of consecutive non-improving k values tolerated before stopping; the counter resets on every improvement, and the selected k is the best-scoring one. |
| `reg_covar`, `threshold_fit` | `float, optional` | `1e-06`, `1e-05` | As in `estimate`. |
| `max_iter`, `seed` | `int, optional` | `500`, `0` | As in `estimate`. |
| `em_chunk` | `int, default=8` | `8` | As in `estimate`, an integer here. |
| `init_type` | `str, default='random_responsibilities'` | `'random_responsibilities'` | As in `estimate`. |
| `scale` | `float, optional` | `None` | As in `estimate`. |
| `n_chunk` | `int, optional` | `None` | As in `estimate`. |
| `init_chunk` | `int, optional` | `None` | Replicates per batch for the initialisation stage (default em_chunk); `estimate` sizes it from the init stage's own memory model. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `k_best` | `int` | Selected number of components. |
| `params` | `tuple of torch.Tensor` | (weights, means, covariances) of the best CV fit at k_best. |
| `history` | `dict` | Per-k (metric, params) from the search. |

### Raises

- `ValueError`: n_folds < 2 or more folds than samples (before any fit); an unknown `init_type` (at the first initialisation).
- `RuntimeError`: When no k yields a finite validation metric; and as in `em_fit` (a covariance that cannot be factorised, a CUDA out-of-memory error at one replicate per batch).

---

<a href="../gmm_mi_torch/selection.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `select_batched`

```python
select_batched(
    X,
    max_components=20,
    n_folds=2,
    n_inits=3,
    threshold_components=1e-05,
    reg_covar=1e-06,
    threshold_fit=1e-05,
    max_iter=500,
    seed=0,
    em_chunk=8,
    init_type='random_responsibilities',
    scale=None,
    metric_method='valid',
    n_chunk=None,
    init_chunk=None,
)
```

Select the number of components with all candidate k fitted at once.

Every k = 1..max_components (times inits and folds) is fitted as a single
batched EM via dead-component padding, so no k's evaluation depends on
any other's — the protocol is immune to the initialisation-noise gating
of the sequential search, and is the fast option on GPU.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `torch.Tensor of shape (n_samples, n_variables)` | - | The data, already on the target device/dtype. |
| `max_components` | `int, default=20` | `20` | Largest k fitted; all are paid for unconditionally. |
| `n_folds`, `n_inits` | `int, optional` | `2`, `3` | As in `estimate`. |
| `threshold_components` | `float, default=1e-5` | `1e-05` | Parsimony rule for metric_method='valid': the smallest k whose validation log-likelihood is within this threshold of the best wins, so noise-level improvements do not inflate k. |
| `reg_covar`, `threshold_fit` | `float, optional` | `1e-06`, `1e-05` | As in `estimate`. |
| `max_iter`, `seed` | `int, optional` | `500`, `0` | As in `estimate`. |
| `em_chunk` | `int, default=8` | `8` | As in `estimate`, an integer here. |
| `init_type` | `str, default='random_responsibilities'` | `'random_responsibilities'` | As in `estimate`. |
| `scale` | `float, optional` | `None` | As in `estimate`. |
| `metric_method` | `{'valid', 'aic', 'bic'}, default='valid'` | `'valid'` | 'valid' scores held-out log-likelihood with the parsimony rule. 'aic'/'bic' follow GMM-MI: each k's best CV parameters are refit on the full data, the criterion is computed on it, and the minimising k wins; the returned parameters are still the CV-selected ones. |
| `n_chunk` | `int, optional` | `None` | As in `estimate`. |
| `init_chunk` | `int, optional` | `None` | Replicates per batch for the initialisation stage (default em_chunk), as in `select_sequential`. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `k_best` | `int` | Selected number of components. |
| `params` | `tuple of torch.Tensor` | (weights, means, covariances) of the best CV fit at k_best. |
| `history` | `dict` | Per-k metrics; params entries are None for every k (the selected parameters are returned separately). |

### Raises

- `ValueError`: n_folds < 2 or more folds than samples (before any fit); an unknown `init_type` (at the initialisation); an unknown `metric_method` (only after the CV fit and scoring).
- `RuntimeError`: When no k yields a finite validation metric (or, for 'aic'/'bic', a finite criterion); and as in `em_fit` (a covariance that cannot be factorised, a CUDA out-of-memory error at one replicate per batch).

---

_This file was generated by `docs/make_api_docs.py` from the docstrings of gmm-mi-torch 0.11.0; edit the docstrings, not this file._
