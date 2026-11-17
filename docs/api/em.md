<!-- markdownlint-disable -->

<a href="../../gmm_mi_torch/em.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `em.py`

Weighted EM batched over a leading replicate dimension: the
GEMM-formulated kernels, the initialisations, `em_fit` and `val_ll`.

**On this page:** [`em_fit`](#function-em_fit) · [`val_ll`](#function-val_ll) · [`EMResult`](#class-emresult) · [`ValLLResult`](#class-valllresult)

---

<a href="../../gmm_mi_torch/em.py#L879"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `em_fit`

```python
em_fit(
    X,
    s,
    w,
    mu,
    cov,
    reg_covar=1e-6,
    threshold_fit=1e-5,
    max_iter=500,
    em_chunk=8,
    n_chunk=None,
)
```

Weighted EM over a leading replicate dimension, in replicate chunks.

Replicates (CV fits, bootstrap realisations) are independent, so they are
processed in chunks of `em_chunk` purely to bound peak memory; results do
not depend on the chunk on CPU (bitwise, with one float32 exception);
on CUDA see `em_chunk`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `torch.Tensor of shape (n_samples, n_variables)` | - | Shared data. |
| `s` | `torch.Tensor of shape (n_replicates, n_samples)` | - | Per-replicate non-negative sample weights: 0/1 masks for CV folds, multinomial counts for bootstrap realisations, ones for a plain fit. The replicate axis may be split over any number of leading dims (shape (..., n_samples), R = their product, row-major): an expand() view then stays a view, and each replicate chunk gathers only its own (chunk, n_samples) rows. |
| `w`, `mu`, `cov` | `torch.Tensor` | - | Initial mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d). |
| `reg_covar`, `threshold_fit` | `float, optional` | `1e-6`, `1e-5` | As in `estimate`. |
| `max_iter` | `int, optional` | `500` | As in `estimate`. |
| `em_chunk` | `int` | `8` | Replicates per batch; a CUDA out-of-memory error halves it and retries (with a RuntimeWarning). Bitwise chunk-invariant on CPU except em_chunk=1 with a one-component model at float32; on CUDA see [docs/estimate_notes.md](../estimate_notes.md), "Reproducibility". |
| `n_chunk` | `int, optional` | `None` | As in `estimate`. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `w`, `mu`, `cov` | `torch.Tensor` | Fitted parameters, same shapes as the inputs. |
| `ll` | `torch.Tensor of shape (n_replicates,)` | Weighted mean per-sample log-likelihood of the parameters that ENTERED the final iteration's M-step: it lags the returned parameters by one M-step (for converged replicates, by less than `threshold_fit`). Score the returned parameters with `val_ll` when an exact final-state likelihood is needed. |

### Raises

- `RuntimeError`: When a covariance cannot be factorised even after escalating diagonal regularisation (raise reg_covar, or use float64), or when a CUDA out-of-memory error persists at one replicate per batch (torch.cuda.OutOfMemoryError). Arguments are not validated here; `estimate` validates them at entry.

### Warns

- `RuntimeWarning`: Once per call for replicates that exhausted `max_iter` without meeting `threshold_fit` (their last iterates are returned), and once for replicates finished on the centred legacy kernels because a component was too narrow for the default kernels' precision. Covariance factorisation also warns when a component needed extra diagonal regularisation, or when an all-zero or non-finite covariance was replaced by the identity.

---

<a href="../../gmm_mi_torch/em.py#L1379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EMResult`

```python
EMResult(
    w: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    ll: torch.Tensor,
    n_unconverged: int,
    n_rerouted: int,
)
```

Result of `_em_fit_core` for one replicate batch.

A tuple, so slicing (`res[:4]`) and positional reads still work.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `w`, `mu`, `cov` | `torch.Tensor` | Fitted parameters, shapes (R, K), (R, K, d), (R, K, d, d). |
| `ll` | `torch.Tensor` | Last E-step's weighted mean per-sample log-likelihood, shape (R,). |
| `n_unconverged` | `int` | Replicates that exhausted `max_iter`. |
| `n_rerouted` | `int` | Replicates the kernel-reroute detector moved to the legacy kernels. |

---

<a href="../../gmm_mi_torch/em.py#L1503"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `val_ll`

```python
val_ll(X, s_val, w, mu, cov, n_chunk=None, em_chunk=None)
```

Weighted mean per-sample log-likelihood of X under each replicate.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `torch.Tensor of shape (n_samples, n_variables)` | - | Shared data. |
| `s_val` | `torch.Tensor of shape (n_replicates, n_samples)` | - | Non-negative weights selecting (and weighting) the scored samples; the replicate axis may span several leading dims, as in `em_fit`. |
| `w`, `mu`, `cov` | `torch.Tensor` | - | Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d). |
| `n_chunk` | `int, optional` | `None` | Stream over samples in chunks of this size (as in `estimate`). |
| `em_chunk` | `int, optional` | `None` | Replicates scored per batch, under the same halve-and-retry CUDA out-of-memory guard as `em_fit`. None scores all replicates in one batch. Results do not depend on it on CPU (bitwise); on CUDA they move at reduction order. |

### Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| - | `torch.Tensor of shape (n_replicates,)` | -inf for a replicate whose weights are all zero (an empty validation fold scores worse than any real fit, instead of a spurious 0). |

### Raises

- `RuntimeError`: As in `em_fit`: a covariance that cannot be factorised even after escalating regularisation, or a CUDA out-of-memory error at one replicate per batch. Arguments are not validated here.

### Warns

- `RuntimeWarning`: Once per call for replicates scored with the centred legacy kernels because a component was too narrow for the default kernels' precision; and, as in `em_fit`, from covariance factorisation (extra diagonal regularisation; all-zero or non-finite covariance replaced by the identity).

---

<a href="../../gmm_mi_torch/em.py#L1607"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValLLResult`

```python
ValLLResult(values: torch.Tensor, n_rerouted: int)
```

Result of `_val_ll_core` for one replicate batch (a tuple).

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `torch.Tensor` | Weighted mean per-sample log-likelihood per replicate, shape (R,). |
| `n_rerouted` | `int` | Replicates scored with the centred legacy kernels. |

---

_This file was generated by `docs/make_api_docs.py` from the docstrings; edit the docstrings, not this file._
