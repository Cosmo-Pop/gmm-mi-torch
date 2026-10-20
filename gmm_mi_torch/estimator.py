"""Pure-torch port of GMM-MI with vectorised bootstrap and TC.

Design notes
------------
- No sklearn. EM is implemented as *weighted* EM over a leading replicate
  dimension R. Per-sample weights unify everything that GMM-MI does with
  loops and index resampling:
    * CV folds  -> 0/1 masks over the full dataset (train mask; complement
      scores validation LL), so unequal fold sizes are free;
    * bootstrap -> multinomial(N, 1/N) counts as weights (identical in
      distribution to index resampling);
    * inits     -> just more replicates.
  One EM loop therefore fits (n_inits x n_folds), or B bootstrap
  realisations, or (K_max x n_inits x n_folds), as a single batched
  tensor computation.
- Marginalisation is analytic index masking of the joint GMM, as in
  GMM-MI, so MI / conditional MI / total correlation all come from one
  joint fit and the marginals are exactly consistent with the joint.
- Two component-selection protocols:
    'sequential' : GMM-MI's protocol (increment k, validation-LL metric,
                   improvement threshold_components + patience early stop).
    'batched'    : fit k = 1..K_max simultaneously (dead-component
                   padding to a common tensor shape), argmax validation LL.
"""

import functools
import math
import operator
import os
import warnings

import torch

# EM kernel implementation. 'gemm' evaluates the Gaussian log-densities
# and the M-step moments as matrix products against per-sample quadratic
# features (see the kernel section below); 'legacy' selects the previous
# broadcast kernels. Both give the same results. A typo must fail here,
# not silently select the legacy kernels.
_EM_KERNEL = os.environ.get('GMM_MI_TORCH_EM_KERNEL', 'gemm')
if _EM_KERNEL not in ('gemm', 'legacy'):
    raise ValueError(f"GMM_MI_TORCH_EM_KERNEL must be 'gemm' or 'legacy'; "
                     f"found '{_EM_KERNEL}'")


# torch >= 2.9 replaces get/set_float32_matmul_precision with per-backend
# fp32_precision settings; once user code has set any of those, the legacy
# getter RAISES ("mix of the legacy and new APIs"), which used to kill every
# public entry point here even on CPU. The per-backend settings are also what
# make an exact restore possible: the legacy setter rewrites every backend at
# once, so restoring through it handed a caller who had only enabled CUDA
# TF32 a changed mkldnn (CPU) setting.
try:
    _FP32_KNOBS = (torch.backends, torch.backends.cuda.matmul,
                   torch.backends.mkldnn.matmul)
    for _knob in _FP32_KNOBS:
        _knob.fp32_precision
except AttributeError:      # torch < 2.9: legacy API only
    _FP32_KNOBS = None


class _pinned_precision:
    """Context manager: float32 matmuls at full (ieee) precision inside the
    block, the caller's exact settings restored on exit. Reentrant."""

    def __enter__(self):
        if _FP32_KNOBS is not None:
            self._prev = [knob.fp32_precision for knob in _FP32_KNOBS]
            for knob in _FP32_KNOBS:
                knob.fp32_precision = 'ieee'
        else:
            self._prev = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision('highest')
        return self

    def __exit__(self, *exc):
        if _FP32_KNOBS is not None:
            for knob, prev in zip(_FP32_KNOBS, self._prev):
                knob.fp32_precision = prev
        else:
            torch.set_float32_matmul_precision(self._prev)
        return False


def _full_precision(fn):
    """Pin float32 matmuls to full precision for the duration of a call:
    frameworks that set torch's matmul precision ('high'/'medium' via the
    legacy API, TF32/bf16 via the per-backend fp32_precision settings of
    torch >= 2.9) would silently degrade the GEMM-formulated EM. The pin
    uses the per-backend settings when torch has them (generic,
    cuda.matmul, mkldnn.matmul — the legacy getter raises once any of
    those has been set in the process) and the legacy get/set pair
    otherwise.

    Applied to estimate/estimate_categorical and, so that direct kernel
    calls are covered too, at the cancellation-critical kernel boundaries
    em_fit and val_ll (the pin is reentrant: nesting saves and restores
    the ambient values). These settings are PROCESS-GLOBAL, not
    thread-local: concurrent callers in other threads of the same process
    will also run these calls' matmuls at full precision, and the
    save/restore can race with theirs. Use processes, not threads, for
    parallel estimates (see examples/mi_grid_multigpu.py)."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _pinned_precision():
            return fn(*args, **kwargs)
    return wrapped


def _entry_precision(fn):
    """_full_precision plus torch.no_grad() for the public estimators: they
    return plain floats, so an input tensor carrying requires_grad (model
    latents taken without .detach()) would otherwise build and retain the
    EM autograd graph of every iteration — several times the peak memory —
    and crash FittedGMM's .numpy() calls under return_fit=True. em_fit and
    val_ll keep plain _full_precision: differentiating a log-likelihood
    with respect to the parameters is legitimate."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _pinned_precision(), torch.no_grad():
            return fn(*args, **kwargs)
    return wrapped

LOG2PI = math.log(2.0 * math.pi)

# torch.multinomial caps the number of categories at 2**24; above this the
# uniform bootstrap resample falls back to a randint histogram (identically
# distributed for uniform probabilities).  Hoisted to a module constant so the
# fallback branch can be exercised at small N by monkeypatching it down.
_MULTINOMIAL_MAX_CATEGORIES = 1 << 24

# Auto-sizing of the replicate chunks (em_chunk / mc_chunk = 'auto').  Peak
# memory of one EM iteration is replicates x N_eff x K x itemsize x (4d+3),
# and of one MC chunk replicates x mc_samples x K x itemsize x (3d+1); both
# measured on GB10 to within 2% over d=1..8, K=5..20, fp32/fp64
# (validation/bench_mem.py).  The models below carry one extra unit of
# slack; the EM model describes the legacy kernels, and the GEMM kernels
# use the smaller model in _em_bytes_per_replicate.  The CUDA floor of 32
# keeps small problems reasonably batched; 'auto' normally sizes far above
# it.  CPU keeps the historical default.
_AUTO_CHUNK_MEM_FRACTION = 0.5      # of the free device memory at call time
_AUTO_CHUNK_MIN = 32
_AUTO_CHUNK_CPU_DEFAULT = 8


def _auto_chunk(n_items, bytes_per_item, device, default=_AUTO_CHUNK_CPU_DEFAULT):
    """Chunk size for `n_items` replicates costing `bytes_per_item` each.

    `default` on CPU; on CUDA, as many items as fit in a fraction of the free
    device memory, floored at _AUTO_CHUNK_MIN and capped at n_items.
    """
    dev = torch.device(device)
    if dev.type != 'cuda':
        return max(1, min(int(n_items), int(default)))
    free, _ = torch.cuda.mem_get_info(dev)
    n = int(free * _AUTO_CHUNK_MEM_FRACTION // max(int(bytes_per_item), 1))
    return max(1, min(int(n_items), max(_AUTO_CHUNK_MIN, n)))


def _resolve_chunk(value, n_items, bytes_per_item, device):
    """Turn an `em_chunk` / `mc_chunk` argument ('auto' or int) into an int."""
    if isinstance(value, str):
        if value != 'auto':
            raise ValueError(f"chunk must be a positive int or 'auto'; found '{value}'")
        return _auto_chunk(n_items, bytes_per_item, device)
    value = int(value)
    if value < 1:
        raise ValueError(f"chunk must be a positive int or 'auto'; found {value}")
    return value


def _index_arg(value, name, minimum=None):
    """Coerce an integer argument at entry: operator.index accepts numpy
    integers (the natural output of numpy RNGs in seed sweeps) and rejects
    floats, so a malformed count fails here under the argument's own name
    instead of deep inside EM with an opaque torch error after the compute
    has already been spent."""
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer; found {value!r} of "
                        f"type {type(value).__name__}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}; found {value}")
    return value


def _validate_entry_args(seed, n_bootstrap, mc_samples, max_components,
                         n_folds, n_inits, max_iter,
                         fixed_components_number, reg_covar,
                         threshold_components):
    """Entry validation shared by estimate and estimate_categorical, so the
    two cannot drift apart. Returns the integer arguments coerced to plain
    Python ints (torch.Generator.manual_seed and torch factory sizes reject
    numpy integers and floats)."""
    seed = _index_arg(seed, 'seed')
    n_bootstrap = _index_arg(n_bootstrap, 'n_bootstrap', 0)
    mc_samples = _index_arg(mc_samples, 'mc_samples', 1)
    max_components = _index_arg(max_components, 'max_components', 1)
    n_folds = _index_arg(n_folds, 'n_folds')   # >= 2 checked by _validate_folds
    n_inits = _index_arg(n_inits, 'n_inits', 1)
    max_iter = _index_arg(max_iter, 'max_iter', 1)
    if fixed_components_number is not None:
        fixed_components_number = _index_arg(
            fixed_components_number, 'fixed_components_number', 1)
    if not reg_covar > 0:
        raise ValueError(f"reg_covar must be > 0 (the covariance floor; 0 "
                         f"lets collapsed components through to garbage "
                         f"fits or factorisation failures); "
                         f"found {reg_covar}")
    if not threshold_components >= 0:
        raise ValueError(f"threshold_components must be >= 0; found "
                         f"{threshold_components}")
    return (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
            max_iter, fixed_components_number)


def _em_bytes_per_replicate(N, K, d, dtype, n_chunk=None):
    """Peak memory per EM replicate, from the measured model above."""
    n_eff = N if n_chunk is None else min(N, int(n_chunk))
    if _EM_KERNEL == 'gemm':
        # one live (R,N,K) buffer plus GEMM workspace and slack
        return n_eff * K * dtype.itemsize * 3
    return n_eff * K * dtype.itemsize * (4 * d + 4)


def _mc_bytes_per_replicate(mc_samples, K, d, dtype):
    """Peak memory per MC-integration replicate, from the model above."""
    return mc_samples * K * dtype.itemsize * (3 * d + 2)


def _run_chunked(n, chunk, fn):
    """Apply fn(slice) over [0, n) in chunks of `chunk`, halving the chunk
    and retrying on a CUDA out-of-memory error.  Results are chunk-invariant,
    so the retry changes only peak memory."""
    outs, i = [], 0
    chunk = max(1, min(int(chunk), int(n)))
    while i < n:
        try:
            outs.append(fn(slice(i, i + chunk)))
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()
            warnings.warn(f"CUDA out of memory; retrying with chunk={chunk}",
                          RuntimeWarning, stacklevel=3)
            continue
        i += chunk
    return outs


# ----------------------------------------------------------------------
# batched Gaussian machinery
# ----------------------------------------------------------------------

def _chol(cov):
    """Cholesky of (R,K,d,d) covariances; returns (chol, cov).

    The returned covariance is the matrix that was actually factorised, and
    every caller uses it in place of its input, so cov == chol @ chol^T is
    an invariant: marginals and refactorisations downstream then describe
    the same mixture the factor does.  Dead components (zero cov) are padded
    to I.  A non-finite component is reset to I (with a RuntimeWarning).  A
    component whose factorisation fails — round-off can make a
    near-degenerate covariance indefinite, especially at float32 — gets
    escalating diagonal regularisation scaled to its own largest absolute
    diagonal entry until the factorisation succeeds (with a RuntimeWarning
    reporting the cumulative bump): the per-component analogue of GMM-MI's
    refit-at-larger-reg_covar fallback, without abandoning the whole fit.
    """
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    dead = cov.abs().sum(dim=(-2, -1)) < 1e-30            # (R,K)
    broken = ~torch.isfinite(cov).all(dim=(-2, -1))       # (R,K)
    safe = torch.where((dead | broken)[..., None, None], eye.expand_as(cov),
                       cov)
    if bool(broken.any()):
        warnings.warn(f"{int(broken.sum())} non-finite covariance(s) "
                      f"reset to identity", RuntimeWarning, stacklevel=2)
    L, info = torch.linalg.cholesky_ex(safe)
    if not bool(info.any()):
        return L, safe
    # positive by construction: a corrupted component can have a negative or
    # near-zero mean diagonal, which must not flip the bump's sign
    scale = torch.diagonal(safe, dim1=-2, dim2=-1).abs().amax(-1)
    scale = scale.clamp_min(torch.finfo(cov.dtype).tiny)      # (R,K)
    bump, total = 1e-6, 0.0
    repaired = info > 0
    for _ in range(10):
        bad = info > 0
        repaired = repaired | bad
        safe = torch.where(bad[..., None, None],
                           safe + (bump * scale)[..., None, None] * eye, safe)
        total += bump
        L, info = torch.linalg.cholesky_ex(safe)
        if not bool(info.any()):
            warnings.warn(
                f"{int(repaired.sum())} covariance factorisation(s) needed "
                f"extra diagonal regularisation (up to {total:g} x the "
                f"component's largest absolute diagonal entry)",
                RuntimeWarning, stacklevel=2)
            return L, safe
        bump *= 10.0
    raise RuntimeError(
        f"covariance factorisation failed even after diagonal "
        f"regularisation up to {total:g} x the component's largest "
        f"absolute diagonal entry")


def log_gauss(X, mu, chol):
    """Component-wise Gaussian log-densities.

    X: (R,N,d), mu: (R,K,d), chol: (R,K,d,d) lower.  Returns (R,N,K).
    """
    diff = X.unsqueeze(2) - mu.unsqueeze(1)               # (R,N,K,d)
    eye = torch.eye(chol.shape[-1], dtype=chol.dtype, device=chol.device)
    Linv = torch.linalg.solve_triangular(chol, eye.expand_as(chol),
                                         upper=False)     # (R,K,d,d)
    z = torch.einsum('rkij,rnkj->rnki', Linv, diff)       # (R,N,K,d)
    maha = (z ** 2).sum(-1)                               # (R,N,K)
    logdet = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)  # (R,K)
    return -0.5 * (maha + X.shape[-1] * LOG2PI) - logdet.unsqueeze(1)


def gmm_score(X, w, mu, chol):
    """Mixture log-density of X under each replicate.

    Parameters
    ----------
    X : torch.Tensor of shape (R, N, d)
    w, mu, chol : torch.Tensor
        Mixture weights (R, K), means (R, K, d) and covariance Cholesky
        factors (R, K, d, d); zero-weight (dead) components are skipped.

    Returns
    -------
    torch.Tensor of shape (R, N)
    """
    logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                       torch.full_like(w, -torch.inf))
    return torch.logsumexp(logw.unsqueeze(1) + log_gauss(X, mu, chol), dim=-1)


def gmm_sample(w, mu, chol, n_samples, generator=None):
    """Draw samples from each replicate's mixture.

    Parameters
    ----------
    w, mu, chol : torch.Tensor
        Mixture parameters as in `gmm_score`.
    n_samples : int
        Samples per replicate.
    generator : torch.Generator, optional
        Device-matched RNG.

    Returns
    -------
    torch.Tensor of shape (R, n_samples, d)
    """
    R, K, d = mu.shape
    idx = torch.multinomial(w.clamp_min(0), n_samples, replacement=True,
                            generator=generator)          # (R,S)
    mu_s = torch.gather(mu, 1, idx.unsqueeze(-1).expand(R, n_samples, d))
    chol_s = torch.gather(
        chol, 1, idx[..., None, None].expand(R, n_samples, d, d))
    eps = torch.randn(R, n_samples, d, dtype=mu.dtype, device=mu.device,
                      generator=generator)
    return mu_s + (chol_s @ eps.unsqueeze(-1)).squeeze(-1)


def marginal_params(mu, cov, idx):
    """Analytic marginal of a GMM onto a variable subset.

    A GMM's marginals are GMMs: component weights are unchanged and the
    means/covariances are index-masked — this is what keeps every marginal
    exactly consistent with the joint fit.

    Parameters
    ----------
    mu, cov : torch.Tensor
        Joint means (R, K, d) and covariances (R, K, d, d).
    idx : sequence of int
        Variable indices to keep.

    Returns
    -------
    mu_m, cov_m : torch.Tensor
        Marginal means and covariances over the subset.
    """
    idx = torch.as_tensor(idx, device=mu.device)
    mu_m = mu[..., idx]
    cov_m = cov[..., idx, :][..., :, idx]
    return mu_m, cov_m


def marginal_score(X_full, w, mu, cov, idx):
    """Log-density of the selected columns under the analytic marginal GMM.

    Parameters
    ----------
    X_full : torch.Tensor of shape (R, N, d)
        Points in the full space; only columns `idx` are used.
    w, mu, cov : torch.Tensor
        Joint mixture parameters.
    idx : sequence of int
        Variable indices defining the marginal.

    Returns
    -------
    torch.Tensor of shape (R, N)
    """
    mu_m, cov_m = marginal_params(mu, cov, idx)
    chol_m, _ = _chol(cov_m)
    return gmm_score(X_full[..., list(idx)], w, mu_m, chol_m)


# ----------------------------------------------------------------------
# GEMM-formulated EM kernels
# ----------------------------------------------------------------------
# log w_k N(x|mu_k, Sigma_k) is a quadratic form in x, so against per-sample
# features F(x) = [x, x_i x_j (i<=j)] it is one matrix product with
# per-component coefficients, and the M-step moments are the transposed
# product resp^T F (Machlica, Vanek & Zajic 2011). The only tensor of shape
# (n_replicates, n_samples, n_components) is the responsibilities.
# The expanded quadratic form cancels almost completely for points inside a
# component, so the coefficients are assembled at float64 and the moments
# accumulated at float64 (see _gauss_coef and _moment_gemm); the data are
# standardised in estimate(), which bounds the magnitudes entering the
# cancellation (Chan, Golub & LeVeque 1983).


def _features(X):
    """Quadratic features [x, x_i x_j (i<=j)] of X (..., d): (..., d + d(d+1)/2)."""
    d = X.shape[-1]
    iu = torch.triu_indices(d, d, device=X.device)
    return torch.cat([X, X[..., iu[0]] * X[..., iu[1]]], -1)


def _tri_inv(L):
    """Batched inverse of lower-triangular (..., d, d) by forward
    substitution in plain tensor ops. The library routines
    (cholesky_inverse, solve_triangular) are slow on some GPUs when batched
    over many small matrices; d^2 vectorised element operations are cheap
    at the small d used here."""
    d = L.shape[-1]
    inv_diag = 1.0 / torch.diagonal(L, dim1=-2, dim2=-1)   # (...,d)
    Y = torch.zeros_like(L)
    for j in range(d):
        Y[..., j, j] = inv_diag[..., j]
        for i in range(j + 1, d):
            Y[..., i, j] = -(L[..., i, j:i] * Y[..., j:i, j]).sum(-1)                 * inv_diag[..., i]
    return Y


def _gauss_coef(w, mu, cov):
    """GEMM coefficients of the weighted log-density quadratic form.

    Returns coef (R,K,q), const (R,K) with
    log w_k N(x|mu_k, cov_k) = _features(x) . coef_k + const_k.
    The factorisation and inversion run at the native dtype; the quadratic
    form is then assembled at float64. The assembly is where precision
    matters: the expansion's terms cancel almost completely, so low-precision
    arithmetic there corrupts the log-density itself, whereas a small error
    in the precision matrix only evaluates the density of a slightly
    perturbed Gaussian.
    """
    dt = cov.dtype
    d = cov.shape[-1]
    L, _ = _chol(cov)
    Y = _tri_inv(L)
    P = (Y.transpose(-2, -1) @ Y).double()                 # (R,K,d,d)
    logdet = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1).double()
    mu64 = mu.double()
    Pm = (P @ mu64.unsqueeze(-1)).squeeze(-1)              # (R,K,d)
    c = (mu64 * Pm).sum(-1)                                # (R,K)
    iu = torch.triu_indices(d, d, device=cov.device)
    off = (iu[0] != iu[1]).to(P.dtype)
    Pq = P[..., iu[0], iu[1]] * (1.0 + off)                # doubled off-diagonals
    coef = torch.cat([Pm, -0.5 * Pq], -1)                  # (R,K,q)
    logw = torch.where(w > 0, torch.log(w.double().clamp_min(1e-300)),
                       torch.full_like(c, -torch.inf))
    const = logw - 0.5 * (c + d * LOG2PI) - logdet
    return coef.to(dt), const.to(dt)


def _estep_gemm(F, s, w, mu, cov):
    """Responsibilities (Ra,N,K) and weighted mean log-likelihood (Ra,)."""
    coef, const = _gauss_coef(w, mu, cov)
    lj = torch.matmul(F.unsqueeze(0), coef.transpose(1, 2))
    lj += const.unsqueeze(1)                               # (Ra,N,K)
    m = lj.amax(-1, keepdim=True)
    lj.sub_(m).exp_()
    Z = lj.sum(-1, keepdim=True)
    ll_n = (m + Z.log()).squeeze(-1)
    resp = lj.div_(Z)
    ll = (s * ll_n).sum(1) / s.sum(1)
    return resp, ll


def _mstep_from_moments(S, Nk, ssum, reg_covar, d, dt):
    """Finalise the M-step from float64 moments S=(resp*s)^T F, Nk, s.sum(1)."""
    dev = S.device
    dead = Nk < 1e-10
    Nks = Nk.clamp_min(1e-10).unsqueeze(-1)
    m1 = S[..., :d] / Nks                                  # (R,K,d)
    M2 = S[..., d:] / Nks
    iu = torch.triu_indices(d, d, device=dev)
    cov = torch.zeros(*S.shape[:-1], d, d, dtype=S.dtype, device=dev)
    cov[..., iu[0], iu[1]] = M2
    cov[..., iu[1], iu[0]] = M2
    cov = cov - m1.unsqueeze(-1) * m1.unsqueeze(-2)
    eye = torch.eye(d, dtype=S.dtype, device=dev)
    cov = cov + reg_covar * eye
    # the accumulated M2 - m1^2 can round below the reg_covar floor for a
    # component collapsed onto near-duplicate points; restore the floor
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    diag.clamp_(min=reg_covar)
    w = torch.where(dead, torch.zeros_like(Nk), Nk / ssum.double().unsqueeze(-1))
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    m1 = torch.where(dead[..., None], torch.zeros_like(m1), m1)
    cov = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
    return w.to(dt), m1.to(dt), cov.to(dt)


_MSTEP_BLOCK = 512


def _moment_gemm(wr, F):
    """S = wr^T F accumulated at float64: one GEMM at float64 input, else
    block-wise float32 GEMMs reduced in float64 (accumulation error scales
    with sqrt(block) instead of sqrt(N), at plain-SGEMM cost)."""
    R, N, K = wr.shape
    q = F.shape[-1]
    if F.dtype == torch.float64:
        return torch.matmul(wr.transpose(1, 2), F.unsqueeze(0))
    nb = N // _MSTEP_BLOCK
    S = torch.zeros(R, K, q, dtype=torch.float64, device=F.device)
    if nb > 0:
        N0 = nb * _MSTEP_BLOCK
        wrb = (wr[:, :N0].transpose(1, 2).reshape(R, K, nb, _MSTEP_BLOCK)
               .permute(0, 2, 1, 3))                       # (R,nb,K,b)
        Fb = F[:N0].reshape(nb, _MSTEP_BLOCK, q)
        S += torch.matmul(wrb, Fb.unsqueeze(0)).double().sum(1)
    if N % _MSTEP_BLOCK:
        N0 = nb * _MSTEP_BLOCK
        S += torch.matmul(wr[:, N0:].transpose(1, 2), F[N0:].unsqueeze(0)).double()
    return S


def _mstep_gemm(F, resp, s, reg_covar, d):
    """M-step via resp^T F; consumes resp in place. float64 accumulation."""
    wr = resp.mul_(s.unsqueeze(-1))
    S = _moment_gemm(wr, F)
    return _mstep_from_moments(S, wr.sum(1).double(), s.sum(1), reg_covar, d,
                               F.dtype)


def _em_step_gemm_nchunked(X, s, w, mu, cov, reg_covar, n_chunk):
    """One GEMM EM iteration streamed over sample chunks (float64 moments)."""
    Ra, N = s.shape
    d = X.shape[-1]
    K = w.shape[-1]
    dt, dev = X.dtype, X.device
    coef, const = _gauss_coef(w, mu, cov)
    q = coef.shape[-1]
    S = torch.zeros(Ra, K, q, dtype=torch.float64, device=dev)
    Nk = torch.zeros(Ra, K, dtype=torch.float64, device=dev)
    ll_sum = torch.zeros(Ra, dtype=torch.float64, device=dev)
    for i in range(0, N, n_chunk):
        Fc = _features(X[i:i+n_chunk])
        sc = s[:, i:i+n_chunk]
        lj = torch.matmul(Fc.unsqueeze(0), coef.transpose(1, 2))
        lj += const.unsqueeze(1)
        m = lj.amax(-1, keepdim=True)
        lj.sub_(m).exp_()
        Z = lj.sum(-1, keepdim=True)
        ll_n = (m + Z.log()).squeeze(-1)
        wr = lj.div_(Z).mul_(sc.unsqueeze(-1))
        Nk += wr.sum(1).double()
        S += _moment_gemm(wr, Fc)
        ll_sum += (sc * ll_n).sum(1).double()
    ll = (ll_sum / s.sum(1).double()).to(dt)
    w_new, mu_new, cov_new = _mstep_from_moments(S, Nk, s.sum(1), reg_covar, d, dt)
    return ll, w_new, mu_new, cov_new


# ----------------------------------------------------------------------
# weighted batched EM
# ----------------------------------------------------------------------

def _init_params(X, K, R, reg_covar, generator=None, active_k=None,
                 init_type='random_responsibilities', scale=None, em_chunk=8):
    """Batched GMM parameter initialisation (torch ports of GMM-MI's inits).

    init_type:
        'random_responsibilities' : GMM-MI's 'random_sklearn' (default);
        'minmax'            : means uniform over the data bounding box,
                              isotropic covs with GMM-MI's n-ball scale rule;
        'kmeans'            : batched Lloyd on a subsample, hard-assignment
                              stats (covariances normalised by N, as in
                              GMM-MI's KmeansInit);
        'randomized_kmeans' : 'kmeans' plus a mean push scale*N(0, cov_k)
                              (scale defaults to 1, as in GMM-MI).
    active_k: optional (R,) tensor of active component counts per replicate
    (for the batched-over-k protocol); components >= active_k[r] are dead.
    The initial M-step / assignment stats run in replicate chunks of
    `em_chunk`, under the same halve-and-retry CUDA out-of-memory guard as
    `em_fit` (results are chunk-invariant).
    """
    N, d = X.shape
    dev, dt = X.device, X.dtype
    ak = (active_k if active_k is not None
          else torch.full((R,), K, device=dev, dtype=torch.long))
    alive = torch.arange(K, device=dev).view(1, K) < ak.view(R, 1)    # (R,K)
    eye = torch.eye(d, dtype=dt, device=dev)

    if init_type == 'random_responsibilities':
        resp = torch.rand(R, N, K, dtype=dt, device=dev, generator=generator)
        if active_k is not None:
            comp = torch.arange(K, device=dev).view(1, 1, K)
            resp = resp * (comp < active_k.view(R, 1, 1))
        resp = resp / resp.sum(-1, keepdim=True).clamp_min(1e-30)

        def _init_mstep(sl):
            r = resp[sl]
            c = r.shape[0]
            return _m_step(X.expand(c, N, d), r,
                           torch.ones(c, N, dtype=dt, device=dev), reg_covar)
        outs = _run_chunked(R, em_chunk, _init_mstep)
        return tuple(torch.cat([o[j] for o in outs]) for j in range(3))

    if init_type == 'minmax':
        lo, hi = X.min(0).values, X.max(0).values
        akf = ak.to(dt)
        if scale is None:
            # n-ball scale rule: components jointly fill the data volume
            # (log-sum instead of prod(): prod is a JIT kernel on CUDA and
            # nvrtc rejects the GB10 arch)
            vol = torch.exp(torch.log(hi - lo).sum())
            s = ((vol / akf * math.gamma(d * 0.5 + 1)) ** (1.0 / d)
                 / math.sqrt(math.pi))                                # (R,)
        else:
            s = torch.full((R,), float(scale), dtype=dt, device=dev)
        mu = lo + (hi - lo) * torch.rand(R, K, d, dtype=dt, device=dev,
                                         generator=generator)
        w = alive.to(dt) / akf.view(R, 1)
        cov = ((s.view(R, 1, 1, 1) ** 2) * eye).expand(R, K, d, d) + reg_covar * eye
        dead = ~alive
        mu = torch.where(dead[..., None], torch.zeros_like(mu), mu)
        cov = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
        return w, mu, cov

    if init_type in ('kmeans', 'randomized_kmeans'):
        M = min(N, 4096)
        sub = torch.randperm(N, device=dev, generator=generator)[:M]
        Xs = X[sub]
        cidx = torch.randint(0, M, (R, K), device=dev, generator=generator)
        C = Xs[cidx]                                                  # (R,K,d)
        big = torch.finfo(dt).max
        xs2 = (Xs ** 2).sum(-1)                                       # (M,)
        for _ in range(25):
            d2 = (xs2.view(1, M, 1) + (C ** 2).sum(-1).unsqueeze(1)
                  - 2.0 * torch.einsum('md,rkd->rmk', Xs, C))         # (R,M,K)
            d2 = torch.where(alive.unsqueeze(1), d2, torch.full_like(d2, big))
            lab = d2.argmin(-1)
            oh = torch.zeros(R, M, K, dtype=dt, device=dev).scatter_(
                2, lab.unsqueeze(-1), 1.0)
            cnt = oh.sum(1)                                           # (R,K)
            Cn = torch.einsum('rmk,md->rkd', oh, Xs) \
                / cnt.clamp_min(1.0)[..., None]
            C = torch.where((cnt > 0)[..., None], Cn, C)
        # hard-assignment statistics on the full dataset, chunked over R
        x2 = (X ** 2).sum(-1)                                         # (N,)

        def _assign_stats(sl):
            Cc, al = C[sl], alive[sl]
            c = Cc.shape[0]
            d2 = (x2.view(1, N, 1) + (Cc ** 2).sum(-1).unsqueeze(1)
                  - 2.0 * torch.einsum('nd,rkd->rnk', X, Cc))
            d2 = torch.where(al.unsqueeze(1), d2, torch.full_like(d2, big))
            lab = d2.argmin(-1)
            oh = torch.zeros(c, N, K, dtype=dt, device=dev).scatter_(
                2, lab.unsqueeze(-1), 1.0)
            Nk = oh.sum(1)
            w = Nk / N
            mu = torch.einsum('rnk,nd->rkd', oh, X) \
                / Nk.clamp_min(1.0)[..., None]
            diff = X.view(1, N, 1, d) - mu.unsqueeze(1)
            # GMM-MI normalises the k-means covariances by N, not by Nk
            cov = torch.einsum('rnk,rnki,rnkj->rkij', oh, diff, diff) / N \
                + reg_covar * eye
            dead = (Nk < 1.0) | ~al
            w = torch.where(dead, torch.zeros_like(w), w)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
            mu = torch.where(dead[..., None], torch.zeros_like(mu), mu)
            cov = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
            return w, mu, cov
        outs = _run_chunked(R, em_chunk, _assign_stats)
        w, mu, cov = (torch.cat([o[j] for o in outs]) for j in range(3))
        if init_type == 'randomized_kmeans':
            s = 1.0 if scale is None else float(scale)
            eps = torch.randn(R, K, d, dtype=dt, device=dev,
                              generator=generator)
            L, cov = _chol(cov)
            push = (L @ eps.unsqueeze(-1)).squeeze(-1)
            mu = torch.where((w > 0)[..., None], mu + s * push, mu)
        return w, mu, cov

    raise ValueError(f"init_type not known; should be one of "
                     f"'random_responsibilities', 'minmax', 'kmeans', "
                     f"'randomized_kmeans'; found '{init_type}'")


def _m_step(X, resp, s, reg_covar):
    """Weighted M-step.  X:(R,N,d), resp:(R,N,K), s:(R,N) sample weights."""
    wr = resp * s.unsqueeze(-1)                           # (R,N,K)
    Nk = wr.sum(1)                                        # (R,K)
    dead = Nk < 1e-10
    Nk_safe = Nk.clamp_min(1e-10)
    mu = torch.einsum('rnk,rnd->rkd', wr, X) / Nk_safe.unsqueeze(-1)
    diff = X.unsqueeze(2) - mu.unsqueeze(1)               # (R,N,K,d)
    cov = torch.einsum('rnk,rnki,rnkj->rkij', wr, diff, diff) \
        / Nk_safe[..., None, None]
    d = X.shape[-1]
    cov = cov + reg_covar * torch.eye(d, dtype=X.dtype, device=X.device)
    # dead components: zero weight, harmless params
    w = torch.where(dead, torch.zeros_like(Nk), Nk / s.sum(1, keepdim=True))
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    mu = torch.where(dead[..., None], torch.zeros_like(mu), mu)
    cov = torch.where(dead[..., None, None],
                      torch.eye(d, dtype=X.dtype, device=X.device).expand_as(cov), cov)
    return w, mu, cov


@_full_precision
def em_fit(X, s, w, mu, cov, reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, em_chunk=8,
           n_chunk=None):
    """Weighted EM over a leading replicate dimension, in replicate chunks.

    Replicates (CV fits, bootstrap realisations) are independent, so they are
    processed in chunks of `em_chunk` purely to bound peak memory; results
    are identical to a single batch.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        Shared data.
    s : torch.Tensor of shape (n_replicates, n_samples)
        Per-replicate non-negative sample weights: 0/1 masks for CV folds,
        multinomial counts for bootstrap realisations, ones for a plain fit.
    w, mu, cov : torch.Tensor
        Initial mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    reg_covar, threshold_fit, max_iter, n_chunk
        As in `estimate`.
    em_chunk : int, default=8
        Replicates per batch. A CUDA out-of-memory error halves it and
        retries (with a RuntimeWarning); results are chunk-invariant.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Fitted parameters, same shapes as the inputs.
    ll : torch.Tensor of shape (n_replicates,)
        Final weighted mean per-sample log-likelihood.
    """
    R = s.shape[0]
    outs = _run_chunked(R, em_chunk, lambda sl: _em_fit_core(
        X, s[sl], w[sl], mu[sl], cov[sl], reg_covar, threshold_fit, max_iter, n_chunk))
    if len(outs) == 1:
        return outs[0]
    return tuple(torch.cat([o[j] for o in outs]) for j in range(4))


def _em_step_nchunked(X, s, w, mu, chol, logw, reg_covar, n_chunk):
    """One EM iteration accumulated over sample chunks (streaming M-step).

    Accumulates N_k, sum resp*(x - mu_old) and sum resp*(x - mu_old)(..)^T
    per sample-chunk; shifting the moments by the incoming means is algebraically
    identical to accumulating resp*x / resp*xx^T but avoids the fp32
    cancellation of the origin-moment form when |mu|^2 >> var.  As in the
    GEMM path, the cross-chunk accumulators run at float64 and the finished
    diagonal is clamped back to the reg_covar floor, which float32 rounding
    of the shifted moments can otherwise undercut."""
    Ra, N = s.shape
    d = X.shape[-1]
    K = w.shape[-1]
    dt, dev = X.dtype, X.device
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    Nk = torch.zeros(Ra, K, dtype=torch.float64, device=dev)
    Sd = torch.zeros(Ra, K, d, dtype=torch.float64, device=dev)
    Sdd = torch.zeros(Ra, K, d, d, dtype=torch.float64, device=dev)
    ll_sum = torch.zeros(Ra, dtype=torch.float64, device=dev)
    for i in range(0, N, n_chunk):
        Xc = X[i:i+n_chunk]
        nc = Xc.shape[0]
        sc = s[:, i:i+n_chunk]
        logjoint = logw.unsqueeze(1) + log_gauss(Xc.expand(Ra, nc, d),
                                                 mu, chol)       # (Ra,nc,K)
        ll_n = torch.logsumexp(logjoint, dim=-1)
        resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
        wr = resp * sc.unsqueeze(-1)
        ll_sum += (sc * ll_n).sum(1).double()
        Nk += wr.sum(1).double()
        diff = Xc.view(1, nc, 1, d) - mu.unsqueeze(1)            # (Ra,nc,K,d)
        Sd += torch.einsum('rnk,rnkd->rkd', wr, diff).double()
        Sdd += torch.einsum('rnk,rnki,rnkj->rkij', wr, diff, diff).double()
    ll = (ll_sum / s.sum(1).double()).to(dt)
    dead = Nk < 1e-10
    Nk_safe = Nk.clamp_min(1e-10)
    delta = Sd / Nk_safe.unsqueeze(-1)                           # mu_new-mu_old
    mu_new = mu.double() + delta
    cov_new = Sdd / Nk_safe[..., None, None] \
        - delta.unsqueeze(-1) * delta.unsqueeze(-2) + reg_covar * eye
    # the shifted moments can round the variance below the reg_covar floor
    # for a component collapsed onto near-duplicate points; restore the floor
    diag = torch.diagonal(cov_new, dim1=-2, dim2=-1)
    diag.clamp_(min=reg_covar)
    w_new = torch.where(dead, torch.zeros_like(Nk),
                        Nk / s.sum(1, keepdim=True).double())
    w_new = w_new / w_new.sum(-1, keepdim=True).clamp_min(1e-30)
    mu_new = torch.where(dead[..., None], torch.zeros_like(mu_new), mu_new)
    cov_new = torch.where(dead[..., None, None], eye.expand_as(cov_new),
                          cov_new)
    return ll, w_new.to(dt), mu_new.to(dt), cov_new.to(dt)


def _em_fit_core(X, s, w, mu, cov, reg_covar=1e-6, threshold_fit=1e-5, max_iter=500,
                 n_chunk=None):
    """Weighted EM, batched over replicates.

    X: (N,d) shared data; s: (R,N) per-replicate sample weights.
    n_chunk: sample-dimension chunk size for the per-iteration E/M
    accumulation (None = original single-pass path, kept verbatim).
    Returns (w, mu, cov, train_ll) with train_ll the final weighted
    mean per-sample log-likelihood, shape (R,).
    """
    R, N = s.shape
    dev = X.device
    # compaction: only still-active replicates stay in the working batch
    ai = torch.arange(R, device=dev)                      # active indices
    w_out, mu_out, cov_out = w.clone(), mu.clone(), cov.clone()
    ll_out = torch.full((R,), -torch.inf, dtype=X.dtype, device=dev)
    prev = torch.full((R,), -torch.inf, dtype=X.dtype, device=dev)
    sa = s
    gemm = _EM_KERNEL == 'gemm'
    F = _features(X) if gemm and n_chunk is None else None
    for _ in range(max_iter):
        Ra = ai.shape[0]
        if gemm:
            if n_chunk is None:
                resp, ll = _estep_gemm(F, sa, w, mu, cov)
                w, mu, cov = _mstep_gemm(F, resp, sa, reg_covar, X.shape[-1])
            else:
                ll, w, mu, cov = _em_step_gemm_nchunked(X, sa, w, mu, cov,
                                                        reg_covar, n_chunk)
        elif n_chunk is None:
            chol, cov = _chol(cov)
            logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                               torch.full_like(w, -torch.inf))
            Xr = X.expand(Ra, N, X.shape[-1])
            logjoint = logw.unsqueeze(1) + log_gauss(Xr, mu, chol)  # (Ra,N,K)
            ll_n = torch.logsumexp(logjoint, dim=-1)                # (Ra,N)
            resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
            ll = (sa * ll_n).sum(1) / sa.sum(1)
            w, mu, cov = _m_step(Xr, resp, sa, reg_covar)
        else:
            chol, cov = _chol(cov)
            logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                               torch.full_like(w, -torch.inf))
            ll, w, mu, cov = _em_step_nchunked(X, sa, w, mu, chol,
                                               logw, reg_covar, n_chunk)
        done = (ll - prev).abs() < threshold_fit
        if done.any():
            di = ai[done]
            w_out[di], mu_out[di], cov_out[di] = w[done], mu[done], cov[done]
            ll_out[di] = ll[done]
            keep = ~done
            ai, sa, prev = ai[keep], sa[keep], ll[keep]
            w, mu, cov = w[keep], mu[keep], cov[keep]
            if ai.shape[0] == 0:
                return w_out, mu_out, cov_out, ll_out
        else:
            prev = ll
    # unconverged stragglers: write back current state
    w_out[ai], mu_out[ai], cov_out[ai], ll_out[ai] = w, mu, cov, prev
    return w_out, mu_out, cov_out, ll_out


@_full_precision
def val_ll(X, s_val, w, mu, cov, n_chunk=None, em_chunk=None):
    """Weighted mean per-sample log-likelihood of X under each replicate.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        Shared data.
    s_val : torch.Tensor of shape (n_replicates, n_samples)
        Non-negative weights selecting (and weighting) the scored samples.
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    n_chunk : int, optional
        Stream over samples in chunks of this size (as in `estimate`).
    em_chunk : int, optional
        Replicates scored per batch, under the same halve-and-retry CUDA
        out-of-memory guard as `em_fit`; results are chunk-invariant.
        None scores all replicates in one batch.

    Returns
    -------
    torch.Tensor of shape (n_replicates,)
        -inf for a replicate whose weights are all zero (an empty validation
        fold scores worse than any real fit, instead of a spurious 0).
    """
    R = s_val.shape[0]
    outs = _run_chunked(R, R if em_chunk is None else em_chunk,
                        lambda sl: _val_ll_core(X, s_val[sl], w[sl], mu[sl],
                                                cov[sl], n_chunk))
    return outs[0] if len(outs) == 1 else torch.cat(outs)


def _val_ll_core(X, s_val, w, mu, cov, n_chunk=None):
    """`val_ll` for one replicate batch."""
    R, N = s_val.shape
    if _EM_KERNEL == 'gemm':
        coef, const = _gauss_coef(w, mu, cov)
        acc = torch.zeros(R, dtype=X.dtype, device=X.device)
        for i in range(0, N, n_chunk or N):
            Fc = _features(X[i:i+(n_chunk or N)])
            lj = torch.matmul(Fc.unsqueeze(0), coef.transpose(1, 2))
            lj += const.unsqueeze(1)
            acc = acc + (s_val[:, i:i+(n_chunk or N)]
                         * torch.logsumexp(lj, -1)).sum(1)
    else:
        chol, _ = _chol(cov)
        if n_chunk is None:
            ll_n = gmm_score(X.expand(R, N, X.shape[-1]), w, mu, chol)
            acc = (s_val * ll_n).sum(1)
        else:
            acc = torch.zeros(R, dtype=X.dtype, device=X.device)
            for i in range(0, N, n_chunk):
                Xc = X[i:i+n_chunk]
                nc = Xc.shape[0]
                ll_n = gmm_score(Xc.expand(R, nc, X.shape[-1]), w, mu, chol)
                acc = acc + (s_val[:, i:i+n_chunk] * ll_n).sum(1)
    tot = s_val.sum(1)
    return torch.where(tot > 0, acc / tot.clamp_min(1e-30),
                       torch.full_like(acc, -torch.inf))


# ----------------------------------------------------------------------
# component selection: the two protocols
# ----------------------------------------------------------------------

# CV fold assignment uses the caller's seed plus this fixed offset, keeping
# the fold stream decoupled from the init streams (seed + K) and the
# bootstrap stream (seed + 10_000).
_CV_MASK_SEED_OFFSET = 30_000


def _validate_folds(N, n_folds, what='X'):
    """Reject CV settings that produce empty or degenerate folds up front:
    they otherwise surface as 0/0 NaN metrics and far-away index errors."""
    if int(n_folds) < 2:
        raise ValueError(f"n_folds must be >= 2 (each fit needs a held-out "
                         f"validation fold); found {n_folds}")
    if int(N) < int(n_folds):
        raise ValueError(f"n_folds={n_folds} exceeds the {N} samples "
                         f"in {what}")


def _cv_masks(N, n_folds, n_inits, dtype, seed=42):
    """0/1 train masks and val masks, shape (n_inits*n_folds, N)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    fold_id = torch.zeros(N, dtype=torch.long)
    for f in range(n_folds):
        fold_id[perm[f::n_folds]] = f
    val = torch.stack([(fold_id == f).to(dtype) for f in range(n_folds)])  # CPU; moved by caller
    train = 1.0 - val
    train = train.repeat(n_inits, 1)                      # (I*F, N)
    val = val.repeat(n_inits, 1)
    return train, val


def _cv_fit_for_k(X, K, n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed, em_chunk=8,
                  init_type='random_responsibilities', scale=None,
                  n_chunk=None):
    """CV fits for a single k, batched over inits x folds."""
    g = torch.Generator(device=X.device).manual_seed(seed + K)
    train, val = _cv_masks(X.shape[0], n_folds, n_inits, X.dtype,
                           seed=seed + _CV_MASK_SEED_OFFSET)
    train, val = train.to(X.device), val.to(X.device)
    R = train.shape[0]
    w0, mu0, cov0 = _init_params(X, K, R, reg_covar, generator=g,
                                 init_type=init_type, scale=scale, em_chunk=em_chunk)
    w, mu, cov, _ = em_fit(X, train, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                           em_chunk=em_chunk, n_chunk=n_chunk)
    v = val_ll(X, val, w, mu, cov, n_chunk=n_chunk,
               em_chunk=em_chunk)                         # (I*F,)
    v_if = v.view(n_inits, n_folds)
    metric = v_if.mean(1).max().item()                    # best init's mean fold LL
    best_i = int(v_if.mean(1).argmax())
    best_f = int(v_if[best_i].argmax())
    r = best_i * n_folds + best_f
    return metric, (w[r], mu[r], cov[r])


def select_sequential(X, max_components=20, n_folds=2, n_inits=3,
                      threshold_components=1e-5, patience=1, reg_covar=1e-6, threshold_fit=1e-5,
                      max_iter=500, seed=0, em_chunk=8,
                      init_type='random_responsibilities', scale=None,
                      n_chunk=None):
    """Select the number of components by GMM-MI's sequential protocol.

    Components are added one at a time; the search stops when the validation
    metric fails to improve by `threshold_components` for `patience`
    consecutive values of k (the counter resets whenever k improves), and
    the best-scoring k seen so far is selected — also when the search
    reaches `max_components` without stopping. Cheaper than the batched
    protocol when it stops early, but the stopping decision is made under
    initialisation noise: one poor fit at a plateau can gate off better
    models at higher k.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        The data, already on the target device/dtype.
    max_components : int, default=20
        Largest k tried if the early stop never triggers.
    threshold_components : float, default=1e-5
        Minimum metric improvement that counts as progress.
    patience : int, default=1
        Number of consecutive non-improving k values tolerated before
        stopping; the counter resets on every improvement, and the selected
        k is the best-scoring one.
    n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed, em_chunk,
    init_type, scale, n_chunk
        As in `estimate`.

    Returns
    -------
    k_best : int
        Selected number of components.
    params : tuple of torch.Tensor
        (weights, means, covariances) of the best CV fit at k_best.
    history : dict
        Per-k (metric, params) from the search.
    """
    _validate_folds(X.shape[0], n_folds)
    best_metric, best_k, pat, history = -torch.inf, None, 0, {}
    for K in range(1, max_components + 1):
        metric, params = _cv_fit_for_k(X, K, n_folds, n_inits, reg_covar, threshold_fit,
                                       max_iter, seed, em_chunk=em_chunk,
                                       init_type=init_type, scale=scale,
                                       n_chunk=n_chunk)
        history[K] = (metric, params)
        if metric - best_metric > threshold_components:
            best_metric, best_k, pat = metric, K, 0
        else:
            pat += 1
            if pat >= patience:
                break
    if best_k is None:
        raise RuntimeError(
            "sequential selection found no finite validation metric at any "
            "k; NaN metrics usually mean degenerate CV folds or data")
    return best_k, history[best_k][1], history


def select_batched(X, max_components=20, n_folds=2, n_inits=3, threshold_components=1e-5,
                   reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0, em_chunk=8,
                   init_type='random_responsibilities', scale=None,
                   metric_method='valid', n_chunk=None):
    """Select the number of components with all candidate k fitted at once.

    Every k = 1..max_components (times inits and folds) is fitted as a single
    batched EM via dead-component padding, so no k's evaluation depends on
    any other's — the protocol is immune to the initialisation-noise gating
    of the sequential search, and is the fast option on GPU.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        The data, already on the target device/dtype.
    max_components : int, default=20
        Largest k fitted; all are paid for unconditionally.
    threshold_components : float, default=1e-5
        Parsimony rule for metric_method='valid': the smallest k whose
        validation log-likelihood is within this threshold of the best wins,
        so noise-level improvements do not inflate k.
    metric_method : {'valid', 'aic', 'bic'}, default='valid'
        'valid' scores held-out log-likelihood with the parsimony rule.
        'aic'/'bic' follow GMM-MI: each k's best CV parameters are refit on
        the full data, the criterion is computed on it, and the minimising k
        wins; the returned parameters are still the CV-selected ones.
    n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed, em_chunk,
    init_type, scale, n_chunk
        As in `estimate`.

    Returns
    -------
    k_best : int
        Selected number of components.
    params : tuple of torch.Tensor
        (weights, means, covariances) of the best CV fit at k_best.
    history : dict
        Per-k metrics; params entries are None for every k (the selected
        parameters are returned separately).
    """
    N, d = X.shape
    _validate_folds(N, n_folds)
    train, val = _cv_masks(N, n_folds, n_inits, X.dtype,
                           seed=seed + _CV_MASK_SEED_OFFSET)
    train, val = train.to(X.device), val.to(X.device)
    IF = train.shape[0]
    R = max_components * IF
    train_r, val_r = train.repeat(max_components, 1), val.repeat(max_components, 1)
    active_k = torch.arange(1, max_components + 1, device=X.device).repeat_interleave(IF)
    g = torch.Generator(device=X.device).manual_seed(seed)
    w0, mu0, cov0 = _init_params(X, max_components, R, reg_covar, generator=g,
                                 active_k=active_k, init_type=init_type,
                                 scale=scale, em_chunk=em_chunk)
    w, mu, cov, _ = em_fit(X, train_r, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                           em_chunk=em_chunk, n_chunk=n_chunk)
    v = val_ll(X, val_r, w, mu, cov, n_chunk=n_chunk,
               em_chunk=em_chunk).view(max_components, n_inits, n_folds)
    metric_k = v.mean(-1).max(-1).values                  # (max_components,)
    if not torch.isfinite(metric_k).any():
        # mirror select_sequential: all-NaN/-inf metrics must not reach the
        # parsimony mask's nonzero()[0] (an opaque, far-away IndexError)
        raise RuntimeError(
            "batched selection found no finite validation metric at any "
            "k; NaN metrics usually mean degenerate CV folds or data")
    if metric_method == 'valid':
        # parsimony rule: smallest k within `threshold_components` of the best metric
        k_best = int((metric_k >= metric_k.max() - threshold_components)
                     .nonzero()[0].item()) + 1
    elif metric_method in ('aic', 'bic'):
        # best (init, fold) per k, refit on the full data from those params
        best_i_k = v.mean(-1).argmax(-1)                  # (max_components,)
        best_f_k = v[torch.arange(max_components), best_i_k].argmax(-1)
        rk = torch.arange(max_components, device=X.device) * IF \
            + best_i_k * n_folds + best_f_k
        ones = torch.ones(max_components, N, dtype=X.dtype, device=X.device)
        _, _, _, llf = em_fit(X, ones, w[rk], mu[rk], cov[rk], reg_covar, threshold_fit,
                              max_iter, em_chunk=em_chunk, n_chunk=n_chunk)
        ks = torch.arange(1, max_components + 1, dtype=X.dtype, device=X.device)
        n_par = (ks - 1) + ks * d + ks * d * (d + 1) / 2
        pen = 2.0 if metric_method == 'aic' else math.log(N)
        crit = -2.0 * llf * N + pen * n_par
        k_best = int(crit.argmin().item()) + 1
    else:
        raise ValueError(f"metric_method must be one of 'valid', 'aic', "
                         f"'bic'; found '{metric_method}'")
    v_k = v[k_best - 1]
    best_i = int(v_k.mean(-1).argmax())
    best_f = int(v_k[best_i].argmax())
    r = (k_best - 1) * IF + best_i * n_folds + best_f
    history = {k + 1: (metric_k[k].item(), None) for k in range(max_components)}
    return k_best, (w[r], mu[r], cov[r]), history


# ----------------------------------------------------------------------
# estimators (all from one joint fit, marginals by masking)
# ----------------------------------------------------------------------

def _integrand_MI(pts, w, mu, cov, chol):
    """log p(x,y) - log p(x) - log p(y) at the sampled points."""
    joint = gmm_score(pts, w, mu, chol)
    mx = marginal_score(pts, w, mu, cov, [0])
    my = marginal_score(pts, w, mu, cov, [1])
    return joint - mx - my


def _integrand_cMI(pts, w, mu, cov, chol):
    """log p(x,y,z) + log p(z) - log p(x,z) - log p(y,z)."""
    d = pts.shape[-1]
    zs = list(range(2, d))
    joint = gmm_score(pts, w, mu, chol)
    mz = marginal_score(pts, w, mu, cov, zs)
    mxz = marginal_score(pts, w, mu, cov, [0] + zs)
    myz = marginal_score(pts, w, mu, cov, [1] + zs)
    return joint + mz - mxz - myz


def _integrand_TC(pts, w, mu, cov, chol):
    """log p(x) - sum_i log p(x_i)."""
    joint = gmm_score(pts, w, mu, chol)
    marg = sum(marginal_score(pts, w, mu, cov, [i])
               for i in range(pts.shape[-1]))
    return joint - marg


def _integrand_KL_forward(pts, w, mu, cov, chol):
    """KL[p(x)||p(y)] between the two 1D marginals (GMM-MI estimate_KL_MC):
    E_{x~p(x)}[log p_x(x) - log p_y(x)], with x the first column of the
    joint samples."""
    x = pts[..., [0]]
    mu_x, cov_x = marginal_params(mu, cov, [0])
    mu_y, cov_y = marginal_params(mu, cov, [1])
    return gmm_score(x, w, mu_x, _chol(cov_x)[0]) \
        - gmm_score(x, w, mu_y, _chol(cov_y)[0])


def _integrand_KL_reverse(pts, w, mu, cov, chol):
    """KL[p(y)||p(x)]: E_{y~p(y)}[log p_y(y) - log p_x(y)]."""
    y = pts[..., [1]]
    mu_x, cov_x = marginal_params(mu, cov, [0])
    mu_y, cov_y = marginal_params(mu, cov, [1])
    return gmm_score(y, w, mu_y, _chol(cov_y)[0]) \
        - gmm_score(y, w, mu_x, _chol(cov_x)[0])


_INTEGRANDS = {'MI': _integrand_MI, 'cMI': _integrand_cMI, 'TC': _integrand_TC,
               'KL_forward': _integrand_KL_forward,
               'KL_reverse': _integrand_KL_reverse}


# Tie fraction (1 - n_unique/N per column) above which gaussianise warns:
# genuinely continuous data has essentially no ties, so a noticeable
# fraction means quantised or zero-inflated input, where the rank-probit
# transform cannot make the marginal Gaussian.
_GAUSSIANISE_TIE_WARN_FRACTION = 0.01


def _rank_gaussianise(X):
    """Per-variable rank-probit transform: X -> Phi^{-1}(rank / (N+1)).

    MI, cMI and TC are invariant under per-variable diffeomorphisms, so no
    correction term is needed.  Tied values share their average rank (the
    scipy `rankdata` convention), so a tied block maps to a single probit
    atom instead of a monotone ramp in row order — a positional tiebreak
    fabricates MI whenever row order correlates with another variable
    (sorted catalogues, concatenated bins, zero-inflated data).
    """
    N = X.shape[0]
    # ranks and quantiles at float64: float32 integer ranks round above
    # 2^24 samples, corrupting the quantiles exactly in the large-N regime
    cols, tie_frac = [], []
    for j in range(X.shape[1]):
        vals, inv, cnt = torch.unique(X[:, j], return_inverse=True,
                                      return_counts=True)
        # average rank of a tied block: (#smaller values) + (count+1)/2
        csum = torch.cumsum(cnt, 0)
        avg_rank = (csum - cnt).double() + (cnt.double() + 1.0) / 2.0
        cols.append(avg_rank[inv])
        tie_frac.append(1.0 - vals.shape[0] / N)
    tied = [j for j, f in enumerate(tie_frac)
            if f > _GAUSSIANISE_TIE_WARN_FRACTION]
    if tied:
        warnings.warn(
            f"gaussianise: column(s) {tied} have tied values (tie fraction "
            f"{', '.join(f'{tie_frac[j]:.2f}' for j in tied)}); tied blocks "
            f"map to one probit atom, so the transformed marginal is not "
            f"Gaussian and the GMM fit may be degraded on heavily "
            f"quantised data", RuntimeWarning, stacklevel=4)
        # 4: _rank_gaussianise -> estimate -> _entry_precision -> caller
    p = torch.stack(cols, 1) / (N + 1.0)
    # ndtri/erfinv are JIT (nvrtc) kernels on CUDA and nvrtc rejects the
    # GB10 arch; this is one-off data prep, so run the probit on CPU
    return torch.special.ndtri(p.cpu()).to(dtype=X.dtype, device=X.device)


def _select_and_fit(X, protocol, fixed_components_number, max_components,
                    n_folds, n_inits, reg_covar, threshold_fit, max_iter,
                    seed, em_chunk, init_type, scale, metric_method,
                    threshold_components, patience, n_chunk):
    """Component selection and best CV fit for one dataset.

    The single home of the selection pipeline shared by `estimate` and
    `estimate_categorical` (per-category), so the two cannot drift apart:
    chunk sizing, the fixed-k shortcut, protocol dispatch and the
    protocol/metric_method compatibility rules live only here.
    Returns (k, (w, mu, cov)) with dead components sliced away and k the
    LIVE component count of the returned parameters — the model that is
    actually bootstrapped, MC-integrated and returned — which can be
    smaller than the selected/requested k when components died during the
    CV fit (a RuntimeWarning says so for a requested
    fixed_components_number).
    """
    N, d = X.shape
    IF = n_inits * n_folds
    # compatibility rules before the fixed-k shortcut: a typo'd
    # metric_method must raise even when selection itself is skipped
    if protocol == 'sequential':
        if metric_method != 'valid':
            raise ValueError("metric_method is only supported by the "
                             "batched protocol")
    elif metric_method not in ('valid', 'aic', 'bic'):
        raise ValueError(f"metric_method must be one of 'valid', "
                         f"'aic', 'bic'; found '{metric_method}'")
    if fixed_components_number is not None:
        k_sel, r_sel = int(fixed_components_number), IF
    elif protocol == 'sequential':
        k_sel, r_sel = max_components, IF
    else:
        k_sel, r_sel = max_components, max_components * IF
    em_chunk_sel = _resolve_chunk(
        em_chunk, r_sel, _em_bytes_per_replicate(N, k_sel, d, X.dtype, n_chunk),
        X.device)
    if fixed_components_number is not None:
        k_best = int(fixed_components_number)
        _, params = _cv_fit_for_k(X, k_best, n_folds, n_inits, reg_covar,
                                  threshold_fit, max_iter, seed,
                                  em_chunk=em_chunk_sel, init_type=init_type,
                                  scale=scale, n_chunk=n_chunk)
    else:
        kw = dict(max_components=max_components, n_folds=n_folds,
                  n_inits=n_inits, reg_covar=reg_covar,
                  threshold_fit=threshold_fit, max_iter=max_iter, seed=seed,
                  threshold_components=threshold_components,
                  em_chunk=em_chunk_sel, init_type=init_type, scale=scale,
                  n_chunk=n_chunk)
        if protocol == 'sequential':
            sel = select_sequential
            kw['patience'] = patience
        else:
            sel = select_batched
            kw['metric_method'] = metric_method
        k_best, params, _ = sel(X, **kw)
    # the batched protocol pads parameters to K=max_components, and any
    # protocol can leave dead components in the best CV fit; a zero-weight
    # component can never resurrect under EM, so drop them here (the same
    # masking FittedGMM applies) instead of paying for them through the
    # bootstrap, refit and MC stages — and letting them shrink the
    # auto-sized chunks by the padding factor.
    w1, mu1, cov1 = params
    alive = w1 > 0
    k_live = int(alive.sum())
    if fixed_components_number is not None and k_live != k_best:
        warnings.warn(
            f"fixed_components_number={k_best}: {k_best - k_live} "
            f"component(s) died during the CV fit, so the model that is "
            f"bootstrapped and returned has {k_live}",
            RuntimeWarning, stacklevel=4)
    return k_live, (w1[alive], mu1[alive], cov1[alive])


def _bootstrap_counts(N, B, dtype, device, generator):
    """Uniform multinomial bootstrap resample counts, shape (B, N).

    Identical in distribution to index resampling. torch.multinomial caps
    its category count at 2^24; above that the counts are accumulated from
    randint draws instead (identically distributed for uniform
    probabilities; this size used to raise, so no existing result changed
    when the fallback was added). This helper is the fallback's only home.
    """
    if N <= _MULTINOMIAL_MAX_CATEGORIES:
        idx = torch.multinomial(
            torch.full((B, N), 1.0 / N, dtype=dtype, device=device),
            N, replacement=True, generator=generator)
    else:
        idx = torch.randint(0, N, (B, N), device=device, generator=generator)
    return torch.zeros(B, N, dtype=dtype, device=device).scatter_add_(
        1, idx, torch.ones(B, N, dtype=dtype, device=device))


def _warm_start(w1, mu1, cov1, B):
    """Expand one CV-selected parameter set to B bootstrap replicates."""
    K, d = mu1.shape
    return (w1.unsqueeze(0).expand(B, K).contiguous(),
            mu1.unsqueeze(0).expand(B, K, d).contiguous(),
            cov1.unsqueeze(0).expand(B, K, d, d).contiguous())


class FittedGMM:
    """The GMM selected by `estimate`, refitted on the full data.

    Mirrors GMM-MI's fitted-model access for plotting and inspection, with
    sklearn-style attributes: `weights_` (K,), `means_` (K, d) and
    `covariances_` (K, d, d) as numpy arrays on the CPU (dead padding
    components dropped), plus `n_components`. Obtained via
    `estimate(..., return_fit=True)`.
    """

    def __init__(self, w, mu, cov):
        # factorise once: if the back-transformed covariance needs repair,
        # covariances_, sample and score_samples all describe the same
        # (repaired) mixture, and the repair warning fires here, not on use
        chol, cov = _chol(cov.unsqueeze(0))
        self._w, self._mu, self._cov = w, mu, cov[0]
        self._chol = chol[0]
        self.weights_ = w.cpu().numpy()
        self.means_ = mu.cpu().numpy()
        self.covariances_ = self._cov.cpu().numpy()
        self.n_components = int(w.shape[0])

    def sample(self, n_samples, seed=0):
        """Draw `n_samples` from the mixture; returns a (n_samples, d) numpy array."""
        g = torch.Generator(device=self._w.device).manual_seed(seed)
        pts = gmm_sample(self._w.unsqueeze(0), self._mu.unsqueeze(0),
                         self._chol.unsqueeze(0), int(n_samples),
                         generator=g)
        return pts[0].cpu().numpy()

    def score_samples(self, X):
        """Mixture log-density at X of shape (n, d); returns a (n,) numpy array."""
        Xt = torch.as_tensor(X, dtype=self._mu.dtype, device=self._mu.device)
        ll = gmm_score(Xt.unsqueeze(0), self._w.unsqueeze(0),
                       self._mu.unsqueeze(0), self._chol.unsqueeze(0))
        return ll[0].cpu().numpy()


@_entry_precision
def estimate(X, quantity='MI', protocol='batched', n_bootstrap=50,
             mc_samples=100_000, max_components=20, n_folds=2, n_inits=3,
             reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0, return_k=False,
             return_fit=False,
             device='cpu', dtype=torch.float64, em_chunk='auto', mc_chunk='auto',
             preprocess='none', init_type='random_responsibilities',
             scale=None, fixed_components_number=None, metric_method='valid',
             threshold_components=1e-5, patience=1, n_chunk=None):
    """Estimate an information-theoretic quantity with bootstrap errorbars.

    One joint GMM is fitted to all columns of `X` (component count chosen by
    cross-validation); every marginal needed by the requested quantity is then
    obtained analytically from that single fit, so marginals are exactly
    consistent with the joint. The bootstrap is a single batched EM:
    multinomial-count sample weights over `n_bootstrap` replicates, each
    warm-started from the CV-selected parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_variables)
        The data. Two columns for 'MI' and the KL quantities; at least three
        for 'cMI' (variables 0 and 1 are X and Y, the rest condition); any
        number of columns for 'TC'.
    quantity : {'MI', 'cMI', 'TC', 'KL_forward', 'KL_reverse'}, default='MI'
        What to estimate. 'MI' is I(X;Y); 'cMI' is I(X;Y|Z) with Z all
        columns beyond the first two; 'TC' is the total correlation
        (KL between the joint and the product of all 1D marginals, zero iff
        the variables are mutually independent); 'KL_forward' is
        KL[p(x0) || p(x1)] between the two 1D marginals, 'KL_reverse' the
        opposite order.
    protocol : {'batched', 'sequential'}, default='batched'
        Component-selection protocol. 'batched' fits k = 1..max_components
        simultaneously and picks the smallest k within `threshold_components`
        of the best validation metric (robust to initialisation noise, and
        the fast option on GPU). 'sequential' is GMM-MI's original protocol:
        increment k and stop early via threshold and patience (cheaper on CPU
        when it stops early, but a single bad initialisation near a plateau
        can gate off better models at higher k).
    n_bootstrap : int, default=50
        Number of bootstrap realisations used for the errorbar. Higher values
        give a smoother MI distribution and more reliable errorbars, while
        being slower; on GPU all realisations are fitted as one batch.
        Set to 0 to skip the bootstrap (single fit on the full data); the
        returned standard deviation is None whenever n_bootstrap < 2, since
        one replicate has no sample deviation.
    mc_samples : int, default=100_000
        Number of Monte Carlo samples used to integrate the estimator for
        each bootstrap replicate. Higher values reduce the MC contribution to
        the error at linear cost in time.
    max_components : int, default=20
        Largest number of GMM components considered in selection. The batched
        protocol pays for all of them unconditionally, so keep this to a
        plausible ceiling for the data.
    n_folds : int, default=2
        Number of cross-validation folds. Each fold should remain large
        enough to be representative of the data.
    n_inits : int, default=3
        Number of GMM initialisations per fold. Higher values reduce the
        chance of a local optimum, at proportional cost (batched away on GPU).
    reg_covar : float, default=1e-6
        Constant added to the diagonal of every covariance matrix to prevent
        singularities. Smaller values perturb the estimate less but risk
        failed Cholesky factorisations; 1e-6 is safe at float32, where
        GMM-MI's 1e-15 would regularise nothing. The data are internally
        standardised (see Notes), so this floor is relative to each
        variable's variance; a component whose factorisation still fails
        receives escalating diagonal regularisation automatically (with a
        RuntimeWarning) instead of aborting the fit.
    threshold_fit : float, default=1e-5
        EM convergence threshold on the change in mean per-sample
        log-likelihood. Smaller values give more converged fits at the cost
        of more EM iterations; equivalent to `tol` in sklearn GMMs.
    max_iter : int, default=500
        Maximum EM iterations per fit. Convergence should normally be reached
        via `threshold_fit` well before this.
    seed : int, default=0
        Seed controlling initialisations, CV splits, bootstrap resampling and
        MC integration.
    return_k : bool, default=False
        If True, also return the selected number of components.
    return_fit : bool, default=False
        If True, also return the selected model as a `FittedGMM` (refitted on
        the full data, warm-started from the CV-selected parameters — the
        same warm start the bootstrap replicates use), e.g. for plotting.
        With preprocess='gaussianise' the returned model describes the
        rank-probit-transformed data, not the original units (the rank
        transform has no parametric inverse); a RuntimeWarning says so.
    device : str, default='cpu'
        Torch device on which to fit and integrate (e.g. 'cuda').
    dtype : torch.dtype, default=torch.float64
        Working precision. float32 is validated and much faster on GPUs with
        limited fp64 throughput; float64 is the reference precision.
    em_chunk : int or 'auto', default='auto'
        Number of EM replicates (CV fits or bootstrap realisations) processed
        per batch; the initialisation and validation-LL stages batch the
        same way. Larger values are faster on GPU at higher peak memory
        (results are chunk-invariant). 'auto' keeps 8 on CPU; on CUDA it
        sizes each stage's batch to fill half of the free device memory from
        the measured peak-memory model: replicates x N_eff x K x itemsize
        x 3 for the default GEMM kernels, x (4d+4) for the legacy kernels
        (N_eff = min(N, n_chunk)), floored at 32 and capped at the stage's
        replicate count. A CUDA out-of-memory error
        during EM halves the chunk and retries, with a RuntimeWarning.
    mc_chunk : int or 'auto', default='auto'
        Number of bootstrap replicates whose MC integration is evaluated per
        batch; 'auto' as for `em_chunk` (model mc_samples x K x (3d+2) x
        itemsize). The MC sample stream is drawn per chunk, so estimates
        differ across `mc_chunk` values at the Monte-Carlo noise level.
    preprocess : {'none', 'gaussianise'}, default='none'
        'gaussianise' applies a per-variable rank-probit transform before
        fitting. MI, cMI and TC are invariant under per-variable
        diffeomorphisms, so this changes only the difficulty of the density
        estimation: heavy tails and monotone nonlinearities that would
        require many components (or defeat the fit entirely) are removed.
        Tied values share their average rank, so quantised data maps tied
        blocks to probit atoms (a RuntimeWarning is raised above a 1% tie
        fraction: atoms are not Gaussian, and the invariance argument needs
        a continuous variable in the first place). Rejected (ValueError) for
        the KL quantities, which are not invariant under marginal
        transforms.
    init_type : {'random_responsibilities', 'minmax', 'kmeans', 'randomized_kmeans'}, default='random_responsibilities'
        GMM initialisation scheme (torch ports of GMM-MI's initialisations).
        'random_responsibilities' is cheap and adequate for easy fits; the
        k-means variants are more robust on well-separated multimodal data.
        At very large N the responsibility draw of the default binds on
        memory first; 'minmax' avoids it.
    scale : float, optional
        Mean-perturbation scale for 'randomized_kmeans' (GMM-MI's default 1
        when None).
    fixed_components_number : int, optional
        If set, skip component selection entirely and fit this many
        components (best CV init/fold retained, as GMM-MI's
        fixed_components). Use when k is already known.
    metric_method : {'valid', 'aic', 'bic'}, default='valid'
        Selection metric for the batched protocol. 'valid' is the validation
        log-likelihood with the parsimony-threshold rule; 'aic'/'bic' refit
        each k's best CV parameters on the full data and minimise the
        criterion, as in GMM-MI.
    threshold_components : float, default=1e-5
        Parsimony threshold for the batched protocol's 'valid' metric (and
        the improvement threshold for the sequential protocol): metric gains
        smaller than this do not justify a larger k.
    patience : int, default=1
        Sequential protocol only: consecutive non-improving k values
        tolerated before the search stops (as in `select_sequential`).
    n_chunk : int, optional
        If set, each EM iteration and validation-LL evaluation streams over
        the sample dimension in chunks of this size, accumulating sufficient
        statistics. Bounds the EM and validation-LL peak memory by
        em_chunk * n_chunk rather than N (CV masks and bootstrap weights
        still scale with N); results agree with the single-pass path to
        floating-point reduction order (ulp-level at float64, not bitwise).
        None keeps the single-pass path.

    Returns
    -------
    mean : float
        Mean of the estimated quantity over the bootstrap distribution
        (or the single-fit value when n_bootstrap=0), in nats.
    std : float or None
        Bootstrap standard deviation (None when n_bootstrap < 2:
        one replicate has no sample deviation).
    k : int
        Selected number of components; only if `return_k` is True.
    fit : FittedGMM
        The selected model refitted on the full data; only if `return_fit`
        is True (appended after `k` when both flags are set).

    Notes
    -----
    The data are standardised internally before fitting: each variable is
    affinely mapped to zero mean and unit variance (the KL quantities use a
    single map shared by both columns, under which KL is invariant). All
    estimated quantities are exactly invariant under these maps, and the
    `return_fit` model is transformed back (only through this affine map:
    with preprocess='gaussianise' the model stays in rank-probit space),
    so the standardisation is
    unobservable except through `reg_covar` (now a scale-free floor), the
    `scale` argument of the minmax/randomized k-means initialisers (now in
    standardised units), and the absence of float32 failures on
    unstandardised data.
    """
    X = torch.as_tensor(X, dtype=dtype, device=device).detach()
    if quantity not in _INTEGRANDS:
        raise ValueError(f"quantity must be one of "
                         f"{sorted(_INTEGRANDS)}; found '{quantity}'")
    if protocol not in ('batched', 'sequential'):
        raise ValueError(f"protocol must be 'batched' or 'sequential'; "
                         f"found '{protocol}'")
    (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
     max_iter, fixed_components_number) = _validate_entry_args(
        seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
        max_iter, fixed_components_number, reg_covar, threshold_components)
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_variables); "
                         f"found shape {tuple(X.shape)}")
    if quantity in ('MI', 'KL_forward', 'KL_reverse') and X.shape[1] != 2:
        raise ValueError(f"quantity='{quantity}' needs exactly 2 columns; "
                         f"found {X.shape[1]}")
    if quantity in ('cMI', 'TC') and X.shape[1] < (3 if quantity == 'cMI' else 2):
        raise ValueError(f"quantity='{quantity}' needs at least "
                         f"{3 if quantity == 'cMI' else 2} columns; "
                         f"found {X.shape[1]}")
    if not torch.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values (NaN or inf in the working "
            "dtype); gaussianise would silently rank them as extreme "
            "values and the standardisation path propagates them into "
            "every replicate — remove or impute them first")
    _validate_folds(X.shape[0], n_folds)
    if n_chunk is not None and int(n_chunk) < 1:
        raise ValueError(f"n_chunk must be a positive int or None; "
                         f"found {n_chunk}")
    if preprocess == 'gaussianise':
        if quantity in ('KL_forward', 'KL_reverse'):
            raise ValueError(
                "preprocess='gaussianise' is invalid for KL quantities: "
                "KL is not invariant under marginal transforms (the "
                "gaussianised marginals have ~zero KL by construction)")
        if return_fit:
            warnings.warn(
                "return_fit with preprocess='gaussianise': the returned "
                "model is a fit to the rank-probit-transformed data, not "
                "to the original units — the rank transform has no "
                "parametric inverse. Its sample/score_samples/contours "
                "live in gaussianised space.", RuntimeWarning,
                stacklevel=3)   # estimate -> _entry_precision -> caller
        X = _rank_gaussianise(X)
    elif preprocess != 'none':
        raise ValueError(f"preprocess must be 'none' or 'gaussianise'; "
                         f"found '{preprocess}'")
    # Internal per-variable affine standardisation.  MI, cMI and TC are
    # exactly invariant under per-variable affine maps, and the KL quantities
    # under a map shared by both columns, so no estimated quantity changes;
    # it makes `reg_covar` a scale-free floor and prevents float32 Cholesky
    # failures on unstandardised, strongly correlated pairs.  The fitted
    # model returned by `return_fit` is mapped back to the original units.
    if quantity in ('KL_forward', 'KL_reverse'):
        col_mean = X.mean().expand(X.shape[1]).contiguous()
        col_std = X.std().expand(X.shape[1]).contiguous()
    else:
        col_mean = X.mean(0)
        col_std = X.std(0)
    col_std = torch.where(col_std > 0, col_std, torch.ones_like(col_std))
    X = (X - col_mean) / col_std
    N, d = X.shape
    k_best, (w1, mu1, cov1) = _select_and_fit(
        X, protocol, fixed_components_number, max_components, n_folds,
        n_inits, reg_covar, threshold_fit, max_iter, seed, em_chunk,
        init_type, scale, metric_method, threshold_components, patience,
        n_chunk)

    B = max(n_bootstrap, 1)
    K = w1.shape[0]
    em_chunk_boot = _resolve_chunk(
        em_chunk, B, _em_bytes_per_replicate(N, K, d, dtype, n_chunk), device)
    mc_chunk = _resolve_chunk(
        mc_chunk, B, _mc_bytes_per_replicate(mc_samples, K, d, dtype), device)
    if return_fit:
        s1 = torch.ones(1, N, dtype=X.dtype, device=X.device)
        wf, muf, covf, _ = em_fit(X, s1, w1.unsqueeze(0), mu1.unsqueeze(0),
                                  cov1.unsqueeze(0), reg_covar, threshold_fit,
                                  max_iter, em_chunk=em_chunk_boot, n_chunk=n_chunk)
        alive = wf[0] > 0
        fit = FittedGMM(wf[0][alive],
                        muf[0][alive] * col_std + col_mean,
                        covf[0][alive] * (col_std[:, None] * col_std[None, :]))
    g = torch.Generator(device=X.device).manual_seed(seed + 10_000)
    if n_bootstrap >= 1:
        s = _bootstrap_counts(N, B, X.dtype, X.device, g)
    else:
        s = torch.ones(1, N, dtype=X.dtype, device=X.device)
    w0, mu0, cov0 = _warm_start(w1, mu1, cov1, B)
    w, mu, cov, _ = em_fit(X, s, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                           em_chunk=em_chunk_boot, n_chunk=n_chunk)

    # write the repaired covariances back: the integrands' marginal scores
    # refactorise cov, so sampling under the repaired factor while scoring
    # marginals from the raw covariance would mix two different mixtures
    chol, cov = _chol(cov)

    def _mc(sl):
        pts = gmm_sample(w[sl], mu[sl], chol[sl], mc_samples, generator=g)
        return _INTEGRANDS[quantity](pts, w[sl], mu[sl], cov[sl], chol[sl]).mean(1)
    vals = torch.cat(_run_chunked(B, mc_chunk, _mc))              # (B,)
    mean = vals.mean().item()
    std = vals.std(unbiased=True).item() if n_bootstrap >= 2 else None
    out = (mean, std)
    if return_k:
        out += (k_best,)
    if return_fit:
        out += (fit,)
    return out


@_entry_precision
def estimate_categorical(X, Y, protocol='batched', n_bootstrap=50,
                         mc_samples=100_000, max_components=10, n_folds=2, n_inits=3,
                         reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0,
                         return_k=False, device='cpu', dtype=torch.float64,
                         em_chunk='auto', mc_chunk='auto',
                         init_type='random_responsibilities', scale=None,
                         fixed_components_number=None, metric_method='valid',
                         threshold_components=1e-5, patience=1,
                         category_weights='empirical', n_chunk=None):
    """Estimate MI between a continuous variable and a categorical variable.

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

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, 1)
        The continuous variable.
    Y : array-like of shape (n_samples,)
        The categorical variable; categories are its unique values.
    category_weights : {'empirical', 'uniform'}, default='empirical'
        'empirical' sets pi_c = N_c / N from the observed category counts —
        the correct weighting for MI of the observed joint (X, Y).
        'uniform' sets pi_c = 1/C, reproducing the reference code, whose
        Appendix B formula assumes equiprobable categories.
    max_components : int, default=10
        Largest number of components considered per category's 1D fit.

    All remaining parameters are shared with `estimate` and have the same
    meaning there (applied per category): protocol, n_bootstrap, mc_samples,
    n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed, return_k,
    device, dtype, em_chunk, mc_chunk, init_type, scale,
    fixed_components_number, metric_method, threshold_components, patience,
    n_chunk.

    Returns
    -------
    mean : float
        Bootstrap-mean MI in nats.
    std : float or None
        Bootstrap standard deviation (None when n_bootstrap < 2:
        one replicate has no sample deviation).
    k : list of int
        Selected component count per category; only if `return_k` is True.
    """
    X = torch.as_tensor(X, dtype=dtype, device=device).detach()
    if X.dim() > 2 or (X.dim() == 2 and X.shape[1] != 1):
        raise ValueError(f"X must be 1-D or a single column; found shape "
                         f"{tuple(X.shape)}")
    X = X.reshape(-1, 1)
    if not torch.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values (NaN or inf in the working "
            "dtype); the standardisation propagates them into every "
            "replicate — remove or impute them first")
    Y = torch.as_tensor(Y, device=device).detach()
    if Y.dim() > 2 or (Y.dim() == 2 and Y.shape[1] != 1):
        raise ValueError(f"Y must be 1-D; found shape {tuple(Y.shape)}")
    Y = Y.reshape(-1)
    if Y.is_floating_point() and not torch.isfinite(Y).all():
        raise ValueError(
            "Y contains non-finite labels (NaN or inf); torch.unique would "
            "make every NaN its own category — remove those samples or "
            "encode missingness as an explicit category")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"X and Y must have the same length; found "
                         f"{X.shape[0]} and {Y.shape[0]}")
    if protocol not in ('batched', 'sequential'):
        raise ValueError(f"protocol must be 'batched' or 'sequential'; "
                         f"found '{protocol}'")
    (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
     max_iter, fixed_components_number) = _validate_entry_args(
        seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
        max_iter, fixed_components_number, reg_covar, threshold_components)
    if n_chunk is not None and int(n_chunk) < 1:
        raise ValueError(f"n_chunk must be a positive int or None; "
                         f"found {n_chunk}")
    # shared affine standardisation of the continuous variable: MI with a
    # categorical variable is invariant, and reg_covar becomes scale-free
    x_std = X.std()
    X = (X - X.mean()) / torch.where(x_std > 0, x_std, torch.ones_like(x_std))
    cats, counts = torch.unique(Y, return_counts=True)
    C = cats.shape[0]
    ci_min = int(counts.argmin())
    _validate_folds(int(counts[ci_min]), n_folds,
                    what=f"the smallest category (Y == {cats[ci_min].item()})")
    if category_weights == 'empirical':
        pi = counts.to(dtype) / Y.shape[0]
    elif category_weights == 'uniform':
        pi = torch.full((C,), 1.0 / C, dtype=dtype, device=X.device)
    else:
        raise ValueError("category_weights must be 'empirical' or 'uniform'")
    logpi = torch.log(pi)
    B = max(n_bootstrap, 1)
    fits, k_bests = [], []
    for ci in range(C):
        Xc = X[Y == cats[ci]]
        Nc = Xc.shape[0]
        seed_c = seed + 997 * ci
        k_c, (w1, mu1, cov1) = _select_and_fit(
            Xc, protocol, fixed_components_number, max_components, n_folds,
            n_inits, reg_covar, threshold_fit, max_iter, seed_c, em_chunk,
            init_type, scale, metric_method, threshold_components, patience,
            n_chunk)
        k_bests.append(k_c)
        g = torch.Generator(device=X.device).manual_seed(seed_c + 10_000)
        if n_bootstrap >= 1:
            s = _bootstrap_counts(Nc, B, X.dtype, X.device, g)
        else:
            s = torch.ones(1, Nc, dtype=X.dtype, device=X.device)
        K = w1.shape[0]
        em_chunk_b = _resolve_chunk(
            em_chunk, B, _em_bytes_per_replicate(Nc, K, 1, dtype, n_chunk), device)
        w0, mu0, cov0 = _warm_start(w1, mu1, cov1, B)
        w, mu, cov, _ = em_fit(Xc, s, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                               em_chunk=em_chunk_b, n_chunk=n_chunk)
        chol, cov = _chol(cov)
        fits.append((w, mu, cov, chol))

    g = torch.Generator(device=X.device).manual_seed(seed + 20_000)
    K_max = max(f[0].shape[1] for f in fits)
    mc_chunk = _resolve_chunk(
        mc_chunk, B, C * _mc_bytes_per_replicate(mc_samples, K_max, 1, dtype), device)
    vals = torch.zeros(B, dtype=X.dtype, device=X.device)
    for ci in range(C):
        w, mu, cov, chol = fits[ci]

        def _mc(sl):
            pts = gmm_sample(w[sl], mu[sl], chol[sl], mc_samples, generator=g)
            scores = torch.stack(
                [gmm_score(pts, fits[cj][0][sl], fits[cj][1][sl],
                           fits[cj][3][sl]) for cj in range(C)])   # (C,b,S)
            log_pbar = torch.logsumexp(scores + logpi[:, None, None], dim=0)
            return pi[ci] * (scores[ci] - log_pbar).mean(1)
        vals += torch.cat(_run_chunked(B, mc_chunk, _mc))
    mean = vals.mean().item()
    std = vals.std(unbiased=True).item() if n_bootstrap >= 2 else None
    return (mean, std, k_bests) if return_k else (mean, std)
