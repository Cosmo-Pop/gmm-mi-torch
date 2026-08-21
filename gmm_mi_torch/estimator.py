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

import math
import warnings

import torch

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
# slack.  Per-replicate EM cost is flat above a batch of ~32, so the CUDA
# floor sits there; CPU keeps the historical default.
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


def _em_bytes_per_replicate(N, K, d, dtype, n_chunk=None):
    n_eff = N if n_chunk is None else min(N, int(n_chunk))
    return n_eff * K * dtype.itemsize * (4 * d + 4)


def _mc_bytes_per_replicate(mc_samples, K, d, dtype):
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
    """Cholesky of (R,K,d,d), with dead components (zero cov) padded to I."""
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    dead = cov.abs().sum(dim=(-2, -1)) < 1e-30            # (R,K)
    safe = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
    return torch.linalg.cholesky(safe)


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
    return gmm_score(X_full[..., list(idx)], w, mu_m, _chol(cov_m))


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
    The initial M-step / assignment stats run in replicate chunks of `em_chunk`.
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
        outs = []
        for i in range(0, R, em_chunk):
            r = resp[i:i+em_chunk]
            c = r.shape[0]
            outs.append(_m_step(X.expand(c, N, d), r,
                                torch.ones(c, N, dtype=dt, device=dev), reg_covar))
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
        outs = []
        for i in range(0, R, em_chunk):
            Cc, al = C[i:i+em_chunk], alive[i:i+em_chunk]
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
            outs.append((w, mu, cov))
        w, mu, cov = (torch.cat([o[j] for o in outs]) for j in range(3))
        if init_type == 'randomized_kmeans':
            s = 1.0 if scale is None else float(scale)
            eps = torch.randn(R, K, d, dtype=dt, device=dev,
                              generator=generator)
            push = (_chol(cov) @ eps.unsqueeze(-1)).squeeze(-1)
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


def _em_step_nchunked(X, s, w, mu, cov, chol, logw, reg_covar, n_chunk):
    """One EM iteration accumulated over sample chunks (streaming M-step).

    Accumulates N_k, sum resp*(x - mu_old) and sum resp*(x - mu_old)(..)^T
    per sample-chunk; shifting the moments by the incoming means is algebraically
    identical to accumulating resp*x / resp*xx^T but avoids the fp32
    cancellation of the origin-moment form when |mu|^2 >> var."""
    Ra, N = s.shape
    d = X.shape[-1]
    K = w.shape[-1]
    dt, dev = X.dtype, X.device
    eye = torch.eye(d, dtype=dt, device=dev)
    Nk = torch.zeros(Ra, K, dtype=dt, device=dev)
    Sd = torch.zeros(Ra, K, d, dtype=dt, device=dev)
    Sdd = torch.zeros(Ra, K, d, d, dtype=dt, device=dev)
    ll_sum = torch.zeros(Ra, dtype=dt, device=dev)
    for i in range(0, N, n_chunk):
        Xc = X[i:i+n_chunk]
        nc = Xc.shape[0]
        sc = s[:, i:i+n_chunk]
        logjoint = logw.unsqueeze(1) + log_gauss(Xc.expand(Ra, nc, d),
                                                 mu, chol)       # (Ra,nc,K)
        ll_n = torch.logsumexp(logjoint, dim=-1)
        resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
        wr = resp * sc.unsqueeze(-1)
        ll_sum += (sc * ll_n).sum(1)
        Nk += wr.sum(1)
        diff = Xc.view(1, nc, 1, d) - mu.unsqueeze(1)            # (Ra,nc,K,d)
        Sd += torch.einsum('rnk,rnkd->rkd', wr, diff)
        Sdd += torch.einsum('rnk,rnki,rnkj->rkij', wr, diff, diff)
    ll = ll_sum / s.sum(1)
    dead = Nk < 1e-10
    Nk_safe = Nk.clamp_min(1e-10)
    delta = Sd / Nk_safe.unsqueeze(-1)                           # mu_new-mu_old
    mu_new = mu + delta
    cov_new = Sdd / Nk_safe[..., None, None] \
        - delta.unsqueeze(-1) * delta.unsqueeze(-2) + reg_covar * eye
    w_new = torch.where(dead, torch.zeros_like(Nk),
                        Nk / s.sum(1, keepdim=True))
    w_new = w_new / w_new.sum(-1, keepdim=True).clamp_min(1e-30)
    mu_new = torch.where(dead[..., None], torch.zeros_like(mu_new), mu_new)
    cov_new = torch.where(dead[..., None, None], eye.expand_as(cov_new),
                          cov_new)
    return ll, w_new, mu_new, cov_new


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
    for _ in range(max_iter):
        Ra = ai.shape[0]
        chol = _chol(cov)
        logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                           torch.full_like(w, -torch.inf))
        if n_chunk is None:
            Xr = X.expand(Ra, N, X.shape[-1])
            logjoint = logw.unsqueeze(1) + log_gauss(Xr, mu, chol)  # (Ra,N,K)
            ll_n = torch.logsumexp(logjoint, dim=-1)                # (Ra,N)
            resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
            ll = (sa * ll_n).sum(1) / sa.sum(1)
            w, mu, cov = _m_step(Xr, resp, sa, reg_covar)
        else:
            ll, w, mu, cov = _em_step_nchunked(X, sa, w, mu, cov, chol,
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


def val_ll(X, s_val, w, mu, cov, n_chunk=None):
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

    Returns
    -------
    torch.Tensor of shape (n_replicates,)
    """
    R, N = s_val.shape
    chol = _chol(cov)
    if n_chunk is None:
        ll_n = gmm_score(X.expand(R, N, X.shape[-1]), w, mu, chol)
        return (s_val * ll_n).sum(1) / s_val.sum(1).clamp_min(1e-30)
    acc = torch.zeros(R, dtype=X.dtype, device=X.device)
    for i in range(0, N, n_chunk):
        Xc = X[i:i+n_chunk]
        nc = Xc.shape[0]
        ll_n = gmm_score(Xc.expand(R, nc, X.shape[-1]), w, mu, chol)
        acc = acc + (s_val[:, i:i+n_chunk] * ll_n).sum(1)
    return acc / s_val.sum(1).clamp_min(1e-30)


# ----------------------------------------------------------------------
# component selection: the two protocols
# ----------------------------------------------------------------------

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
    train, val = _cv_masks(X.shape[0], n_folds, n_inits, X.dtype)
    train, val = train.to(X.device), val.to(X.device)
    R = train.shape[0]
    w0, mu0, cov0 = _init_params(X, K, R, reg_covar, generator=g,
                                 init_type=init_type, scale=scale, em_chunk=em_chunk)
    w, mu, cov, _ = em_fit(X, train, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                           em_chunk=em_chunk, n_chunk=n_chunk)
    v = val_ll(X, val, w, mu, cov, n_chunk=n_chunk)       # (I*F,)
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
    consecutive values of k. Cheaper than the batched protocol when it stops
    early, but the stopping decision is made under initialisation noise: one
    poor fit at a plateau can gate off better models at higher k.

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
        stopping; the selected k is the last improving one.
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
    best_metric, pat, history = -torch.inf, 0, {}
    for K in range(1, max_components + 1):
        metric, params = _cv_fit_for_k(X, K, n_folds, n_inits, reg_covar, threshold_fit,
                                       max_iter, seed, em_chunk=em_chunk,
                                       init_type=init_type, scale=scale,
                                       n_chunk=n_chunk)
        history[K] = (metric, params)
        if metric - best_metric > threshold_components:
            best_metric = metric
        else:
            pat += 1
            if pat >= patience:
                k_best = K - patience
                return k_best, history[k_best][1], history
    return max_components, history[max_components][1], history


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
    train, val = _cv_masks(N, n_folds, n_inits, X.dtype)
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
    v = val_ll(X, val_r, w, mu, cov,
               n_chunk=n_chunk).view(max_components, n_inits, n_folds)
    metric_k = v.mean(-1).max(-1).values                  # (max_components,)
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
    joint = gmm_score(pts, w, mu, chol)
    mx = marginal_score(pts, w, mu, cov, [0])
    my = marginal_score(pts, w, mu, cov, [1])
    return joint - mx - my


def _integrand_cMI(pts, w, mu, cov, chol):
    d = pts.shape[-1]
    zs = list(range(2, d))
    joint = gmm_score(pts, w, mu, chol)
    mz = marginal_score(pts, w, mu, cov, zs)
    mxz = marginal_score(pts, w, mu, cov, [0] + zs)
    myz = marginal_score(pts, w, mu, cov, [1] + zs)
    return joint + mz - mxz - myz


def _integrand_TC(pts, w, mu, cov, chol):
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
    return gmm_score(x, w, mu_x, _chol(cov_x)) \
        - gmm_score(x, w, mu_y, _chol(cov_y))


def _integrand_KL_reverse(pts, w, mu, cov, chol):
    """KL[p(y)||p(x)]: E_{y~p(y)}[log p_y(y) - log p_x(y)]."""
    y = pts[..., [1]]
    mu_x, cov_x = marginal_params(mu, cov, [0])
    mu_y, cov_y = marginal_params(mu, cov, [1])
    return gmm_score(y, w, mu_y, _chol(cov_y)) \
        - gmm_score(y, w, mu_x, _chol(cov_x))


_INTEGRANDS = {'MI': _integrand_MI, 'cMI': _integrand_cMI, 'TC': _integrand_TC,
               'KL_forward': _integrand_KL_forward,
               'KL_reverse': _integrand_KL_reverse}


def _rank_gaussianise(X):
    """Per-variable rank-probit transform: X -> Phi^{-1}(rank / (N+1)).

    MI, cMI and TC are invariant under per-variable diffeomorphisms, so no
    correction term is needed."""
    N = X.shape[0]
    ranks = X.argsort(dim=0).argsort(dim=0).to(X.dtype) + 1.0
    p = ranks / (N + 1.0)
    # ndtri/erfinv are JIT (nvrtc) kernels on CUDA and nvrtc rejects the
    # GB10 arch; this is one-off data prep, so run the probit on CPU
    return torch.special.ndtri(p.cpu()).to(p.device)


class FittedGMM:
    """The GMM selected by `estimate`, refitted on the full data.

    Mirrors GMM-MI's fitted-model access for plotting and inspection, with
    sklearn-style attributes: `weights_` (K,), `means_` (K, d) and
    `covariances_` (K, d, d) as numpy arrays on the CPU (dead padding
    components dropped), plus `n_components`. Obtained via
    `estimate(..., return_fit=True)`.
    """

    def __init__(self, w, mu, cov):
        self._w, self._mu, self._cov = w, mu, cov
        self.weights_ = w.cpu().numpy()
        self.means_ = mu.cpu().numpy()
        self.covariances_ = cov.cpu().numpy()
        self.n_components = int(w.shape[0])

    def sample(self, n_samples, seed=0):
        """Draw `n_samples` from the mixture; returns a (n_samples, d) numpy array."""
        g = torch.Generator(device=self._w.device).manual_seed(seed)
        pts = gmm_sample(self._w.unsqueeze(0), self._mu.unsqueeze(0),
                         _chol(self._cov.unsqueeze(0)), int(n_samples),
                         generator=g)
        return pts[0].cpu().numpy()

    def score_samples(self, X):
        """Mixture log-density at X of shape (n, d); returns a (n,) numpy array."""
        Xt = torch.as_tensor(X, dtype=self._mu.dtype, device=self._mu.device)
        ll = gmm_score(Xt.unsqueeze(0), self._w.unsqueeze(0),
                       self._mu.unsqueeze(0), _chol(self._cov.unsqueeze(0)))
        return ll[0].cpu().numpy()


def estimate(X, quantity='MI', protocol='batched', n_bootstrap=50,
             mc_samples=100_000, max_components=20, n_folds=2, n_inits=3,
             reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0, return_k=False,
             return_fit=False,
             device='cpu', dtype=torch.float64, em_chunk='auto', mc_chunk='auto',
             preprocess='none', init_type='random_responsibilities',
             scale=None, fixed_components_number=None, metric_method='valid',
             threshold_components=1e-5, n_chunk=None):
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
        Set to 0 to skip the bootstrap (single fit on the full data; the
        returned standard deviation is then None).
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
        GMM-MI's 1e-15 would regularise nothing.
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
    device : str, default='cpu'
        Torch device on which to fit and integrate (e.g. 'cuda').
    dtype : torch.dtype, default=torch.float64
        Working precision. float32 is validated and much faster on GPUs with
        limited fp64 throughput; float64 is the reference precision.
    em_chunk : int or 'auto', default='auto'
        Number of EM replicates (CV fits or bootstrap realisations) processed
        per batch. Larger values are faster on GPU at higher peak memory
        (results are chunk-invariant). 'auto' keeps 8 on CPU; on CUDA it
        sizes each stage's batch to fill half of the free device memory from
        the measured peak-memory model (replicates x N x K x (4d+4) x
        itemsize), floored at 32 — where the per-replicate cost flattens out —
        and capped at the stage's replicate count. A CUDA out-of-memory error
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
        Rejected (ValueError) for the KL quantities, which are not invariant
        under marginal transforms.
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
        Bootstrap standard deviation (None when n_bootstrap=0).
    k : int
        Selected number of components; only if `return_k` is True.
    fit : FittedGMM
        The selected model refitted on the full data; only if `return_fit`
        is True (appended after `k` when both flags are set).
    """
    X = torch.as_tensor(X, dtype=dtype, device=device)
    if preprocess == 'gaussianise':
        if quantity in ('KL_forward', 'KL_reverse'):
            raise ValueError(
                "preprocess='gaussianise' is invalid for KL quantities: "
                "KL is not invariant under marginal transforms (the "
                "gaussianised marginals have ~zero KL by construction)")
        X = _rank_gaussianise(X)
    elif preprocess != 'none':
        raise ValueError(f"preprocess must be 'none' or 'gaussianise'; "
                         f"found '{preprocess}'")
    if quantity in ('MI', 'KL_forward', 'KL_reverse'):
        assert X.shape[1] == 2
    if quantity == 'cMI':
        assert X.shape[1] >= 3
    N, d = X.shape
    IF = n_inits * n_folds
    if fixed_components_number is not None:
        k_sel, r_sel = int(fixed_components_number), IF
    elif protocol == 'sequential':
        k_sel, r_sel = max_components, IF
    else:
        k_sel, r_sel = max_components, max_components * IF
    em_chunk_sel = _resolve_chunk(
        em_chunk, r_sel, _em_bytes_per_replicate(N, k_sel, d, dtype, n_chunk), device)
    if fixed_components_number is not None:
        k_best = int(fixed_components_number)
        _, (w1, mu1, cov1) = _cv_fit_for_k(X, k_best, n_folds, n_inits, reg_covar,
                                           threshold_fit, max_iter, seed, em_chunk=em_chunk_sel,
                                           init_type=init_type, scale=scale,
                                           n_chunk=n_chunk)
    else:
        sel = select_sequential if protocol == 'sequential' else select_batched
        kw = dict(max_components=max_components, n_folds=n_folds, n_inits=n_inits, reg_covar=reg_covar,
                  threshold_fit=threshold_fit, max_iter=max_iter, seed=seed,
                  threshold_components=threshold_components,
                  em_chunk=em_chunk_sel, init_type=init_type, scale=scale,
                  n_chunk=n_chunk)
        if protocol == 'sequential':
            if metric_method != 'valid':
                raise ValueError("metric_method is only supported by the "
                                 "batched protocol")
        else:
            kw['metric_method'] = metric_method
        k_best, (w1, mu1, cov1), _ = sel(X, **kw)

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
        fit = FittedGMM(wf[0][alive], muf[0][alive], covf[0][alive])
    g = torch.Generator(device=X.device).manual_seed(seed + 10_000)
    if n_bootstrap >= 1:
        if N <= _MULTINOMIAL_MAX_CATEGORIES:
            s = torch.multinomial(
                torch.full((B, N), 1.0 / N, dtype=X.dtype, device=X.device),
                N, replacement=True, generator=g)
        else:
            # torch.multinomial caps categories at 2^24; for uniform
            # probabilities the resample counts are identically distributed
            # as a histogram of N iid uniform indices (previously this size
            # raised, so no existing result changes)
            s = torch.randint(0, N, (B, N), device=X.device, generator=g)
        s = torch.zeros(B, N, dtype=X.dtype, device=X.device).scatter_add_(
            1, s, torch.ones(B, N, dtype=X.dtype, device=X.device))
    else:
        s = torch.ones(1, N, dtype=X.dtype, device=X.device)
    w0 = w1.unsqueeze(0).expand(B, K).contiguous()
    mu0 = mu1.unsqueeze(0).expand(B, K, d).contiguous()
    cov0 = cov1.unsqueeze(0).expand(B, K, d, d).contiguous()
    w, mu, cov, _ = em_fit(X, s, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                           em_chunk=em_chunk_boot, n_chunk=n_chunk)

    chol = _chol(cov)

    def _mc(sl):
        pts = gmm_sample(w[sl], mu[sl], chol[sl], mc_samples, generator=g)
        return _INTEGRANDS[quantity](pts, w[sl], mu[sl], cov[sl], chol[sl]).mean(1)
    vals = torch.cat(_run_chunked(B, mc_chunk, _mc))              # (B,)
    mean = vals.mean().item()
    std = vals.std(unbiased=True).item() if n_bootstrap >= 1 else None
    out = (mean, std)
    if return_k:
        out += (k_best,)
    if return_fit:
        out += (fit,)
    return out


def estimate_categorical(X, Y, protocol='batched', n_bootstrap=50,
                         mc_samples=100_000, max_components=10, n_folds=2, n_inits=3,
                         reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0,
                         return_k=False, device='cpu', dtype=torch.float64,
                         em_chunk='auto', mc_chunk='auto',
                         init_type='random_responsibilities', scale=None,
                         fixed_components_number=None, metric_method='valid',
                         category_weights='empirical', n_chunk=None):
    """Estimate MI between a continuous variable and a categorical variable.

    Port of GMM-MI's fit_categorical/estimate_categorical semantics
    (Appendix B of Piras et al. 2022): a 1D GMM p(x|c) is fitted per
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
    meaning there: protocol, n_bootstrap, mc_samples, n_folds, n_inits,
    reg_covar, threshold_fit, max_iter, seed, return_k, device, dtype,
    em_chunk, mc_chunk, init_type, scale, fixed_components_number,
    metric_method, n_chunk.

    Returns
    -------
    mean : float
        Bootstrap-mean MI in nats.
    std : float or None
        Bootstrap standard deviation (None when n_bootstrap=0).
    k : list of int
        Selected component count per category; only if `return_k` is True.
    """
    X = torch.as_tensor(X, dtype=dtype, device=device).reshape(-1, 1)
    Y = torch.as_tensor(Y, device=device).reshape(-1)
    cats = torch.unique(Y)
    C = cats.shape[0]
    if category_weights == 'empirical':
        pi = torch.stack([(Y == c).sum() for c in cats]).to(dtype) / Y.shape[0]
    elif category_weights == 'uniform':
        pi = torch.full((C,), 1.0 / C, dtype=dtype, device=X.device)
    else:
        raise ValueError("category_weights must be 'empirical' or 'uniform'")
    logpi = torch.log(pi)
    B = max(n_bootstrap, 1)
    fits, k_bests = [], []
    IF = n_inits * n_folds
    for ci in range(C):
        Xc = X[Y == cats[ci]]
        Nc = Xc.shape[0]
        seed_c = seed + 997 * ci
        if fixed_components_number is not None:
            k_sel, r_sel = int(fixed_components_number), IF
        elif protocol == 'sequential':
            k_sel, r_sel = max_components, IF
        else:
            k_sel, r_sel = max_components, max_components * IF
        em_chunk_c = _resolve_chunk(
            em_chunk, r_sel, _em_bytes_per_replicate(Nc, k_sel, 1, dtype, n_chunk), device)
        if fixed_components_number is not None:
            k_c = int(fixed_components_number)
            _, (w1, mu1, cov1) = _cv_fit_for_k(
                Xc, k_c, n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed_c,
                em_chunk=em_chunk_c, init_type=init_type, scale=scale,
                n_chunk=n_chunk)
        else:
            sel = (select_sequential if protocol == 'sequential'
                   else select_batched)
            kw = dict(max_components=max_components, n_folds=n_folds, n_inits=n_inits, reg_covar=reg_covar,
                      threshold_fit=threshold_fit, max_iter=max_iter, seed=seed_c, threshold_components=1e-5,
                      em_chunk=em_chunk_c, init_type=init_type, scale=scale,
                      n_chunk=n_chunk)
            if protocol != 'sequential':
                kw['metric_method'] = metric_method
            k_c, (w1, mu1, cov1), _ = sel(Xc, **kw)
        k_bests.append(k_c)
        g = torch.Generator(device=X.device).manual_seed(seed_c + 10_000)
        if n_bootstrap >= 1:
            s = torch.multinomial(
                torch.full((B, Nc), 1.0 / Nc, dtype=X.dtype, device=X.device),
                Nc, replacement=True, generator=g)
            s = torch.zeros(B, Nc, dtype=X.dtype, device=X.device) \
                .scatter_add_(1, s, torch.ones(B, Nc, dtype=X.dtype,
                                               device=X.device))
        else:
            s = torch.ones(1, Nc, dtype=X.dtype, device=X.device)
        K = w1.shape[0]
        em_chunk_b = _resolve_chunk(
            em_chunk, B, _em_bytes_per_replicate(Nc, K, 1, dtype, n_chunk), device)
        w0 = w1.unsqueeze(0).expand(B, K).contiguous()
        mu0 = mu1.unsqueeze(0).expand(B, K, 1).contiguous()
        cov0 = cov1.unsqueeze(0).expand(B, K, 1, 1).contiguous()
        w, mu, cov, _ = em_fit(Xc, s, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                               em_chunk=em_chunk_b, n_chunk=n_chunk)
        fits.append((w, mu, cov, _chol(cov)))

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
    std = vals.std(unbiased=True).item() if n_bootstrap >= 1 else None
    return (mean, std, k_bests) if return_k else (mean, std)
