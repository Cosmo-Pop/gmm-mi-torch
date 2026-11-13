"""Pure-torch port of GMM-MI with vectorised bootstrap and TC.

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
constants: docs/design_notes.md.
"""

import functools
import warnings
from typing import NamedTuple

import torch

from .constants import (
    _BOOTSTRAP_SEED_OFFSET, _CATEGORY_SEED_STRIDE, _MC_CAT_STRIDE,
    _MC_SEED_OFFSET, _MULTINOMIAL_MAX_CATEGORIES, _TIE_SEED_OFFSET)
from .precision import _as_array, _entry_precision, _validate_entry_args
from .chunking import (
    _cuda_free_bytes, _em_bytes_per_replicate, _em_fixed_bytes,
    _init_bytes_per_replicate, _mc_bytes_per_replicate, _resolve_chunk,
    _run_chunked)
from .gaussian import (
    _chol, gmm_sample, gmm_score, marginal_params, marginal_score)
from .em import em_fit
from .selection import (
    _cv_fit_for_k, _validate_device, _validate_folds, select_batched,
    select_sequential)

# public names this module defined up to v0.9.0 and does not use
# itself, imported so `from gmm_mi_torch.estimator import ...` still
# finds them
from .constants import LOG2PI  # noqa: F401
from .gaussian import log_gauss  # noqa: F401
from .em import EMResult, ValLLResult, val_ll  # noqa: F401


# ----------------------------------------------------------------------
# integrands: each quantity from one joint fit, marginals by masking
# ----------------------------------------------------------------------

def _integrand_MI(pts, w, mu, cov, chol):
    """
    Integrand of the mutual information I(x; y).

    log p(x,y) - log p(x) - log p(y) at the sampled points.

    Parameters
    ----------
    pts : torch.Tensor of shape (B, S, d)
        Monte Carlo samples from each replicate's mixture.
    w, mu, cov, chol : torch.Tensor
        Mixture weights (B, K), means (B, K, d), covariances (B, K, d, d)
        and their Cholesky factors.

    Returns
    -------
    torch.Tensor of shape (B, S)
        The integrand at each sample; its mean over S is the estimate, in
        nats.
    """
    joint = gmm_score(pts, w, mu, chol)
    mx = marginal_score(pts, w, mu, cov, [0])
    my = marginal_score(pts, w, mu, cov, [1])
    return joint - mx - my


def _integrand_cMI(pts, w, mu, cov, chol):
    """
    Integrand of the conditional mutual information I(x; y | z).

    log p(x,y,z) + log p(z) - log p(x,z) - log p(y,z).

    Parameters
    ----------
    pts : torch.Tensor of shape (B, S, d)
        Monte Carlo samples from each replicate's mixture.
    w, mu, cov, chol : torch.Tensor
        Mixture weights (B, K), means (B, K, d), covariances (B, K, d, d)
        and their Cholesky factors.

    Returns
    -------
    torch.Tensor of shape (B, S)
        The integrand at each sample; its mean over S is the estimate, in
        nats.
    """
    d = pts.shape[-1]
    zs = list(range(2, d))
    joint = gmm_score(pts, w, mu, chol)
    mz = marginal_score(pts, w, mu, cov, zs)
    mxz = marginal_score(pts, w, mu, cov, [0] + zs)
    myz = marginal_score(pts, w, mu, cov, [1] + zs)
    return joint + mz - mxz - myz


def _integrand_TC(pts, w, mu, cov, chol):
    """
    Integrand of the total correlation.

    log p(x) - sum_i log p(x_i).

    Parameters
    ----------
    pts : torch.Tensor of shape (B, S, d)
        Monte Carlo samples from each replicate's mixture.
    w, mu, cov, chol : torch.Tensor
        Mixture weights (B, K), means (B, K, d), covariances (B, K, d, d)
        and their Cholesky factors.

    Returns
    -------
    torch.Tensor of shape (B, S)
        The integrand at each sample; its mean over S is the estimate, in
        nats.
    """
    joint = gmm_score(pts, w, mu, chol)
    marg = sum(marginal_score(pts, w, mu, cov, [i])
               for i in range(pts.shape[-1]))
    return joint - marg


def _integrand_KL_forward(pts, w, mu, cov, chol):
    """
    Integrand of KL[p(x) || p(y)] between the two 1-D marginals.

    As in GMM-MI's estimate_KL_MC: E_{x~p(x)}[log p_x(x) - log p_y(x)], with x
    the first column of the joint samples.

    Parameters
    ----------
    pts : torch.Tensor of shape (B, S, d)
        Monte Carlo samples from each replicate's mixture.
    w, mu, cov, chol : torch.Tensor
        Mixture weights (B, K), means (B, K, d), covariances (B, K, d, d)
        and their Cholesky factors.

    Returns
    -------
    torch.Tensor of shape (B, S)
        The integrand at each sample; its mean over S is the estimate, in
        nats.
    """
    x = pts[..., [0]]
    mu_x, cov_x = marginal_params(mu, cov, [0])
    mu_y, cov_y = marginal_params(mu, cov, [1])
    return gmm_score(x, w, mu_x, _chol(cov_x)[0]) \
        - gmm_score(x, w, mu_y, _chol(cov_y)[0])


def _integrand_KL_reverse(pts, w, mu, cov, chol):
    """
    Integrand of KL[p(y) || p(x)] between the two 1-D marginals.

    E_{y~p(y)}[log p_y(y) - log p_x(y)].

    Parameters
    ----------
    pts : torch.Tensor of shape (B, S, d)
        Monte Carlo samples from each replicate's mixture.
    w, mu, cov, chol : torch.Tensor
        Mixture weights (B, K), means (B, K, d), covariances (B, K, d, d)
        and their Cholesky factors.

    Returns
    -------
    torch.Tensor of shape (B, S)
        The integrand at each sample; its mean over S is the estimate, in
        nats.
    """
    y = pts[..., [1]]
    mu_x, cov_x = marginal_params(mu, cov, [0])
    mu_y, cov_y = marginal_params(mu, cov, [1])
    return gmm_score(y, w, mu_y, _chol(cov_y)[0]) \
        - gmm_score(y, w, mu_x, _chol(cov_x)[0])


_INTEGRANDS = {'MI': _integrand_MI, 'cMI': _integrand_cMI, 'TC': _integrand_TC,
               'KL_forward': _integrand_KL_forward,
               'KL_reverse': _integrand_KL_reverse}


# tie warning threshold; design notes: _GAUSSIANISE_TIE_WARN_FRACTION
_GAUSSIANISE_TIE_WARN_FRACTION = 0.01


def _rank_gaussianise(X, tie_rank='average', generator=None):
    """
    Per-variable rank-probit transform: X -> Phi^{-1}(rank / (N+1)).

    MI, cMI and TC are invariant under per-variable diffeomorphisms, so no
    correction term is needed.  Ranks and quantiles are computed at float64
    (float32 integer ranks round above 2^24 samples).

    tie_rank:
        'random'  : tied values get a seeded random ordering within their
                    block — `estimate`'s default. Equivalent to breaking
                    ties with independent sub-resolution jitter, under
                    which the MI estimand is EXACTLY unchanged (the
                    original values are recoverable from the tie-broken
                    ones, so data processing gives equality both ways) and
                    the transformed marginal is exactly uniform. The
                    permutation is drawn independently per column: one
                    shared permutation would correlate the tie-breaks of
                    blocks shared across columns (joint zeros) and
                    fabricate MI inside the atom.
        'average' : tied values share their average rank (the scipy
                    `rankdata` convention) — this function's default, for
                    deterministic direct use. The tied block maps to a
                    single probit atom.
    Neither convention makes a GMM fit of heavily tied data reliable; the
    warning below recommends preprocess='none' there.

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        The data.
    tie_rank : {'average', 'random'}, default='average'
        How tied values are ranked, as above.
    generator : torch.Generator, optional
        CPU generator for the random tie order.

    Returns
    -------
    torch.Tensor of shape (N, d)
        The transformed data, in X's dtype and on X's device.
    """
    N = X.shape[0]
    cols, tie_frac = [], []
    for j in range(X.shape[1]):
        xj = X[:, j]
        if tie_rank == 'average':
            vals, inv, cnt = torch.unique(xj, return_inverse=True,
                                          return_counts=True)
        elif tie_rank == 'random':
            vals, cnt = torch.unique(xj, return_counts=True)
        else:
            raise ValueError(f"tie_rank must be 'random' or 'average'; "
                             f"found '{tie_rank}'")
        big = cnt[cnt >= 3]
        tie_frac.append(float((big - 1).sum()) / N)
        if tie_rank == 'random':
            # random order within tied blocks; design notes: _rank_gaussianise
            perm = torch.randperm(N, generator=generator,
                                  device='cpu').to(xj.device)
            order = torch.argsort(xj[perm], dim=0, stable=True)
            rp = torch.argsort(order, dim=0, stable=True)
            ranks = torch.empty(N, dtype=torch.float64, device=xj.device)
            ranks[perm] = rp.double() + 1.0
            cols.append(ranks)
        else:
            # average rank of a tied block: (#smaller values) + (count+1)/2
            csum = torch.cumsum(cnt, 0)
            avg_rank = (csum - cnt).double() + (cnt.double() + 1.0) / 2.0
            cols.append(avg_rank[inv])
    tied = [j for j, f in enumerate(tie_frac)
            if f > _GAUSSIANISE_TIE_WARN_FRACTION]
    if tied:
        fracs = ', '.join(f'{tie_frac[j]:.2f}' for j in tied)
        if tie_rank == 'random':
            detail = ("ties were broken by seeded random ranks, which "
                      "leaves the MI estimand exactly unchanged but biases "
                      "the GMM estimate LOW when the tied values are "
                      "shared between columns (e.g. joint zero-inflation), "
                      "and the bootstrap errorbar excludes the rank "
                      "transform's own sampling variability")
        else:
            detail = ("tied blocks share their average rank and map to a "
                      "single probit atom, which can fabricate MI between "
                      "independent heavily tied columns")
        warnings.warn(
            f"gaussianise: column(s) {tied} have heavily tied values (tie "
            f"fraction {fracs}); {detail}. For heavily tied or "
            f"zero-inflated data, preprocess='none' is the more reliable "
            f"choice.", RuntimeWarning, stacklevel=4)
        # 4: _rank_gaussianise -> estimate -> _entry_precision -> caller
    p = torch.stack(cols, 1) / (N + 1.0)
    # ndtri/erfinv are JIT (nvrtc) kernels on CUDA and nvrtc rejects the
    # GB10 arch; this is one-off data prep, so run the probit on CPU
    return torch.special.ndtri(p.cpu()).to(dtype=X.dtype, device=X.device)


def _select_and_fit(X, selection, fit):
    """
    Component selection and best CV fit for one dataset.

    The single home of the selection pipeline shared by `estimate` and
    `estimate_categorical` (per-category), so the two cannot drift apart:
    chunk sizing, the fixed-k shortcut, protocol dispatch and the
    protocol/metric_method compatibility rules live only here.
    `selection` and `fit` are the two plain dicts the entry points build
    from their validated arguments: the selection knobs (protocol,
    fixed_components_number, max_components, n_folds, n_inits,
    metric_method, threshold_components, patience) and the fit settings
    (reg_covar, threshold_fit, max_iter, em_chunk, n_chunk, init_type,
    scale, seed).
    Returns (k, (w, mu, cov)) with dead components sliced away and k the
    LIVE component count of the returned parameters — the model that is
    actually bootstrapped, MC-integrated and returned — which can be
    smaller than the selected/requested k when components died during the
    CV fit (a RuntimeWarning says so for a requested
    fixed_components_number).

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        The standardised data, on the working device and dtype.
    selection : dict
        The selection knobs, as above.
    fit : dict
        The fit settings, as above.

    Returns
    -------
    k : int
        Live component count.
    params : tuple of torch.Tensor
        (w, mu, cov) of the best CV fit, shapes (k,), (k, d), (k, d, d).
    """
    N, d = X.shape
    IF = selection['n_inits'] * selection['n_folds']
    # compatibility rules before the fixed-k shortcut: a typo'd
    # metric_method must raise even when selection itself is skipped
    if selection['protocol'] == 'sequential':
        if selection['metric_method'] != 'valid':
            raise ValueError("metric_method is only supported by the "
                             "batched protocol")
    elif selection['metric_method'] not in ('valid', 'aic', 'bic'):
        raise ValueError(f"metric_method must be one of 'valid', "
                         f"'aic', 'bic'; found '{selection['metric_method']}'")
    if selection['fixed_components_number'] is not None:
        k_sel, r_sel = int(selection['fixed_components_number']), IF
    elif selection['protocol'] == 'sequential':
        k_sel, r_sel = selection['max_components'], IF
    else:
        k_sel, r_sel = (selection['max_components'],
                        selection['max_components'] * IF)
    em_chunk_sel = _resolve_chunk(
        fit['em_chunk'], r_sel,
        _em_bytes_per_replicate(N, k_sel, d, X.dtype, fit['n_chunk']),
        X.device, fixed_bytes=_em_fixed_bytes(N, d, X.dtype, fit['n_chunk']),
        name='em_chunk')
    # the init stage has its own (larger) model, streamed with n_chunk: an
    # integer em_chunk is honoured as given, 'auto' sizes the two apart
    init_bytes = _init_bytes_per_replicate(N, k_sel, d, X.dtype,
                                           fit['init_type'], fit['n_chunk'])
    if X.device.type == 'cuda':
        # a per-replicate feasibility test, whatever the replicate batching
        # (the model's estimate; it is a conservative envelope)
        free = _cuda_free_bytes(X.device)
        if init_bytes > free:
            stage = ("EM stage" if fit['init_type'] == 'minmax'
                     else f"initialisation stage of init_type='{fit['init_type']}'")
            remedy = ("set n_chunk, which streams the samples (reduction-order "
                      "cost only), or use float32"
                      if fit['init_type'] == 'minmax' else
                      "set n_chunk, which streams the initialisation over "
                      "sample chunks (reduction-order cost only), or use "
                      "init_type='minmax', which allocates nothing of this size")
            raise RuntimeError(
                f"by the memory model, the {stage} needs "
                f"{init_bytes / 2**30:.1f} GiB per replicate at N={N}, "
                f"k={k_sel}, d={d} — more than the {free / 2**30:.1f} GiB "
                f"this process can allocate on {X.device}, so not even one "
                f"replicate fits: {remedy}")
    init_chunk_sel = _resolve_chunk(fit['em_chunk'], r_sel, init_bytes,
                                    X.device, name='em_chunk')
    if selection['fixed_components_number'] is not None:
        k_best = int(selection['fixed_components_number'])
        # the CV fit runs at the replicate chunk resolved above
        _, params = _cv_fit_for_k(X, k_best, selection['n_folds'],
                                  selection['n_inits'],
                                  {**fit, 'em_chunk': em_chunk_sel},
                                  init_chunk=init_chunk_sel)
    else:
        kw = dict(max_components=selection['max_components'],
                  n_folds=selection['n_folds'],
                  n_inits=selection['n_inits'], reg_covar=fit['reg_covar'],
                  threshold_fit=fit['threshold_fit'],
                  max_iter=fit['max_iter'], seed=fit['seed'],
                  threshold_components=selection['threshold_components'],
                  em_chunk=em_chunk_sel, init_type=fit['init_type'],
                  scale=fit['scale'], n_chunk=fit['n_chunk'],
                  init_chunk=init_chunk_sel)
        if selection['protocol'] == 'sequential':
            sel = select_sequential
            kw['patience'] = selection['patience']
        else:
            sel = select_batched
            kw['metric_method'] = selection['metric_method']
        k_best, params, _ = sel(X, **kw)
    # drop dead components; design notes: _select_and_fit
    w1, mu1, cov1 = params
    alive = w1 > 0
    k_live = int(alive.sum())
    if selection['fixed_components_number'] is not None and k_live != k_best:
        warnings.warn(
            f"fixed_components_number={k_best}: {k_best - k_live} "
            f"component(s) died during the CV fit, so the model that is "
            f"bootstrapped and returned has {k_live}",
            RuntimeWarning, stacklevel=4)
    return k_live, (w1[alive], mu1[alive], cov1[alive])


def _bootstrap_counts(N, B, dtype, device, generator):
    """
    Uniform multinomial bootstrap resample counts, shape (B, N).

    Identical in distribution to index resampling. Up to
    _MULTINOMIAL_MAX_CATEGORIES (torch.multinomial's cap, 2^24) the counts
    come from ONE torch.multinomial call on the whole (B, N) index matrix —
    the validated resample stream — whose probability input is a float64
    (1, N) row expanded to (B, N), whatever the working dtype (so float32
    and float64 resamples are bitwise identical, on CPU and CUDA), and whose
    scatter source is an expanded ones row: the only full-size transient is
    torch.multinomial's own (B, N) int64 index matrix plus its internal
    prefix sums, outside the OOM guard by design. Above the cap each
    replicate's indices are floor(u N) from float64 uniforms, clamped to
    N - 1 (which guards only u == 1.0), drawn from a per-replicate generator
    seeded off `generator` and histogrammed one replicate at a time:
    chunk- and retry-invariant by construction, with a transient of 16
    B/sample. This helper is the fallback's only home.

    The uniformity bounds of both paths, the CUDA multinomial's
    float32-resolution weight scatter, the clamp proof and the CPU
    per-replicate seed space: docs/design_notes.md, "Bootstrap resample".

    Parameters
    ----------
    N : int
        Number of samples.
    B : int
        Number of bootstrap replicates.
    dtype : torch.dtype
        Dtype of the counts.
    device : str or torch.device
        Device of the counts.
    generator : torch.Generator
        Device-matched RNG.

    Returns
    -------
    torch.Tensor of shape (B, N)
        Resample counts; each row sums to N.
    """
    if N <= _MULTINOMIAL_MAX_CATEGORIES:
        idx = torch.multinomial(
            torch.full((1, N), 1.0 / N, dtype=torch.float64, device=device)
            .expand(B, N), N, replacement=True, generator=generator)
        # the counts are allocated after the draw: torch.multinomial's own
        # transient is the peak, and nothing else is live during it
        return torch.zeros(B, N, dtype=dtype, device=device).scatter_add_(
            1, idx, torch.ones(1, N, dtype=dtype, device=device).expand(B, N))
    s = torch.zeros(B, N, dtype=dtype, device=device)
    seeds = torch.randint(0, 1 << 62, (B,), generator=generator,
                          device=device).tolist()
    one = torch.ones((), dtype=dtype, device=device).expand(N)
    for b in range(B):
        gen = torch.Generator(device=device).manual_seed(seeds[b])
        # floor(u N) from float64 uniforms; design notes: _bootstrap_counts
        s[b].scatter_add_(
            0, torch.rand(N, dtype=torch.float64, device=device,
                          generator=gen).mul_(N).long().clamp_(max=N - 1),
            one)
    return s


def _warm_start(w1, mu1, cov1, B):
    """Tile one parameter set (K,), (K, d), (K, d, d) over B replicates.

    The results are contiguous copies, not expand views.
    """
    K, d = mu1.shape
    return (w1.unsqueeze(0).expand(B, K).contiguous(),
            mu1.unsqueeze(0).expand(B, K, d).contiguous(),
            cov1.unsqueeze(0).expand(B, K, d, d).contiguous())


class FittedGMM:
    """The GMM selected by `estimate`, refitted on the full data.

    Mirrors GMM-MI's fitted-model access for plotting and inspection, with
    sklearn-style attributes: `weights_` (K,), `means_` (K, d) and
    `covariances_` (K, d, d) as float64 numpy arrays on the CPU (dead
    padding components dropped), plus `n_components`. Obtained via
    `estimate(..., return_fit=True)`, which passes a Cholesky factor
    computed (and, when needed, repaired) in the fit's standardised units,
    where reg_covar and the repair thresholds are meaningful. Direct
    construction without `chol` factorises whatever units the caller
    supplies.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Weights (K,), means (K, d) and covariances (K, d, d).
    chol : torch.Tensor of shape (K, d, d), optional
        Lower Cholesky factor of `cov`; computed (and repaired if needed)
        from `cov` when omitted.
    """

    def __init__(self, w, mu, cov, chol=None):
        if chol is None:
            # factorise once: covariances_, sample and score_samples then
            # all describe the same (repaired) mixture, and any repair
            # warning fires here, not on use
            chol_b, cov_b = _chol(cov.unsqueeze(0))
            chol, cov = chol_b[0], cov_b[0]
        self._w, self._mu, self._cov = w, mu, cov
        self._chol = chol
        self.weights_ = w.cpu().numpy()
        self.means_ = mu.cpu().numpy()
        self.covariances_ = self._cov.cpu().numpy()
        self.n_components = int(w.shape[0])

    def sample(self, n_samples, seed=0):
        """
        Draw `n_samples` from the mixture.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int, default=0
            Seed of the sampling generator.

        Returns
        -------
        numpy.ndarray of shape (n_samples, d)
            The samples, in the units of the fitted model.
        """
        g = torch.Generator(device=self._w.device).manual_seed(seed)
        pts = gmm_sample(self._w.unsqueeze(0), self._mu.unsqueeze(0),
                         self._chol.unsqueeze(0), int(n_samples),
                         generator=g)
        return pts[0].cpu().numpy()

    def score_samples(self, X):
        """
        Mixture log-density at X.

        A 1-D X is n points of a 1-D model, or one point of a d-D model
        (sklearn rejects 1-D input); any other shape raises a ValueError naming
        the expected one.

        Parameters
        ----------
        X : array-like of shape (n, d) or (n,)
            Points, in the units of the fitted model.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Log-density at each point.
        """
        Xt = torch.as_tensor(_as_array(X), dtype=self._mu.dtype,
                             device=self._mu.device)
        d = self._mu.shape[-1]
        given = tuple(Xt.shape)
        if Xt.dim() == 1:
            Xt = Xt.view(-1, 1) if d == 1 else Xt.view(1, -1)
        if Xt.dim() != 2 or Xt.shape[1] != d:
            raise ValueError(f"X must have shape (n, {d}) for this "
                             f"{d}-D model (a 1-D input is read as one "
                             f"{d}-D point); found {given}")
        ll = gmm_score(Xt.unsqueeze(0), self._w.unsqueeze(0),
                       self._mu.unsqueeze(0), self._chol.unsqueeze(0))
        return ll[0].cpu().numpy()


def _mc_stage(sl, w, mu, cov, chol, quantity, mc_samples, seed,
              n_replicates):
    """
    Monte Carlo estimate of `quantity` for one chunk of bootstrap replicates.

    The chunk callback of `estimate`'s MC stage. Each replicate draws from
    its own seeded stream, so the samples are independent of `mc_chunk` and
    of the out-of-memory halve-and-retry.

    Parameters
    ----------
    sl : slice
        Replicate range handed over by `_run_chunked`.
    w, mu, cov, chol : torch.Tensor
        Fitted mixture parameters and Cholesky factors, shapes (B, K),
        (B, K, d), (B, K, d, d), (B, K, d, d).
    quantity : str
        Key into `_INTEGRANDS`.
    mc_samples : int
        Monte Carlo samples per replicate.
    seed : int
        Base seed; replicate r uses seed + _MC_SEED_OFFSET + r.
    n_replicates : int
        Total number of replicates B (the slice is clipped to it).

    Returns
    -------
    vals : torch.Tensor
        Estimate per replicate in the chunk, shape (b,), in nats.
    """
    pts = torch.cat([
        gmm_sample(w[r:r + 1], mu[r:r + 1], chol[r:r + 1], mc_samples,
                   generator=torch.Generator(device=w.device).manual_seed(
                       seed + _MC_SEED_OFFSET + r))
        for r in range(sl.start, min(sl.stop, n_replicates))])
    return _INTEGRANDS[quantity](pts, w[sl], mu[sl], cov[sl], chol[sl]).mean(1)


@_entry_precision
def estimate(X, quantity='MI', protocol='batched', n_bootstrap=50,
             mc_samples=100_000, max_components=20, n_folds=2, n_inits=3,
             reg_covar=1e-6, threshold_fit=1e-5, max_iter=500, seed=0, return_k=False,
             return_fit=False,
             device='cpu', dtype=torch.float64, em_chunk='auto', mc_chunk='auto',
             preprocess='none', tie_rank='random',
             init_type='random_responsibilities',
             scale=None, fixed_components_number=None, metric_method='valid',
             threshold_components=1e-5, patience=1, n_chunk=None):
    """
    Estimate an information-theoretic quantity with bootstrap errorbars.

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
        for 'cMI' (variables 0 and 1 are X and Y, the rest condition); at
        least two for 'TC'.
    quantity : {'MI', 'cMI', 'TC', 'KL_forward', 'KL_reverse'}, default='MI'
        'MI': I(X;Y); 'cMI': I(X;Y|Z), Z the columns beyond the first two;
        'TC': total correlation KL[p(x) || prod_i p(x_i)], zero iff the
        variables are mutually independent; 'KL_forward': KL[p(x0) || p(x1)]
        between the two 1D marginals; 'KL_reverse': the opposite order.
    protocol : {'batched', 'sequential'}, default='batched'
        Component selection. 'batched' fits k = 1..max_components at once
        and picks the smallest k within `threshold_components` of the best
        validation metric; 'sequential' is GMM-MI's original protocol
        (increment k, stop early via threshold and patience). See Notes.
    n_bootstrap : int, default=50
        Bootstrap realisations for the errorbar, batched per `em_chunk`. 0
        skips the bootstrap (a single fit on the full data).
    mc_samples : int, default=100_000
        Monte Carlo samples per bootstrap replicate for the integral; more
        reduce the MC error, at linear cost in time.
    max_components : int, default=20
        Largest number of components considered. The batched protocol fits
        them all, so keep this to a plausible ceiling for the data.
    n_folds : int, default=2
        Cross-validation folds; each should stay representative of the data.
    n_inits : int, default=3
        Initialisations per fold; more reduce the chance of a local optimum,
        at proportional cost (batched away on GPU).
    reg_covar : float, default=1e-6
        Constant added to every covariance diagonal to prevent singularities,
        relative to each variable's variance (the data are standardised; see
        Notes). 1e-6 is safe at float32, where GMM-MI's 1e-15 would
        regularise nothing.
    threshold_fit : float, default=1e-5
        EM convergence threshold on the change in mean per-sample
        log-likelihood (sklearn's `tol`).
    max_iter : int, default=500
        Maximum EM iterations per fit.
    seed : int, default=0
        Seed controlling initialisations, CV splits, bootstrap resampling, MC
        integration and gaussianise's tie-breaking. Bitwise reproducible on
        CPU; on CUDA see Notes, "Reproducibility".
    return_k : bool, default=False
        If True, also return the selected number of components.
    return_fit : bool, default=False
        If True, also return the selected model as a `FittedGMM` (refitted on
        the full data, warm-started like the bootstrap replicates from the
        CV-selected parameters), e.g. for plotting; see Notes for its units.
    device : str or torch.device, default='cpu'
        Torch device on which to fit and integrate: 'cpu', or 'cuda' /
        'cuda:i' / a torch.device of those types (other backends are
        rejected at entry).
    dtype : torch.dtype, default=torch.float64
        Working precision. float32 is validated and much faster on GPUs with
        limited fp64 throughput; float64 is the reference precision.
    em_chunk : int or 'auto', default='auto'
        EM replicates (CV fits or bootstrap realisations) per batch; the
        initialisation and validation-LL stages batch the same way. Larger is
        faster on GPU at higher peak memory. 'auto' is 8 on CPU and fills half
        the free device memory on CUDA (see Notes).
    mc_chunk : int or 'auto', default='auto'
        Bootstrap replicates whose MC integration is evaluated per batch;
        'auto' as for `em_chunk`. The MC samples do not depend on it; results
        are bitwise invariant on CPU and move at reduction order on CUDA.
    preprocess : {'none', 'gaussianise'}, default='none'
        'gaussianise' applies a per-variable rank-probit transform before
        fitting, which leaves MI, cMI and TC unchanged but can make the
        density much easier to fit. Not for heavily tied data, and rejected
        for the KL quantities (see Notes).
    tie_rank : {'random', 'average'}, default='random'
        How gaussianise ranks tied values: 'random' breaks each column's ties
        with a seeded random ordering; 'average' is the deterministic scipy
        rankdata convention. See Notes.
    init_type : str, default='random_responsibilities'
        GMM initialisation scheme (torch ports of GMM-MI's initialisations):
        'random_responsibilities', 'minmax', 'kmeans' or 'randomized_kmeans'.
        'random_responsibilities' is cheap and adequate for easy fits; the
        k-means variants are more robust on well-separated multimodal data.
        See Notes for their memory at large N.
    scale : float, optional
        Mean-perturbation scale for 'randomized_kmeans' (GMM-MI's default 1
        when None), or the initial isotropic standard deviation for 'minmax'
        (GMM-MI's n-ball rule when None); in standardised units.
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
        If set, the EM and validation-LL kernels stream over sample chunks of
        this size, bounding their peak memory by em_chunk * n_chunk rather
        than N, and the default/k-means initialisation streams too. Results
        agree with the single pass to reduction order. See Notes.

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

    Raises
    ------
    ValueError
        An unknown option, an out-of-range count, threshold, dtype or device,
        data of the wrong shape or with NaN/inf values, or folds too small for
        the data; raised before any fitting. The full list: Notes.
    TypeError
        A non-integer count, seed, em_chunk or mc_chunk (numpy integers are
        accepted), or a non-numeric threshold, reg_covar or `scale`.
    RuntimeError
        On CUDA, when not even one replicate fits in memory (set `n_chunk`) or
        an out-of-memory error persists at one replicate per batch; when no
        candidate k yields a finite validation metric; or when a covariance
        cannot be factorised (raise `reg_covar`, or use float64).

    Warns
    -----
    RuntimeWarning
        For a result that may need a look: fewer than 100 samples, tied data
        or return_fit under gaussianise, components that died in a fixed-k
        fit, a skipped k in the sequential protocol, replicates that hit
        `max_iter` or finished on the legacy kernels, a CUDA out-of-memory
        retry, or extra covariance regularisation. The full list, with how
        often each is issued: Notes.

    Notes
    -----
    On selection, standardisation, preprocessing, initialisation, memory,
    reproducibility, and the full lists of errors and warnings:
    docs/estimate_notes.md in the repository.
    """
    X = torch.as_tensor(_as_array(X)).detach()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64; "
                         f"found {dtype}")
    device = _validate_device(device)
    # entry pipeline at float64; design notes: estimate, float64 entry
    # a private float64 copy (copy=True also for float64 input, whose
    # storage torch.as_tensor shares with the caller) that the
    # standardisation below then updates in place
    X = X.to(torch.float64, copy=True)
    if quantity not in _INTEGRANDS:
        raise ValueError(f"quantity must be one of "
                         f"{sorted(_INTEGRANDS)}; found '{quantity}'")
    if protocol not in ('batched', 'sequential'):
        raise ValueError(f"protocol must be 'batched' or 'sequential'; "
                         f"found '{protocol}'")
    (seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
     max_iter, fixed_components_number, patience,
     n_chunk) = _validate_entry_args(
        seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
        max_iter, fixed_components_number, reg_covar, threshold_components,
        threshold_fit, patience, n_chunk, mc_chunk=mc_chunk)
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
            "X contains non-finite values (NaN or inf); gaussianise would "
            "silently rank them as extreme values and the standardisation "
            "path propagates them into every replicate — remove or impute "
            "them first")
    _validate_folds(X.shape[0], n_folds, d=X.shape[1])
    if tie_rank not in ('random', 'average'):
        raise ValueError(f"tie_rank must be 'random' or 'average'; "
                         f"found '{tie_rank}'")
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
        X = _rank_gaussianise(
            X, tie_rank=tie_rank,
            generator=torch.Generator().manual_seed(seed + _TIE_SEED_OFFSET))
    elif preprocess != 'none':
        raise ValueError(f"preprocess must be 'none' or 'gaussianise'; "
                         f"found '{preprocess}'")
    # affine standardisation; design notes: estimate, standardisation
    if quantity in ('KL_forward', 'KL_reverse'):
        col_mean = X.mean().expand(X.shape[1]).contiguous()
        col_std = X.std().expand(X.shape[1]).contiguous()
    else:
        col_mean = X.mean(0)
        col_std = X.std(0)
    col_std = torch.where(col_std > 0, col_std, torch.ones_like(col_std))
    # in place on the private copy: elementwise identical to
    # (X - col_mean) / col_std, without the two further full-size copies
    X.sub_(col_mean).div_(col_std)
    # the float64 col_mean/col_std are kept for the return_fit back-transform;
    # the fit itself runs at the working dtype on the target device
    X = X.to(dtype=dtype, device=device)
    if not torch.isfinite(X).all():
        raise ValueError(
            f"X is not representable at {dtype} after standardisation "
            f"(a column's scale lies beyond the working dtype's range); "
            f"use dtype=torch.float64")
    N, d = X.shape
    # the hyperparameters the selection stage reads, as two plain dicts
    selection = {'protocol': protocol,
                 'fixed_components_number': fixed_components_number,
                 'max_components': max_components, 'n_folds': n_folds,
                 'n_inits': n_inits, 'metric_method': metric_method,
                 'threshold_components': threshold_components,
                 'patience': patience}
    fit = {'reg_covar': reg_covar, 'threshold_fit': threshold_fit,
           'max_iter': max_iter, 'em_chunk': em_chunk, 'n_chunk': n_chunk,
           'init_type': init_type, 'scale': scale, 'seed': seed}
    k_best, (w1, mu1, cov1) = _select_and_fit(X, selection, fit)

    B = max(n_bootstrap, 1)
    K = w1.shape[0]
    em_chunk_boot = _resolve_chunk(
        em_chunk, B, _em_bytes_per_replicate(N, K, d, dtype, n_chunk), device,
        fixed_bytes=_em_fixed_bytes(N, d, dtype, n_chunk), name='em_chunk')
    mc_chunk = _resolve_chunk(
        mc_chunk, B, _mc_bytes_per_replicate(mc_samples, K, d, dtype), device,
        name='mc_chunk')
    if return_fit:
        s1 = torch.ones(1, N, dtype=X.dtype, device=X.device)
        wf, muf, covf, _ = em_fit(X, s1, w1.unsqueeze(0), mu1.unsqueeze(0),
                                  cov1.unsqueeze(0), reg_covar, threshold_fit,
                                  max_iter, em_chunk=em_chunk_boot, n_chunk=n_chunk)
        alive = wf[0] > 0
        # factor, then back-transform; design notes: estimate, return_fit units
        chol_s, cov_s = _chol(covf[0][alive].unsqueeze(0))
        D = col_std.to(device=device)                 # float64 scales
        mean64 = col_mean.to(device=device)
        fitted = FittedGMM(wf[0][alive].double(),
                           muf[0][alive].double() * D + mean64,
                           cov_s[0].double() * (D[:, None] * D[None, :]),
                           chol=chol_s[0].double() * D[:, None])
    g = torch.Generator(device=X.device).manual_seed(
        seed + _BOOTSTRAP_SEED_OFFSET)
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
    mc = functools.partial(_mc_stage, w=w, mu=mu, cov=cov, chol=chol,
                           quantity=quantity, mc_samples=mc_samples,
                           seed=seed, n_replicates=B)
    vals = torch.cat(_run_chunked(B, mc_chunk, mc))               # (B,)
    mean = vals.mean().item()
    std = vals.std(unbiased=True).item() if n_bootstrap >= 2 else None
    out = (mean, std)
    if return_k:
        out += (k_best,)
    if return_fit:
        out += (fitted,)
    return out


class CategoryFit(NamedTuple):
    """
    Bootstrap fits of one category in `estimate_categorical` (a tuple).

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Fitted parameters over the B bootstrap replicates, shapes (B, K),
        (B, K, 1), (B, K, 1, 1); `cov` as repaired by `_chol`.
    chol : torch.Tensor
        Cholesky factors of `cov`, shape (B, K, 1, 1).
    """
    w: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    chol: torch.Tensor


def _mc_stage_categorical(sl, ci, fits, pi, logpi, mc_samples, seed,
                         n_replicates):
    """
    Monte Carlo term of category `ci` for one chunk of bootstrap replicates.

    The chunk callback of `estimate_categorical`'s MC stage: samples from
    category ci's fit and returns pi[ci] * E[log p(x|ci) - log pbar(x)],
    with pbar(x) = sum_c' pi[c'] p(x|c'). Per-(category, replicate) seeded
    streams, as in `_mc_stage`; the category stride keeps the streams
    collision-free for any n_bootstrap the entry validation admits.

    Parameters
    ----------
    sl : slice
        Replicate range handed over by `_run_chunked`.
    ci : int
        Index of the category sampled from.
    fits : list of CategoryFit
        Bootstrap fits of every category, each over B replicates.
    pi, logpi : torch.Tensor
        Category weights and their logarithms, shape (C,).
    mc_samples : int
        Monte Carlo samples per replicate.
    seed : int
        Base seed; replicate r of category ci uses
        seed + _MC_SEED_OFFSET + ci * _MC_CAT_STRIDE + r.
    n_replicates : int
        Total number of replicates B (the slice is clipped to it).

    Returns
    -------
    vals : torch.Tensor
        Category ci's contribution per replicate in the chunk, shape (b,),
        in nats.
    """
    w, mu, chol = fits[ci].w, fits[ci].mu, fits[ci].chol
    C = len(fits)
    pts = torch.cat([
        gmm_sample(w[r:r + 1], mu[r:r + 1], chol[r:r + 1],
                   mc_samples,
                   generator=torch.Generator(device=w.device)
                   .manual_seed(seed + _MC_SEED_OFFSET
                                + ci * _MC_CAT_STRIDE + r))
        for r in range(sl.start, min(sl.stop, n_replicates))])
    # running logaddexp over categories; design notes: _mc_stage_categorical
    log_pbar = own = None
    for cj in range(C):
        sc = gmm_score(pts, fits[cj].w[sl], fits[cj].mu[sl],
                       fits[cj].chol[sl])                            # (b,S)
        if cj == ci:
            own = sc
        sc = sc + logpi[cj]
        log_pbar = sc if log_pbar is None \
            else torch.logaddexp(log_pbar, sc)
    return pi[ci] * (own - log_pbar).mean(1)


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
    """
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

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, 1)
        The continuous variable.
    Y : array-like of shape (n_samples,)
        The categorical variable; categories are its unique values. Numeric
        labels are used as-is; string/object labels are factorised through
        numpy's unique (categories in sorted label order). Non-finite
        (NaN/inf) and missing (None/NaN) labels are rejected — torch.unique
        would otherwise make every NaN its own category.
    protocol : {'batched', 'sequential'}, default='batched'
        As in `estimate`, applied per category.
    n_bootstrap, mc_samples : int, optional
        As in `estimate`, applied per category.
    max_components : int, default=10
        Largest number of components considered per category's 1D fit.
    n_folds, n_inits : int, optional
        As in `estimate`, applied per category.
    reg_covar : float, default=1e-6
        As in `estimate`; X is standardised once over all categories, so the
        floor is relative to the variance of X as a whole.
    threshold_fit : float, default=1e-5
        As in `estimate`.
    max_iter, seed : int, optional
        As in `estimate`.
    return_k : bool, default=False
        If True, also return the selected component count per category.
    device : str or torch.device, default='cpu'
        As in `estimate`.
    dtype : torch.dtype, default=torch.float64
        As in `estimate`.
    em_chunk, mc_chunk : int or 'auto', optional
        As in `estimate`.
    init_type : str, default='random_responsibilities'
        As in `estimate`, applied per category.
    scale : float, optional
        As in `estimate`, applied per category.
    fixed_components_number : int, optional
        As in `estimate`, applied per category.
    metric_method : {'valid', 'aic', 'bic'}, default='valid'
        As in `estimate`, applied per category.
    threshold_components : float, default=1e-5
        As in `estimate`, applied per category.
    patience : int, default=1
        As in `estimate`, applied per category.
    category_weights : {'empirical', 'uniform'}, default='empirical'
        'empirical' sets pi_c = N_c / N from the observed category counts —
        the correct weighting for MI of the observed joint (X, Y).
        'uniform' sets pi_c = 1/C, reproducing the reference code, whose
        Appendix B formula assumes equiprobable categories.
    n_chunk : int, optional
        As in `estimate`, applied per category.

    Returns
    -------
    mean : float
        Bootstrap-mean MI in nats.
    std : float or None
        Bootstrap standard deviation (None when n_bootstrap < 2:
        one replicate has no sample deviation).
    k : list of int
        Selected component count per category; only if `return_k` is True.

    Raises
    ------
    ValueError
        Before any fitting, for: `X` not 1-D or a single column, with
        NaN/inf values, or not representable at `dtype` after
        standardisation; `Y` not 1-D, of a different length from `X`, with
        NaN/inf or missing (None/NaN) labels, or mixing label types that
        cannot be ordered; a smallest category too small for `n_folds` (as
        `estimate` checks X, with d = 1); an unknown `category_weights`; and
        those of `estimate`'s argument errors that apply to this signature
        (full list: docs/estimate_notes.md, "Errors and warnings").
    TypeError
        As in `estimate`.
    RuntimeError
        As in `estimate`, for any category's fit.

    Warns
    -----
    RuntimeWarning
        As in `estimate`, for each category's selection, bootstrap and
        Monte-Carlo stages; the small-sample warning checks the smallest
        category, and the gaussianise and return_fit warnings do not
        arise.
    """
    X = torch.as_tensor(_as_array(X)).detach()
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be torch.float32 or torch.float64; "
                         f"found {dtype}")
    device = _validate_device(device)
    # as in estimate: the entry pipeline runs at float64 and the cast to the
    # working dtype happens after the standardisation (design notes:
    # estimate, float64 entry)
    X = X.to(torch.float64, copy=True)
    if X.dim() > 2 or (X.dim() == 2 and X.shape[1] != 1):
        raise ValueError(f"X must be 1-D or a single column; found shape "
                         f"{tuple(X.shape)}")
    X = X.reshape(-1, 1)
    if not torch.isfinite(X).all():
        raise ValueError(
            "X contains non-finite values (NaN or inf); the "
            "standardisation propagates them into every replicate — "
            "remove or impute them first")
    cat_labels = None
    try:
        Y = torch.as_tensor(_as_array(Y), device=device).detach()
    except (TypeError, ValueError):
        # labels via np.unique; design notes: estimate_categorical, labels
        import numpy as np
        arr = np.asarray(Y)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"Y must be 1-D; found shape "
                             f"{arr.shape}") from None
        # pandas hands out object arrays with None/NaN for missing labels;
        # np.unique would die comparing str with float — reject them as
        # the numeric path rejects non-finite labels
        missing = [v is None or (isinstance(v, float) and v != v)
                   or type(v).__name__ in ('NAType', 'NaTType') for v in arr]
        if any(missing):
            raise ValueError(
                f"Y contains {sum(missing)} missing label(s) (None/NaN); "
                f"remove those samples or encode missingness as an explicit "
                f"category")
        try:
            cats_np, inv = np.unique(arr, return_inverse=True)
        except TypeError as e:
            raise ValueError(f"Y mixes label types that cannot be ordered "
                             f"together ({e}); use one type per column") \
                from None
        cat_labels = [str(v) for v in cats_np]
        Y = torch.as_tensor(inv.reshape(-1), device=device)
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
     max_iter, fixed_components_number, patience,
     n_chunk) = _validate_entry_args(
        seed, n_bootstrap, mc_samples, max_components, n_folds, n_inits,
        max_iter, fixed_components_number, reg_covar, threshold_components,
        threshold_fit, patience, n_chunk, mc_chunk=mc_chunk)
    # shared affine standardisation of the continuous variable: MI with a
    # categorical variable is invariant, and reg_covar becomes scale-free
    x_std = X.std()
    X.sub_(X.mean()).div_(torch.where(x_std > 0, x_std, torch.ones_like(x_std)))
    X = X.to(dtype=dtype, device=device)
    if not torch.isfinite(X).all():
        raise ValueError(
            f"X is not representable at {dtype} after standardisation "
            f"(its scale lies beyond the working dtype's range); "
            f"use dtype=torch.float64")
    cats, counts = torch.unique(Y, return_counts=True)
    C = cats.shape[0]
    ci_min = int(counts.argmin())
    label_min = (cat_labels[int(cats[ci_min])] if cat_labels is not None
                 else cats[ci_min].item())
    _validate_folds(int(counts[ci_min]), n_folds,
                    what=f"the smallest category (Y == {label_min})", d=1)
    if category_weights == 'empirical':
        pi = counts.to(dtype) / Y.shape[0]
    elif category_weights == 'uniform':
        pi = torch.full((C,), 1.0 / C, dtype=dtype, device=X.device)
    else:
        raise ValueError("category_weights must be 'empirical' or 'uniform'")
    logpi = torch.log(pi)
    B = max(n_bootstrap, 1)
    # the hyperparameters the selection stage reads, as two plain dicts
    selection = {'protocol': protocol,
                 'fixed_components_number': fixed_components_number,
                 'max_components': max_components, 'n_folds': n_folds,
                 'n_inits': n_inits, 'metric_method': metric_method,
                 'threshold_components': threshold_components,
                 'patience': patience}
    fit = {'reg_covar': reg_covar, 'threshold_fit': threshold_fit,
           'max_iter': max_iter, 'em_chunk': em_chunk, 'n_chunk': n_chunk,
           'init_type': init_type, 'scale': scale, 'seed': seed}
    fits, k_bests = [], []
    for ci in range(C):
        Xc = X[Y == cats[ci]]
        Nc = Xc.shape[0]
        # per-category seed; design notes: estimate_categorical, seeds
        seed_c = seed + _CATEGORY_SEED_STRIDE * ci
        # each category's selection runs on its own seed
        k_c, (w1, mu1, cov1) = _select_and_fit(Xc, selection,
                                               {**fit, 'seed': seed_c})
        k_bests.append(k_c)
        g = torch.Generator(device=X.device).manual_seed(
            seed_c + _BOOTSTRAP_SEED_OFFSET)
        if n_bootstrap >= 1:
            s = _bootstrap_counts(Nc, B, X.dtype, X.device, g)
        else:
            s = torch.ones(1, Nc, dtype=X.dtype, device=X.device)
        K = w1.shape[0]
        em_chunk_b = _resolve_chunk(
            em_chunk, B, _em_bytes_per_replicate(Nc, K, 1, dtype, n_chunk), device,
            fixed_bytes=_em_fixed_bytes(Nc, 1, dtype, n_chunk), name='em_chunk')
        w0, mu0, cov0 = _warm_start(w1, mu1, cov1, B)
        w, mu, cov, _ = em_fit(Xc, s, w0, mu0, cov0, reg_covar, threshold_fit, max_iter,
                               em_chunk=em_chunk_b, n_chunk=n_chunk)
        chol, cov = _chol(cov)
        fits.append(CategoryFit(w, mu, cov, chol))

    K_max = max(f.w.shape[1] for f in fits)
    # the per-category MC stage holds three (b,S) tensors beyond the
    # d=1 integrand (running logaddexp over categories, no (C,b,S) stack)
    mc_chunk = _resolve_chunk(
        mc_chunk, B, _mc_bytes_per_replicate(mc_samples, K_max, 1, dtype)
        + 4 * mc_samples * dtype.itemsize, device, name='mc_chunk')
    vals = torch.zeros(B, dtype=X.dtype, device=X.device)
    for ci in range(C):
        stage = functools.partial(_mc_stage_categorical, ci=ci, fits=fits,
                                  pi=pi, logpi=logpi, mc_samples=mc_samples,
                                  seed=seed, n_replicates=B)
        vals += torch.cat(_run_chunked(B, mc_chunk, stage))
    mean = vals.mean().item()
    std = vals.std(unbiased=True).item() if n_bootstrap >= 2 else None
    return (mean, std, k_bests) if return_k else (mean, std)
