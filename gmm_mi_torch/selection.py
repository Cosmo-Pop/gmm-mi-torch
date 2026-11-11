"""Component selection: the cross-validation masks and the sequential and
batched protocols.
"""

import math
import warnings

import torch

from .em import _init_params, em_fit, val_ll


# ----------------------------------------------------------------------
# component selection: the two protocols
# ----------------------------------------------------------------------

# CV fold assignment uses the caller's seed plus this fixed offset, keeping
# the fold stream decoupled from the init streams (seed + K) and the
# bootstrap stream (seed + 10_000).
_CV_MASK_SEED_OFFSET = 30_000


# small-sample warning threshold; design notes: _SMALL_SAMPLE_WARN
_SMALL_SAMPLE_WARN = 100


def _validate_folds(N, n_folds, what='X', d=None, stacklevel=4):
    """Reject CV settings that produce empty or degenerate folds up front:
    they otherwise surface as 0/0 NaN metrics and far-away index errors.

    With `d` given, also reject sizes whose smallest TRAINING fold holds
    fewer than d+1 samples — a full covariance cannot be estimated from
    them, every component sits on the reg_covar floor, and the estimate
    is meaningless (MI ~ 5-6 nats from N=2-3 points) — and warn below
    _SMALL_SAMPLE_WARN samples."""
    N, n_folds = int(N), int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (each fit needs a held-out "
                         f"validation fold); found {n_folds}")
    if N < n_folds:
        raise ValueError(f"n_folds={n_folds} exceeds the {N} samples "
                         f"in {what}")
    if d is None:
        return
    n_train_min = N - (-(-N // n_folds))      # N minus the largest fold
    if n_train_min < d + 1:
        raise ValueError(
            f"{what} has {N} samples: with n_folds={n_folds} the smallest "
            f"training fold holds {n_train_min}, fewer than the d+1={d + 1} "
            f"needed to estimate a {d}-D covariance, so every fit would sit "
            f"on the reg_covar floor and the estimate would be meaningless")
    if N < _SMALL_SAMPLE_WARN:
        warnings.warn(
            f"{what} has only {N} samples: GMM MI estimates carry a "
            f"positive finite-sample bias that the bootstrap errorbar does "
            f"not include, and the component selection is unreliable at "
            f"this size (the validation campaign started at N ~ 1e3)",
            RuntimeWarning, stacklevel=stacklevel)


def _validate_device(device):
    """Resolve `device` to a torch.device and fail at entry, under the
    argument's own name, for strings torch cannot parse or CUDA devices
    that are not there — instead of deep inside the first transfer."""
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"device must be a torch.device or a device "
                         f"string such as 'cpu', 'cuda' or 'cuda:1'; found "
                         f"{device!r} ({e})") from None
    if dev.type not in ('cpu', 'cuda'):
        raise ValueError(f"device={device!r}: only 'cpu' and 'cuda[:i]' "
                         f"devices are supported (the estimator has not been "
                         f"run on other backends, and torch would fail on an "
                         f"absent one only deep inside the fit)")
    if dev.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError(f"device={device!r} requested but CUDA is not "
                             f"available to this torch build/process")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise ValueError(f"device={device!r}: only "
                             f"{torch.cuda.device_count()} CUDA device(s) "
                             f"are visible")
    return dev


def _cv_masks(N, n_folds, n_inits, dtype, seed=42, device='cpu'):
    """0/1 train masks and val masks, shape (n_inits, n_folds, N), as
    expand() views over the inits of one (n_folds, N) mask pair living on
    `device` — em_fit/val_ll gather the rows each replicate chunk needs, so
    the tiling over inits (and over candidate k in select_batched) costs
    nothing. The base masks are moved to the device BEFORE expanding:
    .to() on an expanded view would materialise it.

    Factories are pinned to the CPU: the masks are built with a CPU
    generator, and torch.set_default_device('cuda') must not reroute them
    mid-construction."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g, device='cpu')
    fold_id = torch.zeros(N, dtype=torch.long, device='cpu')
    for f in range(n_folds):
        fold_id[perm[f::n_folds]] = f
    val = torch.stack([(fold_id == f).to(dtype) for f in range(n_folds)])
    train = (1.0 - val).to(device)
    val = val.to(device)
    return (train.unsqueeze(0).expand(n_inits, n_folds, N),
            val.unsqueeze(0).expand(n_inits, n_folds, N))


def _cv_fit_for_k(X, K, n_folds, n_inits, fit, init_chunk=None):
    """CV fits for a single k, batched over inits x folds. `fit` is the
    dict of fit settings (keys reg_covar, threshold_fit, max_iter,
    em_chunk, n_chunk, init_type, scale, seed; em_chunk an integer here).
    `init_chunk` (default fit['em_chunk']) batches the initialisation
    stage, whose peak per replicate differs from the EM step's (see
    _init_bytes_per_replicate)."""
    g = torch.Generator(device=X.device).manual_seed(fit['seed'] + K)
    train, val = _cv_masks(X.shape[0], n_folds, n_inits, X.dtype,
                           seed=fit['seed'] + _CV_MASK_SEED_OFFSET,
                           device=X.device)
    R = n_inits * n_folds
    w0, mu0, cov0 = _init_params(X, K, R, fit['reg_covar'], generator=g,
                                 init_type=fit['init_type'],
                                 scale=fit['scale'],
                                 em_chunk=fit['em_chunk'] if init_chunk is None
                                 else init_chunk, n_chunk=fit['n_chunk'])
    w, mu, cov, _ = em_fit(X, train, w0, mu0, cov0, fit['reg_covar'],
                           fit['threshold_fit'], fit['max_iter'],
                           em_chunk=fit['em_chunk'], n_chunk=fit['n_chunk'])
    v = val_ll(X, val, w, mu, cov, n_chunk=fit['n_chunk'],
               em_chunk=fit['em_chunk'])                  # (I*F,)
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
                      n_chunk=None, init_chunk=None):
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
    n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed
        As in `estimate`.
    em_chunk, init_type, scale, n_chunk
        As in `estimate`.
    init_chunk : int, optional
        Replicates per batch for the initialisation stage (default
        em_chunk); `estimate` sizes it from the init stage's own memory
        model.

    Returns
    -------
    k_best : int
        Selected number of components.
    params : tuple of torch.Tensor
        (weights, means, covariances) of the best CV fit at k_best.
    history : dict
        Per-k (metric, params) from the search.

    Raises
    ------
    ValueError
        n_folds < 2 or more folds than samples (before any fit); an
        unknown `init_type` (at the first initialisation).
    RuntimeError
        When no k yields a finite validation metric; and as in `em_fit`
        (a covariance that cannot be factorised, a CUDA out-of-memory
        error at one replicate per batch).
    """
    _validate_folds(X.shape[0], n_folds)
    fit = {'reg_covar': reg_covar, 'threshold_fit': threshold_fit,
           'max_iter': max_iter, 'em_chunk': em_chunk, 'n_chunk': n_chunk,
           'init_type': init_type, 'scale': scale, 'seed': seed}
    best_metric, best_k, pat, history = -torch.inf, None, 0, {}
    for K in range(1, max_components + 1):
        metric, params = _cv_fit_for_k(X, K, n_folds, n_inits, fit,
                                       init_chunk=init_chunk)
        history[K] = (metric, params)
        if not math.isfinite(metric):
            # a diverged fit at this k: skip it without consuming patience
            # (counting it as a non-improvement could end the search with
            # higher k untried)
            warnings.warn(f"sequential selection: the validation metric at "
                          f"k={K} is not finite; k={K} skipped without "
                          f"consuming patience", RuntimeWarning, stacklevel=2)
            continue
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
                   metric_method='valid', n_chunk=None, init_chunk=None):
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
    n_folds, n_inits, reg_covar, threshold_fit, max_iter, seed
        As in `estimate`.
    em_chunk, init_type, scale, n_chunk
        As in `estimate`.
    init_chunk : int, optional
        Replicates per batch for the initialisation stage (default
        em_chunk), as in `select_sequential`.

    Returns
    -------
    k_best : int
        Selected number of components.
    params : tuple of torch.Tensor
        (weights, means, covariances) of the best CV fit at k_best.
    history : dict
        Per-k metrics; params entries are None for every k (the selected
        parameters are returned separately).

    Raises
    ------
    ValueError
        n_folds < 2 or more folds than samples (before any fit); an
        unknown `init_type` (at the initialisation); an unknown
        `metric_method` (only after the CV fit and scoring).
    RuntimeError
        When no k yields a finite validation metric (or, for 'aic'/'bic',
        a finite criterion); and as in `em_fit` (a covariance that cannot
        be factorised, a CUDA out-of-memory error at one replicate per
        batch).
    """
    N, d = X.shape
    _validate_folds(N, n_folds)
    train, val = _cv_masks(N, n_folds, n_inits, X.dtype,
                           seed=seed + _CV_MASK_SEED_OFFSET, device=X.device)
    IF = n_inits * n_folds
    R = max_components * IF
    # (K, I, F, N) views of the (F, N) masks: em_fit/val_ll gather one
    # (chunk, N) slice per replicate chunk, so the 2(K+1)*I*F*N*itemsize
    # of repeated masks is never materialised
    train_r = train.unsqueeze(0).expand(max_components, n_inits, n_folds, N)
    val_r = val.unsqueeze(0).expand(max_components, n_inits, n_folds, N)
    active_k = torch.arange(1, max_components + 1, device=X.device).repeat_interleave(IF)
    g = torch.Generator(device=X.device).manual_seed(seed)
    w0, mu0, cov0 = _init_params(X, max_components, R, reg_covar, generator=g,
                                 active_k=active_k, init_type=init_type,
                                 scale=scale,
                                 em_chunk=em_chunk if init_chunk is None
                                 else init_chunk, n_chunk=n_chunk)
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
    # a non-finite metric at ONE k (a replicate that diverged) must not
    # poison the rule: torch.max/argmax/argmin propagate NaN, which would
    # end in the parsimony mask's IndexError or a silently selected NaN k
    finite_k = torch.isfinite(metric_k)
    if metric_method == 'valid':
        # parsimony rule: smallest FINITE k within `threshold_components`
        # of the best finite metric
        best = metric_k[finite_k].max()
        k_best = int((finite_k & (metric_k >= best - threshold_components))
                     .nonzero()[0].item()) + 1
    elif metric_method in ('aic', 'bic'):
        # best FINITE (init, fold) per k, refit on the full data from those
        # params (a k whose every init failed keeps a NaN metric and is
        # masked out of the criterion below)
        vm = v.mean(-1)
        best_i_k = torch.where(torch.isfinite(vm), vm,
                               torch.full_like(vm, -torch.inf)).argmax(-1)
        best_f_k = v[torch.arange(max_components, device=v.device),
                     best_i_k].argmax(-1)
        rk = torch.arange(max_components, device=X.device) * IF \
            + best_i_k * n_folds + best_f_k
        ones = torch.ones(1, N, dtype=X.dtype, device=X.device) \
            .expand(max_components, N)
        wf, muf, covf, _ = em_fit(X, ones, w[rk], mu[rk], cov[rk], reg_covar, threshold_fit,
                                  max_iter, em_chunk=em_chunk, n_chunk=n_chunk)
        # score the parameters the refit RETURNS: em_fit's ll lags them by
        # one M-step (see its docstring), which would shift the criterion
        # by up to 2 * N * threshold_fit
        llf = val_ll(X, ones, wf, muf, covf, n_chunk=n_chunk,
                     em_chunk=em_chunk)
        ks = torch.arange(1, max_components + 1, dtype=X.dtype, device=X.device)
        n_par = (ks - 1) + ks * d + ks * d * (d + 1) / 2
        pen = 2.0 if metric_method == 'aic' else math.log(N)
        crit = -2.0 * llf * N + pen * n_par
        ok = finite_k & torch.isfinite(crit)
        if not bool(ok.any()):
            raise RuntimeError(
                f"{metric_method} selection found no finite criterion at "
                f"any k; NaN metrics usually mean degenerate CV folds or data")
        crit = torch.where(ok, crit, torch.full_like(crit, torch.inf))
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
