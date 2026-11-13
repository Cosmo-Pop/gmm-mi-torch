"""Batched Gaussian machinery: Cholesky factorisation with repair,
component log-densities, mixture scores, sampling and analytic marginals.
"""

import warnings

import torch

from .constants import LOG2PI


# ----------------------------------------------------------------------
# batched Gaussian machinery
# ----------------------------------------------------------------------

def _chol(cov, return_repaired=False):
    """
    Cholesky of (R,K,d,d) covariances; returns (chol, cov).

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

    With return_repaired=True, a third element is returned: the (R,K) bool
    mask of components that needed escalation or a non-finite reset — the
    kernel-reroute detector uses it as corroborating evidence.

    Parameters
    ----------
    cov : torch.Tensor of shape (R, K, d, d)
        Covariances.
    return_repaired : bool, default=False
        Also return the mask of repaired components.

    Returns
    -------
    chol : torch.Tensor of shape (R, K, d, d)
        Lower Cholesky factors.
    cov : torch.Tensor of shape (R, K, d, d)
        The matrices actually factorised.
    repaired : torch.Tensor of shape (R, K)
        Bool mask of escalated or reset components; only with
        return_repaired=True.
    """
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    dead = cov.abs().sum(dim=(-2, -1)) < 1e-30            # (R,K)
    broken = ~torch.isfinite(cov).all(dim=(-2, -1))       # (R,K)
    safe = torch.where((dead | broken)[..., None, None], eye.expand_as(cov),
                       cov)
    if bool(dead.any()):
        # internally produced dead components are already exact identity
        # (the M-steps and inits finalise them so), so this absolute test
        # fires only on externally supplied matrices — say so
        warnings.warn(f"{int(dead.sum())} all-zero covariance(s) replaced "
                      f"by identity", RuntimeWarning, stacklevel=2)
    if bool(broken.any()):
        warnings.warn(f"{int(broken.sum())} non-finite covariance(s) "
                      f"reset to identity", RuntimeWarning, stacklevel=2)
    L, info = torch.linalg.cholesky_ex(safe)
    if not bool(info.any()):
        return (L, safe, broken) if return_repaired else (L, safe)
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
            return ((L, safe, repaired | broken) if return_repaired
                    else (L, safe))
        bump *= 10.0
    raise RuntimeError(
        f"covariance factorisation failed even after diagonal "
        f"regularisation up to {total:g} x the component's largest "
        f"absolute diagonal entry — a component has collapsed below what "
        f"this dtype can represent: raise reg_covar, or use "
        f"dtype=torch.float64")


def log_gauss(X, mu, chol):
    """
    Component-wise Gaussian log-densities.

    X: (R,N,d), mu: (R,K,d), chol: (R,K,d,d) lower.  Returns (R,N,K).

    Parameters
    ----------
    X : torch.Tensor of shape (R, N, d)
        Points, per replicate.
    mu : torch.Tensor of shape (R, K, d)
        Component means.
    chol : torch.Tensor of shape (R, K, d, d)
        Lower Cholesky factors of the component covariances.

    Returns
    -------
    torch.Tensor of shape (R, N, K)
        Log-density log N(x_n | mu_k, chol_k chol_k^T).
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
    """
    Mixture log-density of X under each replicate.

    Parameters
    ----------
    X : torch.Tensor of shape (R, N, d)
        Points, per replicate.
    w, mu, chol : torch.Tensor
        Mixture weights (R, K), means (R, K, d) and covariance Cholesky
        factors (R, K, d, d); zero-weight (dead) components are skipped.

    Returns
    -------
    torch.Tensor of shape (R, N)
        Log-density log sum_k w_k N(x_n | mu_k, chol_k chol_k^T).
    """
    logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                       torch.full_like(w, -torch.inf))
    return torch.logsumexp(logw.unsqueeze(1) + log_gauss(X, mu, chol), dim=-1)


def gmm_sample(w, mu, chol, n_samples, generator=None):
    """
    Draw samples from each replicate's mixture.

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
        The samples: a component drawn by weight, then a Gaussian draw.
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
    """
    Log-density of the selected columns under the analytic marginal GMM.

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
        The marginal mixture's log-density at the selected columns.
    """
    mu_m, cov_m = marginal_params(mu, cov, idx)
    chol_m, _ = _chol(cov_m)
    return gmm_score(X_full[..., list(idx)], w, mu_m, chol_m)
