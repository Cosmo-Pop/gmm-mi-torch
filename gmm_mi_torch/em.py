"""Weighted EM batched over a leading replicate dimension: the
GEMM-formulated kernels, the initialisations, `em_fit` and `val_ll`.
"""

import functools
import math
import warnings
from typing import NamedTuple

import torch

from .constants import LOG2PI, _EM_KERNEL, _INIT_DRAW_BLOCK, _MSTEP_BLOCK
from .precision import _full_precision
from .chunking import (
    _WeightChunks, _as_weight_chunks, _init_sample_chunk,
    _legacy_sample_chunk, _n_rows, _rows, _run_chunked)
from .gaussian import _chol, gmm_score, log_gauss


# ----------------------------------------------------------------------
# GEMM-formulated EM kernels
# ----------------------------------------------------------------------
# how they work and their limits; design notes: GEMM-formulated EM kernels


def _features(X):
    """
    Quadratic features of the samples, for the GEMM kernels.

    Quadratic features [x, x_i x_j (i<=j)] of X (..., d): (..., d + d(d+1)/2).

    Assembled in place — the output, one gathered operand copied into it,
    the second gathered operand multiplied in — so the transient is
    q + d(d+1)/2 units of n x itemsize; the products are bitwise those of
    a concatenated assembly.

    Parameters
    ----------
    X : torch.Tensor of shape (..., d)
        Samples.

    Returns
    -------
    torch.Tensor of shape (..., d + d(d+1)/2)
        The features F, in X's dtype.
    """
    d = X.shape[-1]
    iu = torch.triu_indices(d, d, device=X.device)
    F = torch.empty(*X.shape[:-1], d + iu.shape[1], dtype=X.dtype,
                    device=X.device)
    F[..., :d] = X
    F[..., d:] = X[..., iu[0]]
    F[..., d:].mul_(X[..., iu[1]])
    return F


def _tri_inv(L):
    """
    Batched inverse of lower-triangular matrices.

    Batched inverse of lower-triangular (..., d, d) by forward
    substitution in plain tensor ops. The library routines
    (cholesky_inverse, solve_triangular) are slow on some GPUs when batched
    over many small matrices; d^2 vectorised element operations are cheap
    at the small d used here.

    Parameters
    ----------
    L : torch.Tensor of shape (..., d, d)
        Lower-triangular matrices with non-zero diagonals.

    Returns
    -------
    torch.Tensor of shape (..., d, d)
        Their inverses (lower-triangular).
    """
    d = L.shape[-1]
    inv_diag = 1.0 / torch.diagonal(L, dim1=-2, dim2=-1)   # (...,d)
    Y = torch.zeros_like(L)
    for j in range(d):
        Y[..., j, j] = inv_diag[..., j]
        for i in range(j + 1, d):
            Y[..., i, j] = -(L[..., i, j:i] * Y[..., j:i, j]).sum(-1)                 * inv_diag[..., i]
    return Y


def _gauss_coef(w, mu, cov):
    """
    GEMM coefficients of the weighted log-density quadratic form.

    Returns coef (R,K,q), const (R,K) with
    log w_k N(x|mu_k, cov_k) = _features(x) . coef_k + const_k.
    The factorisation and inversion run at the native dtype; the quadratic
    form is then assembled at float64. The assembly is where precision
    matters: the expansion's terms cancel almost completely, so low-precision
    arithmetic there corrupts the log-density itself, whereas a small error
    in the precision matrix only evaluates the density of a slightly
    perturbed Gaussian.

    Parameters
    ----------
    w : torch.Tensor of shape (R, K)
        Component weights; zero-weight components get a -inf constant.
    mu : torch.Tensor of shape (R, K, d)
        Component means.
    cov : torch.Tensor of shape (R, K, d, d)
        Component covariances.

    Returns
    -------
    coef : torch.Tensor of shape (R, K, q)
        Coefficients of the features, q = d + d(d+1)/2.
    const : torch.Tensor of shape (R, K)
        Constant terms, including log w_k.
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
    """
    E-step on the GEMM kernels, single pass.

    Responsibilities (n_active,N,K) and weighted mean log-likelihood
    (n_active,).

    Parameters
    ----------
    F : torch.Tensor of shape (N, q)
        Quadratic features of the samples (`_features`).
    s : torch.Tensor of shape (n_active, N)
        Sample weights per replicate.
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (n_active, K), (n_active, K, d),
        (n_active, K, d, d).

    Returns
    -------
    resp : torch.Tensor of shape (n_active, N, K)
        Responsibilities.
    ll : torch.Tensor of shape (n_active,)
        Weighted mean per-sample log-likelihood of the parameters given.
    """
    coef, const = _gauss_coef(w, mu, cov)
    lj = torch.matmul(F.unsqueeze(0), coef.transpose(1, 2))
    lj += const.unsqueeze(1)                               # (n_active,N,K)
    m = lj.amax(-1, keepdim=True)
    lj.sub_(m).exp_()
    Z = lj.sum(-1, keepdim=True)
    ll_n = (m + Z.log()).squeeze(-1)
    resp = lj.div_(Z)
    ll = (s * ll_n).sum(1) / s.sum(1)
    return resp, ll


def _restore_floor(cov, reg_covar):
    """
    Restore the reg_covar floor on a finished M-step covariance, in place.

    Rounding of the (origin or shifted) moments can push a collapsed
    component's variance below reg_covar; the diagonal is clamped back. For
    the components WHOSE DIAGONAL WAS CLAMPED the off-diagonal roundoff
    residue (~u x O(1), i.e. ~1e-6 at float32) can then exceed the floor
    and leave [[reg, e], [e, reg]] indefinite beyond the reach of _chol's
    escalation (scaled to the diagonal, capped at ~1e3 x) — at float32 for
    reg_covar <= ~1e-9 on duplicated-point data. Those components'
    off-diagonals are bounded to the PSD-consistent
    |c_ij| <= sqrt(c_ii c_jj) (1 - 1e-6); every other component is copied
    verbatim, so data that never clamps is bitwise unaffected. At d >= 3
    the pairwise bound is necessary, not sufficient — the residue's signs
    can leave lambda_min ~ -reg_covar — but the matrix is then inside the
    escalation's reach (one warning), where the unbounded one is not.

    Parameters
    ----------
    cov : torch.Tensor of shape (..., d, d)
        Covariances, modified in place.
    reg_covar : float
        The floor.

    Returns
    -------
    torch.Tensor of shape (..., d, d)
        `cov` itself.
    """
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    clamped = (diag < reg_covar).any(-1)                        # (R,K)
    diag.clamp_(min=reg_covar)
    if bool(clamped.any()):
        d = cov.shape[-1]
        bound = (diag.unsqueeze(-1) * diag.unsqueeze(-2)).sqrt() * (1 - 1e-6)
        eye = torch.eye(d, dtype=torch.bool, device=cov.device)
        bounded = torch.where(eye, cov, torch.maximum(torch.minimum(cov, bound),
                                                      -bound))
        cov.copy_(torch.where(clamped[..., None, None], bounded, cov))
    return cov


def _mstep_from_moments(S, Nk, ssum, reg_covar, d, dt):
    """
    Finalise the M-step from accumulated float64 moments.

    Finalise the M-step from float64 moments S=(resp*s)^T F, Nk, s.sum(1).

    Parameters
    ----------
    S : torch.Tensor of shape (R, K, q)
        Weighted feature moments, float64.
    Nk : torch.Tensor of shape (R, K)
        Weighted responsibility sums, float64.
    ssum : torch.Tensor of shape (R,)
        Total sample weight per replicate.
    reg_covar : float
        Covariance floor.
    d : int
        Number of variables.
    dt : torch.dtype
        Working dtype of the results.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Weights (R, K), means (R, K, d) and covariances (R, K, d, d) at
        `dt`; dead components (Nk < 1e-10) get weight 0, mean 0 and
        identity covariance.
    """
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
    cov = _restore_floor(cov, reg_covar)
    w = torch.where(dead, torch.zeros_like(Nk), Nk / ssum.double().unsqueeze(-1))
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    m1 = torch.where(dead[..., None], torch.zeros_like(m1), m1)
    cov = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
    return w.to(dt), m1.to(dt), cov.to(dt)


def _moment_gemm(wr, F):
    """
    Weighted feature moments S = wr^T F, accumulated at float64.

    S = wr^T F accumulated at float64: one GEMM at float64 input, else
    block-wise float32 GEMMs reduced in float64 (accumulation error scales
    with sqrt(block) instead of sqrt(N), at plain-SGEMM cost).

    The float32 blocks are batched over BLOCKS only, with (k, r) folded
    into the GEMM's row dimension: one transient copy of wr[:, :N0] (a
    view at K = 1), and F is never broadcast (why: docs/design_notes.md).
    Per block the dot products are 512 float32 terms, the cross-block sum
    float64 terms.

    Parameters
    ----------
    wr : torch.Tensor of shape (R, N, K)
        Weighted responsibilities.
    F : torch.Tensor of shape (N, q)
        Quadratic features of the samples.

    Returns
    -------
    torch.Tensor of shape (R, K, q)
        The moments, float64.
    """
    R, N, K = wr.shape
    q = F.shape[-1]
    if F.dtype == torch.float64:
        return torch.matmul(wr.transpose(1, 2), F.unsqueeze(0))
    nb = N // _MSTEP_BLOCK
    S = torch.zeros(R, K, q, dtype=torch.float64, device=F.device)
    if nb > 0:
        N0 = nb * _MSTEP_BLOCK
        A = (wr[:, :N0].reshape(R, nb, _MSTEP_BLOCK, K)   # view
             .permute(1, 3, 0, 2)                        # (nb,K,R,b), view
             .reshape(nb, K * R, _MSTEP_BLOCK))          # the one copy
        S += (torch.bmm(A, F[:N0].reshape(nb, _MSTEP_BLOCK, q))  # (nb,K*R,q)
              .double().sum(0).view(K, R, q).transpose(0, 1))
    if N % _MSTEP_BLOCK:
        # the tail is a 3-D broadcast matmul that copies nothing (a
        # folded-row tail would add a copy of its own and change the
        # tail-only regime's summation order)
        N0 = nb * _MSTEP_BLOCK
        S += torch.matmul(wr[:, N0:].transpose(1, 2), F[N0:].unsqueeze(0)).double()
    return S


def _mstep_gemm(F, resp, s, reg_covar, d):
    """
    M-step on the GEMM kernels, single pass.

    M-step via resp^T F; consumes resp in place. float64 accumulation.

    Parameters
    ----------
    F : torch.Tensor of shape (N, q)
        Quadratic features of the samples.
    resp : torch.Tensor of shape (R, N, K)
        Responsibilities; overwritten with resp * s.
    s : torch.Tensor of shape (R, N)
        Sample weights.
    reg_covar : float
        Covariance floor.
    d : int
        Number of variables.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Updated weights (R, K), means (R, K, d) and covariances
        (R, K, d, d), in F's dtype.
    """
    wr = resp.mul_(s.unsqueeze(-1))
    S = _moment_gemm(wr, F)
    return _mstep_from_moments(S, wr.sum(1).double(), s.sum(1), reg_covar, d,
                               F.dtype)


def _em_step_gemm_nchunked(X, weight_chunks, w, mu, cov, reg_covar, n_chunk,
                           rows=None):
    """
    One EM iteration on the GEMM kernels, streamed over sample chunks.

    One GEMM EM iteration streamed over sample chunks (float64 moments).
    `weight_chunks` is a _WeightChunks accessor (a dense (R,N) tensor is
    wrapped); with `rows`, only those replicates take part, read one sample
    chunk at a time (an up-front s[rows] would be an (R,N) copy —
    N/n_chunk units of the streamed budget).

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    weight_chunks : _WeightChunks or torch.Tensor
        Sample weights, an accessor or a dense (R, N) tensor.
    w, mu, cov : torch.Tensor
        Parameters of the replicates taking part, shapes (n_active, K),
        (n_active, K, d), (n_active, K, d, d).
    reg_covar : float
        Covariance floor.
    n_chunk : int
        Samples per chunk.
    rows : torch.Tensor, optional
        Replicate rows of the accessor taking part; None for all.

    Returns
    -------
    ll : torch.Tensor of shape (n_active,)
        Weighted mean per-sample log-likelihood of the incoming parameters.
    w, mu, cov : torch.Tensor
        Parameters after the M-step, same shapes as the inputs.
    """
    weight_chunks = _as_weight_chunks(weight_chunks, n_chunk)
    n_active, N = weight_chunks.shape
    if rows is not None:
        n_active = rows.numel()
    d = X.shape[-1]
    K = w.shape[-1]
    dt, dev = X.dtype, X.device
    coef, const = _gauss_coef(w, mu, cov)
    q = coef.shape[-1]
    S = torch.zeros(n_active, K, q, dtype=torch.float64, device=dev)
    Nk = torch.zeros(n_active, K, dtype=torch.float64, device=dev)
    ll_sum = torch.zeros(n_active, dtype=torch.float64, device=dev)
    for i in range(0, N, n_chunk):
        Fc = _features(X[i:i+n_chunk])
        sc = weight_chunks.rows(i, n_chunk, rows)
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
        # not carried into the next chunk's allocation: the next chunk's
        # features would otherwise be assembled while these are still
        # bound (two feature blocks at once)
        del lj, wr, Fc
    # float64
    tot = (weight_chunks.totals() if rows is None
           else weight_chunks.totals()[rows])
    ll = (ll_sum / tot).to(dt)
    w_new, mu_new, cov_new = _mstep_from_moments(S, Nk, tot, reg_covar, d, dt)
    return ll, w_new, mu_new, cov_new


# ----------------------------------------------------------------------
# weighted batched EM
# ----------------------------------------------------------------------

def _init_draw_block(out, generator):
    """
    One block of the default init's responsibility draw.

    One block of the default init's responsibility draw (a seam the
    tests spy on to check the draw is identical across chunkings).

    Parameters
    ----------
    out : torch.Tensor
        Buffer filled in place with uniform draws.
    generator : torch.Generator
        The replicate's generator.
    """
    out.uniform_(generator=generator)


def _draw_init_blocks(r, gens):
    """
    Fill a responsibility buffer block by block from per-replicate generators.

    Samples [i, i + r.shape[1]) of each replicate are drawn in blocks of
    `_INIT_DRAW_BLOCK` from that replicate's own generator; the buffer starts
    and ends on whole blocks, so the block sequence is the single pass's for
    every replicate chunk and sample chunk.

    Parameters
    ----------
    r : torch.Tensor
        Output buffer, shape (c, n, K); written in place.
    gens : list of torch.Generator
        One generator per replicate in the chunk, already advanced to the
        first sample of this block range.

    Returns
    -------
    None
        `r` is filled in place.
    """
    for b in range(0, r.shape[1], _INIT_DRAW_BLOCK):
        for j, g in enumerate(gens):
            _init_draw_block(r[j, b:b + _INIT_DRAW_BLOCK], g)


def _normalise_responsibilities(r, active_k_chunk):
    """
    Mask dead components and normalise the responsibilities, in place.

    Parameters
    ----------
    r : torch.Tensor
        Raw draws, shape (c, n, K).
    active_k_chunk : torch.Tensor or None
        Active component count per replicate in the chunk, shape (c,);
        components >= active_k_chunk[j] are zeroed. None when every
        component is live.

    Returns
    -------
    r : torch.Tensor
        The same buffer, masked and normalised over the last axis.
    """
    if active_k_chunk is not None:
        K = r.shape[-1]
        comp = torch.arange(K, device=r.device).view(1, 1, K)
        r.mul_(comp < active_k_chunk.view(-1, 1, 1))
    r.div_(r.sum(-1, keepdim=True).clamp_min(1e-30))
    return r


def _init_mstep_random(sl, X, seeds, active_k, K, reg_covar, n_chunk):
    """
    Initial M-step from random responsibilities for one replicate chunk.

    The chunk callback of `_init_params`'s default init. Each replicate draws
    from its own generator, seeded from `seeds`, so the draw does not depend
    on the chunking. Single pass over the samples when `n_chunk` is None;
    otherwise two streamed passes over sample chunks of
    `_init_sample_chunk(N, n_chunk)` (means, then centred second moments)
    with float64 accumulators, the same algebra as the single-pass M-step.

    Parameters
    ----------
    sl : slice
        Replicate range handed over by `_run_chunked`.
    X : torch.Tensor
        Data, shape (N, d).
    seeds : list of int
        One generator seed per replicate, length R.
    active_k : torch.Tensor or None
        Active component count per replicate, shape (R,); None when all K
        components are live in every replicate.
    K : int
        Number of components (including dead padding).
    reg_covar : float
        Added to the covariance diagonals.
    n_chunk : int or None
        Sample chunk size for the streamed path; None for the single pass.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Initial parameters for the chunk, shapes (c, K), (c, K, d),
        (c, K, d, d).
    """
    N, d = X.shape
    dev, dt = X.device, X.dtype
    eye = torch.eye(d, dtype=dt, device=dev)
    nc = _init_sample_chunk(N, n_chunk)
    R = len(seeds)
    lo, hi = sl.start, min(sl.stop, R)
    c = hi - lo
    active_k_chunk = None if active_k is None else active_k[lo:hi]
    if n_chunk is None:
        r = torch.empty(c, N, K, dtype=dt, device=dev)
        gens = [torch.Generator(device=dev).manual_seed(seeds[lo + j])
                for j in range(c)]
        _draw_init_blocks(r, gens)
        return _m_step(X.expand(c, N, d),
                       _normalise_responsibilities(r, active_k_chunk),
                       torch.ones(1, N, dtype=dt, device=dev)
                       .expand(c, N), reg_covar)
    # streamed, two passes (the M-step centres on the NEW mean, so
    # one pass would have to use origin moments — beyond reduction
    # order on offset data); the draw is regenerated for pass 2
    Nk = torch.zeros(c, K, dtype=torch.float64, device=dev)
    S1 = torch.zeros(c, K, d, dtype=torch.float64, device=dev)
    for p in range(2):
        gens = [torch.Generator(device=dev).manual_seed(seeds[lo + j])
                for j in range(c)]
        if p == 1:
            mu = S1 / Nk.clamp_min(1e-10).unsqueeze(-1)       # (c,K,d)
            S2 = torch.zeros(c, K, d, d, dtype=torch.float64,
                             device=dev)
        for i in range(0, N, nc):
            Xc = X[i:i + nc]
            n = Xc.shape[0]
            r = torch.empty(c, n, K, dtype=dt, device=dev)
            _draw_init_blocks(r, gens)
            r = _normalise_responsibilities(r, active_k_chunk)
            if p == 0:
                Nk += r.sum(1).double()
                S1 += torch.einsum('rnk,nd->rkd', r, Xc).double()
            else:
                diff = Xc.view(1, n, 1, d) - mu.to(dt).unsqueeze(1)
                S2 += torch.einsum('rnk,rnki,rnkj->rkij', r, diff,
                                   diff).double()
            del r
    dead = Nk < 1e-10
    Nks = Nk.clamp_min(1e-10)
    cov = S2 / Nks[..., None, None] + reg_covar * eye.double()
    w = torch.where(dead, torch.zeros_like(Nk), Nk / N)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    mu = torch.where(dead[..., None], torch.zeros_like(mu), mu)
    cov = torch.where(dead[..., None, None], eye.double().expand_as(cov),
                      cov)
    return w.to(dt), mu.to(dt), cov.to(dt)


def _hard_assignments(Xc, x2c, Cc, al, big):
    """
    One-hot nearest-centre assignments for a block of samples.

    Parameters
    ----------
    Xc : torch.Tensor
        Samples, shape (n, d).
    x2c : torch.Tensor
        Their squared norms, shape (n,).
    Cc : torch.Tensor
        Centres per replicate, shape (c, K, d).
    al : torch.Tensor
        Live-component mask, shape (c, K); dead centres are never chosen.
    big : float
        Distance given to dead centres (the dtype's largest finite value).

    Returns
    -------
    onehot : torch.Tensor
        Assignments, shape (c, n, K), in the dtype of `Xc`.
    """
    K = Cc.shape[1]
    dt, dev = Xc.dtype, Xc.device
    d2 = (x2c.view(1, -1, 1) + (Cc ** 2).sum(-1).unsqueeze(1)
          - 2.0 * torch.einsum('nd,rkd->rnk', Xc, Cc))
    d2 = torch.where(al.unsqueeze(1), d2, torch.full_like(d2, big))
    lab = d2.argmin(-1)
    return torch.zeros(*lab.shape, K, dtype=dt, device=dev).scatter_(
        2, lab.unsqueeze(-1), 1.0)


def _finish_kmeans_params(w, mu, cov, Nk, al, eye):
    """
    Zero the weights of empty or dead components, renormalise, and give
    them harmless means and covariances.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Hard-assignment parameters, shapes (c, K), (c, K, d), (c, K, d, d).
    Nk : torch.Tensor
        Assigned counts per component, shape (c, K).
    al : torch.Tensor
        Live-component mask, shape (c, K).
    eye : torch.Tensor
        Identity, shape (d, d): the covariance of a dead component.

    Returns
    -------
    w, mu, cov : torch.Tensor
        The finished parameters, same shapes.
    """
    dead = (Nk < 1.0) | ~al
    w = torch.where(dead, torch.zeros_like(w), w)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
    mu = torch.where(dead[..., None], torch.zeros_like(mu), mu)
    cov = torch.where(dead[..., None, None], eye.expand_as(cov), cov)
    return w, mu, cov


def _kmeans_assign_stats(sl, X, x2, centres, alive, reg_covar, n_chunk):
    """
    k-means init: hard-assignment statistics on the full data for one
    replicate chunk.

    The chunk callback of `_init_params`'s 'kmeans' and 'randomized_kmeans'
    inits. Covariances are normalised by N, not by the component counts, as
    in GMM-MI's KmeansInit. Streamed in two passes over sample chunks of
    `_init_sample_chunk(N, n_chunk)` when `n_chunk` is set, with float64
    accumulators; the labels depend only on the centres, so both passes see
    identical assignments.

    Parameters
    ----------
    sl : slice
        Replicate range handed over by `_run_chunked`.
    X : torch.Tensor
        Data, shape (N, d).
    x2 : torch.Tensor
        Squared norms of the samples, shape (N,).
    centres : torch.Tensor
        Lloyd centres per replicate, shape (R, K, d).
    alive : torch.Tensor
        Live-component mask, shape (R, K).
    reg_covar : float
        Added to the covariance diagonals.
    n_chunk : int or None
        Sample chunk size for the streamed path; None for the single pass.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Initial parameters for the chunk, shapes (c, K), (c, K, d),
        (c, K, d, d).
    """
    N, d = X.shape
    dev, dt = X.device, X.dtype
    K = centres.shape[1]
    eye = torch.eye(d, dtype=dt, device=dev)
    big = torch.finfo(dt).max
    nc = _init_sample_chunk(N, n_chunk)
    Cc, al = centres[sl], alive[sl]
    c = Cc.shape[0]
    if n_chunk is None:
        oh = _hard_assignments(X, x2, Cc, al, big)
        Nk = oh.sum(1)
        w = Nk / N
        mu = torch.einsum('rnk,nd->rkd', oh, X) \
            / Nk.clamp_min(1.0)[..., None]
        diff = X.view(1, N, 1, d) - mu.unsqueeze(1)
        # GMM-MI normalises the k-means covariances by N, not by Nk
        cov = torch.einsum('rnk,rnki,rnkj->rkij', oh, diff, diff) / N \
            + reg_covar * eye
        return _finish_kmeans_params(w, mu, cov, Nk, al, eye)
    # streamed: labels depend only on the centres, so the two
    # passes see identical assignments
    Nk = torch.zeros(c, K, dtype=torch.float64, device=dev)
    Sx = torch.zeros(c, K, d, dtype=torch.float64, device=dev)
    for i in range(0, N, nc):
        oh = _hard_assignments(X[i:i + nc], x2[i:i + nc], Cc, al, big)
        Nk += oh.sum(1).double()
        Sx += torch.einsum('rnk,nd->rkd', oh, X[i:i + nc]).double()
    mu = (Sx / Nk.clamp_min(1.0)[..., None]).to(dt)
    Sdd = torch.zeros(c, K, d, d, dtype=torch.float64, device=dev)
    for i in range(0, N, nc):
        Xc = X[i:i + nc]
        oh = _hard_assignments(Xc, x2[i:i + nc], Cc, al, big)
        diff = Xc.view(1, -1, 1, d) - mu.unsqueeze(1)
        Sdd += torch.einsum('rnk,rnki,rnkj->rkij', oh, diff,
                            diff).double()
    cov = (Sdd / N).to(dt) + reg_covar * eye
    return _finish_kmeans_params((Nk / N).to(dt), mu, cov, Nk.to(dt), al,
                                 eye)


def _init_params(X, K, R, reg_covar, generator=None, active_k=None,
                 init_type='random_responsibilities', scale=None, em_chunk=8,
                 n_chunk=None):
    """
    Batched GMM parameter initialisation (torch ports of GMM-MI's inits).

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
    The responsibility draw and the initial M-step / assignment stats run
    in replicate chunks of `em_chunk`, under the same halve-and-retry CUDA
    out-of-memory guard as `em_fit`, and — with `n_chunk` — streamed over
    sample chunks of _init_sample_chunk(N, n_chunk) in two passes (means,
    then centred second moments: the same algebra as the single-pass
    M-step, with float64 accumulators). The draw is chunk-invariant: the
    default init seeds one generator per replicate from `generator` and
    draws in fixed blocks of _INIT_DRAW_BLOCK samples, so the draw is the
    same for every em_chunk and n_chunk on every device. The statistics
    computed from it are bitwise chunk-invariant on CPU and move at
    reduction order on CUDA, and the streamed moments agree with the
    single pass at reduction order.

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    K : int
        Number of components (the padded maximum when `active_k` is given).
    R : int
        Number of replicates.
    reg_covar : float
        Covariance floor.
    generator : torch.Generator, optional
        Device-matched RNG; every random draw of the init comes from it.
    active_k : torch.Tensor of shape (R,), optional
        Live component count per replicate; None means K for all.
    init_type : str, default='random_responsibilities'
        One of the four above.
    scale : float, optional
        Mean-push scale of 'randomized_kmeans' (default 1), and the scale
        of the minmax covariances (as in `estimate`).
    em_chunk : int, default=8
        Replicates per batch.
    n_chunk : int, optional
        Stream the samples in chunks of this size (rounded to draw blocks).

    Returns
    -------
    w, mu, cov : torch.Tensor
        Initial weights (R, K), means (R, K, d) and covariances
        (R, K, d, d).
    """
    N, d = X.shape
    dev, dt = X.device, X.dtype
    ak = (active_k if active_k is not None
          else torch.full((R,), K, device=dev, dtype=torch.long))
    alive = torch.arange(K, device=dev).view(1, K) < ak.view(R, 1)    # (R,K)
    eye = torch.eye(d, dtype=dt, device=dev)

    if init_type == 'random_responsibilities':
        # per-replicate seeded draw; design notes: _init_params
        seeds = torch.randint(0, 1 << 62, (R,), generator=generator,
                              device=dev).tolist()
        stage = functools.partial(_init_mstep_random, X=X, seeds=seeds,
                                  active_k=active_k, K=K,
                                  reg_covar=reg_covar, n_chunk=n_chunk)
        outs = _run_chunked(R, em_chunk, stage)
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
        stage = functools.partial(_kmeans_assign_stats, X=X, x2=x2,
                                  centres=C, alive=alive,
                                  reg_covar=reg_covar, n_chunk=n_chunk)
        outs = _run_chunked(R, em_chunk, stage)
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
    """
    Weighted M-step on the legacy (broadcast) kernels.

    Weighted M-step.  X:(R,N,d), resp:(R,N,K), s:(R,N) sample weights.

    Parameters
    ----------
    X : torch.Tensor of shape (R, N, d)
        Samples, per replicate (typically an expand() view).
    resp : torch.Tensor of shape (R, N, K)
        Responsibilities.
    s : torch.Tensor of shape (R, N)
        Sample weights.
    reg_covar : float
        Covariance floor.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Updated weights (R, K), means (R, K, d) and covariances
        (R, K, d, d).
    """
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
    """
    Weighted EM over a leading replicate dimension, in replicate chunks.

    Replicates (CV fits, bootstrap realisations) are independent, so they are
    processed in chunks of `em_chunk` purely to bound peak memory; results do
    not depend on the chunk on CPU (bitwise, with one float32 exception);
    on CUDA see `em_chunk`.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        Shared data.
    s : torch.Tensor of shape (n_replicates, n_samples)
        Per-replicate non-negative sample weights: 0/1 masks for CV folds,
        multinomial counts for bootstrap realisations, ones for a plain fit.
        The replicate axis may be split over any number of leading dims
        (shape (..., n_samples), R = their product, row-major): an
        expand() view then stays a view, and each replicate chunk gathers
        only its own (chunk, n_samples) rows.
    w, mu, cov : torch.Tensor
        Initial mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    reg_covar, threshold_fit : float
        As in `estimate`.
    max_iter : int
        As in `estimate`.
    em_chunk : int, default=8
        Replicates per batch; a CUDA out-of-memory error halves it and
        retries (with a RuntimeWarning). Bitwise chunk-invariant on CPU
        except em_chunk=1 with a one-component model at float32; on CUDA
        see docs/estimate_notes.md, "Reproducibility".
    n_chunk : int, optional
        As in `estimate`.

    Returns
    -------
    w, mu, cov : torch.Tensor
        Fitted parameters, same shapes as the inputs.
    ll : torch.Tensor of shape (n_replicates,)
        Weighted mean per-sample log-likelihood of the parameters that
        ENTERED the final iteration's M-step: it lags the returned
        parameters by one M-step (for converged replicates, by less than
        `threshold_fit`). Score the returned parameters with `val_ll` when
        an exact final-state likelihood is needed.

    Raises
    ------
    RuntimeError
        When a covariance cannot be factorised even after escalating
        diagonal regularisation (raise reg_covar, or use float64), or when
        a CUDA out-of-memory error persists at one replicate per batch
        (torch.cuda.OutOfMemoryError). Arguments are not validated here;
        `estimate` validates them at entry.

    Warns
    -----
    RuntimeWarning
        Once per call for replicates that exhausted `max_iter` without
        meeting `threshold_fit` (their last iterates are returned), and once
        for replicates finished on the centred legacy kernels because a
        component collapsed beyond the GEMM kernels' precision. Covariance
        factorisation also warns when a component needed extra diagonal
        regularisation, or when an all-zero or non-finite covariance was
        replaced by the identity.
    """
    R = _n_rows(s)
    n_chunk = None if n_chunk is None else int(n_chunk)
    # the streamed path reads the weights one sample block at a time
    # (never an (R,N) copy); the single-pass path takes its (chunk, N)
    # gather, one responsibility-buffer-over-K
    outs = _run_chunked(R, em_chunk, lambda sl: _em_fit_core(
        X, _rows(s, sl) if n_chunk is None else _WeightChunks(s, sl, n_chunk),
        w[sl], mu[sl], cov[sl], reg_covar, threshold_fit, max_iter, n_chunk))
    n_unconverged = sum(o.n_unconverged for o in outs)
    if n_unconverged:
        warnings.warn(
            f"{n_unconverged} of {R} EM replicate(s) did not converge "
            f"within max_iter={max_iter} iterations; their last iterates "
            f"are used (raise max_iter or loosen threshold_fit)",
            RuntimeWarning, stacklevel=3)   # em_fit -> _full_precision -> caller
    n_rerouted = sum(o.n_rerouted for o in outs)
    if n_rerouted:
        warnings.warn(
            f"{n_rerouted} of {R} EM replicate(s) carried a component "
            f"collapsed beyond the GEMM kernels' precision at this dtype "
            f"and finished on the (slower, centred) legacy kernels — "
            f"expected for surplus components during selection at "
            f"float32, not a data problem",
            RuntimeWarning, stacklevel=3)
    if len(outs) == 1:
        return outs[0][:4]
    return tuple(torch.cat([o[j] for o in outs]) for j in range(4))


def _em_step_nchunked(X, weight_chunks, w, mu, chol, logw, reg_covar,
                      n_chunk, rows=None):
    """
    One EM iteration on the legacy kernels, streamed over sample chunks.

    One EM iteration accumulated over sample chunks (streaming M-step).

    Accumulates N_k, sum resp*(x - mu_old) and sum resp*(x - mu_old)(..)^T
    per sample-chunk; shifting the moments by the incoming means is
    algebraically identical to accumulating resp*x / resp*xx^T but avoids
    the fp32 cancellation of the origin-moment form when |mu|^2 >> var.  As
    in the GEMM path, the cross-chunk accumulators run at float64 and the
    finished diagonal is clamped back to the reg_covar floor, which float32
    rounding of the shifted moments can otherwise undercut. `weight_chunks`
    and `rows` as in _em_step_gemm_nchunked.

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    weight_chunks : _WeightChunks or torch.Tensor
        Sample weights, an accessor or a dense (R, N) tensor.
    w, mu : torch.Tensor
        Weights (n_active, K) and means (n_active, K, d) of the replicates
        taking part.
    chol : torch.Tensor of shape (n_active, K, d, d)
        Cholesky factors of their covariances.
    logw : torch.Tensor of shape (n_active, K)
        Log weights, -inf for dead components.
    reg_covar : float
        Covariance floor.
    n_chunk : int
        Samples per chunk.
    rows : torch.Tensor, optional
        Replicate rows of the accessor taking part; None for all.

    Returns
    -------
    ll : torch.Tensor of shape (n_active,)
        Weighted mean per-sample log-likelihood of the incoming parameters.
    w, mu, cov : torch.Tensor
        Parameters after the M-step, shapes (n_active, K), (n_active, K, d),
        (n_active, K, d, d).
    """
    weight_chunks = _as_weight_chunks(weight_chunks, n_chunk)
    n_active, N = weight_chunks.shape
    if rows is not None:
        n_active = rows.numel()
    d = X.shape[-1]
    K = w.shape[-1]
    dt, dev = X.dtype, X.device
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    Nk = torch.zeros(n_active, K, dtype=torch.float64, device=dev)
    Sd = torch.zeros(n_active, K, d, dtype=torch.float64, device=dev)
    Sdd = torch.zeros(n_active, K, d, d, dtype=torch.float64, device=dev)
    ll_sum = torch.zeros(n_active, dtype=torch.float64, device=dev)
    for i in range(0, N, n_chunk):
        Xc = X[i:i+n_chunk]
        nc = Xc.shape[0]
        sc = weight_chunks.rows(i, n_chunk, rows)
        # logjoint: (n_active,nc,K)
        logjoint = logw.unsqueeze(1) + log_gauss(Xc.expand(n_active, nc, d),
                                                 mu, chol)
        ll_n = torch.logsumexp(logjoint, dim=-1)
        resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
        wr = resp * sc.unsqueeze(-1)
        ll_sum += (sc * ll_n).sum(1).double()
        Nk += wr.sum(1).double()
        # diff: (n_active,nc,K,d)
        diff = Xc.view(1, nc, 1, d) - mu.unsqueeze(1)
        Sd += torch.einsum('rnk,rnkd->rkd', wr, diff).double()
        Sdd += torch.einsum('rnk,rnki,rnkj->rkij', wr, diff, diff).double()
    # float64
    tot = (weight_chunks.totals() if rows is None
           else weight_chunks.totals()[rows])
    ll = (ll_sum / tot).to(dt)
    dead = Nk < 1e-10
    Nk_safe = Nk.clamp_min(1e-10)
    delta = Sd / Nk_safe.unsqueeze(-1)                           # mu_new-mu_old
    mu_new = mu.double() + delta
    cov_new = Sdd / Nk_safe[..., None, None] \
        - delta.unsqueeze(-1) * delta.unsqueeze(-2) + reg_covar * eye
    # the shifted moments can round the variance below the reg_covar floor
    # for a component collapsed onto near-duplicate points; restore the floor
    cov_new = _restore_floor(cov_new, reg_covar)
    w_new = torch.where(dead, torch.zeros_like(Nk), Nk / tot.unsqueeze(-1))
    w_new = w_new / w_new.sum(-1, keepdim=True).clamp_min(1e-30)
    mu_new = torch.where(dead[..., None], torch.zeros_like(mu_new), mu_new)
    cov_new = torch.where(dead[..., None, None], eye.expand_as(cov_new),
                          cov_new)
    return ll, w_new.to(dt), mu_new.to(dt), cov_new.to(dt)


# kernel reroute; design notes: _REROUTE_TOL_NATS
_REROUTE_TOL_NATS = 1e-3


def _reroute_danger(w, mu, cov):
    """
    Per-component danger statistic of the kernel-reroute detector.

    Per-component danger statistic (R,K): (|mu_k|^2 + 1) / min_j diag
    cov_k, zeroed for dead components, at float64; the estimated
    log-density residue is u x this.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).

    Returns
    -------
    torch.Tensor of shape (R, K)
        The statistic, float64.
    """
    dmin = torch.diagonal(cov, dim1=-2, dim2=-1).double().amin(-1)   # (R,K)
    dgr = (mu.double().pow(2).sum(-1) + 1.0) \
        / dmin.clamp_min(torch.finfo(torch.float64).tiny)
    return torch.where(w > 0, dgr, torch.zeros_like(dgr))            # (R,K)


def _reroute_trips(w, mu, cov, repaired=None):
    """
    Replicates the GEMM kernels cannot evaluate at this dtype's precision.

    (R,) bool: replicates with a component whose GEMM residue would
    exceed _REROUTE_TOL_NATS at this dtype's unit roundoff.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    repaired : torch.Tensor of shape (R, K), optional
        Components `_chol` had to repair; they trip at a tenfold lower
        threshold.

    Returns
    -------
    torch.Tensor of shape (R,)
        Bool, True for the replicates to reroute.
    """
    u = torch.finfo(cov.dtype).eps / 2.0        # unit roundoff, not eps
    thr = _REROUTE_TOL_NATS / u
    dgr = _reroute_danger(w, mu, cov)                                # (R,K)
    trips = dgr > thr
    if repaired is not None:
        # escalation lowers the threshold tenfold; design notes: _reroute_trips
        trips = trips | (repaired & (dgr > thr / 10.0))
    return trips.any(-1)


def _check_reroute(w, mu, cov, legacy_mask):
    """
    Kernel-reroute check on the parameters entering an E-step (GEMM kernel).

    Runs on the parameters entering EVERY E-step, including the first (warm
    starts can already carry a collapsed component). The factorisation
    doubles as the escalation evidence and its repair is written back.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Parameters of the active replicates, shapes (n_active, K),
        (n_active, K, d), (n_active, K, d, d).
    legacy_mask : torch.Tensor
        Sticky per-replicate reroute mask over the active set, shape
        (n_active,).

    Returns
    -------
    cov : torch.Tensor
        The factorised (possibly repaired) covariances.
    legacy_mask : torch.Tensor
        The mask with any newly tripped replicates added.
    n_new : int
        Replicates rerouted by this check (not already in the mask).
    """
    _, cov, rep = _chol(cov, return_repaired=True)
    trips = _reroute_trips(w, mu, cov, repaired=rep)
    new = trips & ~legacy_mask
    n_new = 0
    if bool(new.any()):
        n_new = int(new.sum())
        legacy_mask = legacy_mask | trips
    return cov, legacy_mask, n_new


def _em_step_mixed(X, F, weights_active, weight_chunks, w, mu, cov,
                   legacy_mask, active_idx, reg_covar, n_chunk,
                   legacy_sample_chunk):
    """
    One EM iteration on a batch split between the GEMM and legacy kernels.

    Replicates outside `legacy_mask` take the GEMM step (single pass or
    streamed), rerouted ones the centred legacy step in sample chunks of
    `legacy_sample_chunk`; the results are scattered back into batch order.

    Parameters
    ----------
    X : torch.Tensor
        Shared data, shape (N, d).
    F : torch.Tensor or None
        Quadratic features of `X` (single-pass GEMM path), else None.
    weights_active : torch.Tensor or None
        Dense weights of the active replicates, shape (n_active, N), on the
        single-pass path; None on the streamed path.
    weight_chunks : _WeightChunks or None
        The streamed path's weight accessor; None on the single-pass path.
    w, mu, cov : torch.Tensor
        Parameters of the active replicates, shapes (n_active, K),
        (n_active, K, d), (n_active, K, d, d).
    legacy_mask : torch.Tensor
        Rerouted replicates among the active ones, shape (n_active,).
    active_idx : torch.Tensor
        Indices of the active replicates in the accessor's rows, shape
        (n_active,).
    reg_covar : float
        Covariance floor.
    n_chunk : int or None
        Sample chunk of the streamed path; None for the single pass.
    legacy_sample_chunk : int
        Sample chunk of the legacy step on rerouted replicates.

    Returns
    -------
    ll : torch.Tensor
        Weighted mean per-sample log-likelihood of the incoming parameters,
        shape (n_active,).
    w, mu, cov : torch.Tensor
        Parameters after the M-step, same shapes as the inputs.
    """
    n_active = active_idx.shape[0]
    d = X.shape[-1]
    dev = X.device
    gemm_idx = (~legacy_mask).nonzero(as_tuple=True)[0]
    legacy_idx = legacy_mask.nonzero(as_tuple=True)[0]
    ll = torch.empty(n_active, dtype=X.dtype, device=dev)
    w_next, mu_next, cov_next = (torch.empty_like(w),
                                 torch.empty_like(mu),
                                 torch.empty_like(cov))
    if gemm_idx.numel():
        if n_chunk is None:
            resp, llg = _estep_gemm(F, weights_active[gemm_idx],
                                    w[gemm_idx], mu[gemm_idx],
                                    cov[gemm_idx])
            wg, mug, covg = _mstep_gemm(F, resp,
                                        weights_active[gemm_idx],
                                        reg_covar, d)
            del resp
        else:
            llg, wg, mug, covg = _em_step_gemm_nchunked(
                X, weight_chunks, w[gemm_idx], mu[gemm_idx],
                cov[gemm_idx], reg_covar, n_chunk,
                rows=active_idx[gemm_idx])
        ll[gemm_idx] = llg
        w_next[gemm_idx], mu_next[gemm_idx], cov_next[gemm_idx] = \
            wg, mug, covg
    if legacy_idx.numel():
        chol_l, cov_l = _chol(cov[legacy_idx])
        logw_l = torch.where(
            w[legacy_idx] > 0,
            torch.log(w[legacy_idx].clamp_min(1e-300)),
            torch.full_like(w[legacy_idx], -torch.inf))
        if n_chunk is not None:
            lll, wl, mul, covl = _em_step_nchunked(
                X, weight_chunks, w[legacy_idx], mu[legacy_idx],
                chol_l, logw_l, reg_covar, legacy_sample_chunk,
                rows=active_idx[legacy_idx])
        else:
            # dense compacted weights: a view-only accessor whose
            # totals are the one-pass sums (bitwise the old code)
            lll, wl, mul, covl = _em_step_nchunked(
                X, _WeightChunks(weights_active, slice(0, n_active)),
                w[legacy_idx], mu[legacy_idx], chol_l, logw_l,
                reg_covar, legacy_sample_chunk, rows=legacy_idx)
        ll[legacy_idx] = lll
        w_next[legacy_idx], mu_next[legacy_idx], \
            cov_next[legacy_idx] = wl, mul, covl
    w, mu, cov = w_next, mu_next, cov_next
    return ll, w, mu, cov


def _em_iteration(X, F, weights_active, weight_chunks, w, mu, cov,
                  legacy_mask, active_idx, active_rows, gemm, N, reg_covar,
                  n_chunk, legacy_sample_chunk):
    """
    One EM iteration of the active replicates, on the kernels they need.

    Dispatches to the GEMM step (single pass: `_estep_gemm` and
    `_mstep_gemm`; streamed: `_em_step_gemm_nchunked`), to the mixed batch
    (`_em_step_mixed`) when some replicates are rerouted, or to the legacy
    step (single pass inline; streamed: `_em_step_nchunked`) when the
    legacy kernels are selected by the environment.

    Parameters
    ----------
    X : torch.Tensor
        Shared data, shape (N, d).
    F : torch.Tensor or None
        Quadratic features of `X` (single-pass GEMM path), else None.
    weights_active : torch.Tensor or None
        Dense weights of the active replicates, shape (n_active, N), on the
        single-pass path; None on the streamed path.
    weight_chunks : _WeightChunks or None
        The streamed path's weight accessor; None on the single-pass path.
    w, mu, cov : torch.Tensor
        Parameters of the active replicates, shapes (n_active, K),
        (n_active, K, d), (n_active, K, d, d).
    legacy_mask : torch.Tensor
        Rerouted replicates among the active ones, shape (n_active,).
    active_idx : torch.Tensor
        Indices of the active replicates in the accessor's rows, shape
        (n_active,).
    active_rows : torch.Tensor or None
        `active_idx`, or None while no replicate has converged (all rows).
    gemm : bool
        Whether the GEMM kernels are selected.
    N : int
        Number of samples.
    reg_covar : float
        Covariance floor.
    n_chunk : int or None
        Sample chunk of the streamed path; None for the single pass.
    legacy_sample_chunk : int
        Sample chunk of the legacy step on rerouted replicates.

    Returns
    -------
    ll : torch.Tensor
        Weighted mean per-sample log-likelihood of the incoming parameters,
        shape (n_active,).
    w, mu, cov : torch.Tensor
        Parameters after the M-step, same shapes as the inputs.
    """
    n_active = active_idx.shape[0]
    if gemm and not bool(legacy_mask.any()):
        if n_chunk is None:
            resp, ll = _estep_gemm(F, weights_active, w, mu, cov)
            w, mu, cov = _mstep_gemm(F, resp, weights_active, reg_covar,
                                     X.shape[-1])
            # release the consumed responsibilities as soon as the M-step
            # has used them (they are the (n_active,N,K) peak)
            del resp
        else:
            ll, w, mu, cov = _em_step_gemm_nchunked(
                X, weight_chunks, w, mu, cov, reg_covar, n_chunk,
                rows=active_rows)
    elif gemm:
        ll, w, mu, cov = _em_step_mixed(
            X, F, weights_active, weight_chunks, w, mu, cov,
            legacy_mask=legacy_mask, active_idx=active_idx,
            reg_covar=reg_covar, n_chunk=n_chunk,
            legacy_sample_chunk=legacy_sample_chunk)
    elif n_chunk is None:
        chol, cov = _chol(cov)
        logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                           torch.full_like(w, -torch.inf))
        Xr = X.expand(n_active, N, X.shape[-1])
        # shapes (n_active,N,K), then (n_active,N)
        logjoint = logw.unsqueeze(1) + log_gauss(Xr, mu, chol)
        ll_n = torch.logsumexp(logjoint, dim=-1)
        resp = torch.exp(logjoint - ll_n.unsqueeze(-1))
        ll = (weights_active * ll_n).sum(1) / weights_active.sum(1)
        w, mu, cov = _m_step(Xr, resp, weights_active, reg_covar)
    else:
        chol, cov = _chol(cov)
        logw = torch.where(w > 0, torch.log(w.clamp_min(1e-300)),
                           torch.full_like(w, -torch.inf))
        ll, w, mu, cov = _em_step_nchunked(X, weight_chunks, w, mu, chol,
                                           logw, reg_covar, n_chunk,
                                           rows=active_rows)
    return ll, w, mu, cov


def _compact_converged(done, ll, active_idx, weights_active, w, mu, cov,
                       legacy_mask, w_out, mu_out, cov_out, ll_out):
    """
    Store the converged replicates' results and drop them from the batch.

    Parameters
    ----------
    done : torch.Tensor
        Converged replicates among the active ones, shape (n_active,) bool.
    ll : torch.Tensor
        This iteration's log-likelihoods, shape (n_active,).
    active_idx : torch.Tensor
        Indices of the active replicates in the output rows, shape (n_active,).
    weights_active : torch.Tensor or None
        Dense weights of the active replicates (single-pass path); None on
        the streamed path, whose weights are addressed through `active_idx`.
    w, mu, cov : torch.Tensor
        Parameters of the active replicates.
    legacy_mask : torch.Tensor
        Rerouted replicates among the active ones, shape (n_active,).
    w_out, mu_out, cov_out, ll_out : torch.Tensor
        Results for all R replicates; the converged rows are written in
        place.

    Returns
    -------
    active_idx, prev, weights_active, w, mu, cov, legacy_mask
        The same quantities for the replicates still active; `prev` is
        their log-likelihood from this iteration.
    """
    di = active_idx[done]
    w_out[di], mu_out[di], cov_out[di] = w[done], mu[done], cov[done]
    ll_out[di] = ll[done]
    keep = ~done
    active_idx, prev = active_idx[keep], ll[keep]
    if weights_active is not None:
        weights_active = weights_active[keep]
    w, mu, cov = w[keep], mu[keep], cov[keep]
    legacy_mask = legacy_mask[keep]
    return active_idx, prev, weights_active, w, mu, cov, legacy_mask


class EMResult(NamedTuple):
    """
    Result of `_em_fit_core` for one replicate batch.

    A tuple, so slicing (`res[:4]`) and positional reads still work.

    Parameters
    ----------
    w, mu, cov : torch.Tensor
        Fitted parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    ll : torch.Tensor
        Last E-step's weighted mean per-sample log-likelihood, shape (R,).
    n_unconverged : int
        Replicates that exhausted `max_iter`.
    n_rerouted : int
        Replicates the kernel-reroute detector moved to the legacy kernels.
    """
    w: torch.Tensor
    mu: torch.Tensor
    cov: torch.Tensor
    ll: torch.Tensor
    n_unconverged: int
    n_rerouted: int


def _em_fit_core(X, s, w, mu, cov, reg_covar=1e-6, threshold_fit=1e-5, max_iter=500,
                 n_chunk=None):
    """
    Weighted EM, batched over replicates.

    X: (N,d) shared data; s: (R,N) per-replicate sample weights, or a
    _WeightChunks accessor on the streamed path (n_chunk set), where the
    weights are read one sample block at a time and the compaction of
    converged replicates is index-only — no (R,N) copy is ever made.
    n_chunk: sample-dimension chunk size for the per-iteration E/M
    accumulation (None = the single-pass path).
    Returns an EMResult (w, mu, cov, ll, n_unconverged, n_rerouted) with ll
    the last E-step's weighted mean per-sample log-likelihood, shape (R,),
    n_unconverged the replicates that exhausted max_iter and n_rerouted the
    replicates the kernel-reroute detector moved to the legacy kernels
    (each aggregated and warned about once by em_fit).

    Each iteration is: the reroute check (`_check_reroute`, GEMM kernel
    only), one EM step on the kernels each replicate needs
    (`_em_iteration`), the convergence test, and the compaction of the
    converged replicates (`_compact_converged`).

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    s : torch.Tensor of shape (R, N) or _WeightChunks
        Per-replicate sample weights.
    w, mu, cov : torch.Tensor
        Initial parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    reg_covar : float, default=1e-6
        Covariance floor.
    threshold_fit : float, default=1e-5
        Convergence threshold on the change of `ll`.
    max_iter : int, default=500
        Maximum EM iterations.
    n_chunk : int, optional
        Sample chunk of the streamed path; None for the single pass.

    Returns
    -------
    EMResult
        (w, mu, cov, ll, n_unconverged, n_rerouted), as described above.
    """
    stream = n_chunk is not None
    if isinstance(s, _WeightChunks):
        weight_chunks = s
        if not stream:
            s = weight_chunks.rows(0, weight_chunks.N)
    else:
        weight_chunks = (_WeightChunks(s, slice(0, s.shape[0]), n_chunk)
                         if stream else None)
    R, N = weight_chunks.shape if stream else s.shape
    dev = X.device
    d = X.shape[-1]
    # compact the batch; design notes: _em_fit_core, compaction
    active_idx = torch.arange(R, device=dev)
    w_out, mu_out, cov_out = w.clone(), mu.clone(), cov.clone()
    ll_out = torch.full((R,), -torch.inf, dtype=X.dtype, device=dev)
    prev = torch.full((R,), -torch.inf, dtype=X.dtype, device=dev)
    weights_active = None if stream else s
    gemm = _EM_KERNEL == 'gemm'
    F = _features(X) if gemm and n_chunk is None else None
    # sticky reroute mask; design notes: _em_fit_core, reroute mask
    legacy_mask = torch.zeros(R, dtype=torch.bool, device=dev)
    n_rerouted = 0
    K = w.shape[-1]
    legacy_sample_chunk = _legacy_sample_chunk(N, K, d, X.dtype, n_chunk)
    for _ in range(max_iter):
        n_active = active_idx.shape[0]
        # rows=None (a view for 2-D weights) while nothing has converged:
        # active_idx is then the identity
        active_rows = None if n_active == R else active_idx
        if gemm:
            cov, legacy_mask, n_new = _check_reroute(w, mu, cov, legacy_mask)
            n_rerouted += n_new
        ll, w, mu, cov = _em_iteration(
            X, F, weights_active, weight_chunks, w, mu, cov,
            legacy_mask=legacy_mask, active_idx=active_idx,
            active_rows=active_rows, gemm=gemm, N=N, reg_covar=reg_covar,
            n_chunk=n_chunk, legacy_sample_chunk=legacy_sample_chunk)
        done = (ll - prev).abs() < threshold_fit
        if done.any():
            (active_idx, prev, weights_active, w, mu, cov,
             legacy_mask) = _compact_converged(
                done, ll, active_idx, weights_active, w, mu, cov,
                legacy_mask, w_out, mu_out, cov_out, ll_out)
            if active_idx.shape[0] == 0:
                return EMResult(w_out, mu_out, cov_out, ll_out, 0, n_rerouted)
        else:
            prev = ll
    # unconverged stragglers: write back current state
    w_out[active_idx], mu_out[active_idx], cov_out[active_idx], \
        ll_out[active_idx] = w, mu, cov, prev
    return EMResult(w_out, mu_out, cov_out, ll_out, int(active_idx.shape[0]),
                    n_rerouted)


@_full_precision
def val_ll(X, s_val, w, mu, cov, n_chunk=None, em_chunk=None):
    """Weighted mean per-sample log-likelihood of X under each replicate.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_variables)
        Shared data.
    s_val : torch.Tensor of shape (n_replicates, n_samples)
        Non-negative weights selecting (and weighting) the scored samples;
        the replicate axis may span several leading dims, as in `em_fit`.
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    n_chunk : int, optional
        Stream over samples in chunks of this size (as in `estimate`).
    em_chunk : int, optional
        Replicates scored per batch, under the same halve-and-retry CUDA
        out-of-memory guard as `em_fit`. None scores all replicates in one
        batch. Results do not depend on it on CPU (bitwise); on CUDA they
        move at reduction order.

    Returns
    -------
    torch.Tensor of shape (n_replicates,)
        -inf for a replicate whose weights are all zero (an empty validation
        fold scores worse than any real fit, instead of a spurious 0).

    Raises
    ------
    RuntimeError
        As in `em_fit`: a covariance that cannot be factorised even after
        escalating regularisation, or a CUDA out-of-memory error at one
        replicate per batch. Arguments are not validated here.

    Warns
    -----
    RuntimeWarning
        Once per call for replicates scored with the centred legacy kernels
        because a component collapsed beyond the GEMM kernels' precision;
        and, as in `em_fit`, from covariance factorisation (extra diagonal
        regularisation; all-zero or non-finite covariance replaced by the
        identity).
    """
    R = _n_rows(s_val)
    n_chunk = None if n_chunk is None else int(n_chunk)
    outs = _run_chunked(R, R if em_chunk is None else em_chunk,
                        lambda sl: _val_ll_core(
                            X, _rows(s_val, sl) if n_chunk is None
                            else _WeightChunks(s_val, sl, n_chunk),
                            w[sl], mu[sl], cov[sl], n_chunk))
    n_rerouted = sum(o.n_rerouted for o in outs)
    if n_rerouted:
        warnings.warn(
            f"{n_rerouted} of {R} replicate(s) carried a component "
            f"collapsed beyond the GEMM kernels' precision at this dtype "
            f"and were scored with the (slower, centred) legacy kernels",
            RuntimeWarning, stacklevel=3)   # val_ll -> _full_precision -> caller
    vals = [o.values for o in outs]
    return vals[0] if len(vals) == 1 else torch.cat(vals)


def _legacy_score_sum(X, weight_chunks, w, mu, chol, n_chunk, rows=None):
    """
    Weighted log-likelihood sums on the centred legacy kernels.

    Weighted log-likelihood sums via the centred kernels, float64 acc,
    over sample chunks of `n_chunk` (None: one chunk, which is bitwise the
    unchunked evaluation) for the `rows` of the accessor `weight_chunks`.

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    weight_chunks : _WeightChunks or torch.Tensor
        Sample weights, an accessor or a dense (R, N) tensor.
    w, mu : torch.Tensor
        Weights and means of the rows scored.
    chol : torch.Tensor
        Cholesky factors of their covariances.
    n_chunk : int or None
        Samples per chunk; None scores all samples at once.
    rows : torch.Tensor, optional
        Replicate rows of the accessor to score; None for all.

    Returns
    -------
    torch.Tensor of shape (n_rows,)
        Sum over samples of weight x log-density, float64.
    """
    weight_chunks = _as_weight_chunks(weight_chunks, n_chunk)
    R, N = weight_chunks.shape
    if rows is not None:
        R = rows.numel()
    n = N if n_chunk is None else n_chunk
    acc = torch.zeros(R, dtype=torch.float64, device=X.device)
    for i in range(0, N, n):
        Xc = X[i:i+n]
        nc = Xc.shape[0]
        ll_n = gmm_score(Xc.expand(R, nc, X.shape[-1]), w, mu, chol)
        acc = acc + (weight_chunks.rows(i, n, rows) * ll_n).sum(1).double()
    return acc


class ValLLResult(NamedTuple):
    """
    Result of `_val_ll_core` for one replicate batch (a tuple).

    Parameters
    ----------
    values : torch.Tensor
        Weighted mean per-sample log-likelihood per replicate, shape (R,).
    n_rerouted : int
        Replicates scored with the centred legacy kernels.
    """
    values: torch.Tensor
    n_rerouted: int


def _val_ll_core(X, s_val, w, mu, cov, n_chunk=None):
    """
    `val_ll` for one replicate batch.

    `val_ll` for one replicate batch; returns a ValLLResult
    (values, n_rerouted).

    The cross-chunk accumulator runs at float64, as in the EM kernels: at
    float32 the drift of a working-dtype sum grows as sqrt(#chunks) x ulp
    and reaches a visible fraction of threshold_components at large N — and
    this metric decides k. Replicates the reroute detector trips are scored
    with the centred legacy kernels in em_fit's model-derived sample chunk,
    so the scoring stage stays inside the budget em_chunk was sized with
    (a RuntimeWarning per call reports them, as in em_fit). `s_val` is a dense
    (R,N) tensor or a _WeightChunks accessor; either way the weights are
    read per sample chunk and only the rerouted rows are ever gathered.

    Parameters
    ----------
    X : torch.Tensor of shape (N, d)
        Shared data.
    s_val : torch.Tensor of shape (R, N) or _WeightChunks
        Weights selecting the scored samples.
    w, mu, cov : torch.Tensor
        Mixture parameters, shapes (R, K), (R, K, d), (R, K, d, d).
    n_chunk : int, optional
        Samples per chunk; None scores all samples at once.

    Returns
    -------
    ValLLResult
        (values, n_rerouted).
    """
    weight_chunks = (
        s_val if isinstance(s_val, _WeightChunks)
        else _WeightChunks(s_val, slice(0, s_val.shape[0]), n_chunk))
    R, N = weight_chunks.shape
    n_rerouted = 0
    if _EM_KERNEL == 'gemm':
        chol_all, cov, rep = _chol(cov, return_repaired=True)
        trips = _reroute_trips(w, mu, cov, repaired=rep)
        n_rerouted = int(trips.sum())
        acc = torch.zeros(R, dtype=torch.float64, device=X.device)
        gemm_idx = (~trips).nonzero(as_tuple=True)[0]
        legacy_idx = trips.nonzero(as_tuple=True)[0]
        if gemm_idx.numel():
            coef, const = _gauss_coef(w[gemm_idx], mu[gemm_idx], cov[gemm_idx])
            # all rows: a view
            rows_g = None if legacy_idx.numel() == 0 else gemm_idx
            n = n_chunk or N
            accg = torch.zeros(gemm_idx.numel(), dtype=torch.float64,
                               device=X.device)
            for i in range(0, N, n):
                Fc = _features(X[i:i+n])
                lj = torch.matmul(Fc.unsqueeze(0), coef.transpose(1, 2))
                lj += const.unsqueeze(1)
                accg = accg + (weight_chunks.rows(i, n, rows_g)
                               * torch.logsumexp(lj, -1)).sum(1).double()
                del lj, Fc
            acc[gemm_idx] = accg
        if legacy_idx.numel():
            # rerouted rows: legacy kernels; design notes: _val_ll_core
            legacy_sample_chunk = _legacy_sample_chunk(
                N, w.shape[-1], X.shape[-1], X.dtype, n_chunk)
            acc[legacy_idx] = _legacy_score_sum(
                X, weight_chunks, w[legacy_idx], mu[legacy_idx],
                chol_all[legacy_idx], min(legacy_sample_chunk, n_chunk or N),
                rows=legacy_idx)
    else:
        # legacy kernels selected by the environment: em_chunk was sized
        # from the legacy model, so the caller's n_chunk stands
        chol, _ = _chol(cov)
        acc = _legacy_score_sum(X, weight_chunks, w, mu, chol, n_chunk)
    tot = weight_chunks.totals()
    vals = torch.where(tot > 0, acc / tot.clamp_min(1e-30),
                       torch.full_like(acc, -torch.inf)).to(X.dtype)
    return ValLLResult(vals, n_rerouted)
