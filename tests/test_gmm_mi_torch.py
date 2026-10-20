"""Tests for the torch GMM-MI port, following the conventions of
gmm_mi/tests/test_sample.py (analytic targets, sigma-band assertions)."""

import os
import warnings

import numpy as np
import pytest
import torch

import gmm_mi_torch.estimator
from gmm_mi_torch import estimate, estimate_categorical

SIGMAS = 3


def test_MI_gaussian_analytic():
    # bivariate Gaussian: analytic MI = -0.5 log(1 - rho^2)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 10000)
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    mi_mean, mi_std = estimate(X, 'MI', protocol='batched', max_components=3)
    assert (analytic_mi >= mi_mean - SIGMAS * mi_std) and \
           (analytic_mi <= mi_mean + SIGMAS * mi_std)


def test_TC_gaussian_analytic():
    # d-dimensional Gaussian: TC = -0.5 log det(correlation matrix).
    # The estimator sees one finite dataset, so the correct target is the
    # TC of the *sample* covariance, not the population value.
    rng = np.random.default_rng(1)
    d = 4
    A = rng.normal(size=(d, d)) * 0.4 + np.eye(d)
    cov = A @ A.T
    X = rng.multivariate_normal(np.zeros(d), cov, 10000)
    S = np.cov(X.T)
    Ds = np.diag(1 / np.sqrt(np.diag(S)))
    analytic_tc = -0.5 * np.log(np.linalg.det(Ds @ S @ Ds))
    tc_mean, tc_std = estimate(X, 'TC', protocol='batched', max_components=3)
    assert (analytic_tc >= tc_mean - SIGMAS * tc_std) and \
           (analytic_tc <= tc_mean + SIGMAS * tc_std)


def test_TC_independent_is_zero():
    # mutually independent variables: TC consistent with zero
    rng = np.random.default_rng(2)
    X = rng.normal(size=(10000, 4))
    tc_mean, tc_std = estimate(X, 'TC', protocol='batched', max_components=3)
    assert tc_mean <= SIGMAS * max(tc_std, 1e-4) + 5e-3


def test_cMI_independent_conditioner():
    # cMI(X;Y|Z) with Z independent of (X,Y) must equal MI(X;Y)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(3)
    X2 = rng.multivariate_normal([0, 0], cov, 10000)
    Xc = np.hstack([X2, rng.normal(size=(10000, 1))])
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    cmi_mean, cmi_std = estimate(Xc, 'cMI', protocol='batched', max_components=3)
    assert (analytic_mi >= cmi_mean - SIGMAS * cmi_std) and \
           (analytic_mi <= cmi_mean + SIGMAS * cmi_std)


def test_TC_invariance_under_marginal_transforms():
    # TC is invariant under per-variable diffeomorphisms
    rng = np.random.default_rng(4)
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal([0, 0], cov, 10000)
    S = np.cov(Z.T)
    Ds = np.diag(1 / np.sqrt(np.diag(S)))
    tc_target = -0.5 * np.log(np.linalg.det(Ds @ S @ Ds))
    from scipy.stats import norm, rankdata
    X = Z.copy()
    X[:, 1] = np.exp(0.5 * Z[:, 1])          # warp one marginal
    X = np.column_stack([norm.ppf(rankdata(X[:, i]) / (X.shape[0] + 1))
                         for i in range(2)])  # gaussianise back
    tc_mean, tc_std = estimate(X, 'TC', protocol='batched', max_components=3)
    assert (tc_target >= tc_mean - SIGMAS * tc_std) and \
           (tc_target <= tc_mean + SIGMAS * tc_std)


def test_KL_gaussian_analytic():
    # KL between the two 1D Gaussian marginals, both orders:
    # KL[N(m1,s1)||N(m2,s2)] = log(s2/s1) + (s1^2+(m1-m2)^2)/(2 s2^2) - 1/2
    rho, mu2, s2 = 0.5, 1.0, 1.5
    cov = np.array([[1.0, rho * s2], [rho * s2, s2 ** 2]])
    rng = np.random.default_rng(5)
    X = rng.multivariate_normal([0.0, mu2], cov, 10000)
    kl_f = np.log(s2) + (1.0 + mu2 ** 2) / (2 * s2 ** 2) - 0.5
    kl_r = -np.log(s2) + (s2 ** 2 + mu2 ** 2) / 2.0 - 0.5
    for quantity, target in (('KL_forward', kl_f), ('KL_reverse', kl_r)):
        kl_mean, kl_std = estimate(X, quantity, protocol='batched', max_components=3)
        assert (target >= kl_mean - SIGMAS * kl_std) and \
               (target <= kl_mean + SIGMAS * kl_std)


def test_categorical_MI_analytic():
    # continuous-categorical MI, near-balanced categories; target from
    # 1D quadrature with the empirical category weights
    rng = np.random.default_rng(8)
    N = 10000
    Y = rng.integers(0, 2, N)
    X = rng.normal(2.0 * Y, 1.0)
    pi = np.array([1 - Y.mean(), Y.mean()])          # empirical weights,
    grid = np.linspace(-8, 12, 40001)                # matching the estimator
    p0 = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    p1 = np.exp(-0.5 * (grid - 2.0) ** 2) / np.sqrt(2 * np.pi)
    pbar = pi[0] * p0 + pi[1] * p1
    mi_target = (pi[0] * np.trapezoid(p0 * np.log(p0 / pbar), grid)
                 + pi[1] * np.trapezoid(p1 * np.log(p1 / pbar), grid))
    mi_mean, mi_std = estimate_categorical(X, Y, max_components=3)
    assert (mi_target >= mi_mean - SIGMAS * mi_std) and \
           (mi_target <= mi_mean + SIGMAS * mi_std)


def test_gaussianise_warped_marginals():
    # TC is invariant under per-variable diffeomorphisms; gaussianise must
    # recover the pre-warp sample-covariance TC from exp/cubic/tanh warps
    rng = np.random.default_rng(6)
    d = 3
    A = rng.normal(size=(d, d)) * 0.4 + np.eye(d)
    cov = A @ A.T
    Z = rng.multivariate_normal(np.zeros(d), cov, 10000)
    S = np.cov(Z.T)
    Ds = np.diag(1 / np.sqrt(np.diag(S)))
    tc_target = -0.5 * np.log(np.linalg.det(Ds @ S @ Ds))
    X = np.column_stack([np.exp(Z[:, 0]), Z[:, 1] ** 3, np.tanh(Z[:, 2])])
    tc_mean, tc_std = estimate(X, 'TC', preprocess='gaussianise', max_components=3)
    assert (tc_target >= tc_mean - SIGMAS * tc_std) and \
           (tc_target <= tc_mean + SIGMAS * tc_std)


def _three_component_data(N=10000, seed=9):
    rng = np.random.default_rng(seed)
    weights = np.array([0.3, 0.4, 0.3])
    means = np.array([[-2.0, -2.0], [0.0, 1.5], [2.5, -1.0]])
    covs = np.array([[[1.0, 0.6], [0.6, 1.0]],
                     [[0.5, -0.3], [-0.3, 0.8]],
                     [[0.9, 0.4], [0.4, 0.6]]])
    counts = rng.multinomial(N, weights)
    X = np.concatenate([rng.multivariate_normal(means[c], covs[c], counts[c])
                        for c in range(3)])
    rng.shuffle(X)
    return X


def test_aic_bic_select_sensible_k():
    # on well-separated 3-component data, AIC and BIC should both land on
    # (or immediately next to) the true component count
    X = _three_component_data()
    for metric_method in ('aic', 'bic'):
        _, _, k = estimate(X, 'MI', protocol='batched', max_components=6,
                           metric_method=metric_method, return_k=True)
        assert 3 <= k <= 4, f"{metric_method} selected k={k}"


def test_fixed_k_matches_batched():
    # fixing k to the batched-selection result must reproduce it (sigma band)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 10000)
    mi_b, std_b, k_b = estimate(X, 'MI', protocol='batched', max_components=3,
                                return_k=True)
    mi_f, std_f, k_f = estimate(X, 'MI', fixed_components_number=k_b,
                                return_k=True)
    assert k_f == k_b
    assert abs(mi_b - mi_f) <= SIGMAS * max(std_b, std_f)


def test_init_types_gaussian_analytic():
    # every ported init type must recover the analytic Gaussian MI
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 10000)
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    for init_type in ('minmax', 'kmeans', 'randomized_kmeans'):
        mi_mean, mi_std = estimate(X, 'MI', protocol='batched', max_components=3,
                                   init_type=init_type)
        assert (analytic_mi >= mi_mean - SIGMAS * mi_std) and \
               (analytic_mi <= mi_mean + SIGMAS * mi_std), init_type


def test_categorical_MI_imbalanced():
    # empirical category weights: P(Y=1)=0.25; quadrature target with the
    # true pi, MC-sampled Y so the estimator sees the empirical counts
    rng = np.random.default_rng(10)
    N = 20000
    Y = (rng.random(N) < 0.25).astype(int)
    X = rng.normal(2.0 * Y, 1.0)
    pi = np.array([1 - Y.mean(), Y.mean()])          # empirical weights
    grid = np.linspace(-8, 12, 40001)
    p0 = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    p1 = np.exp(-0.5 * (grid - 2.0) ** 2) / np.sqrt(2 * np.pi)
    pbar = pi[0] * p0 + pi[1] * p1
    mi_target = (pi[0] * np.trapezoid(p0 * np.log(p0 / pbar), grid)
                 + pi[1] * np.trapezoid(p1 * np.log(p1 / pbar), grid))
    mi_mean, mi_std = estimate_categorical(X, Y, max_components=3)
    assert (mi_target >= mi_mean - SIGMAS * mi_std) and \
           (mi_target <= mi_mean + SIGMAS * mi_std)


# ----------------------------------------------------------------------
# extended suite (added tests; the 12 above and their tolerances unchanged)
# ----------------------------------------------------------------------

def test_MI_sequential_analytic():
    # analytic Gaussian MI, but selected via GMM-MI's original sequential
    # (increment-k, threshold + patience early-stop) protocol
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 10000)
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    mi_mean, mi_std = estimate(X, 'MI', protocol='sequential', max_components=3)
    assert (analytic_mi >= mi_mean - SIGMAS * mi_std) and \
           (analytic_mi <= mi_mean + SIGMAS * mi_std)


def test_sequential_early_stop_k_regression():
    # regression pinning the k the sequential early-stop returns on the
    # 3-component data. This pins observed behaviour (identical on CPU fp64
    # and CUDA fp32); it does NOT assert the returned k is "correct".
    X = _three_component_data()
    _, _, k = estimate(X, 'MI', protocol='sequential', max_components=6,
                       return_k=True)
    assert k == 3


def test_cMI_dependent_conditioner():
    # trivariate unit-variance Gaussian with rho_XY=0.5, rho_XZ=0.6,
    # rho_YZ=0.4; here the conditioner Z is dependent on both X and Y, so
    # conditioning genuinely changes the association. Targets are the analytic
    # Gaussian values (pinned, used as given, not re-derived here):
    #   I(X;Y|Z) = 0.0671910473 nats,  I(X;Y) = 0.1438410362 nats.
    rho_xy, rho_xz, rho_yz = 0.5, 0.6, 0.4
    cov = np.array([[1.0, rho_xy, rho_xz],
                    [rho_xy, 1.0, rho_yz],
                    [rho_xz, rho_yz, 1.0]])
    rng = np.random.default_rng(11)
    X = rng.multivariate_normal([0, 0, 0], cov, 10000)
    cmi_target, mi_target = 0.0671910473, 0.1438410362
    cmi_mean, cmi_std = estimate(X, 'cMI', protocol='batched', max_components=3)
    assert (cmi_target >= cmi_mean - SIGMAS * cmi_std) and \
           (cmi_target <= cmi_mean + SIGMAS * cmi_std)
    mi_mean, mi_std = estimate(X[:, :2], 'MI', protocol='batched',
                               max_components=3)
    assert (mi_target >= mi_mean - SIGMAS * mi_std) and \
           (mi_target <= mi_mean + SIGMAS * mi_std)


def test_TC_multimodal_vs_MC_target():
    # total correlation of a genuinely multimodal 3-variable mixture: three
    # asymmetric, non-collinear components with distinct anisotropic
    # covariances. This is fit under the DEFAULT responsibility init (as in
    # test_aic_bic_select_sensible_k), with metric_method='bic' selecting the
    # true k=3; a collinear/isotropic well-separated layout would instead
    # defeat the default init and collapse to k=1. Target is the population TC
    # of the true model, from Monte Carlo at standard error ~1.4e-4 (pinned
    # ground truth, drawn from the true model, used as given):
    TC_TARGET = 1.092080
    weights = np.array([0.3, 0.4, 0.3])
    means = np.array([[-2.0, -2.0, -1.0],
                      [0.0, 1.5, 2.0],
                      [2.5, -1.0, -1.5]])
    covs = np.array([[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 0.8]],
                     [[0.6, -0.2, 0.1], [-0.2, 0.7, -0.3], [0.1, -0.3, 0.9]],
                     [[0.9, 0.4, -0.2], [0.4, 0.8, 0.2], [-0.2, 0.2, 0.7]]])
    rng = np.random.default_rng(9)
    counts = rng.multinomial(12000, weights)
    X = np.concatenate([rng.multivariate_normal(means[c], covs[c], counts[c])
                        for c in range(3)])
    rng.shuffle(X)
    tc_mean, tc_std = estimate(X, 'TC', protocol='batched', max_components=6,
                               metric_method='bic')
    assert (TC_TARGET >= tc_mean - SIGMAS * tc_std) and \
           (TC_TARGET <= tc_mean + SIGMAS * tc_std)


def test_KL_nongaussian_marginals_vs_quadrature():
    # KL between the two 1D marginals of the suite's asymmetric 3-component
    # data; each marginal is itself a genuine 3-mode (non-Gaussian) mixture.
    # Fit under the DEFAULT responsibility init with metric_method='bic'
    # selecting k=3. Targets are the true-marginal KLs from 1D quadrature to
    # 7 sig figs (pinned, used as given):
    #   KL[p(x0)||p(x1)] = 0.1725434 nats, KL[p(x1)||p(x0)] = 0.1309184 nats.
    X = _three_component_data()
    for quantity, target in (('KL_forward', 0.1725434),
                             ('KL_reverse', 0.1309184)):
        kl_mean, kl_std = estimate(X, quantity, protocol='batched',
                                   max_components=6, metric_method='bic')
        assert (target >= kl_mean - SIGMAS * kl_std) and \
               (target <= kl_mean + SIGMAS * kl_std), quantity


def test_nchunk_parity_fp64():
    # sample-dimension chunking must reproduce the single-pass path to
    # floating-point reduction order (ulp-level at fp64). Forced to fp64
    # regardless of the harness dtype, with a tight absolute tolerance;
    # device follows the harness so parity is checked on CPU and CUDA.
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 4000)
    common = dict(protocol='batched', max_components=3, n_bootstrap=20,
                  dtype=torch.float64)
    m0, s0 = estimate(X, 'MI', n_chunk=None, **common)
    m1, s1 = estimate(X, 'MI', n_chunk=512, **common)
    assert abs(m0 - m1) <= 1e-9
    assert abs(s0 - s1) <= 1e-9


def test_category_weights_uniform_regression():
    # near-balanced two-category data; category_weights='uniform' must
    # reproduce the Part-2 balanced reference MI (pinned, used as given):
    # 0.3463397165 nats.
    rng = np.random.default_rng(8)
    N = 10000
    Y = rng.integers(0, 2, N)
    X = rng.normal(2.0 * Y, 1.0)
    ref = 0.3463397165
    mi_mean, mi_std = estimate_categorical(X, Y, max_components=3,
                                           category_weights='uniform')
    assert (ref >= mi_mean - SIGMAS * mi_std) and \
           (ref <= mi_mean + SIGMAS * mi_std)


def test_categorical_MI_three_categories():
    # continuous-categorical MI with three categories; target from 1D
    # quadrature with the empirical category weights (as the estimator uses)
    rng = np.random.default_rng(12)
    N = 15000
    Y = rng.integers(0, 3, N)
    centres = np.array([0.0, 3.0, 6.0])
    X = rng.normal(centres[Y], 1.0)
    pi = np.array([(Y == c).mean() for c in range(3)])
    grid = np.linspace(-8, 14, 60001)
    ps = [np.exp(-0.5 * (grid - centres[c]) ** 2) / np.sqrt(2 * np.pi)
          for c in range(3)]
    pbar = sum(pi[c] * ps[c] for c in range(3))
    mi_target = sum(pi[c] * np.trapezoid(ps[c] * np.log(ps[c] / pbar), grid)
                    for c in range(3))
    mi_mean, mi_std = estimate_categorical(X, Y, max_components=4)
    assert (mi_target >= mi_mean - SIGMAS * mi_std) and \
           (mi_target <= mi_mean + SIGMAS * mi_std)


def test_mc_chunk_not_dividing_n_bootstrap():
    # mc_chunk=7 does not divide n_bootstrap=50 (the final MC chunk is size 1);
    # the chunked integration must still recover the analytic Gaussian MI
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 10000)
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    mi_mean, mi_std = estimate(X, 'MI', protocol='batched', max_components=3,
                               n_bootstrap=50, mc_chunk=7)
    assert (analytic_mi >= mi_mean - SIGMAS * mi_std) and \
           (analytic_mi <= mi_mean + SIGMAS * mi_std)


def test_invalid_argument_paths():
    # the three documented error paths each raise ValueError
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], np.eye(2), 200)
    Y = rng.integers(0, 2, 200)
    # gaussianise is rejected for the (non-invariant) KL quantities
    with pytest.raises(ValueError):
        estimate(X, 'KL_forward', preprocess='gaussianise', max_components=2)
    # unknown preprocess
    with pytest.raises(ValueError):
        estimate(X, 'MI', preprocess='whiten', max_components=2)
    # unknown category_weights
    with pytest.raises(ValueError):
        estimate_categorical(X[:, 0], Y, category_weights='inverse',
                             max_components=2)


def test_bootstrap_randint_fallback(monkeypatch):
    # for N above torch.multinomial's 2**24 category cap the uniform bootstrap
    # falls back to a randint histogram. Monkeypatch the cap down to exercise
    # the fallback at small N, and check it is distributionally consistent with
    # the multinomial path: both cover the analytic MI, and their means agree
    # within the combined bootstrap error.
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 5000)
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    m_mn, s_mn = estimate(X, 'MI', protocol='batched', max_components=3, seed=1)
    monkeypatch.setattr(gmm_mi_torch.estimator, '_MULTINOMIAL_MAX_CATEGORIES', 8)
    m_ri, s_ri = estimate(X, 'MI', protocol='batched', max_components=3, seed=1)
    assert abs(analytic_mi - m_mn) <= SIGMAS * s_mn
    assert abs(analytic_mi - m_ri) <= SIGMAS * s_ri
    combined = (s_mn ** 2 + s_ri ** 2) ** 0.5
    assert abs(m_mn - m_ri) <= SIGMAS * combined


@pytest.mark.skipif(not os.environ.get('GMM_MI_OFFLINE'),
                    reason='bootstrap-sigma calibration; offline only, not CI')
def test_bootstrap_sigma_calibration_offline():
    # GMM-MI-paper-style errorbar calibration. Over many independent datasets
    # the analytic MI should fall within +-1 bootstrap sigma ~68% of the time
    # and +-2 sigma ~95%. Bounds are loose (a few dozen repeats are noisy);
    # the measured coverage is reported in gb10_report.md. Not part of the CI
    # set (gated on GMM_MI_OFFLINE).
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    analytic_mi = -0.5 * np.log(1 - rho ** 2)
    n_rep = 40
    within1 = within2 = 0
    for r in range(n_rep):
        rng = np.random.default_rng(1000 + r)
        X = rng.multivariate_normal([0, 0], cov, 5000)
        m, s = estimate(X, 'MI', protocol='batched', max_components=3, seed=r)
        z = abs(m - analytic_mi) / s
        within1 += int(z <= 1.0)
        within2 += int(z <= 2.0)
    cov1, cov2 = within1 / n_rep, within2 / n_rep
    assert 0.45 <= cov1 <= 0.90, cov1
    assert 0.80 <= cov2 <= 1.0, cov2


def test_return_fit_model():
    # return_fit exposes the selected model refitted on the full data;
    # attributes must be self-consistent and samples must reproduce the
    # data's covariance. Added for the notebook plotting API.
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 5000)
    mi_mean, mi_std, k, fit = estimate(X, 'MI', max_components=3,
                                       return_k=True, return_fit=True)
    assert fit.n_components == len(fit.weights_) == k
    assert abs(fit.weights_.sum() - 1.0) < 1e-5
    assert fit.means_.shape == (k, 2)
    assert fit.covariances_.shape == (k, 2, 2)
    S = fit.sample(200_000, seed=3)
    assert S.shape == (200_000, 2)
    assert np.allclose(np.cov(S.T), cov, atol=0.05)
    assert np.isfinite(fit.score_samples(X[:100])).all()


def test_auto_chunk_sizing(monkeypatch):
    # CPU keeps the historical default (capped at the item count); CUDA takes
    # half the free memory, floored at 32 and capped at the item count
    E = gmm_mi_torch.estimator
    assert E._auto_chunk(100, 10**6, 'cpu') == 8
    assert E._auto_chunk(3, 10**6, 'cpu') == 3
    monkeypatch.setattr(torch.cuda, 'mem_get_info',
                        lambda dev=None: (2 * 10**9, 4 * 10**9))
    assert E._auto_chunk(1000, 10**7, 'cuda') == 100
    assert E._auto_chunk(1000, 10**9, 'cuda') == 32
    assert E._auto_chunk(20, 10**6, 'cuda') == 20
    with pytest.raises(ValueError):
        E._resolve_chunk('big', 10, 1, 'cpu')
    with pytest.raises(ValueError):
        E._resolve_chunk(0, 10, 1, 'cpu')


def test_em_chunk_parity_fp64():
    # em_chunk only partitions independent replicates, so 'auto' and any
    # fixed value must agree to reduction-order precision. mc_chunk is held
    # fixed so the MC sample stream is identical. Forced to fp64; device
    # follows the harness.
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 4000)
    common = dict(protocol='batched', max_components=3, n_bootstrap=20,
                  dtype=torch.float64, mc_chunk=4)
    m0, s0 = estimate(X, 'MI', em_chunk=3, **common)
    m1, s1 = estimate(X, 'MI', em_chunk='auto', **common)
    m2, s2 = estimate(X, 'MI', em_chunk=1000, **common)
    assert abs(m0 - m1) <= 1e-9 and abs(m0 - m2) <= 1e-9
    assert abs(s0 - s1) <= 1e-9 and abs(s0 - s2) <= 1e-9


def test_mc_chunk_auto_within_mc_noise():
    # same fits, possibly different MC sample stream (on CUDA 'auto' differs
    # from 8): agreement at the Monte-Carlo noise level, well inside the
    # bootstrap errorbar
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 10000)
    common = dict(protocol='batched', max_components=3, n_bootstrap=20,
                  em_chunk=8)
    m0, s0 = estimate(X, 'MI', mc_chunk=8, **common)
    m1, s1 = estimate(X, 'MI', mc_chunk='auto', **common)
    assert abs(m0 - m1) < 5e-3
    assert abs(s0 - s1) < 5e-3


def test_chunk_oom_fallback(monkeypatch):
    # a CUDA out-of-memory error inside a chunk halves the chunk and retries:
    # the result is unchanged and a RuntimeWarning is raised (CPU-only test;
    # the error is injected)
    E = gmm_mi_torch.estimator
    orig = E._em_fit_core
    calls = []

    def flaky(X, s, *args, **kwargs):
        calls.append(s.shape[0])
        if s.shape[0] > 4:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return orig(X, s, *args, **kwargs)

    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.multivariate_normal([0, 0], [[1.0, 0.6], [0.6, 1.0]], 2000))
    g = torch.Generator().manual_seed(0)
    w0, mu0, cov0 = E._init_params(X, 3, 16, 1e-6, generator=g)
    s = torch.ones(16, X.shape[0], dtype=X.dtype)
    ref = E.em_fit(X, s, w0, mu0, cov0, em_chunk=4)
    monkeypatch.setattr(E, '_em_fit_core', flaky)
    with pytest.warns(RuntimeWarning):
        out = E.em_fit(X, s, w0, mu0, cov0, em_chunk=16)
    assert calls[:3] == [16, 8, 4]
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_unstandardised_collinear_fp32_regression():
    # float32 used to abort with a Cholesky error on strongly correlated
    # unstandardised pairs (beta-tester failure mode): covariance entries of
    # order 1e4 round to an indefinite matrix, and an absolute reg_covar=1e-6
    # is 1000x too small to save it. With internal standardisation (plus the
    # escalation fallback) the fit must complete.
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 5000)
    X_raw = np.column_stack([x * 100 + 500,
                             x * 100 + 500 + rng.normal(0, 0.1, 5000)])
    m_raw, s_raw = estimate(X_raw, 'MI', dtype=torch.float32, n_bootstrap=8,
                            max_components=8)
    assert np.isfinite(m_raw) and np.isfinite(s_raw)
    assert m_raw > 3.0        # near-collinear pair: MI is large


def test_mi_affine_invariance_fp64():
    # MI is exactly invariant under per-variable affine maps; at float64 the
    # rounding of the map is far below EM's sensitivity on a well-conditioned
    # pair, so raw and rescaled inputs must agree tightly
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 5000)
    common = dict(max_components=3, n_bootstrap=10, dtype=torch.float64)
    m1, s1 = estimate(X, 'MI', **common)
    m2, s2 = estimate(X * [10.0, 0.1] + [100.0, -5.0], 'MI', **common)
    assert abs(m1 - m2) < 1e-6
    assert abs(s1 - s2) < 1e-6


def test_kl_shared_affine_invariance():
    # KL between the two 1D marginals is invariant under a common affine map
    # of both columns; the internal standardisation must therefore share one
    # scale between the columns for the KL quantities.
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0.0, 1.0, 6000),
                         rng.normal(0.5, 1.3, 6000)])
    common = dict(max_components=3, n_bootstrap=10)
    k1, _ = estimate(X, 'KL_forward', **common)
    k2, _ = estimate(X * 3.0 + 7.0, 'KL_forward', **common)
    assert k1 > 0.05          # the marginals genuinely differ
    assert abs(k1 - k2) < 5e-3


def test_return_fit_original_units():
    # the fit runs on standardised data; the returned model must be in the
    # caller's units, including strongly mismatched column scales
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], cov, 5000) * [10.0, 0.1] + [100.0, -5.0]
    _, _, fit = estimate(X, 'MI', max_components=3, n_bootstrap=5,
                         return_fit=True)
    S = fit.sample(200_000, seed=3)
    C_data, C_fit = np.cov(X.T), np.cov(S.T)
    assert np.allclose(C_fit, C_data, rtol=0.1, atol=1e-3)
    mean_fit = (fit.weights_[:, None] * fit.means_).sum(0)
    assert np.allclose(mean_fit, X.mean(0), atol=[0.5, 0.005])


def test_chol_escalation_recovers():
    # a slightly indefinite covariance must be repaired by escalating
    # diagonal regularisation, with a RuntimeWarning, not abort
    E = gmm_mi_torch.estimator
    cov = torch.tensor([[[[1.0, 1.0 + 1e-6], [1.0 + 1e-6, 1.0]]]],
                       dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        L = E._chol(cov)
    assert torch.isfinite(L).all()


def test_gemm_step_matches_legacy_step_fp64():
    # one EM step: the GEMM kernels must reproduce the legacy kernels to
    # reduction-order precision at float64 (CPU)
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(1500, 3, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    w, mu, cov = E._init_params(X, 4, 6, 1e-6, generator=g)
    s = torch.rand(6, 1500, dtype=torch.float64) + 0.5
    chol = E._chol(cov)
    logw = torch.log(w.clamp_min(1e-300))
    Xr = X.expand(6, 1500, 3)
    lj = logw.unsqueeze(1) + E.log_gauss(Xr, mu, chol)
    ll_n = torch.logsumexp(lj, -1)
    resp_l = torch.exp(lj - ll_n.unsqueeze(-1))
    ll_l = (s * ll_n).sum(1) / s.sum(1)
    wl, mul, covl = E._m_step(Xr, resp_l.clone(), s, 1e-6)
    F = E._features(X)
    resp_g, ll_g = E._estep_gemm(F, s, w, mu, cov)
    assert (resp_g - resp_l).abs().max() < 1e-12
    assert (ll_g - ll_l).abs().max() < 1e-12
    wg, mug, covg = E._mstep_gemm(F, resp_g, s, 1e-6, 3)
    assert (wg - wl).abs().max() < 1e-12
    assert (mug - mul).abs().max() < 1e-12
    assert (covg - covl).abs().max() < 1e-12
    ll_c, wc, muc, covc = E._em_step_gemm_nchunked(X, s, w, mu, cov, 1e-6, 400)
    assert (ll_c - ll_g).abs().max() < 1e-12
    assert (covc - covg).abs().max() < 1e-12


def test_gemm_legacy_estimate_parity_fp64(monkeypatch):
    # full estimate(): both kernels, same seed, must agree at float64. The
    # trajectories differ only by floating-point reduction order, so the
    # convergence stop can flip by at most a knife-edge iteration.
    E = gmm_mi_torch.estimator
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X2 = rng.multivariate_normal([0, 0], cov, 4000)
    X3 = rng.multivariate_normal([0, 0, 0], np.eye(3), 3000)
    for X, quantity, kw in [
            (X2, 'MI', dict(protocol='batched', max_components=3)),
            (X2, 'MI', dict(protocol='batched', max_components=3, n_chunk=700)),
            (X3, 'TC', dict(protocol='sequential', max_components=2)),
    ]:
        out = {}
        for kernel in ('gemm', 'legacy'):
            monkeypatch.setattr(E, '_EM_KERNEL', kernel)
            out[kernel] = estimate(X, quantity, n_bootstrap=10, return_k=True,
                                   dtype=torch.float64, **kw)
        monkeypatch.setattr(E, '_EM_KERNEL', 'gemm')
        mg, sg, kg = out['gemm']
        ml, sl, kl = out['legacy']
        assert kg == kl
        assert abs(mg - ml) < 1e-6
        assert abs(sg - sl) < 1e-6


def test_mstep_covariance_floor():
    # float32 moment accumulation can round M2 - m1^2 below the reg_covar
    # floor for a component collapsed onto near-duplicate points; the M-step
    # must restore the floor on the diagonal
    E = gmm_mi_torch.estimator
    Nk = torch.tensor([[100.0]], dtype=torch.float64)
    m1, M2 = 1.0, 1.0 - 2e-6            # variance -2e-6 before regularisation
    S = torch.tensor([[[m1, M2]]], dtype=torch.float64) * 100.0
    ssum = torch.tensor([100.0])
    w, mu, cov = E._mstep_from_moments(S, Nk, ssum, 1e-6, 1, torch.float64)
    assert cov[0, 0, 0, 0].item() >= 1e-6


def test_chol_corrupted_component_recovers():
    # covariances observed in a failing beta-test fit: a component whose
    # diagonal has gone negative (so a mean-based bump scale would be
    # negative and never heal it), and a non-finite component. Both must be
    # repaired with a RuntimeWarning, not abort.
    E = gmm_mi_torch.estimator
    neg = torch.tensor([[[[-1.59e-6, -7.6e-8], [-7.6e-8, 1.03e-6]]]],
                       dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        L = E._chol(neg)
    assert torch.isfinite(L).all()
    nan = torch.tensor([[[[float('nan'), 0.0], [0.0, 1.0]]]],
                       dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        L = E._chol(nan)
    assert torch.isfinite(L).all()


def test_gaussianise_ties_average_rank_regression():
    # review finding: positional (double-argsort) tie-breaking mapped a tied
    # block to a monotone probit ramp in row order, fabricating MI whenever
    # row order correlates with another variable. The worst case — a sorted
    # variable next to a constant column — used to return ~1.06 nats; with
    # average ranks the constant column carries no information and the MI
    # must be consistent with zero. The heavy ties must also warn.
    rng = np.random.default_rng(0)
    X = np.column_stack([np.sort(rng.normal(size=4000)), np.zeros(4000)])
    with pytest.warns(RuntimeWarning, match='tied values'):
        mi_mean, mi_std = estimate(X, 'MI', preprocess='gaussianise',
                                   max_components=3, n_bootstrap=10)
    assert abs(mi_mean) < 0.05


def test_gaussianise_matches_scipy_average_rankdata():
    # the rank-probit transform must follow scipy's 'average' rankdata
    # convention, computed at float64, on data with and without ties
    from scipy.stats import norm, rankdata
    E = gmm_mi_torch.estimator
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.integers(0, 5, 300).astype(float),
                         rng.normal(size=300)])
    with pytest.warns(RuntimeWarning, match='tied values'):
        out = E._rank_gaussianise(torch.as_tensor(X, dtype=torch.float64))
    ref = norm.ppf(np.column_stack(
        [rankdata(X[:, j], method='average') for j in range(2)])
        / (X.shape[0] + 1))
    assert np.allclose(out.cpu().numpy(), ref, atol=1e-12)


def test_return_fit_gaussianise_warns_probit_space():
    # return_fit with gaussianise cannot be returned in original units (the
    # rank transform has no parametric inverse): it must warn, and the model
    # it returns lives in rank-probit space (means ~0, not the data's ~100)
    rho = 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([100.0, 100.0], cov, 4000)
    with pytest.warns(RuntimeWarning, match='gaussianise'):
        _, _, fit = estimate(X, 'MI', preprocess='gaussianise',
                             max_components=3, n_bootstrap=5,
                             return_fit=True)
    mean_fit = (fit.weights_[:, None] * fit.means_).sum(0)
    assert np.all(np.abs(mean_fit) < 1.0)


def test_spike_pair_fp32_end_to_end():
    # beta-test regression: a pair of zero-inflated variables at float32 with
    # the tester's settings used to abort ("covariance factorisation failed
    # even after diagonal regularisation"). Needs the full-strength
    # configuration to trip, so it is skipped on plain CPU runs where it is
    # slow; the mechanisms are unit-tested above.
    if os.environ.get('GMM_MI_DEVICE') != 'cuda' and \
            not os.environ.get('GMM_MI_OFFLINE'):
        pytest.skip('full-strength run; covered on CUDA and offline suites')
    rng = np.random.default_rng(0)

    def zi(p, scale=0.1, N=5000):
        return np.where(rng.random(N) < p, 0.0,
                        np.abs(rng.normal(0, scale, N)))

    zi(0.8)
    rng.normal(0, 1, 5000)              # matches the failing realisation
    X = np.column_stack([zi(0.8), zi(0.7)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m, s = estimate(X, 'MI', dtype=torch.float32, n_bootstrap=20,
                        max_components=20, n_inits=5, n_folds=3,
                        mc_samples=10_000)
    assert np.isfinite(m) and np.isfinite(s)
