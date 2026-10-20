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
    # same fits, different MC sample stream per chunking: agreement at the
    # Monte-Carlo noise level, well inside the bootstrap errorbar. The
    # fixed chunking is 3 so it differs from 'auto' on EVERY leg (CPU
    # 'auto'=8, CUDA 'auto'=20 here): the v0.3.0 version compared
    # mc_chunk=8 against CPU 'auto'=8, which is bitwise the same run and
    # could never fail on the documented default pytest invocation.
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 10000)
    common = dict(protocol='batched', max_components=3, n_bootstrap=20,
                  em_chunk=8)
    m0, s0 = estimate(X, 'MI', mc_chunk=3, **common)
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
    # diagonal regularisation, with a RuntimeWarning, not abort; the
    # returned covariance is the repaired one, so cov == L @ L^T holds
    E = gmm_mi_torch.estimator
    cov = torch.tensor([[[[1.0, 1.0 + 1e-6], [1.0 + 1e-6, 1.0]]]],
                       dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        L, cov_r = E._chol(cov)
    assert torch.isfinite(L).all()
    assert torch.allclose(cov_r, L @ L.transpose(-2, -1), atol=1e-12)
    assert not torch.equal(cov_r, cov)


def test_gemm_step_matches_legacy_step_fp64():
    # one EM step: the GEMM kernels must reproduce the legacy kernels to
    # reduction-order precision at float64 (CPU)
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(1500, 3, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    w, mu, cov = E._init_params(X, 4, 6, 1e-6, generator=g)
    s = torch.rand(6, 1500, dtype=torch.float64) + 0.5
    chol, cov = E._chol(cov)
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
        L, _ = E._chol(neg)
    assert torch.isfinite(L).all()
    nan = torch.tensor([[[[float('nan'), 0.0], [0.0, 1.0]]]],
                       dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        L, nan_r = E._chol(nan)
    assert torch.isfinite(L).all()
    assert torch.isfinite(nan_r).all()


def test_chol_write_back_invariant_and_warning_counts():
    # the repaired covariance returned by _chol must equal L @ L^T for every
    # component (dead -> identity, non-finite -> identity, indefinite ->
    # escalated); when a batch holds both a non-finite and an indefinite
    # component, BOTH warnings must fire (the non-finite reset used to be
    # swallowed whenever escalation also ran), with per-event counts
    E = gmm_mi_torch.estimator
    cov = torch.tensor([[[[float('nan'), 0.0], [0.0, 1.0]],   # non-finite
                         [[-0.5, 0.0], [0.0, 1.0]],           # indefinite
                         [[0.0, 0.0], [0.0, 0.0]],            # dead
                         [[2.0, 0.3], [0.3, 1.0]]]],          # healthy
                       dtype=torch.float64)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        L, cov_r = E._chol(cov)
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, RuntimeWarning)]
    assert any('1 non-finite' in m for m in msgs), msgs
    assert any('1 covariance factorisation' in m for m in msgs), msgs
    assert torch.allclose(cov_r, L @ L.transpose(-2, -1), atol=1e-12)
    eye = torch.eye(2, dtype=torch.float64)
    assert torch.equal(cov_r[0, 0], eye)       # non-finite reset
    assert torch.equal(cov_r[0, 2], eye)       # dead padding
    assert torch.equal(cov_r[0, 3], cov[0, 3])  # healthy untouched


def test_mc_integrand_consistent_after_repair():
    # review finding: the MC stage sampled the joint under the REPAIRED
    # factor but scored marginals from the RAW covariance, mixing two
    # mixtures. With the write-back, an independent (diagonal) repaired
    # mixture must give an MI integrand of ~0: the marginals refactorised
    # from the written-back covariance describe the sampled joint exactly.
    E = gmm_mi_torch.estimator
    cov = torch.tensor([[[[-0.5, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    with pytest.warns(RuntimeWarning):
        chol, cov_r = E._chol(cov)
    w = torch.ones(1, 1, dtype=torch.float64)
    mu = torch.zeros(1, 1, 2, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    pts = E.gmm_sample(w, mu, chol, 20_000, generator=g)
    v = E._integrand_MI(pts, w, mu, cov_r, chol).mean().item()
    assert abs(v) < 1e-10


def test_legacy_nchunk_mstep_covariance_floor():
    # float32 rounding of the shifted moments can push a collapsed
    # component's variance below the reg_covar floor (observed live on the
    # GB10); the chunked legacy M-step must restore the floor. The inputs
    # below give a pre-fix variance of -1.2e-4 (true variance 2.5e-7).
    E = gmm_mi_torch.estimator
    X = torch.tensor([[1.0], [1.0 + 1e-3]] * 50, dtype=torch.float32)
    w = torch.ones(1, 1, dtype=torch.float32)
    mu = torch.tensor([[[-30.0]]], dtype=torch.float32)
    chol, _ = E._chol(torch.ones(1, 1, 1, 1, dtype=torch.float32))
    logw = torch.zeros(1, 1, dtype=torch.float32)
    s = torch.ones(1, X.shape[0], dtype=torch.float32)
    _, _, _, cov2 = E._em_step_nchunked(X, s, w, mu, chol, logw, 1e-6, 16)
    assert cov2[0, 0, 0, 0].item() >= np.float32(1e-6) * (1 - 1e-6)


def test_sequential_patience_reset_exhaustion_and_nan(monkeypatch):
    # review finding: the patience counter never reset on improvement and
    # the selected k assumed the misses were trailing; exhaustion returned
    # max_components regardless of scores; an all-NaN metric crashed with
    # KeyError(0). All three paths, with a scripted metric sequence.
    E = gmm_mi_torch.estimator
    X = torch.zeros(10, 2)

    def scripted(metrics):
        def stub(X, K, *args, **kwargs):
            return metrics[K - 1], (f'w{K}', f'mu{K}', f'cov{K}')
        return stub

    # improve@1, flat@2, improve@3, flat@4+: patience=2 must survive the
    # single miss at k=2 and select k=3 (the old counter returned k=2)
    monkeypatch.setattr(E, '_cv_fit_for_k',
                        scripted([1.0, 1.0, 2.0, 2.0, 2.0]))
    k, params, _ = E.select_sequential(X, max_components=5, patience=2)
    assert k == 3 and params[0] == 'w3'
    # exhaustion without an early stop: the best-scoring k wins, not
    # max_components
    monkeypatch.setattr(E, '_cv_fit_for_k', scripted([1.0, 2.0, 1.5]))
    k, params, _ = E.select_sequential(X, max_components=3, patience=5)
    assert k == 2 and params[0] == 'w2'
    # all-NaN metrics: a clear error instead of history[0]
    monkeypatch.setattr(E, '_cv_fit_for_k', scripted([float('nan')] * 3))
    with pytest.raises(RuntimeError, match='metric'):
        E.select_sequential(X, max_components=3, patience=1)


def test_seed_controls_cv_folds(monkeypatch):
    # review finding: the seed was never forwarded to _cv_masks (hardcoded
    # 42) although the docstring promises seed controls the CV splits
    E = gmm_mi_torch.estimator
    t0, v0 = E._cv_masks(100, 2, 1, torch.float64, seed=0)
    t1, v1 = E._cv_masks(100, 2, 1, torch.float64, seed=1)
    assert not torch.equal(v0, v1)
    seen = []
    orig = E._cv_masks

    def spy(N, n_folds, n_inits, dtype, seed=42):
        seen.append(seed)
        return orig(N, n_folds, n_inits, dtype, seed=seed)

    monkeypatch.setattr(E, '_cv_masks', spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    for s in (7, 8):
        estimate(X, 'MI', max_components=2, n_bootstrap=2, seed=s,
                 mc_samples=1000)
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(s != 42 for s in seen)


def test_fold_feasibility_validation():
    # degenerate CV settings must fail fast with a clear ValueError instead
    # of a NaN cascade into far-away Index/KeyErrors
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match='n_folds'):
        estimate(X, 'MI', n_folds=1, max_components=2)
    with pytest.raises(ValueError, match='n_folds'):
        estimate(X[:3], 'MI', n_folds=5, max_components=2)
    Y = np.array([0] * 48 + [1] * 2)
    with pytest.raises(ValueError, match='category'):
        estimate_categorical(X[:, 0], Y, n_folds=3, max_components=2)


def test_entry_argument_validation():
    # typos and malformed inputs raise at entry, before minutes of compute
    # (and not via bare asserts, which vanish under python -O)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    Y = rng.integers(0, 2, 200)
    with pytest.raises(ValueError, match='protocol'):
        estimate(X, 'MI', protocol='Sequential', max_components=2)
    with pytest.raises(ValueError, match='quantity'):
        estimate(X, 'mi', max_components=2)
    with pytest.raises(ValueError, match='columns'):
        estimate(np.hstack([X, X[:, :1]]), 'MI', max_components=2)
    with pytest.raises(ValueError, match='columns'):
        estimate(X, 'cMI', max_components=2)
    with pytest.raises(ValueError, match='2-D'):
        estimate(X[:, 0], 'MI', max_components=2)
    with pytest.raises(ValueError, match='n_chunk'):
        estimate(X, 'MI', n_chunk=0, max_components=2)
    with pytest.raises(ValueError, match='protocol'):
        estimate_categorical(X[:, 0], Y, protocol='batch', max_components=2)
    with pytest.raises(ValueError, match='single column'):
        estimate_categorical(X, Y, max_components=2)
    with pytest.raises(ValueError, match='same length'):
        estimate_categorical(X[:100, 0], Y, max_components=2)
    # counts and floors that used to fail deep inside EM with opaque torch
    # errors (or silently return garbage: max_iter=0 returned the random
    # init, n_bootstrap=-1 was treated as 0, reg_covar<=0 fitted collapsed
    # components) must raise at entry under the argument's own name
    with pytest.raises(ValueError, match='max_iter'):
        estimate(X, 'MI', max_iter=0, max_components=2)
    with pytest.raises(ValueError, match='reg_covar'):
        estimate(X, 'MI', reg_covar=0.0, max_components=2)
    with pytest.raises(ValueError, match='reg_covar'):
        estimate(X, 'MI', reg_covar=-1e-6, max_components=2)
    with pytest.raises(ValueError, match='mc_samples'):
        estimate(X, 'MI', mc_samples=0, max_components=2)
    with pytest.raises(TypeError, match='mc_samples'):
        estimate(X, 'MI', mc_samples=1e5, max_components=2)
    with pytest.raises(ValueError, match='max_components'):
        estimate(X, 'MI', max_components=0)
    with pytest.raises(ValueError, match='n_inits'):
        estimate(X, 'MI', n_inits=0, max_components=2)
    with pytest.raises(ValueError, match='fixed_components_number'):
        estimate(X, 'MI', fixed_components_number=0)
    with pytest.raises(ValueError, match='fixed_components_number'):
        estimate(X, 'MI', fixed_components_number=-1)
    with pytest.raises(ValueError, match='threshold_components'):
        estimate(X, 'MI', threshold_components=-1.0, max_components=2)
    with pytest.raises(ValueError, match='n_bootstrap'):
        estimate(X, 'MI', n_bootstrap=-1, max_components=2)
    with pytest.raises(TypeError, match='n_bootstrap'):
        estimate(X, 'MI', n_bootstrap=3.0, max_components=2)
    with pytest.raises(TypeError, match='n_folds'):
        estimate(X, 'MI', n_folds=2.5, max_components=2)
    with pytest.raises(ValueError, match='max_iter'):
        estimate_categorical(X[:, 0], Y, max_iter=0, max_components=2)
    with pytest.raises(TypeError, match='mc_samples'):
        estimate_categorical(X[:, 0], Y, mc_samples=1e4, max_components=2)
    with pytest.raises(ValueError, match='reg_covar'):
        estimate_categorical(X[:, 0], Y, reg_covar=0.0, max_components=2)


def test_metric_method_validated_with_fixed_k():
    # the fixed-k shortcut used to skip the protocol/metric_method
    # compatibility rules entirely, so typos ran silently
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    Y = rng.integers(0, 2, 200)
    with pytest.raises(ValueError, match='metric_method'):
        estimate(X, 'MI', fixed_components_number=2, metric_method='bogus')
    with pytest.raises(ValueError, match='metric_method'):
        estimate(X, 'MI', protocol='sequential', fixed_components_number=2,
                 metric_method='aic')
    with pytest.raises(ValueError, match='metric_method'):
        estimate_categorical(X[:, 0], Y, fixed_components_number=2,
                             metric_method='bogus')


def test_numpy_integer_seed():
    # np.int64/np.int32 seeds (what rng.integers yields in seed sweeps)
    # used to crash in _cv_masks with 'TypeError: an integer is required';
    # they must behave exactly like the equal Python int, and a float seed
    # must be rejected under the argument's name
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 300)
    kw = dict(max_components=2, n_bootstrap=2, mc_samples=1000)
    m0, s0 = estimate(X, 'MI', seed=3, **kw)
    m1, s1 = estimate(X, 'MI', seed=np.int64(3), **kw)
    assert m0 == m1 and s0 == s1
    Y = rng.integers(0, 2, 300)
    m2, _ = estimate_categorical(X[:, 0], Y, seed=np.int32(3), **kw)
    assert np.isfinite(m2)
    with pytest.raises(TypeError, match='seed'):
        estimate(X, 'MI', seed=3.0, **kw)


def test_nonfinite_input_rejected():
    # review finding: NaN/inf input was never rejected — gaussianise ranked
    # NaN rows as the largest values (fabricating MI for independent data)
    # and the 'none' path burned max_iter everywhere before a far-away
    # IndexError. Both entry points must refuse it up front.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    for bad in (np.nan, np.inf, -np.inf):
        Xb = X.copy()
        Xb[7, 1] = bad
        with pytest.raises(ValueError, match='non-finite'):
            estimate(Xb, 'MI', max_components=2)
        with pytest.raises(ValueError, match='non-finite'):
            estimate(Xb, 'MI', preprocess='gaussianise', max_components=2)
    Y = rng.integers(0, 2, 400)
    Xc = X[:, 0].copy()
    Xc[3] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        estimate_categorical(Xc, Y, max_components=2)
    Yf = Y.astype(float)
    Yf[3] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        estimate_categorical(X[:, 0], Yf, max_components=2)


def test_select_public_guards():
    # the public selection functions skipped _validate_folds, silently
    # returning k=1 with every metric -inf for infeasible folds; and
    # all-non-finite metrics hit the parsimony mask's nonzero()[0] as an
    # opaque IndexError instead of the sequential path's clear RuntimeError
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X20 = torch.randn(20, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match='n_folds'):
        E.select_batched(X20, max_components=3, n_folds=25)
    with pytest.raises(ValueError, match='n_folds'):
        E.select_sequential(X20, max_components=3, n_folds=25)
    Xn = torch.randn(300, 2, dtype=torch.float64)
    Xn[5, 0] = float('nan')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(RuntimeError, match='no finite validation metric'):
            E.select_batched(Xn, max_components=2, max_iter=20)


def test_val_ll_empty_fold_minus_inf():
    # an all-zero validation weight row used to score exactly 0.0 (clamped
    # denominator), beating any real negative log-likelihood in the fold
    # argmax; it must lose instead
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(50, 2, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    w, mu, cov = E._init_params(X, 2, 2, 1e-6, generator=g)
    s = torch.ones(2, 50, dtype=torch.float64)
    s[1] = 0.0
    v = E.val_ll(X, s, w, mu, cov)
    assert torch.isfinite(v[0])
    assert v[1].item() == -float('inf')


def test_patience_exposed_by_estimate():
    # patience used not to be plumbed through estimate() at all
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 2000)
    m, s = estimate(X, 'MI', protocol='sequential', max_components=3,
                    patience=2, n_bootstrap=2, mc_samples=1000)
    assert np.isfinite(m)


def test_categorical_selection_pipeline_shared():
    # the categorical selection pipeline is now the same code as estimate's:
    # sequential + metric_method raises (it used to be silently dropped and
    # ran 'valid' selection), and threshold_components / patience are
    # exposed instead of hardcoded
    rng = np.random.default_rng(0)
    N = 600
    Y = rng.integers(0, 2, N)
    X = rng.normal(2.0 * Y, 1.0)
    with pytest.raises(ValueError, match='metric_method'):
        estimate_categorical(X, Y, protocol='sequential',
                             metric_method='bic', max_components=2)
    with pytest.raises(ValueError, match='metric_method'):
        estimate_categorical(X, Y, metric_method='aicc', max_components=2)
    m, _ = estimate_categorical(X, Y, protocol='sequential',
                                max_components=2, threshold_components=1e-4,
                                patience=2, n_bootstrap=2, mc_samples=2000)
    assert np.isfinite(m)


def test_single_bootstrap_std_is_none():
    # n_bootstrap=1 used to return std = NaN (sample std of one element);
    # one replicate has no sample deviation, so std is None as for 0
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 1000)
    m, s = estimate(X, 'MI', max_components=2, n_bootstrap=1,
                    mc_samples=2000)
    assert np.isfinite(m) and s is None
    Y = rng.integers(0, 2, 1000)
    Xc = rng.normal(2.0 * Y, 1.0)
    m, s = estimate_categorical(Xc, Y, max_components=2, n_bootstrap=1,
                                mc_samples=2000)
    assert np.isfinite(m) and s is None


def test_categorical_bootstrap_randint_fallback(monkeypatch):
    # the categorical bootstrap used to lack the >2^24 multinomial fallback
    # estimate has (it now lives in the shared _bootstrap_counts); as in
    # test_bootstrap_randint_fallback, the two paths must agree within the
    # combined bootstrap error
    rng = np.random.default_rng(8)
    N = 4000
    Y = rng.integers(0, 2, N)
    X = rng.normal(2.0 * Y, 1.0)
    common = dict(max_components=3, n_bootstrap=20, mc_samples=20_000, seed=1)
    m_mn, s_mn = estimate_categorical(X, Y, **common)
    monkeypatch.setattr(gmm_mi_torch.estimator,
                        '_MULTINOMIAL_MAX_CATEGORIES', 8)
    m_ri, s_ri = estimate_categorical(X, Y, **common)
    combined = (s_mn ** 2 + s_ri ** 2) ** 0.5
    assert abs(m_mn - m_ri) <= SIGMAS * combined


def test_selection_slices_dead_padding(monkeypatch):
    # the batched protocol returns parameters padded to K=max_components;
    # the dead components used to ride through the bootstrap, refit and MC
    # stages (~6.7x waste at k=3/max=20) and shrink the auto-sized chunks.
    # Every em_fit call after selection must run at the alive count.
    E = gmm_mi_torch.estimator
    seen_K = []
    orig = E.em_fit

    def spy(X, s, w, mu, cov, *args, **kwargs):
        seen_K.append(w.shape[1])
        return orig(X, s, w, mu, cov, *args, **kwargs)

    monkeypatch.setattr(E, 'em_fit', spy)
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    X = rng.multivariate_normal([0, 0], cov, 2000)
    _, _, k, fit = estimate(X, 'MI', protocol='batched', max_components=6,
                            n_bootstrap=3, mc_samples=2000, return_k=True,
                            return_fit=True)
    assert seen_K[0] == 6                      # selection fits all k at once
    assert all(K <= k for K in seen_K[1:])     # refit + bootstrap: alive only
    assert fit.n_components <= k


def test_val_ll_replicate_chunking_and_oom_fallback(monkeypatch):
    # val_ll used to score all replicates in one unguarded shot, outside the
    # em_chunk batching and the OOM retry: chunked scoring must be exact,
    # and an injected OOM must halve the chunk and recover the same result
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(500, 2, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    w, mu, cov = E._init_params(X, 3, 12, 1e-6, generator=g)
    s = torch.rand(12, 500, dtype=torch.float64)
    ref = E.val_ll(X, s, w, mu, cov)
    assert torch.allclose(ref, E.val_ll(X, s, w, mu, cov, em_chunk=5),
                          atol=1e-12)
    calls = []
    orig = E._val_ll_core

    def flaky(X, s_val, *args, **kwargs):
        calls.append(s_val.shape[0])
        if s_val.shape[0] > 4:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return orig(X, s_val, *args, **kwargs)

    monkeypatch.setattr(E, '_val_ll_core', flaky)
    with pytest.warns(RuntimeWarning):
        out = E.val_ll(X, s, w, mu, cov, em_chunk=12)
    assert calls[:3] == [12, 6, 3]
    assert torch.allclose(ref, out, atol=1e-12)


def test_init_params_oom_fallback(monkeypatch):
    # _init_params' replicate chunking now runs under the same OOM guard as
    # em_fit: an injected OOM halves the chunk; results are chunk-invariant
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(1000, 2, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    ref = E._init_params(X, 3, 16, 1e-6, generator=g, em_chunk=4)
    calls = []
    orig = E._m_step

    def flaky(Xr, resp, s, reg):
        calls.append(resp.shape[0])
        if resp.shape[0] > 4:
            raise torch.cuda.OutOfMemoryError("synthetic")
        return orig(Xr, resp, s, reg)

    monkeypatch.setattr(E, '_m_step', flaky)
    g = torch.Generator().manual_seed(0)
    with pytest.warns(RuntimeWarning):
        out = E._init_params(X, 3, 16, 1e-6, generator=g, em_chunk=16)
    assert calls[:3] == [16, 8, 4]
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_full_precision_pinned_at_kernel_boundary(monkeypatch):
    # direct em_fit/val_ll calls must run under full-precision float32
    # matmuls even when the ambient setting is TF32, and restore the
    # ambient setting afterwards (estimate was already pinned; the kernels
    # used to run under whatever the process had set). On torch >= 2.9 the
    # pin lives in the per-backend fp32_precision settings (the legacy
    # getter reports the ambient legacy value even while pinned), so the
    # probe reads whichever settings the pin actually drives.
    E = gmm_mi_torch.estimator
    seen = []
    orig = E._em_fit_core

    def pinned_state():
        if E._FP32_KNOBS is not None:
            return tuple(k.fp32_precision for k in E._FP32_KNOBS)
        return torch.get_float32_matmul_precision()

    target = ('ieee',) * 3 if E._FP32_KNOBS is not None else 'highest'

    def probe(*args, **kwargs):
        seen.append(pinned_state())
        return orig(*args, **kwargs)

    monkeypatch.setattr(E, '_em_fit_core', probe)
    torch.manual_seed(0)
    X = torch.randn(300, 2, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    w, mu, cov = E._init_params(X, 2, 2, 1e-6, generator=g)
    s = torch.ones(2, 300, dtype=torch.float32)
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('medium')
        E.em_fit(X, s, w, mu, cov)
        assert seen and all(p == target for p in seen)
        assert torch.get_float32_matmul_precision() == 'medium'
        v = E.val_ll(X, s, w, mu, cov)
        assert torch.isfinite(v).all()
        assert torch.get_float32_matmul_precision() == 'medium'
    finally:
        torch.set_float32_matmul_precision(prev)


def test_entry_points_survive_new_fp32_api():
    # review finding: once torch >= 2.9's fp32_precision API had been used
    # anywhere in the process, _full_precision's legacy getter raised
    # RuntimeError ('mix of the legacy and new APIs') at EVERY public entry
    # point, even on CPU. The pin must work through the new API and restore
    # the caller's settings exactly (the legacy-setter restore also used to
    # rewrite the mkldnn backend a CUDA-only caller never touched).
    E = gmm_mi_torch.estimator
    if E._FP32_KNOBS is None:
        pytest.skip('torch < 2.9: no fp32_precision API')
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 300)
    Y = rng.integers(0, 2, 300)
    saved = torch.backends.cuda.matmul.fp32_precision
    mkldnn_before = torch.backends.mkldnn.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        # the ambient state that used to crash estimate() on CPU:
        with pytest.raises(RuntimeError):
            torch.get_float32_matmul_precision()
        m, _ = estimate(X, 'MI', max_components=2, n_bootstrap=2,
                        mc_samples=1000)
        assert np.isfinite(m)
        m2, _ = estimate_categorical(X[:, 0], Y, max_components=2,
                                     n_bootstrap=2, mc_samples=1000)
        assert np.isfinite(m2)
        # the caller's own settings survive: cuda.matmul still tf32,
        # mkldnn untouched
        assert torch.backends.cuda.matmul.fp32_precision == 'tf32'
        assert torch.backends.mkldnn.matmul.fp32_precision == mkldnn_before
    finally:
        torch.backends.cuda.matmul.fp32_precision = saved


def test_full_precision_tf32_gemm_result_cuda(monkeypatch):
    # asserts a TF32-sensitive GEMM RESULT inside em_fit rather than a
    # precision flag string (which cannot see whether CUDA GEMMs actually
    # run at full precision), under both the legacy API and (when torch has
    # it) the per-backend fp32_precision API. The probe matrix of
    # 1 + 2**-12 entries rounds to exactly 1 under TF32's 10-bit mantissa,
    # so reduced precision shows up as an O(0.1) error against the fp64
    # reference.
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA: TF32 is a CUDA tensor-core mode')
    E = gmm_mi_torch.estimator
    a = torch.full((256, 256), 1.0 + 2 ** -12, device='cuda')
    ref = (a.double() @ a.double()).float()

    def gemm_err():
        return (a @ a - ref).abs().max().item()

    seen = []
    orig = E._em_fit_core

    def probe(*args, **kwargs):
        seen.append(gemm_err())
        return orig(*args, **kwargs)

    monkeypatch.setattr(E, '_em_fit_core', probe)
    torch.manual_seed(0)
    X = torch.randn(400, 2, dtype=torch.float32, device='cuda')
    g = torch.Generator(device='cuda').manual_seed(0)
    w, mu, cov = E._init_params(X, 2, 2, 1e-6, generator=g)
    s = torch.ones(2, 400, dtype=torch.float32, device='cuda')

    # (a) legacy API: 'high' = TF32 matmuls
    prev = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision('high')
        ambient = gemm_err()
        if ambient <= 1e-2:
            pytest.skip('device does not run TF32 matmuls; probe is blind')
        seen.clear()
        E.em_fit(X, s, w, mu, cov)
        assert seen and max(seen) < 1e-3     # full precision inside the pin
        assert gemm_err() > 1e-2             # degraded ambient mode restored
    finally:
        torch.set_float32_matmul_precision(prev)

    # (b) per-backend API (torch >= 2.9)
    if E._FP32_KNOBS is not None:
        saved = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            assert gemm_err() > 1e-2
            seen.clear()
            E.em_fit(X, s, w, mu, cov)
            assert seen and max(seen) < 1e-3
            assert gemm_err() > 1e-2
        finally:
            torch.backends.cuda.matmul.fp32_precision = saved


def test_dataframe_input_accepted():
    # the docstrings promise 'array-like' but torch.as_tensor cannot
    # inspect a pandas DataFrame ("could not determine the shape of object
    # type 'DataFrame'"); the entry points unwrap __array__ containers, so
    # a DataFrame must give bitwise the numpy result
    pd = pytest.importorskip('pandas')
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 1000)
    kw = dict(max_components=2, n_bootstrap=2, mc_samples=2000)
    m_np, s_np = estimate(X, 'MI', **kw)
    m_df, s_df = estimate(pd.DataFrame(X, columns=['a', 'b']), 'MI', **kw)
    assert m_df == m_np and s_df == s_np
    Y = rng.integers(0, 2, 1000)
    x0 = X[:, 0].copy()      # same memory layout for both calls: a strided
    # column and a Series copy reduce in different orders at the ulp level
    m_c, _ = estimate_categorical(pd.Series(x0), pd.Series(Y), **kw)
    m_n, _ = estimate_categorical(x0, Y, **kw)
    assert m_c == m_n


def test_string_labels_estimate_categorical():
    # string/object labels (morphological class, survey field) used to die
    # inside torch.as_tensor with a raw TypeError; they are factorised via
    # numpy's unique. Labels whose sorted order matches the integer codes
    # must give bitwise the integer-label result, and the degenerate-fold
    # error must name the offending label, not a numeric code.
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 900)
    Y = rng.integers(0, 3, 900)
    names = np.array(['a0', 'b1', 'c2'])     # sorted order == code order
    kw = dict(max_components=2, n_bootstrap=2, mc_samples=2000)
    m_int, s_int = estimate_categorical(X[:, 0], Y, **kw)
    m_str, s_str = estimate_categorical(X[:, 0], names[Y], **kw)
    assert m_str == m_int and s_str == s_int
    m_lst, _ = estimate_categorical(X[:, 0], list(names[Y]), **kw)
    assert m_lst == m_int
    Ysmall = np.array(['rare'] + ['common'] * 399)
    with pytest.raises(ValueError, match='rare'):
        estimate_categorical(X[:400, 0], Ysmall, max_components=2)


def test_return_k_reports_live_components(monkeypatch):
    # review finding: v0.3.0's alive-slice dropped components that died
    # during the CV fit without adjusting the reported k, so return_k said
    # e.g. 6 while the model actually bootstrapped, MC-integrated and
    # returned had 5, with no warning. k must report the live count, and a
    # requested fixed_components_number that lost components must warn.
    E = gmm_mi_torch.estimator
    orig = E._cv_fit_for_k

    def kill_one(*args, **kwargs):
        metric, (w, mu, cov) = orig(*args, **kwargs)
        w = w.clone()
        w[0] = 0.0                      # one dead component, as EM leaves it
        w = w / w.sum()
        return metric, (w, mu, cov)

    monkeypatch.setattr(E, '_cv_fit_for_k', kill_one)
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 400)
    with pytest.warns(RuntimeWarning, match='died during the CV fit'):
        m, s, k, fit = estimate(X, 'MI', fixed_components_number=3,
                                n_bootstrap=2, mc_samples=2000,
                                return_k=True, return_fit=True)
    assert k == 2
    assert fit.n_components <= 2
    assert np.isfinite(m)


def test_warning_stacklevels_attribute_to_caller():
    # the gaussianise tie warning and the return_fit warning carried a
    # stacklevel computed without the entry decorator's frame, so both
    # reported estimator.py's wrapper line as their origin; Python's
    # once-per-location filter then collapsed distinct call sites and a
    # module-keyed warnings filter could not match the caller. They must
    # attribute to the caller's own frame.
    E = gmm_mi_torch.estimator
    rng = np.random.default_rng(0)
    Xq = np.column_stack([rng.integers(0, 3, 400).astype(float),
                          rng.normal(size=400)])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        E.estimate(Xq, 'MI', preprocess='gaussianise', return_fit=True,
                   fixed_components_number=1, n_bootstrap=0, mc_samples=1000)
    mine = [w for w in wl if w.category is RuntimeWarning
            and ('tied values' in str(w.message)
                 or 'return_fit' in str(w.message))]
    assert len(mine) >= 2               # the tie and the return_fit warning
    for w in mine:
        assert os.path.basename(w.filename) == os.path.basename(__file__), \
            f"warning attributed to {w.filename}:{w.lineno}"


def test_requires_grad_input_detached(monkeypatch):
    # review finding: an input tensor with requires_grad=True built and
    # retained the full EM autograd graph (peak memory several times the
    # no-grad run, growing with iteration count) and return_fit=True then
    # crashed in FittedGMM on .numpy(). The entry points now run under
    # torch.no_grad() and detach the input; the caller's tensor is untouched.
    E = gmm_mi_torch.estimator
    rng = np.random.default_rng(0)
    Xg = torch.as_tensor(
        rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 500)
    ).requires_grad_(True)
    grad_seen = []
    orig = E._em_fit_core

    def probe(*args, **kwargs):
        grad_seen.append(torch.is_grad_enabled())
        return orig(*args, **kwargs)

    monkeypatch.setattr(E, '_em_fit_core', probe)
    m, s, fit = estimate(Xg, 'MI', max_components=2, n_bootstrap=2,
                         mc_samples=2000, return_fit=True)
    assert np.isfinite(m) and np.isfinite(fit.weights_).all()
    assert grad_seen and not any(grad_seen)
    assert Xg.requires_grad and Xg.grad is None
    Y = rng.integers(0, 2, 500)
    m2, _ = estimate_categorical(Xg[:, 0], Y, max_components=2,
                                 n_bootstrap=2, mc_samples=2000)
    assert np.isfinite(m2)


def test_fp32_large_offset_cast_ordering():
    # review finding: the cast to the working dtype preceded the internal
    # standardisation, so float64 data with offset/sigma >~ 1e6 was quantised
    # at float32 before the centring could protect it (MI -18% at 1e7,
    # destroyed at 1e8, RuntimeError at timestamp scale). Standardising at
    # float64 first, float32 must now track float64 at every offset.
    rho = 0.9
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(0)
    Z = rng.multivariate_normal([0, 0], cov, 2000)
    common = dict(fixed_components_number=1, n_bootstrap=5, mc_samples=20_000)
    for offset in (0.0, 1e6, 1e7, 1e8, 1.7e9):
        X = Z + offset
        m64, _ = estimate(X, 'MI', dtype=torch.float64, **common)
        m32, _ = estimate(X, 'MI', dtype=torch.float32, **common)
        assert abs(m32 - m64) < 0.02, (offset, m32, m64)


def test_entry_pipeline_runs_at_float64(monkeypatch):
    # the rank transform must see float64 data even when dtype=float32:
    # pre-cast input made the tie warning fire spuriously on continuous
    # float32 data at large N and quantised the quantiles
    E = gmm_mi_torch.estimator
    seen = []
    orig = E._rank_gaussianise

    def spy(X, *args, **kwargs):
        seen.append(X.dtype)
        return orig(X, *args, **kwargs)

    monkeypatch.setattr(E, '_rank_gaussianise', spy)
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 500)
    estimate(X, 'MI', preprocess='gaussianise', dtype=torch.float32,
             max_components=2, n_bootstrap=2, mc_samples=1000)
    assert seen == [torch.float64]


def test_tie_warning_ignores_micro_blocks():
    # float32-NATIVE continuous data acquires birthday-pair ties at large N
    # (~2% of samples at N=3e6); pairs are harmless and must not warn, while
    # genuinely quantised data (few-level column) still must
    E = gmm_mi_torch.estimator
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3_000_000, 1, dtype=torch.float64, generator=g)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        E._rank_gaussianise(x.to(torch.float32))   # pairs only: silent
    q = torch.randint(0, 5, (10_000, 1), generator=g).double()
    with pytest.warns(RuntimeWarning, match='tied values'):
        E._rank_gaussianise(q)


def test_integer_input_and_dtype_validation():
    # integer input is promoted to float64 at entry (as_tensor would hand
    # the standardisation an int64 tensor, whose .std() raises); fp16/bf16
    # working dtypes are rejected up front (cholesky unsupported deep inside)
    rng = np.random.default_rng(0)
    Xi = rng.integers(0, 1000, size=(400, 2))
    m, s = estimate(Xi, 'MI', max_components=2, n_bootstrap=2,
                    mc_samples=1000)
    assert np.isfinite(m)
    with pytest.raises(ValueError, match='dtype'):
        estimate(Xi.astype(float), 'MI', dtype=torch.float16,
                 max_components=2)


def _scaled_pair(scale, rho=0.8, N=600, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal([0, 0], cov, N) * scale


def test_return_fit_tiny_scale_units():
    # review finding: FittedGMM factorised the ORIGINAL-units covariance, so
    # columns with std ~1e-21 (strain-scale data) tripped the absolute
    # dead test and silently returned identity covariances. The repair now
    # happens in standardised units and the back-transform runs at float64.
    X = _scaled_pair(1e-21)
    _, _, fit = estimate(X, 'MI', n_bootstrap=0, fixed_components_number=1,
                         return_fit=True)
    assert np.allclose(fit.covariances_[0], np.cov(X.T), rtol=0.2)
    s = fit.score_samples(X[:50])
    assert np.isfinite(s).all() and s.std() > 0
    S = fit.sample(10_000, seed=1)
    assert abs(float(np.log10(S.std(0)).mean()) + 21.0) < 0.5


def test_return_fit_mixed_scale_repair():
    # with reg_covar below float32 resolution and a near-collinear pair at
    # scales (1e-4, 1), a repair at FittedGMM construction used to add
    # 1e-6 x the LARGEST original-unit variance to every diagonal entry,
    # inflating the small column's variance ~100x; repaired in standardised
    # units, the small column keeps its true ~1e-8 variance
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    y = x + 1e-5 * rng.normal(size=2000)
    X = np.column_stack([x * 1e-4, y])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, _, fit = estimate(X, 'MI', n_bootstrap=0,
                             fixed_components_number=1, return_fit=True,
                             dtype=torch.float32, reg_covar=1e-9)
    assert fit.covariances_[0][0, 0] < 5e-8


def test_return_fit_huge_scale_fp32():
    # float32 data at |x| ~ 1e20: the old working-dtype back-transform
    # overflowed to inf and the fit came back as identity covariances under
    # a generic non-finite warning; the float64 back-transform keeps the
    # true ~1e40 covariances
    X = _scaled_pair(1e20, rho=0.9)
    _, _, fit = estimate(X, 'MI', n_bootstrap=0, fixed_components_number=1,
                         return_fit=True, dtype=torch.float32)
    assert np.isfinite(fit.covariances_).all()
    assert fit.covariances_[0][0, 0] > 1e38


def test_mc_chunk_bitwise_invariance():
    # review finding: the MC sample stream used to be sliced per replicate
    # chunk, and mc_chunk='auto' is sized from free device memory at call
    # time, so a fixed seed did not reproduce on a busy GPU. Replicates now
    # draw from their own seeded streams: results are independent of
    # mc_chunk (bitwise on CPU; on CUDA the integrand's batched reductions
    # keep same-seed agreement at fp noise).
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 3000)
    common = dict(protocol='batched', max_components=2, n_bootstrap=8,
                  mc_samples=5000, seed=3, em_chunk=8)
    ref = estimate(X, 'MI', mc_chunk=8, **common)
    for c in (1, 3, 'auto'):
        out = estimate(X, 'MI', mc_chunk=c, **common)
        if os.environ.get('GMM_MI_DEVICE', 'cpu') == 'cpu':
            assert out == ref, c
        else:
            assert abs(out[0] - ref[0]) < 1e-4, c
            assert abs(out[1] - ref[1]) < 1e-4, c


def test_mc_oom_retry_stream_invariant(monkeypatch):
    # an out-of-memory retry inside the MC stage used to re-draw a different
    # sample stream; with per-replicate generators the retried run must
    # return exactly the uncontended result
    E = gmm_mi_torch.estimator
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 2000)
    common = dict(max_components=2, n_bootstrap=8, mc_samples=4000, seed=1,
                  em_chunk=8, mc_chunk=8, device='cpu', dtype=torch.float64)
    ref = estimate(X, 'MI', **common)
    state = {'failed': False}
    orig = E.gmm_score

    def flaky(pts, w, mu, chol):
        if not state['failed'] and pts.shape[0] > 2:
            state['failed'] = True
            raise torch.cuda.OutOfMemoryError('synthetic')
        return orig(pts, w, mu, chol)

    monkeypatch.setattr(E, 'gmm_score', flaky)
    with pytest.warns(RuntimeWarning, match='retrying'):
        out = estimate(X, 'MI', **common)
    assert out == ref


def test_seed_and_bootstrap_bounds():
    # the per-replicate stream offsets require |seed| <= 2**62 and
    # n_bootstrap <= the category stride; both are rejected at entry
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], np.eye(2), 100)
    with pytest.raises(ValueError, match='seed'):
        estimate(X, 'MI', seed=1 << 63, max_components=2)
    with pytest.raises(ValueError, match='n_bootstrap'):
        estimate(X, 'MI', n_bootstrap=(1 << 20) + 1, max_components=2)


def test_autocast_suspended_inside_pin():
    # an ambient torch.autocast region used to run the cancellation-critical
    # float32 GEMMs at bf16 (shifting both the estimate and the selected k);
    # the precision pin now suspends autocast, so inside == outside bitwise
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 2000)
    common = dict(max_components=3, n_bootstrap=4, mc_samples=5000,
                  dtype=torch.float32, device='cpu', return_k=True)
    ref = estimate(X, 'MI', **common)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = estimate(X, 'MI', **common)
    assert out == ref


def test_max_iter_exhaustion_warns():
    # EM used to exhaust max_iter silently (the sklearn-based reference
    # emits a ConvergenceWarning); em_fit now warns once per call with the
    # straggler count, and a converged fit stays silent
    E = gmm_mi_torch.estimator
    torch.manual_seed(0)
    X = torch.randn(400, 2, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    w0, mu0, cov0 = E._init_params(X, 3, 4, 1e-6, generator=g)
    s = torch.ones(4, 400, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match='did not converge'):
        E.em_fit(X, s, w0, mu0, cov0, max_iter=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        E.em_fit(X, s, w0, mu0, cov0, max_iter=500)


def test_cv_masks_survive_default_device():
    # torch.set_default_device('cuda') used to break _cv_masks (CPU
    # generator vs default-device factories); the factories are pinned
    if not torch.cuda.is_available():
        pytest.skip('needs a CUDA device to change the default to')
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], 500)
    try:
        torch.set_default_device('cuda')
        m, s = estimate(X, 'MI', max_components=2, n_bootstrap=2,
                        mc_samples=1000, device='cuda', dtype=torch.float32)
    finally:
        torch.set_default_device(None)
    assert np.isfinite(m)


def test_plotting_one_sigma_contour_constant():
    # the 1-sigma ellipse constant was chi2_2.ppf(0.32) = 0.77 (an inverted
    # quantile enclosing 32% of the mass); it must be the 68.27% quantile
    from gmm_mi_torch import plotting
    from scipy.stats import chi2
    r1, r2, r3 = plotting.find_contours([1, 2, 3])
    assert abs(r1 - np.sqrt(chi2.ppf(0.6827, 2))) < 1e-3
    assert abs(r2 - np.sqrt(chi2.ppf(0.95, 2))) < 1e-3
    assert abs(r3 - np.sqrt(chi2.ppf(0.995, 2))) < 1e-2


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
