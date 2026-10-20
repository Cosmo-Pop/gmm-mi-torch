# GB10 report: GPU benchmark and fp32 validation of `torch_gmm_mi`

**Date:** 2026-07-04
**Machine:** NVIDIA GB10 (DGX Spark), driver 580.126.09, CUDA 13.0, ~128 GB unified memory
**Stack:** torch 2.11.0+cu128, numpy 2.4.4, scipy 1.17.1, pytest 9.1.1, Python 3.12.3

## TL;DR

- **All 5 tests pass on every config (CPU/CUDA × fp64/fp32) with zero code fixes.** No device-placement errors on CUDA; fp32 passed at the default `reg=1e-6`, so the Cholesky-conditioning fallback (`reg=1e-5`) was never needed.
- **Best GPU config (fp32, tuned chunks) beats the sklearn CPU reference by 17× on selection and 59× on bootstrap** (0.90 s / 0.17 s vs 15.4 s / 10.1 s).
- **Caveat for this box: GB10's fp64 CUDA throughput is heavily rate-limited — fp64 on GPU is *slower* than fp64 on CPU** (22.5 s vs 11.5 s selection, even fully batched). If fp64 end-to-end matters, run it on CPU; the GPU case here is fp32.
- Only module changes: the chunk-parameter plumbing requested in the brief (defaults preserve old behaviour exactly).

## 1. Test matrix

Suite run unchanged (device/dtype routed via a new test-only `conftest.py` shim; defaults reproduce the original CPU/fp64 run).

| Config | Result | Wall time |
|---|---|---|
| CPU fp64 (unchanged suite — parity baseline) | 5/5 pass | 5.1–5.6 s |
| CUDA fp64 | 5/5 pass | 9.0 s |
| CUDA fp32, `reg=1e-6` | 5/5 pass | 2.0 s |
| CPU fp32 | 5/5 pass | 3.1 s |

Nothing broke at fp32, so there is no sigma-deviation to report. On the benchmark workload the fp32 and fp64 MI estimates agree to ~0.1σ (0.55902 vs 0.55911, σ ≈ 0.005).

## 2. Benchmark: 3-component mixture, N=20 000, max_k=8, n_bootstrap=50

The transcript's exact mixture construction was not available in the working directory, so a representative well-separated 3-component bivariate mixture was used (weights 0.3/0.4/0.3, distinct means, correlated covariances; construction in `bench.py`). All configs select k=3 and agree on MI ≈ 0.559 ± 0.005. Warm timings (one small warmup run to load CUDA kernels).

| Config (em_chunk/mc_chunk) | `select_batched` | bootstrap phase |
|---|---|---|
| sklearn reference (CPU) | 15.4 s | 10.1 s |
| CPU fp64 (8/8) | 11.5 s | 1.9 s |
| CPU fp32 (8/8) | 4.6 s | 0.95 s |
| CUDA fp64 (8/8) | 38.6 s | 4.0 s |
| CUDA fp64 (16/16) | 28.7 s | 3.0 s |
| CUDA fp64 (48/48) | 22.5 s | 2.4 s |
| CUDA fp32 (8/8) | 1.9 s | 0.24 s |
| CUDA fp32 (16/16) | 1.3 s | 0.20 s |
| **CUDA fp32 (48/48)** | **0.90 s** | **0.17 s** |

The torch port on CPU/fp64 already beats sklearn (the batched-replicate EM does in one loop what sklearn does across inits × folds). The chunk sweep shows chunk=8 serialises the 48 selection replicates into 6 kernel launches; chunk=48 batches them all and roughly halves GPU wall time.

## 3. Scale test: N=200 000, d=8 correlated Gaussian, TC, n_bootstrap=50

8-dim correlated Gaussian (A = 0.3·N(0,1) + I, cov = AAᵀ). Selection picks k=1 (correct — the data is a single Gaussian) and TC matches the sample-covariance analytic value **4.56739** well within 1σ in every run.

| Config (em_chunk/mc_chunk) | select | bootstrap | total | peak GPU mem | TC |
|---|---|---|---|---|---|
| CUDA fp32 (8/8) | 2.2 s | 1.9 s | 4.1 s | 9.86 GiB | 4.5675 ± 0.0065 |
| CUDA fp32 (16/16) | 2.0 s | 1.7 s | 3.7 s | 9.86 GiB | 4.5677 ± 0.0075 |
| CUDA fp32 (48/25) | 1.8 s | 1.5 s | 3.3 s | 10.15 GiB | 4.5677 ± 0.0063 |
| CUDA fp32 (48/50) | 1.8 s | 1.5 s | 3.3 s | 10.15 GiB | 4.5677 ± 0.0064 |
| CUDA fp64 (8/8) | 28.4 s | 27.0 s | 55.4 s | 19.71 GiB | 4.5668 ± 0.0075 |
| CUDA fp64 (48/50) | 18.0 s | 16.5 s | 34.5 s | 20.28 GiB | 4.5671 ± 0.0083 |

(CPU at this scale was not run: extrapolating from the N=20 000, d=2 timings it would take tens of minutes and the brief's scale target is GPU memory/tuning.)

**Chunk-tuning finding.** Larger chunks give ~20–35 % speedup at this scale, but peak memory is *insensitive* to both knobs: the high-water mark is set by `_init_params`/`_m_step` running un-chunked over all 48 selection replicates (the `(R,N,K,d)` diff tensor and its einsum workspace). At ~10 GiB (fp32) / ~20 GiB (fp64) out of 128 GB unified memory this does not bind, so it was left alone per the "EM internals only if needed" constraint. **If a future workload does bind on memory, chunking the init M-step is the lever** — not `em_chunk`/`mc_chunk`. The MC chunk did need exposing for the tuning sweep, so it is now a parameter (`mc_chunk`, default 8 = old behaviour).

## 4. Recommended settings on GB10

```python
estimate(X, ..., device='cuda', dtype=torch.float32,
         em_chunk=48, mc_chunk=50)  # em_chunk >= max_components * n_inits * n_folds
```

fp32 is validated (§1) and 20–25× faster than fp64 on this GPU; set `em_chunk` to the full replicate count so EM runs as a single batch.

## 5. Changes

`torch_gmm_mi.py` (parameter plumbing only; all defaults reproduce the old behaviour exactly — the selection rule, estimator integrands, and test tolerances are untouched):

- `estimate()` gains `em_chunk=8, mc_chunk=8`; the hardcoded MC chunk of 8 now uses `mc_chunk`.
- `select_batched` / `select_sequential` / `_cv_fit_for_k` gain a `chunk=8` passthrough to `em_fit`.

`conftest.py` (new, test harness only): routes the unchanged suite to a device/dtype via `GMM_MI_DEVICE` / `GMM_MI_DTYPE` / `GMM_MI_REG` env vars.

```diff
--- torch_gmm_mi.py (original)
+++ torch_gmm_mi.py
@@ -212,14 +212,15 @@
     return train, val
 
 
-def _cv_fit_for_k(X, K, n_folds, n_inits, reg, tol, max_iter, seed):
+def _cv_fit_for_k(X, K, n_folds, n_inits, reg, tol, max_iter, seed, chunk=8):
     """CV fits for a single k, batched over inits x folds."""
     g = torch.Generator(device=X.device).manual_seed(seed + K)
     train, val = _cv_masks(X.shape[0], n_folds, n_inits, X.dtype)
     train, val = train.to(X.device), val.to(X.device)
     R = train.shape[0]
     w0, mu0, cov0 = _init_params(X, K, R, reg, generator=g)
-    w, mu, cov, _ = em_fit(X, train, w0, mu0, cov0, reg, tol, max_iter)
+    w, mu, cov, _ = em_fit(X, train, w0, mu0, cov0, reg, tol, max_iter,
+                           chunk=chunk)
     v = val_ll(X, val, w, mu, cov)                        # (I*F,)
     v_if = v.view(n_inits, n_folds)
     metric = v_if.mean(1).max().item()                    # best init's mean fold LL
@@ -231,12 +232,12 @@
 
 def select_sequential(X, max_k=20, n_folds=2, n_inits=3,
                       threshold=1e-5, patience=1, reg=1e-6, tol=1e-5,
-                      max_iter=500, seed=0):
+                      max_iter=500, seed=0, chunk=8):
     """GMM-MI's protocol: increment k, threshold + patience early stop."""
     best_metric, pat, history = -torch.inf, 0, {}
     for K in range(1, max_k + 1):
         metric, params = _cv_fit_for_k(X, K, n_folds, n_inits, reg, tol,
-                                       max_iter, seed)
+                                       max_iter, seed, chunk=chunk)
         history[K] = (metric, params)
         if metric - best_metric > threshold:
             best_metric = metric
@@ -249,7 +250,7 @@
 
 
 def select_batched(X, max_k=20, n_folds=2, n_inits=3, threshold=1e-5,
-                   reg=1e-6, tol=1e-5, max_iter=500, seed=0):
+                   reg=1e-6, tol=1e-5, max_iter=500, seed=0, chunk=8):
     """Simplified protocol: all k at once via dead-component padding; argmax."""
     N, d = X.shape
     train, val = _cv_masks(N, n_folds, n_inits, X.dtype)
@@ -261,7 +262,8 @@
     g = torch.Generator(device=X.device).manual_seed(seed)
     w0, mu0, cov0 = _init_params(X, max_k, R, reg, generator=g,
                                  active_k=active_k)
-    w, mu, cov, _ = em_fit(X, train_r, w0, mu0, cov0, reg, tol, max_iter)
+    w, mu, cov, _ = em_fit(X, train_r, w0, mu0, cov0, reg, tol, max_iter,
+                           chunk=chunk)
     v = val_ll(X, val_r, w, mu, cov).view(max_k, n_inits, n_folds)
     metric_k = v.mean(-1).max(-1).values                  # (max_k,)
     # parsimony rule: smallest k within `threshold` of the best metric
@@ -309,7 +311,7 @@
 def estimate(X, quantity='MI', protocol='batched', n_bootstrap=50,
              mc_samples=100_000, max_k=20, n_folds=2, n_inits=3,
              reg=1e-6, tol=1e-5, max_iter=500, seed=0, return_k=False,
-             device='cpu', dtype=torch.float64):
+             device='cpu', dtype=torch.float64, em_chunk=8, mc_chunk=8):
     """Fit + bootstrap-estimate MI, conditional MI, or total correlation.
 
     Bootstrap is a single batched EM: multinomial-count sample weights over
@@ -322,7 +324,8 @@
         assert X.shape[1] >= 3
     sel = select_sequential if protocol == 'sequential' else select_batched
     kw = dict(max_k=max_k, n_folds=n_folds, n_inits=n_inits, reg=reg,
-              tol=tol, max_iter=max_iter, seed=seed, threshold=1e-5)
+              tol=tol, max_iter=max_iter, seed=seed, threshold=1e-5,
+              chunk=em_chunk)
     k_best, (w1, mu1, cov1), _ = sel(X, **kw)
 
     N = X.shape[0]
@@ -340,12 +343,13 @@
     w0 = w1.unsqueeze(0).expand(B, K).contiguous()
     mu0 = mu1.unsqueeze(0).expand(B, K, X.shape[1]).contiguous()
     cov0 = cov1.unsqueeze(0).expand(B, K, X.shape[1], X.shape[1]).contiguous()
-    w, mu, cov, _ = em_fit(X, s, w0, mu0, cov0, reg, tol, max_iter)
+    w, mu, cov, _ = em_fit(X, s, w0, mu0, cov0, reg, tol, max_iter,
+                           chunk=em_chunk)
 
     chol = _chol(cov)
     vals = []
-    for i in range(0, B, 8):
-        sl = slice(i, i + 8)
+    for i in range(0, B, mc_chunk):
+        sl = slice(i, i + mc_chunk)
         pts = gmm_sample(w[sl], mu[sl], chol[sl], mc_samples, generator=g)
         vals.append(_INTEGRANDS[quantity](pts, w[sl], mu[sl], cov[sl],
                                           chol[sl]).mean(1))
```

---

# Part 2 — Extensions (same day, extensions brief)

Features ported from the GMM-MI source (github.com/dpiras/GMM-MI, cloned and read at implementation time — semantics taken from `initializations.py`, `gmm.py`, `mi.py`, `cross_validation.py`, not from the brief text).

## 6. What was added

- **Init types** (`init_type` on both selectors and `estimate`, default `'random_responsibilities'` = old behaviour): `'minmax'` (means uniform over the data bounding box, isotropic covariances via GMM-MI's n-ball scale rule, per-replicate k-dependent scale), `'kmeans'` (batched Lloyd on a ≤4096-point subsample, hard-assignment statistics on the full data — covariances normalised by N, not by cluster count, matching GMM-MI's `KmeansInit`), `'randomized_kmeans'` (same plus a mean push `scale·N(0, cov_k)`, scale default 1). All run batched over replicates with per-replicate `active_k` masking for the batched-over-k protocol.
- **`_init_params` M-step chunking** (the peak-memory lever from Part 1): the initial M-step / assignment statistics now run in replicate chunks of the same `chunk` used by `em_fit`.
- **Marginal Gaussianisation**: `preprocess='gaussianise'` in `estimate()` — per-variable rank-probit `Φ⁻¹(rank/(N+1))` before fitting; MI/cMI/TC are invariant under per-variable diffeomorphisms so no correction term.
- **`fixed_components_number`** in `estimate()` (and `estimate_categorical`): skips selection, fits that k via the usual CV (best init/fold), mirroring GMM-MI's `fixed_components`.
- **`metric_method='valid'|'aic'|'bic'`** in `select_batched` (and passthrough from `estimate`): AIC/BIC per GMM-MI — each k's best CV parameters are refit on the full dataset and sklearn's criteria (`-2·ΣlogL + 2p` / `-2·ΣlogL + p·ln N`, `p = (k−1) + kd + kd(d+1)/2`) are computed on that refit; k = argmin. `'valid'` keeps the parsimony-threshold rule unchanged.
- **Marginal KL divergence**: quantities `'KL_forward'`/`'KL_reverse'` in `estimate()`, porting `estimate_KL_MC`: sample the joint, evaluate both 1-D marginals (analytic masking) on the x-column (forward) or y-column (reverse). Bootstrap errorbars via the existing batched path, unchanged.
- **Continuous–categorical MI**: `estimate_categorical(X, Y, ...)`, porting `fit_categorical`/`estimate_categorical`: per-category 1-D GMM fit with full component selection, bootstrap = per-category multinomial-weight batched EM warm-started from that category's CV parameters, MI = `(1/C)·Σ_c E_{x~p̂(x|c)}[log p̂(x|c) − log((1/C)Σ_c' p̂(x|c'))]` (uniform category weighting, exactly as in the reference code / Piras et al. 2022 App. B), assembled with logsumexp and chunked by `mc_chunk`.

Untouched, per the brief: parsimony rule, existing estimator integrands, existing test tolerances.

## 7. Test results (11 tests: 5 original + 6 new)

New tests (sigma-band style): KL forward+reverse vs analytic Gaussian KL; categorical MI vs 1-D quadrature target; gaussianisation on exp/cubic/tanh-warped correlated Gaussian vs pre-warp sample-covariance TC; AIC and BIC selecting k on 3-component mixture data (both select k=3; assertion allows 3–4); fixed-k reproducing the batched result; all three new init types recovering analytic Gaussian MI.

| Config | Result | Wall time |
|---|---|---|
| CPU fp64 (parity) | 11/11 pass | 23.5 s |
| CUDA fp32, `em_chunk=64` (full batch, production) | 11/11 pass | 4.0 s |

Original 5 tests reproduce their previous values bit-for-bit on CPU fp64 (the default init path draws the same random numbers; M-step chunking is exact).

**GB10 platform note:** two ops had to be routed around nvrtc — `Tensor.prod()` and `torch.special.ndtri`/`erfinv` are JIT (jiterator) kernels on CUDA in torch 2.11, and nvrtc in this build rejects GB10's GPU architecture. The volume product now uses `exp(Σlog)` and the probit runs on CPU (one-off data prep). Anything jiterator-backed will hit the same wall on this box.

## 8. Timing/memory deltas vs Part 1

N=20 000 benchmark: timings within noise of Part 1 (e.g. CUDA fp32 (8/8) select 1.89 s vs 1.88 s; CPU fp64 11.2 s vs 11.5 s); MI values identical. Peak GPU memory drops from the init chunking: fp64 (8/8) 0.61→0.43 GiB, fp32 (8/8) 0.31→0.22 GiB.

Scale test (N=200 000, d=8, TC), CUDA:

| Config (em/mc chunk) | total | peak GPU mem | Part-1 peak | Δ |
|---|---|---|---|---|
| fp32 (8/8) | 4.3 s | 7.25 GiB | 9.86 GiB | −26 % |
| **fp32 (16/50)** | **3.8 s** | **7.25 GiB** | — | new sweet spot |
| fp32 (48/50) | 3.3 s | 10.15 GiB | 10.15 GiB | 0 (init not chunked at full batch) |
| fp64 (8/8) | 59.1 s | 14.49 GiB | 19.71 GiB | −26 % |
| fp64 (48/50) | 34.5 s | 20.28 GiB | 20.28 GiB | 0 |

TC values bit-identical to Part 1 in every config. `chunk` is now a clean speed/memory dial; the remaining ~7.25 GiB floor at small chunks is `val_ll` scoring all 48 selection replicates unchunked — the next lever if memory ever binds, not needed at 128 GB unified.

## 9. Files

[changes.diff](changes.diff) is the cumulative diff (Parts 1+2) of `torch_gmm_mi.py` and `test_torch_gmm_mi.py` against the pristine originals; `torch_gmm_mi_orig.py` is the pristine module. `conftest.py` additionally gained `GMM_MI_EM_CHUNK` routing and now wraps `estimate_categorical` as well.

---

# Part 3 — Validation of the `category_weights` update

Verification-only pass on the supervisor-updated `torch_gmm_mi.py` / `test_torch_gmm_mi.py` (empirical-π categorical MI, 12-test suite).

| Config | Result | Wall time |
|---|---|---|
| CPU fp64 (parity) | 12/12 pass | 24.0 s |
| CUDA fp32, `em_chunk=64` (full batch, production) | 12/12 pass | 3.9 s |

Categorical MI on CUDA fp32 (`max_k=3`, empirical π):

| Case | MI ± σ | Quadrature target | Deviation |
|---|---|---|---|
| Balanced (P₁≈0.5, N=10 000) | 0.3463 ± 0.0050 | 0.3368 | +1.91σ |
| Imbalanced (P₁=0.25, N=20 000) | 0.2630 ± 0.0044 | 0.2644 | −0.32σ |

The balanced-case bias matches CPU fp64 (+1.72σ), i.e. finite-N GMM-fit bias, not an fp32 effect. No new device-placement or nvrtc-JIT issues.

**Code review of the update:** π wiring verified correct (weighted logsumexp inner, π_c outer, uniform case algebraically subsumes the old `/C` form); `category_weights='uniform'` reproduces the Part-2 balanced value to full precision (0.3463397165). Notes: (1) `changes.diff` had not been regenerated after the file swap — fixed, the shipped diff now includes the categorical change and new test; (2) the stratified bootstrap holds π̂ fixed, so σ excludes π-sampling uncertainty (documented; an underestimate only for small N / extreme imbalance); (3) `test_categorical_MI_analytic` compares an empirical-π estimate to an exact-π=0.5 target — benign (MI is stationary in π at 0.5; difference 4×10⁻⁵ here) but strictly inconsistent; the imbalanced test does it consistently.

---

# Part 4 — Sample-dimension chunking (`n_chunk`): lifting the large-N ceiling

## 10. What was added

- `_em_fit_core` now takes `n_chunk`: with it set, each EM iteration streams over sample chunks, accumulating N_k, Σ resp·(x−μ_old), Σ resp·(x−μ_old)(x−μ_old)ᵀ and the weighted LL sum, then forms the M-step from the accumulated statistics (`_em_step_nchunked`). **Note on the accumulators:** the second moment is shifted by the incoming means rather than raw Σ resp·xxᵀ — algebraically identical, but immune to the fp32 cancellation of origin moments when |μ|² ≫ var (production is fp32). `val_ll` chunks the same way. `n_chunk=None` (default) keeps the original single-pass code path verbatim.
- `n_chunk` is threaded through `em_fit`, `_cv_fit_for_k`, both selectors (including the AIC/BIC full-data refit), `estimate`, and `estimate_categorical`. (This report originally said conftest gained a `GMM_MI_N_CHUNK` routing variable; that variable was never shipped — conftest routes device/dtype/reg/`GMM_MI_EM_CHUNK` only, and `n_chunk` is exercised by the parity tests directly.)
- **One flagged deviation from "no other changes":** `torch.multinomial` hard-caps categories at 2²⁴ = 16.7M, so the bootstrap-weight construction raised for N > 1.6×10⁷ — an API ceiling well below the new memory ceiling. `estimate` now falls back to `randint`+scatter for N > 2²⁴ (identical in distribution for uniform probabilities: multinomial counts = histogram of N iid uniform indices). The fallback triggers **only where the old code raised**, so no previously computable result changes. The same cap exists untouched in `estimate_categorical` (per-category N_c > 16.7M would hit it).

## 11. Test results

| Config | n_chunk=None | n_chunk=2048 |
|---|---|---|
| CPU fp64 | 12/12 pass | 12/12 pass |
| CUDA fp32, `em_chunk=64` | 12/12 pass | 12/12 pass |

**On bit-identity at fp64:** strict bitwise equality between chunked and unchunked paths is not attainable in principle — chunked accumulation reorders the floating-point reductions (and the streaming covariance form differs algebraically-equivalently from the centered single-pass form). Measured agreement at fp64 (`n_chunk=2048` vs `None`): MI mean bitwise identical, TC mean 1 ulp (1.2×10⁻¹⁶ relative), bootstrap σ within 6×10⁻¹⁵ relative. I.e. machine-epsilon-level, ~11 orders of magnitude below the bootstrap uncertainties.

## 12. Scale demonstration: TC, 8-dim correlated Gaussian, fp32, n_chunk tuned

Data generated on-GPU (`bench_nchunk.py`); `init_type='minmax'` (see below); `max_k=8`, `n_bootstrap=50`, `mc_chunk=50`. Tuning at N=10⁷ gave (`n_chunk=2.5×10⁵`, `em_chunk=48`) — same speed as larger n_chunk but 3× lower peak (E-step einsum temporaries scale with `em_chunk × n_chunk`).

| N | total wall | peak GPU mem | k | TC | sample-cov target | dev |
|---|---|---|---|---|---|---|
| 10⁷ | 123 s | 17.3 GiB | 1 | 4.56699 ± 0.00589 | 4.56776 | −0.13σ |
| 4×10⁷ | 498 s | 31.0 GiB | 1 | 4.56813 ± 0.00544 | 4.56835 | −0.04σ |
| 6×10⁷ | 756 s | 46.5 GiB | 1 | 4.56789 ± 0.00563 | 4.56835* | −0.08σ |
| **10⁸** | **1379 s** | **77.5 GiB** | 1 | **4.56871 ± 0.00491** | 4.56856 | +0.03σ |

(*same generator seed; targets recomputed per-N from the sample covariance.)

**N reached: 10⁸ samples** (d=8, full pipeline: selection over k=1..8 × 6 CV replicates, 50-replicate bootstrap EM, MC integration) in 23 minutes at 77.5 GiB — a **500× larger N** than the Part-1 ceiling demonstration, with wall time and peak memory now scaling linearly in N. Without `n_chunk` this N is impossible: the unchunked E-step alone would need ~10 TiB for the (48, 10⁸, 8, 8) responsibility-weighted diff tensor.

Remaining N-scalers (the next ceilings, all outside this task's scope): the CV-mask replicates ((48,N) float each for train/val), the bootstrap weight construction (~1 kB/sample transient), and with the default init the `_init_params` responsibility draw ((R,N,K) — the reason `minmax` was used here; default init binds at roughly N ≈ 2–3×10⁷).

## 13. Recommended large-N settings on GB10

```python
estimate(X, ..., device='cuda', dtype=torch.float32,
         em_chunk=48, mc_chunk=50,
         n_chunk=250_000,          # only helps for N >~ 10^6; None below that
         init_type='minmax')       # avoids the (R,N,K) init responsibility draw
```

---

# Part 5 — Final pass: polished module validated

Verification of the supervisor's documentation pass + parameter renames (`reg`→`reg_covar`, `tol`→`threshold_fit`, `threshold`→`threshold_components`, `max_k`→`max_components`, `chunk`→`em_chunk`).

## 14. Validation

| Config | Result | Wall time |
|---|---|---|
| CPU fp64 | 12/12 pass | 23.9 s |
| CUDA fp32, `em_chunk=64` (full batch) | 12/12 pass | 4.5 s |

- **Renames are behaviour-neutral:** the benchmark on CPU fp64 reproduces the Part-2 reference exactly (k=3, MI = 0.55930 ± 0.00682, matching to all printed digits), and the module greps clean for every old parameter name — no missed call site.
- **`bench.py` / `bench_nchunk.py` call sites updated** to the new kwargs (flagged: the §4 recommendation snippet's `max_k` comment was also renamed).
- **Scale reproduction under renamed kwargs:** `bench_nchunk.py` at N=10⁷ (`n_chunk=2.5×10⁵`, `em_chunk=48`, minmax): 122.2 s, 17.25 GiB, TC = 4.56699 ± 0.00589 — identical value and timing (Part 4: 122.7 s) to the pre-rename run.

## 15. Docstring review findings (no code errors found)

The numpydoc pass was checked against the code; the algorithms are untouched and the renames are consistent. Four documentation inaccuracies to correct in a future pass:

1. **`select_batched`, Returns/history:** claims "params entries are None for k != k_best" — in fact `history` stores `None` params for *every* k, including `k_best` (the selected parameters are returned separately).
2. **`estimate`, `n_chunk`:** "at identical results" overstates — chunked accumulation reorders floating-point reductions; measured agreement is ulp-level at fp64 (Part 4 §11), not bitwise. Likewise "makes peak memory independent of N" holds for the EM/val-LL passes only; the CV-mask replicates and bootstrap-weight construction still scale with N (Part 4 §12).
3. **`estimate`, `preprocess`:** "Not valid for the KL quantities" is documented but not enforced — `preprocess='gaussianise'` with `quantity='KL_forward'` runs silently and returns the KL of the gaussianised marginals (which is ~0 by construction). Worth a `ValueError`.
4. **`estimate`, `protocol`:** the text refers to `threshold_components` as if it were tunable from `estimate`, but `estimate` hardcodes `threshold_components=1e-5` and does not expose it.

---

# Close-out — corrected module verified (final)

The §15-corrected module was reviewed and validated; **the project is closed from the GB10 side.**

**Review.** All six changes are present and correct: the four §15 docstring fixes (history-params wording, n_chunk precision/memory claims now stated exactly as measured in Part 4, preprocess restriction wording, threshold_components documentation) plus the two behavioural ones — `preprocess='gaussianise'` with a KL quantity now raises `ValueError` (verified to fire), and `estimate()` exposes `threshold_components=1e-5`, correctly threaded to both selectors (verified functional: `threshold_components=100` forces k=1 on 3-component data). Nothing else changed: the module greps clean, and the benchmark reproduces the Part-2 reference exactly (k=3, MI = 0.55930 ± 0.00682), confirming the default behaviour is untouched. One cosmetic nit, no action needed: the `threshold_components=threshold_components,` line in `estimate`'s kw dict is under-indented relative to its neighbours.

**Handoff note.** The shipped `changes.diff` was stale for the second time (it predated the corrected module — still carried the pre-fix `select_batched` docstring). Regenerated from the shipped files (1181 lines); suggest making diff regeneration part of the channel's file-replacement step.

**Final validation.**

| Config | Result | Wall time |
|---|---|---|
| CPU fp64 | 12/12 pass | 23.7 s |
| CUDA fp32, `em_chunk=64` (full batch) | 12/12 pass | 3.9 s |

---

# Part 6 — Extended test suite (tests only + one module seam)

Goal: broaden coverage of the shipped estimator. The 12 existing tests and their
tolerances are untouched; **12 new tests** were added (11 in the CI set + 1
offline-gated), plus a single, behaviour-neutral module seam. New tests follow
the suite's existing style (analytic / ground-truth targets, 3σ-band asserts)
and pass on both target configs.

## 16. What was added

**Module seam (the only permitted module change).** The bare `1 << 24`
(torch.multinomial's category cap) at the `estimate` bootstrap branch is hoisted
to a module constant `_MULTINOMIAL_MAX_CATEGORIES = 1 << 24`, and the branch now
reads `if N <= _MULTINOMIAL_MAX_CATEGORIES:`. Branch logic and defaults are
unchanged; the constant only lets a test monkeypatch the cap down to exercise the
randint fallback at small N. Verified behaviour-neutral (both configs unchanged).

**New tests.**

| Test | Target / provenance |
|---|---|
| `test_MI_sequential_analytic` | Gaussian MI via `protocol='sequential'` vs analytic −½log(1−ρ²) |
| `test_sequential_early_stop_k_regression` | pins the sequential early-stop `k=3` on `_three_component_data` (observed behaviour, identical on both configs; not asserted "correct") |
| `test_cMI_dependent_conditioner` | trivariate Gaussian ρ_XY=.5, ρ_XZ=.6, ρ_YZ=.4; pinned I(X;Y\|Z)=0.0671910473, I(X;Y)=0.1438410362 |
| `test_TC_multimodal_vs_MC_target` | 3-var asymmetric 3-component mixture; TC vs MC ground truth **1.092080** (SE 1.4e-4) |
| `test_KL_nongaussian_marginals_vs_quadrature` | KL between the two 3-mode marginals of `_three_component_data`; quadrature targets KL_fwd=0.1725434, KL_rev=0.1309184 |
| `test_nchunk_parity_fp64` | small `n_chunk` vs `None`, fp64, tight tol (\|Δ\|≤1e-9); device follows harness |
| `test_category_weights_uniform_regression` | `category_weights='uniform'` vs Part-2 balanced reference 0.3463397165 |
| `test_categorical_MI_three_categories` | 3-category continuous-categorical MI vs 1D quadrature (empirical weights) |
| `test_mc_chunk_not_dividing_n_bootstrap` | `mc_chunk=7`, `n_bootstrap=50` (final chunk size 1) vs analytic MI |
| `test_invalid_argument_paths` | ValueError: gaussianise+KL; invalid `preprocess`; invalid `category_weights` |
| `test_bootstrap_randint_fallback` | monkeypatch the cap down → randint path; consistent with the multinomial path |
| `test_bootstrap_sigma_calibration_offline` | **offline only** (gated on `GMM_MI_OFFLINE`); errorbar coverage, GMM-MI-paper style |

The mixture-TC MC target and the KL quadrature targets were each computed by a
dedicated subagent (returning only the reference value + generating script);
derivations were kept out of the main line of work.

## 17. Multimodal tests: why the default init, not k-means

First cut used `init_type='kmeans'` for the TC and KL tests, because
well-separated isotropic blobs along the diagonal collapse to `k=1` under the
default responsibility init (recovering only the Gaussian-correlation value,
e.g. TC≈1.66 vs true 1.789) and k-means finds the modes. **This was wrong:**
k-means init is not the GMM-MI default and is brittle; a test that only passes
under it is not exercising the recommended path and can flake across seed/device.

Root cause is deeper than the init alone: with the default `'valid'` metric even
moderately-separated mixtures select `k=1` (the validation-LL parsimony rule is
reluctant to add components), and the bootstrap σ (~0.007) is so small that any
`k=1` bias blows past 3σ. And `random_responsibilities` genuinely cannot fit
*symmetric* well-separated blobs (every component initialises at the global mean
→ saddle), regardless of `n_inits` or metric.

The fix uses the estimator's own sanctioned levers: **(a)** an *asymmetric,
non-collinear* mixture with distinct anisotropic covariances (the layout of the
existing `test_aic_bic_select_sensible_k` data, which the default init *does*
fit), and **(b)** `metric_method='bic'` to select the true `k`. Both TC and KL
then recover their targets to <1σ on both configs under the **default init** —
no k-means, no separation hacks. The KL test reuses the suite's own
`_three_component_data`, whose 1D marginals are genuinely trimodal.

## 18. Results (both target configs)

| Config | CI set | Wall time |
|---|---|---|
| CPU fp64, default `em_chunk` (parity baseline) | 23 passed, 1 skipped | 50.6 s |
| CUDA fp32, full-batch `em_chunk=1000` | 23 passed, 1 skipped | 7.2 s |

Every new test lands well inside its 3σ band on both configs (TC −0.2σ, KL
−0.3σ/−0.1σ, cMI/MI <1σ, categorical <1σ). Independently re-verified by a
read-only subagent (diff consistency + module-seam-only + both suites).

**Offline calibration** (`GMM_MI_OFFLINE=1`, MI errorbar coverage over 40
independent datasets, ρ=0.6):

| Config | ±1σ coverage (nom. 0.68) | ±2σ coverage (nom. 0.95) |
|---|---|---|
| CPU fp64 | 0.675 | 0.925 |
| CUDA fp32 | 0.575 | 0.925 |

Well-calibrated; the in-suite test asserts only loose bounds (few-dozen repeats
are noisy) and is kept out of the CI set.

## 19. `changes.diff` baseline correction

The prior `changes.diff` diffed against `torch_gmm_mi_orig.py` (355-line
*pre-extension* pristine) and `test_orig.py` (77-line *5-test* original) — a very
old baseline that buried this task's change under the entire Parts 1–5 history.
`changes.diff` is now **task-scoped**: it diffs the current-shipped files
(`torch_gmm_mi_shipped.py` = module minus the seam; `test_torch_gmm_mi_shipped.py`
= the 12-test file) against the shipped ones, so it shows exactly the seam + 12
new tests (264 lines). Both baseline snapshots are kept in-repo for
reproducibility.

## 20. Files

`test_torch_gmm_mi.py` (extended, 24 tests), `torch_gmm_mi.py` (seam only),
`torch_gmm_mi_shipped.py` / `test_torch_gmm_mi_shipped.py` (task-start baselines),
`changes.diff` (task-scoped, regenerated from the shipped files).
