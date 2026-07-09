"""Benchmark torch_gmm_mi: selection vs bootstrap wall-clock, CPU vs GPU, fp64 vs fp32.

Workload: 3-component bivariate mixture, N=20000, max_k=8, n_bootstrap=50
(mirrors estimate()'s internals so the two phases can be timed separately).
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/hvp20/GMM-MI torch port")
from torch_gmm_mi import (select_batched, em_fit, _chol, gmm_sample,
                          _INTEGRANDS)


def make_data(N=20000, seed=0):
    """3-component bivariate GMM sample."""
    rng = np.random.default_rng(seed)
    weights = np.array([0.3, 0.4, 0.3])
    means = np.array([[-2.0, -2.0], [0.0, 1.5], [2.5, -1.0]])
    covs = np.array([[[1.0, 0.6], [0.6, 1.0]],
                     [[0.5, -0.3], [-0.3, 0.8]],
                     [[0.9, 0.4], [0.4, 0.6]]])
    counts = rng.multinomial(N, weights)
    parts = [rng.multivariate_normal(means[c], covs[c], counts[c])
             for c in range(3)]
    X = np.concatenate(parts)
    rng.shuffle(X)
    return X


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_once(Xnp, device, dtype, max_k=8, n_bootstrap=50, mc_samples=100_000,
             em_chunk=8, mc_chunk=8, reg=1e-6, tol=1e-5, max_iter=500, seed=0,
             quantity="MI"):
    X = torch.as_tensor(Xnp, dtype=dtype, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    t0 = time.perf_counter()
    k_best, (w1, mu1, cov1), _ = select_batched(
        X, max_components=max_k, reg_covar=reg, threshold_fit=tol,
        max_iter=max_iter, seed=seed, em_chunk=em_chunk)
    sync(device)
    t_select = time.perf_counter() - t0

    # bootstrap phase (mirrors estimate())
    N, d = X.shape
    B = n_bootstrap
    g = torch.Generator(device=X.device).manual_seed(seed + 10_000)
    t0 = time.perf_counter()
    s = torch.multinomial(
        torch.full((B, N), 1.0 / N, dtype=X.dtype, device=X.device), N,
        replacement=True, generator=g)
    s = torch.zeros(B, N, dtype=X.dtype, device=X.device).scatter_add_(
        1, s, torch.ones(B, N, dtype=X.dtype, device=X.device))
    K = w1.shape[0]
    w0 = w1.unsqueeze(0).expand(B, K).contiguous()
    mu0 = mu1.unsqueeze(0).expand(B, K, d).contiguous()
    cov0 = cov1.unsqueeze(0).expand(B, K, d, d).contiguous()
    w, mu, cov, _ = em_fit(X, s, w0, mu0, cov0, reg, tol, max_iter,
                           em_chunk=em_chunk)
    chol = _chol(cov)
    vals = []
    for i in range(0, B, mc_chunk):
        sl = slice(i, i + mc_chunk)
        pts = gmm_sample(w[sl], mu[sl], chol[sl], mc_samples, generator=g)
        vals.append(_INTEGRANDS[quantity](pts, w[sl], mu[sl], cov[sl],
                                          chol[sl]).mean(1))
    vals = torch.cat(vals)
    sync(device)
    t_boot = time.perf_counter() - t0

    mean = vals.mean().item()
    std = vals.std(unbiased=True).item()
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device == "cuda" else float("nan"))
    return dict(k=k_best, mean=mean, std=std, t_select=t_select,
                t_boot=t_boot, peak_gib=peak)


if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    Xnp = make_data()

    # warmup: tiny run to load CUDA kernels / JIT caches
    if device_arg in ("cuda", "both"):
        run_once(Xnp[:2000], "cuda", torch.float32, max_k=2, n_bootstrap=4,
                 mc_samples=1000)

    devices = ["cpu", "cuda"] if device_arg == "both" else [device_arg]
    for device in devices:
        for dtype in (torch.float64, torch.float32):
            r = run_once(Xnp, device, dtype)
            print(f"{device:4s} {str(dtype):15s} k={r['k']} "
                  f"MI={r['mean']:.5f}+-{r['std']:.5f} "
                  f"select={r['t_select']:.2f}s boot={r['t_boot']:.2f}s "
                  f"peak={r['peak_gib']:.2f}GiB", flush=True)
