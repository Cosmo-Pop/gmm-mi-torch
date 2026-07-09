"""Large-N scale demo for n_chunk: TC on an 8-dim correlated Gaussian, fp32.

Data are generated directly on the GPU (Z @ L^T); the sample-covariance TC
target is accumulated in fp64. Usage:
    python bench_nchunk.py <N> <n_chunk> [em_chunk] [init_type]
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/hvp20/GMM-MI torch port")
from torch_gmm_mi import estimate


def make_data_gpu(N, d=8, seed=7, dtype=torch.float32):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) * 0.3 + np.eye(d)
    cov = A @ A.T
    L = torch.as_tensor(np.linalg.cholesky(cov), dtype=dtype, device='cuda')
    g = torch.Generator(device='cuda').manual_seed(seed)
    X = torch.empty(N, d, dtype=dtype, device='cuda')
    for i in range(0, N, 10_000_000):
        nc = min(10_000_000, N - i)
        X[i:i+nc] = torch.randn(nc, d, dtype=dtype, device='cuda',
                                generator=g) @ L.T
    # sample-covariance TC target, fp64 accumulation
    S = torch.zeros(d, d, dtype=torch.float64, device='cuda')
    m = torch.zeros(d, dtype=torch.float64, device='cuda')
    for i in range(0, N, 10_000_000):
        Xc = X[i:i+10_000_000].double()
        S += Xc.T @ Xc
        m += Xc.sum(0)
    m /= N
    S = (S / N - torch.outer(m, m)).cpu()   # logdet JIT-fails on GB10 CUDA
    Dinv = torch.diag(1.0 / S.diag().sqrt())
    tc = -0.5 * torch.logdet(Dinv @ S @ Dinv)
    return X, tc.item()


if __name__ == "__main__":
    N = int(float(sys.argv[1]))
    n_chunk = int(float(sys.argv[2]))
    em_chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    init_type = sys.argv[4] if len(sys.argv) > 4 else 'minmax'

    X, tc_target = make_data_gpu(N)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tc_mean, tc_std, k = estimate(X, 'TC', protocol='batched', max_components=8,
                                  n_bootstrap=50, mc_samples=100_000,
                                  device='cuda', dtype=torch.float32,
                                  em_chunk=em_chunk, mc_chunk=50,
                                  n_chunk=n_chunk, init_type=init_type,
                                  return_k=True)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"N={N:.0e} n_chunk={n_chunk:.0e} em_chunk={em_chunk} "
          f"init={init_type}: total={t:.1f}s peak={peak:.2f}GiB k={k} "
          f"TC={tc_mean:.5f}+-{tc_std:.5f} (sample-cov TC={tc_target:.5f})",
          flush=True)
