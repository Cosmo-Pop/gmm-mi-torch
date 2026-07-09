"""Scale test: N=200000, d=8 correlated Gaussian, TC, chunk tuning + peak mem."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/hvp20/GMM-MI torch port")
from bench import run_once, make_data


def make_data_8d(N=200_000, d=8, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) * 0.3 + np.eye(d)
    cov = A @ A.T
    X = rng.multivariate_normal(np.zeros(d), cov, N)
    S = np.cov(X.T)
    Ds = np.diag(1 / np.sqrt(np.diag(S)))
    tc_sample = -0.5 * np.log(np.linalg.det(Ds @ S @ Ds))
    return X, tc_sample


if __name__ == "__main__":
    dtype = {"float64": torch.float64, "float32": torch.float32}[sys.argv[1]]
    em_chunk = int(sys.argv[2])
    mc_chunk = int(sys.argv[3])
    device = sys.argv[4] if len(sys.argv) > 4 else "cuda"

    X, tc_sample = make_data_8d()
    if device == "cuda":
        run_once(make_data()[:2000], "cuda", dtype, max_k=2, n_bootstrap=4,
                 mc_samples=1000)  # warmup
    t0 = time.perf_counter()
    r = run_once(X, device, dtype, max_k=8, n_bootstrap=50,
                 em_chunk=em_chunk, mc_chunk=mc_chunk, quantity="TC")
    t_tot = time.perf_counter() - t0
    print(f"{device} {sys.argv[1]:8s} em_chunk={em_chunk:2d} mc_chunk={mc_chunk:2d} "
          f"select={r['t_select']:.2f}s boot={r['t_boot']:.2f}s total={t_tot:.2f}s "
          f"peak={r['peak_gib']:.2f}GiB k={r['k']} "
          f"TC={r['mean']:.5f}+-{r['std']:.5f} (sample-cov TC={tc_sample:.5f})",
          flush=True)
