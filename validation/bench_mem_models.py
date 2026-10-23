"""Measured peak CUDA memory of every auto-sized stage against its byte model
(ratio = measured peak / model; the models must stay >= 1 on every row).
Source of the per-replicate models in gmm_mi_torch/estimator.py (hardening
pass, 2026-10-21). Run from the repo root on a CUDA machine:
    PYTHONPATH=. python validation/bench_mem_models.py"""
import torch, warnings
import gmm_mi_torch.estimator as E
warnings.simplefilter('ignore'); dev = 'cuda'
def peak(fn):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base
def params(X, K, R, dt, collapse=0):
    g = torch.Generator(device=dev).manual_seed(0)
    w, mu, cov = E._init_params(X, K, R, 1e-6, generator=g, init_type='minmax', em_chunk=R)
    if collapse:
        mu[:collapse, 0] = 4.0; cov[:collapse, 0] = 1e-6 * torch.eye(X.shape[-1], dtype=dt, device=dev)
    return w, mu, cov
dt = torch.float32
print("EM GEMM, n_chunk path at allocator-relevant size: N=400000, R=32, n_chunk=65536; ratio = peak / (R * model)")
for d in (2, 8):
    for K in (1, 3, 20):
        X = torch.randn(400000, d, device=dev, dtype=dt); s = torch.ones(32, X.shape[0], device=dev, dtype=dt)
        w, mu, cov = params(X, K, 32, dt)
        E._em_fit_core(X, s, w, mu, cov, max_iter=1, n_chunk=65536)
        p = peak(lambda: E._em_fit_core(X, s, w, mu, cov, max_iter=3, n_chunk=65536))
        m = 32 * E._em_bytes_per_replicate(X.shape[0], K, d, dt, 65536) + E._em_fixed_bytes(X.shape[0], d, dt, 65536)
        print(f"  d={d} K={K:2d}: peak {p/2**20:8.1f} MiB, model {m/2**20:8.1f} MiB, ratio {p/m:5.2f}")
print("EM GEMM single pass N=20000 R=64, ratio to model (+F fixed)")
for d in (1, 2, 4, 8):
    for K in (1, 2, 3, 5, 20):
        X = torch.randn(20000, d, device=dev, dtype=dt); s = torch.ones(64, 20000, device=dev, dtype=dt)
        w, mu, cov = params(X, K, 64, dt)
        E._em_fit_core(X, s, w, mu, cov, max_iter=1)
        p = peak(lambda: E._em_fit_core(X, s, w, mu, cov, max_iter=3))
        m = 64 * E._em_bytes_per_replicate(20000, K, d, dt) + E._em_fixed_bytes(20000, d, dt)
        print(f"  d={d} K={K:2d}: ratio {p/m:5.2f}")
print("rerouted (all / half), N=20000 R=64, ratio to GEMM model")
for d in (2, 4, 8):
    for K in (1, 2, 3):
        for frac, nc in ((64, None), (32, None), (64, 4096), (32, 4096)):
            X = torch.randn(20000, d, device=dev, dtype=dt); s = torch.ones(64, 20000, device=dev, dtype=dt)
            w, mu, cov = params(X, K, 64, dt, collapse=frac)
            E._em_fit_core(X, s, w, mu, cov, max_iter=1, n_chunk=nc)
            p = peak(lambda: E._em_fit_core(X, s, w, mu, cov, max_iter=3, n_chunk=nc))
            m = 64 * E._em_bytes_per_replicate(20000, K, d, dt, nc) + E._em_fixed_bytes(20000, d, dt, nc)
            print(f"  d={d} K={K} rerouted={frac:2d}/64 n_chunk={nc}: ratio {p/m:5.2f}")
print("init stage ratio to model (R=64, N=20000)")
for init in ('random_responsibilities', 'kmeans'):
    for d in (1, 2, 8):
        for K in (1, 3, 20):
            X = torch.randn(20000, d, device=dev, dtype=dt)
            g = torch.Generator(device=dev).manual_seed(0)
            E._init_params(X, K, 4, 1e-6, generator=g, init_type=init, em_chunk=4)
            p = peak(lambda: E._init_params(X, K, 64, 1e-6, generator=g, init_type=init, em_chunk=64))
            print(f"  {init:24s} d={d} K={K:2d}: ratio {p/(64*E._init_bytes_per_replicate(20000, K, d, dt, init)):5.2f}")
print("MC stage ratio to model (b=8 and b=1, S=1e5)")
for d, q in ((2, 'MI'), (3, 'cMI'), (4, 'TC'), (8, 'TC')):
    for K in (1, 2, 5, 20):
        for b in (8, 1):
            X = torch.randn(2000, d, device=dev, dtype=dt)
            w, mu, cov = params(X, K, b, dt); chol, cov = E._chol(cov)
            def mc():
                pts = torch.cat([E.gmm_sample(w[r:r+1], mu[r:r+1], chol[r:r+1], 100_000, generator=torch.Generator(device=dev).manual_seed(r)) for r in range(b)])
                return E._INTEGRANDS[q](pts, w, mu, cov, chol).mean(1)
            mc(); p = peak(mc)
            print(f"  d={d} {q:3s} K={K:2d} b={b}: ratio {p/(b*E._mc_bytes_per_replicate(100_000, K, d, dt)):5.2f}")

print("streamed path with expand-view CV masks and converging replicates (2026-10-23 review F6), N=4e5, R=32, d=2")
N, d, R = 400000, 2, 32
X = torch.randn(N, d, device=dev, dtype=dt)
train, val = E._cv_masks(N, 2, 16, dt, device=dev)
for nc in (4096, 65536):
    for K in (1, 3, 20):
        w, mu, cov = params(X, K, R, dt)
        m = R * E._em_bytes_per_replicate(N, K, d, dt, nc) + E._em_fixed_bytes(N, d, dt, nc)
        E.em_fit(X, train, w, mu, cov, em_chunk=R, max_iter=2, n_chunk=nc)
        pe = peak(lambda: E.em_fit(X, train, w, mu, cov, em_chunk=R, max_iter=6, n_chunk=nc))
        pv = peak(lambda: E.val_ll(X, val, w, mu, cov, em_chunk=R, n_chunk=nc))
        print(f"  n_chunk={nc:5d} K={K:2d}: em_fit ratio {pe/m:5.2f}  val_ll ratio {pv/m:5.2f}  (model {m/2**20:.0f} MiB; at K=20, n_chunk=4096 the fixed cuBLAS workspace shows against a 23 MiB model)")
print("single pass with expand-view CV masks and converging replicates, N=20000, R=64, d=2")
N, R = 20000, 64
X = torch.randn(N, 2, device=dev, dtype=dt)
train, val = E._cv_masks(N, 2, 32, dt, device=dev)
for K in (1, 3, 20):
    w, mu, cov = params(X, K, R, dt)
    m = R * E._em_bytes_per_replicate(N, K, 2, dt) + E._em_fixed_bytes(N, 2, dt)
    E.em_fit(X, train, w, mu, cov, em_chunk=R, max_iter=2)
    pe = peak(lambda: E.em_fit(X, train, w, mu, cov, em_chunk=R, max_iter=30))
    pv = peak(lambda: E.val_ll(X, val, w, mu, cov, em_chunk=R))
    print(f"  K={K:2d}: em_fit ratio {pe/m:5.2f}  val_ll ratio {pv/m:5.2f}")
