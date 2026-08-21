"""Hardware sweep for gmm-mi-torch: em_chunk x dtype on a synthetic hard pair,
plus the per-replicate EM cost vs batch size (saturation curve).

Self-contained (synthetic data, no downloads).  Writes <out>.json and <out>.md.

    python validation/bench_sweep.py --out sweep_gh200
    python validation/bench_sweep.py --quick          # smoke test, ~1 min on GB10

Workload (the beta tester's Figure 7 settings on a deliberately hard pair):
N=20000, bivariate Gaussian rho=0.3 with one heavy-tailed marginal,
protocol='batched', max_components=20, n_inits=5, n_folds=3, n_bootstrap=100,
mc_samples=1e5.  The batched protocol selects k=19 on this pair.
"""
import argparse
import json
import platform
import sys
import time

import numpy as np
import torch

import gmm_mi_torch
import gmm_mi_torch.estimator as E
from gmm_mi_torch import estimate


def hard_pair(N=20000, rho=0.3, seed=0):
    X = np.random.default_rng(seed).multivariate_normal([0, 0], [[1, rho], [rho, 1]], N)
    X[:, 1] = np.sign(X[:, 1]) * np.abs(X[:, 1]) ** 1.5
    return X


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_info(dev):
    info = dict(torch=torch.__version__, python=platform.python_version(),
                gmm_mi_torch=getattr(gmm_mi_torch, '__version__', 'n/a'),
                device=dev, matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    if dev == 'cuda':
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.update(gpu=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}",
                    total_gb=round(total / 1e9, 1), free_gb=round(free / 1e9, 1))
    else:
        info.update(cpu_threads=torch.get_num_threads(), machine=platform.machine())
    return info


def run_estimate(X, dev, dtype, em_chunk, mc_chunk, quick):
    kw = dict(device=dev, dtype=dtype, n_bootstrap=100, max_components=20,
              n_inits=5, n_folds=3, mc_samples=100_000, return_k=True,
              em_chunk=em_chunk, mc_chunk=mc_chunk)
    if quick:
        kw.update(max_components=6, n_bootstrap=20)
    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(); t = time.perf_counter()
    m, s, k = estimate(X, 'MI', **kw)
    sync(); dt = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else None
    return dict(MI=m, sigma=s, k=k, wall_s=dt, peak_gb=peak)


def saturation(Xt, dtype, batches, K=20):
    """ms per replicate-iteration of one EM step vs batch size."""
    N, d = Xt.shape
    out = {}
    for Ra in batches:
        g = torch.Generator(device=Xt.device).manual_seed(0)
        w, mu, cov = E._init_params(Xt, K, Ra, 1e-6, generator=g, em_chunk=Ra)
        s = torch.ones(Ra, N, device=Xt.device, dtype=dtype)

        def step():
            chol = E._chol(cov)
            logw = torch.log(w.clamp_min(1e-300))
            Xr = Xt.expand(Ra, N, d)
            lj = logw.unsqueeze(1) + E.log_gauss(Xr, mu, chol)
            ll_n = torch.logsumexp(lj, -1)
            return E._m_step(Xr, torch.exp(lj - ll_n.unsqueeze(-1)), s, 1e-6)
        step(); sync(); t = time.perf_counter()
        for _ in range(5):
            step()
        sync()
        out[Ra] = (time.perf_counter() - t) / 5 / Ra * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sweep')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtypes', default='float32,float64')
    ap.add_argument('--em-chunks', default='8,32,128,300')
    ap.add_argument('--quick', action='store_true', help='small smoke-test configuration')
    a = ap.parse_args()
    dev = a.device
    dtypes = [getattr(torch, t) for t in a.dtypes.split(',')]
    chunks = [int(c) for c in a.em_chunks.split(',')]

    res = dict(info=device_info(dev), args=vars(a), estimate=[], saturation={})
    print(json.dumps(res['info'], indent=1), flush=True)
    X = hard_pair()
    # warm-up (kernel compilation, allocator)
    estimate(X[:2000], 'MI', device=dev, dtype=dtypes[0], max_components=3, n_bootstrap=4)

    for dt in dtypes:
        for c in chunks:
            r = run_estimate(X, dev, dt, c, 100, a.quick)
            r.update(dtype=str(dt).split('.')[-1], em_chunk=c)
            res['estimate'].append(r)
            print(f"{r['dtype']:8s} em_chunk={c:4d}: MI={r['MI']:.4f} sd={r['sigma']:.4f} "
                  f"k={r['k']:2d}  {r['wall_s']:7.1f}s  peak={r['peak_gb']}", flush=True)
        r = run_estimate(X, dev, dt, 'auto', 'auto', a.quick)
        r.update(dtype=str(dt).split('.')[-1], em_chunk='auto')
        res['estimate'].append(r)
        print(f"{r['dtype']:8s} em_chunk=auto: MI={r['MI']:.4f} sd={r['sigma']:.4f} "
              f"k={r['k']:2d}  {r['wall_s']:7.1f}s  peak={r['peak_gb']}", flush=True)

    batches = [1, 8, 32, 128, 300] if a.quick else [1, 8, 32, 64, 128, 300, 600, 1200]
    for dt in dtypes:
        Xt = torch.as_tensor(X, dtype=dt, device=dev)
        sat = saturation(Xt, dt, batches)
        res['saturation'][str(dt).split('.')[-1]] = sat
        print(f"saturation {str(dt).split('.')[-1]} (ms per replicate-iter): " +
              "  ".join(f"Ra={k}:{v:.3f}" for k, v in sat.items()), flush=True)

    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    lines = [f"# gmm-mi-torch sweep: {res['info'].get('gpu', dev)}", "",
             "```", json.dumps(res['info'], indent=1), "```", "",
             "## estimate(): hard pair, tester settings", "",
             "| dtype | em_chunk | MI | sigma | k | wall (s) | peak (GB) |", "|---|---|---|---|---|---|---|"]
    for r in res['estimate']:
        lines.append(f"| {r['dtype']} | {r['em_chunk']} | {r['MI']:.4f} | {r['sigma']:.4f} | {r['k']} | "
                     f"{r['wall_s']:.1f} | {r['peak_gb'] if r['peak_gb'] is None else round(r['peak_gb'], 2)} |")
    lines += ["", "## one EM step: ms per replicate-iteration vs batch size", "",
              "| dtype | " + " | ".join(f"Ra={b}" for b in batches) + " |",
              "|---|" + "---|" * len(batches)]
    for dt, sat in res['saturation'].items():
        lines.append(f"| {dt} | " + " | ".join(f"{sat[b]:.3f}" for b in batches) + " |")
    with open(a.out + '.md', 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}.json and {a.out}.md")


if __name__ == '__main__':
    sys.exit(main())
