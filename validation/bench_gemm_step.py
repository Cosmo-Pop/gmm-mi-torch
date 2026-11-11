import torch, time, numpy as np
import gmm_mi_torch.estimator as E
from gmm_mi_torch import em, gaussian

# --- exact per-step parity at fp64 (CPU)
torch.manual_seed(0)
X = torch.randn(2000, 3, dtype=torch.float64)
g = torch.Generator().manual_seed(0)
w, mu, cov = em._init_params(X, 4, 6, 1e-6, generator=g)
s = torch.rand(6, 2000, dtype=torch.float64) + 0.5
# legacy step
chol, cov = gaussian._chol(cov); logw = torch.log(w.clamp_min(1e-300))
Xr = X.expand(6, 2000, 3)
lj = logw.unsqueeze(1) + E.log_gauss(Xr, mu, chol)
ll_n = torch.logsumexp(lj, -1); resp_l = torch.exp(lj - ll_n.unsqueeze(-1))
ll_l = (s * ll_n).sum(1) / s.sum(1)
wl, mul, covl = em._m_step(Xr, resp_l.clone(), s, 1e-6)
# gemm step
F = em._features(X)
resp_g, ll_g = em._estep_gemm(F, s, w, mu, cov)
resp_cmp = resp_g.clone()
wg, mug, covg = em._mstep_gemm(F, resp_g, s, 1e-6, 3)
print("step parity fp64: |dresp|=%.2e |dll|=%.2e |dw|=%.2e |dmu|=%.2e |dcov|=%.2e" % (
    (resp_cmp-resp_l).abs().max(), (ll_g-ll_l).abs().max(),
    (wg-wl).abs().max(), (mug-mul).abs().max(), (covg-covl).abs().max()))
# nchunk gemm parity vs full gemm
ll_c, wc, muc, covc = em._em_step_gemm_nchunked(X, s, w, mu, cov, 1e-6, 300)
print("nchunk parity fp64: |dll|=%.2e |dmu|=%.2e |dcov|=%.2e" % (
    (ll_c-ll_g).abs().max(), (muc-mug).abs().max(), (covc-covg).abs().max()))

# --- GB10 speed: one step at Ra=300, N=20k, K=20, d=2, fp32
Xc = torch.randn(20000, 2, device='cuda')
g = torch.Generator(device='cuda').manual_seed(0)
w, mu, cov = em._init_params(Xc, 20, 300, 1e-6, generator=g, em_chunk=300)
s1 = torch.ones(300, 20000, device='cuda')
F = em._features(Xc)
def step():
    r, ll = em._estep_gemm(F, s1, w, mu, cov)
    return em._mstep_gemm(F, r, s1, 1e-6, 2)
step(); torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(5): step()
torch.cuda.synchronize()
print(f"gemm step GB10 fp32 Ra=300: {(time.perf_counter()-t)/5*1e3:.1f} ms (legacy was 206)  peak={torch.cuda.max_memory_allocated()/1e9:.2f} GB")
