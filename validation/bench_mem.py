"""Peak CUDA memory per replicate for one EM step and for one MC chunk, to fit the auto-chunk model."""
import torch, itertools
import gmm_mi_torch.estimator as E
from gmm_mi_torch import em, gaussian
dev='cuda'
def peak_em(N,K,d,Ra,dt,n_chunk=None):
    X=torch.randn(N,d,device=dev,dtype=dt)
    g=torch.Generator(device=dev).manual_seed(0)
    w,mu,cov=em._init_params(X,K,Ra,1e-6,generator=g,em_chunk=Ra)
    s=torch.ones(Ra,N,device=dev,dtype=dt)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base=torch.cuda.memory_allocated()
    em._em_fit_core(X,s,w,mu,cov,max_iter=2,n_chunk=n_chunk)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()-base
print("EM step: units = peak / (Ra*Neff*K*itemsize)")
for dt in (torch.float32,torch.float64):
    for d in (1,2,3,5,8):
        for K in (5,20):
            N,Ra=20000,64
            p=peak_em(N,K,d,Ra,dt)
            u=p/(Ra*N*K*dt.itemsize)
            print(f"  {str(dt)[6:]} d={d} K={K:2d}: peak={p/1e9:5.2f} GB  units={u:6.2f}   (d^2+3d+3={d*d+3*d+3})")
print("EM step, n_chunk path (N=200000, n_chunk=4096, Neff=4096)")
for d in (2,5):
    p=peak_em(200000,20,d,64,torch.float32,n_chunk=4096)
    print(f"  d={d}: peak={p/1e9:5.2f} GB units={p/(64*4096*20*4):6.2f}")
print("MC chunk: units = peak / (b*S*K*itemsize)")
for d in (2,3,5):
    K,S,b=20,100_000,8
    g=torch.Generator(device=dev).manual_seed(0)
    X=torch.randn(2000,d,device=dev)
    w,mu,cov=em._init_params(X,K,b,1e-6,generator=g,em_chunk=b)
    chol, cov=gaussian._chol(cov)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base=torch.cuda.memory_allocated()
    pts=E.gmm_sample(w,mu,chol,S,generator=g)
    q='MI' if d==2 else ('TC')
    E._INTEGRANDS[q](pts,w,mu,cov,chol).mean(1)
    torch.cuda.synchronize(); p=torch.cuda.max_memory_allocated()-base
    print(f"  d={d} {q}: peak={p/1e9:5.2f} GB units={p/(b*S*K*4):6.2f}")
