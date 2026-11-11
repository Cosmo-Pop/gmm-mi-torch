import torch, time
import gmm_mi_torch.estimator as E
from gmm_mi_torch import em, gaussian
K=2
for N in (2000,5000,20000):
    Xt=torch.randn(N,2,device='cuda')
    print(f"N={N}: per replicate-iter (ms), K=20 d=2")
    for Ra in (8,32,128,512):
        g=torch.Generator(device='cuda').manual_seed(0)
        w,mu,cov=em._init_params(Xt,20,Ra,1e-6,generator=g,em_chunk=Ra)
        s=torch.ones(Ra,N,device='cuda')
        def step():
            chol=gaussian._chol(cov)[0]; logw=torch.log(w.clamp_min(1e-300)); Xr=Xt.expand(Ra,N,2)
            lj=logw.unsqueeze(1)+E.log_gauss(Xr,mu,chol); ll_n=torch.logsumexp(lj,-1)
            return em._m_step(Xr,torch.exp(lj-ll_n.unsqueeze(-1)),s,1e-6)
        step(); torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(5): step()
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/5
        print(f"   Ra={Ra:4d}: {dt/Ra*1e3:6.3f}")
