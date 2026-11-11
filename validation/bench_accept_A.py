"""Acceptance for item A: the hard pair with default (auto) chunks vs the old fixed default."""
import torch, time, numpy as np
from gmm_mi_torch import chunking
from gmm_mi_torch import estimate
N=20000; rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
kw=dict(device='cuda',dtype=torch.float32,n_bootstrap=100,max_components=20,n_inits=5,n_folds=3,mc_samples=100_000,return_k=True)
estimate(X[:2000],'MI',device='cuda',dtype=torch.float32,max_components=3,n_bootstrap=4)
free,_=torch.cuda.mem_get_info()
print(f"free={free/1e9:.0f} GB; auto em_chunk (selection, 300 reps) -> {chunking._auto_chunk(300, chunking._em_bytes_per_replicate(N,20,2,torch.float32), 'cuda')}, "
      f"mc_chunk -> {chunking._auto_chunk(100, chunking._mc_bytes_per_replicate(100_000,20,2,torch.float32), 'cuda')}")
for label,extra in (("old default em_chunk=8, mc_chunk=8",dict(em_chunk=8,mc_chunk=8)),("new default 'auto'",{})):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.time()
    m,s,k=estimate(X,'MI',**kw,**extra); torch.cuda.synchronize()
    print(f"{label:36s}: MI={m:.4f} sd={s:.4f} k={k} {time.time()-t:.1f}s peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
