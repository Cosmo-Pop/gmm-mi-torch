import torch, time, numpy as np
from gmm_mi_torch import estimate
from gmm_mi_torch.estimator import select_batched, em_fit, _chol, gmm_sample, _INTEGRANDS
N=20000; rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
kw=dict(device='cuda',dtype=torch.float32,n_bootstrap=100,max_components=20,n_inits=5,n_folds=3,mc_samples=100_000)
estimate(X[:2000],'MI',device='cuda',dtype=torch.float32,max_components=3,n_bootstrap=4)  # warm-up
for em_chunk in (8,64,300):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.time()
    out=estimate(X,'MI',em_chunk=em_chunk,return_k=True,**kw)
    torch.cuda.synchronize(); dt=time.time()-t
    print(f"estimate em_chunk={em_chunk}: MI={out[0]:.4f} sd={out[1]:.4f} k={out[2]} {dt:.1f}s peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",flush=True)
# phase breakdown
Xt=torch.as_tensor(X,dtype=torch.float32,device='cuda')
for em_chunk in (8,300):
    torch.cuda.synchronize(); t=time.time()
    k,(w1,mu1,cov1),_=select_batched(Xt,max_components=20,n_inits=5,n_folds=3,em_chunk=em_chunk)
    torch.cuda.synchronize(); t1=time.time()
    B=100; g=torch.Generator(device='cuda').manual_seed(1)
    s=torch.multinomial(torch.full((B,N),1/N,device='cuda'),N,replacement=True,generator=g)
    s=torch.zeros(B,N,device='cuda').scatter_add_(1,s,torch.ones(B,N,device='cuda'))
    K=w1.shape[0]
    w,mu,cov,_=em_fit(Xt,s,w1.expand(B,K).contiguous(),mu1.expand(B,K,2).contiguous(),cov1.expand(B,K,2,2).contiguous(),em_chunk=em_chunk)
    torch.cuda.synchronize(); t2=time.time()
    chol=_chol(cov); vals=[]
    for i in range(0,B,8):
        sl=slice(i,i+8); pts=gmm_sample(w[sl],mu[sl],chol[sl],100_000,generator=g)
        vals.append(_INTEGRANDS['MI'](pts,w[sl],mu[sl],cov[sl],chol[sl]).mean(1))
    torch.cuda.synchronize(); t3=time.time()
    print(f"em_chunk={em_chunk}: k={k} select={t1-t:.1f}s bootstrapEM={t2-t1:.1f}s MC={t3-t2:.1f}s",flush=True)
