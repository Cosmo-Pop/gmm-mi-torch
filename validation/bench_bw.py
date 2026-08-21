import torch, time, numpy as np
import gmm_mi_torch.estimator as E
from gmm_mi_torch import estimate
def T(f,n=10):
    f(); torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
R,N,K=300,20000,20
t=torch.randn(R,N,K,device='cuda'); nbytes=t.numel()*4
print(f"tensor (R,N,K) = {nbytes/1e6:.0f} MB")
ms=T(lambda: torch.exp(t)); print(f"exp            : {ms:6.1f} ms  -> {2*nbytes/ms/1e6:6.0f} GB/s effective")
ms=T(lambda: t*2.0+1.0); print(f"mul+add (2 ops): {ms:6.1f} ms")
ms=T(lambda: torch.logsumexp(t,-1)); print(f"logsumexp(-1)  : {ms:6.1f} ms  -> {nbytes/ms/1e6:6.0f} GB/s read")
ms=T(lambda: t.sum(1)); print(f"sum(dim=1)     : {ms:6.1f} ms")
F=torch.randn(1,N,6,device='cuda'); C=torch.randn(R,6,K,device='cuda')
ms=T(lambda: torch.matmul(F,C)); print(f"matmul (1,N,6)@(R,6,K): {ms:6.1f} ms")
ms=T(lambda: torch.einsum('nf,rfk->rnk',F[0],C)); print(f"einsum same    : {ms:6.1f} ms")
wr=torch.rand(R,N,K,device='cuda'); G=torch.randn(1,N,5,device='cuda')
ms=T(lambda: torch.matmul(wr.transpose(1,2),G)); print(f"matmul (R,K,N)@(1,N,5): {ms:6.1f} ms")
ms=T(lambda: torch.matmul(wr.transpose(1,2).contiguous(),G.expand(R,N,5))); print(f"  same, contiguous+expand: {ms:6.1f} ms")
# fused alternative for resp: softmax along K
ms=T(lambda: torch.softmax(t,-1)); print(f"softmax(-1)    : {ms:6.1f} ms  (vs logsumexp+sub+exp)")
ms=T(lambda: torch.exp(t-torch.logsumexp(t,-1,keepdim=True))); print(f"logsumexp+sub+exp: {ms:6.1f} ms")

# MI under both protocols for the benchmark pair
rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
true_mi=-0.5*np.log(1-rho**2)
kw=dict(device='cuda',dtype=torch.float32,n_bootstrap=100,max_components=20,n_inits=5,n_folds=3,em_chunk=300,return_k=True)
for proto in ('batched','sequential'):
    torch.cuda.synchronize(); t0=time.time()
    m,s,k=estimate(X,'MI',protocol=proto,**kw); torch.cuda.synchronize()
    print(f"{proto:10s}: MI={m:.4f} +- {s:.4f}  k={k}  {time.time()-t0:.1f}s   (analytic, invariant to the monotone warp: {true_mi:.4f})")
