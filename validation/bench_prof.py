import torch, time, numpy as np
import gmm_mi_torch.estimator as E
from gmm_mi_torch import em, gaussian
from torch.profiler import profile, ProfilerActivity
N=20000; rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
Xt=torch.as_tensor(X,dtype=torch.float32,device='cuda')
Ra=300
g=torch.Generator(device='cuda').manual_seed(0)
w,mu,cov=em._init_params(Xt,20,Ra,1e-6,generator=g,em_chunk=Ra)
s=torch.ones(Ra,N,device='cuda')
def step():
    chol=gaussian._chol(cov)[0]; logw=torch.log(w.clamp_min(1e-300))
    Xr=Xt.expand(Ra,N,2)
    lj=logw.unsqueeze(1)+E.log_gauss(Xr,mu,chol)
    ll_n=torch.logsumexp(lj,-1); resp=torch.exp(lj-ll_n.unsqueeze(-1))
    return em._m_step(Xr,resp,s,1e-6)
step(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14))

# --- stacking-gain proxy: 1200 replicates in one selection vs 4 x 300 serial
def run(n_inits, em_chunk):
    torch.cuda.synchronize(); t=time.time()
    E.select_batched(Xt,max_components=20,n_inits=n_inits,n_folds=3,em_chunk=em_chunk)
    torch.cuda.synchronize(); return time.time()-t
t_serial=sum(run(5,300) for _ in range(4))
t_stack=run(20,1200)
print(f"\nstacking proxy (GB10): 4 serial x (300 reps) = {t_serial:.1f}s ; 1 x 1200 reps = {t_stack:.1f}s ; gain = {t_serial/t_stack:.2f}x")
t_default=sum(run(5,8) for _ in range(4))
print(f"4 serial with default em_chunk=8: {t_default:.1f}s ; gain vs default = {t_default/t_stack:.2f}x")
