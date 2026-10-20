import torch, time, numpy as np
import gmm_mi_torch.estimator as E
N=20000; rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
Xt=torch.as_tensor(X,dtype=torch.float32,device='cuda')

# --- A: trace active replicate count per iteration inside _em_fit_core
trace=[]
orig=E._em_fit_core
def traced(X,s,w,mu,cov,reg_covar=1e-6,threshold_fit=1e-5,max_iter=500,n_chunk=None):
    R,N=s.shape; dev=X.device
    ai=torch.arange(R,device=dev)
    w_out,mu_out,cov_out=w.clone(),mu.clone(),cov.clone()
    ll_out=torch.full((R,),-torch.inf,dtype=X.dtype,device=dev)
    prev=torch.full((R,),-torch.inf,dtype=X.dtype,device=dev)
    sa=s; it_trace=[]
    for it in range(max_iter):
        Ra=ai.shape[0]; torch.cuda.synchronize(); t=time.perf_counter()
        chol, cov=E._chol(cov)
        logw=torch.where(w>0,torch.log(w.clamp_min(1e-300)),torch.full_like(w,-torch.inf))
        Xr=X.expand(Ra,N,X.shape[-1])
        logjoint=logw.unsqueeze(1)+E.log_gauss(Xr,mu,chol)
        ll_n=torch.logsumexp(logjoint,dim=-1)
        resp=torch.exp(logjoint-ll_n.unsqueeze(-1))
        ll=(sa*ll_n).sum(1)/sa.sum(1)
        w,mu,cov=E._m_step(Xr,resp,sa,reg_covar)
        done=(ll-prev).abs()<threshold_fit
        torch.cuda.synchronize(); it_trace.append((Ra,time.perf_counter()-t))
        if done.any():
            di=ai[done]
            w_out[di],mu_out[di],cov_out[di]=w[done],mu[done],cov[done]
            ll_out[di]=ll[done]
            keep=~done
            ai,sa,prev=ai[keep],sa[keep],ll[keep]
            w,mu,cov=w[keep],mu[keep],cov[keep]
            if ai.shape[0]==0: break
        else: prev=ll
    else:
        w_out[ai],mu_out[ai],cov_out[ai],ll_out[ai]=w,mu,cov,prev
    trace.append(it_trace)
    return w_out,mu_out,cov_out,ll_out
E._em_fit_core=traced
torch.cuda.synchronize(); t=time.time()
k,_,_=E.select_batched(Xt,max_components=20,n_inits=5,n_folds=3,em_chunk=300)
torch.cuda.synchronize(); print(f"select k={k} total={time.time()-t:.1f}s, em_fit calls={len(trace)}")
tr=trace[0]
print(f"iterations={len(tr)}")
tot=sum(t for _,t in tr)
print(f"sum iter time={tot:.1f}s")
# time spent by active-count bucket
buckets=[(300,151),(150,65),(64,17),(16,5),(4,0)]
for hi,lo in buckets:
    sel=[(r,t) for r,t in tr if lo<r<=hi]
    print(f"  Ra in ({lo},{hi}]: iters={len(sel):4d} time={sum(t for _,t in sel):6.1f}s  mean/iter={np.mean([t for _,t in sel])*1e3 if sel else 0:6.1f}ms")
print("active count at iters 0,10,20,50,100,200,300,400:",[tr[i][0] for i in (0,10,20,50,100,200,300,400) if i<len(tr)])

# --- B: per-iteration throughput vs batch size (full EM step, K=20, N=20k, d=2)
E._em_fit_core=orig
print("\nthroughput: one EM step, K=20 d=2 N=20000")
for Ra in (1,4,8,32,128,300,1200):
    g=torch.Generator(device='cuda').manual_seed(0)
    w,mu,cov=E._init_params(Xt,20,Ra,1e-6,generator=g,em_chunk=Ra)
    s=torch.ones(Ra,N,device='cuda')
    def step():
        chol=E._chol(cov)[0]; logw=torch.log(w.clamp_min(1e-300))
        Xr=Xt.expand(Ra,N,2)
        lj=logw.unsqueeze(1)+E.log_gauss(Xr,mu,chol)
        ll_n=torch.logsumexp(lj,-1); resp=torch.exp(lj-ll_n.unsqueeze(-1))
        return E._m_step(Xr,resp,s,1e-6)
    step(); torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(5): step()
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/5
    print(f"  Ra={Ra:5d}: {dt*1e3:8.2f} ms/iter  {dt/Ra*1e3:7.3f} ms per replicate-iter  peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    torch.cuda.reset_peak_memory_stats()
