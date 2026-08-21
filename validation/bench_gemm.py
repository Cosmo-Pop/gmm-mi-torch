import torch, time, math, numpy as np
import gmm_mi_torch.estimator as E
N=20000; rho=0.3
X=np.random.default_rng(0).multivariate_normal([0,0],[[1,rho],[rho,1]],N)
X[:,1]=np.sign(X[:,1])*np.abs(X[:,1])**1.5
Xt=torch.as_tensor(X,dtype=torch.float32,device='cuda')
Ra,K,d=300,20,2
g=torch.Generator(device='cuda').manual_seed(0)
w,mu,cov=E._init_params(Xt,K,Ra,1e-6,generator=g,em_chunk=Ra)
s=torch.ones(Ra,N,device='cuda')
def T(f,n=5):
    f(); torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): out=f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3, out

# ---- current pieces
chol=E._chol(cov); logw=torch.log(w.clamp_min(1e-300)); Xr=Xt.expand(Ra,N,d)
t_lg,lj=T(lambda: logw.unsqueeze(1)+E.log_gauss(Xr,mu,chol))
def resp_f():
    ll_n=torch.logsumexp(lj,-1); return torch.exp(lj-ll_n.unsqueeze(-1)), ll_n
t_resp,(resp,ll_n)=T(resp_f)
t_m,(w1,mu1,cov1)=T(lambda: E._m_step(Xr,resp,s,1e-6))
print(f"current: log_gauss={t_lg:.1f}ms  resp={t_resp:.1f}ms  m_step={t_m:.1f}ms  total={t_lg+t_resp+t_m:.1f}ms")

# ---- GEMM-formulated E-step: maha = x^T P x - 2 x^T P mu + mu^T P mu, with P = cov^-1
# features F = [x_i x_j (i<=j), x_i, 1]  (N, d(d+1)/2 + d + 1), per-component coefficient vector
iu=torch.triu_indices(d,d).cuda()
def feats(X):
    q=X[:,iu[0]]*X[:,iu[1]]                      # (N, d(d+1)/2)
    return torch.cat([q,X,torch.ones(X.shape[0],1,device=X.device,dtype=X.dtype)],1)
F=feats(Xt)                                       # (N,6)
def e_step_gemm(w,mu,cov):
    L=torch.linalg.cholesky(cov)                  # (R,K,d,d)
    P=torch.cholesky_inverse(L)                   # (R,K,d,d)
    logdet=torch.log(torch.diagonal(L,dim1=-2,dim2=-1)).sum(-1)
    Pm=(P@mu.unsqueeze(-1)).squeeze(-1)           # (R,K,d)
    c=(mu*Pm).sum(-1)                             # (R,K)
    offdiag=(iu[0]!=iu[1]).to(P.dtype)            # off-diagonal terms appear twice
    Pq=P[...,iu[0],iu[1]]*(1+offdiag)             # (R,K,d(d+1)/2)
    coef=torch.cat([Pq,-2*Pm,c.unsqueeze(-1)],-1) # (R,K,6)
    maha=torch.matmul(F.unsqueeze(0),coef.transpose(1,2))  # (1,N,6)@(R,6,K) -> (R,N,K)
    lj=torch.log(w.clamp_min(1e-300)).unsqueeze(1)-0.5*(maha+d*E.LOG2PI)-logdet.unsqueeze(1)
    ll_n=torch.logsumexp(lj,-1); resp=torch.exp(lj-ll_n.unsqueeze(-1))
    return resp,ll_n
t_e2,(resp2,ll2)=T(lambda: e_step_gemm(w,mu,cov))
print(f"gemm E-step: {t_e2:.1f}ms   max|resp diff|={(resp2-resp).abs().max().item():.2e}  max|ll diff|={(ll2-ll_n).abs().max().item():.2e}")

# ---- GEMM-formulated M-step: centre data at per-replicate data mean (component-independent) for fp32 stability
xbar=Xt.mean(0); Xc=Xt-xbar                        # (N,d)
XX=(Xc[:,iu[0]]*Xc[:,iu[1]])                       # (N, d(d+1)/2)
G=torch.cat([Xc,XX],1)                             # (N, d + d(d+1)/2)
def m_step_gemm(resp,s):
    wr=resp*s.unsqueeze(-1)                        # (R,N,K)
    Nk=wr.sum(1)                                   # (R,K)
    S=torch.matmul(wr.transpose(1,2),G.unsqueeze(0))   # (R,K,N)@(1,N,5) -> (R,K,5)
    Nks=Nk.clamp_min(1e-10).unsqueeze(-1)
    m=S[...,:d]/Nks                                # centred means (R,K,d)
    q=S[...,d:]/Nks                                # E[x_i x_j] centred (R,K,3)
    C=torch.zeros(Ra,K,d,d,device=Xt.device,dtype=Xt.dtype)
    C[...,iu[0],iu[1]]=q; C[...,iu[1],iu[0]]=q
    C=C-m.unsqueeze(-1)*m.unsqueeze(-2)+1e-6*torch.eye(d,device=Xt.device)
    mu=m+xbar
    wn=Nk/s.sum(1,keepdim=True); wn=wn/wn.sum(-1,keepdim=True)
    return wn,mu,C
t_m2,(w2,mu2,cov2)=T(lambda: m_step_gemm(resp,s))
print(f"gemm M-step: {t_m2:.1f}ms   max|mu diff|={(mu2-mu1).abs().max().item():.2e}  max|cov diff|={(cov2-cov1).abs().max().item():.2e}  max|w diff|={(w2-w1).abs().max().item():.2e}")
print(f"gemm total per iter: {t_e2+t_m2:.1f}ms vs current {t_lg+t_resp+t_m:.1f}ms  -> {(t_lg+t_resp+t_m)/(t_e2+t_m2):.1f}x")

# ---- sequential protocol cost on this pair
torch.cuda.synchronize(); t=time.time()
k,_,_=E.select_sequential(Xt,max_components=20,n_inits=5,n_folds=3,em_chunk=15)
torch.cuda.synchronize(); print(f"\nsequential protocol: k={k} time={time.time()-t:.1f}s (batched was ~42s)")
