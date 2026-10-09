import json
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
gen=torch.Generator().manual_seed(1009)
N,C,H=2,10,12
rr=torch.arange(H)/ (H-1)
grid0=rr[:,None].expand(H,H)
grid1=rr[None,:].expand(H,H)
x=torch.stack([.2+.5*grid0+.05*grid1,.2+.5*grid1+.05*grid0])[:,None,:,:].repeat(1,3,1,1)
x=x+.02*torch.rand(x.shape,generator=gen)
mu=torch.cat([F.unfold(x,5,padding=2).mean(dim=(0,2)),torch.ones(1)])
kappa=torch.linalg.vector_norm(mu); r=mu/kappa
u1=torch.tensor([.6,.8]);u2=torch.tensor([.8,.6]); a1=1.1;a2=.9;c=0
k2=.5+torch.rand((5,5),generator=gen);k2=k2/k2.sum()
B=.07*torch.randn((C,9),generator=gen);B=B-B.mean(dim=0,keepdim=True)
p0=[(a1*u1[:,None]*r[:-1]).reshape(2,3,5,5),a1*u1*r[-1],a2*u2[:,None,None,None]*u1[None,:,None,None]*k2[None,None,:,:],torch.zeros(2),(B[:,None,:]*u2[None,:,None]).reshape(C,18),torch.zeros(C)]

def pack(ps):return torch.cat([p.reshape(-1) for p in ps])
def shapes(ps):return [p.shape for p in ps]
def unpack(t,ss):
    ans=[];i=0
    for s in ss:
        n=s.numel();ans.append(t[i:i+n].reshape(s));i+=n
    return ans
ss=shapes(p0)

def forward(t,ss,extra=False):
    w1,b1,w2,b2,v,b=unpack(t,ss)
    z1=F.conv2d(x,w1,b1,padding=2)
    h1,ind1=F.max_pool2d(F.relu(z1),2,return_indices=True)
    z2=F.conv2d(h1,w2,b2,padding=2)
    h2,ind2=F.max_pool2d(F.relu(z2),2,return_indices=True)
    f=F.linear(h2.flatten(1),v,b)
    if extra:return f,z1,h1,z2,h2,ind1,ind2
    return f.reshape(-1)

def get_u(ss):
    up=[torch.zeros(s) for s in ss]
    up[0][0]=mu[:-1].reshape(3,5,5);up[1][0]=1.
    return pack(up)

def calc(t,ss):
    u=get_u(ss)
    fun=lambda q:forward(q,ss)
    jf=lambda q:torch.autograd.functional.jacobian(fun,q,create_graph=True,vectorize=True)
    J,Jp=torch.autograd.functional.jvp(jf,t,u,create_graph=False)
    f=fun(t); R=J@u;K=J@J.T;Kp=Jp@J.T+J@Jp.T
    probs=f.reshape(N,C).softmax(dim=1)
    ce=((probs-1/C)*R.reshape(N,C)).sum()/N
    cap=.5*torch.trace(torch.linalg.solve(torch.eye(N*C)+K,Kp))
    return dict(J=J,Jp=Jp,f=f,R=R,K=K,Kp=Kp,ce=ce,cap=cap)

def selfpars(ps):
    w1,b1,w2,b2,v,b=ps
    return [w1[c:c+1],b1[c:c+1],w2[:,c:c+1],b2,v,b]

t0=pack(p0);q0=selfpars(p0);st=shapes(q0);ts0=pack(q0)
full=calc(t0,ss);self0=calc(ts0,st)
f,z1,h1,z2,h2,i1,i2=forward(t0,ss,True)
X2=h2[:,0].flatten(1)/(a1*a2*u2[0]);Q=torch.kron(X2@X2.T,torch.eye(C))
k=float(kappa*u1[c])
starts=[0]
for s in ss:starts.append(starts[-1]+s.numel())
Jw2=full['J'][:,starts[2]:starts[3]]
G2=Jw2@Jw2.T/a1**2
expected=2*a1*k*(G2+a2*a2*Q)
expected_s=2*a1*k*(G2+a2*a2*u1[c]**2*Q)
assert torch.max(torch.abs(full['Kp']-expected))<1e-10
assert torch.max(torch.abs(self0['Kp']-expected_s))<1e-10
assert torch.max(torch.abs(full['R']-(k/a1)*full['f']))<1e-10

def marginpool(z):
    vals=F.unfold(z,2,stride=2).reshape(N,z.shape[1],4,-1).sort(dim=2).values
    return float((vals[:,:,3]-vals[:,:,2]).min())

rho=float(torch.linalg.eigvalsh(X2@X2.T).min())
low=2*a1*k*a2*a2*rho
lows=low*float(u1[c]**2)
# A finite generic change of every trainable raw parameter, biases included.
pert=5e-6*torch.randn(t0.shape,generator=gen)
t=t0+pert;ps=unpack(t,ss);ts=pack(selfpars(ps))
act=calc(t,ss);acts=calc(ts,st)

def cert(a,ref,lower):
    d=float(torch.linalg.matrix_norm(a['J']-ref['J']))
    e=float(torch.linalg.matrix_norm(a['Jp']-ref['Jp']))
    bound=2*(e*float(torch.linalg.matrix_norm(ref['J']))+d*float(torch.linalg.matrix_norm(ref['Jp']))+d*e)
    return dict(d=d,e=e,derivative_perturbation_bound=bound,reference_eigen_lower=lower,certified_eigen_lower=lower-bound,actual_min_eigen=float(torch.linalg.eigvalsh(a['Kp']).min()),capacity_slope_lambda1=float(a['cap']))
cf=cert(act,full,low);cs=cert(acts,self0,lows)
dR=float(torch.linalg.vector_norm(act['R']-full['R']));df=float(torch.linalg.vector_norm(act['f']-full['f']))
ceerr=dR*((1-1/C)/N)**.5+float(torch.linalg.vector_norm(full['R']))*df/(2*N)
fe,ze,he,zz,hh,ii,jj=forward(t,ss,True)
nonprop_h1=float(torch.linalg.svdvals(he.transpose(0,1).flatten(1)).min())
nonprop_h2=float(torch.linalg.svdvals(hh.transpose(0,1).flatten(1)).min())
assert cf['certified_eigen_lower']>0 and cs['certified_eigen_lower']>0
assert float(full['ce'])-ceerr>0
assert torch.equal(i1,ii) and torch.equal(i2,jj)
assert nonprop_h1>0 and nonprop_h2>0
out=dict(architecture='RGB12x12 -> Conv5x5(2,pad2)+bias -> ReLU+MaxPool2 -> Conv5x5(2,pad2)+bias -> ReLU+MaxPool2 -> flatten18 -> linear10+bias',N=N,C=C,all_trainable_parameter_count=t0.numel(),mu_norm=float(kappa),k=k,X2_gram_eigenvalues=torch.linalg.eigvalsh(X2@X2.T).tolist(),reference_min_preactivation=[float(z1.min()),float(z2.min())],reference_min_pool_winner_gap=[marginpool(z1),marginpool(z2)],formula_full_max_error=float((full['Kp']-expected).abs().max()),formula_self_max_error=float((self0['Kp']-expected_s).abs().max()),mean_ray_max_error=float((full['R']-(k/a1)*full['f']).abs().max()),full=cf,literal_self=cs,raw_perturbation_l2=float(torch.linalg.vector_norm(pert)),reference_CE_mean_gradient=float(full['ce']),CE_perturbation_error_bound=ceerr,CE_certified_lower=float(full['ce'])-ceerr,actual_CE_mean_gradient=float(act['ce']),channel_nonproportional_singular_min=[nonprop_h1,nonprop_h2],winner_routing_unchanged=True)
print(json.dumps(out,indent=2))
