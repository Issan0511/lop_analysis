"""Finite reused-label task certificate: mixed ReLU gates, negative mean, all raw SGD."""
import itertools,json,math
from fractions import Fraction as F
import torch
import torch.nn.functional as TF
import sympy as sp

torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
N,C,H=2,3,8;eta=1e-4;radius=F(1,200)
rho=torch.tensor([[1.,.8],[.6,.4]])
x=torch.zeros(N,3,2,2);x[0,0]=rho;x[1,1]=rho
w=torch.tensor([[1.3,1.03,0.],[1.03,1.3,0.]])[:,:,None,None]
b=torch.full((2,),-1.)
a=torch.tensor([-1.,0.,1.]);V=a[:,None]*torch.tensor([.24,.15])[None,:];bo=.018*a
p0=tuple(t.clone().requires_grad_() for t in (w,b,V,bo))

def forward(pp,self_only=False):
    ww,bb,vv,oo=pp
    if self_only:ww=ww[:1];bb=bb[:1];vv=vv[:,:1]
    z=TF.conv2d(x,ww,bb)
    hh=TF.max_pool2d(TF.relu(z),2).flatten(1)
    return hh@vv.T+oo,z,hh

def mean(pp):return forward(pp)[1][:,0].mean()

f,z,hh=forward(p0);m0=float(mean(p0).detach())
u=torch.autograd.grad(mean(p0),p0,allow_unused=True)
u=tuple(torch.zeros_like(t) if q is None else q for t,q in zip(p0,u))
_,R=torch.autograd.functional.jvp(lambda *p:forward(p)[0],p0,u)
G=float((R*(f.softmax(1)-1/C)).sum().detach()/N)
# Explicit rational bounds, valid for every parameter point in the radius ball.
rho_bound=F(3,2);V0_bound=F(401,1000);h0_bound=F(302,1000)
J=F(5,4);grad_bound=F(15,8);hess_bound=F(97,32);u_bound=F(9,8)
assert (V0_bound+radius)**2*rho_bound**2+(h0_bound+rho_bound*radius)**2+1<J**2
assert hess_bound==J**2/2+rho_bound**2
assert float(sum(v.square().sum() for v in u).sqrt())<float(u_bound)
assert 2*rho_bound*radius<F(3,100) # strict gate + pooled winner margins
eta_q=F(1,10000);tau=H*eta_q
assert tau*grad_bound<radius
# A'(t)=Var_softmax(t*a)(a)>=8/15 for 0<=t<=1/10.
Gamma=F(81,250)*F(8,15)*F(1647,20000)
error_bound=u_bound*hess_bound*grad_bound*F(H*(H-1),2)*eta_q**2
certificate=tau*Gamma-error_bound
assert certificate>0 and G>float(Gamma)

rows=[]
for labels in itertools.product(range(C),repeat=N):
    pp=tuple(t.detach().clone().requires_grad_() for t in p0)
    y=torch.tensor(labels)
    initial_grad=torch.autograd.grad(TF.cross_entropy(forward(pp)[0],y),pp)
    frozen=tau*float(sum((uu*gg).sum() for uu,gg in zip(u,initial_grad)).detach())
    max_move=0.
    for _ in range(H):
        grads=torch.autograd.grad(TF.cross_entropy(forward(pp)[0],y),pp)
        pp=tuple((t-eta*g).detach().requires_grad_() for t,g in zip(pp,grads))
        move=float(sum((t-t0).square().sum() for t,t0 in zip(pp,p0)).sqrt().detach())
        max_move=max(max_move,move)
        assert move<float(radius)
        _,zz,_=forward(pp)
        assert torch.equal(zz.detach()>0,z.detach()>0)
        assert torch.equal(zz.flatten(2).argmax(2),z.flatten(2).argmax(2))
    actual=m0-float(mean(pp).detach())
    assert abs(actual-float(frozen))<=float(error_bound)+1e-14
    rows.append(dict(labels=labels,actual_sink=actual,frozen_sink=float(frozen),absolute_reference_error=abs(actual-float(frozen)),max_all_parameter_displacement=max_move))
actual_expectation=sum(r['actual_sink'] for r in rows)/len(rows)
frozen_expectation=sum(r['frozen_sink'] for r in rows)/len(rows)
assert actual_expectation>=float(certificate)>0
assert abs(frozen_expectation-float(tau)*G)<1e-14
# Same label is reused H times; variance is H times the fresh-label-per-step variance.
centered_R=R-R.mean(1,keepdim=True)
variances=centered_R.square().mean(1)
variance_reused=float((float(tau)/N)**2*variances.sum())
variance_fresh=float(H*(eta/N)**2*variances.sum())
enumerated_variance=sum((r['frozen_sink']-frozen_expectation)**2 for r in rows)/len(rows)
assert abs(variance_reused-enumerated_variance)<1e-18
assert abs(variance_reused/H-variance_fresh)<1e-18
# Literal all-parameter NTK/self: exact rational positive signs.
Q=sp.Rational
HS=sp.Matrix([[Q(3,10),Q(3,100)],[Q(3,100),Q(3,10)]])
VS=sp.Matrix([[-Q(24,100),-Q(15,100)],[0,0],[Q(24,100),Q(15,100)]])
Xi=sp.Matrix([[1,0,0,1],[0,1,0,1]])
rs=sp.Matrix([Q(27,20),Q(27,20)]);hc=HS[:,0]
DKS=sp.kronecker_product(rs*hc.T+hc*rs.T,sp.eye(C))
capacity=[]
K_delta=2*N*J*rho_bound*radius
DK_delta=2*N*rho_bound**2*u_bound*radius
capacity_perturbation_bound=F(N*C,2)*(DK_delta+K_delta*F(11,10))
assert capacity_perturbation_bound==F(441,1600)
for isolated in (False,True):
    HS0=hc if isolated else HS;VS0=VS[:,0] if isolated else VS
    KS=sp.kronecker_product(Xi*Xi.T,VS0*VS0.T)+sp.kronecker_product(HS0*HS0.T+sp.ones(N),sp.eye(C))
    cap=(Q(1,2)*((sp.eye(N*C)+KS).inv()*DKS).trace()).factor();assert cap>Q(capacity_perturbation_bound.numerator,capacity_perturbation_bound.denominator)
    def kernel(*pp):
        jac=torch.autograd.functional.jacobian(lambda *v:forward(v,isolated)[0].flatten(),pp,create_graph=True)
        jj=torch.cat([j.reshape(N*C,-1) for j in jac],1)
        return jj@jj.T
    KK,DK=torch.autograd.functional.jvp(kernel,p0,u)
    kerr=float((KK-torch.tensor(KS.tolist(),dtype=torch.float64)).abs().max())
    derr=float((DK-torch.tensor(DKS.tolist(),dtype=torch.float64)).abs().max())
    assert kerr<1e-12 and derr<1e-12
    capacity.append(dict(isolated=isolated,capacity_derivative_exact=str(cap),capacity_derivative=float(cap),uniform_ball_derivative_lower_bound=float(cap)-float(capacity_perturbation_bound),kernel_error=kerr,derivative_error=derr))
print(json.dumps(dict(scope='One finite reused-label task, all raw SGD, mixed ReLU gates and negative mean; not a long-time or Adam theorem',images=N,classes=C,updates_per_task=H,learning_rate=eta,labels_reused=True,label_assignments_enumerated=len(rows),initial_mean=m0,pooled_features=hh.detach().tolist(),pooled_rank=int(torch.linalg.matrix_rank(hh)),initial_expected_mean_gradient=G,rigorous_gradient_lower_bound=str(Gamma),parameter_radius=float(radius),gradient_norm_bound=float(grad_bound),gradient_lipschitz_bound=float(hess_bound),reference_error_bound=float(error_bound),certified_expected_sink_lower_bound=float(certificate),certified_expected_sink_lower_bound_exact=str(certificate),actual_expected_sink=actual_expectation,frozen_expected_sink=frozen_expectation,frozen_label_reuse_variance=variance_reused,fresh_per_step_label_variance=variance_fresh,variance_ratio=variance_reused/variance_fresh,assignments=rows,capacity_perturbation_bound=float(capacity_perturbation_bound),full_and_literal_self_capacity=capacity),indent=2))
