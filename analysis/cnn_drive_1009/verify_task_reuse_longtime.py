"""Verify damped raw-parameter CNN SGD with taskwise label reuse."""
import json, math
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(100925)
C,N,S,L=10,4,4,2
u0=torch.tensor([2.,3.,5.]);u0=u0/u0.norm()
u1=torch.tensor([4.,8.,3.]);u1=u1/u1.norm()
u2=torch.tensor([5.,4.,7.]);u2=u2/u2.norm()
coarse=torch.tensor([[1.4,.2,.3,.4],[.3,1.5,.2,.4],[.4,.3,1.6,.2],[.2,.4,.3,1.7]])
rho=1+torch.arange(4)[:,None]*.01+torch.arange(4)[None,:]*.001
scalar_x=torch.zeros(N,8,8)
for n in range(N):
 for pos in range(S):
  row,col=divmod(pos,2);scalar_x[n,row*4:(row+1)*4,col*4:(col+1)*4]=coarse[n,pos]*rho
x=u0[None,:,None,None]*scalar_x[:,None]
X=F.max_pool2d(F.max_pool2d(scalar_x[:,None],2),2).flatten(1)
assert torch.linalg.matrix_rank(X)==4
B=.13*torch.randn(C,S);B-=B.mean(0,keepdim=True)
a=1.2
assert a*a>float(B.square().sum())
B_initial=B.clone();a_initial=a
K=math.sqrt(2)*float(X.norm(dim=1).max())

def params(aa,BB,self_only=False):
 w1=aa*torch.outer(u1,u0)[:,:,None,None]
 w2=aa*torch.outer(u2,u1)[:,:,None,None]
 head=(BB[:,None,:]*u2[None,:,None]).reshape(C,-1)
 if self_only:w1=w1[0:1];w2=w2[:,0:1]
 return tuple(t.detach().clone().requires_grad_() for t in (w1,w2,head))

def forward(pp,ret=False):
 w1,w2,head=pp
 z1=F.conv2d(x,w1);h1=F.max_pool2d(F.relu(z1),2)
 z2=F.conv2d(h1,w2);h2=F.max_pool2d(F.relu(z2),2)
 logits=F.linear(h2.flatten(1),head)
 return (logits,z1,z2,h2) if ret else logits

H=8;tasks=80;kappa=.15;power=.75
assert H%2==0
D0=a*a-float(B.square().sum())
d_lower=D0*math.exp(-4*kappa*kappa/H)
CF=math.sqrt(2)*min(1,math.sqrt(D0))
R=K/math.sqrt(2)
LF=2/math.sqrt(d_lower)+(L+1)/(2*math.sqrt(2))+math.sqrt(2)+(R/2)*math.sqrt(L*L+1)
p=params(a,B);max_param_error=0.;max_D_error=0.;max_block_error=0.;max_block_ratio=0.;min_D=D0
rows=[]
for task in range(tasks):
 labels=torch.randint(C,(N,))
 delta=kappa/(H*(task+1)**power)
 aa0=a;BB0=B.clone();z0=torch.cat((torch.tensor([a]),B.flatten()))
 A0=(((a**L*(X@B.T)).softmax(1)-F.one_hot(labels,C)).T@X)/N
 scale0=math.sqrt(a*a-float(B.square().sum()))/(K*(1+a**(L+1)))
 F0=-scale0*torch.cat((torch.tensor([float((B*A0).sum())/a]),A0.flatten()))
 counts=torch.zeros(N,dtype=torch.int64)
 for epoch in range(H//2):
  order=torch.randperm(N)
  for jj in range(2):
   idx=order[2*jj:2*jj+2];counts[idx]+=1
   logits=forward(p)
   loss=F.cross_entropy(logits[idx],labels[idx]);grad=torch.autograd.grad(loss,p)
   ff=a**L*(X[idx]@B.T)
   A=((ff.softmax(1)-F.one_hot(labels[idx],C)).T@X[idx])/len(idx)
   DD=a*a-float(B.square().sum());eta=delta*math.sqrt(DD)/(K*a**L*(1+a**(L+1)))
   scalar=float((B*A).sum())
   anext=a-eta*a**(L-1)*scalar;Bnext=B-eta*a**L*A
   Dnext=anext*anext-float(Bnext.square().sum())
   Dformula=DD-eta*eta*a**(2*L-2)*(a*a*float(A.square().sum())-scalar*scalar)
   max_D_error=max(max_D_error,abs(Dnext-Dformula))
   assert Dnext>=DD*(1-delta*delta)-1e-12 and Dnext<=DD+1e-12
   p=tuple((v-eta*g).detach().requires_grad_() for v,g in zip(p,grad))
   predicted=params(anext,Bnext)
   max_param_error=max(max_param_error,max(float((v-w).abs().max().detach()) for v,w in zip(p,predicted)))
   min_D=min(min_D,Dnext);a,B=anext,Bnext
 assert torch.equal(counts,torch.full_like(counts,H//2))
 zfinal=torch.cat((torch.tensor([a]),B.flatten()))
 err=float((zfinal-z0-H*delta*F0).norm())
 bound=.5*LF*CF*H*(H-1)*delta*delta
 assert err<=bound+1e-12
 max_block_error=max(max_block_error,err);max_block_ratio=max(max_block_ratio,err/bound)
 rows.append(dict(task=task+1,a=a,head_squared_norm=float(B.square().sum()),task_freeze_error=err,task_freeze_error_bound=bound))
assert max_param_error<1e-12 and max_D_error<1e-12 and min_D>d_lower
print(json.dumps(dict(scope='Actual all-raw two-Conv/ReLU/MaxPool SGD, fresh iid labels only per task, repeated labels and equal-coverage minibatches; proof needs specified decaying/damped learning rate',seed=100925,classes=C,images=N,minibatch=2,layers=L,widths=[3,3,3],input_feature_rank=int(torch.linalg.matrix_rank(X)),updates_per_task=H,tasks=tasks,total_updates=H*tasks,epochs_per_task=H//2,kappa=kappa,decay_power=power,initial_a=a_initial,initial_head_squared_norm=float(B_initial.square().sum()),proved_limiting_a_squared_upper_bound=a_initial*a_initial-float(B_initial.square().sum()),deterministic_D_lower_bound=d_lower,minimum_D=min_D,global_lipschitz_bound=LF,global_vector_field_bound=CF,max_raw_parameter_recurrence_error=max_param_error,max_balance_identity_error=max_D_error,max_task_reference_error=max_block_error,max_task_error_to_bound_ratio=max_block_ratio,final_a=a,final_head_squared_norm=float(B.square().sum()),first_and_last_task_records=[rows[0],rows[-1]]),indent=2))
