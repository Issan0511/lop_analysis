"""Verify full raw-parameter CNN SGD invariant family and literal NTK signs."""
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

def kernel(pp):
 JJ=torch.autograd.functional.jacobian(lambda *v:forward(v).flatten(),pp,create_graph=True)
 J=torch.cat([j.reshape(N*C,-1) for j in JJ],1)
 return J@J.T

def kernel_record(aa,BB,self_only):
 pp=params(aa,BB,self_only)
 f,z1,_,_=forward(pp,True)
 m=z1[:,0].mean()
 uu=torch.autograd.grad(m,pp,retain_graph=True,allow_unused=True)
 uu=tuple(torch.zeros_like(t) if q is None else q for t,q in zip(pp,uu))
 KK,DK=torch.autograd.functional.jvp(lambda *v:kernel(v),pp,uu)
 _,R=torch.autograd.functional.jvp(lambda *v:forward(v),pp,uu)
 bb=(X@BB.T).flatten()
 QQ=torch.kron(X@X.T,torch.eye(C))
 k=float(scalar_x.mean())*float(u1[0])
 if self_only:
  K_formula=2*aa*aa*u1[0]**2*torch.outer(bb,bb)+aa**4*u1[0]**4*QQ
  DK_formula=2*aa*k*(torch.outer(bb,bb)+aa*aa*u1[0]**2*QQ)
 else:
  K_formula=2*aa*aa*torch.outer(bb,bb)+aa**4*QQ
  DK_formula=2*aa*k*(torch.outer(bb,bb)+aa*aa*QQ)
 kernel_error=float((KK-K_formula).abs().max())
 derivative_error=float((DK-DK_formula).abs().max())
 assert kernel_error<1e-10 and derivative_error<1e-10
 lam=.1
 gradient_capacity=.5*torch.trace(torch.linalg.solve(KK+lam*torch.eye(N*C),DK))
 gradient_ce=(R*(f.softmax(1)-.1)).sum()/N
 eig=torch.linalg.eigvalsh((DK+DK.T)/2)
 assert gradient_capacity>0 and gradient_ce>0
 assert float(eig.min())>-1e-9
 return {'self_only':self_only,'exact_kernel_formula_error':kernel_error,'exact_kernel_derivative_formula_error':derivative_error,'NTK_shape':list(KK.shape),'capacity_mean_direction':float(gradient_capacity.detach()),'CE_mean_direction':float(gradient_ce.detach()),'minimum_eigenvalue_NTK_derivative':float(eig.min()),'maximum_eigenvalue_NTK_derivative':float(eig.max()),'conv1_channels':pp[0].shape[0]}

initial_kernel=[kernel_record(a,B,False),kernel_record(a,B,True)]
p=params(a,B);max_param_error=0.;max_forward_error=0.;max_D_error=0.;min_D=math.inf
for step in range(300):
 labels=torch.randint(C,(N,))
 logits=forward(p)
 expected_logits=a**L*(X@B.T)
 max_forward_error=max(max_forward_error,float((logits-expected_logits).abs().max().detach()))
 loss=F.cross_entropy(logits,labels)
 grads=torch.autograd.grad(loss,p)
 G=((expected_logits.softmax(1)-F.one_hot(labels,C)).T@X)/N
 D=a*a-float(B.square().sum())
 gamma=.08/((step+1)**.75)
 eta=gamma*math.sqrt(D)/(K*a**L)
 scalar=float((B*G).sum())
 a_next=a-eta*a**(L-1)*scalar
 B_next=B-eta*a**L*G
 D_next=a_next*a_next-float(B_next.square().sum())
 D_formula=D+eta*eta*a**(2*L-2)*(scalar*scalar-a*a*float(G.square().sum()))
 max_D_error=max(max_D_error,abs(D_next-D_formula))
 assert D_next>=D*(1-gamma*gamma)-1e-12 and D_next<=D+1e-12
 p=tuple((t-eta*g).detach().requires_grad_() for t,g in zip(p,grads))
 predicted=params(a_next,B_next)
 max_param_error=max(max_param_error,max(float((t-q).abs().max().detach()) for t,q in zip(p,predicted)))
 min_D=min(min_D,D_next);a,B=a_next,B_next
assert max_param_error<1e-12 and max_forward_error<1e-12 and max_D_error<1e-12
final_kernel=[kernel_record(a,B,False),kernel_record(a,B,True)]
result={'seed':100925,'scope':'Actual full raw-parameter 1x1 Conv/ReLU/MaxPool twice plus free multiclass spatial head; no biases, fresh iid labels each update, adaptive scalar SGD; not standard RL-CIFAR Adam','batch':N,'classes':C,'input_feature_rank':4,'all_channels_overlap':True,'layers':L,'widths':[3,3,3],'initial_a':a_initial,'initial_B_squared_norm':float(B_initial.square().sum()),'limiting_a_squared_upper_bound_proved_in_companion_note':a_initial*a_initial-float(B_initial.square().sum()),'actual_autograd_steps':300,'final_a':a,'final_B_squared_norm':float(B.square().sum()),'minimum_balanced_gap_D':min_D,'max_full_parameter_invariance_error':max_param_error,'max_forward_error':max_forward_error,'max_exact_balance_identity_error':max_D_error,'initial_NTK_and_CE':initial_kernel,'after_300_updates_NTK_and_CE':final_kernel}
print(json.dumps(result,indent=2))
