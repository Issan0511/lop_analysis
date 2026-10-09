import itertools,json
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
x=torch.zeros(2,3,6,6);x[0,0,:5,:5]=1;x[1,1,:5,:5]=1
w0=torch.zeros(2,3,5,5);w0[0,0]=1/25;w0[0,1]=1/250;w0[1,0]=1/250;w0[1,1]=1/25
b0=torch.zeros(2);a0=torch.tensor([[[[1.]],[[.01]]],[[[.01]],[[1.]]]]);c0=torch.zeros(2)
base=[w0,b0,a0,c0];lam=.1;alpha=5e-6;beta=.2;T=100

def representation(p):
 w,b,a,c=p
 z=F.conv2d(x,w,b);p1=F.max_pool2d(F.relu(z),2)
 z2=F.conv2d(p1,a,c);h=F.relu(z2).flatten(1)
 return torch.cat([h,torch.ones(2,1)],1),z[:,0].mean(),z,z2

def data(p):
 H,m,z,z2=representation(p);u=torch.autograd.grad(m,p,retain_graph=True,allow_unused=True)
 u=[torch.zeros_like(t) if g is None else g for g,t in zip(u,p)]
 R=torch.zeros_like(H)
 for i in range(2):
  for j in range(2):
   J=torch.autograd.grad(H[i,j],p,retain_graph=True)
   R[i,j]=sum((a*b).sum() for a,b in zip(u,J))
 return H.detach(),R.detach(),min(float(z.detach().min()),float(z2.detach().min()))

p0=[v.clone().requires_grad_() for v in base];H0,R0,_=data(p0)
K0=H0@H0.T;P0=torch.linalg.inv(K0+lam*torch.eye(2));Q0=K0@P0@P0
M0=float(torch.sum(R0*(Q0@H0))/2)
D=Q0.diag();eps=float(torch.linalg.matrix_norm(Q0/torch.sqrt(D[:,None]*D[None,:])-torch.eye(2),2))
ref_lower=sum(float(((R0[:,j]*H0[:,j]*D).sum()-eps*(R0[:,j]*D.sqrt()).norm()*(H0[:,j]*D.sqrt()).norm())/2) for j in range(2))
h0=float(torch.linalg.matrix_norm(H0,2));r0=float(torch.linalg.matrix_norm(R0,2))
labels=list(itertools.product([-1.,1.],repeat=2))
records=[];B0=E0=dh=dr=dv=sres=0.;actualmean=0.
for yy in labels:
 y=torch.tensor(yy);V0=H0.T@P0@y
 B0=max(B0,float(V0.norm()));E0=max(E0,float((y-H0@V0).norm()))
 p=[v.clone().requires_grad_() for v in base];v=torch.zeros(3,requires_grad=True)
 pool_margin=float('inf')
 for _ in range(T):
  H,_,z_current,_=representation(p)
  ordered=z_current.detach().flatten(2).sort(dim=2,descending=True).values
  pool_margin=min(pool_margin,float((ordered[:,:,0]-ordered[:,:,1]).min()))
  loss=.5*((H@v-y)**2).sum()+lam/2*(v*v).sum()
  gg=torch.autograd.grad(loss,p+[v])
  p=[(a-alpha*g).detach().requires_grad_() for a,g in zip(p,gg[:-1])]
  v=(v-beta*gg[-1]).detach().requires_grad_()
 H,R,activationmargin=data(p)
 dh=max(dh,float(torch.linalg.matrix_norm(H-H0,2)));dr=max(dr,float(torch.linalg.matrix_norm(R-R0,2)))
 dv=max(dv,float((v.detach()-V0).norm()));res=(H.T@H+lam*torch.eye(3))@v.detach()-H.T@y
 sres=max(sres,float(res.norm()))
 g=float((R@v.detach())@(H@v.detach())/2);actualmean+=g/4
 displacement=[float((a.detach()-b).norm()) for a,b in zip(p,base)]
 records.append(dict(label=yy,mean_new_gradient=g,hidden_displacements=displacement,activation_margin=activationmargin,min_pool_margin_during_old_training=pool_margin))

# Common-y pointwise comparison: finite label dependence is retained.
e_stationarity=sres/lam+dh*E0/lam+dh*B0/(2*lam**.5)

def bound(e):
 deltaA=dr*h0+r0*dh+dr*dh
 return ((B0+e)**2*deltaA+e*(2*B0+e)*r0*h0)/2
out=dict(alpha_hidden=alpha,beta_head=beta,old_steps=T,reference_margin=M0,reference_spectral_lower_bound=ref_lower,actual_joint_mean_gradient=actualmean,deltaH=dh,deltaR=dr,deltaV=dv,head_stationarity_residual=sres,deltaV_bound_from_stationarity=e_stationarity,error_bound_direct=bound(dv),error_bound_from_stationarity=bound(e_stationarity),certified_margin=M0-bound(e_stationarity),certificate_using_spectral_reference=ref_lower-bound(e_stationarity),records=records)
assert out['certificate_using_spectral_reference']>0
assert abs(actualmean-M0)<=bound(dv)
assert all(min(r['hidden_displacements'][0],r['hidden_displacements'][2])>1e-10 for r in records)
print(json.dumps(out,indent=2))
