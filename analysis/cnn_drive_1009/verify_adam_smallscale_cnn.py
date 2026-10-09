import json,torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
B=16;rho=torch.tensor([[1.,.8],[.6,.4]]);x=torch.zeros(B,3,2,2);x[:8,0]=rho;x[8:,1]=rho
a=torch.arange(10)-4.5;q=torch.tensor([.2,.1]);records=[]
for s in [.1,.01,.001]:
 w=torch.zeros(2,3,1,1);w[0,0]=1+s;w[0,1]=1+.1*s;w[1,0]=1+.1*s;w[1,1]=1+s
 w.requires_grad_();b=torch.full((2,),-1.,requires_grad=True)
 head=(s*q[:,None]*a[None,:]).detach().requires_grad_();outbias=(s*s*.05*a).detach().requires_grad_()
 z=F.conv2d(x,w,b);h=F.max_pool2d(F.relu(z),2).flatten(1);logits=h@head+outbias
 expect=torch.tensor([[1.,.1]]*8+[[.1,1.]]*8)*s
 assert torch.allclose(h,expect,atol=1e-14,rtol=0)
 mean=z[:,0].mean();u=torch.autograd.grad(mean,[w,b],retain_graph=True)
 # Exact new-uniform-label CE expectation, holding the constructed parameters fixed.
 mean_loss=-F.log_softmax(logits,1).mean()
 gmean=torch.autograd.grad(mean_loss,[w,b],retain_graph=True)
 projection=sum((aa*bb).sum() for aa,bb in zip(u,gmean)).item()
 labels=torch.tensor(list(range(10))+list(range(6)))
 loss=F.cross_entropy(logits,labels);grad=torch.autograd.grad(loss,[w,b])
 Aa=(logits.softmax(1)*a).sum(1).detach()
 noise_R=-.2*float(a[labels[:8]].sum())/16
 noise_G=-.2*float(a[labels[8:]].sum())/16
 noise_b=-.2*float(a[labels].sum())/16
 mu_R=.2*float(Aa[:8].sum())/(16*s*s)
 mu_G=.2*float(Aa[8:].sum())/(16*s*s)
 mu_b=.2*float(Aa.sum())/(16*s*s)
 actual=torch.tensor([float(grad[0][0,0]),float(grad[0][0,1]),float(grad[1][0])])
 formula=torch.tensor([s*noise_R+s**3*mu_R,s*noise_G+s**3*mu_G,s*noise_b+s**3*mu_b])
 error=float((actual-formula).abs().max());assert error<1e-14 and projection>0
 ordered=z.detach().flatten(2).sort(dim=2,descending=True).values
 records.append(dict(scale=s,mean_preactivation=float(mean.detach()),predicted_mean=-.3+.385*s,mean_CE_gradient=projection,gradient_divided_by_s_cubed=projection/s**3,coordinate_mu=[mu_R,mu_G,mu_b],gradient_decomposition_error=error,minimum_winner_gap=float((ordered[:,:,0]-ordered[:,:,1]).min())))
assert abs(records[-1]['gradient_divided_by_s_cubed']-.4789125)<1e-6
print(json.dumps(dict(records=records,limit_mean_coefficient=.4789125),indent=2))
