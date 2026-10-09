import math,json
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
B=16;C=10;p=.301;q=.3;eps=1e-8;eta=1e-4
s=math.log(p/(1-p)*(1-q)/q)
x=torch.zeros(B,3,2,2);x[:,0]=torch.tensor([[1.,.8],[.6,.4]])
w0=torch.zeros(2,3,1,1);w0[:,0]=1
b0=torch.full((2,),s/2-1)
head0=torch.full((2,C),-.3);head0[:,:3]=.7
weights=[math.comb(B,k)*q**k*(1-q)**(B-k) for k in range(B+1)]
out={}
for mode,t,vprev in [('fresh',1,0.),('history',3,1.)]:
 expected_dir=expected_raw=0.;max_formula_error=0.;min_margin=float('inf')
 for k in range(B+1):
  w=w0.clone().requires_grad_();b=b0.clone().requires_grad_();head=head0.clone().requires_grad_()
  z=F.conv2d(x,w,b);pooled=F.max_pool2d(F.relu(z),2).flatten(1);logits=pooled@head
  labels=torch.full((B,),3,dtype=torch.long);labels[:k]=0
  loss=F.cross_entropy(logits,labels)
  grad=torch.autograd.grad(loss,[w,b,head])
  g_theory=p-k/B
  max_formula_error=max(max_formula_error,abs(float(grad[0][0,0,0,0])-g_theory),abs(float(grad[1][0])-g_theory))
  beta1=.9;beta2=.999
  steps=[]
  for gg in grad:
   moment=(1-beta1)*gg;second=beta2*vprev+(1-beta2)*gg**2
   steps.append((moment/(1-beta1**t))/((second/(1-beta2**t)).sqrt()+eps))
  m0=float(z[:,0].mean().detach())
  zafter=F.conv2d(x,w.detach()-eta*steps[0],b.detach()-eta*steps[1])
  direction=-(float(zafter[:,0].mean())-m0)/eta
  expected_dir+=weights[k]*direction;expected_raw+=weights[k]*1.7*g_theory
  ordered=z.detach().flatten(2).sort(dim=2,descending=True).values
  min_margin=min(min_margin,float((ordered[:,:,0]-ordered[:,:,1]).min()))
 assert max_formula_error<1e-12
 out[mode]=dict(expected_Adam_mean_direction=expected_dir,expected_SGD_mean_direction=expected_raw,max_CE_gradient_formula_error=max_formula_error,winner_margin=min_margin)
assert out['fresh']['expected_Adam_mean_direction']<0<out['fresh']['expected_SGD_mean_direction']
assert out['history']['expected_Adam_mean_direction']>0
print(json.dumps(out,indent=2))
