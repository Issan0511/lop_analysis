import json, math
from pathlib import Path
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(100920)
N,C=2,10
x=.1+.9*torch.rand(N,3,32,32)
shapes=[(16,3,5,5),(16,16,5,5),(100,1024),(100,100),(10,100)]
p=[]
for shape in shapes[:-1]:
    fan=math.prod(shape[1:])
    p.extend([(.6+.4*torch.rand(shape)/1)/fan, .05+.02*torch.rand(shape[0])])
q=.5+.5*torch.rand(100)
a=torch.arange(10)-4.5
w=a[:,None]*q[None,:]/100+.0001*torch.rand(10,100)
b=.01*a+.0001*torch.rand(10)
p.extend([w,b])
p=tuple(v.requires_grad_() for v in p)

def forward(*pp, return_z=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5=pp
    z1=F.conv2d(x,w1,b1,padding=2); y1=F.max_pool2d(F.relu(z1),2)
    z2=F.conv2d(y1,w2,b2,padding=2); y2=F.max_pool2d(F.relu(z2),2).flatten(1)
    z3=F.linear(y2,w3,b3);y3=F.relu(z3)
    z4=F.linear(y3,w4,b4);y4=F.relu(z4)
    f=F.linear(y4,w5,b5)
    return (f,[z1,z2,z3,z4]) if return_z else f

f,z=forward(*p,return_z=True)
probs=f.softmax(1)
uniform_loss=torch.logsumexp(f,1).mean()-f.mean()
g=torch.autograd.grad(uniform_loss,p,retain_graph=True)
records=[]
for layer in range(4):
    m=z[layer][:,0].mean()
    u0=torch.autograd.grad(m,p,retain_graph=True,allow_unused=True)
    u=tuple(torch.zeros_like(v) if v0 is None else v0 for v,v0 in zip(p,u0))
    _,r=torch.autograd.functional.jvp(forward,p,u)
    direct=sum((ui*gi).sum() for ui,gi in zip(u,g)).item()
    formula=((probs-1/C)*r).sum().item()/N
    pair=sum(((probs[:,j]-probs[:,i])*(r[:,j]-r[:,i])).sum().item() for i in range(C) for j in range(i+1,C))/(N*C)
    centered_f=f-f.mean(1,keepdim=True);centered_r=r-r.mean(1,keepdim=True)
    min_order=(r[:,1:]-r[:,:-1]).min().item()
    record={'layer':layer+1,'mean':m.item(),'direction_norm':sum(v.square().sum() for v in u).sqrt().item(),'all_parameter_gradient_projection':direct,'new_uniform_CE_formula':formula,'pairwise_formula':pair,'minimum_adjacent_R_order_gap':min_order,'centered_logit_sensitivity_cosine':(centered_f.flatten()@centered_r.flatten()/(centered_f.norm()*centered_r.norm())).item(),'minimum_nonzero_hidden_direction_entry':min(v[v>0].min().item() for v in u[:-2] if (v>0).any())}
    assert direct>0 and min_order>0
    assert abs(direct-formula)<1e-10*max(1,abs(direct))
    assert abs(direct-pair)<1e-10*max(1,abs(direct))
    records.append(record)

def pool_gap(zz):
    v=F.unfold(zz,kernel_size=2,stride=2).reshape(N,16,4,-1).sort(dim=2,descending=True).values
    return (v[:,:,0]-v[:,:,1]).min().item()
result={'seed':100920,'architecture':'Conv5(3,16,pad2)-ReLU-MaxPool2-Conv5(16,16,pad2)-ReLU-MaxPool2-FC1024,100-ReLU-FC100,100-ReLU-FC100,10; all biases trainable','images':N,'input_range':[x.min().item(),x.max().item()],'all_hidden_weights_positive':all((v>0).all().item() for v in p[:-2:2]),'minimum_head_class_coordinate_order_gap':(p[-2][1:]-p[-2][:-1]).min().item(),'minimum_output_bias_order_gap':(p[-1][1:]-p[-1][:-1]).min().item(),'minimum_activation':min(v.min().item() for v in z),'minimum_pool_gap':min(pool_gap(z[0]),pool_gap(z[1])),'minimum_logit_order_gap':(f[:,1:]-f[:,:-1]).min().item(),'records':records,'scope':'Nonempty strict ordered-readout certificate in actual architecture, arbitrary specified state; not a training-trajectory claim'}
assert result['minimum_pool_gap']>0
print(json.dumps(result,indent=2))
