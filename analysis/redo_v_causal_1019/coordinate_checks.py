"""Independent double-precision chain-rule checks for proposed coordinates."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
g=torch.Generator().manual_seed(1019)
h=torch.randn(23,7,generator=g,dtype=torch.float64)
W=torch.randn(5,7,generator=g,dtype=torch.float64)
b=torch.randn(5,generator=g,dtype=torch.float64)
V=torch.randn(3,5,generator=g,dtype=torch.float64)
y=torch.randint(3,(23,),generator=g)

def evaluate(w,bias,v,mu=None):
    ps=[x.clone().requires_grad_() for x in (w,bias,v)]
    z=h@ps[0].T+ps[1]
    if mu is not None:z=z-(ps[0]*mu).sum(1)
    f=torch.relu(z)@ps[2].T
    grads=torch.autograd.grad(F.cross_entropy(f,y),ps)
    return f.detach(),grads

f0,g0=evaluate(W,b,V)
rows=[]
for c in (.25,4.):
    scale=torch.tensor([c,1,c,1,c],dtype=torch.float64)
    f,grad=evaluate(W/scale[:,None],b/scale,V*scale[None,:])
    expected=(g0[0]*scale[:,None],g0[1]*scale,g0[2]/scale[None,:])
    rows.append(dict(test='gauge',c=c,logit_max_error=float((f-f0).abs().max()),gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(grad,expected))))
mu=torch.randn(5,7,generator=g,dtype=torch.float64)
f,grad=evaluate(W,b+(W*mu).sum(1),V,mu)
expected=(g0[0]-g0[1][:,None]*mu,g0[1],g0[2])
rows.append(dict(test='centering',logit_max_error=float((f-f0).abs().max()),gradient_max_error=max(float((a-b).abs().max()) for a,b in zip(grad,expected))))
assert max(r['logit_max_error'] for r in rows)<1e-12
assert max(r['gradient_max_error'] for r in rows)<1e-12
root=Path(__file__).resolve().parents[2]
p=root/'results/redo_v_causal_1019/coordinate_checks.json'
p.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
