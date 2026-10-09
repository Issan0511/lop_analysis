"""Independent exact-construction check; torch only for autodiff validation."""
import json
import math
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(100927)
P = 1 + torch.arange(1024).reshape(32, 32) / 1024
x = torch.tensor([1., 6.])[:, None, None, None] * P[None, None, :, :] / 12
x = x.repeat(1, 3, 1, 1)
c0 = 1585 / 1024
w1 = torch.zeros(16, 3, 5, 5)
w1[:, :, 2, 2] = 4
w2 = torch.zeros(16, 16, 5, 5)
w2[:, :, 2, 2] = 1 / 16
w3 = torch.full((100, 1024), 1 / (1024 * c0))
b3 = torch.cat([torch.zeros(50), torch.full((50,), -2.)])
w4 = torch.empty(100, 100)
w4[:50, :50] = 1/50
w4[:50, 50:] = .1/50
w4[50:, :50] = .1/50
w4[50:, 50:] = 1/50
b4 = torch.cat([torch.zeros(50), torch.full((50,), -.5)])
w5 = torch.full((10, 100), 1/100)
w5[0, 50:] += 6/50
w5[1, :50] += 3/50
b5 = torch.full((10,), -3.)
b5[:2] = 0
base = [w1, torch.zeros(16), w2, torch.zeros(16),
        w3, b3, w4, b4, w5, b5]

def forward(pp, detail=False):
    w1,b1,w2,b2,w3,b3,w4,b4,w5,b5 = pp
    z1 = F.conv2d(x,w1,b1,padding=2)
    a1 = F.max_pool2d(F.relu(z1),2)
    z2 = F.conv2d(a1,w2,b2,padding=2)
    a2 = F.max_pool2d(F.relu(z2),2).flatten(1)
    z3 = F.linear(a2,w3,b3)
    z4 = F.linear(F.relu(z3),w4,b4)
    f = F.linear(F.relu(z4),w5,b5)
    return (f,[z1,z2,z3,z4],a2.mean(1)/c0) if detail else f

def pool_gap(z):
    chunks=F.unfold(z,kernel_size=2,stride=2).reshape(2,z.shape[1],4,-1)
    sort=chunks.sort(dim=2,descending=True).values
    return (sort[:,:,0]-sort[:,:,1]).min().item()

def check(pp, name):
    pp=tuple(v.detach().clone().requires_grad_() for v in pp)
    f,z,h=forward(pp,True)
    m=z[0][:,0].mean()
    u0=torch.autograd.grad(m,pp,retain_graph=True,allow_unused=True)
    u=tuple(torch.zeros_like(v) if v0 is None else v0 for v,v0 in zip(pp,u0))
    _,r=torch.autograd.functional.jvp(lambda *v: forward(v),pp,u)
    _,rh=torch.autograd.functional.jvp(lambda *v: forward(v,True)[2],pp,u)
    p=f.softmax(1)
    loss=torch.logsumexp(f,1).mean()-f.mean()
    grad=torch.autograd.grad(loss,pp)
    direct=sum((gi*ui).sum() for gi,ui in zip(grad,u)).item()
    gn=((p-.1)*r).sum(1)
    pair=torch.zeros(2)
    for c in range(10):
        for d in range(c+1,10):
            pair+=(p[:,c]-p[:,d])*(r[:,c]-r[:,d])/10
    sample_records=[]
    for n in range(2):
        t=p[n].argmax().item()
        loser=torch.arange(10)!=t
        contrast=r[n,t]-r[n,loser]
        a,b=contrast.min(),contrast.max()
        B=(p[n,loser]-.1).clamp_min(0).sum()
        spread_bound=a*(p[n,t]-.1)-(b-a)*B
        sample_records.append({
            "top_class_1based":t+1,
            "minimum_top_logit_gap":(f[n,t]-f[n,loser]).min().item(),
            "minimum_top_R_gap":a.item(),
            "maximum_top_R_gap":b.item(),
            "spread_certificate":spread_bound.item(),
            "g":gn[n].item(),
            "g_per_rh":(gn[n]/rh[n]).item(),
            "r_h":rh[n].item(),
            "top_probability":p[n,t].item()
        })
        assert a>0 and spread_bound>0 and gn[n]>0
    assert abs(direct-gn.mean().item())<1e-11
    assert (gn-pair).abs().max()<1e-11
    assert min(pool_gap(z[0]),pool_gap(z[1]))>0
    assert min(v.abs().min().item() for v in z)>0
    assert (rh>=64/1585-1e-12).all() if name.endswith("exact") else True
    return {"name":name,"h":h.tolist(),"mean_gradient_projection":direct,
            "minimum_abs_gate_margin":min(v.abs().min().item() for v in z),
            "minimum_pool_gap":min(pool_gap(z[0]),pool_gap(z[1])),
            "all_conv_channels_active_on_both_images":bool((z[0]>0).all() and (z[1]>0).all()),
            "samples":sample_records}

records=[]
for perturb in [False,True]:
    pp=[v.clone() for v in base]
    if perturb:
        for v in pp:
            v.add_(1e-8*torch.rand_like(v))
    label="perturbed" if perturb else "exact"
    records.append(check(pp,"full_"+label))
    self_pp=[v.clone() for v in pp]
    self_pp[0]=self_pp[0][:1]
    self_pp[1]=self_pp[1][:1]
    self_pp[2]=self_pp[2][:,:1]
    records.append(check(self_pp,"self_"+label))

result={"architecture":"Actual RLCIFAR: Conv5 3->16 pad2; pool; Conv5 16->16 pad2; pool; FC1024->100->100->10, ReLU and all trainable biases",
        "input_range":[x.min().item(),x.max().item()],
        "c0":c0,"r_h_analytic_lower_bound":64/1585,
        "scope":"Constructed arbitrary-state open-family witness, not a measured training trajectory",
        "records":records}
print(json.dumps(result,indent=2))
