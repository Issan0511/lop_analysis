"""Finite exact-label-sum checks of CE/Adam signs in the RL-CIFAR architecture.

A constructed state certificate, not a claim about the RL-CIFAR trajectory.
"""
import json,math
import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(100921)
N,C,BATCH=2,10,16
x=.1+.9*torch.rand(N,3,32,32)
shapes=[(16,3,5,5),(16,16,5,5),(100,1024),(100,100)]
base=[]
for shape in shapes:
    fan=math.prod(shape[1:]);base.extend([(.6+.4*torch.rand(shape))/fan,.05+.02*torch.rand(shape[0])])
p=tuple(v.requires_grad_() for v in base)
q=(.5+.5*torch.rand(100))/100
EPS=1e-8

def hidden(*pp):
    w1,b1,w2,b2,w3,b3,w4,b4=pp
    z1=F.conv2d(x,w1,b1,padding=2);y1=F.max_pool2d(F.relu(z1),2)
    z2=F.conv2d(y1,w2,b2,padding=2);y2=F.max_pool2d(F.relu(z2),2).flatten(1)
    z3=F.linear(y2,w3,b3);z4=F.linear(F.relu(z3),w4,b4)
    return F.relu(z4),z1,[z1,z2,z3,z4]

h,z1,zs=hidden(*p);t=h@q+.01
# u is the actual augmented mean input patch for target channel 0.
m=z1[:,0].mean();u0=torch.autograd.grad(m,p,retain_graph=True,allow_unused=True)
u=torch.cat([u0[0][0].flatten(),u0[1][0:1]]).detach()
D=[]
for n in range(N):
    gg=torch.autograd.grad(t[n],p,retain_graph=True)
    D.append(torch.cat([gg[0][0].flatten(),gg[1][0:1]]).detach())
D=torch.stack(D)
assert (D>0).all() and (u>0).all()

def sum_counts(draws):
    counts=[1]
    for _ in range(draws):
        nxt=[0]*(len(counts)+9)
        for i,c in enumerate(counts):
            for k in range(C):nxt[i+k]+=c
        counts=nxt
    assert sum(counts)==10**draws
    return np.array(counts,dtype=np.int64)

# Symmetric ten-class coefficients: convolution compresses 10^16 label vectors
# to 73^2 = 5329 group-sum configurations without approximation.
a=torch.arange(C)-4.5
f=t[:,None]*a[None,:]
mean_a=(f.softmax(1)*a).sum(1).detach()
cnt=sum_counts(8)
prob=torch.tensor(np.outer(cnt,cnt).astype(np.float64)/1e16).flatten()
ss=torch.arange(len(cnt),dtype=torch.float64)-8*4.5
noise0=ss[:,None].expand(-1,len(cnt)).flatten()/16
noise1=ss[None,:].expand(len(cnt),-1).flatten()/16
mu=(D*mean_a[:,None]).mean(0)
G=mu[None,:]-noise0[:,None]*D[0]-noise1[:,None]*D[1]
Eadam=(prob[:,None]*(G/(G.abs()+EPS))).sum(0)
Esgd=(prob[:,None]*G).sum(0)
assert (Eadam>0).all()
assert torch.max((Esgd-mu).abs())<1e-14
# Compare actual torch.optim.Adam at several explicit 16-label configurations.
max_autograd_error=0.;max_adam_error=0.
for labels in ([0]*16,[9]*16,list(range(10))+list(range(6)),[0,9]*8):
    actual=[v.detach().clone().requires_grad_() for v in p]
    output_w=(a[:,None]*q[None,:]).detach().clone().requires_grad_()
    output_b=(.01*a).detach().clone().requires_grad_()
    opt=torch.optim.Adam(actual+[output_w,output_b],lr=.001,eps=EPS)
    hh,_,_=hidden(*actual);logits=F.linear(hh,output_w,output_b)
    expanded=logits.repeat_interleave(8,0)
    loss=F.cross_entropy(expanded,torch.tensor(labels));loss.backward()
    actual_g=torch.cat([actual[0].grad[0].flatten(),actual[1].grad[0:1]])
    expected_g=mu-D[0]*(sum(a[labels[:8]]).item()/16)-D[1]*(sum(a[labels[8:]]).item()/16)
    max_autograd_error=max(max_autograd_error,float((actual_g-expected_g).abs().max()))
    before=torch.cat([actual[0][0].detach().flatten(),actual[1][0:1].detach()])
    opt.step()
    after=torch.cat([actual[0][0].detach().flatten(),actual[1][0:1].detach()])
    direction=(before-after)/.001
    expected_direction=expected_g/(expected_g.abs()+EPS)
    max_adam_error=max(max_adam_error,float((direction-expected_direction).abs().max()))
assert max_autograd_error<1e-13 and max_adam_error<1e-10

# Asymmetric 3/7 class split: same full architecture, one image repeated 16x.
# Set head bias to make P(first 3 classes)=.301 exactly in mathematical logits.
fav=3;p_group=.301
logit_gap=math.log(p_group*(10-fav)/((1-p_group)*fav))
group_a=torch.tensor([.7]*3+[-.3]*7)
group_bias=logit_gap-float((h[0]@q).detach())
f_group=group_a*(float((h[0]@q).detach())+group_bias)
computed_group_probability=float(f_group.softmax(0)[:3].sum())
probs=torch.tensor([math.comb(16,k)*(.3**k)*(.7**(16-k)) for k in range(17)])
ks=torch.arange(17,dtype=torch.float64)
group_G=(p_group-ks[:,None]/16)*D[0][None,:]
group_Eadam=(probs[:,None]*group_G/(group_G.abs()+EPS)).sum(0)
group_Esgd=(probs[:,None]*group_G).sum(0)
# The statistic for this case averages only the single image's spatial patches.
ug=torch.autograd.grad(z1[0,0].mean(),p,allow_unused=True)
u_group=torch.cat([ug[0][0].flatten(),ug[1][0:1]]).detach()
assert (group_Esgd>0).all() and (group_Eadam<0).all()
assert abs(computed_group_probability-p_group)<1e-14

def pool_gap(z):
    vals=F.unfold(z,kernel_size=2,stride=2).reshape(N,16,4,-1).sort(dim=2,descending=True).values
    return float((vals[:,:,0]-vals[:,:,1]).min().detach())
minimum_pool_gap=min(pool_gap(zs[0]),pool_gap(zs[1]))
assert minimum_pool_gap>0
out={'seed':100921,'minimum_pool_gap':minimum_pool_gap,'actual_optimizer_check_scope':'Four symmetric-example label vectors; asymmetric full-network case checked by analytic coordinate formula','scope':'Exact finite label-sum expectations at constructed full CNN states; not observed training trajectories','architecture':'Conv5(3,16,pad2)-ReLU-MaxPool2-Conv5(16,16,pad2)-ReLU-MaxPool2-FC(1024,100)-ReLU-FC(100,100)-ReLU-FC(100,10); all biases trainable','batch':16,'classes':10,'adam':{'step':1,'beta1':.9,'beta2':.999,'epsilon':EPS,'previous_first_and_second_moments':0},'symmetric_example':{'different_images':2,'copies_per_image':8,'compressed_label_configurations':len(cnt)**2,'number_label_vectors':10**16,'probability_sum':float(prob.sum()),'minimum_target_coordinate_expected_adam_direction':float(Eadam.min()),'expected_target_mean_gradient_SGD':float(u@Esgd),'expected_target_mean_update_Adam_lr_001':float(-.001*(u@Eadam)),'max_autograd_gradient_error':max_autograd_error,'max_actual_adam_update_error':max_adam_error},'asymmetric_counterexample':{'favored_classes':3,'favored_probability':computed_group_probability,'expected_target_mean_gradient_SGD':float(u_group@group_Esgd),'expected_target_mean_update_Adam_lr_001':float(-.001*(u_group@group_Eadam)),'minimum_coordinate_expected_adam_direction':float(group_Eadam.min()),'maximum_coordinate_expected_adam_direction':float(group_Eadam.max()),'output_bias_group_coefficient':group_bias},'versions':{'torch':torch.__version__,'numpy':np.__version__}}
print(json.dumps(out,indent=2))
