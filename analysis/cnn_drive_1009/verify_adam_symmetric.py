import math,json

def coeff_counts(B):
 counts=[1]
 for _ in range(B):
  nxt=[0]*(len(counts)+9)
  for i,c in enumerate(counts):
   for k in range(10):nxt[i+k]+=c
  counts=nxt
 assert sum(counts)==10**B
 return counts

def mean_a(t):
 a=[k-4.5 for k in range(10)];p=[math.exp(t*z) for z in a];Z=sum(p)
 return sum(z*v for z,v in zip(a,p))/Z
mu_r=mean_a(.26);mu_g=mean_a(.17)
eps=1e-8
specs=[('w_red',8,.1,mu_r,.35),('w_green',8,.1,mu_g,.35),('bias',16,.2,(mu_r+mu_g)/2,1.)]
records=[];direction=0
for name,B,scale,mu,u in specs:
 counts=coeff_counts(B);expected=0
 for i,count in enumerate(counts):
  noise=(i-4.5*B)/B;g=scale*(mu-noise)
  expected+=count/(10**B)*g/(abs(g)+eps)
 assert expected>0
 direction+=u*expected
 records.append(dict(coordinate=name,labels=B,mean_gradient=scale*mu,expected_fresh_Adam=expected,mean_direction_weight=u))
print(json.dumps(dict(classes=10,batch=16,features_red=[1.,.1],features_green=[.1,1.],class_coefficients=[k-4.5 for k in range(10)],q=[.2,.1],class_bias_scalar=.05,mean_a_red=mu_r,mean_a_green=mu_g,coordinates=records,expected_Adam_mean_direction=direction),indent=2))
