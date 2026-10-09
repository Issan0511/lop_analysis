from fractions import Fraction as F
from math import comb, sqrt, log
import json
B=16;C=10;q=F(3,10);p=F(301,1000);eps=F(1,10**8)
weights=[F(comb(B,k))*q**k*(1-q)**(B-k) for k in range(B+1)]
g=[p-F(k,B) for k in range(B+1)]
mean=sum(w*x for w,x in zip(weights,g));var=sum(w*(x-mean)**2 for w,x in zip(weights,g))
fresh=sum(w*x/(abs(x)+eps) for w,x in zip(weights,g))
sign=sum(w*(1 if x>0 else -1) for w,x in zip(weights,g))
assert mean==F(1,1000) and fresh<0
beta1=.9;beta2=.999;t=3;mold=0.;vold=1.
a=beta1*mold;b=1-beta1;c=beta2*vold;d=1-beta2
bc1=1-beta1**t;bc2=1-beta2**t;kappa=sqrt(bc2)/bc1;eps_t=float(eps)*sqrt(bc2)
h=lambda x: 1/(sqrt(c+d*x*x)+eps_t)
lo=float(min(g));hi=float(max(g));absmin=0 if lo<=0<=hi else min(abs(lo),abs(hi));absmax=max(abs(lo),abs(hi))
hmin=h(absmax);hmax=h(absmin);w0=(hmin+hmax)/2;rho=(hmax-hmin)/(hmax+hmin)
mu=float(mean);sigma=sqrt(float(var))
main=w0*(a+b*mu);error=rho*w0*(abs(a+b*mu)+b*sigma)
cert=kappa*(main-error)
exact=kappa*sum(float(w)*(a+b*float(x))*h(float(x)) for w,x in zip(weights,g))
assert exact>=cert>0
# Two historical gradients G,-beta1*G realize m_prev=0,v_prev=1 at t=2.
G=sqrt(1/((1-beta2)*(beta2+beta1*beta1)))
m1=(1-beta1)*G;m2=beta1*m1+(1-beta1)*(-beta1*G)
v1=(1-beta2)*G*G;v2=beta2*v1+(1-beta2)*(beta1*G)**2
assert abs(m2)<1e-12 and abs(v2-1)<1e-12
s=log(float(p/(1-p)*(1-q)/q))
print(json.dumps(dict(batch=B,classes=C,favored_classes=3,predicted_favored_mass=float(p),mean_raw_gradient=mu,variance=float(var),expected_fresh_Adam=fresh.numerator/fresh.denominator,expected_sign=float(sign),firstlayer_mean_direction_factor=1.7,fresh_Adam_mean_direction=1.7*float(fresh),history_v=vold,history_m=mold,history_time=2,relative_denominator_distortion=rho,certificate_history_Adam=cert,actual_history_Adam=exact,historical_gradients=[G,-beta1*G],two_channel_each_activation=s/2),indent=2))
