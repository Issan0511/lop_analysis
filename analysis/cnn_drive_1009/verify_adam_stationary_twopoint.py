import math,json
from fractions import Fraction as F
q=F(3,5);beta1=F(9,10);beta2=F(999,1000);eps=F(1,10**8);scale=F(1,16);eps_eff=eps/scale
omega=(1-beta1)*(1-beta2)/(1-beta1*beta2)
gamma=omega*(1-q)*(2*q-1)/(2*(q+eps_eff)**2)
delta=F(1,1000);r=1-q-delta;M=q+delta
L=1/(r+eps_eff)+M*M/(r*(r+eps_eff)**2)
mu=F(1,100000)
upper=-gamma+L*mu
assert 0<mu<delta and upper<0
print(json.dumps(dict(label_probability=q,predicted_probability=q+mu,batch=16,sensitive_examples=1,gradient_scale=scale,epsilon=eps,effective_epsilon=eps_eff,beta1=beta1,beta2=beta2,omega=omega,strict_negative_bound_at_zero_mean=-gamma,uniform_shift_Lipschitz_bound=L,positive_mean_shift=mu,stationary_Adam_upper_bound=upper,raw_SGD_mean=scale*mu,certified_positive_shift_interval=min(delta,gamma/L)),indent=2,default=float))
