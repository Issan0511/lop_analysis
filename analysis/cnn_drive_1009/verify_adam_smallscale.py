from fractions import Fraction as F
import json
beta1=F(9,10);beta2=F(999,1000);omega=(1-beta1)*(1-beta2)/(1-beta1*beta2)
q=F(3,5);K=q;variance=q*(1-q);third=q*(1-q)*(2*q-1)
C1lo=omega*(1-q)*(2*q-1)/2;C1hi=omega*q*(2*q-1)/2;C2=omega*third
eps=F(1,10**8);s=F(1,10**12)

def remainder(k,mu):
 delta=mu*s**(k-1);Q=K+abs(delta)
 return s*s*abs(delta)*(2*K+abs(delta))/(eps*eps)+s**3*(abs(delta)*variance*(1+2*omega)+abs(delta)**3)/eps**3+s**4*Q**4/eps**4

rows=[]
for k,mu in [(3,F(1)),(2,F(1)),(2,F(10000))]:
 R=remainder(k,mu)
 lower=mu*s**k/eps-C1hi*s*s/eps**2+C2*s**3/eps**3-R
 upper=mu*s**k/eps-C1lo*s*s/eps**2+C2*s**3/eps**3+R
 if mu==1:assert upper<0
 else:assert lower>0
 rows.append(dict(k=k,mu=mu,SGD_mean=mu*s**k,stationary_Adam_lower=lower,stationary_Adam_upper=upper,finite_remainder=R))
Ksym=F(9,2);musym=F(33,4);k=3;delta=musym*s**2
rel=s*(2*Ksym+abs(delta))/eps
symlo=musym*s**3/eps*(1-rel)
assert rel<1 and symlo>0
# Vanishing third moment is weaker than symmetry / C1=0.
noise=[(F(-2),F(1,7)),(F(-1),F(2,5)),(F(3,2),F(16,35))]
assert sum(prob for x,prob in noise)==1
assert sum(x*prob for x,prob in noise)==0
assert sum(x**3*prob for x,prob in noise)==0
signed_second=sum(x*abs(x)*prob for x,prob in noise)
assert signed_second==F(2,35)>0
print(json.dumps(dict(beta1=beta1,beta2=beta2,omega=omega,epsilon=eps,scale=s,noise_bound=K,epsilon_dominance_ratio=s*K/eps,skew_C1_lower=C1lo,skew_C1_upper=C1hi,skew_C2=C2,skew_cases=rows,symmetric_case=dict(k=3,mu=musym,noise_bound=Ksym,relative_error_bound=rel,stationary_Adam_lower=symlo),zero_third_moment_example=dict(noise=noise,E_x_abs_x=signed_second)),indent=2,default=float))
