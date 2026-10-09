"""Independent rational recurrence audit; not the primary certificate."""
from fractions import Fraction as F
from functools import lru_cache
from math import comb
from decimal import Decimal as D, localcontext
import json

a,b=F(9,10),F(999,1000)
values=[F(900001,1000000)-F(k,16) for k in range(17)]
probs=[F(comb(16,k)*9**k,10**16) for k in range(17)]
assert sum(probs)==1
mu=sum(p*g for p,g in zip(probs,values))
r=sum(p*g*g for p,g in zip(probs,values))
zmin=min(g*g for g in values)
gmax=max(abs(g) for g in values)

@lru_cache(None)
def noise(p,q):
    return sum(prob*g**p*(g*g-r)**q for prob,g in zip(probs,values))

@lru_cache(None)
def moment(p,q):
    if p==0 and q==0:
        return F(1)
    total=F(0)
    for i in range(p+1):
        for j in range(q+1):
            if i==p and j==q:
                continue
            total+=comb(p,i)*comb(q,j)*a**i*(1-a)**(p-i)*b**j*(1-b)**(q-j)*moment(i,j)*noise(p-i,q-j)
    return total/(1-a**p*b**q)

def dec(x):
    return D(x.numerator)/D(x.denominator)

assert moment(1,0)==mu
assert moment(0,1)==0
assert moment(2,0)==mu*mu+(r-mu*mu)*(1-a)/(1+a)
# Exact rational upper bounds, independent of Decimal sqrt or isqrt code.
# T2=A/sqrt(r), and A<0. Square only after establishing the sign.
A=mu-moment(1,1)/(2*r)+3*moment(1,2)/(8*r*r)
center_bound=F(-3150831,10**10)
good_bound=F(391610,10**10)
bad_bound=F(40891,10**10)
eps_bound=F(2753,10**10)
assert A<0 and A*A>center_bound*center_bound*r
assert F(25,256)*moment(2,0)*moment(0,6)/(F(7,10)*r)**7<good_bound**2
assert r>F(3,40)**2 and zmin==F(25001,1000000)**2
bad_prob=moment(0,12)/(F(3,10)*r)**12
assert gmax*bad_prob*(F(1000000,25001)+25)<bad_bound
assert F(1,10**8)**2*moment(2,0)/zmin**2<eps_bound**2
exact_upper=center_bound+good_bound+bad_bound+eps_bound
assert exact_upper==F(-2715577,10**10) and exact_upper<0
with localcontext() as ctx:
    ctx.prec=80
    rd,z=dec(r),dec(zmin)
    center=dec(mu)/rd.sqrt()-dec(moment(1,1))/(2*rd*rd.sqrt())+3*dec(moment(1,2))/(8*rd*rd*rd.sqrt())
    remainder=D(5)/16*(dec(moment(2,0)*moment(0,6))/(D('.7')*rd)**7).sqrt()
    prob_bad=dec(moment(0,12))/(D('.3')*rd)**12
    bad=dec(gmax)*(1/z.sqrt()+D(15)/(8*rd.sqrt()))*prob_bad
    eps=D('1e-8')*dec(moment(2,0)).sqrt()/z
    result={"exact_rational_certified_upper":str(exact_upper),
            "exact_decimal_certified_upper":str(dec(exact_upper)),
            "mu":str(dec(mu)),"r":str(rd),"zmin":str(z),"gmax":str(dec(gmax)),
            "Em2":str(dec(moment(2,0))),"EmDelta":str(dec(moment(1,1))),
            "EmDelta2":str(dec(moment(1,2))),
            "EDelta6":str(dec(moment(0,6))),"EDelta12":str(dec(moment(0,12))),
            "center":str(center),"good_remainder":str(remainder),
            "prob_bad":str(prob_bad),"bad_error":str(bad),"eps_error":str(eps),
            "expectation_upper":str(center+remainder+bad+eps)}
print(json.dumps(result,indent=2))
