import itertools
import json
import math

def transpose(a):
    return list(map(list, zip(*a)))

def matmul(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in transpose(b)] for row in a]

def softmax(row):
    exponentials = [math.exp(x-max(row)) for x in row]
    return [x/sum(exponentials) for x in exponentials]

H=[[1.,2.],[2.,1.]]
D=[[9.,0.],[1.5,0.]]
N=C=2
h=3.
d=math.sqrt(83.25)
cy=math.sqrt(.5)
B=h*cy/math.sqrt(N)
L=h*h/(2*N)
Q,R=138.,9.
eta,T=.001,18
S=eta*T
Sstar=min(N/R,2*N/h**2,8*Q/(5*math.e*C*d*h**5))
margin=cy*cy*Q*S*S*math.exp(-S*R/N)/(N**3*C)
eps_fit=L*B*S*S/2
M=B*S
eps_H=eps_D=1e-4
eps_hidden=eps_H*cy*S/math.sqrt(N)+eps_H*(2*h+eps_H)*B*S*S/(4*N)
eps_total=eps_fit+eps_hidden
pert_bound=(eps_D*(h+eps_H)*(M+eps_total)**2
            +d*(h+eps_H)*eps_total*(2*M+eps_total)
            +d*eps_H*M*M)/(2*N)

def fnorm(x):
    return math.sqrt(sum(v*v for row in x for v in row))

def delta(a,b):
    return [[x-y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]

def feature_at(k,labels,perturbed):
    e=eps_H*math.sin(k+sum(labels)) if perturbed else 0.
    return [[H[i][j]+(e if i==j else 0.) for j in range(2)] for i in range(2)]

def run(labels,perturbed):
    V=[[0.]*C for _ in range(2)]
    for k in range(T):
        Hk=feature_at(k,labels,perturbed)
        P=[softmax(row) for row in matmul(Hk,V)]
        residual=[[P[n][c]-float(labels[n]==c) for c in range(C)] for n in range(N)]
        grad=matmul(transpose(Hk),residual)
        V=[[V[a][c]-eta*grad[a][c]/N for c in range(C)] for a in range(2)]
    return V

def switching_gradient(Hx,Dx,V):
    P=[softmax(row) for row in matmul(Hx,V)]
    Dv=matmul(Dx,V)
    return sum(Dv[n][c]*(P[n][c]-1/C) for n in range(N) for c in range(C))/N

results=[]
for labels in itertools.product(range(C),repeat=N):
    Y=[[float(labels[n]==c)-1/C for c in range(C)] for n in range(N)]
    Vlin=[[S*x/N for x in row] for row in matmul(transpose(H),Y)]
    V=run(labels,False)
    Vp=run(labels,True)
    Hcurrent=feature_at(T,labels,True)
    Dcurrent=[[9.,-eps_D],[1.5,0.]]
    fiterr=fnorm(delta(V,Vlin))
    hiddenerr=fnorm(delta(Vp,V))
    assert fiterr<=eps_fit+1e-14
    assert hiddenerr<=eps_hidden+1e-14
    gp=switching_gradient(Hcurrent,Dcurrent,Vp)
    gl=switching_gradient(H,D,Vlin)
    assert abs(gp-gl)<=pert_bound+1e-14
    results.append({'labels':labels,'g_frozen':switching_gradient(H,D,V),
                    'g_perturbed':gp,'g_linear':gl,
                    'fit_error':fiterr,'hidden_error':hiddenerr})
g=sum(x['g_frozen'] for x in results)/len(results)
gp=sum(x['g_perturbed'] for x in results)/len(results)
assert S<=Sstar
assert g>=margin/2
assert gp>=margin-pert_bound>0
print(json.dumps({'S':S,'S_star':Sstar,'reference_margin_lower':margin,
                  'fit_error_bound':eps_fit,'hidden_error_bound':eps_hidden,
                  'perturbation_gradient_error_bound':pert_bound,
                  'expected_frozen_gradient':g,'expected_perturbed_gradient':gp,
                  'per_label':results},indent=2))
