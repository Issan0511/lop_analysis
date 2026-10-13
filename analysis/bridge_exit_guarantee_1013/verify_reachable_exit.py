"""Exact exit witness in the frozen training-Jacobian reachable space.

Capacity is evaluated from the nonlinear ReLU parameters at BOTH points.
This is an algebra certificate, not a simulation of eventual exit.
"""
import json
from pathlib import Path
import sympy as s

R = s.Rational
N, H, target = 5, 3, 2
z0 = 2*s.Matrix([[-4,-4,-4],[-7,-4,-9],[2,4,1],[-8,-3,-9],[8,5,6]])
v0 = s.ones(H, 1)
x = s.eye(N)
theta_gram = s.eye(N)+s.ones(N, N)
xbar = s.Matrix([R(1,N)]*N+[1])


def pack(w, b, v, c):
    return s.Matrix(list(w)+list(b)+list(v)+[c])


def unpack(theta):
    return (s.Matrix(H,N,list(theta[:H*N])),
            theta[H*N:H*N+H, :],
            theta[H*N+H:H*N+2*H, :],theta[-1])


def features(theta):
    w,b,v,c = unpack(theta)
    z = w.T+s.ones(N,1)*b.T
    h = z.applyfunc(lambda a: max(a,0))
    d = z.applyfunc(lambda a: 1 if a>0 else 0)
    return z,h,d,v,c


def jacobian(theta):
    z,h,d,v,c = features(theta)
    jac = s.zeros(N, H*N+2*H+1)
    for n in range(N):
        for j in range(H):
            jac[n,j*N+n] = d[n,j]*v[j]
            jac[n,H*N+j] = d[n,j]*v[j]
            jac[n,H*N+H+j] = h[n,j]
        jac[n,-1] = 1
    return jac


def push(theta, alone=False):
    z,h,d,v,c = features(theta)
    kernel = s.eye(N)+s.ones(N,N)  # lambda_cap=1 plus output-bias kernel.
    for j in ([target] if alone else range(H)):
        kernel += v[j]**2*(d[:,j]*d[:,j].T).multiply_elementwise(theta_gram)
        kernel += h[:,j]*h[:,j].T
    q = R(6,5)*d[:,target]
    return s.factor((q.T*kernel.inv()*h[:,target])[0])


initial = pack(z0.T,s.zeros(H,1),v0,-7)
j0 = jacobian(initial)
direction = j0.T*s.ones(N,1)
witness = initial+direction
a0,s0 = push(initial),push(initial,True)
a1,s1 = push(witness),push(witness,True)
assert a0 == -R(27,1325) < 0
assert s0 == R(84,1075) > 0
assert a1>0 and s1>0
z1,h1,d1,v1,c1 = features(witness)
assert features(initial)[2] == d1
assert min(abs(a) for a in z1)>0
assert j0.rank()==3
# This gives an explicit stable step size for every finite task length T>=1,
# without estimating a largest eigenvalue numerically.
k = j0*j0.T
alpha = 1/(s.trace(k)+1)
assert alpha>0 and alpha*s.trace(k)<1

report = {
    'kind':'exact rational certificate, frozen training J; nonlinear capacity recomputed',
    'initial_A':str(a0),'initial_self':str(s0),
    'witness_parameter_displacement':'J0.T @ ones(5)',
    'witness_A':str(a1),'witness_self':str(s1),
    'witness_A_decimal':float(a1),'witness_self_decimal':float(s1),
    'witness_Z':[[str(a) for a in z1.row(n)] for n in range(N)],
    'witness_readout':[str(a) for a in v1],
    'witness_output_bias':str(c1),
    'gate_chamber_unchanged':True,
    'minimum_gate_margin':str(min(abs(a) for a in z1)),
    'reachable_dimension':j0.rank(),
    'training_ridge':'0',
    'capacity_ridge':'1',
    'explicit_stable_step':str(alpha),
    'task_length':'any fixed positive integer T',
    'label_assumption':'iid Gaussian vectors, held fixed within each task',
    'scope':'Certifies the geometric hypothesis of the a.s. exit theorem; not real CE/Adam training.'
}
Path(__file__).with_name('reachable_exit_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
