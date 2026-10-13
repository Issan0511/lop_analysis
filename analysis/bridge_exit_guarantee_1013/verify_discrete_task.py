"""Exact finite-label task-end obstruction in a globally frozen-J model.

All assertions use rational SymPy arithmetic. The training model is f0+J u
with J fixed across ALL tasks; both-layer coordinates are included in u.
At task endpoints we evaluate the original nonlinear ReLU capacity kernel
at theta0+u. This does NOT assert nonlinear-network SGD invariance or absence
of exit inside a task. Each task reaches its least-squares minimum; J has
rank three, so not every five-label vector is fitted with zero residual.
"""
from itertools import product
import json
from pathlib import Path
import sympy as s

R = s.Rational
N,H,target = 5,3,2
Z0 = 2*s.Matrix([[-4,-4,-4],[-7,-4,-9],[2,4,1],[-8,-3,-9],[8,5,6]])
W0 = Z0.T
b0 = s.zeros(H,1)
v0 = s.ones(H,1)
c0 = s.Integer(-7)
theta0 = s.Matrix(list(W0)+list(b0)+list(v0)+[c0])
Theta = s.eye(N)+s.ones(N,N)

def unpack(theta):
    w=s.Matrix(H,N,list(theta[:15]))
    b=s.Matrix(theta[15:18])
    v=s.Matrix(theta[18:21])
    return w,b,v,theta[21]

def parts(theta):
    w,b,v,c=unpack(theta)
    z=w.T+s.ones(N,1)*b.T
    d=z.applyfunc(lambda x: 1 if x>0 else 0)
    f=z.applyfunc(lambda x: max(x,0))
    return z,d,f,v,c

def capacity_force(theta,alone=False):
    z,d,f,v,c=parts(theta)
    matrix=s.eye(N)+s.ones(N,N)
    for j in ([target] if alone else range(H)):
        dj,fj=d[:,j],f[:,j]
        matrix+=v[j]**2*(dj*dj.T).multiply_elementwise(Theta)+fj*fj.T
    q=R(6,5)*d[:,target]
    return s.factor((q.T*matrix.inv()*f[:,target])[0])

z,d,f,v,c=parts(theta0)
f0=f*v+s.ones(N,1)*c
assert list(f0)==[-7,-7,7,-7,31]
A0,S0=capacity_force(theta0),capacity_force(theta0,True)
assert A0==-R(27,1325) and S0==R(84,1075)

# Exact Jacobian of the original network at theta0. Coordinates are all
# hidden weights, hidden biases, readouts and output bias, in that order.
J=s.zeros(N,len(theta0))
for n in range(N):
    for j in range(H):
        J[n,j*N+n]=v[j]*d[n,j]
        J[n,15+j]=v[j]*d[n,j]
        J[n,18+j]=f[n,j]
    J[n,21]=1
K=J*J.T
B=J.T*K.pinv()
assert J.rank()==3
assert s.trace(K)==601
# For mean MSE, eta=1/601 gives eta*lambda_max(J^T J)/N <= 1/5 < 2.
# Ordinary fixed-step full-batch GD therefore converges to the displayed
# task minimum in every nonzero singular mode; no task-dependent reset is used.
stable_mean_mse_eta=R(1,601)
assert J*B*J==J
assert B*J*B==B
assert J*B==(J*B).T
assert B*J==(B*J).T
assert (s.eye(len(theta0))-B*J)*B==s.zeros(len(theta0),N)

rows=[]
for labels in product((-1,1),repeat=N):
    y=s.Matrix(labels)
    u=B*(y-f0)
    state=theta0+u
    zy,dy,fy,vy,cy=parts(state)
    assert dy==d
    assert min(abs(z) for z in zy)>0
    ay,sy=capacity_force(state),capacity_force(state,True)
    assert ay<0<sy
    residual=f0+J*u-y
    assert J.T*residual==s.zeros(len(theta0),1)
    assert (s.eye(len(theta0))-B*J)*u==s.zeros(len(theta0),1)
    rows.append(dict(labels=labels,A=str(ay),self=str(sy),
                     A_decimal=float(ay),self_decimal=float(sy),
                     least_squares_loss=str((residual.T*residual)[0]/(2*N)),
                     nonlinear_minus_linear_logits=[str(x) for x in fy*vy+s.ones(N,1)*cy-(f0+J*u)]))

# From any preceding endpoint, fully training the SAME global linear model
# on the next task gives B(y-f0)+(I-BJ)u_prev = B(y-f0). This exact invariant
# proves the endpoint claim for arbitrary infinite label sequences without
# enumerating sequence lengths or making an iid approximation.
assert all(s.Rational(row['A'])<0<s.Rational(row['self']) for row in rows)

report=dict(
    model="Globally frozen joint Jacobian; unregularized squared-error task training to its minimum",
    capacity="Original hard-ReLU two-layer kernel at lifted endpoint theta0+u; lambda=1",
    rank=int(J.rank()),label_vectors=len(rows),
    convergent_joint_GD=dict(loss="mean squared error / 2",eta=str(stable_mean_mse_eta),
                            kernel_trace="601",task_training="infinite-step convergence"),
    initial=dict(A=str(A0),self=str(S0),logits=list(map(str,f0))),
    exact_J=[[str(x) for x in J.row(n)] for n in range(N)],
    exact_endpoint_map_B=[[str(x) for x in B.row(n)] for n in range(B.rows)],
    all_endpoints_strictly_held=True,
    invariant="u_prev in range(J^T); u_next=B(y-f0)+(I-BJ)u_prev=B(y-f0)",
    ranges=dict(A_min=min(row['A_decimal'] for row in rows),A_max=max(row['A_decimal'] for row in rows),
                self_min=min(row['self_decimal'] for row in rows),self_max=max(row['self_decimal'] for row in rows)),
    scope=["Every task endpoint is genuinely held for every infinite sequence of the 32 finite labels.",
           "The capacity holding predicate is nonlinear and is evaluated exactly at each endpoint.",
           "The TRAINING map is globally frozen-linear, not repeated linearization or actual nonlinear SGD.",
           "Least-squares minima need not have zero residual because J has rank three, not five.",
           "No claim here excludes holding exit followed by return during a task.",
           "Hence a continuous Gaussian-support escape argument cannot automatically be transferred to this finite label alphabet."],
    rows=rows)
path=Path(__file__).with_name('discrete_task_checks.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('exact_J','exact_endpoint_map_B','rows')},indent=2))
