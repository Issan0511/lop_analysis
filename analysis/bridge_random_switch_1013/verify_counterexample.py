"""Exact high-confidence CE counterexample to an automatic outward mean drift.

Run with Python + SymPy. This script is standalone. It certifies a local
directional expectation, not permanent holding, non-reachability, or a
previous-task stationary point. Floating-point values are presentation only;
every sign and identity assertion is an exact rational calculation.
"""
import json
from pathlib import Path
import sympy as s

R = s.Rational
N, H, target = 5, 3, 2
Z = 2 * s.Matrix([[-4,-4,-4],[-7,-4,-9],[2,4,1],[-8,-3,-9],[8,5,6]])
v = s.ones(H,1)
output_bias = s.Integer(-7)
old_labels = s.Matrix([0,0,1,0,1])
F = s.Matrix([[4,8,2],[16,10,12]])  # The two active input rows (3 and 5).
B = s.Matrix([[2,1],[1,2]])       # Their augmented-input Gram matrix.
q = s.ones(2,1) * R(6,5)
base = s.eye(2) + s.ones(2,2)/4  # lambda=1, after eliminating inactive inputs.

def value(f, readout, alone=False):
    matrix = base.copy()
    for j in ([target] if alone else range(H)):
        matrix += readout[j]**2*B + f[:,j]*f[:,j].T
    return s.factor((q.T*matrix.inv()*f[:,target])[0])

def ratio(f, readout):
    return s.factor(value(f,readout)/value(f,readout,True))

A, S, chi = value(F,v), value(F,v,True), ratio(F,v)
assert A == -R(27,1325) < 0
assert S == R(84,1075) > 0
assert chi == -R(387,1484)

# Independently check the active-input Schur reduction against the full kernel.
D = Z.applyfunc(lambda x: 1 if x > 0 else 0)
Fall = Z.applyfunc(lambda x: max(x,0))
Theta = s.eye(N) + s.ones(N,N)
for alone, expected in ((False,A),(True,S)):
    K = s.eye(N) + s.ones(N,N)
    for j in ([target] if alone else range(H)):
        dj,fj = D[:,j],Fall[:,j]
        K += v[j]**2*(dj*dj.T).multiply_elementwise(Theta) + fj*fj.T
    qfull = R(6,5)*D[:,target]
    assert (qfull.T*K.inv()*Fall[:,target])[0] == expected

# Calculate b_n = J_n grad(chi) by an independent exact parameter perturbation
# in the direction J_n^T. All active gates equal one, and both biases train.
# The hidden-bias component is included in B, the output-bias value does not
# enter capacity, and inactive input Jacobian rows cannot affect chi.
t = s.symbols('t')
bchi, bA = [], []
for n in range(2):
    ft = F + t*B[:,n]*v.T
    vt = v + t*F[n,:].T
    bchi.append(s.factor(s.diff(ratio(ft,vt),t).subs(t,0)))
    bA.append(s.factor(s.diff(value(ft,vt),t).subs(t,0)))
assert bchi == [R(75779237,88090240),R(113881973,88090240)]
assert bA == [R(819627,11236000),R(342111,2247200)]
assert all(x > 0 for x in bchi+bA)

logits = Fall*v + s.ones(N,1)*output_bias
assert list(logits) == [-7,-7,7,-7,31]
assert all((2*old_labels[n]-1)*logits[n] >= 7 for n in range(N))

# e^7 > sum_{k=0}^{11} 7^k/k! > 999 certifies every old correct probability
# exceeds 999/1000. This confidence bound uses no transcendental approximation.
exp7_lower = sum(s.Integer(7)**k/s.factorial(k) for k in range(12))
assert exp7_lower > 999

# Fresh labels have marginal mean 1/2. Cross-input independence is not needed
# for this first-order expectation. With L=mean CE and p=sigmoid(logit),
# E[D chi] = -(1/N) sum_n (p_n-1/2) b_n.
# Only active inputs 3 and 5 have nonzero b_n; their logits are 7 and 31.
pactive = [1/(1+s.exp(-7)),1/(1+s.exp(-31))]
EDchi = -sum((p-R(1,2))*b for p,b in zip(pactive,bchi))/N
EDA = -sum((p-R(1,2))*b for p,b in zip(pactive,bA))/N
strict_upper_chi = -R(499,5000)*sum(bchi)
strict_upper_A = -R(499,5000)*sum(bA)
assert strict_upper_chi < 0 and strict_upper_A < 0

report = dict(
    model=dict(inputs="x_n=e_n in R^5",activation="ReLU",lambda_="1",
               both_layers_and_biases_trained=True,loss="mean binary CE",
               Z=[[str(x) for x in Z.row(n)] for n in range(N)],
               readout=[str(x) for x in v],output_bias=str(output_bias),
               old_labels=list(map(int,old_labels)),logits=list(map(int,logits))),
    exact=dict(A=str(A),self=str(S),chi=str(chi),
               J_grad_chi_active=list(map(str,bchi)),J_grad_A_active=list(map(str,bA)),
               minimum_old_correct_probability_strict_lower="999/1000",
               exp7_rational_lower=str(exp7_lower),
               expected_direction_chi_strict_upper=str(strict_upper_chi),
               expected_direction_A_strict_upper=str(strict_upper_A)),
    presentation_only=dict(minimum_old_correct_probability=float(pactive[0].evalf()),
                           expected_direction_chi=float(EDchi.evalf()),
                           expected_direction_A=float(EDA.evalf())),
    scope=["True strict hold, with open fraction 2/5 and old accuracy 100%.",
           "Local fresh-label expected derivative points farther into A<0; chi decreases too.",
           "By smoothness this gives negative expected changes for sufficiently small positive steps.",
           "Not asserted to be previous-task stationary or an exact MSE/minimum-norm fit.",
           "Does not disprove relative-pattern noise, finite-word exit, or eventual exit."])
path = Path(__file__).with_name('counterexample_checks.json')
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
