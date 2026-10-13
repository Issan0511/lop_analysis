"""Exact counterexample to universal one-step label exit for the two-layer NTK.

No numerical tolerance is used in any mathematical assertion. The printout's
decimals are presentation only. Labels are all elements of {-1,+1}^5 and the
loss is (1/2N)*sum_n (f_n-y_n)^2. One full-batch SGD step trains both layers
and both biases, with eta=1/1000, no momentum or decay.
"""
from itertools import product
import json
from pathlib import Path
import sympy as s

N, H = 5, 3
R = s.Rational
Z = s.Matrix([[-4, -4, -4], [-7, -4, -9], [2, 4, 1],
              [-8, -3, -9], [8, 5, 6]])
v = s.ones(H, 1)
eta = R(1, 1000)
Th = s.eye(N) + s.ones(N, N)
kv = s.ones(N, 1) * R(6, 5)
lam = s.Integer(1)
target = 2

def parts(z):
    return z.applyfunc(lambda x: 1 if x > 0 else 0), z.applyfunc(lambda x: max(x, 0))

def force(z, readout, alone=False):
    d, f = parts(z)
    indices = [target] if alone else range(H)
    K = s.ones(N, N) + lam * s.eye(N)
    for j in indices:
        dj, fj = d[:, j], f[:, j]
        K += readout[j] ** 2 * (dj * dj.T).multiply_elementwise(Th) + fj * fj.T
    p = kv.multiply_elementwise(d[:, target])
    return s.factor((p.T * K.inv() * f[:, target])[0])

A0, S0 = force(Z, v), force(Z, v, True)
assert A0 == -R(36, 3377)
assert S0 == R(336, 2285)
Azero, Szero = force(Z, s.zeros(H, 1)), force(Z, s.zeros(H, 1), True)
assert Azero == -R(27, 640)
assert Szero == R(168, 895)
D, F = parts(Z)
pred = F * v  # output bias starts at zero
rows = []
for label in product((-1, 1), repeat=N):
    y = s.Matrix(label)
    r = pred - y
    # X=I: z'_{nj}=z_{nj}-eta/N*v_j*(d_nj*r_n+sum_m d_mj*r_m).
    z1 = Z.copy()
    v1 = v.copy()
    for j in range(H):
        r_gate = r.multiply_elementwise(D[:, j])
        z1[:, j] -= eta / N * v[j] * (r_gate + s.ones(N, 1) * sum(r_gate))
        v1[j] -= eta / N * (F[:, j].T * r)[0]
    out_bias1 = -eta / N * sum(r)
    # Capacity is independent of the value of output bias but retains its 11^T kernel.
    assert parts(z1)[0] == D
    A1, S1 = force(z1, v1), force(z1, v1, True)
    assert A1 < 0 < S1
    rows.append(dict(label=label, A=str(A1), self=str(S1),
                     A_decimal=float(A1), self_decimal=float(S1),
                     output_bias=str(out_bias1)))

report = dict(N=N, H=H, activation="ReLU", lambda_=str(lam), eta=str(eta),
              input="X=I_5; both biases trained", target_zero_based=target,
              Z=[[str(x) for x in Z.row(i)] for i in range(N)],
              v=[str(x) for x in v], A_initial=str(A0), self_initial=str(S0),
              zero_readout_variant=dict(A=str(Azero), self=str(Szero),
                                       output_J_times_grad_A="exactly zero"),
              all_label_count=len(rows), rows=rows)
path = Path(__file__).with_name("exact_counterexample.json")
path.write_text(json.dumps(report, indent=2))
print("Initial:", A0, S0)
print("Zero-readout variant:", Azero, Szero, "; J grad(A) = 0 exactly")
print("All", len(rows), "labels remain strictly held after eta=", eta)
print("A range:", min(x['A_decimal'] for x in rows), max(x['A_decimal'] for x in rows))
print("Self range:", min(x['self_decimal'] for x in rows), max(x['self_decimal'] for x in rows))
print("Exact rational certificate:", path)
