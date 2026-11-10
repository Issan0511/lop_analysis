#!/usr/bin/env python3
"""Reviewer's independent CPU checks of arm B (bwdswap_1111).  Writes review_cpu.json next to this file.

A  element level, three activations (F.elu, RM's expm1 ELU, 5+1's clamp ReLU), special and random z, d:
   bwd_only value == phi(z) bit for bit; d/dz == autograd of phi at z + d (upstream w) bit for bit;
   the brief's plain formula phi(z).detach() + (phi(z+d) - sg) for comparison (signed zeros).
B  RM, check seed 100, 2-epoch prefix to task 10, both directions, 2 epochs of training (150 updates):
   (i)  B (src rm_bwd_forward) vs an independent construction (custom autograd.Function: forward phi(z),
        backward = re-entrant autograd of phi at z + d): state sha256 and per-update CE/acc bit for bit;
   (ii) B with d = 0 vs the natural forward: bit for bit (no extra rounding anywhere in B);
   (iii) the main-run path bw.rm_arm('B') (RM.run_arm with make_forward swapped) vs RM.run_arm with the
        independent forward: rows' state sha256 / online for k = 1, 2.
No B accuracy numbers are printed (pre-registration): only equalities.
"""
import json
import sys
from pathlib import Path

ROOT = Path('/home/issan/Projects/claude/wt/bwdswap_1111')
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from src import bwdswap_1111 as bw

RM, F51 = bw.RM, bw.F51
OUT = Path(__file__).resolve().parent / 'review_cpu.json'


def ibits(t):
    t = t.detach().contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int64)


def nbits_diff(a, b):
    return int((ibits(a) != ibits(b)).sum())


def autograd_phi_prime(phi, q, w):
    q = q.detach().clone().requires_grad_(True)
    return torch.autograd.grad((phi(q) * w).sum(), q)[0]


def part_a():
    torch.manual_seed(0)
    sp = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 1e-7, -1e-7, 1.0, -1.0, 5.0, -5.0, -16.6, -16.7,
                       -17.0, -88.0, -104.0, -200.0, 30.0, 100.0], dtype=torch.float32)
    zs, ds = torch.meshgrid(sp, sp, indexing='ij')
    zs, ds = zs.flatten(), ds.flatten()
    zr = torch.randn(20000) * 3
    dr = torch.randn(20000) * 5
    z_cancel = torch.randn(2000) * 4
    d_cancel = -z_cancel                                   # z + d == 0 exactly (incl. -0 cases)
    d_near = torch.nextafter(-z_cancel, torch.full_like(z_cancel, 1e9))
    z = torch.cat([zs, zr, z_cancel, z_cancel]).contiguous()
    d = torch.cat([ds, dr, d_cancel, d_near]).contiguous()
    w = torch.randn_like(z)
    w[::7] = 0.0
    w[1::7] = -0.0
    res = {}
    for name, phi in (('F.elu', lambda q: F.elu(q)), ('RM_expm1_elu', RM.ACT.phi), ('F51_clamp_relu', F51.phi)):
        zz = z.clone().requires_grad_(True)
        a = bw.bwd_only(phi, zz, d)
        nat = phi(z)
        g_b = torch.autograd.grad((a * w).sum(), zz)[0]
        g_ref = autograd_phi_prime(phi, z + d, w)
        g_nat = autograd_phi_prime(phi, z, w)
        zz2 = z.clone().requires_grad_(True)
        sh = phi(zz2 + d)
        naive = phi(zz2).detach() + (sh - sh.detach())
        g_naive = torch.autograd.grad((naive * w).sum(), zz2)[0]
        res[name] = dict(
            n=int(z.numel()),
            value_bits_differ=nbits_diff(a, nat),
            grad_bits_differ_vs_autograd_at_z_plus_d=nbits_diff(g_b, g_ref),
            grad_values_differ_vs_autograd_at_z_plus_d=int((g_b != g_ref).sum()),
            grad_equals_natural_pairs_where_derivs_differ=int(((g_b == g_nat) & (g_ref != g_nat)).sum()),
            pairs_where_phi_prime_shift_differs=int((g_ref != g_nat).sum()),
            naive_value_bits_differ=nbits_diff(naive, nat),
            naive_value_values_differ=int((naive != nat).sum()),
            naive_grad_bits_differ=nbits_diff(g_naive, g_ref))
    # NaN field (5+1 table outside the task): must stay fail closed (NaN value), not silently natural
    zn = torch.randn(10)
    dn = torch.full((10,), float('nan'))
    res['nan_field_fail_closed'] = {nm: bool(torch.isnan(bw.bwd_only(phi, zn, dn)).all())
                                    for nm, phi in (('F.elu', F.elu), ('RM', RM.ACT.phi), ('F51', F51.phi))}
    return res


class IndepSwap(torch.autograd.Function):
    """Independent construction: value phi(z) (computed inside forward), backward = autograd of phi at z + d."""

    @staticmethod
    def forward(ctx, z, d):
        ctx.save_for_backward(z, d)
        return RM.ACT.phi(z)

    @staticmethod
    def backward(ctx, g):
        z, d = ctx.saved_tensors
        with torch.enable_grad():
            q = (z + d).detach().requires_grad_(True)
            gz = torch.autograd.grad(RM.ACT.phi(q), q, g)[0]
        return gz, None


def indep_forward(sh):
    d2 = sh[2]

    def fwd(params, xb, ob):
        W1, b1, W2, b2, W3, b3 = params
        z1 = xb @ W1.T + b1
        a1 = RM.ACT.phi(z1)
        z2 = a1 @ W2.T + b2
        a2 = IndepSwap.apply(z2, d2[ob])
        return z1, a1, z2, a2, a2 @ W3.T + b3
    return fwd


def start(ck, reset):
    params = [p.detach().clone().requires_grad_(True) for p in ck['params']]
    if reset:
        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])
    else:
        adam = ([q.clone() for q in ck['m']], [q.clone() for q in ck['v']], [ck['tc']])
    return params, adam, RM.gen_from(ck['g_lab']), RM.gen_from(ck['g_batch'])


def train(ck, reset, fwd, epochs):
    params, adam, gl, gb = start(ck, reset)
    y = RM.RL.task_labels(gl)
    ce, acc, stp, epsf, _ = RM.train_task(params, adam, X, y, gb, epochs, fwd)
    return dict(state=RM.SH.state_sha256(params, adam, RM.ACT), ce=ce, acc=acc,
                params=[p.detach().clone() for p in params], m=[q.clone() for q in adam[0]])


def part_b(seed=100, epochs=2):
    global X
    mnist = bw.rm_setup()
    prows, units, cks, X, info = RM.run_prefix(seed, mnist, epochs, n_tasks=12, save_at=(2, 10))
    out = {}
    for direction in bw.DIRECTIONS:
        spec = bw.SETTINGS['RM'][direction]
        arm = RM.ARMS[spec['X']]
        ck = cks[spec['branch']]
        sh = RM.build_shift(arm, cks)
        runs = dict(B=train(ck, arm['reset'], bw.rm_bwd_forward(sh), epochs),
                    I=train(ck, arm['reset'], indep_forward(sh), epochs),
                    B0=train(ck, arm['reset'], bw.rm_bwd_forward({'p0': sh['p0'], 2: torch.zeros_like(sh[2])}), epochs),
                    N=train(ck, arm['reset'], None, epochs),
                    X=train(ck, arm['reset'], RM.make_forward(sh), epochs))
        r = dict(
            B_eq_indep_state=runs['B']['state'] == runs['I']['state'],
            B_eq_indep_ce_bits=nbits_diff(runs['B']['ce'], runs['I']['ce']) == 0,
            B_eq_indep_acc_bits=nbits_diff(runs['B']['acc'], runs['I']['acc']) == 0,
            B_eq_indep_param_values=all(torch.equal(a, b) for a, b in zip(runs['B']['params'], runs['I']['params'])),
            Bd0_eq_N_state=runs['B0']['state'] == runs['N']['state'],
            Bd0_eq_N_ce_bits=nbits_diff(runs['B0']['ce'], runs['N']['ce']) == 0,
            B_differs_from_N=runs['B']['state'] != runs['N']['state'],
            B_differs_from_X=runs['B']['state'] != runs['X']['state'],
            X_differs_from_N=runs['X']['state'] != runs['N']['state'])
        # (iii) the main-run path: bw.rm_arm (RM.run_arm with make_forward swapped, monitor on) vs the same
        # RM.run_arm with the independent forward swapped in
        rows_b, mon, expected, _ = bw.rm_arm('B', spec, cks, X, prows, seed, epochs, {}, monitor=True)

        def make_indep(sh_):
            return None if sh_ is None else indep_forward(sh_)
        with bw.swapped(RM, 'make_forward', make_indep):
            rows_i = RM.run_arm(spec['B'], arm, cks, X, prows, seed, epochs, units={})
        r['run_arm_B_eq_indep_state_k1_k2'] = [a['state_sha256'] == b['state_sha256'] for a, b in zip(rows_b, rows_i)]
        r['run_arm_B_eq_indep_online_k1_k2'] = [a['online_acc'] == b['online_acc'] for a, b in zip(rows_b, rows_i)]
        r['run_arm_B_monitor'] = dict(expected=expected, calls=mon['eager_calls'], mismatch=mon['eager_mismatch'])
        r['run_arm_B_branch_logits'] = bool(rows_b[0]['logits_equal_full'] and rows_b[0]['logits_equal_mb'])
        out[direction] = r
    return out


if __name__ == '__main__':
    torch.set_num_threads(1)
    res = dict(part_a=part_a())
    res['part_b'] = part_b()
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
