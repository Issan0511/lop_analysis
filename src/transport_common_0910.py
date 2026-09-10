"""Shared glue for spec_transport_holes_0910 (sub-runs A/B/C/D).

Committed modules are imported and NEVER edited: their sha256 is recorded in the
provenance of already-committed results.  Everything here is either a new
function or an in-process fix-up, and every fix-up is recorded in provenance.

The traps this file exists to close are enumerated in spec §7.5.  The two that
bite silently:

  * `src/boundary_groups_0908.py:10` pins `H.DATA_DIR` at another machine's
    absolute path.  `data_dir()` re-resolves it and checks the bytes against the
    sha256 recorded in committed provenance, so a correct copy in a different
    directory yields identical provenance.
  * `C.make_act` raises KeyError for GELU/SILU, and `C.restore` calls it through
    the module global -- so `restore()` here is a local 12-line copy rather than
    a monkeypatch of a committed module.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from src import width_sink_clamp_0909 as C
from src import valley_acts_0910 as VA

H = C.H
ROOT = C.ROOT
VALLEY_ARMS = tuple(VA.VALLEY)                     # ('GELU', 'SILU')
UNIT_KEYS = ('zbar_i', 'sd_i', 'cnorm_i', 'm_i', 'bias_i', 'star_i')

# sha256 of the four idx files, copied from results/width_sink_clamp_0909/
# LR_none_s0_provenance.json.  A different MNIST would change every trajectory,
# so this is checked rather than trusted.
DATA_SHA = {
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
    't10k-images-idx3-ubyte.gz': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6',
    't10k-labels-idx1-ubyte.gz': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6',
}


def data_dir():
    """Point H.DATA_DIR at a directory that actually holds the committed MNIST."""
    for d in (Path(H.DATA_DIR), ROOT / 'data' / 'mnist'):
        if all((d / f).exists() for f in H.Mnist.FILES.values()):
            got = {f: hashlib.sha256((d / f).read_bytes()).hexdigest()
                   for f in H.Mnist.FILES.values()}
            bad = {f: (got[f], DATA_SHA[f]) for f in got if got[f] != DATA_SHA[f]}
            assert not bad, ('MNIST bytes differ from the committed provenance', d, bad)
            H.DATA_DIR = d
            return d
    raise FileNotFoundError(f'no MNIST under {H.DATA_DIR} or {ROOT / "data" / "mnist"}')


def make_act(arm):
    """R / LR / ELU1 / SNA from the committed dispatcher; GELU / SILU from valley_acts."""
    a = VA.make_valley(arm)
    return a if a is not None else C.make_act(arm)


def restore(snap, arm):
    """C.restore, but through the make_act above (C.restore KeyErrors on GELU/SILU)."""
    p = [q.clone().requires_grad_(True) for q in snap['p']]
    act = make_act(arm)
    if snap['V'] is not None:
        act.V = [v.clone() for v in snap['V']]
    adam = ([x.clone() for x in snap['adam'][0]], [x.clone() for x in snap['adam'][1]],
            [snap['adam'][2]])
    gens = []
    for s in snap['gens']:
        g = torch.Generator()
        g.set_state(s.clone())
        gens.append(g)
    return p, act, adam, gens


# --------------------------------------------------------------- gate counters
def gate_stats(p, act, probe, perm, arm):
    """dead_hard / dead_soft / sat keep their committed definitions so the leaky
    and ELU numbers stay comparable.  They threshold the SIGNED gate, so for a
    valley activation a unit past z_c (phi' down to -0.129 for GELU, -0.0998 for
    SiLU -- both bigger in magnitude than leaky's 0.1 floor) is counted 'dead'
    while being MORE mobile than leaky, with the opposite sign.  The four new
    counters separate that: immobile_units uses |phi'|, and gate_neg_frac /
    inv_units / beyond_frac measure the inversion itself."""
    with torch.no_grad():
        z = probe.px[:, perm] @ p[0].detach().T + p[1].detach()
        g = act.dphi(z, 0) if isinstance(act, H.AdaptiveSnake) else act.dphi(z)
        gmax = g.max(0).values
        out = dict(dead_hard=int((gmax < 1e-6).sum()),
                   dead_soft=int((gmax < .05).sum()),
                   sat=float((g < .05).float().mean()),
                   immobile_units=int((g.abs().max(0).values < 1e-6).sum()),
                   gate_neg_frac=float((g < 0).float().mean()),
                   inv_units=int((((g < 0).float().mean(0)) > .5).sum()))
        zc = VA.ZC.get(arm)
        out['beyond_frac'] = float((z < zc).float().mean()) if zc is not None else 0.
    return out


def check_gates(rows, arm, ck):
    """Falsifiable in BOTH directions: a valley arm must show inversion, a
    non-valley arm must show exactly none.  Either half alone would be vacuous."""
    neg = max(r['gate_neg_frac'] for r in rows)
    bey = max(r['beyond_frac'] for r in rows)
    inv = max(r['inv_units'] for r in rows)
    ck.update(gate_neg_frac_max=neg, beyond_frac_max=bey, inv_units_max=inv)
    if arm in VALLEY_ARMS:
        assert neg > 0.05, (arm, 'valley arm never puts 5% of the probe past z_c', neg)
        assert bey > 0.05, (arm, 'valley arm never reaches beyond z_c', bey)
        # phi'(z) < 0  <=>  z < z_c is an identity, so the two fractions must agree
        # exactly.  This is what actually checks the tabulated z_c against the
        # activation's own derivative on live data, task by task.
        worst = max(abs(r['gate_neg_frac'] - r['beyond_frac']) for r in rows)
        ck['gate_neg_vs_beyond_maxabs'] = worst
        assert worst == 0., (arm, 'phi\'<0 and z<z_c disagree: z_c is wrong', worst)
    else:
        assert neg == 0. and bey == 0. and inv == 0, \
            (arm, 'non-valley arm shows a negative gate', neg, bey, inv)


def finite_guard(rows, tag):
    """There is no NaN guard anywhere in the committed loop: on divergence
    `(cn0 > CAP*c).any()` is False for NaN, the step counts as idle, and the run
    dies ~1000 s later complaining about an unexercised code path."""
    for r in rows:
        bad = [k for k, v in r.items()
               if isinstance(v, float) and not np.isfinite(v)]
        assert not bad, ('non-finite measurement -- divergence?', tag, r.get('task'), bad)


# ------------------------------------------------------------------ G1 anchors
def anchor_path(arm, seed):
    """Which committed per-unit file this arm's reference trajectory reproduces.
    Returns (path, provenance-note).  See spec §3 for how strong each one is."""
    if arm in VALLEY_ARMS:
        return (ROOT / 'results/long_horizon_acts_0910' / f'{arm}_none_s{seed}_units.npz',
                'sub-run A of this same spec (not an independent registered run)')
    if arm == 'R':
        return (ROOT / 'results/leak_ladder_force_posthoc_0910' / f'R_s{seed}_units.npz',
                'leak_ladder_force_posthoc_0910 (post-hoc, unregistered; itself 0.0 '
                'against gate_scale_invariance_0909)')
    return (ROOT / 'results/width_sink_clamp_0909' / f'{arm}_none_s{seed}_units.npz',
            'width_sink_clamp_0909 (registered)')


def g1_ref_units(arm, seed, units, t_from, t_to, ck, prefix='g1'):
    """Compare the `ref` arm's per-unit arrays against the anchor, over t_from..t_to
    and all six arrays.  Records the comparison COUNT next to the maxabs: a check
    with zero keys matched reports 0.0 and would otherwise pass (追補 1 R8)."""
    path, note = anchor_path(arm, seed)
    ref = np.load(path)
    worst, n = 0., 0
    for k in UNIT_KEYS:
        for t in range(t_from, t_to + 1):
            key = f'ref_{k}_t{t}'
            if key in ref and key in units:
                worst = max(worst, float(np.abs(ref[key] - units[key]).max()))
                n += 1
    ck[f'{prefix}_units_maxabs'] = worst
    ck[f'{prefix}_units_compared'] = n
    ck[f'{prefix}_units_expected'] = len(UNIT_KEYS) * (t_to - t_from + 1)
    ck[f'{prefix}_anchor'] = str(path.relative_to(ROOT))
    ck[f'{prefix}_anchor_note'] = note
    return worst, n


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance_base(mnist):
    return dict(data_sha256=mnist.sha256,
                spec_sha256=sha(ROOT / 'specs/spec_transport_holes_0910.md'),
                common_sha256=sha(Path(__file__)),
                base_sha256=sha(Path(C.__file__)),
                host_sha256=sha(Path(H.__file__)),
                valley_sha256=sha(Path(VA.__file__)),
                torch_version=torch.__version__,
                numpy_version=np.__version__,
                data_dir=str(H.DATA_DIR))


def write_rows(path, rows):
    keys = []
    [keys.append(k) for r in rows for k in r if k not in keys]
    C.G.B.csvwrite(path, [{k: r.get(k) for k in keys} for r in rows])


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=str))
