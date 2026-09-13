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
ZC_BAND = 1e-3          # how close to z_c a sign disagreement is allowed to be
ZC_MUT = 0.05           # the wrong z_c the mutation control uses


def gate_stats(p, act, probe, perm, arm):
    """dead_hard / dead_soft / sat keep their committed definitions so the leaky
    and ELU numbers stay comparable.  They threshold the SIGNED gate, so for a
    valley activation a unit past z_c (phi' down to -0.129 for GELU, -0.0998 for
    SiLU -- both bigger in magnitude than leaky's 0.1 floor) is counted 'dead'
    while being MORE mobile than leaky, with the opposite sign.

    The far side of the valley has TWO regimes, and float32 is where the second
    one lives.  GELU's dphi = Phi(z) + z*phi(z) evaluates to exactly +0.0 for
    float32 z <= -14.3439 (SiLU's at -90), so a unit that has escaped far enough
    stops moving altogether.  That is not an artefact to be tolerated: it is the
    end state the derivation predicts ("past the valley a unit escapes until
    phi' -> 0 and freezes").  So it gets its own counter rather than being
    silently folded into the inverted ones:

        beyond_frac    z < z_c                     (the whole far side)
        gate_neg_frac  phi' < 0                    (escaped AND still mobile)
        frozen_frac    z < z_c and phi' >= 0       (escaped AND frozen)
        inv_units      units >50% inverted
        frozen_units   units >50% frozen past z_c
        immobile_units |phi'| < 1e-6 on EVERY sample

    zc_bad_lo / zc_bad_hi are the sign disagreements that underflow does NOT
    explain; they are what actually tests the tabulated z_c (see check_gates)."""
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
        if zc is None:
            out.update(beyond_frac=0., frozen_frac=0., frozen_units=0,
                       zc_bad_lo=0, zc_bad_hi=0, zc_mut_fires=0)
            return out
        below, neg = z < zc, g < 0
        frozen = below & ~neg
        near = (z - zc).abs() <= ZC_BAND
        out['beyond_frac'] = float(below.float().mean())
        out['frozen_frac'] = float(frozen.float().mean())
        out['frozen_units'] = int(((frozen.float().mean(0)) > .5).sum())
        # a pair past z_c whose gate is not negative must be an exact underflow
        # (or sit within ZC_BAND of the bottom); a pair before z_c must not be
        # negative at all.  Both counts are 0 iff the tabulated z_c is right.
        out['zc_bad_lo'] = int((below & ~neg & (g != 0) & ~near).sum())
        out['zc_bad_hi'] = int((~below & neg & ~near).sum())
        # in-run mutation control: the same test against a z_c that is wrong by
        # ZC_MUT must find disagreements, otherwise the test above is vacuous
        zw = zc + ZC_MUT
        out['zc_mut_fires'] = int(((z < zw) & ~neg & (g != 0) & ((z - zw).abs() > ZC_BAND)).sum()
                                  + ((z >= zw) & neg & ((z - zw).abs() > ZC_BAND)).sum())
    return out


def check_gates(rows, arm, ck):
    """Falsifiable in BOTH directions: a valley arm must show inversion, a
    non-valley arm must show exactly none.  Either half alone would be vacuous.

    The z_c test is NOT "gate_neg_frac == beyond_frac".  That equality is false
    in float32 once the deepest units underflow, and asserting it destroyed three
    finished 400-task runs (see spec 追補 2).  What is tested instead is that
    every sign disagreement is accounted for by underflow or by sitting within
    ZC_BAND of the bottom -- with an in-run control showing a z_c wrong by
    ZC_MUT would be caught."""
    neg = max(r['gate_neg_frac'] for r in rows)
    bey = max(r['beyond_frac'] for r in rows)
    inv = max(r['inv_units'] for r in rows)
    ck.update(gate_neg_frac_max=neg, beyond_frac_max=bey, inv_units_max=inv,
              frozen_frac_max=max(r['frozen_frac'] for r in rows),
              frozen_units_max=max(r['frozen_units'] for r in rows),
              zc_bad_lo_max=max(r['zc_bad_lo'] for r in rows),
              zc_bad_hi_max=max(r['zc_bad_hi'] for r in rows),
              zc_mut_fires_max=max(r['zc_mut_fires'] for r in rows))
    if arm in VALLEY_ARMS:
        assert neg > 0.05, (arm, 'valley arm never puts 5% of the probe past z_c', neg)
        assert bey > 0.05, (arm, 'valley arm never reaches beyond z_c', bey)
        assert ck['zc_bad_lo_max'] == 0 and ck['zc_bad_hi_max'] == 0, \
            (arm, 'sign disagreement that underflow does not explain: z_c is wrong',
             ck['zc_bad_lo_max'], ck['zc_bad_hi_max'])
        assert ck['zc_mut_fires_max'] > 0, \
            (arm, 'vacuous z_c test: a z_c wrong by %g would not be caught' % ZC_MUT)
    else:
        # beyond_frac is a hard-coded 0 for a non-valley arm, so it is left out:
        # asserting it would be 0. == 0.  neg and inv ARE reductions over phi'.
        assert neg == 0. and inv == 0, \
            (arm, 'non-valley arm shows a negative gate', neg, inv)


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
