"""Reproduce the ordinary-learning intervention comparisons from saved reports.

Contrasts are descriptive within the same 40 states, not independent unit tests
of a population hypothesis. Linear accounting always uses unnormalized fields.
"""
import json
from pathlib import Path
import pandas as pd
import common as T


def main():
    out = T.ROOT / 'results/vit_two_block_1025'
    families = ['ordinary_drive', 'ordinary_residual', 'ordinary_tokens']
    frames = {f: pd.read_csv(out / f'{f}_cases.csv') for f in families}
    keys = ['seed', 'task', 'start', 'layer']
    expected = {(s, t, k, l) for s in T.SEEDS for t in (10, 60)
                for k in (100, 500) for l in (1, 2)}
    checks = {}
    inputs = []
    for family, data in frames.items():
        assert set(data[keys].itertuples(index=False, name=None)) == expected
        summary = json.loads((out / f'{family}_inputs.json').read_text())
        entries = summary['inputs']
        assert len(entries) == 40
        for entry in entries:
            path = Path(entry['path'])
            assert T.sha256(path) == entry['sha256']
            marker = json.loads(path.read_text())
            assert marker['status'] == 'complete' and marker['frozen_unchanged']
            for rel, digest in marker['source_hashes'].items():
                assert T.sha256(T.ROOT / rel) == digest, rel
            assert T.sha256(Path(marker['checkpoint'])) == marker['checkpoint_sha256']
        inputs.append(dict(path=str(out / f'{family}_cases.csv'),
                           sha256=T.sha256(out / f'{family}_cases.csv')))
        checks[family] = dict(states=40, source_and_checkpoint_hashes_match=True)

    drive = frames['ordinary_drive']
    q = drive[(drive['kind'] == 'linear') & (drive['mode'] == 'fixed')]
    pairs = q.pivot(index=keys, columns='arm', values='r_mean')
    pairs['image_gate_pairing'] = pairs['relu'] - pairs['rate_gate']
    pairs['sign_additivity_error'] = (
        pairs['positive_request'] + pairs['negative_request'] - pairs['relu'])
    assert pairs['sign_additivity_error'].abs().max() < 1e-9
    pairs[['relu', 'rate_gate', 'image_gate_pairing', 'positive_request',
           'negative_request', 'sign_additivity_error']].reset_index().to_csv(
        out / 'ordinary_gate_pairing.csv', index=False)

    residual = frames['ordinary_residual']
    q = residual[(residual['kind'] == 'linear') & (residual['mode'] == 'fixed')]
    terms = q.pivot(index=keys, columns='arm', values='r_mean')
    terms['wrong_class_redistribution'] = terms['stable'] - terms['uniform_wrong']
    terms.reset_index().to_csv(out / 'ordinary_residual_terms.csv', index=False)

    # Compact table: finite single updates with matched parameter-gradient norm.
    selection = [
        ('ordinary_drive', 'relu', 'baseline'),
        ('ordinary_drive', 'all_open_norm', 'backward_gate_one'),
        ('ordinary_drive', 'rate_gate_norm', 'image_gate_pairing_removed'),
        ('ordinary_residual', 'equal_error', 'equal_error'),
        ('ordinary_residual', 'uniform_wrong', 'uniform_wrong'),
        ('ordinary_residual', 'both', 'equal_correct_margin'),
        ('ordinary_tokens', 'cls_only', 'cls_gradient_only'),
        ('ordinary_tokens', 'patch_only', 'patch_gradient_only'),
    ]
    compact = []
    for family, arm, name in selection:
        q = frames[family]
        q = q[(q.arm == arm) & (q['kind'].isin(['finite', 'finite_norm']))]
        if 'population' in q:
            q = q[q.population == 'cls']
        for (layer, mode), group in q.groupby(['layer', 'mode']):
            assert len(group) == 20
            compact.append(dict(intervention=name, layer=int(layer), mode=mode,
                states=len(group), independent_seeds=group.seed.nunique(),
                r_down=int((group.r_mean < 0).sum()),
                r_up=int((group.r_mean > 0).sum()),
                r_mean=float(group.r_mean.mean()),
                ce_down=int((group.ce_delta < 0).sum()),
                ce_delta_mean=float(group.ce_delta.mean())))
    pd.DataFrame(compact).to_csv(out / 'ordinary_intervention_counts.csv', index=False)

    # Identical margin direction for every image, computed independently from
    # saved logits. This checks the interpretation, not a claim about its sign.
    import numpy as np
    margin_checks = []
    for folder in sorted((T.ARCHIVE / 'ordinary_residual').glob('seed*')):
        with np.load(folder / 'arrays.npz') as a:
            f, eps = a['logits'].astype('float64'), a['epsilon']
            assert np.isfinite(f).all() and (eps > 0).all()
            # All selected states are fit to 100%; the true class is therefore
            # exactly the argmax, verified against per-state report accuracy.
            ident = json.loads((folder / 'complete.json').read_text())
            subset = residual
            for key in keys:
                subset = subset[subset[key] == ident[key]]
            assert (subset.accuracy0 == 1).all()
            y = f.argmax(1)
            e = np.full_like(f, eps.mean() / (f.shape[1] - 1))
            e[np.arange(len(f)), y] = -eps.mean()
            margin_gradient = np.full_like(f, -1 / (f.shape[1] - 1))
            margin_gradient[np.arange(len(f)), y] = 1
            error = float(np.abs(e + eps.mean() * margin_gradient).max())
            assert error < 1e-15
            margin_checks.append(error)
    checks['equal_margin_signal_identity_max_error'] = max(margin_checks)
    checks['linear_sign_additivity_max_error'] = float(pairs['sign_additivity_error'].abs().max())
    T.atomic_json(out / 'ordinary_synthesis_checks.json', dict(
        status='passed', checks=checks, inputs=inputs,
        report_sha256=T.sha256(Path(__file__)),
        warning='20 states per layer are repeated observations of five seeds. '
                'Finite one-step results do not establish long-run interventions.'))
    print(json.dumps(dict(status='passed', families=3, states_per_family=40,
                          compact_rows=len(compact))))


if __name__ == '__main__':
    main()
