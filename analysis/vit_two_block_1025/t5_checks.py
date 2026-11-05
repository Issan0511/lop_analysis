"""Synthetic operational gates, including re-reading completed replica files."""
import json
import subprocess
import sys
from pathlib import Path
import common as T
from t5_run import source_hashes
from t5_ledger import verification


def main():
    gate = T.ARCHIVE / 'gate/t5_clamp_boundary'
    gate.mkdir(exist_ok=True)
    checks = []
    math = verification()
    for layer in (1, 2):
        for mode in ('fixed', 'adam'):
            out = gate / f't5_L{layer}_{mode}'
            cmd = [sys.executable, '-u', str(Path(__file__).with_name('t5_run.py')),
                   '--layer', str(layer), '--mode', mode, '--gate', '--output', str(out)]
            print(json.dumps(dict(starting=out.name)), flush=True)
            with (gate / f'{out.name}.log').open('x') as log:
                subprocess.run(cmd, cwd=T.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((out / 'complete.json').read_text())
            assert result['status'] == 'complete'
            assert result['config']['source_hashes'] == source_hashes()
            assert result['frozen_unchanged'] and result['prefix_bit_equal']
            checks.append(dict(layer=layer, mode=mode, checks=result['checks']))
            if layer == 1 and mode == 'fixed':
                files = sorted(out.glob('*.pt'))
                before = {str(p): T.sha256(p) for p in files}
                with T.np.load(out / 'measurements.npz') as data:
                    prior = {name: data[name].copy() for name in data.files}
                with (gate / 't5_reread.log').open('x') as log:
                    subprocess.run(cmd, cwd=T.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert before == {str(p): T.sha256(p) for p in files}
                with T.np.load(out / 'measurements.npz') as data:
                    assert set(data.files) == set(prior)
                    for name in data.files:
                        assert T.np.array_equal(data[name], prior[name]), name
    result = dict(status='passed', source_hashes=source_hashes(),
        git_hash=T.git_state()[0], math=math, cases=checks,
        reread_replicas_unchanged=True, reread_measurements_bit_equal=True)
    T.atomic_json(gate / 't5.json', result)
    T.atomic_json(T.ARCHIVE / 'gate/t5.json', result)
    print('T5 operational gates passed', flush=True)


if __name__ == '__main__':
    main()
