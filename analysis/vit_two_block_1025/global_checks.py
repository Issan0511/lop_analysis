"""Re-read gate plus manifest/initial-state checks for full-model replicas."""
import json
import subprocess
import sys
from pathlib import Path
import common as T
from global_noise import hashes


def main():
    gate = T.ARCHIVE / 'gate'
    out = gate / 'global_noise'
    result = json.loads((out / 'complete.json').read_text())
    assert result['status'] == 'complete' and result['config']['source_hashes'] == hashes()
    files = sorted(out.glob('*.pt'))
    assert len(files) == 4
    before = {str(p): T.sha256(p) for p in files}
    with T.np.load(out / 'measurements.npz') as data:
        prior = {k: data[k].copy() for k in data.files}
    for name, value in prior.items():
        if name.startswith('s0_'):
            assert T.np.array_equal(value, T.np.repeat(value[:1], 4, axis=0)), name
    for path in files:
        record = T.torch.load(path, map_location='cpu', weights_only=False)
        assert record['non_fc1_changed']
    with (gate / 'global_noise_reread.log').open('x') as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('global_noise.py')),
            '--gate', '--output', str(out)], cwd=T.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert before == {str(p): T.sha256(p) for p in files}
    with T.np.load(out / 'measurements.npz') as data:
        assert set(prior) == set(data.files)
        for name, value in prior.items():
            assert T.np.array_equal(value, data[name]), name
    T.atomic_json(gate / 'global_noise.json', dict(status='passed', source_hashes=hashes(),
        git_hash=T.git_state()[0], checker_sha256=T.sha256(Path(__file__)), checks=result['checks'],
        identical_initial_observations=True, non_fc1_updated=True,
        reread_replicas_unchanged=True, reread_measurements_bit_equal=True))
    print('Full-model replica gate passed', flush=True)


if __name__ == '__main__':
    main()
