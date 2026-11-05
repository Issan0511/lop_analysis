"""Explicit reuse of unchanged training records after the clamp-boundary fix."""
import json
import shutil
from pathlib import Path
import common as T

LEGACY = json.loads(Path(__file__).with_name('t5_legacy_sources.json').read_text())
TRAINING_SOURCES = ('analysis/vit_two_block_1025/common.py', 'analysis/vit_two_block_1025/train.py',
    'specs/spec_vit_two_block_1025.md', 'analysis/vit_two_block_1025/t5_model.py',
    'specs/spec_vit_two_block_t5_1025.md')


def legacy_sources(sources):
    return sources == LEGACY


def training_compatible(saved, current):
    if saved == current:
        return True
    old, new = dict(saved), dict(current)
    a, b = old.pop('source_hashes'), new.pop('source_hashes')
    return old == new and legacy_sources(a) and all(a[k] == b[k] for k in TRAINING_SOURCES)


def preserve_previous_evaluation(out, current, head):
    path = out / 'provenance.json'
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    if previous['config'] == current:
        return
    assert training_compatible(previous['config'], current)
    backup = out / 'before_clamp_boundary_fix'
    backup.mkdir(exist_ok=True)
    copied = []
    for name in ('provenance.json', 'complete.json', 'manifest.json', 'measurements.npz', 'summary.csv', 'input.json', 'status.json'):
        source, target = out / name, backup / name
        if source.exists():
            if target.exists():
                assert T.sha256(source) == T.sha256(target), target
            else:
                shutil.copy2(source, target)
            copied.append(dict(path=name, bytes=source.stat().st_size, sha256=T.sha256(source)))
    T.atomic_json(backup / 'preservation_manifest.json', copied)
    T.atomic_json(path, dict(config=current, git_hash=head,
        previous_provenance=str(backup / 'provenance.json'),
        reason='Re-evaluate unchanged replicas with the original clamp derivative at zero.',
        training_records_reused_without_rewriting=True))
