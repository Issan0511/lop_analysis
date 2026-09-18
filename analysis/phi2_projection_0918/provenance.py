"""Write results/phi2_projection_0918/provenance.json: git state, input checkpoint hashes, library versions, commands."""
import hashlib, json, subprocess, sys, platform
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phi2_projection_0918"
CK = Path("/home/issan/Projects/obsidian-research-data/zero_attraction_0913/training/ckpts")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    inputs = {f"{a}_step{s}.pt": sha(CK / f"{a}_step{s}.pt") for a in ("LR_a0p1_q0", "LR_a0p7_q0") for s in (200000, 5000000)}
    outputs = {p.name: sha(p) for p in sorted(OUT.glob("*.csv"))}
    prov = dict(
        experiment="phi2_projection_0918",
        kind="post-hoc theory verification (spec_phi2_projection_0918.md); no new long training",
        git_hash=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=bool(git("status", "--porcelain", "--", "analysis/phi2_projection_0918", "specs/spec_phi2_projection_0918.md")),
        inputs=dict(checkpoint_dir=str(CK), sha256=inputs),
        outputs_sha256=outputs,
        commands=[
            "run.py perturb --T 10000 --eps 1e-2 --n_pos 5 --n_neg 5 --margin 0.03",
            "run.py perturb --T 2000 --eps 1e-3 --n_pos 5 --n_neg 5 --margin 0.03 --tag _lin",
            "run.py switch --T 10000 --margin 0.0",
            "run.py perturb --T 2000 --eps 1e-2 --n_pos 5 --n_neg 5 --margin 0.03 --qdir munull --tag _null",
            "run.py perturb --T 2000 --eps 1e-2 --n_pos 5 --n_neg 5 --margin 0.03 --freeze_others --tag _frz",
            "run.py kick --T 2000 --delta 0.2 --n_units 5 --zmax 0.3",
            "run.py kick --T 2000 --delta 0.005 --n_units 5 --zmax 0.3 --tag _d0005",
            "run.py kick --T 2000 --delta 0.05 --n_units 5 --zmax 0.3 --tag _d005",
            "report.py",
        ],
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
        dtype="float64", lr=0.005, threads=2,
    )
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(json.dumps({k: prov[k] for k in ("git_hash", "git_branch", "git_dirty")}))
