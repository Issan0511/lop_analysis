# NaP on F51, registered 2026-10-18

The implementation fixes the spec's sole candidate: two bias-free hidden linear maps,
affine LayerNorm before native ReLU, a bias-free 100-way head, and projection after
every Adam update. W1/W2/W3 each return to their own initial per-seed Frobenius sphere.
Each hidden layer's gamma and beta share a single projection onto the sphere of radius
sqrt(100). There is no cap, weight decay, moment projection, or output LayerNorm.

All six original affine tensors are drawn from the host stream before discarding the
biases. Class, image, and batch streams remain the host's. Production runs fix R=10,
30 tasks, 780 updates per task, lr=1e-4, batch=32, and CIFAR-100 standardization.
The engine captures one step as a CUDA graph and restores all warmup-mutated tensors.
Both NaP-self-fresh and the shared method-free R-fresh use exactly the saved t29 batches.

Run commands from this worktree, with `PY` set to the existing environment:

```bash
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
$PY -m analysis.nap_cifar5p1_1018.checks --device cpu
$PY -m src.nap_cifar5p1_1018 preflight --phase calib
```

The CPU suite uses synthetic fixtures only. Its `cpu_synthetic_all_pass` is separate
from `all_pass` and `production_ready`; neither of the latter can pass without CUDA.
The first execution attempt in this environment is blocked by CUDA and the outstanding
root-owned G0 audit. No production performance has been obtained.

On a scheduled GPU after the independent G0 audit has passed:

```bash
$PY -m analysis.nap_cifar5p1_1018.checks --device cuda
$PY -m src.nap_cifar5p1_1018 run --phase calib
$PY -m src.nap_cifar5p1_1018 select
git add results/nap_cifar5p1_1018/calib/selected.json
git commit -m 'Fix sole NaP candidate and calibration artifacts before evaluation'
$PY -m src.nap_cifar5p1_1018 run --phase eval
```

Both source and calibration artifacts are hashed. Production refuses CPU, missing G0,
unverified current sources, uncommitted implementation, uncommitted calibration selection,
changed calibration artifacts, and existing performance outputs. A checkpoint containing
parameters, moments, initial state, time, radii, stream states, and the saved t29 batches is
written atomically after every task. There is deliberately no automatic resume: a correct
restart must explicitly validate the checkpoint and retain exactly the same conditions.

The required comparator-reuse qualification is separate from G0:

```bash
$PY -m analysis.nap_cifar5p1_1018.verdict qualify --arm R --directory results/cifar5p1_mlp_0920/R_std_lr0.0001 --out results/nap_cifar5p1_1018/reuse_R.json
$PY -m analysis.nap_cifar5p1_1018.verdict qualify --arm SNA --directory results/cifar5p1_mlp_0920/SNA_std_lr0.0001 --out results/nap_cifar5p1_1018/reuse_SNA.json
$PY -m analysis.nap_cifar5p1_1018.verdict qualify --arm KKT1 --directory results/cifar5p1_mlp_0920/KKT1_std_lr0.0001 --out results/nap_cifar5p1_1018/reuse_KKT1.json
$PY -m analysis.nap_cifar5p1_1018.verdict verdict --r-qualification results/nap_cifar5p1_1018/reuse_R.json --sna-qualification results/nap_cifar5p1_1018/reuse_SNA.json --kkt1-qualification results/nap_cifar5p1_1018/reuse_KKT1.json
```

Historical source differences are rejected pending an independent equivalence audit;
similar aggregate accuracy is never a reuse proof. The verdict requires all ten seeds
and all thirty tasks, joins by seed, and applies two-sided exact sign tests plus Holm
across the two registered NaP-versus-KKT1/SNA comparisons. Equivalence uses [d_(2),d_(9)]
within [-.005,.005]. Unresolved evidence remains unresolved. NaP-versus-R is report-only.
NaP-versus-SNA/KKT1 is an **OPEN_COMPARISON**, not a paper-predicted winner.

## Added ranking predictions, before NaP production

The OPEN_COMPARISON statement above records the initial registration. Issa added
**KKT1 > NaP** and **SNA > NaP** at 2026-10-18 02:07:26/02:08:13 JST. No KKT1
versus SNA order or quantitative probability was supplied. These are Issa's added
predictions, rather than rankings already established by the paper.
The authoritative append-only record is
`specs/amendments/spec_nap_cifar5p1_1018_prediction_20261018T020813JST.md` and its
JSON. Filename/status-only independent inspection found no NaP calibration or
evaluation data at 02:08:57 JST; the only calibration launch was BLOCKED_CUDA.
Sources, original spec/checks, two-sided tests and Holm family of two are unchanged.
Preserve original runtime prediction metadata and cite the amendment commit/hash
alongside it in the final report. UNRESOLVED is not a comparator victory.
