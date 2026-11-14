# Matched readout interventions for kunekune activations

The preregistered question is whether the adaptive KKA, KKA23 and KKT1 activations
can retain their derivatives while still incurring a readout-dependent relearning
cost. SNA is a related reference; ELU and ReLU anchor the existing no-LN MLP
readout controls. See `specs/spec_kunekune_readout_1114.md` for the frozen design.

The experiment pairs 10 seeds across 6 activations, 3 checkpoint ages and 8 arms.
All probes run for 4000 updates. The scale intervention is a one-time 0.1 scaling
of the class-centered readout, with mean-logit bias compensation. It is not the
earlier per-column initial-norm constraint or a continuing intervention. P uses
independent class permutations within each column; it is not a pure alignment
intervention. Adaptive activation EMA state is retained except in the fresh-body
reference F0.

From the repository root, using the project Python environment:

```bash
python -m analysis.kunekune_readout_1114.checks
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m analysis.kunekune_readout_1114.launch --workers 12
python -m analysis.kunekune_readout_1114.analyze --raw results/kunekune_readout_1114/raw --out results/kunekune_readout_1114
```

`checks` is a synthetic preproduction test using seed 9001. It does not select
study settings. The source runner checks all 30 ELU/ReLU task endpoints against
the original saved parameters, and eligible continuation curves against earlier
studies. Those anchors are exact replays, not independent replications.

The analyzer requires all 1440 curves, verifies paired streams and hashes, and
reloads every final model state to reproduce endpoint accuracy, CE and derivative
diagnostics. It independently computes the integral by two methods and bootstrap
means by indexing and counts. `summary.md` reports every age; `contrasts.csv`
contains all registered contrasts and paired activation differences. All eight
arms appear in the learning-curve figures. T90 is the first observed grid point
with at least 1080 correct predictions; nonarrival is censored.

Raw states, per-probe curves and worker logs are archived under
`/home/issan/Projects/obsidian-research-data/kunekune_readout_1114/`, with their
original repository-relative structure. `backup_manifest.json` records each
source, destination, size and SHA-256. Reanalysis after archive uses:

```bash
python -m analysis.kunekune_readout_1114.analyze --raw /home/issan/Projects/obsidian-research-data/kunekune_readout_1114/results/kunekune_readout_1114/raw --out /tmp/kunekune_readout_reanalysis
```

Positive B_P means 0.1 scaling helps after P. Positive R_body means the learned
body remains slower than the fresh reference after zeroing the readout and
clearing Adam. A small readout benefit alone establishes neither body health nor
readout health. These are fixed-training-set results from one no-LN MLP, with
adaptive shape/scale histories; no ViT or generalization claim follows.
