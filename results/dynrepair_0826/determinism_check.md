# Determinism check

Two independent in-memory analysis builds produced identical canonical CSV bytes.
Per-arm replay checks (S8-<arm>) are listed in sanity.csv and runner_meta.json.

The cross-run half of S8 is a separate operator step, because it re-executes
the same command from the same commit into a second output directory:

```
OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair --config <cfg> --outdir <dir2>
OMP_NUM_THREADS=1 .venv/bin/python -m src.dynrepair --compare-outdirs <dir1> <dir2>
```

| artifact | SHA-256 |
|---|---|
| traj.csv | fd73be6c3acb88f497a0b8820fba33445d65c5cfd77d40c083f8a3a1a5ff17ce |
| utility.csv | 5f2208f00a4d3128d7d25937605b55d836f207c663fee08b820422194dd16b07 |
| units.csv | 4281e4c233d53244e6bfe893e922e6eebd804cc2e1c6ed21fc8b01ef68d2ec77 |
| km.csv | 3894aab194eeaedb3f9112e32a8a6594981de0d1f97c5fe8837e7e7dde708695 |
| verdict.csv | 85f8a6a82c577cfab975a5912d0a6b2f249ec070959bc6d514717bc26d550277 |
| placebo.csv | b8f01b447360883d1e2a58dae94f569265bb0bf79905e341c1b004beaecf7b9e |
| sanity.csv | e3c21b3a1f66d3afdd4f3d38da9ef2eadda07490251b62888dbcfb020d2dd259 |
| manifest.csv | f9eaed1545cf69c29de929e3dffc64c01c9a2cc8d694eef264493212b8472174 |
| raw_sha256.csv | ad6f769692d075c9a56df159fee040383e68072496dac4717b78d6f77c587e1b |

## S8 cross-run (registered §9 S8)

Re-executed the same command from the same commit into a separate output
directory and compared. `git_untracked_sources` is normalised: the second run
sees the first run's own stdout log, so that provenance field grows between
executions while nothing about the code changes.

- replay outdir: `results/dynrepair_0826_s8replay` (not committed)
- compared: 2026-08-27 06:44 JST

```
  traj.csv: identical
  utility.csv: identical
  units.csv: identical
  km.csv: identical
  verdict.csv: identical
  placebo.csv: identical
  sanity.csv: identical
  manifest.csv: identical
  raw_sha256.csv: identical
  logs/unit_traj_A0.npz: identical (20 keys)
  logs/unit_traj_A1.npz: identical (20 keys)
  logs/unit_traj_A2.npz: identical (20 keys)
  logs/unit_traj_A3.npz: identical (20 keys)
  logs/unit_traj_A1_lo.npz: identical (20 keys)
  logs/unit_traj_A1_hi.npz: identical (20 keys)
S8 cross-run: PASS
```
