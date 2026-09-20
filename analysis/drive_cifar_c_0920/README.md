# A2 implementation

`src/drive_cifar_c_0920.py` implements the C host, full1200 every-update observations,
Adam certificates, self/upstream/total signed transport, shared-denominator conf/label/history,
epoch atomic checkpoints, failure fixtures and fixed first/last epoch audit replay.
`numerics.py` records each independent roundoff bound. At exactly zero the unchanged host's
clamp backward is one; see the preregistration implementation appendix.

The current user authorization is implementation/testing only. No production run was started.
Test seeds are always 100–109 (R10); scientific seeds are 0–9.

```bash
python -m analysis.drive_cifar_c_0920.checks --full --out results/_checks_drive_cifar_c_0920/attempt
python -m analysis.drive_cifar_c_0920.audit --src results/_checks_drive_cifar_c_0920/attempt/observed/audit/t2_e400.pt
```

A later, explicitly authorized production launch uses:

```bash
python -m src.drive_cifar_c_0920 --mode production --production-go --checks CHECKS.json --epochs 400 --tasks 5 --out results/drive_cifar_c_0920/run
python -m analysis.drive_cifar_c_0920.report calibrate --src results/drive_cifar_c_0920/run --window results/drive_cifar_c_0920/window_calibration.json
# Commit the window file before the next command.
python -m analysis.drive_cifar_c_0920.report report --src results/drive_cifar_c_0920/run --window results/drive_cifar_c_0920/window_calibration.json --out results/drive_cifar_c_0920/report
```

Touch `STOP` inside the run directory to stop at the next epoch boundary; remove it
and add `--resume` to continue with identical source/input/config. Failed checks are
saved before abort. Resume treats the atomic checkpoint as the committed progress.
Production report refuses check seeds, partial shards, altered source and forged markers.
It reads calibration and primary observations in separate stages. File hashes are checked
before reading contents, and the window's current committed bytes must match exactly.

The full host qualification compares parameters, both moments, step, label/batch RNG,
actual labels and every existing per-task numeric column against the unchanged host for
2 × 400 epochs. Old production seed0–9 trajectory comparison remains report-only work for
a future production run; no old production outcomes are opened by these implementation checks.
