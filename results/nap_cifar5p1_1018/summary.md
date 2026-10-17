# NaP on F51: implementation complete, production blocked

The full calibration command was attempted on 2026-10-17 16:22 UTC. The installed torch build is 2.13.0+cu130, but CUDA is unavailable. The run stopped at the registered preflight and produced no calibration/evaluation performance. G0 host reproduction and mutation proofs remain incomplete; the CUDA graph S check has not run.

All eight CPU synthetic S categories pass, including exact host R=10 online/parameter reproduction (maximum residual 0), an independent float64 LN/gradient reference, full-matrix/affine projection mutation checks, fresh-state and diagnostic-isolation checks, exhaustive exact-sign/Holm checks, and fail-closed audit admission mutations. `all_pass` and `production_ready` remain false.

The sole candidate, seed streams, full 30-task schedule, two fresh controls, per-unit readouts, checkpointing, and exact paired verdict implementation are ready. NaP-versus-SNA/KKT1 remains OPEN_COMPARISON. No GPU production was ported to CPU or reduced. See analysis/nap_cifar5p1_1018/README.md for commands.
