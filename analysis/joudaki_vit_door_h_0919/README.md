# Joudaki ViT / Tiny ImageNet: door H

Specification: `specs/spec_joudaki_vit_door_h_0919.md` (addendum 10 to the 13-activation
battle's spec). No main-run results have yet been inspected.

This is a **separate experiment** from `analysis/joudaki_vit_battle_0919/`, not an
extension of it. The battle's report refuses runs whose `source_hashes` differ, and
`source_hashes()` covers every `.py` in the experiment directory, so adding arms to the
battle would have invalidated the 50 runs it had already completed. That package is left
byte-identical; this one is a copy with the door added. Spec §0 records the decision.

Four arms — `R`, `GELU`, `RH`, `GH` — times seeds 0–9, 40 tasks of 500 updates: 40 runs.
`R` and `GELU` are re-run here so every run in the final table shares one source hash.
The battle's own `R`/`GELU` runs are not merged into it.

The door keeps one scalar per FFN channel:

    out = phi_raw(z) - M,   M <- .99 M + .01 mean_{batch x token}(phi_raw(z).detach())

`M` is a buffer, so no gradient reaches it and it rides in the state_dict — a task-boundary
resume restores the EMA instead of restarting it at zero. The same `M` is used in training
and evaluation; only training updates it, and only after `optimizer.step()`, never inside
the compiled forward. The mean is taken over `phi_raw` before the subtraction, on the same
sample axis as the adaptive `V`: batch times every token, CLS included.

`src/rlcifar_mlp_battle_0918.py` is frozen by the 0918 run's provenance and is not touched.
`RH`/`GH` are resolved in `model.py` to base arm plus `door_h`, so `make_act` never sees them.

Run from the repository root:

```bash
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
$PY -m analysis.joudaki_vit_door_h_0919.check_speed
$PY -m analysis.joudaki_vit_door_h_0919.checks --engine compile
$PY -m analysis.joudaki_vit_door_h_0919.check_report
$PY -m analysis.joudaki_vit_door_h_0919.launch
```

Preflight beyond the battle's own: `S-off` (door pinned to zero reproduces the base arm bit
for bit, via `--door-frozen`, which is smoke-only and refused in production), `S-H` (the door
drives the output DC to zero, against a tolerance derived from the EMA's `1/beta` gain rather
than a guessed constant, with a `2M` mutant required to separate by 100x), `S-grad` (the local
Jacobian is bit-equal to the base arm's and `M.grad` is None), `S-resume` and `S-compile` for
both door arms, and `S-nochange` (`R` and `GELU` run through this build and the battle build
agree bit for bit, so the two extra arms move nothing that already existed).

Registered family: `RH-R` and `GH-GELU`, Holm-adjusted across those two. `readouts.csv` carries
the descriptive secondary quantities from spec §6 — negative-preactivation mass and the channel
median of `z/sd(z)` per layer at tasks 1, 21 and 40. They are not decision endpoints; they exist
so that a null on `RH-R` still answers whether LN was already doing the door's job.
