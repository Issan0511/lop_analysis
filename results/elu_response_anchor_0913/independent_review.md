# Independent response-anchor audit

## Pre-run code review

The committed response engine implements the four registered cells directly:
`A=F0K0`, `B=F0K1`, `C=F1K0`, and `D=F1K1`.  Its manual derivative uses
`ELU'(z+K*delta)`, and AF/BF restore all W1 and b1 values after every Adam step
while retaining their updated Adam moments.  I found no defect in the engine
formula, update arithmetic, freeze behavior, snapshot restoration, or online
step logging.

Two coverage gaps in the first synthetic validator were closed before launch.
The source-checkpoint validation directly checked both corrected layer-2
features and logits within F on all 1,200 inputs and the first scheduled
16-sample batch.  The independent synthetic audit constructed perturbations
whose z1, z2, and z2+delta arguments stayed more than 1 from zero and obtained
finite-difference relative error below 7e-7 in every F/K cell.
The preserved machine-readable pre-run record is
`independent_prerun_audit.json`; it is tied to the then-current auditor SHA256
`a818bc2f46379b4f91a6ab7034f384fbdc3be170a21c818f490c003bff62343d`.

The first runner draft had two critical pre-run issues: equality checks over
snapshots containing NaNs could falsely report mutation, and the anchor cache
was initially formed in a different matrix-multiplication geometry from the
18-model training batch.  Both were corrected before the runner was committed,
validated, or launched.  The launched runner commit was
`372cd4b85634761d709f3cafbddd79447c5069fc`; its runner SHA256 was
`ccda2f500db44c94f03c5d48f9c4a0e7abadb47eb8a818e19819465a3347db08`.
The pre-run validation then passed with exact full-1,200 within-F feature/logit
equality and maximum first-batch difference 1.068115234375e-4, inside the fixed
atol 2e-4 + rtol 1e-6 criterion.

## Post-run direct audit

The independent audit passed without finding an experiment defect.  It imports
neither the response engine, runner, nor report.  It reconstructs the anchored
forward, manual gradients, Adam decomposition, source RNG continuation, and
primary aggregation directly from raw artifacts.
This post-run audit used auditor SHA256
`9bd058c9a3dcef3a70eab84a6df8f11ddddca3afa2b28a99544861fa840e573a`.

- The fixed target IDs have past-only q>=.95 at tasks 16-20; the minimum saved q
  is 0.9891666769981384.  Source checkpoint, selections, units, RNG, spec, and
  engine hashes match their pinned values.
- The 18-model source z2 and both anchor formulas recompute exactly.  Initial
  source parameters and Adam states, within-F corrected features/logits, and
  unselected ordinary activations all have maximum absolute error 0.
- All 25 saved task/step states were checked.  Independent raw-gradient and Adam
  diagnostic maximum absolute error is 4.76837158203125e-7.  Replaying the
  actual step 0 to step 1 in each task differs by at most
  1.1920928955078125e-7.  Task-boundary state continuity is exact.
- AF/BF W1 and b1 equal their branch-start values in every saved state; maximum
  absolute error is 0.  The final checkpoint exactly matches the task25 step6000
  state and has Adam t=150000.
- The mechanism archive is finite, all 23 target-unit gathered arrays exactly
  equal their all-unit sources, and task-step interval arrays are zero at each
  task's step 0.

The task21 primary A-B loss benefits recomputed from every one of the 6,000
saved pre-update cross-entropies in float64 are 1.3261828400070468,
1.4174294878418245, and 1.3608590399697422 for seeds 0, 1, and 2.  Their mean is
1.3681571226062044; the seed SD is 0.04605902943937771 and the df=2 t95% interval
is [1.2537401506049246, 1.4825740946074841].  These values and the registered
`DIRECTIONAL_RESPONSE_SUPPORT` label match the separate aggregation exactly.

Machine-readable evidence is in `independent_audit.json`; the direct seed-level
losses and contrasts are in `independent_primary.csv`.
