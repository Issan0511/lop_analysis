# ELU Q audit: meaning, sign and relation to readout/input growth

Status: post-hoc audit of existing data; analysis plan committed before this audit's numerical evaluation. This is not a preregistered training experiment or a prospective test. Date: 2026-09-29 JST.

## Question

Does Q = d(||v||^2 - ||w||^2 - b^2)/dt add an independently interpretable diagnostic for the ELU/readout discussion, and does it predict the actual Adam trajectory? Do not assume that the ratio tends to one, or that an algebraic decomposition defines independent forces.

## Sources and scope

Primary stored states: obsidian-research-data/valley_anchor_1layer_0928/runs/{ELU,LR}_s{0,1}.npz. Tasks 10,20,30,40; switch, +200 and end (4000 updates). N=1200, H=100, output classes=10. End states include W2 and labels; other phases need only saved z and D=e W2. These are states trained with Adam; evaluating a Euclidean gradient at them does not reproduce Adam.

Inspect schemas of other existing files only to identify available norms/moments. A matching norm trajectory may be joined only after identical source configuration and numerical agreement of overlapping z states are checked. Missing W1 or optimizer moments must remain a reported limitation; do not substitute a stationary RMS approximation for actual Adam.

## Fixed calculations

1. Mathematical ELU: phi=z for z>0 and exp(z)-1 otherwise; gate=1 or exp(z). With unit demand D=v^T e and eta=1 for rates, Sv=-2 mean(D phi), Sr=-2 mean(D z gate), Q=Sv-Sr=2 mean(D r), r=z gate-phi. LR(.1) is a homogeneous negative control. Check roundoff residuals, not a hypothesis test.
2. Interpret Q using the fixed-product parameter path v(s)=exp(s)v, u(s)=exp(-s)u (u includes hidden bias). Its loss slope is -Q/2. Verify by central differences at s=1e-4 and 1e-5. Only end states with recoverable probabilities are used for loss differences. Check recovery residuals, conditioning, probability bounds and normalization.
3. Compare a path additionally changing output bias c(s)=c+(exp(s)-1)v to preserve the deeply saturated contribution. Define Qdc=2 mean(D)=2 v^T mean(e), Qcomp=Q-Qdc. Its slope is -Qcomp/2. This is a different, explicitly specified direction, not a coordinate-free invariant. Check finite differences and report sign changes and magnitude ratios.
4. Stratify all units and alive units (at least one z>0); keep seeds and phases separate. Report signed medians, positive fractions, contributions z<=-3 versus -3<z<=0, and cancellation magnitudes. Do not infer net sign from the sign of individual terms or from a median of another quantity.
5. At fixed end-state input/output weights, minimize CE over output bias only (centered class gauge). Recompute Q and report bias-gradient residual, loss change, and changes in Q's sign and size. This is a local diagnostic intervention, not a resumed training run. A bias-balanced Q is not assumed to equal the old Qcomp because probabilities change.
6. Where available, compare end-state Q to later actual norm changes only as a descriptive association. Endpoint gradients are not integrated gradients; lack of agreement cannot by itself identify a particular optimizer mechanism.
7. Include the full norm increment identity 2 theta dot delta + ||delta||^2 when interpreting finite Adam updates. A transverse step does not imply a constant norm.

## Deliverables and limits

Save source hashes, per-state/unit outputs, numerical checks, summary.md, and reproducible analysis code. Report the finite number of seeds and the within-task windows. Do not declare a closed dynamical law, an asymptotic limit, a natural-trajectory causal decomposition, or exact Adam verification without the required observations.
