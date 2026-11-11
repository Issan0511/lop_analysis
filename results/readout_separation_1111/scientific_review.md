# Independent scientific review: readout_separation_1111

Reviewed after completion and archive, before the final synthesis. This is a read-only interpretation audit of `summary.md`, `contrasts.csv`, and `seed_metrics.csv`, informed by the registered operators and the previously audited frozen assays. No new training, hypothesis test, confidence interval, or experiment condition was introduced. All effect sizes below concern the registered primary **U**, unless explicitly marked acute/late; `100 × ΔU` is percentage points of period-average **training** accuracy deficit, not endpoint accuracy or a percentage reduction in delay.

## Independent numerical checks

The saved coefficient dictionary for every estimable contrast was applied independently to the paired seed metrics. All **447** point estimates matched. All **92** nonestimable contrasts were first/sustained T90 contrasts containing a genuinely missing event time; none substituted the horizon for the event. This check did not regenerate bootstrap intervals. The reported 280 complete continuations, 2520 frozen records, 30 bit-exact legacy KC controls, and post-archive integrity checks are separate execution-validation results.

Within each model, seeds are the statistical units (10 ELU-MLP; 5 LN-ViT). The 10000-replicate 95% intervals are unadjusted, descriptive paired-seed bootstrap intervals. The many arms, examples, units, and probe cells are not additional independent replicates. Zero inside an interval is unresolved, not equivalence. The registered practical threshold is 1 pp of U; a signed interval need not exclude effects smaller than that threshold.

## Keep the contrast signs explicit

For U, lower is better. These orientations do not automatically carry over to the endpoint-accuracy metric.

| Contrast family | Saved difference | Interpretation of positive U difference |
| --- | --- | --- |
| Scale | scale 1 − scale 0.1 | Shrinking improves period-average accuracy |
| Direction | S − P or S − O | The named P/O intervention improves period-average accuracy |
| Memory | keep − clear | Clearing body Adam improves period-average accuracy |
| Forward / body gain | unmodified factor − modified factor, conditional on the other factor | The specified conditional policy improves period-average accuracy |
| Transmission | altered Jacobian − native | The altered Jacobian harms period-average accuracy |
| Parameterization | physical 0.1 − TG | TG is better; therefore the observed negative differences favor physical 0.1 |

Interaction signs must be read from the explicit coefficient dictionaries. In particular, `practical_positive` is a numerical positive-direction flag, not a universal benefit flag for every contrast family.

## ViT: P benefit and scale benefit are conditional on each other

At scale 1 with body memory kept, S−P is **+9.178 pp [5.292, 14.442]**, positive in all five seeds. The S scale benefit is **+6.797 [2.149, 11.466]**, but positive in only **4/5** seeds. The O scale benefit is **+7.283 [0.461, 14.140]**, also positive in 4/5. Thus neither S nor O supports an all-seed shrink benefit.

After P, further shrinking has estimated benefit **−1.424 [−3.753, 1.021]** with memory kept: four of five seed differences point toward worse U after shrinking, while the descriptive interval remains unresolved. With body memory cleared the P scale difference is **+0.884 [−0.156, 2.047]**, also unresolved. This is not evidence that shrinking is universally unnecessary, harmless, or equivalent after P. It is evidence that the benefit is strongly conditional: the keep-memory scale-by-P interaction is **+8.222 [2.707, 14.460]**.

S−O at scale 1 averages nearly zero (**−0.028 [−6.562, 3.750]**), but that masks heterogeneity. Four seed differences are +1.90 to +4.10 pp, whereas seed 10252013 is −12.86 pp. “O has no effect” or “S and O are equivalent” would overstate this result.

The crossed frozen map figure helps specify the local effect: at matched full probability vectors, P backward columns have smaller ViT trunk-gradient norms than S/O columns. It does not identify a unique long-run mediator. P changes both the Gram matrix and its other relationships to data/residuals; O preserves WᵀW while changing class-space directions, predictions and label relationships. The difference between P and O is not an isolated intervention on one scalar notion of alignment, nor proof that Gram change alone explains P's learning benefit.

## ELU: a large scale effect survives all tested readout directions

With body memory kept, shrinking improves full-horizon U by **+19.461 pp** for S, **+19.479** for P, and **+19.799** for O; all ten paired seed differences are positive in each case. At scale 1 the S−P and S−O full-U intervals are unresolved. At scale 0.1 S−P is slightly negative (**−0.746 [−1.376, −0.110]** with memory kept), so it should not be summarized as exact invariance to readout direction.

The important cross-model distinction is the pattern of the registered interventions, not a causal effect of “having LN.” The models differ in architecture, depth, activation, data/task history, optimizer state and horizon (4000 versus 1500 updates). Their first-200 windows represent different fractions of the primary averaging horizon. Directly subtracting their U effects would not isolate LN.

## Native transmission can be protective in these conditions

Setting the ELU derivative to 1 while retaining its forward output **worsens** full U: **+11.302 [10.827, 11.831]** at S scale 1 and **+2.148 [1.946, 2.367]** at scale 0.1. The result contradicts the simple claim that suppressive ELU derivatives alone cause the measured delay and that transmitting all gradients should rescue it. It does not establish that saturation is harmless in general.

For ViT at S scale 1, removing final-LN radial subtraction worsens U by **+10.853 [8.826, 12.880]**; removing both radial and mean subtractions gives **+10.680 [8.668, 12.737]**. Both harm directions occur in all five seeds. These are surrogate input-VJP changes with native forward outputs and final-affine gradients, not models with the LN layer removed.

At scale 0.1, full-U harm from radial removal is unresolved (**+4.355 [−2.035, 12.355]**), but acute harm remains **+3.294 [1.400, 4.683]**. “Shrinking removes the transmission effect” would therefore be too strong. The similar radial-only and radial-plus-mean point estimates are not an equivalence test or an independently established null effect of the mean term.

Dynamic Jacobian interventions change both gradient magnitude and direction. Norm-matched frozen assays show different Adam directions even after matching the trunk-gradient L2 norm, but this does not establish that direction mediates the long-run difference. The dynamic arms do not hold gradient norm, Adam displacement norm, features, or future probability vectors fixed.

## Adam memory is not uniformly harmful

For S at scale 1, keep−clear is negative: **−4.356 pp [−4.686, −4.016]** in ELU and **−5.990 [−6.601, −5.222]** in ViT. Clearing body state is harmful there. At scale 0.1, the same contrast reverses: **+6.156 [5.755, 6.595]** and **+2.557 [0.367, 4.768]**, favoring clear.

Every continuation already clears head Adam. These are conditional effects of clearing **body** m, v and step together, not a comparison of globally remembered versus entirely fresh optimizers, and not a separation of the individual Adam-state components. Frozen backward scaling also shows that a 0.1 gradient-norm ratio is not a 0.1 virtual-update ratio; this should not be generalized into exact Adam scale invariance.

## Early behavior and full-horizon behavior can have opposite signs

ELU S shrink with memory kept has a slightly unfavorable acute effect, **−0.564 pp [−0.977, −0.164]**, despite its **+19.461 pp** full-U benefit. P and O have the same acute-versus-full sign pattern. ELU body-gain-only G similarly has **−2.763 pp [−2.974, −2.546]** acute benefit (i.e. acute harm), but **+2.372 [2.076, 2.698]** full-U benefit. The earliest average accuracy or a single immediate gradient cannot stand in for later learning.

ViT forward contraction under P strongly worsens full U: forward-only T versus P has benefit **−21.001 pp [−28.412, −12.109]**. The same forward policy is beneficial in ELU P (**+8.850 [8.479, 9.231]**). This is not a universal pure-confidence effect: contraction is toward a fixed class-specific mean and can also change argmax decisions.

“First 200 updates” describes the policy duration, not the horizon of an effect labeled full U. The U_acute/U_late columns are secondary temporal partitions, not causal mediator shares. Their definitions intentionally use different step-200 evaluation policies.

## Release at 200 is an output-policy discontinuity

No weights or optimizer state are reset at release. Primary U uses the post-release evaluation at step 200; U_acute uses pre-release 200 and U_late begins with post-release 200. The step-199 and step-201 records help expose the transition. Because the primary trapezoid uses post-release 200 at the end of [199,200], it should not be described as exactly the weighted sum of the two differently defined secondary partitions.

For ViT T/TG, release increases mean CE by roughly **10.14–14.01**, while mean accuracy increases by **0.65–2.98 pp**. Thus the CE jump is not an equally large accuracy injury; the two metrics measure different changes. A post-release deficit may reflect both the learned state and removal of the forward policy. For G and Jacobian-only arms, native forward predictions do not jump just because the backward policy is removed.

## Physical 0.1 versus TG does not uniquely isolate head relearning

The observed physical-0.1−TG U differences favor physical scaling: **−16.169 pp [−17.144, −15.340]** in ELU S; **−8.096 [−10.983, −6.034]** in ViT S; **−23.479 [−32.413, −16.103]** in ViT P.

At the common initial state and matched body-memory setting, physical 0.1 and TG match p, body gradient and head gradient. With the same cleared head Adam state, the Adam formula consequently gives the same initial raw head-coordinate update, up to implementation rounding. The head-to-output sensitivity differs: physical-head changes pass through the ordinary forward map, whereas TG multiplies their forward effect by 0.1. This identifies an initial tenfold difference in the functional sensitivity to the same head-coordinate change.

It does **not** identify the entire U contrast as a tenfold head-learning-rate effect. After the first divergence, head/body parameters, probabilities and gradients interact; TG also releases its forward/body-gain policy after 200 updates, while the physical model continues ordinary learning from its scaled parameterization. Neither the future probabilities nor the head contribution are held fixed. The result supports the importance of parameterization and evolving readout learning, without assigning a unique causal share to that one initial multiplier.

## T90 and endpoint limits

All continuations ran to the fixed horizon. Endpoint accuracy is therefore measured at 4000 ELU or 1500 ViT updates, not at early stopping. T90 remains an observed-grid hitting time, rather than a continuously observed exact crossing time.

For ELU, all physical scale-1 S/P/O arms fail to reach 90% within 4000 updates, whereas all physical scale-0.1 S/P/O arms reach it. The unscaled surrogate T/G/TG arms also remain below that criterion over the available budget. Failure to reach the threshold is not proof of irreversible damage.

For ViT, radial-removal scale-1 arms have first T90 in 5/5 seeds but sustained T90 in only 4/5; seed 10252010 later falls below 90%. First and sustained events must remain distinct. “Sustained” here means all subsequent **observed** points until the horizon, not a fixed minimum persistence duration. In particular, seed 10252013, physical_S01_body_keep has sustained T90 = 1500 and sustained_observation_duration = 0. A final-only threshold crossing therefore qualifies under this registered definition.

Do not average only uncensored event times and present that as a full-cohort T90 difference. Keep censored contrasts nonestimable unless using a separately justified estimand; no such new analysis was added here.

## Synthesis wording recommended

The results separate several **conditional interventions and local maps**. They do not reduce readout growth harm to “LN versus no LN,” “alignment alone,” “confidence alone,” or an additive percentage decomposition. Useful concise claims are: the ViT benefit of P depends on scale and differs from the Gram-preserving O control; native transmission can protect learning in both models; body-memory effects reverse with physical scale; matching the initial p and body gradient does not match the later readout learning process.

## Final synthesis audit

The integrated draft correctly distinguishes local versus full-horizon results, P versus O, the current mean-logit-compensated scale intervention versus the earlier zero-head intervention, and U versus event-time outcomes. No numerical or contrast-sign blocker was found. The final integrated synthesis was re-read after clarification of the two boundaries below; both are now explicit, and no unresolved scientific blocker remains:

1. T preserves the class-specific mean logits **at the fork**. Its contraction center zbar0 is then fixed; once the ordinary logits evolve, the current mean logits are not generally preserved.
2. “Removing the downward mean shift” and “P recreates the downward action” refer to the frozen-state operator comparison, not a demonstrated elimination/recreation throughout the 200-update trajectory. For example, at frozen ELU S1/S1 with actual labels, native mean virtual-Adam shift is −0.006520 [−0.007259, −0.005799], whereas derivative-one gives +0.002441 [0.001127, 0.004077]. The corresponding dynamic policy is nevertheless harmful. Both observations should be stated without conflating their estimands.

## Reviewed source fingerprints

- `summary.md`: `fd399093459dbd1cf3fe71e3080fb3e91ff013f6220435b78d466cd95ca16362`
- `contrasts.csv`: `2e85548c3e55d3d83e9a705e105cca3726256471f7fa23473e1db45706860bfa`
- `seed_metrics.csv`: `16c2a5d0391dcee338ce9b202f5028260a5e7c284cc11fa7e17af2a9a65e5bc4`
- `probe_estimates.csv`: `263efc3e638137fd681589d1ffdf5de8e6a2c497512b02edeb786bfa3554f039`
- `synthesis.md` (final scientific review): `1fbbae0b496d8498cdcdd29709b3bef77b30d8a02ba544e1ed751632d53c0007`
