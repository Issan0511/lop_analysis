# Readout separation: reporting commitments

This is a pre-result reporting template, written after implementation freeze and before inspecting complete main data. Fill the final report from all registered sources; this template is not a result.

1. Report the within-direction physical scale effect at S, P and O, paired by seed. A scale effect after P is not automatically a pure norm-mediated effect; P also changes the forward probabilities and geometry.
2. Contrast frozen probability banks at the same backward map, and backward maps at the same complete per-example probability vector. Keep actual next labels primary and uniform-label expected gradients secondary.
3. The forward policy can change the argmax because its anchor is class-specific; do not label it a pure confidence/temperature intervention. Report the initial200-update forward-only, body-gain-only and joint policies and their interaction, then distinguish physical0.1 versus joint surrogate. The latter starts with matching probabilities/body/head gradients but has different head parameterization and subsequent learning response.
4. Report unchanged-forward ELU/LN Jacobian interventions and their scale dependence. Explain gradient norms, norm-matched directions, virtual Adam angles and actual training delays separately. A larger norm alone does not prove harm.
5. Report body-memory keep−clear and interactions. All new continuation conditions clear head Adam; this conditions the claim, while previous head-memory results provide separate context.
6. Use raw W and effective Wdiag(gamma) carefully in ViT. The frozen gauge identity preserves logits/upstream VJP but not coordinate learning under an unchanged optimizer.
7. Distinguish effects that are supported, uncertain and not identified. Use paired-seed descriptive intervals and effect sizes. Do not equate intervals crossing zero with equivalence, or compare MLP/ViT update counts as identical time units.
8. No claim that LN alone causes all cross-architecture differences, no percentages assigning the total delay to independent mechanisms, no identity of microscopic mechanism from a common response to readout removal.

Final numerical evidence must cite `contrasts.csv`, `seed_metrics.csv`, the complete probe tables, all validation artifacts, frozen execution provenance and the raw backup manifest. The primary U integrates post-release200 output; acute and late partitions show the release's timing separately.
