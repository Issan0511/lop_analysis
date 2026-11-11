# Preregistered follow-up: differentiate through the prescribed body mean

This addendum is written AFTER inspecting the54 original outcomes and BEFORE any follow-up continuation. Original P2/P3/P5 passed; original P4 matched-hard versus matched-uniform did not pass. V01UM had large top/gap growth and poor uniform/true-label losses. Its positive external bias corrections can oppose descent and alter the optimization dynamics. We retain that failed criterion and the original outputs. Do not relabel the original matched-uniform arm as an implementation bug or discard it.

Question: does persistent-label selective growth and its small-readout advantage survive a body constraint whose gradient is the exact derivative of the constrained model? This is an additional controlled model, not a natural-trajectory mediation estimate.

Use the same6 saved ELU states and the original V1 body reference at every step; all100units in all6states satisfy the original eligibility threshold. Parameterize

    z_ni(t)=w_i^T(x_n-xbar_Bi)+m_ref,i(t).

Thus hidden beta is determined by W and m_ref, NOT an independently optimized parameter. Differentiate through beta(W)=m_ref-W*xbar_B, so grad_W=delta^T X - sum_n(delta_ni)*xbar_Bi. This is the gradient of the actual constrained loss at each step. Body-reference changes still inject a known exogenous mean shift. Keep native float32 activation and output forward; compute the effective beta in double and cast to float32 to bound mean-matching error. Evaluate the actual constrained forward.

Four arms, each4000steps, seeds0,1,2 and tasks10,25 (24new continuations): C1 original fixed V/hard labels; C01 V*.1 with original mean-logit compensation/hard labels; C01U same small V/uniform targets; C01R same small V/fresh labels using the SAME private label array as original V01R. Output bias remains trainable with original Adam state. Clear W Adam state in all four arms at the fork to initialize the new coordinate system consistently; no trainable hidden bias or V. This reset differs from the original matched arms and is identical across all follow-up comparisons.

Initial body/activities must agree across four arms within float32 tolerance; C01/U/R must have identical initial parameters/optimizer and predicted logits. All use the same reference and batches. At step t use reference[t] for forward/gradient; after the update evaluate at reference[t+1]. Body match tolerance2e-5*max(1,abs(ref)); report measured maxima. No projection of the old unconstrained W gradient is used.

Same evaluation times and definitions as original spec. Save source states/endpoints, unit/body traces, and G,h/activity/opening, fixed-label CE/accuracy AND uniform-target loss. Plot/source analysis labeled follow-up. Trained source histories split exact constrained W gradients into uniform and label components. Next-step z contributions use (x-xbar_Bi)^T DeltaW and the prescribed Delta m_ref separately, with actual Adam denominators and arithmetic residual. Source probes at0,200,500,1000,2000; means across the latter four are descriptive.

F1 exact-gradient and coordinate checks: float64 autograd versus delta^T X-sum(delta)*xbar_B <1e-10, finite difference<1e-7; actual constrained forward and body path within float tolerance; fixed V exact. The uniform-target end loss must not exceed its start by more than.01 in each seed; failure flags continuing optimizer/forcing instability and precludes a clean label comparison.

F2 small-readout selective expansion under the same body path: at4000 C01 exceeds C1 in both G and median h in all3seeds, separately per task. Report absolute h/activity changes and within-unit body-down/top-up fractions. Matching assures equal means but not equal full distributions.

F3 persistent-label dependence: C01 exceeds BOTH C01U and C01R in G and median h in all3seeds per task, provided F1 stability passes. Report if either fails; no retuning or replacement after outcomes. Accuracy/margins distinguish useful fitting from amplitude growth.

This is a hypothesis refinement motivated by the failed original control; it is not independent discovery or validation of a preregistered original prediction. Additional budget24task equivalents, at most15wall minutes/30CPU minutes, two single-thread workers, raw<300MB. Commit addendum before synthetic/real smoke; commit implementation before main. Endpoints and failed criteria remain alongside the original54 curves. No manuscript edit.
