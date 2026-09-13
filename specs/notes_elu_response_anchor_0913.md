# Response-anchor interpretation note, before execution

Date 2026-09-13. This supplements the limitations of prereg c4965d09c16534ceec23e074c11171404f8840ee without changing arms, outcomes, thresholds, horizon or technical tolerances. No substantive outcome has been run.

For B, a_B(theta,x)=phi(z_theta+delta)-phi(z0+delta)+phi(z0).
If the trainable argument later becomes deeply negative and phi(z_theta+delta) approaches -1, then
a_B approaches -1-phi(z0+delta)+phi(z0).
This residual can remain input-dependent even though the trainable ELU argument has re-saturated.
Thus re-saturation does NOT necessarily remove all corrected feature variation. This is a consequence of the fixed correction defined by the intervention, not a new intervention or label leakage.

Report the registered actual corrected-activation spread separately from the effective response gate q.
If B fits new labels after its trainable response re-saturates, consider the persisted fixed residual features and readout adaptation.
Do not call the anchored architecture permanently revived ordinary ELU. B-vs-A identifies the total consequence of replacing the selected response functions while matching INITIAL features; it does not isolate a gradient-only effect for the entire future trajectory.

F1K1 (D) has no fixed residual because the anchors cancel; it is ordinary one-shot lift under the registered parameter mapping.
C has a different fixed residual, also input-dependent. Effects need not be additive across F and K.
A short-window initial comparison and full learning trajectory are complementary. Their endpoints remain exactly as preregistered.

Upstream-fixed AF/BF still allow layer2 and output updates, so any effect there concerns that restricted network, not selected incoming weights alone.
W norms may diverge after intervention. Initial matching is not whole-trajectory control.
