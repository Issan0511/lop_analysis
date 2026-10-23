# Interpretation limits and secondary observations

The registered labels concern the finite-budget treatment effect of a first-layer norm clamp introduced after task10. They do not identify an activation's intrinsic sensitivity to weight growth.

- PM reference online accuracy is already near99%, and endpoint training accuracy is100%. This matched-sample PM setting has a ceiling and differs from the historical10000-image single-pass PM benchmark.
- RL ELU has already degraded before intervention: mean endpoint train accuracy across3seeds is98.83% at task1,100% at task5,31.61% at task10,11.11% at task50. The registered early11-20 online mean11.71% lies near chance. A small late-clamp response cannot show that ELU weight growth is harmless; the intervention is late relative to the collapse.
- RL leaky reference early endpoint84.66% is also flagged BUDGET_LIMITED. Every paired contrast remains included. This result concerns80epochs/task, not the previous400-epoch RL protocol.
- Negative D_RL_ELU=-0.197pp does not mean the clamp lowers late accuracy: late online accuracy is10.365% with clamp versus10.222% without. D is the difference in early-to-late degradation, so an early benefit that exceeds the late benefit gives negative D.
- Registered secondary perunit metrics locate a large layer2 change. For RL ELU reference, layer2 low-gate occupancy(phi'<.05; average over inputs/units/seeds) goes0.00809 at task1 to0.98696 at task10 to0.99997 at task50; mean layer2 preactivation goes0.298,-43.25,-154.13. Mean W1 centered norm goes3.38,10.64,13.31 over those same tasks. These are descriptive associations, not an intervention establishing sinking causality.
- Only W1 is projected. This protocol does not test whether preventing layer2 sinking or weight growth before collapse rescues ELU.

Source: rows.csv, levels.csv, paired.csv, verdict.csv and units.npz. selected_task_diagnostics.csv is a secondary summary of registered metrics at tasks1,5,10,20,50; taskselection is for descriptive display, not a new inferential endpoint.
Completeness:24models x50tasks=1200taskrows;7200learning-curve rows;700unit arrays. All arrays finite. Warmup paired parameter/moment maximum difference0; largest clamp norm relative error2.384e-7. Training155.48seconds on RTX5060Ti.
