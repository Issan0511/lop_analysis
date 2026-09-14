# ELU environment comparison 0913

Status: preregistration draft; no full training before this file is committed and pushed by coordinator.
Question: does the causal benefit of restricting centered first-layer weight growth differ between ELU and leaky, and does this interaction depend on task construction?

## Fixed protocol
- 3 seeds (0,1,2); 2 environments x 2 activations x 2 interventions = 24 trajectories.
- MLP 784-100-100-10; float32, default Linear uniform initialization using the historical pmnist_0905 init RNG. Shared initial parameters across all arms/environments for each seed.
- ELU alpha=1 at both hidden layers versus leaky ReLU negative slope 0.1.
- Adam lr=0.001, beta=(0.9,0.999), eps=1e-8, no regularizer, no parameter or optimizer reset.
- Same 1200 MNIST training images selected uniformly once per seed, with historical rl_subset RNG, raw pixels/255.
- PM: fresh iid pixel permutation each task, true digit labels. RL: identity pixels, fresh iid uniform labels for each image each task. The input image identities remain fixed in both environments.
- Exactly 50 tasks, 80 epochs/task, batch16 = 6000 updates/task. Shuffle once each epoch, with identical sample orders across environments and activations within a seed.
- This is a matched-sample task comparison. PM differs from historical PMNIST (10000 images/single pass) and RL differs from historical 400-epoch RL (here80). No claim of reproducing previous benchmark levels.
- Both control and clamp trajectories train identically through task10. At its end record each unit's centered W1 row norm. From task11, after each Adam update, replace W1 row by its current row mean plus its centered row scaled to exactly that recorded norm. W1 row means, biases, higher layers and Adam moments are not projected. This intervention is identical in both environments.
- Batched independent GPU models allowed: the minibatch CE is averaged across samples and SUMMED across models, preserving each model's gradient scale.

## Endpoints and contrasts
- Task-level primary endpoint: online accuracy, the mean PRE-update batch accuracy over all6000 updates. Also save online CE, endpoint train accuracy/CE, full-training-set learning curves at steps0,75,375,1500,3000,6000.
- Save centered W1 norms, perunit preactivation mean/std, mean gates, low-gate fraction(phi'<.05), near-zero occupancy(abs(z)<1), unit gate RMS at taskend; W2/W3 row norms as secondary.
- Early window tasks11-20; late41-50. LoP L=early online accuracy minus late online accuracy, in percentage points. Report absolute early/late levels and all-task mean.
- For each activation/environment/seed: D=L_ref-L_clamp. Positive D means clamp reduces degradation. Also late clamp-ref accuracy and online CE counterpart.
- Activation interaction I_env=D_leaky-D_ELU: positive means weight growth restriction helps leaky more. Environment interaction J=I_PM-I_RL.
- Unit of replication is seed, NOT task. For all primary paired contrasts report all3 seed values, mean, SD, 95% t interval(df2). Wide intervals with3seeds are expected.
- Descriptive threshold0.5pp: CONSISTENT_INTERACTION if all3 I_env share a sign and abs(mean)>=.5; ENV_DEPENDENT if all3 J share a sign and abs(mean)>=.5. Otherwise INCONCLUSIVE. These labels are pilot strength only, not broad equivalence or universal causal claims.
- Failure-to-fit flag: any activation/environment reference arm with early window mean endpoint train accuracy<.90 is BUDGET_LIMITED. Retain every measured contrast; interpret as learning within fixed budget, not asymptotic capacity. Also report endpoint ceiling fraction(acc>=.99) and warn primary endpoint interpretation if online accuracy>=.99.
- No adaptive sample size, budget adjustment, arm selection or changed windows after full run begins. Unexpected failures are recorded.

## Validation gates before full run
- CPU single-model autograd Adam versus batched GPU step on the same parameters, inputs and labels: gradients maxabs<2e-5; updated parameters maxabs<2e-5. Verify both activation derivatives with autograd.
- Batched model independence: sum across model losses yields same gradients as single-model reference (same tolerance).
- Graph/eager update equivalence over25steps, maxabs<5e-5; RNG/data hashes recorded. Float32/backend drift is not exact replay of old CPU experiments.
- Clamp centered norm relative error<2e-5 and rowmean absolute change<2e-6, other tensors unchanged by projection. Mutation of whole-row scaling must fail rowmean-preservation when nonzero.
- Matched input indices, minibatch orders, initialization, and unclamped warmup pairing verified. No nonfinite params/loss; record failures, never drop a seed.
- A synthetic/timing smoke is allowed before registration, but no substantive training on the50-task protocol.

## Outputs
src/elu_environment_0913.py; results/elu_environment_0913/{rows.csv,learning.csv,units.npz,paired.csv,verdict.csv,summary.md,validation.json,provenance.json,checkpoint.pt}.
Provenance includes actual prereg commit, spec/code/data SHA256, torch/device versions, deterministic/precision settings, parameters and wallclock.
