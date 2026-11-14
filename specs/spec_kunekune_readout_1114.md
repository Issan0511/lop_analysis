# kunekune_readout_1114: matched readout controls for the kunekune family

Preregistration, 2026-11-14. Requested by Issa in the chat 仮1〜4の査読結果を確認.
Commit this file before production histories or branches. Code checks and timing use
synthetic data and seed9001 only. No production outcome is used to select conditions.

## Question and scope

Do adaptive kunekune activations avoid readout-dependent adaptation costs, or retain
activation derivatives while still paying such a cost? Extend the no-LN MLP arm of
elu_specificity_1112/readout_separation_1111, not the ViT box. A small reset benefit
alone does not show health: compare reset bodies with fresh bodies and absolute
performance. This is a one-time readout intervention study, not Qall prevention.

## Fixed design

- Canonical stage1_ctrl_1111 inputs, initialization and private label/batch streams:
  MNIST raw/255,1200 examples,784-100-10,10 uniform random classes,batch16,
  4000 updates/task,Adam lr.001 betas(.9,.999) eps1e-8,CPU float32,one thread/job.
- Seeds0..9 paired across all activations and arms. Histories30 tasks. Probe saved
  task ends1,10,30 on their respective next task,4000 updates without early stop.
- Activations: KKA,KKA23,KKT1,SNA,ELU,R. ELU/R use canonical stage1 arithmetic,
  including ELU expm1(clamp_max), not F.elu. Replay archived controls must match.
- Adaptive family uses existing src/rlcifar_mlp_battle_0918.py SnakeFamily functions:
  KKA=kk,KKA23=kk23,KKT1=kkt1,SNA=snake; c=.6,beta=.01,alpha limits[.005,3].
  One unit-wise variance EMA starts at1; alpha=clip(.6/sqrt(EMA)), stop-gradient.
  After each optimizer step update EMA from that step's pre-update batch z,
  biased batch variance. Do not reset it at task switches or head interventions.
  Adaptation history is part of each activation treatment; no claim about shape alone.
- Arms at each saved state: N (native head and Adam); S1 (same head,head Adam clear);
  S01 (class-centered head*.1,mean-logit bias compensation,head Adam clear);
  P1 (independent per-column class permutation,compensation,head Adam clear);
  P01 (same P then centered*.1,compensation,head Adam clear);
  Z_keep (head weight/bias zero,head Adam clear,body Adam keep);
  Z_clear (head zero,all Adam clear); F0 (same seed's initial body and activation EMA,
  head zero,fresh Adam). All arms train all parameters afterwards; head is not fixed.
  All learned-body arms start with identical body parameters/EMA. All but Z_clear
  and F0 retain body Adam. P uses the exact prior ops.geometry_bank permutation.
- 60 histories and1440 full probe curves. No activation/seed exclusion based on
  performance. Divergence is reported; technical mismatches stop affected work.

## Outcomes and decisions

Primary at task-end10: period-average training accuracy deficit U, trapezoidal
integral over the exact previous grid {0,1,5,25,50,100,199,200,201,4000,100-step grid}.
Positive benefit means U(baseline)-U(intervention). Contrasts:
 D=U(S1)-U(P1); B_S=U(S1)-U(S01); B_P=U(P1)-U(P01);
 Z_S=U(S1)-U(Z_keep); Z_P=U(P1)-U(Z_keep);
 M=U(Z_keep)-U(Z_clear); H=U(N)-U(S1);
 R_body=U(Z_clear)-U(F0) (positive is a residual learned-body cost).
For each activation report seed-paired means and10000 paired-seed percentile
bootstrap intervals. Use one shared resampling matrix with fixed seed11142026.
Intervals are descriptive, unadjusted; do not label a family of tests significant.
Support an improvement only if its95% interval is wholly above0; wholly below0
means harm; otherwise unresolved, not equivalence. Also report whether the interval
lies within[-1,+1]pp as a separately named descriptive practical-size check.
Task ends1 and30 and activation-difference contrasts are secondary; report all.
Primary family comparisons are KKA/KKA23/KKT1 against ELU for B_P and R_body;
SNA is the related smooth reference. Do not sum nonadditive cause fractions.

Secondary metrics: early0..200 and late200..4000 U,endpoint accuracy/CE,
first T90 with integer correct_count>=1080; nonarrival is right-censored at4000,
never an observed T90=4000. Repeated baseline native curves show age-dependence.
Do not compare probabilities of reaching90 with the prior 4-consecutive protocol.

Diagnostics at0,25,200,4000: m_i,sd_i,r_i=m_i/sd_i; |phi'| mean,small(|phi'|<.05)
fraction, derivative input variance; activity centered RMS; head centered norm,
logit RMS,confidence,entropy; alpha/EMA; activity-increase demands and uniform-label
expected SGD mean-preactivation push (not Adam movement). On steps1,25,200,4000
also record actual head/body gradient and update norms and body mean-z movement.
Keep signed derivatives for gradient calculations, abs only for response diagnostics.
Unit/input counts are not independent seeds. Response-cost coexistence is descriptive;
not causal mediation. Report continuous diagnostics, no post-hoc healthy/dead cutoff.

## Registered competing expectations

E1: KKA/KKA23/KKT1 may keep derivatives yet have positive B_P or Z_P. If supported,
response preservation does not eliminate readout-dependent cost in this box.
E2: Small readout benefits may instead coexist with low U and reset bodies at least
as fast as fresh bodies. This is compatible with little detectable cost here.
E3: Small benefits with large R_body or failure after Z_clear indicate constrained
body recovery, not proof of harmless readouts. These are alternatives, not ranked
claims about which activation must win. ELU control's known t10 response is a replay
anchor, not an independent replication or a new successful prediction.

## Validation, resource bounds, reporting

Before production: check family forward/backward against independent piecewise
formula and original class; EMA timing; no EMA update during evaluation;
snapshot/replay restores weights,Adam,EMA; interventions preserve initial body/EMA,
column multisets/centered scales and mean logits; head/body Adam rules.
Validate control replay against archived stage1 task endpoints and existing t10
ELU continuation curves, without relaxing equality after inspecting differences.
After production: completeness/hash inventory; independently reload final states
and recompute endpoint accuracy,CE and response diagnostics; independently recompute
primary U and contrasts from curves. Plot all conditions, including adverse results.

Use at most12 CPU workers,one thread each,24GiB combined RSS target,5GiB raw target,
120000 worker-seconds ceiling and6h wall ceiling. No paid external worker or cloud.
If limits are reached, save checkpoints and report incomplete work; do not silently
change seeds/horizons. Native GPU is unavailable, so no CUDA arithmetic substitution.
Progress check no more often than5min unless expected completion sooner or an error.

Each source/fork records spec/code/data hashes,source state,labels and batch sequence.
Failures/smokes are kept separate. Save source snapshots and branch final state;
reuse of archived results is explicitly marked. Code/results go to the experiment
branch and main without rewriting history. Raw files are SHA-verified into
/home/issan/Projects/obsidian-research-data/kunekune_readout_1114/ before cleanup.
Save an Obsidian result note; do not edit the manuscript or launch further experiments.
