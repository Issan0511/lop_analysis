# Matched-sample ELU environment comparison 0913

50 tasks × 6000 updates/task; fixed1200 images/seed; same 784-100-100-10 MLP and Adam in both environments.
PM: fresh pixel permutations and true labels. RL: fixed pixels and fresh iid random labels. Three seed replicas.
W1 centered-row norm projection starts task11 at task10 targets. Primary LoP=early(11–20) online accuracy minus late(41–50), in pp.
This PM is repeated-sample80-epoch PM, not the historical single-pass10000-image benchmark. RL uses80epochs, not historical400.

|Contrast|Seed0|Seed1|Seed2|Mean pp|95% t CI|Label|
|---|---:|---:|---:|---:|---|---|
|D_PM_ELU1|0.3017|0.2857|0.2860|0.2911|[0.2685, 0.3138]|ESTIMATE|
|D_PM_LR|0.4911|0.4882|0.5270|0.5021|[0.4485, 0.5557]|ESTIMATE|
|D_RL_ELU1|-0.2648|-0.0212|-0.3045|-0.1968|[-0.5778, 0.1841]|ESTIMATE|
|D_RL_LR|4.6248|4.4267|5.3669|4.8061|[3.5749, 6.0373]|ESTIMATE|
|I_PM|0.1895|0.2025|0.2409|0.2110|[0.1445, 0.2774]|INCONCLUSIVE|
|I_RL|4.8896|4.4479|5.6714|5.0030|[3.4639, 6.5420]|CONSISTENT_INTERACTION|
|J_PM_minus_RL|-4.7001|-4.2454|-5.4304|-4.7920|[-6.2770, -3.3069]|ENV_DEPENDENT|

D = LoP_ref−LoP_clamp; I = D_leaky−D_ELU; J = I_PM−I_RL.
Positive I means clamp reduces degradation more for leaky; positive J means that difference is larger in PM.
All inferential replication is across seeds. Task entries are repeated observations, not independent replicas.

|Environment|Activation|Iv|Early online %|Late online %|Early endpoint %|Late endpoint %|LoP pp|
|---|---|---|---:|---:|---:|---:|---:|
|PM|ELU1|ref|99.164|98.847|100.000|100.000|0.317|
|PM|ELU1|wclamp|99.202|99.176|100.000|100.000|0.026|
|PM|LR|ref|99.104|98.487|100.000|100.000|0.617|
|PM|LR|wclamp|99.170|99.055|100.000|100.000|0.115|
|RL|ELU1|ref|11.712|10.222|14.847|10.789|1.489|
|RL|ELU1|wclamp|12.051|10.365|15.836|11.050|1.686|
|RL|LR|ref|51.838|48.818|84.661|78.644|3.020|
|RL|LR|wclamp|53.442|55.228|88.003|89.014|-1.786|

Budget-limited references (early endpoint train accuracy<90%): RL/ELU1/seed0, RL/LR/seed0, RL/ELU1/seed1, RL/LR/seed1, RL/ELU1/seed2, RL/LR/seed2.
Budget flags retain every measured contrast: these are finite-budget learning effects, not claims about asymptotic capacity.
Three seeds and matched-sample construction bound the scope; a null interaction does not establish environment invariance.
Raw online CE and endpoint trajectories are in rows.csv/learning.csv; perunit distributions are in units.npz.
