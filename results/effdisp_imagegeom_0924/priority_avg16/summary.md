# Priority avg16 versus rgb32

Separate priority analysis of the completed rgb32 main arm and separately launched avg16 arm.
This is not the registered 50-series final image-geometry verdict.

Both conditions use the same five image IDs and task-label schedules per seed, 50 tasks, 30000 updates/task and frozen training source.
The avg16 transform removes spatial information; its native fan-in initialization does not match the RGB initial function. The same learning rate also acts in different optimizer coordinates. Paired ratios are descriptive and do not isolate input dimension alone.
raw_W_RMS = ||W1||F / sqrt(100*d) removes the elementary sqrt(d) factor from comparing raw Frobenius norms. The effective R and D use each condition's own fixed input covariance.
Rstar_local = D_eff/(2|c|) is reported only when c<0; R/Rstar is a local diagnostic, not proof of convergence.
Early is task6–15, late is task41–50. Window values are medians within seed; avg16/rgb32 ratios pair seeds before the five-seed median and bootstrap CI (5000 resamples, RNG924).
`checkpoint_points.csv` gives task-end R, raw Frobenius and raw RMS at task0/10/20/30/40/50; `checkpoint_pairs.csv` pairs the same seed and task. These trajectories show observed approach only, not an inferred limiting value.
P1–P3 use the prior fixed-covariance thresholds separately per condition; P3 calibrates on task6–25 and predicts task26–50. Group PASS requires four of five seeds. P2 is a finite task41–50 plateau criterion; even a pass does not establish asymptotic convergence or a fixed-point height.

## P1–P3 per condition

geometry optimizer question  n_seed  n_pass  n_uninformative  n_low_response       verdict
   avg16      adam       P1       5       5                0               0          PASS
   avg16      adam       P2       5       1                0               0          FAIL
   avg16      adam       P3       5       0                0               0          FAIL
   avg16       sgd       P1       5       5                0               0          PASS
   avg16       sgd       P2       5       4                0               0          PASS
   avg16       sgd       P3       5       0                0               0          FAIL
   rgb32      adam       P1       5       5                0               0          PASS
   rgb32      adam       P2       5       0                0               0          FAIL
   rgb32      adam       P3       5       0                0               0          FAIL
   rgb32       sgd       P1       5       5                0               0          PASS
   rgb32       sgd       P2       5       4                0               0          PASS
   rgb32       sgd       P3       5       0                4               0 UNINFORMATIVE

## Paired late-window metrics

optimizer window             metric statistic  n_seed  n_valid  paired_seed_median  ci95_low  ci95_high
     adam   late                  R     ratio       5        5            0.333933  0.331372   0.337613
     adam   late R_over_Rstar_local     ratio       5        5            0.961171  0.928783   1.052353
     adam   late        Rstar_local     ratio       5        5            0.344757  0.322771   0.360100
     adam   late                  c     ratio       5        0                 NaN       NaN        NaN
     adam   late              d_eff     ratio       5        5            0.314537  0.305099   0.330869
     adam   late            raw_D_F     ratio       5        5            0.564091  0.549408   0.570073
     adam   late            raw_W_F     ratio       5        5            0.504082  0.501225   0.507281
     adam   late          raw_W_RMS     ratio       5        5            1.008165  1.002451   1.014562
     adam   late   top10_Q_fraction     ratio       5        5            1.048551  1.030342   1.072504
     adam   late          train_acc     ratio       5        5            0.916388  0.901715   0.920547
      sgd   late                  R     ratio       5        5            0.657824  0.651017   0.673814
      sgd   late R_over_Rstar_local     ratio       5        5            0.926200  0.831758   1.171628
      sgd   late        Rstar_local     ratio       5        5            0.706943  0.554013   0.811906
      sgd   late                  c     ratio       5        0                 NaN       NaN        NaN
      sgd   late              d_eff     ratio       5        5            0.510445  0.487734   0.517256
      sgd   late            raw_D_F     ratio       5        5            0.722603  0.709413   0.743612
      sgd   late            raw_W_F     ratio       5        5            0.817393  0.808017   0.833113
      sgd   late          raw_W_RMS     ratio       5        5            1.634787  1.616033   1.666225
      sgd   late   top10_Q_fraction     ratio       5        5            1.081490  1.079892   1.112355
      sgd   late          train_acc     ratio       5        5            0.645449  0.632919   0.656537
