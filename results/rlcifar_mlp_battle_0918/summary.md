# rlcifar_mlp_battle_0918 -- RL-CIFAR x MLP activation battle

box: {"lr": 0.001, "n_tasks": 50, "epochs_per_task": 400, "steps_per_task": 30000, "dims": [3072, 100, 100, 10], "optimizer": "adam", "engine": "stacked baddbmm, autograd, elementwise Adam (spec \u00a72.2), one step captured as a CUDA graph", "device": "cuda"}

## windows (median over seeds; * = collapsed, d = diverged runs)

| arm | raw | std | std-raw | raw drop | std drop | raw memo min | collapse raw/std |
|---|---|---|---|---|---|---|---|
| `SNA` | 0.9549 | 0.9870 | +0.0319 (STD_HELPS) | -0.0245 | -0.0011 | 0.9188 | 0/0 |
| `KKA` | 0.9535 | 0.9871 | +0.0338 (STD_HELPS) | -0.0211 | -0.0011 | 0.8575 | 0/0 |
| `KKA23` | 0.9484 | 0.9878 | +0.0393 (STD_HELPS) | -0.0143 | -0.0004 | 0.8129 | 0/0 |
| `KKT1` | 0.9565 | 0.9871 | +0.0307 (STD_HELPS) | -0.0221 | -0.0009 | 0.9542 | 0/0 |
| `R` | 0.1132* | 0.1132* | +0.0000 (TIE) | +0.0032 | +0.1572 | 0.1067 | 10/10 |
| `LK001` | 0.3881* | 0.8967 | +0.5118 (STD_HELPS) | +0.0509 | +0.0665 | 0.5721 | 10/0 |
| `LR` | 0.7264 | 0.9655 | +0.2390 (STD_HELPS) | +0.0493 | +0.0139 | 0.7671 | 0/0 |
| `LK03` | 0.8447 | 0.9789 | +0.1345 (STD_HELPS) | +0.0214 | +0.0020 | 0.7446 | 0/0 |
| `SL` | 0.7229 | 0.9645 | +0.2421 (STD_HELPS) | +0.0437 | +0.0157 | 0.7379 | 0/0 |
| `RSL` | 0.8653 | 0.9404 | +0.0754 (STD_HELPS) | +0.0151 | +0.0178 | 0.9450 | 0/0 |
| `ELU` | 0.1011* | 0.1005* | -0.0000 (TIE) | +0.1590 | +0.2166 | 0.0904 | 10/10 |
| `SILU` | 0.1233* | 0.1132* | -0.0098 (STD_HURTS) | +0.0036 | +0.1206 | 0.1163 | 10/10 |
| `GELU` | 0.1160* | 0.1132* | -0.0029 (STD_HURTS) | +0.0017 | +0.1375 | 0.1092 | 10/10 |

## registered calls

- **A (raw)**: `SNA_BEATEN` — beaten by ['KKT1']
    - SNA - KKA: median +0.0021, 6/10 seeds, p=0.7539 → TIE
    - SNA - KKA23: median +0.0058, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - KKT1: median -0.0018, 1/10 seeds, p=0.0215 → B_WINS
    - SNA - R: median +0.8417, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LK001: median +0.5667, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LR: median +0.2291, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LK03: median +0.1103, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - SL: median +0.2317, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - RSL: median +0.0895, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - ELU: median +0.8544, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - SILU: median +0.8313, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - GELU: median +0.8389, 10/10 seeds, p=0.0020 → A_WINS
- **A (std)**: `SNA_BEATEN` — beaten by ['KKA', 'KKA23']
    - SNA - KKA: median -0.0002, 0/10 seeds, p=0.0020 → B_WINS
    - SNA - KKA23: median -0.0009, 0/10 seeds, p=0.0020 → B_WINS
    - SNA - KKT1: median -0.0001, 2/10 seeds, p=0.1094 → TIE
    - SNA - R: median +0.8738, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LK001: median +0.0901, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LR: median +0.0213, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - LK03: median +0.0078, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - SL: median +0.0225, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - RSL: median +0.0467, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - ELU: median +0.8863, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - SILU: median +0.8738, 10/10 seeds, p=0.0020 → A_WINS
    - SNA - GELU: median +0.8738, 10/10 seeds, p=0.0020 → A_WINS
- **main**: `NOT_BOTH`
- **K1 (raw)**: `PERIOD_FREE` — KKA - SNA median -0.0021, 4/10 seeds, p=0.7539
- **K1 (std)**: `PERIOD_HURTS` — KKA - SNA median +0.0002, 10/10 seeds, p=0.0020
- **K2 (raw)**: `SCALE_MATTERS_DOWN` — KKA23 - KKA median -0.0051, 1/10 seeds, p=0.0215
- **K2 (std)**: `SCALE_MATTERS_UP` — KKA23 - KKA median +0.0007, 10/10 seeds, p=0.0020
- **K3 (raw)**: `TAIL_FREE` — KKT1 - KKA median +0.0031, 8/10 seeds, p=0.1094
- **K3 (std)**: `TAIL_FREE` — KKT1 - KKA median -0.0001, 3/10 seeds, p=0.3438

## course of the snake arms (median over seeds)

| arm/cond | t | online | seat l1 | seat l2 | mob l1 | mob l2 | W l1 | W l2 | 2aW l1 |
|---|---|---|---|---|---|---|---|---|---|
| SNA_raw | 1 | 0.8529 | -0.82 | -0.25 | 1.10 | 0.87 | 7.2 | 10.1 | 1.20 |
| SNA_raw | 10 | 0.9417 | -0.92 | -1.42 | 0.68 | 0.54 | 12.6 | 30.3 | 1.20 |
| SNA_raw | 50 | 0.9585 | -1.12 | -1.42 | 0.62 | 0.54 | 14.5 | 35.4 | 1.20 |
| SNA_std | 1 | 0.9772 | -0.12 | -0.62 | 0.93 | 0.72 | 13.9 | 15.5 | 1.20 |
| SNA_std | 10 | 0.9870 | -0.20 | -1.04 | 0.90 | 0.60 | 27.1 | 40.3 | 1.20 |
| SNA_std | 50 | 0.9867 | -0.41 | -1.12 | 0.81 | 0.57 | 34.2 | 49.4 | 1.20 |
| KKA_raw | 1 | 0.8776 | -0.01 | -0.20 | 1.16 | 0.93 | 5.3 | 9.3 | 1.20 |
| KKA_raw | 10 | 0.9423 | -0.91 | -1.03 | 0.74 | 0.58 | 10.8 | 30.6 | 1.20 |
| KKA_raw | 50 | 0.9547 | -1.09 | -1.35 | 0.65 | 0.54 | 11.9 | 32.5 | 1.20 |
| KKA_std | 1 | 0.9785 | -0.13 | -0.61 | 0.95 | 0.74 | 11.0 | 12.0 | 1.20 |
| KKA_std | 10 | 0.9873 | -0.22 | -0.99 | 0.92 | 0.62 | 23.9 | 37.5 | 1.20 |
| KKA_std | 50 | 0.9872 | -0.44 | -1.07 | 0.81 | 0.59 | 30.4 | 45.9 | 1.20 |
| KKA23_raw | 1 | 0.8884 | +0.04 | -0.16 | 0.78 | 0.63 | 6.4 | 9.4 | 1.20 |
| KKA23_raw | 10 | 0.9418 | -1.01 | -1.13 | 0.46 | 0.38 | 14.4 | 35.2 | 1.20 |
| KKA23_raw | 50 | 0.9517 | -1.10 | -1.43 | 0.41 | 0.36 | 16.2 | 42.5 | 1.20 |
| KKA23_std | 1 | 0.9801 | -0.09 | -0.54 | 0.65 | 0.51 | 14.7 | 13.2 | 1.20 |
| KKA23_std | 10 | 0.9883 | -0.17 | -1.07 | 0.63 | 0.40 | 31.5 | 45.8 | 1.20 |
| KKA23_std | 50 | 0.9880 | -0.33 | -1.08 | 0.57 | 0.39 | 40.7 | 62.8 | 1.20 |
| KKT1_raw | 1 | 0.8658 | -0.34 | -0.15 | 1.18 | 0.94 | 5.7 | 8.4 | 1.20 |
| KKT1_raw | 10 | 0.9445 | -1.05 | -1.23 | 0.67 | 0.55 | 11.4 | 29.0 | 1.20 |
| KKT1_raw | 50 | 0.9605 | -1.12 | -1.30 | 0.62 | 0.55 | 14.2 | 35.0 | 1.20 |
| KKT1_std | 1 | 0.9788 | -0.12 | -0.62 | 0.94 | 0.72 | 12.2 | 13.5 | 1.20 |
| KKT1_std | 10 | 0.9871 | -0.20 | -1.01 | 0.90 | 0.60 | 26.4 | 39.9 | 1.20 |
| KKT1_std | 50 | 0.9871 | -0.42 | -1.12 | 0.81 | 0.57 | 33.2 | 48.7 | 1.20 |

## phase: share of 2 alpha_i z outside the kunekune band (median over seeds)

| arm/cond | task/layer | above +pi/2 | below -3pi/2 | median theta |
|---|---|---|---|---|
| SNA_raw | t1_l1 | 0.164 | 0.054 | -0.84 |
| SNA_raw | t1_l2 | 0.110 | 0.006 | -0.29 |
| SNA_raw | t10_l1 | 0.061 | 0.015 | -0.96 |
| SNA_raw | t10_l2 | 0.017 | 0.011 | -1.36 |
| SNA_raw | t25_l1 | 0.053 | 0.014 | -1.07 |
| SNA_raw | t25_l2 | 0.017 | 0.010 | -1.33 |
| SNA_raw | t50_l1 | 0.048 | 0.010 | -1.11 |
| SNA_raw | t50_l2 | 0.016 | 0.010 | -1.39 |
| SNA_std | t1_l1 | 0.088 | 0.000 | -0.12 |
| SNA_std | t1_l2 | 0.038 | 0.003 | -0.61 |
| SNA_std | t10_l1 | 0.078 | 0.000 | -0.19 |
| SNA_std | t10_l2 | 0.015 | 0.006 | -0.99 |
| SNA_std | t25_l1 | 0.068 | 0.001 | -0.28 |
| SNA_std | t25_l2 | 0.015 | 0.006 | -1.02 |
| SNA_std | t50_l1 | 0.060 | 0.001 | -0.39 |
| SNA_std | t50_l2 | 0.014 | 0.006 | -1.06 |
| KKA_raw | t1_l1 | 0.186 | 0.031 | -0.23 |
| KKA_raw | t1_l2 | 0.099 | 0.004 | -0.21 |
| KKA_raw | t10_l1 | 0.050 | 0.018 | -1.02 |
| KKA_raw | t10_l2 | 0.024 | 0.008 | -1.03 |
| KKA_raw | t25_l1 | 0.036 | 0.015 | -1.17 |
| KKA_raw | t25_l2 | 0.019 | 0.010 | -1.23 |
| KKA_raw | t50_l1 | 0.031 | 0.014 | -1.16 |
| KKA_raw | t50_l2 | 0.016 | 0.009 | -1.30 |
| KKA_std | t1_l1 | 0.085 | 0.001 | -0.13 |
| KKA_std | t1_l2 | 0.038 | 0.003 | -0.56 |
| KKA_std | t10_l1 | 0.077 | 0.001 | -0.20 |
| KKA_std | t10_l2 | 0.017 | 0.006 | -0.92 |
| KKA_std | t25_l1 | 0.066 | 0.001 | -0.31 |
| KKA_std | t25_l2 | 0.017 | 0.005 | -0.95 |
| KKA_std | t50_l1 | 0.056 | 0.001 | -0.42 |
| KKA_std | t50_l2 | 0.016 | 0.005 | -1.04 |
| KKA23_raw | t1_l1 | 0.182 | 0.030 | -0.20 |
| KKA23_raw | t1_l2 | 0.099 | 0.003 | -0.14 |
| KKA23_raw | t10_l1 | 0.042 | 0.020 | -1.09 |
| KKA23_raw | t10_l2 | 0.022 | 0.008 | -1.08 |
| KKA23_raw | t25_l1 | 0.031 | 0.012 | -1.19 |
| KKA23_raw | t25_l2 | 0.018 | 0.009 | -1.20 |
| KKA23_raw | t50_l1 | 0.028 | 0.011 | -1.18 |
| KKA23_raw | t50_l2 | 0.015 | 0.010 | -1.39 |
| KKA23_std | t1_l1 | 0.090 | 0.000 | -0.10 |
| KKA23_std | t1_l2 | 0.046 | 0.003 | -0.48 |
| KKA23_std | t10_l1 | 0.083 | 0.000 | -0.15 |
| KKA23_std | t10_l2 | 0.015 | 0.007 | -0.99 |
| KKA23_std | t25_l1 | 0.074 | 0.001 | -0.22 |
| KKA23_std | t25_l2 | 0.015 | 0.006 | -1.01 |
| KKA23_std | t50_l1 | 0.066 | 0.001 | -0.30 |
| KKA23_std | t50_l2 | 0.015 | 0.006 | -1.02 |
| KKT1_raw | t1_l1 | 0.169 | 0.047 | -0.52 |
| KKT1_raw | t1_l2 | 0.106 | 0.004 | -0.20 |
| KKT1_raw | t10_l1 | 0.040 | 0.021 | -1.12 |
| KKT1_raw | t10_l2 | 0.020 | 0.009 | -1.19 |
| KKT1_raw | t25_l1 | 0.030 | 0.020 | -1.17 |
| KKT1_raw | t25_l2 | 0.018 | 0.009 | -1.23 |
| KKT1_raw | t50_l1 | 0.031 | 0.018 | -1.14 |
| KKT1_raw | t50_l2 | 0.016 | 0.009 | -1.28 |
| KKT1_std | t1_l1 | 0.088 | 0.001 | -0.12 |
| KKT1_std | t1_l2 | 0.037 | 0.003 | -0.60 |
| KKT1_std | t10_l1 | 0.078 | 0.001 | -0.19 |
| KKT1_std | t10_l2 | 0.016 | 0.006 | -0.98 |
| KKT1_std | t25_l1 | 0.068 | 0.001 | -0.29 |
| KKT1_std | t25_l2 | 0.015 | 0.006 | -1.02 |
| KKT1_std | t50_l1 | 0.058 | 0.001 | -0.40 |
| KKT1_std | t50_l2 | 0.014 | 0.006 | -1.08 |

## prediction score (spec §6, Claude)

hits 8/19, Brier 0.230

| key | claim | p | hit |
|---|---|---|---|
| P1 | raw: A == SNA_TOP | 0.80 | no |
| P2 | raw: R collapses | 0.95 | yes |
| P3 | raw: LK001 collapses | 0.75 | yes |
| P4 | raw: LR collapses | 0.65 | no |
| P5 | raw: LK03 collapses | 0.50 | no |
| P6 | raw: SL collapses | 0.60 | no |
| P7 | raw: RSL collapses | 0.50 | no |
| P8 | raw: ELU collapses | 0.70 | yes |
| P9 | raw: SILU collapses | 0.70 | yes |
| P10 | raw: GELU collapses | 0.70 | yes |
| P11 | std: A == SNA_TOP | 0.45 | no |
| P12 | std: R collapses | 0.60 | yes |
| P13 | SNA_WINS_BOTH | 0.40 | no |
| P14 | K1 == PERIOD_FREE in both conditions | 0.60 | no |
| P15 | K2 == SCALE_FREE in both conditions | 0.60 | no |
| P16 | K3 == TAIL_FREE in both conditions | 0.50 | yes |
| P17 | raw: median window order SNA ~ KKA ~ KKT1 > KKA23 > SL,RSL > LK03 > LR > LK001 > SILU,GELU > ELU > R (checked as: the four snake arms are the top four and R is last) | 0.40 | no |
| P18 | every leaky arm (LK001, LR, LK03) is STD_HELPS | 0.50 | yes |
| P19 | SNA is TIE between raw and std | 0.45 | no |

## prediction score (spec §6, Issa)

hits 1/4

| key | claim | hit |
|---|---|---|
| I1 | A == SNA_TIED_TOP in both conditions | no |
| I2 | K1 in (PERIOD_FREE, PERIOD_HURTS) in both conditions | yes |
| I3 | K2 == SCALE_FREE in both conditions | no |
| I4 | K3 == TAIL2_BETTER in both conditions | no |
