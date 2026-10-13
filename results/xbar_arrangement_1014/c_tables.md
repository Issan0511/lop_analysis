
### a01 seed 0 (alive units 82, lambda 12.82, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 5.0e-15, Ef 9.4e-16, sd_f 2.0e-14, mc_f 1.2e-15, Aw 3.6e-13, Eb 1.4e-15, sd_b 1.3e-14, Ar 3.9e-14, Er 1.1e-14

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.21 [0.11, 0.51] | 0.27 [0.15, 0.76] | -6.3 [-14.1, -2.7] | -10.6 [-23.6, -3.5] | 8.1 | 14.7 | 0.41 |
| probe | 0.59 [0.19, 0.88] | 0.75 [0.26, 1.02] | -3.4 [-6.1, -1.5] | -3.9 [-9.1, -0.9] | 8.2 | 15.8 | 0.40 |
| readout | 0.03 [-13.37, 1.30] | 0.08 [-0.33, 0.73] | -4.2 [-27.9, +16.4] | -9.4 [-30.6, -2.9] | 5.3 | 10.4 | 1.02 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | 0.32 [0.02, 0.56] | -0.10 | 0.82 | 0 |
| probe | 0.59 [0.04, 0.83] | -0.46 | 1.05 | 0 |
| readout | -0.24 [-3.26, -0.03] | -19.29 | 0.05 | 31 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 82/82 | 82/82 | 82/82 | 75/82 | 82/82 | 0/82 | 81/82 | 0/82 | 2.8 |
| probe | 82/82 | 82/82 | 82/82 | 74/82 | 79/82 | 3/82 | 64/82 | 0/82 | 3.4 |
| readout | 52/82 | 81/82 | 82/82 | 3/82 | 81/82 | 1/82 | 77/82 | 1/82 | 2.7 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 82 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.5 | 24.6 | 28.1 | 30.9 | 33.2 | 35.8 | 38.4 | 41.2 | 45.5 | 53.9 |
| abs(x~)^2 mean | 42.8 | 56.2 | 66.9 | 73.7 | 83.3 | 88.9 | 95.4 | 104.3 | 118.5 | 144.1 |
| e' mass frac (median) | 0.163 | 0.138 | 0.117 | 0.110 | 0.088 | 0.069 | 0.037 | 0.019 | 0.000 | 0.000 |
| e' mass frac (mean) | 0.233 | 0.161 | 0.141 | 0.133 | 0.108 | 0.085 | 0.048 | 0.045 | 0.028 | 0.020 |
| (d-a) mass frac (median) | 0.249 | 0.164 | 0.111 | 0.119 | 0.076 | 0.038 | 0.010 | 0.005 | 0.000 | 0.000 |
| (d-a) mass frac (mean) | 0.312 | 0.174 | 0.135 | 0.136 | 0.088 | 0.060 | 0.035 | 0.029 | 0.018 | 0.011 |
| M_nn rel (full) | 0.43 | 0.59 | 0.71 | 0.80 | 0.92 | 1.01 | 1.12 | 1.23 | 1.43 | 1.77 |
| row sum rel (full) | 0.37 | 0.61 | 0.78 | 0.87 | 1.01 | 1.09 | 1.20 | 1.22 | 1.36 | 1.51 |
| row sum / M_nn (full) | 0.07 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.08 | 0.07 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.43 | 0.60 | 0.72 | 0.80 | 0.92 | 1.01 | 1.12 | 1.23 | 1.42 | 1.76 |
| row sum rel (probe) | 0.38 | 0.62 | 0.79 | 0.87 | 1.01 | 1.09 | 1.19 | 1.22 | 1.35 | 1.49 |
| row sum / M_nn (probe) | 0.08 | 0.09 | 0.10 | 0.09 | 0.10 | 0.09 | 0.09 | 0.09 | 0.08 | 0.07 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.006 [0.001, 0.087])

### a03 seed 0 (alive units 100, lambda 54.52, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 5.3e-15, Ef 7.7e-16, sd_f 2.6e-14, mc_f 9.9e-16, Aw 1.3e-14, Eb 6.7e-16, sd_b 2.1e-14, Ar 5.1e-14, Er 1.0e-14

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.47 [0.34, 0.65] | 0.50 [0.37, 0.77] | -10.9 [-16.4, -4.8] | -20.9 [-27.6, -9.6] | 20.8 | 41.8 | 0.48 |
| probe | 0.70 [0.49, 0.87] | 0.72 [0.55, 0.92] | -6.3 [-8.6, -3.0] | -12.1 [-17.0, -5.2] | 21.1 | 42.3 | 0.49 |
| readout | 0.03 [-0.92, 10.81] | 0.03 [0.01, 0.14] | -26.4 [-49.5, +21.4] | -42.7 [-58.6, -16.0] | 27.7 | 43.8 | 0.89 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | 0.08 [-0.14, 0.40] | -0.27 | 0.72 | 0 |
| probe | 0.12 [-0.30, 0.52] | -1.06 | 0.84 | 0 |
| readout | -0.39 [-1.88, -0.11] | -91.11 | -0.02 | 20 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 100/100 | 100/100 | 100/100 | 63/100 | 100/100 | 0/100 | 100/100 | 0/100 | 2.7 |
| probe | 100/100 | 100/100 | 100/100 | 62/100 | 100/100 | 0/100 | 98/100 | 0/100 | 2.1 |
| readout | 81/100 | 100/100 | 100/100 | 0/100 | 100/100 | 0/100 | 100/100 | 0/100 | 2.5 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 100 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.5 | 24.6 | 28.1 | 30.9 | 33.2 | 35.8 | 38.4 | 41.2 | 45.5 | 53.9 |
| abs(x~)^2 mean | 42.8 | 56.2 | 66.9 | 73.7 | 83.3 | 88.9 | 95.4 | 104.3 | 118.5 | 144.1 |
| e' mass frac (median) | 0.101 | 0.104 | 0.107 | 0.106 | 0.107 | 0.106 | 0.095 | 0.089 | 0.079 | 0.059 |
| e' mass frac (mean) | 0.119 | 0.114 | 0.115 | 0.114 | 0.107 | 0.113 | 0.092 | 0.085 | 0.080 | 0.061 |
| (d-a) mass frac (median) | 0.161 | 0.128 | 0.129 | 0.108 | 0.104 | 0.100 | 0.081 | 0.072 | 0.059 | 0.033 |
| (d-a) mass frac (mean) | 0.180 | 0.144 | 0.128 | 0.113 | 0.101 | 0.095 | 0.075 | 0.068 | 0.059 | 0.036 |
| M_nn rel (full) | 0.49 | 0.64 | 0.75 | 0.83 | 0.93 | 1.00 | 1.09 | 1.20 | 1.37 | 1.70 |
| row sum rel (full) | 0.49 | 0.64 | 0.73 | 0.88 | 0.94 | 0.96 | 1.13 | 1.16 | 1.37 | 1.69 |
| row sum / M_nn (full) | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.09 | 0.09 | 0.09 | 0.09 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.49 | 0.64 | 0.75 | 0.83 | 0.93 | 1.00 | 1.09 | 1.20 | 1.37 | 1.70 |
| row sum rel (probe) | 0.50 | 0.64 | 0.73 | 0.88 | 0.94 | 0.96 | 1.13 | 1.16 | 1.37 | 1.68 |
| row sum / M_nn (probe) | 0.09 | 0.09 | 0.09 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.056 [0.006, 0.297])

### a05 seed 0 (alive units 100, lambda 45.27, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 7.0e-15, Ef 1.3e-15, sd_f 2.4e-14, mc_f 1.4e-15, Aw 1.5e-14, Eb 1.4e-15, sd_b 4.3e-14, Ar 6.3e-14, Er 1.6e-14

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.58 [0.44, 0.75] | 0.56 [0.48, 0.75] | -12.1 [-14.8, -8.7] | -19.6 [-23.6, -13.9] | 29.0 | 45.7 | 0.67 |
| probe | 0.71 [0.56, 0.98] | 0.70 [0.62, 0.98] | -8.1 [-10.7, -4.7] | -13.7 [-17.7, -9.0] | 29.5 | 46.6 | 0.66 |
| readout | 0.01 [-1.99, 3.83] | 0.01 [0.00, 0.02] | -31.4 [-45.6, -6.8] | -51.8 [-58.3, -40.5] | 31.8 | 52.2 | 0.83 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | -0.08 [-0.25, 0.14] | -0.39 | 0.34 | 0 |
| probe | -0.13 [-0.46, 0.18] | -0.93 | 0.44 | 0 |
| readout | -0.32 [-1.74, -0.03] | -182.13 | 0.08 | 8 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 100/100 | 100/100 | 100/100 | 25/100 | 100/100 | 0/100 | 100/100 | 0/100 | 2.2 |
| probe | 100/100 | 100/100 | 100/100 | 24/100 | 100/100 | 0/100 | 99/100 | 0/100 | 2.3 |
| readout | 93/100 | 100/100 | 100/100 | 5/100 | 100/100 | 0/100 | 100/100 | 0/100 | 3.3 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 100 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.5 | 24.6 | 28.1 | 30.9 | 33.2 | 35.8 | 38.4 | 41.2 | 45.5 | 53.9 |
| abs(x~)^2 mean | 42.8 | 56.2 | 66.9 | 73.7 | 83.3 | 88.9 | 95.4 | 104.3 | 118.5 | 144.1 |
| e' mass frac (median) | 0.087 | 0.089 | 0.095 | 0.100 | 0.097 | 0.099 | 0.110 | 0.102 | 0.101 | 0.097 |
| e' mass frac (mean) | 0.094 | 0.098 | 0.101 | 0.102 | 0.098 | 0.101 | 0.109 | 0.100 | 0.099 | 0.096 |
| (d-a) mass frac (median) | 0.122 | 0.116 | 0.110 | 0.108 | 0.107 | 0.101 | 0.092 | 0.087 | 0.080 | 0.061 |
| (d-a) mass frac (mean) | 0.132 | 0.122 | 0.112 | 0.110 | 0.106 | 0.102 | 0.092 | 0.086 | 0.078 | 0.060 |
| M_nn rel (full) | 0.51 | 0.66 | 0.77 | 0.84 | 0.94 | 1.00 | 1.09 | 1.19 | 1.34 | 1.66 |
| row sum rel (full) | 0.64 | 0.72 | 0.81 | 0.87 | 0.90 | 0.93 | 1.04 | 1.10 | 1.32 | 1.68 |
| row sum / M_nn (full) | 0.08 | 0.07 | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.07 | 0.07 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.51 | 0.66 | 0.77 | 0.84 | 0.94 | 1.00 | 1.09 | 1.19 | 1.34 | 1.66 |
| row sum rel (probe) | 0.64 | 0.72 | 0.82 | 0.88 | 0.90 | 0.93 | 1.04 | 1.10 | 1.31 | 1.66 |
| row sum / M_nn (probe) | 0.09 | 0.08 | 0.07 | 0.07 | 0.07 | 0.06 | 0.07 | 0.06 | 0.07 | 0.07 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.161 [0.037, 0.386])

### a01 seed 1 (alive units 91, lambda 11.96, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 3.6e-15, Ef 7.1e-16, sd_f 8.9e-15, mc_f 7.9e-16, Aw 4.8e-13, Eb 1.4e-15, sd_b 1.0e-14, Ar 2.7e-14, Er 8.7e-15

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.21 [0.10, 0.42] | 0.27 [0.13, 0.68] | -6.0 [-13.0, -2.8] | -9.4 [-23.1, -4.3] | 7.9 | 13.9 | 0.44 |
| probe | 0.60 [0.28, 1.13] | 0.75 [0.42, 0.99] | -3.1 [-5.4, -1.4] | -3.8 [-9.3, -1.1] | 7.8 | 14.5 | 0.42 |
| readout | 0.03 [-0.70, 0.71] | 0.07 [-0.41, 1.42] | -5.9 [-24.1, +15.3] | -10.6 [-26.6, -3.1] | 6.9 | 11.2 | 1.03 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | 0.23 [-0.00, 0.52] | -0.25 | 0.71 | 0 |
| probe | 0.46 [-0.02, 0.75] | -1.12 | 0.97 | 1 |
| readout | -0.19 [-3.75, -0.04] | -16.64 | 0.04 | 27 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 91/91 | 91/91 | 91/91 | 80/91 | 91/91 | 0/91 | 90/91 | 0/91 | 2.7 |
| probe | 91/91 | 91/91 | 91/91 | 78/91 | 91/91 | 0/91 | 72/91 | 0/91 | 2.4 |
| readout | 64/91 | 90/91 | 91/91 | 1/91 | 89/91 | 2/91 | 87/91 | 0/91 | 2.4 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 91 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.7 | 25.2 | 28.5 | 31.6 | 33.9 | 36.4 | 39.0 | 42.3 | 47.1 | 55.7 |
| abs(x~)^2 mean | 41.9 | 57.9 | 66.6 | 75.5 | 82.8 | 89.5 | 97.7 | 106.9 | 121.3 | 148.6 |
| e' mass frac (median) | 0.188 | 0.135 | 0.136 | 0.098 | 0.091 | 0.083 | 0.041 | 0.046 | 0.008 | 0.000 |
| e' mass frac (mean) | 0.203 | 0.148 | 0.136 | 0.108 | 0.106 | 0.087 | 0.066 | 0.076 | 0.051 | 0.018 |
| (d-a) mass frac (median) | 0.240 | 0.145 | 0.133 | 0.113 | 0.088 | 0.056 | 0.016 | 0.031 | 0.002 | 0.000 |
| (d-a) mass frac (mean) | 0.270 | 0.172 | 0.146 | 0.118 | 0.097 | 0.067 | 0.046 | 0.047 | 0.027 | 0.009 |
| M_nn rel (full) | 0.44 | 0.61 | 0.69 | 0.77 | 0.90 | 0.99 | 1.12 | 1.22 | 1.43 | 1.84 |
| row sum rel (full) | 0.39 | 0.66 | 0.73 | 0.83 | 0.99 | 1.08 | 1.17 | 1.26 | 1.37 | 1.53 |
| row sum / M_nn (full) | 0.07 | 0.09 | 0.08 | 0.09 | 0.09 | 0.09 | 0.08 | 0.08 | 0.08 | 0.07 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.44 | 0.61 | 0.69 | 0.77 | 0.90 | 0.99 | 1.12 | 1.22 | 1.43 | 1.83 |
| row sum rel (probe) | 0.40 | 0.66 | 0.73 | 0.84 | 0.99 | 1.08 | 1.16 | 1.25 | 1.36 | 1.52 |
| row sum / M_nn (probe) | 0.08 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.07 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.007 [0.001, 0.068])

### a03 seed 1 (alive units 100, lambda 56.65, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 6.7e-15, Ef 8.8e-16, sd_f 1.9e-14, mc_f 1.3e-15, Aw 1.0e-14, Eb 8.8e-16, sd_b 2.0e-14, Ar 1.0e-13, Er 1.7e-14

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.49 [0.39, 0.64] | 0.51 [0.38, 0.85] | -10.1 [-14.8, -5.0] | -21.8 [-28.4, -11.4] | 19.8 | 43.8 | 0.41 |
| probe | 0.69 [0.50, 0.92] | 0.73 [0.57, 0.91] | -5.8 [-8.5, -3.5] | -12.6 [-17.1, -6.5] | 20.1 | 46.0 | 0.40 |
| readout | 0.02 [-0.87, 1.17] | 0.03 [-0.07, 0.09] | -22.6 [-51.7, +19.6] | -43.3 [-60.8, -21.5] | 23.7 | 44.3 | 0.85 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | 0.11 [-0.11, 0.42] | -0.39 | 0.76 | 0 |
| probe | 0.18 [-0.24, 0.59] | -2.15 | 0.84 | 0 |
| readout | -0.41 [-1.58, -0.08] | -64.50 | 0.06 | 26 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 100/100 | 100/100 | 100/100 | 63/100 | 100/100 | 0/100 | 100/100 | 0/100 | 3.0 |
| probe | 100/100 | 100/100 | 100/100 | 63/100 | 100/100 | 0/100 | 99/100 | 0/100 | 2.5 |
| readout | 75/100 | 99/100 | 100/100 | 1/100 | 99/100 | 1/100 | 99/100 | 1/100 | 2.8 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 100 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.7 | 25.2 | 28.5 | 31.6 | 33.9 | 36.4 | 39.0 | 42.3 | 47.1 | 55.7 |
| abs(x~)^2 mean | 41.9 | 57.9 | 66.6 | 75.5 | 82.8 | 89.5 | 97.7 | 106.9 | 121.3 | 148.6 |
| e' mass frac (median) | 0.101 | 0.110 | 0.113 | 0.107 | 0.113 | 0.096 | 0.093 | 0.092 | 0.074 | 0.051 |
| e' mass frac (mean) | 0.122 | 0.116 | 0.115 | 0.110 | 0.114 | 0.103 | 0.094 | 0.092 | 0.075 | 0.060 |
| (d-a) mass frac (median) | 0.139 | 0.128 | 0.130 | 0.121 | 0.110 | 0.094 | 0.079 | 0.065 | 0.050 | 0.023 |
| (d-a) mass frac (mean) | 0.167 | 0.137 | 0.130 | 0.126 | 0.115 | 0.091 | 0.079 | 0.068 | 0.054 | 0.034 |
| M_nn rel (full) | 0.48 | 0.65 | 0.74 | 0.82 | 0.91 | 0.98 | 1.10 | 1.20 | 1.38 | 1.74 |
| row sum rel (full) | 0.51 | 0.71 | 0.73 | 0.81 | 0.95 | 0.98 | 1.13 | 1.25 | 1.38 | 1.58 |
| row sum / M_nn (full) | 0.09 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.48 | 0.65 | 0.74 | 0.82 | 0.91 | 0.98 | 1.10 | 1.20 | 1.38 | 1.74 |
| row sum rel (probe) | 0.52 | 0.71 | 0.73 | 0.81 | 0.95 | 0.98 | 1.13 | 1.25 | 1.37 | 1.56 |
| row sum / M_nn (probe) | 0.10 | 0.10 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.050 [0.003, 0.323])

### a05 seed 1 (alive units 100, lambda 36.51, threads 8)

MATCH vs P149: log lines identical 5/5 -> MATCH; per-unit max relative diff vs P149 json: Ag 7.0e-15, Ef 2.5e-15, sd_f 3.7e-14, mc_f 1.9e-15, Aw 1.4e-14, Eb 2.4e-15, sd_b 3.7e-14, Ar 5.9e-14, Er 4.7e-15

C1 (median [min, max] for ratios; median [10%, 90%] for z)

| M | actual/E_full | actual/E_strat | z_full | z_strat | E_full/sd_full | E_strat/sd_strat | sd_strat/sd_full |
|---|---|---|---|---|---|---|---|
| full | 0.56 [0.47, 1.22] | 0.55 [0.46, 1.00] | -12.6 [-15.1, -8.4] | -20.7 [-25.0, -11.3] | 28.6 | 45.1 | 0.69 |
| probe | 0.73 [0.57, 1.37] | 0.69 [0.56, 1.12] | -8.1 [-10.6, -2.6] | -14.1 [-18.4, -6.5] | 29.2 | 46.6 | 0.69 |
| readout | 0.01 [-0.11, 0.16] | 0.01 [0.00, 0.02] | -34.1 [-45.5, -11.1] | -51.1 [-58.5, -41.7] | 34.3 | 51.3 | 0.83 |

(readout row: actual = u^T Ebar f, E_full = P149 lemma; its sd_full is new (uniform MC, rng 15).)

C2 gap closed = (E_full - E_strat)/(E_full - actual), units with E_full > actual

| M | median [10%, 90%] | min | max | units with E_full <= actual |
|---|---|---|---|---|
| full | -0.09 [-0.30, 0.12] | -0.75 | 0.36 | 1 |
| probe | -0.16 [-0.59, 0.17] | -3.78 | 8.35 | 3 |
| readout | -0.31 [-1.51, 0.01] | -13.77 | 0.13 | 2 |

C3 counts

| M | E_full > 0 | E_strat > 0 | actual > 0 | E_strat < E_full | actual < E_strat | actual > E_strat | z_strat < -2 | z_strat > +2 | exact vs MC max abs(diff)/(sd/sqrt150) |
|---|---|---|---|---|---|---|---|---|---|
| full | 100/100 | 100/100 | 100/100 | 29/100 | 100/100 | 0/100 | 99/100 | 0/100 | 2.1 |
| probe | 100/100 | 100/100 | 100/100 | 29/100 | 95/100 | 5/100 | 95/100 | 0/100 | 3.4 |
| readout | 98/100 | 100/100 | 100/100 | 12/100 | 100/100 | 0/100 | 100/100 | 0/100 | 2.3 |

C4 per K-decile profiles (decile 1 = dimmest; pooled = median over 100 units; M profiles are decile mean / all-input mean per unit, then median)

| quantity | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| K mean | 19.7 | 25.2 | 28.5 | 31.6 | 33.9 | 36.4 | 39.0 | 42.3 | 47.1 | 55.7 |
| abs(x~)^2 mean | 41.9 | 57.9 | 66.6 | 75.5 | 82.8 | 89.5 | 97.7 | 106.9 | 121.3 | 148.6 |
| e' mass frac (median) | 0.084 | 0.090 | 0.092 | 0.099 | 0.098 | 0.101 | 0.107 | 0.100 | 0.105 | 0.105 |
| e' mass frac (mean) | 0.090 | 0.092 | 0.095 | 0.100 | 0.099 | 0.104 | 0.108 | 0.105 | 0.104 | 0.102 |
| (d-a) mass frac (median) | 0.113 | 0.116 | 0.111 | 0.109 | 0.106 | 0.101 | 0.093 | 0.091 | 0.080 | 0.057 |
| (d-a) mass frac (mean) | 0.126 | 0.119 | 0.113 | 0.111 | 0.106 | 0.099 | 0.093 | 0.089 | 0.081 | 0.062 |
| M_nn rel (full) | 0.50 | 0.67 | 0.76 | 0.83 | 0.92 | 0.99 | 1.10 | 1.18 | 1.34 | 1.70 |
| row sum rel (full) | 0.61 | 0.72 | 0.77 | 0.86 | 0.92 | 0.96 | 1.05 | 1.17 | 1.27 | 1.66 |
| row sum / M_nn (full) | 0.07 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| units with decile row-sum mean < 0 (full) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| M_nn rel (probe) | 0.50 | 0.67 | 0.76 | 0.83 | 0.92 | 0.99 | 1.10 | 1.18 | 1.35 | 1.70 |
| row sum rel (probe) | 0.61 | 0.72 | 0.77 | 0.86 | 0.93 | 0.97 | 1.06 | 1.17 | 1.27 | 1.63 |
| row sum / M_nn (probe) | 0.08 | 0.07 | 0.06 | 0.07 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| units with decile row-sum mean < 0 (probe) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(units with all-input mean row sum < 0: full 0, probe 0; units with decile M_nn mean < 0: full max 0, probe max 0; open fraction p median 0.186 [0.051, 0.545])
