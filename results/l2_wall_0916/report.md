# l2_wall_0916 report

- **B_EE**: `MIXED` delta=0.0423 eps1e-6/s0:+0.0035 eps1e-6/s1:-0.0095 eps1e-6/s2:-0.0487 eps1e-30/s0:+0.0189 eps1e-30/s1:+0.0145 eps1e-30/s2:+0.0149
- **B_LE**: `MIXED` delta=0.0213 eps1e-6/s0:-0.0042 eps1e-6/s1:-0.0189 eps1e-6/s2:-0.0055 eps1e-30/s0:+0.0488 eps1e-30/s1:+0.0408 eps1e-30/s2:+0.0392
- **C1_RL_EE_inLN**: `RESCUED` RESCUED RESCUED RESCUED
- **C2_RL_LE_inLN**: `RESCUED` 
- **C3_RL_EE_preLN**: `RESCUED` 
- **C4_RL_EE_preLN_a**: `RESCUED` RESCUED RESCUED RESCUED

| id | owner | prediction | outcome | detail |
|---|---|---|---|---|
| P-A1 | Claude | RL EE: cone entry no later than W2 eps regime (3/3) | HIT | cone=(2, 75) eps=(17, 375); cone=(2, 75) eps=(15, 75); cone=(2, 75) eps=(20, 1500) |
| P-A2 | Claude | RL EE t0-t20: W2 has the largest Shapley share of Δd2 (3/3) | MISS | {"W": 2.713, "mu": 6.582, "S": 5.821}; {"W": 1.986, "mu": 7.112, "S": 5.689}; {"W": 2.124, "mu": 6.331, "S": 5.323} |
| P-A3 | Claude | RL EE t0-t10: |self|>|up| and |cross|<0.1|Δz̄2| (3/3) | MISS | meas -39.97 self -8.36 up -14.52 cross -16.96 bias -0.12; meas -52.13 self -8.91 up -24.04 cross -19.05 bias -0.13; meas -37.66 self -9.26 up -12.28 cross -15.96 bias -0.15 |
| P-A4 | Claude | RL EE after eps regime to t50: |up|>|self| (3/3) | HIT | t17+375-t50 self -3.56 up -59.30 cross -0.72; t15+75-t50 self -4.59 up -68.03 cross -0.81; t20+1500-t50 self -5.84 up -38.50 cross 0.35 |
| P-A5 | Claude | RL LE t50: min e2'a1>0 and |d2+c2|<0.2 (3/3) | HIT | pmin 65.24 c2 3.25 k2 2.16 d2 -3.13; pmin 46.23 c2 3.28 k2 2.28 d2 -3.15; pmin 21.09 c2 3.16 k2 2.33 d2 -3.03 |
| P-A6 | Claude | ext150 PM GELU t150: median rho>=0.8 and median |b|/sigma<=0.3 (3/3) | HIT | rho 0.95 |b|/s 0.03 d2 -3.38 c2 3.44 k2 2.18; rho 0.98 |b|/s 0.04 d2 -3.20 c2 3.19 k2 2.12; rho 0.97 |b|/s 0.02 d2 -3.13 c2 3.14 k2 2.19 |
| P-A7 | Claude | ext150 RL SILU s0,s1 t150: same pin criterion (2/2) | HIT | rho 1.00 |b|/s 0.00 d2 -3.55 c2 3.54 k2 2.18; rho 1.01 |b|/s 0.00 d2 -3.56 c2 3.55 k2 2.35 |
| P-B-EE | Claude | Part B EE = PIN | MISS | delta=0.0423 eps1e-6/s0:+0.0035 eps1e-6/s1:-0.0095 eps1e-6/s2:-0.0487 eps1e-30/s0:+0.0189 eps1e-30/s1:+0.0145 eps1e-30/s2:+0.0149 |
| P-B1-EE | Claude | EE: eps1e-30 deeper |z̄2(t50)| than none (3/3) | MISS | s0 150.3 vs 155.7; s1 123.2 vs 155.8; s2 116.2 vs 147.7 |
| P-B2-EE | Claude | EE: eps1e-6 shallower |z̄2(t50)| than none (3/3) | MISS | s0 155.8 vs 155.7; s1 118.3 vs 155.8; s2 125.0 vs 147.7 |
| P-B-LE | Claude | Part B LE = PIN | MISS | delta=0.0213 eps1e-6/s0:-0.0042 eps1e-6/s1:-0.0189 eps1e-6/s2:-0.0055 eps1e-30/s0:+0.0488 eps1e-30/s1:+0.0408 eps1e-30/s2:+0.0392 |
| P-B1-LE | Claude | LE: eps1e-30 deeper |z̄2(t50)| than none (3/3) | MISS | s0 99.8 vs 168.3; s1 110.8 vs 152.0; s2 104.0 vs 138.4 |
| P-B2-LE | Claude | LE: eps1e-6 shallower |z̄2(t50)| than none (3/3) | MISS | s0 150.4 vs 168.3; s1 152.8 vs 152.0; s2 158.0 vs 138.4 |
| P-B3 | Claude | all eps arms at floor (12/12) | HIT | 12/12 |
| P-C1 | Claude | C1 = MU_PATH | MISS | RESCUED |
| P-C1-other | pasted reading (via Issa) | C1 = B_PATH | MISS | RESCUED |
| P-C1b | Claude | RL EE inLN: late d2 deeper than -c2(none) (3/3) | MISS | d2 -2.84 vs -3.38; d2 -2.98 vs -3.51; d2 -2.94 vs -3.45 |
| P-C2 | Claude | C2 = RESCUED | HIT | RESCUED |
| P-C2-other | pasted reading (via Issa) | C2 = COLLAPSED | MISS | RESCUED |
| P-C3 | Claude | C3 = RESCUED | HIT | RESCUED |
| P-C4 | Claude | C4 = RESCUED | HIT | RESCUED |
