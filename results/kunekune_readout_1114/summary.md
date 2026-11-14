# Matched kunekune readout controls — complete results

6 activations ×10 paired seeds ×3 source ages ×8 arms =1440 curves. CPU native stage1 arithmetic.
ELU/ReLU histories were replayed exactly, not counted as independent replication. All endpoint states reloaded and checked.
U is period-average fixed-training accuracy deficit. Positive benefit is faster learning. CIs: paired-seed10000 bootstrap, descriptive and unadjusted.
P changes column geometry and predictions as well as class association. Head changes are one-time, followed by full learning.
S01/P01 preserve mean logits by compensating the head bias. Zero-head arms reset both weights and bias, so they are not merely the zero endpoint of that scale intervention.
Small benefits alone do not establish health; compare Z_clear and F0 and absolute U. R_body positive means a residual cost after resetting head and Adam.
No causal mediation shares, universal family guarantee, ViT inference, or sustained-prevention claim.

## Task end 1 (secondary)

| Activation | P benefit D | Shrink B_S | Shrink after P B_P | Zero after P Z_P | Residual body R_body |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | +2.87 [+2.61,+3.14] | -11.41 [-11.76,-11.03] | -12.30 [-12.57,-12.04] | -15.11 [-15.43,-14.77] | -1.90 [-2.13,-1.64] |
| KKA23 | +3.67 [+3.42,+3.88] | -14.56 [-14.94,-14.18] | -15.97 [-16.24,-15.68] | -19.10 [-19.44,-18.75] | -1.88 [-2.10,-1.65] |
| KKT1 | +2.86 [+2.60,+3.12] | -10.96 [-11.20,-10.71] | -11.84 [-12.07,-11.61] | -14.66 [-15.01,-14.30] | -2.20 [-2.41,-1.97] |
| SNA | +2.94 [+2.64,+3.26] | -11.00 [-11.22,-10.80] | -11.82 [-12.11,-11.53] | -14.74 [-15.10,-14.36] | -2.09 [-2.32,-1.84] |
| ELU | +13.18 [+12.80,+13.60] | -8.65 [-9.19,-8.09] | -19.45 [-19.78,-19.12] | -21.92 [-22.25,-21.57] | -6.80 [-7.12,-6.48] |
| R | +14.47 [+13.56,+15.35] | -0.34 [-1.32,+0.75] | -12.75 [-13.57,-11.93] | -16.17 [-16.95,-15.42] | -4.17 [-4.99,-3.41] |

| Activation | Native U (%) | Zero-head / Adam reset U (%) | Fresh-body U (%) | Source mean abs derivative | Source head centered norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | 13.58 | 19.42 | 21.32 | 1.1564 | 8.323 |
| KKA23 | 16.36 | 22.74 | 24.62 | 0.8148 | 9.387 |
| KKT1 | 13.05 | 18.90 | 21.10 | 1.0986 | 8.941 |
| SNA | 13.13 | 19.01 | 21.10 | 1.1002 | 8.978 |
| ELU | 28.40 | 25.81 | 32.61 | 0.7501 | 10.032 |
| R | 30.18 | 23.02 | 27.19 | 0.4637 | 10.287 |

## Task end 10 (primary)

| Activation | P benefit D | Shrink B_S | Shrink after P B_P | Zero after P Z_P | Residual body R_body |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | +0.30 [-0.02,+0.64] | -11.75 [-12.00,-11.47] | -11.96 [-12.41,-11.48] | -13.85 [-14.40,-13.23] | -1.56 [-1.88,-1.21] |
| KKA23 | +0.13 [-0.46,+0.66] | -14.32 [-14.61,-14.07] | -14.63 [-15.17,-14.10] | -17.02 [-17.61,-16.44] | -1.11 [-1.42,-0.83] |
| KKT1 | +0.44 [-0.02,+0.85] | -11.71 [-11.98,-11.43] | -12.53 [-12.98,-12.04] | -14.16 [-14.62,-13.74] | -1.46 [-1.59,-1.30] |
| SNA | +0.51 [+0.11,+0.90] | -12.00 [-12.33,-11.68] | -12.81 [-13.24,-12.42] | -14.51 [-15.05,-14.02] | -1.08 [-1.41,-0.74] |
| ELU | -0.76 [-1.80,+0.23] | +19.46 [+18.62,+20.64] | +19.48 [+18.32,+20.83] | +21.87 [+20.69,+23.15] | -8.32 [-8.75,-7.86] |
| R | +0.12 [-0.03,+0.28] | +1.40 [+0.82,+2.13] | +1.53 [+0.91,+2.30] | +2.80 [+1.74,+4.13] | +56.50 [+54.57,+58.12] |

| Activation | Native U (%) | Zero-head / Adam reset U (%) | Fresh-body U (%) | Source mean abs derivative | Source head centered norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | 14.20 | 20.05 | 21.62 | 0.6138 | 10.561 |
| KKA23 | 16.62 | 23.67 | 24.78 | 0.4088 | 12.953 |
| KKT1 | 13.98 | 19.87 | 21.33 | 0.6112 | 10.888 |
| SNA | 14.10 | 20.15 | 21.23 | 0.6101 | 10.835 |
| ELU | 57.55 | 24.37 | 32.68 | 0.0756 | 16.224 |
| R | 88.32 | 83.67 | 27.17 | 0.0004 | 28.240 |

## Task end 30 (secondary)

| Activation | P benefit D | Shrink B_S | Shrink after P B_P | Zero after P Z_P | Residual body R_body |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | -0.54 [-0.94,-0.12] | -10.23 [-10.55,-9.90] | -10.18 [-10.60,-9.74] | -10.64 [-11.10,-10.11] | +0.39 [+0.00,+0.75] |
| KKA23 | -0.92 [-1.76,-0.07] | -13.10 [-13.51,-12.72] | -12.69 [-13.33,-12.06] | -13.34 [-14.04,-12.59] | +0.18 [-0.27,+0.64] |
| KKT1 | -0.63 [-1.11,-0.13] | -10.36 [-10.85,-9.89] | -10.15 [-10.47,-9.78] | -10.85 [-11.30,-10.34] | +0.54 [+0.32,+0.75] |
| SNA | -0.35 [-1.05,+0.40] | -9.92 [-10.34,-9.46] | -10.11 [-10.62,-9.59] | -10.61 [-11.21,-10.01] | +0.55 [+0.11,+0.99] |
| ELU | -1.34 [-2.28,-0.50] | +15.61 [+14.96,+16.23] | +16.00 [+15.06,+16.85] | +19.77 [+18.38,+21.03] | +11.48 [+10.00,+13.03] |
| R | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | +0.00 [+0.00,+0.00] | +0.19 [+0.12,+0.25] | +62.04 [+60.99,+63.17] |

| Activation | Native U (%) | Zero-head / Adam reset U (%) | Fresh-body U (%) | Source mean abs derivative | Source head centered norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| KKA | 18.10 | 21.88 | 21.50 | 0.6108 | 7.778 |
| KKA23 | 20.16 | 24.87 | 24.69 | 0.4040 | 10.140 |
| KKT1 | 18.20 | 21.77 | 21.23 | 0.6063 | 7.815 |
| SNA | 18.36 | 21.69 | 21.14 | 0.6099 | 7.739 |
| ELU | 74.32 | 44.09 | 32.61 | 0.0183 | 15.865 |
| R | 88.75 | 88.56 | 26.51 | 0.0000 | 28.524 |

## Validation and scope

integrity.json and endpoint_audit.csv document1440 saved-state checks, canonical replay, paired streams and independent U aggregation.
per_seed.csv includes censoring and response diagnostics; contrasts.csv includes all registered contrasts and paired activation differences.
The family variants adapt alpha from variance; treatment differences include that history. Positive response plus positive shrink benefit is coexistence, not mediation.
Mean derivatives and positive occupancy are distinct for oscillatory activations. Do not equate negative preactivation with a dead unit.

T90_first is the first observed grid point with at least1080 correct; it is not an exact per-step crossing time. Nonarrival is right-censored, not assigned4000.
