# Joudaki ViT / Tiny ImageNet: 13 activations

40 tasks × 500 steps; 10 seeds; online window = tasks 21–40.
Raw paired sign-test labels are exploratory. Holm-adjusted p values accompany all 14 distinct comparisons.
Non-significance does not establish equivalence. This is an activation comparison, not a causal mechanism test.

| Arm | n | Window (5-class) | Window (200-class) | Early−late | Validation window | Collapsed |
|---|---:|---:|---:|---:|---:|---:|
| R | 10 | 0.8695 | 0.8691 | -0.1214 | 0.8014 | 0 |
| LK001 | 10 | 0.8674 | 0.8670 | -0.1205 | 0.8010 | 0 |
| GELU | 10 | 0.8588 | 0.8584 | -0.1305 | 0.7948 | 0 |
| LR | 10 | 0.8448 | 0.8444 | -0.1088 | 0.7925 | 0 |
| KKA23 | 10 | 0.8356 | 0.8352 | -0.1091 | 0.7933 | 0 |
| KKA | 10 | 0.8319 | 0.8315 | -0.1134 | 0.7980 | 0 |
| KKT1 | 10 | 0.8314 | 0.8310 | -0.1122 | 0.7943 | 0 |
| SNA | 10 | 0.8296 | 0.8292 | -0.1108 | 0.7933 | 0 |
| SL | 10 | 0.8275 | 0.8271 | -0.1149 | 0.7923 | 0 |
| SILU | 10 | 0.8215 | 0.8211 | -0.1158 | 0.7811 | 0 |
| LK03 | 10 | 0.8043 | 0.8039 | -0.0952 | 0.7795 | 0 |
| RSL | 10 | 0.8025 | 0.8021 | -0.1008 | 0.7736 | 0 |
| ELU | 10 | 0.7816 | 0.7812 | -0.0948 | 0.7663 | 0 |

Registered labels (unadjusted): `{"A": "SNA_BEATEN", "K1": "PERIOD_FREE", "K2": "SCALE_MATTERS_UP", "K3": "TAIL_FREE"}`

Diverged runs: 0. Raw data: /home/i_nakatsuka/Projects/obsidian-research-data/joudaki_vit_battle_0919/runs.
Source: Joudaki et al., arXiv:2510.00304v3 Appendix B; upstream commit 161217078ba52107c94a16602af958a321d62ce3.
