# chargpt_detlabel_1023 の判定（judge.py）

- 作成: 2026-10-24 01:32:01 +0900　judge.py sha256 a2e51af0d8b7704f…　spec sha256 42f4bcaa1c17db9f…
- A2 の記録: /home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019/results/chargpt_cl_1019/M2/C3（使える seed 10/10）。A2 の J4 は再生の出力 /tmp/claude-1000/-home-kubo-project-Nakatsuka/fff216c2-f017-401b-869a-e3d0a73aac33/scratchpad/smoke/none（使える seed 0）。A2 の J6 は loadsign（/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_loadsign_1023）。
- ラベルの境は §7.0（spec:284〜291）。使える seed が 10 未満なら k/10 → ceil(k·n/10)、8 未満なら INSUFFICIENT。

## A4 − A2（主）

使える seed 10/10

| 判定 | ラベル | 数 | 併記 |
| --- | --- | --- | --- |
| J1 喪失（spec:293） | **LOSS_PERSISTS** | LOSS 10/10（境 ≥ 8・≤ 2） | 相ごと T015/E1000/ELast [10, 10, 10]、学習成立 10、ΔE 1.08 [0.769, 2.01]（n 10）、ΔE の A4 < A2 の対 0/10、符号 +0 / −0 / 0 10（p = NaN）、比の中央値 1 |
| J1b 条件 1（記録、spec:310） | — | 10/10 | — |
| J2 L0 の伸び（spec:314） | **L0_GROWTH_SIMILAR** | ρ2 = 1 | g 15.1 [14, 16.7]（n 10） 対 A2 15.1 [14, 16.7]（n 10）、g 小の対 0、符号 +0 / −0 / 0 10（p = NaN） |
| J3 L0 の死（spec:322） | **L0_DEATH_PERSISTS** | median d 0.865（境 < 0.433） | d 0.865 [0, 0.959]（n 10）、d 小の対 0、符号 +0 / −0 / 0 10（p = NaN）、d ≥ 0.5 の seed 8 |
| J4 後半の動き net5k（spec:329・454） | **INSUFFICIENT** | ρ4 = NaN | M NaN [NaN, NaN]（n 0） 対 A2 再生 NaN [NaN, NaN]（n 0）、符号 +0 / −0 / 0 0（p = NaN）、窓の ρ4 early NaN, middle NaN, late NaN |
| J5a L1 の早期の死（spec:344） | **L1_EARLY_DEATH** | 10/10 | — |
| J5b 後期の戻り（spec:345） | **L1_REVIVES** | 9/10 | R_k {0: 55, 1: 50, 2: 84, 3: None, 4: 56, 5: 37, 6: 68, 7: 44, 8: 65, 9: 57}、R_k の対の符号 +0 / −0 / 0 10（p = NaN） |
| J6 他 unit 項（spec:359） | **OTHER_TERM_FLIPS** | 0/10（境 ≥ 7） | 後期 s_original_other の A − A2 の符号 +0 / −0 / 0 10（p = NaN） |
| J6-LN（spec:360） | **LN_HALVES** | 10/10 | — |

記録（判定に入れない）:

- J1: ΔE（早期 1〜10）1.07 [0.765, 2]（n 10）。10 課題ごとの E1000 の平均の中央値: 0.194, 0.28, 0.388, 0.63, 0.889, 1.13, 1.06, 1.07, 1.3, 1.27
- J2: g0_t10 4.23 [3.95, 4.6]（n 10）（A2 4.23 [3.95, 4.6]（n 10））; g0_t50 10.4 [9.09, 11.7]（n 10）（A2 10.4 [9.09, 11.7]（n 10））; g1 7.97 [6.72, 11.1]（n 10）（A2 7.97 [6.72, 11.1]（n 10））; g0_alive 14.9 [14, 15.7]（n 10）（A2 14.9 [14, 15.7]（n 10））; g1_alive 8.03 [6.94, 11.1]（n 10）（A2 8.03 [6.94, 11.1]（n 10））
- J2: 課題末どうしの 1 − cos（L0、seed 内の窓の中央値の seed 間の中央値、全行 / 生きた unit）: 2-10 0.116 / 0.116; 41-50 0.0307 / 0.0314; 91-100 0.00097 / 0.0114
- J3: d_91_100 0.832 [0, 0.955]（n 10）; d50 0.166 [0.00586, 0.682]（n 10）; d_chain100 0.858 [0, 0.967]（n 10）; ge099_100 0 [0, 0]（n 10）; 死が初めて 0.5 以上の課題 {0: 65, 1: 64, 2: 87, 3: 78, 4: None, 5: 51, 6: 98, 7: 44, 8: None, 9: 57}
- J4（A / A2 再生）: L0_M NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net_over_path NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net_over_norm NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_rot5k NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net100_end NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net100_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net_over_path NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net_over_norm NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_rot5k NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net100_end NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net100_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）
- J4 の欠けた課題の数（seed ごと）: {}、A2 再生 {}
- J4 net100 の A2 の値 L0: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.66 [1.37, 1.78]・全行 1.17 [1.08, 1.78]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit NaN [NaN, NaN]（n 0）・全行 NaN [NaN, NaN]（n 0）　(iii) A4 生きた unit NaN [NaN, NaN]（n 0）
- J4 net100 の A2 の値 L1: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.36 [1.22, 1.48]・全行 1.03 [0.462, 1.36]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit NaN [NaN, NaN]（n 0）・全行 NaN [NaN, NaN]（n 0）　(iii) A4 生きた unit NaN [NaN, NaN]（n 0）
- J5: L1d50 0.977 [0, 0.99]（n 10）; L1d100 0.124 [0, 0.965]（n 10）; spearman99 -0.106 [-0.29, 0.134]（n 10）; spearman31 -0.432 [-0.617, -0.0665]（n 10）; Spearman 負の seed 99 対 8・31 対 10
- J6 early: s_original_self 1 [0.997, 1]（n 10）; s_original_other 0.815 [0.755, 0.895]（n 10）; s_original_residual 0.852 [0.687, 0.898]（n 10）; s_original_direct_K 0.991 [0.986, 0.996]（n 10）; s_radial 0.00549 [0.00255, 0.0104]（n 10）; s_original_exact_K 0.771 [0.719, 0.798]（n 10）; original_other_abs_share 0.196 [0.169, 0.202]（n 10）; n_alive 460 [442, 482]（n 10）
- J6 middle: s_original_self 0.968 [0.936, 1]（n 10）; s_original_other 0.661 [0.0244, 0.699]（n 10）; s_original_residual 0.643 [0.483, 0.689]（n 10）; s_original_direct_K 0.828 [0.664, 0.868]（n 10）; s_radial 0.0115 [0.00223, 0.344]（n 10）; s_original_exact_K 0.408 [0.392, 0.473]（n 10）; original_other_abs_share 0.314 [0.199, 0.471]（n 10）; n_alive 156 [126, 512]（n 10）
- J6 late: s_original_self 0.997 [0.75, 1]（n 10）; s_original_other 0.148 [0.00586, 0.512]（n 10）; s_original_residual 0.492 [0.476, 0.507]（n 10）; s_original_direct_K 0.6 [0.493, 0.749]（n 10）; s_radial 0.376 [0.176, 0.439]（n 10）; s_original_exact_K 0.485 [0.46, 0.519]（n 10）; original_other_abs_share 0.487 [0.395, 0.59]（n 10）; n_alive 488 [109, 512]（n 10）
- §7.7 縮退した課題（列 chain_stats.csv が無いか列が読めない）: seed ごとの数 {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}、除いた J1 LOSS_PERSISTS（10/10）、除いた J4 INSUFFICIENT（ρ4 NaN）
- §7.7 追補 6 の条件 2（judge_m2.py）を満たす seed 9/10。条件 3 は出さない（読みの選択 11）。
- §7.7 位置 r の中央値（窓の平均の seed 間の中央値）: rmed_uniform_L0 1-10 -0.601, 41-50 -1.36, 91-100 -3.53; rmed_uniform_L1 1-10 -2.39, 41-50 -3.25, 91-100 1.11; rmed_chain_L0 1-10 -0.597, 41-50 -1.38, 91-100 -3.51; rmed_chain_L1 1-10 -2.35, 41-50 -3.21, 91-100 1.17

## A3 − A2（記録）

使える seed 10/10

| 判定 | ラベル | 数 | 併記 |
| --- | --- | --- | --- |
| J1 喪失（spec:293） | **LOSS_GONE** | LOSS 0/10（境 ≥ 8・≤ 2） | 相ごと T015/E1000/ELast [0, 0, 1]、学習成立 10、ΔE -0.0353 [-0.0508, -0.0217]（n 10）、ΔE の A3 < A2 の対 10/10、符号 +0 / −10 / 0 0（p = 0.00195）、比の中央値 -0.0362 |
| J1b 条件 1（記録、spec:310） | — | 0/10 | — |
| J2 L0 の伸び（spec:314） | **L0_GROWTH_SMALLER** | ρ2 = 0.013 | g 0.197 [0.179, 0.453]（n 10） 対 A2 15.1 [14, 16.7]（n 10）、g 小の対 10、符号 +0 / −10 / 0 0（p = 0.00195） |
| J3 L0 の死（spec:322） | **L0_DEATH_REDUCED** | median d 0（境 < 0.433） | d 0 [0, 0.00195]（n 10）、d 小の対 9、符号 +0 / −9 / 0 1（p = 0.00391）、d ≥ 0.5 の seed 0 |
| J4 後半の動き net5k（spec:329・454） | **INSUFFICIENT** | ρ4 = NaN | M NaN [NaN, NaN]（n 0） 対 A2 再生 NaN [NaN, NaN]（n 0）、符号 +0 / −0 / 0 0（p = NaN）、窓の ρ4 early NaN, middle NaN, late NaN |
| J5a L1 の早期の死（spec:344） | **L1_EARLY_DEATH** | 10/10 | — |
| J5b 後期の戻り（spec:345） | **L1_REVIVES** | 10/10 | R_k {0: 41, 1: 59, 2: 50, 3: 49, 4: 51, 5: 50, 6: 51, 7: 60, 8: 50, 9: 49}、R_k の対の符号 +3 / −7 / 0 0（p = 0.344） |
| J6 他 unit 項（spec:359） | **OTHER_TERM_KEPT** | 10/10（境 ≥ 7） | 後期 s_original_other の A − A2 の符号 +10 / −0 / 0 0（p = 0.00195） |
| J6-LN（spec:360） | **LN_HALVES** | 10/10 | — |

記録（判定に入れない）:

- J1: ΔE（早期 1〜10）-0.0397 [-0.0541, -0.027]（n 10）。10 課題ごとの E1000 の平均の中央値: 0.194, 0.232, 0.261, 0.278, 0.275, 0.2, 0.162, 0.154, 0.15, 0.148
- J2: g0_t10 0.233 [0.192, 0.362]（n 10）（A2 4.23 [3.95, 4.6]（n 10））; g0_t50 0.109 [0.0961, 0.216]（n 10）（A2 10.4 [9.09, 11.7]（n 10））; g1 0.142 [0.129, 0.313]（n 10）（A2 7.97 [6.72, 11.1]（n 10））; g0_alive 0.197 [0.179, 0.453]（n 10）（A2 14.9 [14, 15.7]（n 10））; g1_alive 0.142 [0.129, 0.313]（n 10）（A2 8.03 [6.94, 11.1]（n 10））
- J2: 課題末どうしの 1 − cos（L0、seed 内の窓の中央値の seed 間の中央値、全行 / 生きた unit）: 2-10 0.566 / 0.566; 41-50 0.769 / 0.769; 91-100 0.666 / 0.666
- J3: d_91_100 0 [0, 0.000977]（n 10）; d50 0 [0, 0.00391]（n 10）; d_chain100 0 [0, 0.0684]（n 10）; ge099_100 0 [0, 0]（n 10）; 死が初めて 0.5 以上の課題 {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None}
- J4（A / A2 再生）: L0_M NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net_over_path NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net_over_norm NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_rot5k NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net100_end NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L0_M_net100_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net_over_path NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net_over_norm NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_rot5k NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net100_end NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）; L1_M_net100_allrows NaN [NaN, NaN]（n 0） / NaN [NaN, NaN]（n 0）
- J4 の欠けた課題の数（seed ごと）: {}、A2 再生 {}
- J4 net100 の A2 の値 L0: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.66 [1.37, 1.78]・全行 1.17 [1.08, 1.78]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit NaN [NaN, NaN]（n 0）・全行 NaN [NaN, NaN]（n 0）　(iii) A3 生きた unit NaN [NaN, NaN]（n 0）
- J4 net100 の A2 の値 L1: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.36 [1.22, 1.48]・全行 1.03 [0.462, 1.36]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit NaN [NaN, NaN]（n 0）・全行 NaN [NaN, NaN]（n 0）　(iii) A3 生きた unit NaN [NaN, NaN]（n 0）
- J5: L1d50 0.435 [0, 0.912]（n 10）; L1d100 0 [0, 0]（n 10）; spearman99 -0.124 [-0.22, 0.0408]（n 10）; spearman31 -0.16 [-0.323, 0.0888]（n 10）; Spearman 負の seed 99 対 9・31 対 8
- J6 early: s_original_self 0.995 [0.987, 1]（n 10）; s_original_other 0.849 [0.769, 0.903]（n 10）; s_original_residual 0.818 [0.431, 0.901]（n 10）; s_original_direct_K 0.983 [0.963, 0.994]（n 10）; s_radial 0.00211 [0, 0.0137]（n 10）; s_original_exact_K 0.753 [0.719, 0.771]（n 10）; original_other_abs_share 0.195 [0.155, 0.221]（n 10）; n_alive 376 [210, 508]（n 10）
- J6 middle: s_original_self 0.993 [0.973, 1]（n 10）; s_original_other 0.938 [0.879, 0.997]（n 10）; s_original_residual 0.903 [0.756, 0.928]（n 10）; s_original_direct_K 0.978 [0.95, 1]（n 10）; s_radial 0.00634 [0, 0.0314]（n 10）; s_original_exact_K 0.641 [0.529, 0.713]（n 10）; original_other_abs_share 0.175 [0.131, 0.651]（n 10）; n_alive 133 [112, 512]（n 10）
- J6 late: s_original_self 1 [1, 1]（n 10）; s_original_other 0.997 [0.99, 1]（n 10）; s_original_residual 0.727 [0.66, 0.785]（n 10）; s_original_direct_K 1 [0.998, 1]（n 10）; s_radial 0.0859 [0.0439, 0.144]（n 10）; s_original_exact_K 0.465 [0.446, 0.514]（n 10）; original_other_abs_share 0.77 [0.735, 0.802]（n 10）; n_alive 512 [512, 512]（n 10）
- §7.7 縮退した課題（列 chain_stats.csv が無いか列が読めない）: seed ごとの数 {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}、除いた J1 LOSS_GONE（0/10）、除いた J4 INSUFFICIENT（ρ4 NaN）
- §7.7 追補 6 の条件 2（judge_m2.py）を満たす seed 10/10。条件 3 は出さない（読みの選択 11）。
- §7.7 位置 r の中央値（窓の平均の seed 間の中央値）: rmed_uniform_L0 1-10 -0.702, 41-50 -0.773, 91-100 -0.789; rmed_uniform_L1 1-10 -2.86, 41-50 -4.2, 91-100 -0.019; rmed_chain_L0 1-10 -0.696, 41-50 -0.77, 91-100 -0.805; rmed_chain_L1 1-10 -2.78, 41-50 -4.16, 91-100 -0.0154

## A2 の再生（§6.4・§12.3。この台での再生）

- seed 0: 置き場が無い
- seed 1: 置き場が無い
- seed 2: 置き場が無い
- seed 3: 置き場が無い
- seed 4: 置き場が無い
- seed 5: 置き場が無い
- seed 6: 置き場が無い
- seed 7: 置き場が無い
- seed 8: 置き場が無い
- seed 9: 置き場が無い

## S-entropy の記録（Issa の判断 2026-10-24: 止める検査から記録に替えた。判定は変えない。外れた連鎖を除かない・引き直さない）

- A4: 連鎖 0、旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差）で外れ 0、そのうち代案（系列ごとと真の分散の大きい方の標準誤差）でも外れ 0
- A3: 連鎖 0、旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差）で外れ 0、そのうち代案（系列ごとと真の分散の大きい方の標準誤差）でも外れ 0

## 予測の採点（§3、A4 について。数は書き換えない）

| 予測 | 誰 | p | 規則 | 結果 | p(結果) | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| P1（spec:102） | Fable | 0.55 | J1 が LOSS_PERSISTS、かつ ΔE が A4 < A2 の seed が 8/10 以上 | False | 0.45 | 0.303 |
| P1（spec:102） | Opus | 0.35 | J1 が LOSS_PERSISTS、かつ ΔE が A4 < A2 の seed が 8/10 以上 | False | 0.65 | 0.122 |
| P2（spec:103） | Fable | 0.6 | J2 が L0_GROWTH_SMALLER | False | 0.4 | 0.36 |
| P2（spec:103） | Opus | 0.35 | J2 が L0_GROWTH_SMALLER | False | 0.65 | 0.122 |
| P3（spec:104） | Fable | 0.8 | J4 の比 ρ4 ≤ 1/3（net5k、§12.1 の 2） | INSUFFICIENT・未完 | — | — |
| P3（spec:104） | Opus | 0.15 | J4 の比 ρ4 ≤ 1/3（net5k、§12.1 の 2） | INSUFFICIENT・未完 | — | — |
| P6（spec:105） | Fable | 0.4 | J6 が OTHER_TERM_KEPT | False | 0.6 | 0.16 |
| P6-LN（spec:106） | Fable | 0.7 | J6-LN が LN_HALVES | True | 0.7 | 0.09 |

## 読みの選択

- **(1) 欠けた seed の境** spec:290 の「k/10 の境を k × n/10 の切り上げに読み替える」を、以上の境（8/10・7/10）にも以下の境（2/10）にも同じく当てる（thr(k, n) = ceil(k·n/10)、整数演算）。n は判定に使える seed の数。ρ の比（J2・J4）と J3 の中央値の比較は、両方の腕で使える seed が 8 以上のときだけラベルを出す。
- **(2) 使える seed** 腕の seed は、per_task.csv が 100 行・online.npz が [100, 10000]・fresh_online_t100.npz・fresh.csv（課題 10・100）・unit_stats.npz・layer_stats.csv（課題 100 の u = 10000）が揃うとき使える。J4 はさらに within_motion.npz（A2 は再生の出力）が読めて M_k が有限のとき、J6 は後期の 12 状態（課題末 89〜100）が全部あるとき使える。
- **(3) 死の量** J3・J5 の死は spec:324・343 のとおり layer_stats.csv（probe = uniform、u = 10000）の frac_p_lt005 を使う。登録前の a2_table.py は unit_stats.npz の uniform_q から (1 − q) < 0.005 を数えた。検査 (a) で両者の一致を確かめた。
- **(4) J4 の生きた unit** net5k の m_{k,t} は、(r2) の更新 5,000 と 10,000 の鎖 probe の印（within_motion.npz の alive[..., u_half]・alive[..., u_end]、True = z > 0 のトークンがある）が両方とも生きている unit（spec:332）。記録の net100 は §7.4 の代案の定義（課題末の鎖 probe、spec:339）に合わせ、更新 10,000 の印だけで決める（A4・A3・S2 は (r2) の alive10k、無ければ unit_stats.npz の chain_dead の否定。A2 の再生は再生の alive10k）。
- **(5) J4 の欠けた課題** T_J4 の課題が within_motion に無い・生きた unit が無い課題は欠測（NaN）にし、M_k は残りの課題の中央値（nanmedian）。欠測の数を記録に書く。29 課題すべてが欠測なら、その seed は J4 に使えない。
- **(6) 縮退した課題（§7.7）** chain_stats.csv の最初の 500 更新の異なる文脈の数（列名は CTX500_COLS のどれか）が 100 以下の課題。A2 は chain_stats.csv を持たないので縮退した課題 0 とする（登録前の計算 §4.3 で α = 0.01 は 0/30、最小 4,151）。縮退を除いた J1 は、早期・後期の窓の平均からその課題を除く（本体(100) と新品(100) は課題 100 のまま。課題 100 が縮退なら印を付ける）。窓の課題がすべて縮退なら、その seed はその J1 に使えない。縮退を除いた J4 は T_J4 からその課題を除く。
- **(7) J5b の数** J5b は、J5a で早期に死んだ seed の数 m を分母にした割合（spec:345）。m < 6 なら INSUFFICIENT。使える seed が 8 未満なら §7.0 により INSUFFICIENT（m の条件より先に見る）。
- **(8) J6 の行** J6 の入力（compute_J の出力、loadsign と同じ列名）で layer 列があれば layer = 1 の行だけを判定に使い、layer = 0 は L0 の記録にする。components_states.csv に layer 列が無ければ全行を L1 とみなす。腕の名は arm 列で引く（既定 A4・A3・S2。A2 の参照は loadsign の C3）。
- **(9) Spearman（J5 の記録）** chargpt_handover_1023 の P4 の記録の形（spec_chargpt_handover_1023.md:97）に従い、layer_stats.csv（chain probe、u = 10000）の Δ(1 − frac_p_lt005)_L1 と Δr_med_L0 の連続する課題末の 99 対で seed ごとの ρ を出す（同順位は平均順位、片方が定数なら未定義）。checkpoint のある 31 対に限った値も並べる。自前の forward を使う handover の主の数とは別物。
- **(10) 予測の採点** 予測は真偽の 2 値。結果が INSUFFICIENT・未完のときは採点しない。p_of_outcome は結果が真なら p、偽なら 1 − p、Brier は (p − 1[真])²。
- **(11) §7.7 の条件 3** 追補 6 の条件 3 は decomp_dmu.csv（別の解析の出力）を要るので、この judge.py では出さない（記録だけ。判定に入らない）。条件 2 と切替の押し s_{l,t} は unit_stats.npz から judge_m2.py の式で出す。
- **(12) S2 の J1** S2 の online.npz は loss がサンプルした目標の CE（A2 と同じ量）、aux が勾配に使った真の分布の CE（compute_W の報告 §2）。J1・J1b・ΔE は loss と per_task.csv の online_all（loss − H の平均。試走で確かめた）を使い、aux は使わない（§4.5 の記録の揃え方）。
- **(13) net100 の A2 の値** Issa の判断（2026-10-24）により 2 つを区別して並べる。(i) 登録前の参照 1.66 [1.37, 1.78]（white-san の保存状態 u100 から、出典 prereg/a2_j4_fallback.{py,txt}。judge.py は txt を読み sha256 を残す）。(ii) byakko での再生の net100（再生の出力の alive[..., u_end] で生きた unit）。(ii) は元の軌道の値と呼ばない（spec:395・469）。
- **(14) S-entropy** Issa の判断（2026-10-24）により止める検査から記録に替わった。chain_stats.csv の ok_entropy（旧規則）が False の連鎖を、3 通りの標準誤差（トークンごと true_ce_eval_se・系列ごと _se_seq・真の分散 _se_truevar）と CE − H、z とともに一覧にする。判定・連鎖・データは変えない。
