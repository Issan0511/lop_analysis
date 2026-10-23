# chargpt_detlabel_1023 の判定（judge.py）

- 作成: 2026-10-24 01:31:49 +0900　judge.py sha256 a2e51af0d8b7704f…　spec sha256 42f4bcaa1c17db9f…
- A2 の記録: /home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_detlabel_1023/checks/judge/cjW_afua1rzl/A2（使える seed 10/10）。A2 の J4 は再生の出力 /home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_detlabel_1023/checks/judge/cjW_afua1rzl/res/replay_A2（使える seed 10）。A2 の J6 は loadsign（/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_loadsign_1023）。
- ラベルの境は §7.0（spec:284〜291）。使える seed が 10 未満なら k/10 → ceil(k·n/10)、8 未満なら INSUFFICIENT。

## A4 − A2（主）

使える seed 10/10

| 判定 | ラベル | 数 | 併記 |
| --- | --- | --- | --- |
| J1 喪失（spec:293） | **LOSS_PERSISTS** | LOSS 10/10（境 ≥ 8・≤ 2） | 相ごと T015/E1000/ELast [10, 10, 10]、学習成立 10、ΔE 0.721 [0.721, 0.721]（n 10）、ΔE の A4 < A2 の対 0/10、符号 +0 / −0 / 0 10（p = NaN）、比の中央値 1 |
| J1b 条件 1（記録、spec:310） | — | 10/10 | — |
| J2 L0 の伸び（spec:314） | **L0_GROWTH_SIMILAR** | ρ2 = 1 | g 15 [15, 15]（n 10） 対 A2 15 [15, 15]（n 10）、g 小の対 0、符号 +0 / −0 / 0 10（p = NaN） |
| J3 L0 の死（spec:322） | **L0_DEATH_PERSISTS** | median d 0.9（境 < 0.45） | d 0.9 [0.9, 0.9]（n 10）、d 小の対 0、符号 +0 / −0 / 0 10（p = NaN）、d ≥ 0.5 の seed 10 |
| J4 後半の動き net5k（spec:329・454） | **MOTION_SMALLER** | ρ4 = 0.0336 | M 0.0161 [0.0161, 0.0161]（n 10） 対 A2 再生 0.479 [0.479, 0.479]（n 10）、符号 +0 / −10 / 0 0（p = 0.00195）、窓の ρ4 early 0.0468, middle NaN, late NaN |
| J5a L1 の早期の死（spec:344） | **L1_EARLY_DEATH** | 10/10 | — |
| J5b 後期の戻り（spec:345） | **L1_REVIVES** | 10/10 | R_k {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 50, 8: 50, 9: 50}、R_k の対の符号 +0 / −0 / 0 10（p = NaN） |
| J6 他 unit 項（spec:359） | **OTHER_TERM_FLIPS** | 0/10（境 ≥ 7） | 後期 s_original_other の A − A2 の符号 +0 / −0 / 0 10（p = NaN） |
| J6-LN（spec:360） | **LN_HALVES** | 10/10 | — |

記録（判定に入れない）:

- J1: ΔE（早期 1〜10）0.721 [0.721, 0.721]（n 10）。10 課題ごとの E1000 の平均の中央値: 0.0458, 0.0458, 0.0458, 0.0458, 0.0458, 0.766, 0.766, 0.766, 0.766, 0.766
- J2: g0_t10 2.27 [2.27, 2.27]（n 10）（A2 2.27 [2.27, 2.27]（n 10））; g0_t50 7.93 [7.93, 7.93]（n 10）（A2 7.93 [7.93, 7.93]（n 10））; g1 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））; g0_alive 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））; g1_alive 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））
- J3: d_91_100 0.86 [0.86, 0.86]（n 10）; d50 0.45 [0.45, 0.45]（n 10）; d_chain100 0.9 [0.9, 0.9]（n 10）; ge099_100 0 [0, 0]（n 10）; 死が初めて 0.5 以上の課題 {0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 56, 6: 56, 7: 56, 8: 56, 9: 56}
- J4（A / A2 再生）: L0_M 0.0161 [0.0161, 0.0161]（n 10） / 0.479 [0.479, 0.479]（n 10）; L0_M_allrows 0.0161 [0.0161, 0.0161]（n 10） / 0.478 [0.478, 0.478]（n 10）; L0_M_net_over_path 0.586 [0.586, 0.586]（n 10） / 0.543 [0.543, 0.543]（n 10）; L0_M_net_over_norm 0.0246 [0.0246, 0.0246]（n 10） / 0.19 [0.19, 0.19]（n 10）; L0_M_rot5k 0.000343 [0.000343, 0.000343]（n 10） / 0.0152 [0.0152, 0.0152]（n 10）; L0_M_net100_end 0.0284 [0.0284, 0.0284]（n 10） / 1.32 [1.32, 1.32]（n 10）; L0_M_net100_allrows 0.0284 [0.0284, 0.0284]（n 10） / 1.31 [1.31, 1.31]（n 10）; L1_M 0.014 [0.014, 0.014]（n 10） / 0.621 [0.621, 0.621]（n 10）; L1_M_allrows 0.014 [0.014, 0.014]（n 10） / 0.317 [0.317, 0.317]（n 10）; L1_M_net_over_path 0.664 [0.664, 0.664]（n 10） / 0.498 [0.498, 0.498]（n 10）; L1_M_net_over_norm 0.0207 [0.0207, 0.0207]（n 10） / 0.181 [0.181, 0.181]（n 10）; L1_M_rot5k 0.000243 [0.000243, 0.000243]（n 10） / 0.0122 [0.0122, 0.0122]（n 10）; L1_M_net100_end 0.0273 [0.0273, 0.0273]（n 10） / 1.14 [1.14, 1.14]（n 10）; L1_M_net100_allrows 0.0273 [0.0273, 0.0273]（n 10） / 0.671 [0.671, 0.671]（n 10）
- J4 の欠けた課題の数（seed ごと）: {0: 27, 1: 27, 2: 27, 3: 27, 4: 27, 5: 27, 6: 27, 7: 27, 8: 27, 9: 27}、A2 再生 {0: 27, 1: 27, 2: 27, 3: 27, 4: 27, 5: 27, 6: 27, 7: 27, 8: 27, 9: 27}
- J4 net100 の A2 の値 L0: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.66 [1.37, 1.78]・全行 1.17 [1.08, 1.78]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit 1.32 [1.32, 1.32]（n 10）・全行 1.31 [1.31, 1.31]（n 10）　(iii) A4 生きた unit 0.0284 [0.0284, 0.0284]（n 10）
- J4 net100 の A2 の値 L1: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.36 [1.22, 1.48]・全行 1.03 [0.462, 1.36]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit 1.14 [1.14, 1.14]（n 10）・全行 0.671 [0.671, 0.671]（n 10）　(iii) A4 生きた unit 0.0273 [0.0273, 0.0273]（n 10）
- J5: L1d50 0.3 [0.3, 0.3]（n 10）; L1d100 0.3 [0.3, 0.3]（n 10）; spearman99 -0.112 [-0.112, -0.112]（n 10）; spearman31 -0.203 [-0.203, -0.203]（n 10）; Spearman 負の seed 99 対 10・31 対 10
- J6 early: s_original_self 1 [0.997, 1]（n 10）; s_original_other 0.813 [0.755, 0.895]（n 10）; s_original_residual 0.852 [0.687, 0.898]（n 10）; s_original_direct_K 0.991 [0.986, 0.996]（n 10）; s_radial 0.00549 [0.00255, 0.0104]（n 10）; s_original_exact_K 0.771 [0.719, 0.798]（n 10）; original_other_abs_share 0.196 [0.169, 0.202]（n 10）; n_alive 460 [442, 482]（n 10）
- J6 middle: s_original_self 0.968 [0.936, 1]（n 10）; s_original_other 0.661 [0.0244, 0.7]（n 10）; s_original_residual 0.643 [0.483, 0.689]（n 10）; s_original_direct_K 0.828 [0.664, 0.868]（n 10）; s_radial 0.0115 [0.00223, 0.344]（n 10）; s_original_exact_K 0.408 [0.392, 0.473]（n 10）; original_other_abs_share 0.314 [0.199, 0.471]（n 10）; n_alive 156 [126, 512]（n 10）
- J6 late: s_original_self 0.997 [0.75, 1]（n 10）; s_original_other 0.148 [0.00586, 0.512]（n 10）; s_original_residual 0.492 [0.476, 0.507]（n 10）; s_original_direct_K 0.6 [0.493, 0.749]（n 10）; s_radial 0.376 [0.176, 0.439]（n 10）; s_original_exact_K 0.485 [0.46, 0.519]（n 10）; original_other_abs_share 0.487 [0.395, 0.59]（n 10）; n_alive 488 [109, 512]（n 10）
- §7.7 縮退した課題（列 distinct_ctx_500upd）: seed ごとの数 {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}、除いた J1 LOSS_PERSISTS（10/10）、除いた J4 MOTION_SMALLER（ρ4 0.0336）
- §7.7 追補 6 の条件 2（judge_m2.py）を満たす seed 0/10。条件 3 は出さない（読みの選択 11）。
- §7.7 位置 r の中央値（窓の平均の seed 間の中央値）: rmed_uniform_L0 1-10 -0.183, 41-50 -1.52, 91-100 -3.18; rmed_uniform_L1 1-10 -1, 41-50 -1, 91-100 -1; rmed_chain_L0 1-10 -0.183, 41-50 -1.52, 91-100 -3.18; rmed_chain_L1 1-10 -1, 41-50 -1, 91-100 -1
- §7.7 chain_stats.csv（10 課題ごと、seed 間の中央値）: alpha [0.0001, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; H [0.0215, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; pi_iters [1e+06, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; pi_last_step_L1 [9.33e-08, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; B [2.49e-05, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; rowsum_err [2.22e-16, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_data [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_probe [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_eval [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; label_nonmode_frac_all [0.0119, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; label_nonmode_frac_16k [0.0119, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; distinct_ctx_all [273, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; distinct_ctx_500upd [237, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; copy_correct_frac_16k [0.653, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; copy_correct_frac_16k_second_half [0.795, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval [0.0219, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se [0.000653, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_minus_H [0.000366, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_data [0.0215, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se_seq [0.000566, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se_truevar [0.000617, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; cpu_task_sec [0.192, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; rec_sec [0.119, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
- §7.7 logit_stats.csv（10 課題ごと、seed 間の中央値）: gap12_med [10.2, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; maxprob_mean [0.987, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; gamma_f_norm [12.9, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; emb_rownorm_med [1.02, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]

## S2 − A2（判定（§12.1 の 1））

使える seed 10/10

| 判定 | ラベル | 数 | 併記 |
| --- | --- | --- | --- |
| J1 喪失（spec:293） | **LOSS_PERSISTS** | LOSS 10/10（境 ≥ 8・≤ 2） | 相ごと T015/E1000/ELast [10, 10, 10]、学習成立 10、ΔE 0.721 [0.721, 0.721]（n 10）、ΔE の S2 < A2 の対 0/10、符号 +0 / −0 / 0 10（p = NaN）、比の中央値 1 |
| J1b 条件 1（記録、spec:310） | — | 10/10 | — |
| J2 L0 の伸び（spec:314） | **L0_GROWTH_SIMILAR** | ρ2 = 1 | g 15 [15, 15]（n 10） 対 A2 15 [15, 15]（n 10）、g 小の対 0、符号 +0 / −0 / 0 10（p = NaN） |
| J3 L0 の死（spec:322） | **L0_DEATH_PERSISTS** | median d 0.9（境 < 0.45） | d 0.9 [0.9, 0.9]（n 10）、d 小の対 0、符号 +0 / −0 / 0 10（p = NaN）、d ≥ 0.5 の seed 10 |
| J4 後半の動き net5k（spec:329・454） | **MOTION_SMALLER** | ρ4 = 0.079 | M 0.0378 [0.0378, 0.0378]（n 10） 対 A2 再生 0.479 [0.479, 0.479]（n 10）、符号 +0 / −10 / 0 0（p = 0.00195）、窓の ρ4 early 0.11, middle NaN, late NaN |
| J5a L1 の早期の死（spec:344） | **L1_EARLY_DEATH** | 10/10 | — |
| J5b 後期の戻り（spec:345） | **L1_REVIVES** | 10/10 | R_k {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 50, 8: 50, 9: 50}、R_k の対の符号 +0 / −0 / 0 10（p = NaN） |
| J6 他 unit 項（spec:359） | **INSUFFICIENT** | 0/0（境 ≥ None） | 後期 s_original_other の A − A2 の符号 +0 / −0 / 0 0（p = NaN） |
| J6-LN（spec:360） | **INSUFFICIENT** | 0/0 | — |

記録（判定に入れない）:

- J1: ΔE（早期 1〜10）0.721 [0.721, 0.721]（n 10）。10 課題ごとの E1000 の平均の中央値: 0.0458, 0.0458, 0.0458, 0.0458, 0.0458, 0.766, 0.766, 0.766, 0.766, 0.766
- J2: g0_t10 2.27 [2.27, 2.27]（n 10）（A2 2.27 [2.27, 2.27]（n 10））; g0_t50 7.93 [7.93, 7.93]（n 10）（A2 7.93 [7.93, 7.93]（n 10））; g1 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））; g0_alive 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））; g1_alive 15 [15, 15]（n 10）（A2 15 [15, 15]（n 10））
- J3: d_91_100 0.86 [0.86, 0.86]（n 10）; d50 0.45 [0.45, 0.45]（n 10）; d_chain100 0.9 [0.9, 0.9]（n 10）; ge099_100 0 [0, 0]（n 10）; 死が初めて 0.5 以上の課題 {0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 56, 6: 56, 7: 56, 8: 56, 9: 56}
- J4（A / A2 再生）: L0_M 0.0378 [0.0378, 0.0378]（n 10） / 0.479 [0.479, 0.479]（n 10）; L0_M_allrows 0.0378 [0.0378, 0.0378]（n 10） / 0.478 [0.478, 0.478]（n 10）; L0_M_net_over_path 0.411 [0.411, 0.411]（n 10） / 0.543 [0.543, 0.543]（n 10）; L0_M_net_over_norm 0.0631 [0.0631, 0.0631]（n 10） / 0.19 [0.19, 0.19]（n 10）; L0_M_rot5k 0.0021 [0.0021, 0.0021]（n 10） / 0.0152 [0.0152, 0.0152]（n 10）; L0_M_net100_end 0.0921 [0.0921, 0.0921]（n 10） / 1.32 [1.32, 1.32]（n 10）; L0_M_net100_allrows 0.0921 [0.0921, 0.0921]（n 10） / 1.31 [1.31, 1.31]（n 10）; L1_M 0.091 [0.091, 0.091]（n 10） / 0.621 [0.621, 0.621]（n 10）; L1_M_allrows 0.091 [0.091, 0.091]（n 10） / 0.317 [0.317, 0.317]（n 10）; L1_M_net_over_path 0.574 [0.574, 0.574]（n 10） / 0.498 [0.498, 0.498]（n 10）; L1_M_net_over_norm 0.107 [0.107, 0.107]（n 10） / 0.181 [0.181, 0.181]（n 10）; L1_M_rot5k 0.00541 [0.00541, 0.00541]（n 10） / 0.0122 [0.0122, 0.0122]（n 10）; L1_M_net100_end 0.262 [0.262, 0.262]（n 10） / 1.14 [1.14, 1.14]（n 10）; L1_M_net100_allrows 0.262 [0.262, 0.262]（n 10） / 0.671 [0.671, 0.671]（n 10）
- J4 の欠けた課題の数（seed ごと）: {0: 27, 1: 27, 2: 27, 3: 27, 4: 27, 5: 27, 6: 27, 7: 27, 8: 27, 9: 27}、A2 再生 {0: 27, 1: 27, 2: 27, 3: 27, 4: 27, 5: 27, 6: 27, 7: 27, 8: 27, 9: 27}
- J4 net100 の A2 の値 L0: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.66 [1.37, 1.78]・全行 1.17 [1.08, 1.78]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit 1.32 [1.32, 1.32]（n 10）・全行 1.31 [1.31, 1.31]（n 10）　(iii) S2 生きた unit 0.0921 [0.0921, 0.0921]（n 10）
- J4 net100 の A2 の値 L1: (i) 登録前の参照（white-san の保存状態 u100 から、出典 results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt・sha256 5eb9d37e420737ab…）生きた unit 1.36 [1.22, 1.48]・全行 1.03 [0.462, 1.36]　(ii) byakko での再生（この台、元の軌道の値ではない）生きた unit 1.14 [1.14, 1.14]（n 10）・全行 0.671 [0.671, 0.671]（n 10）　(iii) S2 生きた unit 0.262 [0.262, 0.262]（n 10）
- J5: L1d50 0.3 [0.3, 0.3]（n 10）; L1d100 0.3 [0.3, 0.3]（n 10）; spearman99 -0.112 [-0.112, -0.112]（n 10）; spearman31 -0.203 [-0.203, -0.203]（n 10）; Spearman 負の seed 99 対 10・31 対 10
- J6 early: s_original_self NaN [NaN, NaN]（n 0）; s_original_other NaN [NaN, NaN]（n 0）; s_original_residual NaN [NaN, NaN]（n 0）; s_original_direct_K NaN [NaN, NaN]（n 0）; s_radial NaN [NaN, NaN]（n 0）; s_original_exact_K NaN [NaN, NaN]（n 0）; original_other_abs_share NaN [NaN, NaN]（n 0）; n_alive NaN [NaN, NaN]（n 0）
- J6 middle: s_original_self NaN [NaN, NaN]（n 0）; s_original_other NaN [NaN, NaN]（n 0）; s_original_residual NaN [NaN, NaN]（n 0）; s_original_direct_K NaN [NaN, NaN]（n 0）; s_radial NaN [NaN, NaN]（n 0）; s_original_exact_K NaN [NaN, NaN]（n 0）; original_other_abs_share NaN [NaN, NaN]（n 0）; n_alive NaN [NaN, NaN]（n 0）
- J6 late: s_original_self NaN [NaN, NaN]（n 0）; s_original_other NaN [NaN, NaN]（n 0）; s_original_residual NaN [NaN, NaN]（n 0）; s_original_direct_K NaN [NaN, NaN]（n 0）; s_radial NaN [NaN, NaN]（n 0）; s_original_exact_K NaN [NaN, NaN]（n 0）; original_other_abs_share NaN [NaN, NaN]（n 0）; n_alive NaN [NaN, NaN]（n 0）
- §7.7 縮退した課題（列 distinct_ctx_500upd）: seed ごとの数 {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}、除いた J1 LOSS_PERSISTS（10/10）、除いた J4 MOTION_SMALLER（ρ4 0.079）
- §7.7 追補 6 の条件 2（judge_m2.py）を満たす seed 0/10。条件 3 は出さない（読みの選択 11）。
- §7.7 位置 r の中央値（窓の平均の seed 間の中央値）: rmed_uniform_L0 1-10 -0.183, 41-50 -1.52, 91-100 -3.18; rmed_uniform_L1 1-10 -1, 41-50 -1, 91-100 -1; rmed_chain_L0 1-10 -0.183, 41-50 -1.52, 91-100 -3.18; rmed_chain_L1 1-10 -1, 41-50 -1, 91-100 -1
- §7.7 chain_stats.csv（10 課題ごと、seed 間の中央値）: alpha [0.01, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; H [0.726, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; pi_iters [270, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; pi_last_step_L1 [8.69e-15, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; B [2.32e-12, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; rowsum_err [7.4e-16, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_data [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_probe [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; zero_drawn_eval [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; label_nonmode_frac_all [0.283, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; label_nonmode_frac_16k [0.283, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; distinct_ctx_all [4.18e+03, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; distinct_ctx_500upd [4.16e+03, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; copy_correct_frac_16k [0.0226, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; copy_correct_frac_16k_second_half [0.0335, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval [0.72, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se [0.00491, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_minus_H [-0.0061, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_data [0.726, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se_seq [0.00482, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; true_ce_eval_se_truevar [0.00495, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; cpu_task_sec [0.199, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; rec_sec [0.154, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
- §7.7 logit_stats.csv（10 課題ごと、seed 間の中央値）: gap12_med [1.46, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; maxprob_mean [0.719, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; gamma_f_norm [16.3, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]; emb_rownorm_med [1.49, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]

## A2 の再生（§6.4・§12.3。この台での再生）

- seed 0: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 1: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 2: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 3: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 4: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 5: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 6: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 7: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 8: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552
- seed 9: complete True、課題 2（T_J4 の欠け [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]）、online のビット一致 0/2・差の最大 0.0236・平均の中央値 0.00322、終わりの状態のビット一致 0/2・相対差の最大 0.552

## S-entropy の記録（Issa の判断 2026-10-24: 止める検査から記録に替えた。判定は変えない。外れた連鎖を除かない・引き直さない）

- A4: 連鎖 30、旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差）で外れ 0、そのうち代案（系列ごとと真の分散の大きい方の標準誤差）でも外れ 0
- S2: 連鎖 30、旧規則（|CE − H| ≤ 4 × トークンごとの標準誤差）で外れ 0、そのうち代案（系列ごとと真の分散の大きい方の標準誤差）でも外れ 0

## 予測の採点（§3、A4 について。数は書き換えない）

| 予測 | 誰 | p | 規則 | 結果 | p(結果) | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| P1（spec:102） | Fable | 0.55 | J1 が LOSS_PERSISTS、かつ ΔE が A4 < A2 の seed が 8/10 以上 | False | 0.45 | 0.303 |
| P1（spec:102） | Opus | 0.35 | J1 が LOSS_PERSISTS、かつ ΔE が A4 < A2 の seed が 8/10 以上 | False | 0.65 | 0.122 |
| P2（spec:103） | Fable | 0.6 | J2 が L0_GROWTH_SMALLER | False | 0.4 | 0.36 |
| P2（spec:103） | Opus | 0.35 | J2 が L0_GROWTH_SMALLER | False | 0.65 | 0.122 |
| P3（spec:104） | Fable | 0.8 | J4 の比 ρ4 ≤ 1/3（net5k、§12.1 の 2） | True | 0.8 | 0.04 |
| P3（spec:104） | Opus | 0.15 | J4 の比 ρ4 ≤ 1/3（net5k、§12.1 の 2） | True | 0.15 | 0.722 |
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
