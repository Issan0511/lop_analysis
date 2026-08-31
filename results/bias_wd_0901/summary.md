# bias_wd_0901 — 本走の結果

事前登録: [`specs/spec_bias_wd_0901.md`](../../specs/spec_bias_wd_0901.md)。lambda グリッドは `results/bias_wd_pilot_0901/grid_selection.json` が凍結済み規則で決めた値。

## Verdict

| pred | scope | verdict | dead_only_flag | evidence | n_seeds_below_dead_threshold | cp95_lo | cp95_hi | ci_basis | ci_degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P-main | tasks 451-500 (block 10) W1_main lambda=0.001 | BIAS_WD_PROTECTS | 0 | (a) dead 0.463940 -> 0.010480 (threshold 0.232; ratio 0.0226; paired -0.453460 CI [-0.502681, -0.401400]); (b) mean(log10 unfit) -2.5172 -> -2.7557, paired -0.2385 CI [-0.2691, -0.2082] vs margin +0.10; (c) B10-B02 drift diff -0.1323 CI [-0.1811, -0.0844] | 10 | 0.691503 | 1 | percentile | 0 |
| P-dose | block 10 W1_sub1 lambda=0.0001 | REPORT_ONLY | 0 | dead 0.089680; mean(log10 unfit) -2.6743; B10-B02 drift diff vs none -0.1324 |  |  |  |  |  |
| P-dose | block 10 W1_sub2 lambda=0.01 | REPORT_ONLY | 0 | dead 0.000360; mean(log10 unfit) -2.7360; B10-B02 drift diff vs none -0.1436 |  |  |  |  |  |
| P-dose | block 10 W1_sub3 lambda=0.1 | REPORT_ONLY | 0 | dead 0.000120; mean(log10 unfit) -2.3410; B10-B02 drift diff vs none -0.1872 |  |  |  |  |  |
| W2 | block 10 layer-1 activation eff_rank (W2_Aall_main) | SATURATION_PREVENTED | 0 | none B02 10.9413 -> none B10 3.1370; lambda arm B10 15.8712; keep>= 0.70x B02 = 7.6589; paired +12.6238 CI [+12.0217, +13.1798] |  |  |  | percentile | 0 |
| W3 | W1_main layer-1 alive median margin (kappa*sigma - \|b\|), B02->B10 | FLAT | 0 | +0.62179 -> +0.60540; paired -0.01639 CI [-0.03832, +0.00762]; W1_none same contrast +0.33599 CI [+0.28342, +0.39230] |  |  |  | percentile | 0 |
| W4 | block 10 layer-1 alive p_hat>=30/32 (W2_Aall_none) | REPORT_ONLY | 0 | sat_frac B02 0.1926 -> B10 0.6775; alive median b B10 +3.3681 |  |  |  |  |  |
| W4 | block 10 layer-1 alive p_hat>=30/32 (W2_Aall_main) | REPORT_ONLY | 0 | sat_frac B02 0.0384 -> B10 0.0314; alive median b B10 +0.0957 |  |  |  |  |  |
| W4 | block 10 layer-1 alive p_hat>=30/32 (W2_Aall_sub) | REPORT_ONLY | 0 | sat_frac B02 0.0000 -> B10 0.0000; alive median b B10 +0.0002 |  |  |  |  |  |
| S5 | tautology guard (W1_sub3, strongest W1 lambda) | REPORT_ONLY | 0 | alive median b B10 -0.00100; dead 0.000120; geometric-mean unfit (10^mean_log10_unfit, averaged over seeds) 0.00458251; frozen 参照 dead 0 / unfit 0.228112 は代数的帰結であり証拠ではない |  |  |  |  |  |

## 終盤窓（task 451–500 = block 10）の腕別水準（seed 平均）

| arm | wd_b | L1_strict_dead_frac | mean_log10_unfit | log10_mean_unfit | L1_b_median_alive | L1_wall_frac | L1_margin_median_alive | L1_eff_rank | L1_p_hat_sat_frac | floor_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1_none | 0 | 0.46394 | -2.5172 | -2.4034 | -0.807013 | 0.461293 | 0.96658 | 19.4721 | 0.00197017 | 0 |
| W1_main | 0.001 | 0.01048 | -2.75573 | -2.65291 | -0.0996895 | 0.191436 | 0.605402 | 19.1687 | 0.00192215 | 0 |
| W1_sub1 | 0.0001 | 0.08968 | -2.67426 | -2.5736 | -0.383595 | 0.380432 | 0.673053 | 20.5218 | 0.00215744 | 0 |
| W1_sub2 | 0.01 | 0.00036 | -2.73601 | -2.63422 | -0.010592 | 0.0885739 | 0.508845 | 17.8068 | 4e-05 | 0 |
| W1_sub3 | 0.1 | 0.00012 | -2.34105 | -2.24339 | -0.00104176 | 0.0487538 | 0.503343 | 16.9496 | 0 | 0 |
| W2_Aall_none | 0 | 0.01982 | -2.21115 | -2.04595 | 3.25908 | 6.03907 | -2.55936 | 3.27063 | 0.68436 | 0 |
| W2_Aall_main | 0.001 | 0 | -2.78553 | -2.70998 | 0.0957684 | 0.0588624 | 1.85222 | 15.8944 | 0.03826 | 0 |
| W2_Aall_sub | 0.1 | 0 | -2.75724 | -2.68754 | 0.0002041 | 0.0273542 | 2.07974 | 15.831 | 0 | 0 |

## 読み方

- 主判定は **BIAS_WD_PROTECTS**（(a) 死 0.46394 → 0.01048 = 44 分の 1、10 seed 中 10 本がしきい 0.232 を下回る / (b) `mean(log10 unfit)` は同値マージン内どころか **0.239 dex 改善**（1.73 倍）/ (c) 劣化も有意に小さい）。**死の抑制と静的水準の保持がトレードオフになっていない**
- 用量反応は**上に凸**。死は $\lambda$ とともに単調に減るが、`mean(log10 unfit)` は 1e-4 → 1e-3 → 1e-2 で改善したあと **1e-1 で対照より悪化する**（−2.517 / −2.674 / −2.756 / −2.736 / −2.341）。静的コストが現れるのは最強水準だけで、主 $\lambda$=1e-3 はその手前にある
- **W3 の `FLAT` を「保護が働いていない」と読んではいけない。** 対照 `W1_none` 側のマージンが広がる（+0.336）のは、壁に達したユニットが dead 側へ抜けて alive 母集団から外れるためで、**生存者バイアスを含む**。$\lambda$=1e-3 では死ぬのが 1% なので alive 母集団はほぼ層全体であり、両者は同じ母集団を見ていない。W3 は 2 腕の直接比較には使えない
- **frozen の静的コストは払っていない。** 最強水準 $\lambda$=1e-1 でも 幾何平均 `unfit` は 0.00458 で、`centered_freeze_0901` の frozen 腕 0.228112 の 1/50 である。$b$ を 0 に固定するのと、$b$ に有限の復元力を与えるのは別物である
- **W2（上方暴走）側も止まっている。** 対照の第1層は alive 中央 $b$ が +3.27、常時発火率 0.68、活性 `eff_rank` 10.94 → 3.14 まで潰れるのに対し、$\lambda$=1e-3 では $b$ +0.096、常時発火率 0.031、`eff_rank` 15.9 で **B02 水準を上回る**。同じ 1 つの knob が下向きと上向きの 2 病理を同時に止めている

## 集計の約束

- 主判定に使うのは **`mean(log10 unfit)`**（seed 内でブロック内 task 末の log10 を平均）。`log10(mean unfit)` も上表に併記するが判定には使わない
- 床は系ごとに別。深さ1系 `1e-16`（`dose_const_5m_0830` の S6 較正を継承）、深さ2系 `1e-23`（`mlp2_phase1_0829` の S6 較正を継承）。本走では再較正しない
- ブロックは 50 task 刻み。B02 = task 51–100、B10 = task 451–500
- CI は seed 水準の paired percentile bootstrap（B=20000、seed 20260902）。studentized も計算して `ci_degenerate` を出すが、**主は percentile**（この repo では Phase 0b 以降ほぼ全行で studentized が退化する）
- 二値割合の CI は Clopper–Pearson

## サニティ

- **S0**: `W1_none` は committed `L1w100_A1`、`W2_Aall_none` は `L2_Aall` と 30k・1k 格子で `unfit`・`eval_loss_exact` の差ちょうど 0、各層 `strict_dead_frac` も完全一致
- **S1/S2**: $\lambda=0$ 経路は無 WD 実装と bitwise 一致。$W$・$v$・$c$ は $\lambda$ を変えても bitwise 一致で、$b$ の差は $-\eta\lambda b$ とfloat32 の丸め内で一致。`nets.py` を AST で読み、`wd_b` を参照する更新行が `self.b` / `self.bs[i]` の 1 本ずつだけであることも確認
- **S3**: 全 8 腕・501 記録点で壁恒等式違反 0、$\hat p$ の 1/32 量子化違反 0、凍結済み `exact_layer_record_p1` との独立実装一致は最大 3.2e-14（許容 1e-10）。**ただし $\beta$ の一致尺度は事前登録から変更している**（下記）
- **S4**: 非有限は 1 腕も出ていない（`probe_every=1000`）
- **S5**: 恒真ガードは `verdict.csv` の S5 行に REPORT_ONLY で出す。高 $\lambda$ 端の収束は予測の確認であって証拠ではない

### ★ 事前登録からの逸脱（S3 の一致尺度）

- spec §6 S3 は「独立実装一致（許容 1e-10）」とだけ書いており、第1回の走では**要素ごとの相対誤差**で測っていた。$\beta$ は符号を変えながら 0 を通過する量なので、この尺度は分母ゼロで発散する（$\lambda$ が大きく $b\to0$ の腕ほど顕著）
- step 5M のチェックポイントで直接確かめたところ、全 8 腕・全層で絶対誤差は 4e-16〜1.2e-12、$\beta$ のスケールで割ると 2.5e-16〜7.3e-15（float64 の数 ULP）。相対誤差が大きく出るのは $|\beta|$ が 1e-4〜1e-2 のユニットだけだった
- 分母が 0 に近づかない量はすべて ULP 一致している: `unfit` は全記録点で誤差ちょうど 0、`p_hat` は `torch.equal` で完全一致、$\sigma$ 3.5e-15、$\kappa$ の閉形式 9.9e-14
- 対処: $\beta$ の一致尺度を `max|a-b| / max|b_ref|` に変え、要素ごとの相対誤差は `beta_elementwise_rel` として診断だけ残した。指標だけを直して**全 8 腕を同一 config で再走**しており、shard CSV は第1回と sha256 が完全一致する（この走が決定的であることの確認も兼ねる）。**結果の数値は 1 つも動いていない**

## 引いてはいけない線（HANDOFF §7）

1. 高 $\lambda$ 端で dead が 0 になることを証拠にしない。$b\equiv0$ かつ centered なら task 末に消灯できないのは恒等式の帰結であって観測ではない
2. 「WD が LoP を治す」と一般に書かない。スコープは condA・centered・幅100・$T=10^4$・batch=1・lr=0.01・5M に限る
3. std 腕へ外挿しない（台帳の逃げ道が std にはある）
4. 新規性は「WD が効く」ではなく「**$b$ だけの減衰で足りるか**」に置く。先行（`docs/lit_bias_wd_0901.md`）では全パラメータ L2 が dead を上げ effective rank を下げており、符号が逆である
5. `strict_dead` の低下を機能改善と読み替えない。(a) と (b)(c) は独立に読む
6. パイロット（`results/bias_wd_pilot_0901/`）の数値を結果として引用しない
