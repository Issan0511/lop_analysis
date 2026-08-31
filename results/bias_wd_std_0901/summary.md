# bias_wd_std_0901 — 本走の結果

事前登録: [`specs/spec_bias_wd_std_0901.md`](../../specs/spec_bias_wd_std_0901.md)。encoding は std、3腕とも無中心化。

## Verdict

| pred | scope | verdict | evidence | ci_basis | ci_degenerate |
| --- | --- | --- | --- | --- | --- |
| P-main | mean(log10 unfit), B10-B02, S_main/S_none | NUMERIC_DIVERGENCE | S_main stopped by registered S4 at step 3381000 (seeds=[7]); partial trajectory excluded; completed S_none drift +9.101652 dex | not computed |  |
| P-dose | mean(log10 unfit), B10-B02, S_sub/S_none | NUMERIC_DIVERGENCE | S_sub stopped by registered S4 at step 1591000 (seeds=[2]); partial trajectory excluded | not computed |  |
| dead | B10 strict_dead_frac (S_none) | REPORT_ONLY | L1 0.780020; L2 0.890900 |  |  |
| dead | B10 strict_dead_frac (S_main) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| dead | B10 strict_dead_frac (S_sub) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| ledger | alive median channels B02->B10 (S_none, L1) | REPORT_ONLY | M -0.655741->-0.920601, delta -0.264859 CI [-0.344456, -0.189274]; B -0.241193->-0.211662, delta +0.029532 CI [-0.039776, +0.096417] | paired percentile bootstrap | 0 |
| ledger | alive median channels B02->B10 (S_none, L2) | REPORT_ONLY | M -0.801581->-0.883125, delta -0.081544 CI [-0.190999, +0.038819]; B -0.211491->-0.192821, delta +0.018669 CI [-0.090172, +0.091221] | paired percentile bootstrap | 0 |
| ledger | alive median channels B02->B10 (S_main, L1) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| ledger | alive median channels B02->B10 (S_main, L2) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| ledger | alive median channels B02->B10 (S_sub, L1) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| ledger | alive median channels B02->B10 (S_sub, L2) | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| static | B10 mean(log10 unfit), S_main-S_none | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |
| static | B10 mean(log10 unfit), S_sub-S_none | NUMERIC_DIVERGENCE | B10 unavailable | not computed |  |

主判定は **NUMERIC_DIVERGENCE**。B02 = task 51–100、B10 = task 451–500 の `mean(log10 unfit)` 劣化比を seed 対応で評価する設計。

- `S_main` が S4 で停止したため、登録 endpoint の劣化比・対応劣化差は算出不能（部分軌道は除外）
- 完走した `S_none` の B10−B02 劣化は平均 +9.101652 dex

## 終盤窓（task 451–500 = block 10）の腕別水準（seed 平均）

| arm | wd_b | mean_log10_unfit | log10_mean_unfit | L1_strict_dead_frac | L2_strict_dead_frac | L1_M_median_alive | L1_B_median_alive | L2_M_median_alive | L2_B_median_alive | floor_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_none | 0 | -1.81784 | -0.84033 | 0.78002 | 0.8909 | -0.920601 | -0.211662 | -0.883125 | -0.192821 | 0 |
| S_main | 0.001 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| S_sub | 0.1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 台帳移動（alive 中央、B02→B10）

| arm | layer | M_B02 | M_B10 | M_delta | M_ci_lo | M_ci_hi | B_B02 | B_B10 | B_delta | B_ci_lo | B_ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_none | 1 | -0.655741 | -0.920601 | -0.264859 | -0.344456 | -0.189274 | -0.241193 | -0.211662 | 0.0295318 | -0.0397756 | 0.0964172 |
| S_none | 2 | -0.801581 | -0.883125 | -0.0815442 | -0.190999 | 0.038819 | -0.211491 | -0.192821 | 0.0186695 | -0.0901722 | 0.0912213 |

## 集計規約

- 主 endpoint は `mean(log10 unfit)`。床 `1e-23` を各 task 末に当ててから log10 を取り、seed 内50 taskを平均
- paired percentile bootstrap: B=20000, bootstrap_seed=20260903。studentized は退化診断のみ
- `log10(mean unfit)`、dead、M/B 台帳、`S_sub`、B10静的差は REPORT_ONLY

## サニティ

- **S0: PASS**。`S_none` と committed `mlp2_phase1_0829/L2_none` を 30k・1k格子で replay
- **S1/S2: PASS**。lambda=0 の bit identity と、WD が隠れ層 bias だけを触る代数検査
- **S3: PASS**（完走腕。停止腕も S4 検出前の記録点では壁恒等式・1/32量子化・第1層 kappa 閉形式・独立実装一致に違反なし）。beta は前件で修正済みのスケール正規化尺度
- **S4 数値安定性**（probe_every=1000）: S_none PASS; S_main FAIL (step 3381000, seeds=[7]); S_sub FAIL (step 1591000, seeds=[2])

## 事前登録後の実装補正

- S4 で停止した腕の部分ログを除外し、該当 endpoint に `NUMERIC_DIVERGENCE` を出す処理を本走開始後に追加した。判定式・窓・しきい・完走腕の数値には触れていない

## 引いてはいけない線

- `LOP_PERSISTS` でも b-WD が無意味とは書かない。centered では効いている
- `LOP_REMOVED` でも mu 駆動説の棄却まで飛ばず、裁定を Issa に返す
- dead の変化を機能改善・悪化と読み替えない
