# function_blind_direct_0823 confirmation

> generator_offset=20260830の独立20系列。pilotとは合算していない。

## 主判定

- **PROTECTIVE**
- 調整RD (high−low): -0.2353 [-0.2812, -0.2325]
- 意味のある差の境界: ±0.05
- low/high pooled率（記述値）: 0.4018 / 0.1668

判定規則は EQUIV → PROTECTIVE → HARMFUL → INCONCLUSIVE の順に排他的に適用した。主判定は strict_dead × utility_nmse × exact p_count・pre_max五分位だけである。

## 事前登録した意味

- この **PROTECTIVE** 判定は、作業6の操作的定義の下で「選抜は機能を見ない」を否定する。
- 柱3全体、Oで確認した保持容量、壁機構まで否定する結果ではない。

## データとセル

- 曝露: 15,582 （seed=20, t0=61）
- 幾何セル: 6,002、有効 2,839、除外 3,163
- 無効セルに属する除外曝露: 3,163
- bootstrap: seed block B=10,000, RNG seed=20260831, nonfinite=0

## 固定副次解析

- S1_primary_cells_dead_0_05: RD=-0.2544 [-0.2941, -0.2490] (PROTECTIVE; secondary)
- S2_margin10_sensitivity: RD=-0.2405 [-0.2770, -0.2312] (PROTECTIVE; secondary)
- S3_unadjusted_t0_tertiles: RD=-0.3171 [-0.3427, -0.2943] (PROTECTIVE; secondary)
- S4_utility_raw: RD=-0.2332 [-0.2797, -0.2305] (PROTECTIVE; secondary)
- utility sign negative: rate=0.3942 (2472/6271)
- utility sign zero: rate=nan (0/0)
- utility sign positive: rate=0.1392 (1296/9311)
- S6_epoch_primary_rd/200-390k: RD=-0.2116 [-0.2701, -0.2013] (PROTECTIVE; secondary)
- S6_epoch_primary_rd/400-590k: RD=-0.2476 [-0.3157, -0.2231] (PROTECTIVE; secondary)
- S6_epoch_primary_rd/600-800k: RD=-0.2717 [-0.3347, -0.2497] (PROTECTIVE; secondary)

副次解析は主結果の差し替えに使わない。

## サニティ

- C-S1〜C-S8: **PASS**
- runner implementation: `647316c`
- 全CSVは入力・commit・RNGを固定した独立二重集計でbyte一致。

## 解釈範囲

- ΔLは現在タスク32入力上の単独消音損失であり、普遍的価値や将来タスク価値ではない。
- 観察解析であり、機能を人工的に入れ替えた因果介入ではない。
- exact p_countとpre_max分位で層別した範囲を越えて交絡消失を主張しない。
- condB、他幅、他教師、長期将来へ外挿しない。
