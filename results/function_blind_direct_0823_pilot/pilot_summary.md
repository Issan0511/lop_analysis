# function_blind_direct_0823 pilot

> **パイロット専用。以下の効果量・CIを機能盲目性の確認結果として引用しない。**

## 1. サニティ

- 集計sanity: **PASS**
- p̂量子化最大誤差: 0
- x/r幾何最大相対誤差: 4.32e-16
- strict_deadとpre_max符号の不一致: 0
- runner全必須検査: **PASS**
- ΔL逐一消音照合: n=20、最大誤差=3.885780586188048e-16
- 100k probe無擾乱比較: **PASS**
- 旧ratchetログ一致: **PASS**

## 2. リスク集合と転帰

- 曝露: 7,423（seed=10, task=61）
- endpoint strict_dead: 0.2545 (1,889/7,423)
- endpoint dead_0.05: 0.3596

## 3. ΔL 診断

- utility_nmse: min=-4.233, p10=-0.1826, median=+0.008332, p90=+0.6217, max=+33.94
- utility_raw: min=-4.696, p10=-0.2475, median=+0.01227, p90=+0.8896, max=+14.95

## 4. 探索的候補（confirmationの結果ではない）

- unadjusted_t0_tertile/end_strict_dead: RD(high−low)=-0.3224 [-0.3591, -0.2838], rough seed=NA
- unadjusted_t0_tertile/end_dead_0_05: RD(high−low)=-0.3724 [-0.3971, -0.3493], rough seed=NA
- phat_margin_3x3_adjusted/end_strict_dead: RD(high−low)=-0.2344 [-0.2749, -0.2022], rough seed=NA
- phat_margin_3x3_adjusted/end_dead_0_05: RD(high−low)=-0.2615 [-0.2928, -0.2314], rough seed=NA

この節を見てconfirmation specを固定する。好都合な候補だけを主解析にしない。

## 5. 限界

- pilotはgenerator_offset=0で旧軌道を再計装したもので、独立確認ではない。
- S2の100kという長さはpilot specで数値固定していなかった。confirmationでは実行前に固定する。
- ΔLは現在タスク上の単独消音効果。冗長性、相互作用、将来タスク価値を表さない。
- 同一unitの反復曝露がある。unit独立のSEを使わない。
