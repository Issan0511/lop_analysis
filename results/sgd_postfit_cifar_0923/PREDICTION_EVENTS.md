# sgd_postfit_cifar_0923 — 予測する事象（Claude と Codex が独立に確率を付ける）

定義はすべて `specs/spec_sgd_postfit_cifar_0923.md` v2 の §3–§4 のとおり（失敗込みの主規則、上限付き hit999、ce_st の e-fold、中央値の比、0.25/0.75 の境界）。三分岐以上の問いは確率の和を 1 にする。二値の問いは「はい」の確率を付ける。

| # | 事象 | 形式 |
|---|---|---|
| E0 | 参照差が成立する: Ĥ_A − Ĥ_F ≥ 600 かつ Ĝ_A − Ĝ_F ≥ 10,000 | 二値 |
| E1 | Q1（S_hi = 0.03、E1、失敗込み）: SGD_RESCUE / PARTIAL / NO_RESCUE | 三分岐 |
| E2 | Q2（S_lo = 0.001、E1）: SGD_RESCUE / PARTIAL / NO_RESCUE | 三分岐 |
| E2b | Q2b（S_mid = 0.003、E1）: SGD_RESCUE / PARTIAL / NO_RESCUE | 三分岐 |
| E3 | Q3（S_hi、E2）: SGD_RESCUE_W / PARTIAL_W / NO_RESCUE_W | 三分岐 |
| E4 | Q4（AR、E1）: RESET_RESCUE / PARTIAL / NO_RESCUE | 三分岐 |
| E5 | Q5（AR、E2）: RESET_RESCUE_W / PARTIAL_W / NO_RESCUE_W | 三分岐 |
| E6 | Q7（A_ce 対 参照腕: S_hi、生存 seed が 8 本未満なら S_mid、それも未満なら S_lo）: MATCHED_CE_HARMLESS / MATCHED_CE_HARMFUL / ADAM_WORSE_AT_MATCHED_CE / SGD_WORSE_AT_MATCHED_CE / MIXED（+MATCH_INSUFFICIENT は別に数えず、付いたら E6 は採点から外す） | 五分岐 |
| E7 | S_hi の本走で、50 課題のどこかで 1 slot 以上が発散する | 二値 |
| E7b | S_hi の本走で、5 slot 以上が発散する | 二値 |
| E7c | S_mid の本走で、1 slot 以上が発散する | 二値 |
| E8 | S_mid の efolds_mid（seed ごとの t26–50 の ce_st e-fold 中央値、の seed 中央値）< 1.0 | 二値 |
| E9 | η 依存: efolds_mid(S_mid) > efolds_mid(S_lo) かつ Ĝ(S_mid) > Ĝ(S_lo) | 二値 |
| E10 | 伸びと速さの乖離: S_mid か AR のどちらかで、ρ と ρ^W の片方が ≥ 0.75、もう片方が ≤ 0.25 | 二値 |
| E11 | Q8: 状態の効果（W = 課題末で固定）の 90% 区間が 0 を含まない | 二値 |
| E12 | Q8: 重みの効果（状態 = s_sw で固定）の 90% 区間が 0 を含まず、かつ平均 > 0（Adam の 1 課題ぶんの漂流が次の当てはめを遅くする） | 二値 |
| E13 | C2: A が LR_iid に 50 課題全部で bit 一致する | 二値 |
| E14 | Q6: seed 0–4 の Ĥ_F と停止チェーンの 2,100 の差が ±300 以内 | 二値 |
| E15 | ρ(S_mid) の点推定の値（数値予測: 中央値と 80% 区間） | 数値 |
| E16 | Ĝ(S_mid) の値（数値予測: 中央値と 80% 区間） | 数値 |
