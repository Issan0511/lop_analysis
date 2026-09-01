# channel_2x2_0901 — チャネル遮断 2×2 本走

事前登録: [`specs/spec_channel_2x2_0901.md`](../../specs/spec_channel_2x2_0901.md)（repo commit `31f3792`）。

主判定は **TWO_CHANNELS_BOTH_NECESSARY**。4腕共通の完走 seed = [0, 1, 2, 3, 4, 5, 6, 8, 9] (n=9)。

窓は B02 = task 51–100、B10 = task 451–500。主 endpoint は `mean(log10 unfit)` の B10−B02、床は 1e-23。

## 事前予測との対応

| item | prediction | result | match |
| --- | --- | --- | --- |
| 主判定 | TWO_CHANNELS_BOTH_NECESSARY | TWO_CHANNELS_BOTH_NECESSARY | 一致 |
| both E-drift ±0.15内 | yes | yes | 一致 |
| 交互作用 E-drift | SUBADDITIVE | SUPERADDITIVE | 外れた |
| E-level 最良 | bwd | bwd | 一致 |
| 外れた場合の改稿先 | わからない | 本走から自動決定しない | N/A |

## 判定

| pred | scope | verdict | evidence | ci_basis | ci_degenerate |
| --- | --- | --- | --- | --- | --- |
| P-main | E-drift = mean(log10 unfit) B10-B02 | TWO_CHANNELS_BOTH_NECESSARY | common complete seeds=[0, 1, 2, 3, 4, 5, 6, 8, 9]; n=9; both drift -0.026062 CI [-0.075295, +0.021083]; ci_degenerate=False; bwd drift +2.060933 CI [+1.127182, +3.091682]; ci_degenerate=False; bwd-none -6.116520 CI [-7.374467, -4.660924]; ci_degenerate=False; both-bwd -2.086995 CI [-3.126430, -1.142651]; ci_degenerate=False; conditions={'a': True, 'b': True, 'c': True, 'c_i': True, 'c_ii': True}; S-floor=PASS | paired percentile | 0 |
| S-floor | B02/B10 floor_frac, all four cells | PASS | none/B02=0; none/B10=0; bwd/B02=0; bwd/B10=0; cen/B02=0; cen/B10=0; both/B02=0; both/B10=0 |  |  |
| S-ceiling | B02 four-cell level range | CEILING_CONTAMINATED | range=8.980274 dex; threshold=3.0; levels={'none': -11.601911141093385, 'bwd': -7.26134183527655, 'cen': -2.621637607605784, 'both': -2.757324148122739} |  |  |
| L | E-drift/E-level ladder | LADDER_INVERTS | drift rank=['both', 'cen', 'bwd', 'none']; E-level rank=['bwd', 'none', 'both', 'cen']; B10={'none': -3.4244574167072295, 'bwd': -5.200408500559412, 'cen': -2.171918654825814, 'both': -2.7833862041338016} |  |  |
| E-level | none B10 tasks 451-500 | REPORT_ONLY | -3.424457 CI [-4.126955, -2.784164]; ci_degenerate=False | paired percentile | 0 |
| E-level | bwd B10 tasks 451-500 | REPORT_ONLY | -5.200409 CI [-6.058111, -4.221610]; ci_degenerate=False | paired percentile | 0 |
| E-level | cen B10 tasks 451-500 | REPORT_ONLY | -2.171919 CI [-2.269729, -2.066196]; ci_degenerate=False | paired percentile | 0 |
| E-level | both B10 tasks 451-500 | REPORT_ONLY | -2.783386 CI [-2.800831, -2.766521]; ci_degenerate=False | paired percentile | 0 |
| I | (bwd-none)-(both-cen), E-drift | SUPERADDITIVE | -5.640739 CI [-6.848781, -4.217600]; ci_degenerate=False | paired percentile | 0 |
| I | (bwd-none)-(both-cen), E-level | INCONCLUSIVE_WIDE | -1.164484 CI [-2.113677, -0.145000]; ci_degenerate=False | paired percentile | 0 |
| R | L1 eff_rank B10, both-cen | SATURATION_PREVENTED | +12.722975 CI [+12.397145, +13.036054]; ci_degenerate=False | paired percentile | 0 |
| D | none L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.659889->0.778933; delta=+0.119044 |  |  |
| ledger | none L1 M_median_alive B02->B10 | REPORT_ONLY | -0.656855->-0.958922; -0.302067 CI [-0.369180, -0.228568]; ci_degenerate=False | paired percentile | 0 |
| ledger | none L1 B_median_alive B02->B10 | REPORT_ONLY | -0.240487->-0.233844; +0.006642 CI [-0.022282, +0.035509]; ci_degenerate=False | paired percentile | 0 |
| D | none L2 strict_dead_frac B02->B10 | REPORT_ONLY | 0.503200->0.851089; delta=+0.347889 |  |  |
| ledger | none L2 M_median_alive B02->B10 | REPORT_ONLY | -0.675920->-0.681462; -0.005542 CI [-0.122497, +0.145599]; ci_degenerate=False | paired percentile | 0 |
| ledger | none L2 B_median_alive B02->B10 | REPORT_ONLY | -0.261166->-0.379033; -0.117867 CI [-0.281127, +0.026731]; ci_degenerate=False | paired percentile | 0 |
| D | bwd L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.588156->0.664978; delta=+0.076822 |  |  |
| ledger | bwd L1 M_median_alive B02->B10 | REPORT_ONLY | -0.921492->-1.191295; -0.269803 CI [-0.364966, -0.197051]; ci_degenerate=False | paired percentile | 0 |
| ledger | bwd L1 B_median_alive B02->B10 | REPORT_ONLY | -0.011727->+0.006149; +0.017876 CI [+0.014878, +0.020903]; ci_degenerate=False | paired percentile | 0 |
| D | bwd L2 strict_dead_frac B02->B10 | REPORT_ONLY | 0.348289->0.495756; delta=+0.147467 |  |  |
| ledger | bwd L2 M_median_alive B02->B10 | REPORT_ONLY | -0.948660->-1.165157; -0.216497 CI [-0.344082, -0.094799]; ci_degenerate=False | paired percentile | 0 |
| ledger | bwd L2 B_median_alive B02->B10 | REPORT_ONLY | -0.011799->+0.004558; +0.016358 CI [+0.005763, +0.027134]; ci_degenerate=False | paired percentile | 0 |
| D | cen L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.026933->0.016867; delta=-0.010067 |  |  |
| ledger | cen L1 M_median_alive B02->B10 | REPORT_ONLY | +0.000737->-0.000564; -0.001301 CI [-0.002783, +0.000273]; ci_degenerate=False | paired percentile | 0 |
| ledger | cen L1 B_median_alive B02->B10 | REPORT_ONLY | +0.195203->+14.104361; +13.909158 CI [+10.365740, +17.754516]; ci_degenerate=False | paired percentile | 0 |
| D | cen L2 strict_dead_frac B02->B10 | REPORT_ONLY | 0.101200->0.612311; delta=+0.511111 |  |  |
| ledger | cen L2 M_median_alive B02->B10 | REPORT_ONLY | +0.000398->+0.004218; +0.003820 CI [-0.003020, +0.010999]; ci_degenerate=False | paired percentile | 0 |
| ledger | cen L2 B_median_alive B02->B10 | REPORT_ONLY | -0.674724->-1.149522; -0.474799 CI [-0.531157, -0.419317]; ci_degenerate=False | paired percentile | 0 |
| D | both L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.000511->0.000000; delta=-0.000511 |  |  |
| ledger | both L1 M_median_alive B02->B10 | REPORT_ONLY | -0.000371->-0.000559; -0.000188 CI [-0.000987, +0.000553]; ci_degenerate=False | paired percentile | 0 |
| ledger | both L1 B_median_alive B02->B10 | REPORT_ONLY | +0.163255->+0.103659; -0.059596 CI [-0.146260, +0.017158]; ci_degenerate=False | paired percentile | 0 |
| D | both L2 strict_dead_frac B02->B10 | REPORT_ONLY | 0.001533->0.000000; delta=-0.001533 |  |  |
| ledger | both L2 M_median_alive B02->B10 | REPORT_ONLY | +0.002435->-0.000596; -0.003031 CI [-0.005766, -0.000162]; ci_degenerate=False | paired percentile | 0 |
| ledger | both L2 B_median_alive B02->B10 | REPORT_ONLY | -0.212020->-0.146341; +0.065679 CI [+0.035577, +0.094126]; ci_degenerate=False | paired percentile | 0 |
| exclusion | none | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | bwd | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | cen | ARM_VALID | status=COMPLETE_WITH_EXCLUSIONS; excluded=[7]; included=[0, 1, 2, 3, 4, 5, 6, 8, 9] |  |  |
| exclusion | both | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |

## B02 / B10 水準

| arm | window | mean_log10_unfit | log10_mean_unfit | L1_dead | L2_dead | L1_eff_rank | floor_frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | B02 | -11.6019 | -5.54975 | 0.659889 | 0.5032 | 14.017 | 0 |
| none | B10 | -3.42446 | -1.17435 | 0.778933 | 0.851089 | 10.9127 | 0 |
| bwd | B02 | -7.26134 | -5.5903 | 0.588156 | 0.348289 | 15.0041 | 0 |
| bwd | B10 | -5.20041 | -2.29627 | 0.664978 | 0.495756 | 14.4704 | 0 |
| cen | B02 | -2.62164 | -2.53516 | 0.0269333 | 0.1012 | 11.2281 | 0 |
| cen | B10 | -2.17192 | -2.02015 | 0.0168667 | 0.612311 | 3.1317 | 0 |
| both | B02 | -2.75732 | -2.66724 | 0.000511111 | 0.00153333 | 15.5058 | 0 |
| both | B10 | -2.78339 | -2.71539 | 0 | 0 | 15.8547 | 0 |

## フラグ

- `CEILING_CONTAMINATED`: B02 の4セル間差が3 dexを超えたため E-drift 単独では読まない。
- `LADDER_INVERTS`: E-drift と E-level の順位が異なるため、どちらも単独では引かない。

## 解釈上の制限

- 交互作用は事前登録の字義どおり `(bwd-none)-(both-cen)` で計算し、-5.640739 dex の `SUPERADDITIVE` だった。
- spec §2.1 の旧配置 `+5.889` と §8.1 の `SUBADDITIVE` 予測は、§6.3 に固定した式とは符号規約が逆である。結果後に符号を反転せず、字義どおりの式とラベルを維持した。
- centered セルは B02 水準が低く、落ちる余地の差だけでも劣加法が生じる。本走はこれを分離しない。
- EMA 中心化は µ とタスク可識別性を同時に消すため、『µ を消した』とは書かない。
- `strict_dead` は REPORT_ONLY で、主判定には使っていない。
- スコープは condA・幅100・hidden [100,100]・T=10^4・batch=1・lr=0.01・plain SGD・5M・lambda=1e-3。
