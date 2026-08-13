# rank_int_0814 summary (spec_rank_int_0814 §6 事前登録判定)

## サニティ

- S1 (resume bit 一致) / S2 (介入の数値保証): [{'width': 10, 'S1': 'PASS', 'S2': 'PASS'}, {'width': 20, 'S1': 'PASS', 'S2': 'PASS'}]
- S2 最大誤差: sv_preserved 1.16e-15, dF_match 4.56e-07, normF 7.03e-16 (許容 1e-06)
- svdrec ε clipped (target 到達不能): 0/10 件, shuffle abort: 0/10 件
- **逸脱**: shuffle の G が単発抽選で ΔF 目標に届かず、同一 generator 列からの棄却サンプリングで再抽選した seed: w20/s4 (n_draw=2)

## Phase 0 ラベル分布 (回復/予防 — 結論の主張文言を規定)

width  label
10     予防       2
       回復       3
20     予防       5

注: 仕様字義の離陸定義は退化 (phase0_summary.md 参照)。ラベルはロバスト定義。
- t_int=150k 適格性: srank t50 通過 10/10, dead≤0.15 は 2/10 (不適格も除外せず全 seed 使用)

## 主判定 (M = 介入後 20 タスクの online loss、paired seed bootstrap 95%CI)

 width metric                 pair     mean        lo       hi  excl_zero
    10      M          svdrec-none 0.170416  0.054030 0.289473       True
    10      M         shuffle-none 0.020981 -0.058392 0.100355      False
    10      M       svdrec-shuffle 0.149435  0.071379 0.218726       True
    10      M    svdrec_alive-none 0.138994  0.028969 0.278613       True
    10      M svdrec_alive-shuffle 0.118013  0.068911 0.181082       True
    20      M          svdrec-none 0.155573  0.054750 0.277365       True
    20      M         shuffle-none 0.016965 -0.038525 0.085798      False
    20      M       svdrec-shuffle 0.138608 -0.009027 0.301247      False
    20      M    svdrec_alive-none 0.123776  0.030073 0.217479       True
    20      M svdrec_alive-shuffle 0.106812 -0.011146 0.241355      False


### width 10
- P-int-1 (svdrec < none): 不成立 (diff 0.1704, CI [0.05403, 0.2895])
- P-int-2 (摂動一般で説明不能): 成立 (shuffle−none CI [-0.05839, 0.1004], svdrec−shuffle CI [0.07138, 0.2187])
- **判定表**: **判定表想定外**: svdrec が none より有意に悪化 (ランク回復介入は有害)。ランク因果 (ランク回復→可塑性回復) は不支持
- G1 (介入直後 Δdead svdrec−shuffle ±0.05 内): FAIL → dead 復活交絡を疑い感度分析を追試 (CI [0.040, 0.220])
- P-int-3 (dead 増分 svdrec<none ∧ shuffle≈none): 不成立 (svdrec−none CI [-0.040, 0.380], shuffle−none CI [-0.040, 0.240])
- 反証条件 (80%回復 済 かつ M/dead とも none と同等): 非該当

### width 20
- P-int-1 (svdrec < none): 不成立 (diff 0.1556, CI [0.05475, 0.2774])
- P-int-2 (摂動一般で説明不能): 成立 (shuffle−none CI [-0.03852, 0.0858], svdrec−shuffle CI [-0.009027, 0.3012])
- **判定表**: **判定表想定外**: svdrec が none より有意に悪化 (ランク回復介入は有害)。ランク因果 (ランク回復→可塑性回復) は不支持
- G1 (介入直後 Δdead svdrec−shuffle ±0.05 内): FAIL → dead 復活交絡を疑い感度分析を追試 (CI [0.150, 0.340])
- P-int-3 (dead 増分 svdrec<none ∧ shuffle≈none): 不成立 (svdrec−none CI [0.040, 0.340], shuffle−none CI [0.060, 0.380])
- 反証条件 (80%回復 済 かつ M/dead とも none と同等): 非該当

- 両 width の svdrec−none 符号一致: 一致 (頑健)

## 結論と所見

+200k 時点のアーム別平均 (srank_alive / dead_frac):

      dead_end                             srank_end                            
arm       none shuffle svdrec svdrec_alive      none shuffle svdrec svdrec_alive
width                                                                           
10        0.36    0.42   0.58         0.72     1.894   1.876  1.971        1.631
20        0.31    0.35   0.57         0.49     2.230   2.230  2.433        2.510

1. **ランク回復は一過性**: svdrec / svdrec_alive は介入直後に srank_target 付近まで
   回復するが、訓練の継続で ~100k step かけて none と同水準まで再崩壊する
   (fig_ri_series 上段)。低ランク整列はこの regime の学習力学のアトラクターであり、
   スペクトルだけ戻しても維持されない。
2. **svdrec 系は dead をむしろ加速**: P-int-3 の予測 (svdrec が dead 蓄積を抑える) の
   逆で、介入直後および +200k の dead_frac は none より高い (fig_ri_series 下段)。
   ノルム保存の一様再スケールが支配方向 (学習済み解) を縮め、持ち上げられた
   小特異値方向はタスクと不整合なため、SGD がユニットごと殺す方向に働くと解釈できる。
3. **判定**: 両幅・両ラベル (回復/予防) を通じて svdrec (alive 限定版含む) は M を有意に
   悪化させ、等 ΔF の shuffle は none と区別できない。事前登録の判定表では
   「ランク因果不支持」側だが、想定された「全アーム同等」ではなく
   **ランク回復介入が積極的に有害**という、より強い形の不支持である。
   反証条件の字義 (「none と区別できない」) には該当しないが、
   「低ランクは LoP の原因ではなく随伴症状」という結論は M の悪化方向によって
   さらに強く支持される。

## 感度分析 (G1 破れ時の追試, spec §6): alive 行のみ SVD 介入 (svdrec_alive)

- 介入直後 Δdead: 平均 +0.110 (svdrec 本体は dead 行も再構成するため直後 Δdead が正になりがちだが、alive 限定版は dead 行を触らない)
- 介入直後 srank_alive 回復: w10/s0 2.12→3.28(target 3.28), w10/s1 2.17→3.55(target 3.55), w10/s2 2.67→3.92(target 4.48), w10/s3 1.99→3.18(target 3.42), w10/s4 1.64→3.01(target 4.20), w20/s0 2.78→4.77(target 5.48), w20/s1 2.48→4.52(target 5.47), w20/s2 2.16→4.57(target 4.70), w20/s3 2.37→4.73(target 5.59), w20/s4 2.03→5.34(target 6.00)

 width metric                 pair     mean        lo       hi  excl_zero
    10      M    svdrec_alive-none 0.138994  0.028969 0.278613       True
    10      M svdrec_alive-shuffle 0.118013  0.068911 0.181082       True
    20      M    svdrec_alive-none 0.123776  0.030073 0.217479       True
    20      M svdrec_alive-shuffle 0.106812 -0.011146 0.241355      False

## 副指標 (M_tail = タスク末尾 2k 平均)

 width metric                 pair     mean        lo       hi  excl_zero
    10 M_tail          svdrec-none 0.175654  0.044582 0.293138       True
    10 M_tail         shuffle-none 0.017492 -0.064067 0.105239      False
    10 M_tail       svdrec-shuffle 0.158162  0.074752 0.230852       True
    10 M_tail    svdrec_alive-none 0.135124  0.010113 0.282805       True
    10 M_tail svdrec_alive-shuffle 0.117632  0.060796 0.188773       True
    20 M_tail          svdrec-none 0.143274  0.049825 0.256511       True
    20 M_tail         shuffle-none 0.002503 -0.057423 0.070450      False
    20 M_tail       svdrec-shuffle 0.140770  0.002518 0.301410       True
    20 M_tail    svdrec_alive-none 0.105445  0.019547 0.191343       True
    20 M_tail svdrec_alive-shuffle 0.102942 -0.015156 0.245213      False

## seed 別 (回復/予防 層別用)

 width          arm  seed      M  M_tail  dead_t_int  dead_end  d_dead  srank_end  eval_end label
    10         none     0 0.2130  0.1832        0.10      0.30    0.20     2.3832    0.2464    予防
    10      shuffle     0 0.3315  0.2981        0.10      0.60    0.50     1.4066    0.4571    予防
    10       svdrec     0 0.5787  0.5608        0.10      0.90    0.80     1.0000    0.6368    予防
    10 svdrec_alive     0 0.4251  0.4036        0.10      0.70    0.60     1.2469    0.5025    予防
    10         none     1 0.1978  0.1698        0.10      0.20    0.10     1.9097    0.1118    回復
    10      shuffle     1 0.1283  0.1004        0.10      0.20    0.10     2.7957    0.0695    回復
    10       svdrec     1 0.3280  0.2964        0.20      0.30    0.10     2.1755    0.2424    回復
    10 svdrec_alive     1 0.2620  0.2357        0.10      0.80    0.70     1.7888    0.3536    回復
    10         none     2 0.2847  0.2716        0.30      0.50    0.20     1.6189    0.2865    回復
    10      shuffle     2 0.2170  0.1970        0.30      0.40    0.10     1.7219    0.1355    回復
    10       svdrec     2 0.2417  0.2248        0.40      0.60    0.20     2.3932    0.2874    回復
    10 svdrec_alive     2 0.2768  0.2613        0.40      0.60    0.20     2.2704    0.1402    回復
    10         none     3 0.1932  0.1414        0.20      0.20    0.00     2.3265    0.0208    予防
    10      shuffle     3 0.3343  0.2904        0.20      0.50    0.30     1.5970    0.3002    予防
    10       svdrec     3 0.4173  0.3783        0.30      0.50    0.20     2.0897    0.2498    予防
    10 svdrec_alive     3 0.5716  0.5384        0.30      0.80    0.50     1.4107    0.4777    予防
    10         none     4 0.3892  0.3580        0.40      0.60    0.20     1.2334    0.5171    回復
    10      shuffle     4 0.3718  0.3257        0.20      0.40    0.20     1.8604    0.1304    回復
    10       svdrec     4 0.5643  0.5421        0.50      0.60    0.10     2.1944    0.6095    回復
    10 svdrec_alive     4 0.4374  0.3606        0.60      0.70    0.10     1.4396    0.5268    回復
    20         none     0 0.0556  0.0286        0.35      0.20   -0.15     2.8612    0.0000    予防
    20      shuffle     0 0.0305  0.0052        0.30      0.45    0.15     2.4186    0.0085    予防
    20       svdrec     0 0.1652  0.1222        0.45      0.60    0.15     2.8608    0.0191    予防
    20 svdrec_alive     0 0.0674  0.0331        0.45      0.45    0.00     2.9147    0.0081    予防
    20         none     1 0.1086  0.0704        0.55      0.35   -0.20     1.6678    0.1146    予防
    20      shuffle     1 0.2598  0.1951        0.20      0.45    0.25     2.0094    0.0034    予防
    20       svdrec     1 0.1734  0.1383        0.55      0.55    0.00     2.5342    0.1997    予防
    20 svdrec_alive     1 0.1928  0.1306        0.65      0.40   -0.25     2.4580    0.0347    予防
    20         none     2 0.0220  0.0004        0.35      0.25   -0.10     2.0105    0.0000    予防
    20      shuffle     2 0.0225  0.0021        0.20      0.10   -0.10     2.3722    0.0000    予防
    20       svdrec     2 0.0392  0.0104        0.35      0.45    0.10     2.3719    0.1883    予防
    20 svdrec_alive     2 0.0432  0.0147        0.45      0.35   -0.10     2.6627    0.0327    予防
    20         none     3 0.2626  0.2232        0.45      0.50    0.05     2.4787    0.0317    予防
    20      shuffle     3 0.1998  0.1264        0.20      0.60    0.40     2.0619    0.0467    予防
    20       svdrec     3 0.6306  0.5870        0.35      0.80    0.45     1.6667    0.2286    予防
    20 svdrec_alive     3 0.5221  0.4638        0.55      0.80    0.25     2.0519    0.1900    予防
    20         none     4 0.0212  0.0038        0.25      0.25    0.00     2.1313    0.0000    予防
    20      shuffle     4 0.0422  0.0103        0.15      0.15    0.00     2.2892    0.0000    予防
    20       svdrec     4 0.2394  0.1851        0.55      0.45   -0.10     2.7332    0.0319    予防
    20 svdrec_alive     4 0.2633  0.2115        0.55      0.45   -0.10     2.4603    0.0056    予防

## 先生への確認事項 (仕様 §8)

1. **shuffle の連続化**: 先生の記述は「空間内要素のみシャッフル」だが、ΔF を svdrec と厳密一致させるため top-k 部分空間内の連続ランダム回転 W'(θ)=U_k Q(θ) S_k V_kᵀ + 残差 を採用した (特異値・両 span・ノルム・ランク不変)。この連続化が意図と整合するかご確認ください。
2. **ε の決定規則**: svdrec の ε は「介入直後の stable_rank_W_alive ≈ step0 の値 (srank_target)」を pre-intervention の dead マスク下で bisect。介入がゲートを開き直すため、実測の介入後 srank_alive は target をやや上回ることがある (intervention_log の post_svdrec_stable_rank_W_alive 参照)。
3. (Phase 0 逸脱) eval_loss 離陸時刻の字義定義は full-batch の高い初期損失で退化するため、ロバスト定義 (argmin 以降で min+0.5*(v_1M−min) 上抜き) でラベル付けした。
4. (Phase 1 逸脱) shuffle の G は「seed 固定の単発抽選」だと回転角スペクトル次第で ΔF 目標に僅かに届かないケースがある (w20/s4 で 9.08 < 9.24)。同一 seed 列からの決定論的棄却サンプリング (到達可能な G が出るまで再抽選、最大50回) に拡張した。
