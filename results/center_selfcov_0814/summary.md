# center_selfcov_0814 summary (spec §6 事前登録判定)

## 判定表 (null 結果も同じ体裁)

 pred                                        scope    verdict                                                                                                                                                             evidence
 P3-1    condA w100 dead_frac 最終値 (centered < std)       PASS                                                                                                          centered 0.294 vs std 0.964, diff -0.67 CI [-0.724, -0.612]
 P3-2     condA w5 wcos_mean 前半傾き (centered < std)       FAIL                                                                                      centered 3.475e-07 vs std 2.358e-07, diff +1.117e-07 CI [-4.452e-08, 3.503e-07]
 P3-2   condA w100 wcos_mean 前半傾き (centered < std)       PASS                                                                                      centered 1.065e-08 vs std 1.832e-07, diff -1.726e-07 CI [-1.956e-07, -1.42e-07]
 P3-3                  ‖w‖ 交絡統制 (差が ‖w‖ 差で説明されうるか) CONFOUNDED                                    w5: ‖w‖ centered 3.39 vs std 4.54 (diff CI [-1.64, -0.659], 有意); w100: ‖w‖ centered 2.01 vs std 4.44 (diff CI [-2.85, -2.13], 有意)
P3-2b condB κ=1 wcos_mean 前半傾き (c=0 < c=2、µ=0 厳密版)       PASS                                                                                         c=0 -1.015e-07 vs c=2 4.819e-08, diff -1.497e-07 CI [-3.736e-07, -6.675e-09]
 P3-4         κ=16 cos_e1W_e1Sig: (a) 単調増加 (b) 床超え       FAIL (a) [0,300k] 傾き +3.78e-08/step CI [5.9e-09, 8.37e-08]、init→final -0.037 CI [-0.096, 0.034] → 増加と言えない (系列は床付近で往復); (b) 最終 0.105 CI [0.021, 0.200] vs 床 0.174 → 床を超えない
 P3-6      先生の予言: cos_e1W_e1Sig → 1.0 (判定は 0.9 到達)       FAIL                                                                    κ=16 最終 0.105 (0.9 未到達なら部分整列)。srank_alive=2.20, top1_frac=0.46 → rank-1 に落ちていないので e1 が支配的でないのは整合的
 P3-5                        κ 単調性 (実質 κ=4 < κ=16)       FAIL                                                                    κ16 0.105 − κ4 0.312 = -0.207 CI [-0.442, 0.001] (予測と逆符号)。ただし両 κ の最終値 CI が床 0.174 を含むため大小関係の解釈は弱い
 P3-7  κ=16 最終 ckpt: |cos(E[g],u)| > cos_e1W_e1Sig       PASS                                                                                                                 E[g] 0.907 vs W 0.105, diff +0.801 CI [0.747, 0.849]


## Phase 0 / Phase 1

- Phase 0 (aniso_perp_0812 再解析): 仕様 §3 の期待値6項目を全て相対5%以内で再現 (PASS)。「勾配場は Σ 軸を向くが重みは床付近」の乖離を確認 (P3-7 の予備証拠)
- Phase 1 (レジーム探索): 採用セル: 100     5    16  0.01    0.711511  2.530425  1.771555      0.0   0.090621  0.154205    0.063098    1.072168   True 2      True     True    True  False

## アーム1: 条件A std vs centered (項目2)

 width      metric  std_mean  centered_mean  diff_mean  diff_lo  diff_hi  diff_n  diff_excl_zero
     5  dead_final   0.80000        0.72000   -0.08000 -0.28000  0.08000       5           False
     5  wcos_slope   0.00000        0.00000    0.00000 -0.00000  0.00000       5           False
     5 wnorm_final   4.53721        3.38807   -1.14914 -1.64115 -0.65901       5            True
     5 wnorm_slope   0.00001        0.00000   -0.00001 -0.00001 -0.00000       5            True
   100  dead_final   0.96400        0.29400   -0.67000 -0.72400 -0.61200       5            True
   100  wcos_slope   0.00000        0.00000   -0.00000 -0.00000 -0.00000       5            True
   100 wnorm_final   4.44241        2.00795   -2.43446 -2.85377 -2.13254       5            True
   100 wnorm_slope   0.00001        0.00000   -0.00001 -0.00001 -0.00000       5            True

## アーム2: 条件B c=0 vs c=2 (µ=0 厳密、κ=1)

    metric  c0_mean  c2_mean  diff_mean  diff_lo  diff_hi  diff_n  diff_excl_zero
      dead  0.00000  0.00000    0.00000  0.00000  0.00000       5           False
wcos_slope -0.00000  0.00000   -0.00000 -0.00000 -0.00000       5            True
     srank  2.53149  2.39735    0.13415 -0.42319  0.59945       5           False
      eval  0.14681  0.15655   -0.00975 -0.04184  0.03066       5           False

## アーム3: C_self 残存 (κ 別、c=0)

 kappa  init_mean  final_mean  final_lo  final_hi  slope_mean  slope_lo  slope_hi  srank_final  top1_final  pca_final  e1stab_min
     1        NaN         NaN       NaN       NaN         NaN       NaN       NaN       2.5315      0.4084        NaN      0.0017
     4     0.1341      0.3120    0.1414    0.5004        -0.0      -0.0       0.0       2.3732      0.4249     0.3218      0.0016
    16     0.1422      0.1055    0.0207    0.2000         0.0       0.0       0.0       2.2001      0.4581     0.1085      0.0009

- ランダム床 |cos| ≈ 0.174 (d=21)

## サニティ (§7)

- S1 (target_hidden 未指定で既存 condB と bit 一致): PASS (coupling_ab_0813 B_w5 の共通カラムが全行一致、追加カラムなし)
- S2 (条件A centered で教師は生入力): PASS — コード引用:

```
src/train.py (eval_batch / 学習ループ) — 中心化は学習器入力のみに適用され、
教師は生入力 x_raw を受け取る:

    x_raw = env.step()                               # [R,d]
    y = teacher(x_raw)                               # [R]   ← 生入力
    x_in = x_raw - cmask * st["running_mean"]        # ← 学習器のみ中心化
    pre, a, yhat = net.forward(x_in)

eval 側も同様 (eval_batch は y = teacher(x) を生 x で計算し、
呼び出し側が x_ev_in = x_ev - cmask*running_mean を net に渡す)。
```
- S3 (EMA 中心化の実効残差 ‖running_mean−µ_true‖/‖µ_true‖): {0: 1.0, 10000: 0.0301, 50000: 0.0239, 100000: 0.0212, 300000: 0.0273, 1000000: 0.0281}
- S4 (κ=1 で Σ 系が全て NaN): PASS
- S5 (step0 の cos が床付近): PASS (mean 0.131 vs floor 0.174)
- S6 (e1_stability < 0.9 の区間割合): 0.1296 — 大きい場合、第1特異値が縮退して主方向が意味を持たない区間がある

## 結論

1. **項目2 (増幅因子) は dead 経路で強く成立、整列経路では条件付き**。条件A w100 で
   centered の dead_frac は 0.294 vs std 0.964 と大差 (P3-1 PASS、既報 0.96→0.28 を再現)。
   一方 wcos_mean の傾きは w100 で PASS だが **w5 では差なし** (P3-2 は幅依存)。
   µ=0 が厳密な条件B (P3-2b) でも整列傾きは低下するが CI はゼロをかろうじて外す程度。
2. **ただし P3-3 は CONFOUNDED**。centered 腕は ‖w‖ も有意に小さい (w100: 2.01 vs 4.44)。
   理論 v2 §3(d) の通りノイズは ‖w‖⁻²・ドリフトは ‖w‖⁻¹ でスケールするため、
   µ の効果とノルム媒介効果が本実験では分離できていない。
   仕様 §9 のノルム固定アームが本命の追試になる。
3. **項目3 (C_self 残存) は不支持**。µ=0 厳密・κ=16 で cos_e1W_e1Sig は
   最終 0.105 (CI [0.021, 0.200])、ランダム床 0.174 を超えない (P3-4 FAIL)。
   系列は床付近を往復するノイズ支配で、単調増加も認められない。
   先生の予言 (→1.0) は成立しない (P3-6 FAIL、0.9 に遠く及ばない)。
   κ 単調性も逆符号 (P3-5 FAIL) だが両 κ とも床付近なので方向の主張自体が弱い。
   なお srank_alive は 2.2–2.5 で rank-1 に落ちておらず (top1_frac 0.41–0.46)、
   「e₁ が支配的でないのは当然」という但し書きが該当する。
4. **P3-7 が最も強い所見 (PASS)**。同一 checkpoint で勾配場は Σ 軸をほぼ完全に向く
   (|cos(E[g],u)| = 0.907) のに、重みは床以下 (0.105) に留まる。差 +0.801
   CI [0.747, 0.852]。Phase 0 の乖離 (0.71 vs 0.10) が、教師幅を分離してレジームを
   変えた後も、むしろ拡大して再現した。
   **「drift は Σ 軸を向いているが重みはそこに蓄積しない」** = 理論 v2 §5(b) の
   1/‖w‖ による操舵切断、および rank_int_0814 の「病理は状態ではなく力場」と同方向。
5. **本実験の重要な限界**: Phase 1 でどのセルも LoP 発現基準 (基準4) を満たさず、
   採用セルも eval_loss は低下する (LoP 非発現) レジームである。項目3 の null は
   「LoP が起きている状況で C_self 整列が残らない」ことの証明にはなっていない。


## 先生への確認事項 (§10)

1. **e₁^W の定義**: 「PCA」はユニット方向の中心化を含みますか。含む場合、全ユニット共通の方向成分 (w̄) が除去され、測ろうとしている整列そのものが落ちます。本実験は中心化なしの第1右特異ベクトルを主判定とし、PCA 版も併記しました (fig_cs4)。
2. **条件A では項目3が測定不能**: SCR の入力共分散は (1/4)I で完全等方のため e₁^Σ が一意に定まりません。項目3はスパイク型 Σ を入れた条件B で実施しました。
3. **収束先は 1.0 ではなく |cos| → 1**: Cov(e,x) の軸吸引は ± 対称(aniso_perp_0812 の符号付き解析) なので符号は自発的に決まります。
4. **レジーム**: 既存の条件B設定は教師幅が学習器と同一で LoP が発現しないため、教師幅を分離 (target_hidden=100) した上でレジームを選定しました。
