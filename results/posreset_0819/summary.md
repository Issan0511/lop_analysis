# posreset_0819 summary (spec_posreset_0819 §6 事前登録判定)

同方向・小ノルムリセット判別 (2×2 要因)。レジーム A = condA A_w100 (µ 経路 dead)、レジーム B = cbp_harm routeK K=100 の w20 (b 経路 dead)。t_int=500000、窓 [t_int, t_int+500000]。

**判定は clean eval_loss のみ。dead_frac は PASS/FAIL のいかなる経路にも入れない** [§5, §9]。統計は §6 凍結の paired seed bootstrap (rng=default_rng(20260819), B=10000, percentile 95%CI)。**「CI が 0 を除外」は、正が予測されている量については 『CI 下限 > 0』と読む** (片側の読み。本 summary・verdict.csv 全体で同じ規約)。

## 1. 一行結論

G0: A PASS / B FAIL / P1(B) void / P2(B) void / P3(B) void / P4(A) PASS / Δ_posonly/Δ_full = 0.86 → §6 の帰結マッピングに該当する組合せなし (個別行を参照)

## 2. G0 (前提ゲート: Δ_full > 0)

     id regime    point    ci_lo   ci_hi result
     G0      A  0.29458  0.22302 0.36063   PASS
G0_late      A  0.25199  0.15466 0.34091   PASS
     G0      B  0.01326 -0.00890 0.03367   FAIL
G0_late      B -0.00063 -0.02385 0.02202   FAIL

- **G0 不成立: レジーム B の P 判定は全て void (記録して保留) [§6]**。

## 3. Δ 表 (4 アーム × 2 レジーム、M と M_late、点推定 + 95%CI)

regime     arm  M(none)     Δ_M                 CI_M  M_late(none)  Δ_M_late            CI_M_late
     A    none  0.42671 0.00000            — (基準アーム)       0.43306   0.00000            — (基準アーム)
     A posonly  0.42671 0.20726     [0.1613, 0.2517]       0.43306   0.16852    [0.09162, 0.2386]
     A dironly  0.42671 0.27577      [0.211, 0.3325]       0.43306   0.21250     [0.1339, 0.2851]
     A    full  0.42671 0.29458      [0.223, 0.3606]       0.43306   0.25199     [0.1547, 0.3409]
     B    none  0.68725 0.00000            — (基準アーム)       0.69584   0.00000            — (基準アーム)
     B posonly  0.68725 0.01145 [-0.008389, 0.03037]       0.69584   0.00034  [-0.02439, 0.02538]
     B dironly  0.68725 0.00382 [-0.001864, 0.01382]       0.69584   0.00603 [-0.002674, 0.02111]
     B    full  0.68725 0.01326 [-0.008901, 0.03367]       0.69584  -0.00063  [-0.02385, 0.02202]

Δ_arm = M(none) − M(arm)。**正 = そのアームが none より良い**。ペアは base_run_id ソート順の seed 対応 [§6]。

## 4. P 表 (P1–P7、M 基準が主判定。_late は §5 の併記)

            id regime                                                statistic    point    ci_lo    ci_hi result
            G0      A                               Δ_full = M(none) − M(full)  0.29458  0.22302  0.36063   PASS
       G0_late      A                     Δ_full = M_late(none) − M_late(full)  0.25199  0.15466  0.34091   PASS
            G0      B                               Δ_full = M(none) − M(full)  0.01326 -0.00890  0.03367   FAIL
       G0_late      B                     Δ_full = M_late(none) − M_late(full) -0.00063 -0.02385  0.02202   FAIL
            P1      B                                            Δ_posonly (M)  0.01145 -0.00839  0.03037   void
            P2      B                               Δ_posonly − 0.5·Δ_full (M)  0.00481 -0.00886  0.02062   void
     P2_strong      B                              Δ_posonly − 0.75·Δ_full (M)  0.00150 -0.01271  0.01739   void
      P2_ratio      B                                   Δ_posonly / Δ_full (M)  0.86300      NaN      NaN   void
            P3      B                              0.25·Δ_full − Δ_dironly (M) -0.00051 -0.01152  0.00794   void
            P4      A                                            Δ_posonly (M)  0.20726  0.16126  0.25171   PASS
            P5      A                                   Δ_full − Δ_posonly (M)  0.08732  0.04959  0.12843 report
       P1_late      B                                       Δ_posonly (M_late)  0.00034 -0.02439  0.02538   void
       P2_late      B                          Δ_posonly − 0.5·Δ_full (M_late)  0.00065 -0.01805  0.02397   void
P2_strong_late      B                         Δ_posonly − 0.75·Δ_full (M_late)  0.00081 -0.01882  0.02400   void
 P2_ratio_late      B                              Δ_posonly / Δ_full (M_late) -0.53879      NaN      NaN   void
       P3_late      B                         0.25·Δ_full − Δ_dironly (M_late) -0.00619 -0.02248  0.00491   void
       P4_late      A                                       Δ_posonly (M_late)  0.16852  0.09162  0.23855   PASS
       P5_late      A                              Δ_full − Δ_posonly (M_late)  0.08347  0.01470  0.16686 report
    P6_posonly      B                        reopen_frac(posonly) @ t_int+100k  0.49819      NaN      NaN   void
    P6_dironly      B                        reopen_frac(dironly) @ t_int+100k  0.01053      NaN      NaN   void
            P6      B reopen_frac(posonly) − reopen_frac(dironly) @ t_int+100k  0.48766  0.42502  0.55111   void
            P7      A    median_unit Δcos(u, µ̂) (posonly, t_int → t_int+post) -0.03266 -0.05364 -0.00713 report

根拠と注記 (verdict.csv の note 列):

- **G0** (A) [PASS] Δ_full = M(none) − M(full): 0.2946 CI [0.223, 0.3606] — 閾値 CI 下限 > 0。前提ゲート。M(none) 平均 0.4267; n_seed=10
- **G0_late** (A) [PASS] Δ_full = M_late(none) − M_late(full): 0.252 CI [0.1547, 0.3409] — 閾値 CI 下限 > 0。前提ゲート。M_late(none) 平均 0.4331; n_seed=10; 主判定は M (本行は §5 の併記)
- **G0** (B) [FAIL] Δ_full = M(none) − M(full): 0.01326 CI [-0.008901, 0.03367] — 閾値 CI 下限 > 0。前提ゲート。M(none) 平均 0.6872; n_seed=10
- **G0_late** (B) [FAIL] Δ_full = M_late(none) − M_late(full): -0.0006293 CI [-0.02385, 0.02202] — 閾値 CI 下限 > 0。前提ゲート。M_late(none) 平均 0.6958; n_seed=10; 主判定は M (本行は §5 の併記)
- **P1** (B) [void] Δ_posonly (M): 0.01145 CI [-0.008389, 0.03037] — 閾値 CI 下限 > 0。G0 不成立のため void (点推定は参考値)。座標完全復元 (‖w‖ と b) が単独で効くか。FAIL なら B1 棄却の主成分。n_seed=10
- **P2** (B) [void] Δ_posonly − 0.5·Δ_full (M): 0.004814 CI [-0.008858, 0.02062] — 閾値 CI 下限 > 0 で PASS / 点推定のみ正で weak PASS。G0 不成立のため void (点推定は参考値)。**主判定**。FAIL なら混合説 (座標＋特徴鮮度) へ改訂。n_seed=10
- **P2_strong** (B) [void] Δ_posonly − 0.75·Δ_full (M): 0.001499 CI [-0.01271, 0.01739] — 閾値 CI 下限 > 0 なら「強」。G0 不成立のため void (点推定は参考値)。75% 水準は CI 下限が 0 を超えず「強」は付かない。n_seed=10
- **P2_ratio** (B) [void] Δ_posonly / Δ_full (M): 0.863 CI [NA, NA] — 閾値 参考値 (閾値なし)。G0 不成立のため void (点推定は参考値)。分母が正の bootstrap 標本 0.89 ≤ 0.95 のため **CI は報告しない** (分母が 0 を跨ぐと比の CI は発散する。coupling_fbw_0813 の家内規約)。n_seed=10
- **P3** (B) [void] 0.25·Δ_full − Δ_dironly (M): -0.0005058 CI [-0.01152, 0.007942] — 閾値 CI 下限 > 0。G0 不成立のため void (点推定は参考値)。b が深い負のままゲートが開かない所で新方向が効いてはいけない。FAIL は **β/ゲート機構への警報**。n_seed=10
- **P4** (A) [PASS] Δ_posonly (M): 0.2073 CI [0.1613, 0.2517] — 閾値 CI 下限 > 0。**H_feat の主戦場**: 新特徴ゼロで µ 経路の便益が出るか。FAIL は B1 の棄却ではなく改訂 (操舵単独では不十分)。n_seed=10
- **P5** (A) [report] Δ_full − Δ_posonly (M): 0.08732 CI [0.04959, 0.1284] — 閾値 報告のみ (PASS/FAIL なし)。マージン寄与の分解量。符号: 正 (full が優位 = マージン寄与あり)。CI は記述用。n_seed=10
- **P1_late** (B) [void] Δ_posonly (M_late): 0.0003391 CI [-0.02439, 0.02538] — 閾値 CI 下限 > 0。G0 不成立のため void (点推定は参考値)。座標完全復元 (‖w‖ と b) が単独で効くか。FAIL なら B1 棄却の主成分。n_seed=10; 主判定は M (本行は §5 の併記)
- **P2_late** (B) [void] Δ_posonly − 0.5·Δ_full (M_late): 0.0006537 CI [-0.01805, 0.02397] — 閾値 CI 下限 > 0 で PASS / 点推定のみ正で weak PASS。G0 不成立のため void (点推定は参考値)。**主判定**。FAIL なら混合説 (座標＋特徴鮮度) へ改訂。n_seed=10; 主判定は M (本行は §5 の併記)
- **P2_strong_late** (B) [void] Δ_posonly − 0.75·Δ_full (M_late): 0.0008111 CI [-0.01882, 0.024] — 閾値 CI 下限 > 0 なら「強」。G0 不成立のため void (点推定は参考値)。75% 水準は CI 下限が 0 を超えず「強」は付かない。n_seed=10; 主判定は M (本行は §5 の併記)
- **P2_ratio_late** (B) [void] Δ_posonly / Δ_full (M_late): -0.5388 CI [NA, NA] — 閾値 参考値 (閾値なし)。G0 不成立のため void (点推定は参考値)。分母が正の bootstrap 標本 0.474 ≤ 0.95 のため **CI は報告しない** (分母が 0 を跨ぐと比の CI は発散する。coupling_fbw_0813 の家内規約)。n_seed=10; 主判定は M (本行は §5 の併記)
- **P3_late** (B) [void] 0.25·Δ_full − Δ_dironly (M_late): -0.006192 CI [-0.02248, 0.00491] — 閾値 CI 下限 > 0。G0 不成立のため void (点推定は参考値)。b が深い負のままゲートが開かない所で新方向が効いてはいけない。FAIL は **β/ゲート機構への警報**。n_seed=10; 主判定は M (本行は §5 の併記)
- **P4_late** (A) [PASS] Δ_posonly (M_late): 0.1685 CI [0.09162, 0.2386] — 閾値 CI 下限 > 0。**H_feat の主戦場**: 新特徴ゼロで µ 経路の便益が出るか。FAIL は B1 の棄却ではなく改訂 (操舵単独では不十分)。n_seed=10; 主判定は M (本行は §5 の併記)
- **P5_late** (A) [report] Δ_full − Δ_posonly (M_late): 0.08347 CI [0.0147, 0.1669] — 閾値 報告のみ (PASS/FAIL なし)。マージン寄与の分解量。符号: 正 (full が優位 = マージン寄与あり)。CI は記述用。n_seed=10; 主判定は M (本行は §5 の併記)
- **P6_posonly** (B) [void] reopen_frac(posonly) @ t_int+100k: 0.4982 CI [NA, NA] — 閾値 報告のみ。G0 不成立のため void (点推定は参考値)。treated のうち p̂ > 0.05 の割合 (seed 平均)。seed 別 [0.474, 0.474, 0.4, 0.556, 0.368, 0.684, 0.526, 0.5, 0.368, 0.632]
- **P6_dironly** (B) [void] reopen_frac(dironly) @ t_int+100k: 0.01053 CI [NA, NA] — 閾値 報告のみ。G0 不成立のため void (点推定は参考値)。treated のうち p̂ > 0.05 の割合 (seed 平均)。seed 別 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.053, 0.053]
- **P6** (B) [void] reopen_frac(posonly) − reopen_frac(dironly) @ t_int+100k: 0.4877 CI [0.425, 0.5511] — 閾値 点推定の順序のみ (PASS/FAIL なし)。G0 不成立のため void (点推定は参考値)。順序: posonly > dironly (予測どおり)。posonly 0.4982 / dironly 0.01053 / none 0.01608 / full 0.5676。CI は記述用 (事前登録は点推定のみ)。機構署名の不発は判定に波及しない。
- **P7** (A) [report] median_unit Δcos(u, µ̂) (posonly, t_int → t_int+post): -0.03266 CI [-0.05364, -0.007131] — 閾値 点推定のみ (PASS/FAIL なし)。seed 別 median [-0.0448, 0.0661, -0.0366, -0.0415, -0.0251, -0.0278, -0.0894, -0.0466, -0.0625, -0.0183]。正なら二段回復署名 (操舵回復 → u が +µ̂ 半空間へ)。CI は記述用。

## 5. treated_frac 表 (§3.4 適格性)

regime  seed  n_treated  treated_frac  n_guard_fallback  pre_dead_frac  pre_eval_loss
     A     0         85          0.85                 0           0.85        0.00002
     A     1         98          0.98                 0           0.98        0.34416
     A     2         97          0.97                 0           0.97        0.18165
     A     3         96          0.96                 0           0.96        0.32622
     A     4         97          0.97                 0           0.97        0.73449
     A     5         92          0.92                 0           0.92        0.05731
     A     6         81          0.81                 0           0.81        0.00001
     A     7         95          0.95                 0           0.95        0.07157
     A     8         76          0.76                 0           0.76        0.00000
     A     9         98          0.98                 0           0.98        1.18753
     B     0         19          0.95                 0           0.95        0.56651
     B     1         19          0.95                 0           0.95        0.54810
     B     2         20          1.00                 0           1.00        0.76029
     B     3         18          0.90                 0           0.90        0.66470
     B     4         19          0.95                 0           0.95        0.69955
     B     5         19          0.95                 0           0.95        0.48013
     B     6         19          0.95                 0           0.95        0.54758
     B     7         18          0.90                 0           0.90        0.70849
     B     8         19          0.95                 0           0.95        0.53123
     B     9         19          0.95                 0           0.95        0.86806

- レジーム A: treated_frac ≥ 0.3 は 10/10 seed → 適格 (主判定として扱う)
- レジーム B: treated_frac ≥ 0.3 は 10/10 seed → 適格 (主判定として扱う)
- ガード発動 (‖w_i‖ < norm_guard で posonly→full にフォールバック): 合計 0 件 (0 seed で発生) [§3.4]

## 6. サニティ S1–S4 (§7)

- **S1** (OMP_NUM_THREADS=1): 解析プロセス `1` / ランナー申告 `1`
- **S2** (resume bit 一致) ランナー申告: [{"regime": "A", "gbase": "A_w100", "t_int": 500000, "total": 1000000, "snapshot_sha256": {"W": "bf8ef95e1dc549ea289f6d47716e57d9c194d00882c161cbe9e20eeab8ca930b", "b": "36ed4527ccdec40f11efad409cd6ab77fe6c34dbff27ab1e4b99a21380a00731", "v": "1df3e599e37c096ec89dff09d4f881d4eb08eb2cdf9990dbe9a6f557ec4c2a94", "c": "3c7a3c39db6567a862527cab907ef42e5763b3ebbbf0273cf2326991bd114b94", "running_mean": "ed1963daf2993144579f6cda4353e3af50b5526bf1bc734693de4084df2d096a"}, "n_nonfinite_at_tint": 0, "treated_eq_neg_gate": true, "S3": "PASS", "s3_worst_f64": {"s3_posonly_cos_err_f64": 4.440892098500626e-16, "s3_posonly_norm_relerr_f64": 3.417331519034691e-16, "s3_dironly_norm_relerr_f64": 2.902908719095189e-16, "s3_dironly_cos_g_f64": 4.440892098500626e-16, "s3_full_exact_f64": 0.0, "s3_guard_full_exact_f64": 0.0}, "s3_worst_f32": {"s3_posonly_cos_err_f32": 7.771561172376096e-16, "s3_posonly_norm_relerr_f32": 3.330845684737138e-08, "s3_dironly_norm_relerr_f32": 2.5915662486559647e-08, "s3_dironly_cos_g_f32": 7.771561172376096e-16, "s3_full_exact_f32": 0.0, "s3_guard_full_exact_f32": 0.0}, "n_guard_fallback_total": 0, "S2": "PASS", "S2_note": "step>t_int は全列厳密一致。境界行 (step==t_int) のみ ['dead_persist_frac', 'p_zero_persist_frac'] を除外 (resume 側は直前の lop step を持たず定義上 NaN)", "treated_frac": {"values": [0.85, 0.98, 0.97, 0.96, 0.97, 0.92, 0.81, 0.95, 0.76, 0.98], "mean": 0.915, "min": 0.76, "n_ge_min": 10, "n_seeds": 10, "eligible": true}}, {"regime": "B", "gbase": "B_w20", "t_int": 500000, "total": 1000000, "snapshot_sha256": {"W": "5b8c52ba9542df947dbb3137d5c4cd90f8442e67dc340284c151346bcdac8d03", "b": "fce53ef5bda94d488e61eacbebe4019294247c4ab4ef869051dfd6b4680d321c", "v": "15b986a37db244cd6f26a4b3151dcf10fc47c2deac9b9914fa2a3a39bf523947", "c": "ba7a33f2c74db62be7a82ee8077dccd538d2d1fbab4a3347f73dca0ffbbbd591", "running_mean": "a0e79a5f915a733c2aea33c223eeea9d5a91afafb94f475459119cc9c5343646"}, "n_nonfinite_at_tint": 0, "treated_eq_neg_gate": true, "S3": "PASS", "s3_worst_f64": {"s3_posonly_cos_err_f64": 4.440892098500626e-16, "s3_posonly_norm_relerr_f64": 2.5410778867497687e-16, "s3_dironly_norm_relerr_f64": 3.0765033219438983e-16, "s3_dironly_cos_g_f64": 4.440892098500626e-16, "s3_full_exact_f64": 0.0, "s3_guard_full_exact_f64": 0.0}, "s3_worst_f32": {"s3_posonly_cos_err_f32": 6.661338147750939e-16, "s3_posonly_norm_relerr_f32": 3.11633230496464e-08, "s3_dironly_norm_relerr_f32": 2.049850304314241e-08, "s3_dironly_cos_g_f32": 6.661338147750939e-16, "s3_full_exact_f32": 0.0, "s3_guard_full_exact_f32": 0.0}, "n_guard_fallback_total": 0, "S2": "PASS", "S2_note": "step>t_int は全列厳密一致。境界行 (step==t_int) のみ ['dead_persist_frac', 'p_zero_persist_frac'] を除外 (resume 側は直前の lop step を持たず定義上 NaN)", "treated_frac": {"values": [0.95, 0.95, 1.0, 0.9, 0.95, 0.95, 0.95, 0.9, 0.95, 0.95], "mean": 0.945, "min": 0.9, "n_ge_min": 10, "n_seeds": 10, "eligible": true}}]
- **S2 独立再検査** (cont と none アームの lop_metrics を step ≥ t_int で文字列比較): A PASS, B PASS
- **S3** (介入の数値保証) float64 最大誤差: {'s3_posonly_cos_err_f64': 4.440892098500626e-16, 's3_posonly_norm_relerr_f64': 3.417331519034691e-16, 's3_dironly_norm_relerr_f64': 3.0765033219438983e-16, 's3_dironly_cos_g_f64': 4.440892098500626e-16} (許容 1e-12) → PASS
  - float32 丸め後 (学習再開に使う値、eps≈1.2e-7 律速): {'s3_posonly_cos_err_f32': 7.771561172376097e-16, 's3_posonly_norm_relerr_f32': 3.3308456847371386e-08, 's3_dironly_norm_relerr_f32': 2.5915662486559647e-08, 's3_dironly_cos_g_f32': 7.771561172376097e-16}
  - 厳密一致列 (w_post ≡ g / ガード時の full フォールバック) の最大差: {'s3_full_exact_f64': 0.0, 's3_guard_full_exact_f64': 0.0, 's3_full_exact_f32': 0.0, 's3_guard_full_exact_f32': 0.0} → PASS (全て厳密 0)
  - 論理判定 (a←0 / b 規約 / treated 外 hash 不変 / ランナー総合): {'s3_readout_zero_ok': True, 's3_bias_ok': True, 's3_untreated_hash_ok': True, 's3_pass': True} → 総合 True
- **S4** (treated 集合の hash が seed 内 4 アームで一致): PASS (80 npz を照合)
- 発散 (eval_loss = NaN) の内訳 — **判定には使わないが必ず報告する**:

regime     arm  n_nan  n_pts  n_run_all_nan
     A    cont      0   5010              0
     A dironly      0   5010              0
     A    full      0   5010              0
     A    none      0   5010              0
     A posonly      0   5010              0
     B    cont      0   5010              0
     B dironly      0   5010              0
     B    full      0   5010              0
     B    none      0   5010              0
     B posonly      0   5010              0

## 7. 逸脱記録

- 窓・閾値の設定は `results/config_used.yaml` から解決した (t_int=500000, post=500000, p_hat_tau=0.05, probe_every=10000)
- M 窓の左端 (step=t_int) は **none では介入前・reset 3 アームでは介入直後**の値 (ランナーが介入直後に t_int 行を書くため)。窓が閉区間 [t_int, t_int+post] である以上 §5 のとおりで除外はしないが、この 1 行だけで Δ_arm が受ける系統差 (seed 平均, 窓 501 点で割った値): A/dironly -0.0002016, A/full -0.0002016, A/posonly -0.0002016, B/dironly -1.926e-07, B/full -1.926e-07, B/posonly -1.926e-07。3 アーム共通の下方シフトなので G0/P1/P2/P4 は保守側、P5 は不偏、**P3 のみ通りやすくなる向き**に働く
- P2_ratio: 分母 (Δ_full) が bootstrap で 0 を跨ぐため比の CI は報告しない (coupling_fbw_0813 の家内規約)
- P2_ratio_late: 分母 (Δ_full) が bootstrap で 0 を跨ぐため比の CI は報告しない (coupling_fbw_0813 の家内規約)

## 8. 主張してはいけないこと (spec_posreset_0819 §9 の逐語再掲・厳守)

- dead_frac に基づくいかなる判定・主張（clean eval_loss のみ）
- P1–P4 が全て PASS しても「B1 確定」（E2 の燃料溶接・E3 の組合せ代数・E4 の基準対決が残る）
- 「新特徴は無用」への飛躍（B の等価性は弱い証拠。強い証拠は A の P4 のみ）
- CBP（継続適用）・K=10⁴・実スケール・Transformer への外挿
- 忘却側（OP10）への言及
