# sna_loss_gate_1018 — CNN 本走の事前登録

状態: 本走前に固定。Issa 2026-10-18「本走投入して」。
規則は `spec_sna_loss_gate_1018.md` のまま。学習済み本走結果は未観測。

## 実行

- RC CNN、SNA / SNAfrz1 / SNA_lossgate、各seed 10–19、30課題、各400 epoch。
- 学習率1e-3、Adam、c=.6、beta=.01、clip [.005,3]。R=30で束ねる。
- 全腕を同じエンジンとseedで新規に実行。過去の精度は対照データへ流用しない。
- 課題1末のLref校正から課題2以降を制御。ε=1e-4を変更しない。
- 出力 `results/sna_loss_gate_1018/cnn/`、実行ログはその外の `logs/`。
- 課題境界のcheckpointから同じ設定で再開可。異なる設定を拒否。
- 非有限損失・CUDA OOM等は失敗として扱い、科学的ラベルは付けない。
  自動再試行・腕の除外・パラメータ変更はしない。

## 主量と判定

seedごとに課題21–30のonline_accを平均し、10 seedの対応差の平均を主量とする。
95%区間はseedを対応付きで復元抽出するpercentile bootstrap（20,000回、
NumPy default_rng(1018)）。単位は精度の比率（.005 = 0.5ポイント）。

1. **主判定** lossgate − SNA: 区間下端>0なら `BETTER_THAN_SNA`、
   上端<0なら `WORSE_THAN_SNA`、それ以外は `UNRESOLVED_VS_SNA`。
2. **事前指定した副判定** lossgate − SNAfrz1: 区間下端>−.005なら
   `NONINFERIOR_TO_FREEZE`、上端<−.005なら `INFERIOR_TO_FREEZE`、
   それ以外は `UNRESOLVED_VS_FREEZE`。0.5ptは今回の実用的な許容差として
   起草時に選んだ値で、検出力の保証やFableの点予測ではない。
   二つの判定を独立な検証成功として数えず、主判定と対照の比較として示す。

30課題の全3腕×10 seedの行（900行）が揃い、全主量が有限、出力statusが
COMPLETEDのときだけ判定する。不足時は `INCOMPLETE_OR_INVALID`、科学的ラベルnull。
task1で3腕が同じseedごとに一致することも出力から確認する。

## 機構・安全性の報告（主判定を差し替えない）

- 課題2–30の最初10 epochと最後10 epochの平均g・CE（seed単位）。
- 課題末alpha、幅、2alpha幅、c1上限張付率、f2の幅・座席、切替直後CE。
- 全腕の切替CE>100の回数、幅が増え続けるか。過去のgrowとの比較は参考。
- 主判定が良くても、ゲートがほぼ固定へ退化した可能性や尺度の暴走を別に読む。
- 不調だからLrefやεを変更しない。変更を試す場合は別登録・別出力とする。
