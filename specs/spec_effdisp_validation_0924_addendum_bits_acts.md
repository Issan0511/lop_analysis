# 0924 追補 — Issa のビット数・活性化の追加依頼

本走前の追補。元spec commit290c4f8は保持。2026-09-24 JST、性能を読む前に固定。

## 1. CondAの二つのビット数

実装上の元CondAは、m20のうちf15が課題内固定・境界で反転し、r5が毎step独立に変わる。
元specのkは境界1回で反転する個数。これに加え、rそのものを変える。

- 元の7腕を保持。
- Adam lr.001 epsilon1e-8、leaky.1、k1、T10000、m20のままr2/f18とr10/f10を追加。
- 同条件r5/f15/k1へReLU、ELU(alpha1)、exact GELU(erf)、SiLUを追加。leakyと合わせ5活性化。
- 合計13腕、各seed100–104、100tasks。全交互作用の総当たりはしない。
- Σは各腕diag(0×f,.25×r)。full supportは2^r点（最大1024）を列挙して評価。
- r変更は入力分散のrankと教師の制限先をともに変える。「反転頻度だけの効果」と呼ばない。r増で機能的困難さが同じとは仮定しない。

## 2. MNIST活性化比較

RL-MNISTとPMの両方で、leaky.1、ReLU、ELU(alpha1)、exact GELU、SiLUの5腕を同条件で実行。各seed100–104、50tasks。
元specのELU任意追加条項をこの明示依頼により置換。活性化を結果で選別しない。
canonical updates（RL30000/task、PM625/task）は維持。短いsmokeの速度から4時間超が見込まれた場合は、結果を見る前の追加登録で全活性化共通の地平短縮等を定め、変更理由を残す。未完を成功扱いしない。

## 3. 判定の適用と追加予測

P1–P4は固定Σの系列（CondA、RL、CIFAR、PM reference）に適用。
PM actualは別列で全掲載し、共分散変化項Gを含めた恒等式の残差・late幅の傾き・平均幅比・sum|G|/sumQを報告する。P3のGを含まないclosureに無条件に当てはめない。
tableのtask=tは末尾時点。D_t=W_t−W_(t−1)、V列は前時点、V_next列は末尾。校正6..T/2、予測はT/2+1..T。

Codexの追加予測（Issaの予測ではない）:

1. ReLU/ELUでは応答消失による停止がleakyより多く、幅の頭打ちだけなら偽陽性が出る。P1の活動量と精度/応答を合わせる必要がある。
2. GELU/SiLUにleakyと同じcの範囲を予測しない。更新が続く期間には負のXと供給の相殺が見える可能性があるが、全5活性化へのP2/P3成立は予測しない。
3. CondA rを増すとrawとcenteredの見ている空間の差が縮むと予測する。ただし実際にどの方向へWが蓄積するか次第であり、raw/effective成長差の単調減少はreport-only。
4. k増とr増は違う介入である。mean shift、effective width、raw normの3系列を併記し、中心化で見えないpersistent成分を無害と断定しない。

非有限値の腕や未学習の腕を除外して「全環境で成立」と書かない。活性化の応答・精度を記録し、active balance / transient growth / stopped / divergent を区別する。
