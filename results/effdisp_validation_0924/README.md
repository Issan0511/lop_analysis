# 実効変位と幅の環境横断検証 — 結果 0924

Codex、2026-09-24。**完了。新規205系列＋既存参照10系列、全215系列を解析。** 本走は00:33–03:43 JST（約3時間10分）。
予測と判定を先に登録し、実装をgpt-6-solへ分担した。主比較はAdam。CondA22条件、RL-MNIST/PermutedMNIST各9活性化、PMのSGD参照を実行した。

## 結論

**「実効変位と削りの釣り合いで、どの環境でも幅が決まる」という強い一般化は支持されなかった。一方、RL-MNISTの適応Snake c=.3では、学習と非零の更新を保ちながら幅が頭打ちになる現象を5/5 seedで確認した。**

ただし、このSnakeの後半はαがほぼ下限へ張り付く。適応機構による救済と断定できない。前半の収支係数から後半を予測する試験も通らなかった。
残せるのは、**収支の恒等式と、成立する条件を限定した観測**。更新則から平衡の高さを予測する理論は未完成である。

## 1. 何を別々に判定したか

第一層について、固定した中心化入力共分散Σで

\[
V_t=\|W_t\|_\Sigma^2,\quad Q_t=\|W_{t+1}-W_t\|_\Sigma^2,\quad X_t=\langle W_t,W_{t+1}-W_t\rangle_\Sigma,
\qquad \Delta V_t=Q_t+2X_t.
\]

- **P1・近い相殺**: 最後20%の課題で、供給Qと削り−2Xが近く、非零の相対更新と負の角度が続く。
- **P2・頭打ち**: P1に加え、同じ窓でVの傾きと前半/後半比が小さい。無限時間の定常性の証明ではない。
- **P3・独立予測**: 課題6〜前半末で収支係数を推定し、後半を再帰予測する。単に前半末の値を保つ予測より20%以上良く、MAPE≤20%等を満たす。
- **P4・素のノルムとの分離**: P2と、raw/effectiveの二乗量の傾き差≥.2。

群PASSは5 seed中4以上。lateはCondAのtask81–100、MNIST/CIFARのtask41–50。
同じ窓の `R=D/(2|c|)` の一致と恒等式の残差は、独立した科学的支持に数えない。

## 2. 環境をまたいだ結果

数値の出所: [verdict.csv](verdict.csv)、[summary.md](summary.md)。括弧内は通過seed数。

| 環境・条件 | P1・相殺 | P2・頭打ち | P3・後半予測 |
|---|---|---|---|
| RL-MNIST leaky | PASS (4/5) | FAIL (0/5) | FAIL (0/5) |
| RL-MNIST 適応Snake c=.3 | PASS (5/5) | PASS (5/5) | FAIL (0/5) |
| RL-MNIST 適応Snake c=.6 | PASS (5/5) | FAIL (1/5) | FAIL (0/5) |
| RL-MNIST その他6活性化 | 群PASSなし | 群PASSなし | 群PASSなし（ReLUは判別不能） |
| PermutedMNIST Adam・9活性化 | 群PASSなし | 群PASSなし | 7群PASS、ELU/SiLUはFAIL |
| CondA・22条件 | 群PASSなし | 群PASSなし | r10と適応Snake c=.3だけPASS (各4/5) |
| PM SGD leaky参照 | FAIL | FAIL | FAIL |
| 既存RL-CIFAR leaky raw/std | 各PASS (5/5) | 各FAIL (0/5) | 各FAIL (0/5) |

PMでP3を通った7群はleaky、ReLU、GELU、固定Snake、適応Snakeの3設定。それぞれ5/5で通るが、主に**成長する軌道**の予測である。
CIFARは0918の保存runの事後参照であり、0923の主張ノートのrunの厳密な再現ではない。

![RL-MNISTの成長・凍結・活動を保つ頭打ち](overview_rl.png)

図は全5 seedの中央値と範囲。説明のためleaky/ReLU/適応Snake c=.3を選んだ事後図で、全群は登録表と個別図に残している。上段の縦軸範囲は各条件で異なる。

### 有望だった条件の限定

RL適応Snake c=.3、late task41–50、seed中央値:

- 収支比B=1.0016、Vのlog傾き=.0215、rawとの傾き差=.6164。P2/P4は5/5。
- 課題開始精度9.95%→課題末100%。課題内オンライン精度98.02%。低応答の登録基準には達しない。
- 第一層α中央値=.05、αclip率99.8%。第二層clip率100%。αの下限は.05。**後半はほぼ固定αになっている**。
- P3のMAPEは8.60%。一定値予測の3.23%より悪く、独立予測の優位は支持されない。

出所: `summary.md` のRL表・alpha表、`verdict.csv` のmnist_rl/SNA03/P1–P4。
オンライン精度は `seed_summary.csv` のmnist_rl/SNA03/effective、`late_online_acc`（41–50）のseed中央値。

## 3. 先に勾配が尽きるという懸念

RL-MNISTでは、その懸念に対応する低応答と成績低下が実際に見られた。

| 活性化 | LOW_RESPONSEに達したseed | onsetの中央値 | late課題末精度のseed中央値 |
|---|---:|---:|---:|
| ReLU | 5/5 | task5 | 11.6% |
| ELU | 5/5 | task11 | 10.9% |
| GELU | 3/5 | task12（到達した3seed内） | 11.6%（全5seed） |
| SiLU | 0/5 | 未到達 | 57.4% |
| leaky | 0/5 | 未到達 | 100% |
| 固定Snake α=1 | 0/5 | 未到達 | 99.2% |
| 適応Snake c=.3/.6/1 | 各0/5 | 未到達 | 各100% |

出所: `summary.md` のRL表とLow-response表。LOW_RESPONSEはどちらかの隠れ層の平均絶対微分<1e−8またはactive unit fraction<.1が3課題連続、最初の課題をonsetとする。
到達13seedは全て、登録した停止前20遷移を確保できず `INSUFFICIENT_ACTIVE_HISTORY`。短い窓へ事後変更して平衡の成否を判定し直していない。
ELUは今回F.eluとexpによる微分を使い、以前のexpm1実装の丸めによる床とは異なる。低応答でも数学的な微分が厳密に0とは限らない。

**STOPPEDとLOW_RESPONSEは別物。** STOPPEDは相対更新量が小さいという登録ラベルで、学習不能やゼロ勾配を意味しない。CondAにはSTOPPEDでもMSEが大きく改善する条件がある。
SiLUの未到達も健全性の保証ではなく、この50課題で登録閾値に届かなかったという意味である。

## 4. 入力次元・ビット数の結果

定義の上では、実効変位は

\[
\|D\|_\Sigma^2=\sum_j\lambda_j\|Dv_j\|^2
\]

である。必要なのは**分散方向、分散の大きさ、更新の向きとの重なり**。座標数やrankだけでは決まらない。
CondAではrandom bit数rが課題内の共分散rankで、traceΣ=r/4。persistent bitは課題内分散0だが、平均を通じて学習へ効く。

以下は追加の事後診断。late task81–100、seedごとに比を作ってから5seedの中央値。Dは中心化実効変位、Rは中心化幅。

| 比較 | D比 | R比 | 読める範囲 |
|---|---:|---:|---|
| m20/r5 → m40/r5、k1 | .979 | .823 | 生次元だけ増えても単純な増大は出ない。ただし教師/初期化も変わる |
| m20/r10 → m40/r10、k1 | .762 | .565 | 同じrankでも水準は変わる。純粋な次元効果ではない |
| m40/r10、k1 → k2 | 1.273 | 1.168 | 同じ次元・教師・初期化で反転数を増やすとDが増えた |

出所: [dimension_diagnostic/summary.md](dimension_diagnostic/summary.md)、`paired_median.csv`のlate行。
同じm20でrを2/5/10とした比較も `addon_pairs.csv` に全seedを掲載している。r変更は条件付き入力分布と課題難易度も変える。

事後の平均成分診断では、m20/r5/k1の `||Dμ_prev+Δb||` は4.33、中心化Dは.935（各seedのlate平均を取った後の中央値）。平均変化の二乗量/中心化Qの窓比の中央値は19.7。
中心化Σに見えない成分を、無害な「死重」と呼ぶのは適切でない。

ユーザーの画像複製案も理論メモで整理した。q回の単純複製はrankを保つが非零固有値とtraceをq倍にする。分散を保つ複製と、初期関数・第一層学習率を揃える対照を分ける必要がある。**このCIFAR複製介入自体は今回実行していない。**

## 5. 事前予測との照合

- RL leakyで近い相殺が出るという予測はP1で支持。厳しい頭打ちP2は不支持。
- PMでは共分散の変化項が必要という予測と整合。actualでのsum|G|/sumQのseed中央値はAdamの各群で1.85–4.50。固定referenceの幅だけで実入力上の幅を説明できない。
- CondAのk7/k1でD>1.2倍という予測P5はFAIL、比の中央値1.066、閾値通過0/5。
- CondAのAdam>SGDかつ高ε<通常Adamという予測P6はFAIL、両方向が揃うのは2/5。倍率中央値は1.005と.983。学習率も異なるのでoptimizerだけの因果比較ではない。
- 適応SnakeはRLで低応答を避けたが、固定Snakeも今回の地平では避けている。適応そのものの優位は確定しない。
- m40/r5でraw/effectiveの乖離が増すという追補予測はreport-only。幅比の比は1.04程度で、強い分離の証拠として扱わない。

## 6. 理論として残したもの

[理論メモ](../../docs/effdisp_validation_0924_theory.md)に、以下を導出した。

1. 恒等式と確率的定常条件 `E[Q]+2E[X]=0` の区別。平均の比への置換には追加仮定が要る。
2. 条件付き供給q(V)・削りa(V)から固定点と局所安定性を調べる式。
3. `W_next=(1−a)W+ξ` の収縮＋供給モデル。幅の定常二乗量は `ν/[a(2−a)]`。DがWに依存していても、小さな見かけの弾性を持ち得る。
4. 共分散の小さい固有方向が遅く緩和すると、rawが増えながらeffectiveが落ち着き得るという条件付き説明。
5. 特徴複製時のSGD/Adamの正確な学習率・ε対応。

これは今回の観測に合わせた条件付きモデルであり、最適化器から係数を導いた普遍理論や新規性の証明ではない。元ノートの強い因果の一方通行を、この追加実験だけで確定することもできない。

## 7. 監査・再現

- 訓練固定commit: `e83f02d9457d23af34172562a4a45061fdf43d42`。全20suiteのmetadataで一致。
- 登録: `290c4f8`、`2ed7321`、`89b8789`、`5502cc1`。予測本文は結果に合わせて書き換えていない。
- 最終解析コード: `6ee47fe41f593e13c82e4feec87a93c026223a54`。
- 32テスト通過。訓練の勾配/optimizer/境界、heldout漏れ、ゼロQ、混合環境の低応答判定を含む。
- 最終監査で、CondA専用の空列をMNISTが誤選択する低応答集計bugを修正し、全系列を再解析した。P1–P6の閾値や訓練データは変更していない。
- MNISTの実画像とCondAの全支持点から前活性変化を直接再計算し、保存共分散によるQ/Vと一致。[independent_input_check.json](independent_input_check.json)。
- 35,000計量別遷移行。恒等式の相対残差最大1.95e−15は帳簿の検算で、仮説支持には数えない。
- 生データ・全ログ: `/home/issan/Projects/obsidian-research-data/effdisp_validation_0924/`。全ファイルのサイズとSHA256は [backup_manifest.json](backup_manifest.json)。部分集計は参考履歴として保存し、**最終結果はこのフォルダ**を正本とする。
- 再現コマンド・データハッシュ・条件差は [provenance.json](provenance.json)。全seed表は [seed_summary.csv](seed_summary.csv)。
