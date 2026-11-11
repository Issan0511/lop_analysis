# ViTのLN作用と読み出し費用：旧証拠の範囲

2026-11-12 JST。新走Study Bの結果を見る前に、既存の保存結果・実装を照合した記録。Study Aの結果は `study_A_review.md`、有限εを含む代数は `theory_ln_scope.md` に分ける。旧ノートの記述を新実験の結果として扱わない。

## 三つの問いを区別する

| 問い | 旧証拠が支持すること | 旧証拠だけでは確定しないこと |
| --- | --- | --- |
| 最終LNは装填読み出しから戻る押しを変えるか | はい。平均方向を消し、中心化特徴の半径方向を有限εに応じて抑制する | この局所量だけによる、学び直し遅延の大きさ・媒介割合 |
| 読み出しパラメータの除去で消える遅れにLNは必要か | いいえ。LNなしViTでも同一保存状態からheadをゼロ化すると5 seedすべて速くなる | alignmentだけ、normだけ、または最終LNだけが費用の原因であること |
| P後に残る費用が小さいのはLNのためか | 旧task100の比較はLNありViTのみで、対応するLNなし比較がない | LNの因果的な説明。新走Bはtask30のLN履歴比較を追加するが、旧task100と同一条件ではない |

## LNなしViTでの直接対照

既存 `vit_long_1024_2blk_rlfix/addendum_f/summary.md` のtask30・S・次課題学習。T90は更新数。元headとゼロheadは同じ保存bodyから始める。

| seed | 元head | ゼロhead |
| --- | --- | --- |
| 10252010 | 4250 | 725 |
| 10252011 | 2150 | 975 |
| 10252012 | 2900 | 800 |
| 10252013 | 2075 | 1150 |
| 10252014 | 3150 | 1275 |

`analysis/vit_long_1024/posthoc_2blk/fable_f_head0.py` は全head weight/biasをゼロ化する一方、bodyとheadのAdam履歴を保持する。今回の全腕head Adam消去とは異なる介入である。旧新品参照はtask50なので、「新品より速い」「遅れは全部head側」という旧ノートの言い方を、課題を揃えた完全分解として再使用しない。直接支持されるのは、LNなしでもheadパラメータへの介入で短縮できる遅れが存在すること。

この比較ではPをかけていない。したがってP後の残存費用への証拠に読み替えない。

## 半径方向の伝達

既存 `analysis/vit_long_1024/posthoc_2blk/j5_rlfix.py` のLN逆伝播と、別worktree `ln_loading_1026/analysis/ln_loading_1026/radial_decomp.py` の有限ε実装を照合。最終LNの中心化特徴分散をvとすると、affine γを掛けた勾配について、半径方向と接線方向の伝達係数比は ε/(v+ε)。厳密な半径除去はε=0の理想化で、実装の正のεでは小さな残りがある。

RMSNormにも半径抑制があり、LNの平均中心化だけが担う効果ではない。既存 `results/vit_long_1024_2blk_rlfix/addendum_e/summary_e.md` の5 seedでは、装填成分を通した押しが沈める側のunit割合は直通86–93%から実際54–57%へ減る。これは一様な次ラベルへの期待誤差での局所的分解であり、回復曲線・T90とは別指標である。

## 旧P後比較

旧 `results/p_residual_loading_1111/summary.md` では、LNありReLU ViTのtask100でP後ゼロ化利益は2.29［−1.75,6.79］pp。旧 `results/readout_separation_1111/synthesis.md` ではP後0.1倍化利益は−1.42［−3.75,1.02］pp。いずれも区間がゼロをまたぐので未解決であり、費用ゼロや同等性の証明ではない。対応するLNなしtask100は調べていない。

## 出典と作業範囲

一次出典はすべて `/home/issan/Projects/claude/proj_004_drift/` 以下（半径分解だけ `/home/issan/Projects/claude/wt/ln_loading_1026/`）。この作業では既存ファイルを読んだだけで、旧結果・旧ノート・他worktreeを編集していない。担当の独立照合と親のObsidian読取は、canonical `a434dedc`・共同編集差分なしの時点で実施した。新走Bの途中値はこの文書に使用していない。
