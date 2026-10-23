# spec: C3の相2予測を全経路・担い手ごとに照合する

2026-10-24 / Issaの明示依頼。保存状態のみ、新しい学習・optimizer作成・API worker・subagentなし。

## 問いと既知
以前のC3の相2適合はL0自身のup行経路だけだった。論文V13通し稿§3.2は位置を運ぶ重み・上流・biasを分け、§6.2にはL1の100更新の沈める成分を上流が主に運ぶ既知結果がある。これはL0の一歩へ転用できない。今回、同じ相2予測と実CE勾配を同じ全経路へ運び、合計が沈むか、何がそれを運ぶかを直接測る。

## 対象
C3-2、seed0–9、課題末1,4,5,6,9,39,50,89,100（90状態）。L0主、L1記録。時間変化はこの指定端点で報告し、未測定の窓全体の中央値とは呼ばない。checkpoint/Adam状態はchargpt_seed10_1022の検証済みtarから読み出し、展開しない。probeとモデルは既存load10.py/chargpt_earlyphase_1024と同一。相対位置・平均・幅・開口率はseed固定一様probe、損失・最終特徴は現課題chain（最初の位置を除く）。CPU float64、8 threads。

## 信号
H=最終LN特徴、G=H^T H、rank cutoff=最大固有値×1e-10。1e-8/1e-12のrankも記録し異なる場合は該当予測を追加する。
- 相2 iid: J_H^T[H G^+]。旧ラベルの独立等方共分散を仮定する既存のfree-head LS模型。
- 相2 context: J_H^T[H G^+ H^T K H G^+]。Kは同じ2文字文脈の指示行列。正の共通係数は方向評価には不要。各予測内の担い手の加法性は保持する。
- switch: 現入力に独立一様の新ラベルを置いたCE期待勾配 p−1/V。これを切替の押しと呼ぶ。
- fit: 現課題の真の遷移確率Qに対するCE勾配p−Q。実標本ミニバッチのノイズではなく現課題の期待勾配。
switch/fitは共有headの直接微分も含む全parameter勾配を計算する。body由来とhead直接由来を別途保存し、合計を独立autogradと照合。相2のfree-head模型と実網のtied-headの相違を明記し、直接head経路の押しを記録する。

## 担い手と厳密な帳簿
gを各parameter群の排他的な和へ分ける。埋め込み（token共有重み・position）、attention0（その前LN含む）、FFN0前LN、FFN0 up、FFN0 down、attention1（その前LN含む）、FFN1前LN、FFN1 up、その他の後段（FFN1 down・最終LN）。実装のparameter名を保存して全parameterの被覆と非重複を検査する。FFN upにbiasが無いことを確認し、bias経路は0と明記。後段は対象前活性に直接影響しないため、JVPが0であることを検算する。
L0ではown=FFN0 up、上流=embedding+attention0+FFN0前LN。L1ではown=FFN1 up、上流=embedding+attention0+FFN0前LN+FFN0 up+FFN0 down+attention1+FFN1前LN。L1向けにFFN0群の合計も併記。自unit up行と自己出力成分を混同しない。
各群g_aについて JVPでdmu_a、ds_aを計算し、S_r,a=dmu_a/s−mu ds_a/s^2（正ならΔr=−ηS_rが沈む）。第一項を平均の項、第二項を幅の項と呼ぶ。全群の和を、全parameterを同時に動かす独立JVPとunitごとに照合する。自分のup行は既存解析の解析式とも照合する。上流は全体からの引き算だけで済ませず、群ごとの微分から足す。
SGDと、保存Adamのbias補正済み二次モーメント分母を固定した対角重みを使う。momentumを含む実Adam更新の再生ではない。学習ステップは実行しない。

## 主集計と方向照合
主は全512unit。閉じたunitも上流が動けば位置が変わるため除外しない。生きた集合を記録し、以前との比較用集計も残す。
各状態・各層・各metric・各信号・各担い手について、S_rの全unit平均、中央値、正/負/数値ゼロの割合、平均の項と幅の項、n_aliveを出す。最も大きい正の平均を持つ担い手と、最も大きい負の平均を持つ担い手を記録する（全体が沈むかは合計で判定）。実勾配の全unit平均が正のseed数/10を端点ごとに報告する。中央値やunit割合を平均の代わりにしない。全seed同符号ならその事実、混在なら混在と報告し、後付けの支持閾値を置かない。
相2予測対switch/fitを全体・own・上流合計・各群で比較し、符号一致、balanced accuracy（各符号10unit以上の時のみ）、Spearman、cosを出す。数値ゼロは各vectorのmaxabs×1e-10、感度1e-8/1e-12。同一state内のunitを独立seed扱いしない。主の範囲は10seedのmin/max。

## 先行gate（本測定前に通す）
seed0の課題1・9・100末。共有head込みCEの独立autograd、body+directhead、carrier合計、平均+幅、own解析式の相対誤差<=1e-8。旧配列がある課題1/9はr・own P2・own switchを相対1e-8で一致確認。L0/L1のr全体JVPとP2のlog pseudodet微分を正規化方向の中央有限差分で確認（epsilon1e-4,1e-5,1e-6、少なくとも一つ相対1e-4以内。ReLU折れ目を越える試行は記録）。不一致なら本走を止めて修正・全gate再実行。gateで実行時間を記録する。数式・判定を結果に合わせて変えない。

## 限界と置き場
固定状態の微分帳簿で、長期軌道や100更新の因果寄与率ではない。担い手の一致は相2平均の仮定成立の証明ではない。以前の判定は変更しない。
analysis/chargpt_phase2carrier_1024、results/chargpt_phase2carrier_1024（states/comparisons/carriers/checks/provenance/summary、raw npz）。生配列は既定の研究データ置き場へSHA256付き退避。結果は既存の統合ノートに追記し、mainへpushしてworktree/branchを片付ける。
