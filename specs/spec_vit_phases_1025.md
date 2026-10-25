# ViT の層別沈降: 読み出し整列・切替・通常学習（1025）

状態: 既存走の探索的な保存状態解析。1025、Issa の「入力に近い層は相2、出力に近い層は相1が優勢」という仮説を調べる。既存の μ/σ と FFN 経路比率は閲覧済み。以下の押しの計算結果は未閲覧であり、実行前にこの方法を commit する。

## 対象

6-block ReLU/LN、SL と R0L、全5 seed。task 10 と60の開始直前（task9/59末）、update100、update500の計60保存状態。開始前の重みも保存されているため、この2課題を選ぶ。R0Lの開始状態では元の手順どおりheadを0にする。SLのheadは保持する。学習やoptimizer更新は行わない。

実際の通常学習の移動は、固定の64画像probeに対する各unitの μ、σ（不偏）、r=μ/(σ+1e-12) を、開始→100、100→500で比較する。早期窓100更新の全体を純粋な相1とは呼ばない。

## 信号

最終LN後・head直前の課題250画像の特徴を H、使用する10行をクラス中心化したheadを Vc、乱数ラベルを one-hot にしたものを Q、u=1/10 とする。開始時の装填は旧課題（task9/59）の250画像と旧ラベルで、F=(Q_old-u)ᵀH_old、κ=<Vc,F>/||F||²、VL=κF。100/500時点の装填は同じ定義を現課題へ適用する。

以下のbodyへの信号を、後段のattention・FFN・LNを含めて全parameterに逆伝播する。実測側は画像数 N=250 で割る。

- `fit`: (p-Q)Vc/N。実際の現課題の平均CE。
- `switch`: (p-u)Vc/N。開始時は実際の次課題画像・独立新ラベルに対する期待。100/500では同じ入力のラベルを独立に変えた反実仮想。
- `load`: (p-u)VL/N。整列成分が担う切替の押し。
- `remainder`: (p-u)(Vc-VL)/N。load + remainder = switch。remainderを相2とは呼ばない。
- `P1`: J_HᵀH。既存 `chargpt_earlyphase_1024` と同じ特徴固定・相1型head模型。
- `P2`: J_Hᵀ[H(HᵀH)^+]。同じ特徴固定・共同最小二乗readoutの条件付き相2模型。

P1/P2は正の共通係数を省いた予測器。演算のため各head信号のFrobenius normを1にするが、実測への係数fitはしない。予測器の絶対振幅を実測や相互で比べない。rankはfloat64 SVDで s²>1e-8 s_max²。1e-6/1e-10でrankが変わる場合はそのP2も計算する。Hは250×384であり列独立な共同LS仮定を満たせないので、P2は最小ノルム・擬似逆による拡張である。対応が良くても元の相2定理が成立したとはしない。

R0L開始時の実headとVLは0なので、fit/switch/load/remainderのbody勾配は0。この時点のP1/P2は条件付き予測であり、実headが共同fit済みという主張ではない。

## 位置への運び方と層別集約

同じ固定64画像のCLS位置に対して、6 FFNのunitごとの μ、σ、r のparameter JVPを計算する。勾配方向 g に対し S_r=J_r g、勾配降下の一次変化は dr=-ηS_r。S_r>0が沈降方向。σ>1e-8を有効とし、全unitとprobeでReLUが一度以上開くunitを別に集計する。

- full: すべてのbody parameterの更新を含める。
- local: 自unitのfc1 weightとbiasだけを動かす解析的微分。probe入力の平均をm、標準偏差をs、Cov(u,z)をcとすると、dμ=gW·m+gb、ds=c·gW/s、dr=dμ/(s+eps)-μ ds/(s+eps)²。
- upstream=full-local。これは更新経路の分解であり、相1/相2の分解ではない。

各状態・層で平均S、中央値、沈めるunit割合、零点割合を出し、seed中央値と範囲を報告。符号の零点許容差は max(1e-12,1e-6 max|S|)。P1/P2とswitch/fitの符号・順位対応を同じ非ゼロ集合で計算する。balanced accuracyは実測の沈む/浮く群が各10unit以上の場合のみ。相1と相2を足して100%にした寄与率は出さない。

仮説は、(a)開始時のloadが後段で強く沈めるか、(b)通常学習のfitが前段で沈めるか、(c)そのfitをP2がP1よりよく説明するか、(d)実際の区間変位と整合するか、を分けて読む。平均の符号とunit割合の違い、入力経路による符号反転も残す。SGD方向の診断であり、Adamの実更新や期間全体の因果寄与とは区別する。

## 検算・保存

元tarのcheckpoint SHA256とsource hashesを照合。手動bodyのbaseline CE/精度を元intaskの開始/100/500と照合。load+remainder=switchをparameter勾配と位置への押しで検算。fc1だけの解析微分とJVPを第1・最終層で検算。モデルとoptimizerは更新しない。コード・結果は同名のanalysis/resultsへ、生配列・ログは `~/Projects/obsidian-research-data/vit_phases_1025/` に保存しmanifestをcommitする。
