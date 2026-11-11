# ViT装填費用とMLP stage1_ctrlをつなぐ文献照合（2026-11-11）

担当範囲: 一次資料7本の照合。実験・原稿編集は行っていない。Obsidian研究contextとcollaboration_changesを確認した。COLLABORATION_NOTICE.mdは見つからず、共同編集の未反映・未保存差分は通知されなかった。旧 `results/vit_long_1024_2blk_rlfix/loading_cost_literature_1029.md` は調査の入口として読んだが、その結論は引き継がない。

以下は引用文の転記ではなく要約。各文献の由来要約は200英語語相当未満に抑えた。ページはPDF内のページ番号であり、通しのPMLRページとは区別する。

## 1. 読んだ一次資料と今回との距離

### Lyle et al., *Understanding Plasticity in Neural Networks*（ICML 2023）

[正式書誌](https://proceedings.mlr.press/v202/lyle23b.html)・[会議版PDF](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf)。PMLR 202:23190–23211。

確認箇所: §4.1 pp4–5、§5、§6.2 p8、Appendix A.1 p13、B.1 p14。

乱数ラベルMNISTの急な切替で、Adamの一次・二次モーメントの追従速度の不一致が大きな更新を生み、ReLUを飽和させ得る。β2とεの調整がその例を改善する。ノルム・特徴rank・dead unitsのいずれか一つを普遍的な原因とする読みには反例を示す。最終層resetはtoy RLで有益な介入として検証され、悪い条件付けの源を一時的に取り去ると解釈される。B.1のoptimizer-only resetの不成功は、そのtoy RL設定での結果。今回の切替直後25更新のAdam記憶とは時間・介入が異なる。A.1は5000枚・隠れ2層×1024であり、stage1_ctrlの784–100–10とは異なる。論文に列挙されるreset効果だけから、体が無傷だとは判定できない。

### Lyle et al., *Disentangling the Causes of Plasticity Loss in Neural Networks*

[正式書誌](https://proceedings.mlr.press/v274/lyle25a.html)・[会議版PDF](https://raw.githubusercontent.com/mlresearch/v274/main/assets/lyle25a/lyle25a.pdf)・[arXiv初版](https://arxiv.org/abs/2402.18762)。**会議はCoLLAs 2024、PMLR公開書誌は274:750–783, 2025。ICML 2024ではない。** PDF表紙とPMLRの著者表示順にも差があるため、引用する版の公式BibTeXを使う。

確認箇所: §3.2 pp5–6、§4.1 pp8–9、Appendix F.3 pp24–25・Fig.20。arXiv v1では同じReLU説明がE.3/Fig.18。

切替後のerasing相（確信減少・entropy増加）と、その後の新課題への適応を区別する。F.3ではlogitを弱めるhead側と特徴側の経路を挙げ、後者の縮小、前活性を下げる勾配、古い二次モーメントによる過大更新を順に説明する。Fig.20はoptimizer resetも比較し、死の増加を減らしても縮小過程が残る。前活性の平均・分散の変化と線形化も対象で、dead ReLUだけの論文ではない。旧ラベル対応を壊すP/P0、ViTの再学習時間、確信消失後の残存費用は同定していない。§4.1のγ=0は**RL割引率**であり、LN利得でもlabel-smoothing係数でもない。

### Lyle et al., *Normalization and effective learning rates in reinforcement learning*（NeurIPS 2024）

[会議版PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/c04d37be05ba74419d2d5705972a9d64-Paper-Conference.pdf)・[arXiv](https://arxiv.org/abs/2407.01800)。Normalize-and-Project（NaP）の論文。

確認箇所: §2.2 p3、§3–4 pp4–6、Appendix A.8 pp19–20、C.4 pp24–25、**D p26**、code checklist p31。

正規化が作る尺度不変性の下では、重みノルム増大が実効学習率を低下させ得る。NaPは非線形前の正規化と重み射影を組み合わせるが、**最終の線形出力層は尺度不変でないため射影しない**（§4 p6）。正規化の勾配混合は休眠unitの復帰を助け得る。Appendix Dはscale/offsetのdriftを明示し、ReLU等の正斉次条件では両者の共同ノルム制御、一般の非線形では初期値1/0への弱い正則化を提案する。「層重みを固定すれば全ての尺度が固定」は先行研究でも成り立たない。ただしViTのhead行ノルム固定による最終LN利得の補償を直接実験したものではない。checklistでは当時の専用コード公開許可を得ていないと明記する。

### Nikishin et al., *The Primacy Bias in Deep Reinforcement Learning*（ICML 2022）

[正式書誌](https://proceedings.mlr.press/v162/nikishin22a.html)・[会議版PDF](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf)・[著者コード](https://github.com/evgenii-nikishin/rl_with_resets)。PMLR 162:16828–16847。

確認箇所: §5.1 p4、§5.4 pp6–7、Appendix B pp12–14/Figs.9,13。

reset範囲はアルゴリズムごとに異なる。SPRは5層Q網の最終線形層だけ、DrQは7層中後ろ3層、SACは3層網全体。対応するoptimizer統計も既定ではresetし、replay bufferは保持する。DrQではoptimizerだけのresetに明瞭な利益がなく、重みだけresetも標準resetと近い。SPRで体まで戻すと不利な例があり、保存された表現の有用性を示す。これはheadだけで有益な介入が可能という先行例だが、自己生成価値・探索・replay再学習を含む。今回のランダムラベル切替、T90、新品との差の分解、出力尺度の機構を検証したものではない。

### Lee et al., *PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning*（NeurIPS 2023）

[正式会議頁](https://papers.neurips.cc/paper_files/paper/2023/hash/c464fc4516aca4e68f2a14e67c6f0402-Abstract-Conference.html)・[会議版PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/c464fc4516aca4e68f2a14e67c6f0402-Paper-Conference.pdf)・[著者コード](https://github.com/dojeon-ai/PLASTIC)。

確認箇所: §3/Figs.2–3 pp4–5、§4.1 p6、Appendix A.2 p16、公開 `synthetic/output_adaptation.py`。

CIFAR-10のクラス対応を変える合成課題では、LN/SAMよりCReLU/resetがlabel plasticityに効く。網はconv3層とFC3層head、SGD+momentum、1切替500更新、100切替、30seed。したがって「label変化に対してLN単独で十分でない」は先行するが、headは最終1層を意味しない。コードの`reinit_policy()`はpolicy内Linear等を再初期化し、公開版は10chunkごと、optimizerを作り直す処理はそのreset節にない。論文では間隔5/10/20を探索。評価は主に最終test精度で、ViTの初期確信消失後の時間遅延とは違う。PLASTIC†の一部設定には体のShrink-and-Perturbも入り、全てをheadだけの救済と呼べない。

### Dohare et al., *Loss of plasticity in deep continual learning*（Nature 2024）

[正式論文](https://www.nature.com/articles/s41586-024-07711-7)・[正式版全文のPMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/)・[著者コード](https://github.com/shibhansh/loss-of-plasticity)。Nature 632:768–774、2024-08-21公開。

確認箇所: Methods「Continual ImageNet」「Class-incremental CIFAR-100」、Extended Data Fig.1a、本文CBP説明、Methods「Understanding loss of plasticity」。

Continual ImageNetは課題境界で最終2出力への重みをゼロ化する。それでも長期劣化が起きる。Methodsがidealでないとする理由は、境界時刻という特権情報を与えるため。加えてclass-incremental CIFAR-100には各increment開始時にoutput layerを再初期化する**head-reset baseline**がある。旧ノートの「最終層reset ablationは見つからない」は修正が必要。CBPは各隠れ層の低利用unitを選んでincomingを再初期化しoutgoingをゼロにするため、headだけのresetではない。head処置の異なる課題を横断して「同じ障害」と判定することはできない。

### Wei et al., *Mitigating Neural Network Overconfidence with Logit Normalization*（ICML 2022）

[正式書誌](https://proceedings.mlr.press/v162/wei22d.html)・[会議版PDF](https://proceedings.mlr.press/v162/wei22d/wei22d.pdf)・[著者コード](https://github.com/hongxin001/logitnorm_ood)。PMLR 162:23631–23644。

確認箇所: §3 pp2–4、§5 p7、Appendices A–C p12、`common/loss_function.py:47–57`。

正のlogit全体スケールはargmaxを保ちつつ最大softmax確率を変える。LogitNormは学習時に各入力のlogitベクトルを正規化してCEへ渡す。固定の後処理temperatureとは異なる。論文はOOD検出・calibrationを扱い、継続学習の切替費用やAdamの記憶を実験しない。公式実装は`norm`をdetachしないため、forwardの確信だけでなくbackwardの射影・尺度も変わる。従って今回の識別実験でLogitNormを「確信だけを変えた対照」と呼べない。またsoftmaxは共通logit平行移動に不変なので、今回の尺度測定ではraw logit normとクラス中心化logit normを区別する必要がある（後者は今回の解析上の注意）。

## 2. 旧照合から直す点

- `Disentangling`のICML 2024表記は誤り。引用はCoLLAs 2024/PMLR 2025の版を明示する。ReLU説明の参照は正式版F.3/Fig.20へ更新する。
- 同論文§4.1のγをLN利得に結びつけない。割引率である。
- Dohare正式版のhead-reset baselineを記録する。「見つからなかった」を「存在しない」に変えない。
- 旧ノートの「最終LNが特徴縮小を閉じ、体に押しが届かない」「全遅延が確信を消す時間」「悪いのは向きでなく自信」は、これらの文献から導けない。現行のViT追記に基づく再評価が必要。
- LNのscale/offset driftは既知の注意点。今回の特有部分は、head行ノルム制約下で利得が伸び、確信と切替費用が残る具体的な測定に置く。

## 3. 文献の要約とは独立の代数的予測

以下は今回の識別実験のための直接計算であり、上の論文に帰属させない。`h`をhead入力、`z=Wh+b`、`p=softmax(z)`、`u=1/C`とする。新ラベルが入力から独立な一様乱数なら

\[
\mathbb E_y\nabla_h\ell = W^T(p-u).
\]

旧ラベルとの対応はこの式に現れない。さらに

\[
z^T(p-u)=\int_0^1 \mathrm{Var}_{\operatorname{softmax}(tz)}(z)\,dt\ge0.
\]

従ってbiasがゼロなら`hᵀ E∇hℓ ≥ 0`であり、logitが非一様であるだけで特徴の半径を減らす**勾配成分**が生じ得る。biasありでは

\[
h^TW^T(p-u)=z^T(p-u)-b^T(p-u)
\]

と分ける。これは各unitが沈む証明ではない。hは独立に最適化するパラメータではなく、実際の前活性変位はbody Jacobian・activation微分・入力平均・bias・optimizerに依存する。

P0を`W'=ΠW,b'=Πb`とすると、logitは`Πz`、確信とentropyは不変で、期待body勾配も完全に不変。**旧クラス名への対応を失うことと押しを失うことは、そもそも別である。** biasを同じΠで動かさないP0では、この厳密不変性は適用できない。独立列置換Pも単一Πではないため、不変性はない。

headの単純縮小は、forwardの`p-u`とbackwardの`Wᵀ`を同時に変える。bias=0かつ十分小さいscale `a`なら`p(az)-u≈a z_c/C`なので期待body勾配は`a²`の次数になる。一方、同じノルムでもWとhの幾何でzは変わる。このため「ノルム」「logit尺度」「確信」「期待勾配」を別々に測る設計に意味がある。

## 4. 今回まだ判定していないこと

この7本から支持されるのは、切替後の出力抑制圧、前活性分布の変化、optimizerによる増幅、正規化と尺度制御の限界という**候補部品**である。P/P0/方向ランダム化の同一条件比較、同じ確信での異なるbody更新、再装填前の期待押し、確信消失後の停滞、予防と事後回復の差は、今回の既存・追加データで確かめる事項である。

文献上の限定: 検索対象7本でこれらの完全な分解は確認できなかった。網羅的新規性の証明ではない。Lyle三論文の専用実験コードは今回特定できず、NaPについては会議版の非公開説明までを確認した。RL基盤コードの存在を専用実験コードの公開と数えない。

## 5. 実装参照（読取のみ）

- [PLASTICの合成label課題](https://github.com/dojeon-ai/PLASTIC/blob/37360714ee07e23ffe810c71f9783da83d497777/synthetic/output_adaptation.py): `reinit_policy`行156–161、reset行297–302。Linearのbiasも`reset_parameters`に含む。LNのscale/offsetを再初期化する処理ではない。確認commit `37360714ee07e23ffe810c71f9783da83d497777`。
- [LogitNorm loss](https://github.com/hongxin001/logitnorm_ood/blob/0a60eeffb7dfc970fe68e07c5649ea1c9c8244c6/common/loss_function.py): 行47–57。分母normにautogradが通る。確認commit `0a60eeffb7dfc970fe68e07c5649ea1c9c8244c6`。
- [Nikishinの公式実装](https://github.com/evgenii-nikishin/rl_with_resets): READMEはSAC/DrQのreset差分の位置とSPR用設定ファイルを明記。
- [Dohareの公式実装](https://github.com/shibhansh/loss-of-plasticity/tree/3138c5556859e0e2918d40f113b40699b6575cfa): 今回は論文との対応先を確認。全実装の監査はしていない。参照commit `3138c5556859e0e2918d40f113b40699b6575cfa`。
