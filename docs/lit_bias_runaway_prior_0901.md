# bias 暴走に関する最近接先行 2 本の精読記録（2026-09-01）

## 読み方と証拠格

- **[定理]** 論文中で定理・系として証明された記述。
- **[表の数値]** 表に明記された値。
- **[図の目視読み]** 軸・曲線からの概数。生データではない。
- **[本文]** 本文・脚注・キャプション・付録の明記。
- **[式]** 論文中の定義式・更新式。
- **[代数照合]** 論文の定義式から本メモで行った展開。論文自身の主張ではない。
- **[コード]** 公式コードで確認した事項。
- **[TeX 注釈]** arXiv source に残る非表示コメント。公開 PDF 本文ではない。
- **[不在検索]** 検索語、検索対象、見つからなかった範囲を併記した記録。悉皆性は主張しない。

対象版は Paper 1 が **arXiv:2111.02154v1（2021-11-03、18 pp.）**、Paper 2 が **arXiv:2402.18762v1（2024-02-29、32 pp.）** である。[本文: 各 arXiv abstract page / PDF] Paper 2 は CoLLAs 2024 採択論文で、PMLR 274 の書誌年は 2025、proceedings PDF は 34 pp. だった。[本文: PMLR landing page / proceedings PDF] arXiv 版と proceedings 版の結論上の差は見つからず、後者では付録挿入により後半の図番号が概ね 2 つ進む。[本文: 両 PDF の節・図キャプション照合]

---

## 1. Regularization by Misclassification in ReLU Neural Networks

書誌: [Cornacchia et al., arXiv:2111.02154v1](https://arxiv.org/abs/2111.02154)。

### 1.1 設定の完全な記録

**理論設定。** ネットワークは全結合 ReLU で、一般の深さを扱う Theorem 1 と、単一ニューロン又は 1 隠れ層を扱う後続結果に分かれる。[本文: §2–§4] Theorem 1 は homogeneous activation を持つ任意の深さについて、誤分類例に対する 1 回の十分小さい勾配更新が各層の重み Frobenius norm を減少させる。[定理: Theorem 1] 同定理では `x` に定数 1 を付加し、bias を行列 `W_i` に暗黙に含めるが、通常の明示的 bias parameterization とは gradient が異なると本文が注意する。[本文: §2, Equation (2) 直後] Corollary 1 は hinge loss、勾配流、margin parameter `β=0` の下で、各層の重み norm が時間に対して非増加であるとする。[定理: Corollary 1] これは**各隠れ bias の符号付き単調減少**を述べる定理ではない。[定理: Theorem 1 / Corollary 1 の対象変数]

単一ニューロン解析では Gaussian input と pure label noise を用いる。[本文: §3] 主結果は bias なしの 1 ニューロンで、入力 `x ~ N(0,I)`、ラベル `y` は独立な一様 `{-1,+1}`、minibatch size 1 の SGD である。[定理: Theorem 2] Theorem 2 は norm が step size 依存の小さい近傍まで減衰することを示す。[定理: Theorem 2] その前段の弱い命題では bias を augmented weight に含め、その norm が期待値で減ることを示すが、bias 座標の符号を分離していない。[本文・式: §3 の bias を含む予備計算]

**Figure 4 の多ニューロン実験。** 1 隠れ層 ReLU、入力次元 `d=30`、隠れ幅 `4d=120`、cross-entropy、minibatch size 1、step size `h=1/d²=1/900`、weight initialization `U[-√3,√3]` を用いる。[本文: §3、Figure 4 周辺] 入力分布は標準 Gaussian、ラベルは一様二値という pure-noise 条件である。[本文: §3、Figure 4 caption] §3 の定義は各 step で `(x,y)` を独立に選ぶと書く一方、Figure 4 周辺の公開 PDF 本文・caption は input を毎回再標本化したか固定データを反復したかを明記しない。[本文・不在検索: §3、Figure 4 本文・caption] arXiv TeX source には、Theorem 2 は fresh iid だが当該箇所以降の実験は固定 dataset へ切り替わり説明が曖昧、という非表示の reviewer note が残る。[TeX 注釈: `arxiv.tex`, Figure 4 直前] したがって Figure 4 の fresh/fixed は公開本文だけでは未確定として扱う。[本文・TeX 注釈]

この実験は分布・タスク切替のない stationary setting であり、非定常・継続学習実験ではない。[本文: §3 の sampling 設定] 論文冒頭の一般機構は、訓練ラベルを確率 `p` で一様ランダムラベルに置換し、誤分類が起きる更新を implicit regularization として読むものである。[本文: Introduction / §2] Figure 4 はその極端な `p=1` に相当する pure-noise 条件である。[本文: §3]

### 1.2 「ReLU death」と bias 有無の分岐

著者は、bias ありで全隠れユニットがほぼ全入力に対して発火しなくなる減衰を ReLU death と呼ぶ。[本文: §3、Figure 4] 引用候補は “We call this kind of decay ‘ReLU death’.” である。[本文: §3、Figure 4 直後]

標準 Gaussian は `x` と `-x` が対称なので、bias なしでは非零な線形形式が両方に負となることはなく、重みが厳密に零にならない限りユニットは全入力で死なない、という説明である。[本文: §3、Figure 3 周辺] bias なしの曲線は重み norm が零方向へ減衰する一方、bias ありでは first-layer norm が零まで行かず、負 bias によって active neuron count が 0 へ落ちる。[本文: §3、Figure 4 caption][図の目視読み: Figure 4]

Figure 4 の概数は、隠れ bias の代表曲線が初期の 0 近傍から `2×10^8` updates までに約 `-2.6`、active count が約 60 から 0、first-layer norm が約 60 から約 30 の plateau へ移る。[図の目視読み: Figure 4a–c] 曲線画像からの概数であり、表又は raw data の値ではない。[図の目視読み: Figure 4]

bias がなくても死が絶対に起きないという一般主張ではない。[本文: §4] 非負離散入力を使う 1 隠れ層、`2k` ReLU、bias なし、出力重み固定、hidden weights のみ学習、hinge loss `β=0` の構成では、`h≥1` なら全 hidden activation が零になる一方、`h≤1/k` なら各 support point に非零 activation が残る。[定理: Theorem 3 / Theorem 4] したがって Figure 4 の bias 有無の分岐は Gaussian 対称性と当該実験条件に位置づく。[本文: §3–§4]

### 1.3 bias 単調減少の格

“the bias term of a neuron is monotonically decreasing” という記述は本文にあり、Figure 4c を参照している。[本文: §3、Figure 4c 参照文] 同段落は初期化時には backpropagation formula から分かると説明するが、任意時刻の signed bias monotonicity を定理として掲げてはいない。[本文: §3][不在検索: Theorem / Lemma / Corollary と bias の全出現]

よって格は、(i) 初期化近傍の符号に関する本文説明、(ii) Figure 4c の経験曲線であり、大域単調性の定理ではない。[本文: §3][図の目視読み: Figure 4c] Theorem 1 / Corollary 1 の norm 単調性を hidden bias 座標の単調減少へ読み替えることはできない。[定理: Theorem 1 / Corollary 1 の主張範囲]

引用候補 “significant negative drift in the bias weights” は、この経験的観察の記述である。[本文: §3、Figure 4 直前]

### 1.4 b 勾配と本 repo の「税」の比較

1 ユニットを `z_i=w_i^T x+b_i`, `a_i=ReLU(z_i)`、出力を `N=Σ_i v_i a_i`、損失を論文どおり `L(-yN)` と書くと、minibatch 1 の更新は

```text
Δb_i = h y L'(-yN) v_i 1[z_i>0].
```

[代数照合: §2 の loss と SGD update から chain rule]

pure noise の `y∈{-1,+1}` で条件付き平均を取れば、

```text
E_y[Δb_i | x,θ]
  = (h v_i 1[z_i>0]/2) {L'(-N)-L'(N)}.
```

[代数照合: 上式の二値平均]

logistic loss を `N=0` の近傍で展開し `L''(0)=1/4` を用いると、

```text
E_y[Δb_i | x,θ] ≈ -(h/4) v_i N 1[z_i>0]
 = -(h/4) v_i² a_i 1[z_i>0]
   -(h/4) v_i R_i 1[z_i>0],
```

ただし `R_i=Σ_{j≠i}v_j a_j` である。[代数照合: Taylor 展開と `N=v_i a_i+R_i`] 第一項は repo 側の `-2η v_i² E[a_i gate_i]` と `v_i² a_i gate_i` の因子・負符号が同じだが、係数は異なり、論文の loss、random-label average、局所展開に依存し、cross term も残る。[代数照合: 2 式の項別比較] この b-drift 式自体は Paper 1 に明記された定理又は式ではない。[不在検索: PDF / TeX の bias-gradient 記述と式]

Paper 1 が直接示す Figure 4 の負 drift は pure label noise 条件である。[本文: §3、Figure 4] 同一設定で `p=0` にした bias trajectory の対照曲線は提示されていないため、Figure 4 だけから「ノイズを切ると drift が消える」とは確定できない。[不在検索: Figure 4 周辺、全 figure caption、`p=0` / `bias`]

### 1.5 治療、bias weight decay、復元力

Paper 1 の中心は誤分類更新による implicit norm regularization の解析であり、ReLU death を戻す介入実験は提示しない。[本文: Introduction / Conclusion][不在検索: treatment, recover, restore, reset, freeze, weight decay, regularization と Figure/Table] explicit L2 は implicit regularization との文脈比較に現れるが、bias-only weight decay、bias への復元力、bias freeze/reset の提案は見つからなかった。[不在検索: PDF / TeX 全文の `bias`, `weight decay`, `L2`, `restore`, `recovery`, `reset`, `freeze`]

### 1.6 継続学習・plasticity 文献との接続

論文は label noise、implicit regularization、ReLU sparsity/death を扱い、タスク列又は continual-learning benchmark は扱わない。[本文: 全節構成・実験設定] `continual`, `lifelong`, `plasticity`, `catastrophic` の本文出現と、Paper 2 又は arXiv:2402.18762 への参照は見つからなかった。[不在検索: PDF / TeX 全文、references] Paper 2 の references に Paper 1 の題名、arXiv ID、主要著者名の一致も見つからなかった。[不在検索: Paper 2 PDF / TeX references]

### 1.7 引用可能な一行候補

1. “significant negative drift in the bias weights” — §3、Figure 4 直前。[本文]
2. “We call this kind of decay ‘ReLU death’.” — §3、Figure 4 直後。[本文]
3. “the bias term of a neuron is monotonically decreasing” — §3、Figure 4c 参照文。[本文]

3 行はいずれも定理文ではなく、§3 の本文による経験的挙動の記述である。[本文: §3][定理不在検索: 全 theorem environment]

---

## 2. Disentangling the Causes of Plasticity Loss

書誌: [Lyle et al., arXiv:2402.18762v1](https://arxiv.org/abs/2402.18762)、[PMLR 274 proceedings page](https://proceedings.mlr.press/v274/lyle25a.html)。

### 2.1 preactivation shift の測定法

preactivation は elementwise nonlinearity に入る直前の値として定義される。[本文: §3.2] Figure 2 中央は、タスク切替後の複数時点について、各 unit の training set 上の preactivation mean を計算し、その unit 間 histogram を描く。[本文: Figure 2 caption / §3.2] 同図は第一層入力に対する gradient alignment の histogram も示す。[本文: Figure 2 caption]

本文の主な統計量は unit ごとの sample-axis mean と variance、及びその分布である。[本文: §3.2、Appendix F] Wasserstein distance、KL divergence 等の単一の distribution-distance scalar を主指標として定義した箇所は見つからなかった。[不在検索: PDF / TeX 全文の distance/divergence/shift metric と §3.2]

BatchNorm は minibatch の sample 軸で各 unit を mean 0・variance 1 にし、LayerNorm は各 sample の unit 軸で mean 0・variance 1 にする、という区別で議論される。[本文: §3.2、Appendix F]

### 2.2 bias パラメータへの帰属・分解

隠れ preactivation shift を `b` と `Wx` 又は input mean/covariance に分解した解析、hidden-bias-only ablation、hidden-bias の 2×2 attribution は見つからなかった。[不在検索: arXiv PDF / TeX、PMLR PDF / landing page を `bias`, `biases`, `biased`, `affine`, `preactivation`, `mean`, `covariance`, `ablation` で検索し、全出現周辺を確認]

見つかった bias の実質的な用例は、(i) affine layer が bias vector を加えるという一般記述、(ii) parameters は通常 weights and biases を含むという一般記述、(iii) linearized-network の理論導出では biases を置かないという仮定、(iv) 出力 target offset を fixed bias と呼ぶ箇所、(v) final-layer bias norm と target offset の比較である。[本文: §3 / Appendix の linearized-network derivation / output-scale experiments]

arXiv Figure 15／PMLR Figure 17 は、final-layer bias が target mean を符号化できるにもかかわらず、その bias norm は target offset と単調対応せず、penultimate feature の上位 singular subspace が effective bias term として働くと報告する。[本文: 該当 figure caption と周辺本文] これは output side の表現解析であり、hidden preactivation shift の parameter attribution ではない。[本文: 同図の対象 layer / metric]

PMLR landing page と両 PDF に公式 code link はなく、題名、arXiv ID、著者名、図ファイル名による GitHub 検索でも公式 repository を同定できなかった。[不在検索: PMLR landing page、PDF、GitHub/web 検索] したがってこの項目には **[コード] の確証なし**である。[不在検索]

### 2.3 per-unit mean / per-sample mean の一節

Appendix F は、他条件が同じなら “preserves the per-unit mean over samples is less damaging” と述べ、sample 軸で unit mean を保つ shift の方が、unit 軸で sample mean を保つ shift より害が小さいとする。[本文: Appendix F, “A discussion on independence between mechanisms”] 続く説明は、unit を完全に gradient flow 不能にすることが、各 sample 内の unit 間信号を少し変えることより深刻だ、という機構説明である。[本文: Appendix F]

本文はこの点を Figure 21 に参照するが、その番号は両公開版の実図と一致しない。[本文: Appendix F の参照文][図の目視読み: 両 PDF の Figure 21] arXiv Figure 21 は contextual bandit の長期曲線、PMLR Figure 21 は Seaquest/C51 の L2 曲線である。[本文: 各 Figure 21 caption][図の目視読み: 各 Figure 21]

記述と caption が対応する normalization-axis ablation は **arXiv Figure 25／PMLR Figure 27** である。[本文: Figure 25/27 caption、Appendix F][図の目視読み: figure panels] 設定は CIFAR-10 random-label continual classification、CNN/MLP/ResNet、Adam、task 間隔 50,000 steps、全 100 iterations、L2 coefficient `10^-5` である。[本文: Figure 25/27 caption と Appendix experimental details] mean subtraction と standard-deviation division を LN/BN 間で入れ替え、BN-style mean subtraction と LN-style standard-deviation normalization の hybrid が標準 LN/BN より頑健だったと報告する。[本文: Appendix F][図の目視読み: Figure 25/27]

### 2.4 weight decay の掛け先と λ

付録は L2/weight decay を parameter vector の squared norm への penalty と定義し、別の一般記述では parameters は通常 weights and biases から成るとする。[本文: Appendix の regularization 説明] ただし optimizer parameter group、bias 除外、normalization affine 除外を明示する記述は見つからず、公式コードも同定できないため、実装上 bias と LN affine に同じ λ が掛かったかは未確定である。[不在検索: PDF / TeX の `weight decay`, `L2`, `bias`, `exclude`, `parameter group`; 公式コード探索]

supervised continual experiments の固定値は `10^-5`、sweep は no L2、`10^-6`、`10^-5` である。[本文: experimental details][図の目視読み: L2 sweep figure legend] 長期 contextual bandit は 0 又は `10^-6`、Atari C51 の subset figure は `10^-7`, `10^-5`, `10^-3` を比較する。[本文: RL experimental details][図の目視読み: corresponding legends] Atari では `10^-7` が norm growth を少し遅らせる一方で interference があり、main evaluation は LN のみを使うと説明される。[本文: Atari discussion]

### 2.5 LayerNorm の位置、affine、WD

supervised MLP/CNN/ResNet では normalization を各 nonlinearity の前、すなわち preactivation に置く。[本文: architecture / experimental details] C51/Rainbow も preactivation normalization として記述される。[本文: RL methods] SAC の actor/critic encoder は 3 hidden layers で、第一層と第三層の出力の activation 前に LN を置く。[本文: DMC/SAC experimental details] LN と BN を併用する条件では LN を BN より前に置く。[本文: normalization ablation details]

LN の learnable affine `γ,β` を有効化したか、無効化したか、及びそれらを WD 対象にしたかを確定する記述は見つからなかった。[不在検索: PDF / TeX の `affine`, `gamma`, `beta`, `scale`, `shift`, `elementwise_affine`; 公式コード探索]

### 2.6 “Swiss cheese” と介入組合せ

論文が区別する介入群は、preactivation distribution shift への LN/BN/ReDO/activation、parameter norm growth への L2/WD/constraint、target magnitude への two-hot/distributional targets 又は label smoothing である。[本文: §3–§4、Figure 4/5]

Figure 4 は各介入の単独効果、Figure 5 は非定常性と target scale の双方を含む条件で LN+L2 と two-hot/label smoothing を組み合わせる。[本文: Figure 4/5 caption と周辺本文] “additive benefit” はこの介入 package の利得を表す本文語である。[本文: §4] ただし A、B、A+B の factorial interaction contrast 又は `effect(A+B)-effect(A)-effect(B)` は定義・報告されていない。[不在検索: §4、Figure 4/5、Appendix の `additive`, `interaction`, `factorial`, `subadditive`]

Appendix F の independent mechanisms は、片方を緩和しても他方により plasticity loss が残る learning problem が存在する、という存在量化の定義である。[本文: Appendix F] 統計的独立又は無交互作用の意味ではない。[本文: 同定義の形式]

### 2.7 死／線形化 unit の指標と代表値

dead unit は incoming gradient を伝播できない unit と定義され、ReLU では全入力が負側、tanh では constant regime にある場合が例示される。[本文: §3.2] linearized 又は zombie unit は activation が線形又はほぼ線形に振る舞う unit で、ReLU では全入力が正側、smooth activation では preactivation variance が小さい場合が例示される。[本文: §3.2] 引用候補 “a ReLU unit with only positive inputs will behave like the identity” はこの定義の説明である。[本文: §3.2]

計数に使う閾値又は tolerance の数式定義は PDF / TeX に見つからなかった。[不在検索: §3.2、Figure 2 caption、Appendix methods の `dead`, `linear`, `threshold`, `tolerance`] したがって repo の `p_hat_thin_frac` / `p_hat_sat_frac` と数値を直接同一視できる定義根拠は本文にはない。[本文・不在検索: 指標定義の照合]

Figure 2 右の最終層幅は 256 である。[本文: Figure 2 caption / architecture details] 曲線の代表概数は、GeLU・LN なしで per-example label reset が約 250/256、class permutation が約 170/256 へ達し、LN ありでは多くの区間でほぼ 0、task boundary spike が約 30–40 である。[図の目視読み: Figure 2 right] LeakyReLU・LN なしは boundary 付近約 200–250、終盤約 80–120、LN ありは概ね約 40 対約 10 の plateau である。[図の目視読み: Figure 2 right] いずれも raw data ではなく図からの概数である。[図の目視読み]

Figure 2 左の dead trace は LN なしが軸値約 0.2 から約 0.7、LN ありが約 0 から約 0.9 へ上がった後およそ 400 steps で 0 へ戻る。[図の目視読み: Figure 2 left] 混合図の軸表記から count/fraction の換算は確定せず、ここでは軸値としてのみ記録する。[図の目視読み: Figure 2 left]

eNTK は batch 64 の各 example の parameter gradient 内積を取り、non-diagonal component の low-rank 化を診断する。[本文: §3.2 / eNTK experimental details] これは ReLU positive-side linearization と関連づけられるが、bias 上方 drift への帰属は行われない。[本文: §3.2][不在検索: `bias` 周辺と eNTK 節]

### 2.8 ベンチマーク一覧と condA/SCR との機械的照合

- stationary large-offset regression: MNIST/CIFAR image を入力に random target network 又は high-frequency transform を使い、constant target offset を加えた pretrain/fine-tune。[本文: target-scale experiments]
- sequential random-label CIFAR-10: task ごとに labels の 1%–100% を reset して nonstationarity を連続的に変える。[本文: supervised setup / nonstationarity sweep]
- composite supervised tasks: input transformation、label reset、target growth を組み合わせる。[本文: Figure 5 / Appendix]
- contextual bandit: CIFAR-10 image を使う DQN、target networks と replay buffer を含む。[本文: contextual-bandit setup]
- Atari: ALE 57 games の C51 / Rainbow。[本文: Atari setup]
- continuous control: DeepMind Control Suite 28 domains の SAC。[本文: DMC setup]
- real-world shift: iWildCam の location shift 20 phases、各 location 2,000 steps、5 seeds。[本文: iWildCam setup]

repo の condA/SCR は `m=20, f=15` bits、15 bits は 10,000 steps ごとに 1 bit だけ flip、残る 5 bits は各 step 独立 Bernoulli、teacher は固定 LTU、online batch 1 の plain SGD である。[コード: `configs/channel_2x2_0901.yaml`; `src/envs.py` の `LTUTarget` / `SCREnv`; `src/train.py`] Paper 2 の partial label reset は変化量を滑らかにする点だけが近く、変える対象は labels/objective である。[本文: Paper 2 nonstationarity sweep] condA/SCR は input support を 1 bit ずつ変え、teacher を固定するため、両者の intervention axis は異なる。[コード: `src/envs.py`][本文: Paper 2 setup]

### 2.9 引用可能な一行候補

1. “a ReLU unit with only positive inputs will behave like the identity” — §3.2。[本文]
2. “preserves the per-unit mean over samples is less damaging” — Appendix F。[本文]
3. “additive benefit” — §4、Figure 5 周辺。[本文]

---

## 3. 対応表

セルの値は指定の 3 値に限定し、その後ろに出典位置又は不在検索範囲を付した。

| 本 repo の主張要素 | arXiv:2111.02154 | arXiv:2402.18762 / PMLR 274 | 該当なし |
|---|---|---|---|
| **b 下方ドリフト → 死** | **先行あり** — pure label noise 下の負 bias drift と ReLU death。ただし signed monotonicity は定理でなく本文＋Figure 4c。[本文・図の目視読み: §3, Figure 4] | **現象論のみ** — negative-side preactivation shift と dead units は扱うが、hidden `b` へ帰属しない。[本文: §3.2][不在検索: bias 全出現] | — |
| **b 上方暴走 → 線形化・ランク崩壊** | **無し** — positive-side linearization / rank collapse / upward bias drift は見つからない。[不在検索: PDF / TeX 全文] | **現象論のみ** — positive-side ReLU linearization と low-rank eNTK を扱うが、hidden `b` の上方 drift へ帰属しない。[本文: §3.2][不在検索: bias と eNTK 節] | **hidden b に帰属した連鎖は両論文で見つからない**。[不在検索: 上記範囲] |
| **パラメータ帰属の 2×2** | **無し** — bias 有無の比較はあるが `b` 対 `Wx/input statistics` の 2×2 分解ではない。[本文: Figure 4][不在検索] | **無し** — normalization axis ablation はあるが hidden parameter-channel attribution ではない。[本文: Appendix F, arXiv Figure 25 / PMLR Figure 27][不在検索] | **両論文で見つからない**。[不在検索: bias / ablation / mean / covariance] |
| **b 限定 weight decay** | **無し** — bias-only WD の提案・実験なし。[不在検索: 全文] | **無し** — L2 は扱うが parameter groups と bias-only 条件なし。[本文: Appendix][不在検索: WD/bias、公式コード不在] | **両論文で見つからない**。[不在検索: 上記範囲] |
| **復元力による治療** | **無し** — bias restore/reset/freeze の介入なし。[不在検索: 全文] | **無し** — LN+L2 等は扱うが bias restoring force はなし。[本文: §4][不在検索: bias/restore/reset/freeze] | **両論文で見つからない**。[不在検索: 上記範囲] |

---

## 4. 一行単位の境界（事実のみ）

1. Paper 1 は stationary pure-label-noise 条件で hidden bias の負 drift と ReLU death を Figure 4 に報告する。[本文・図の目視読み: §3, Figure 4]
2. Paper 1 の signed bias monotonicity は theorem ではなく本文説明と経験曲線である。[本文: §3][不在検索: theorem environments]
3. Paper 1 の式から random-label average と logistic loss の局所展開を行うと repo の bias 税と同じ `v_i² a_i gate_i` 因子が現れるが、論文明記式ではなく、cross term と条件依存係数が残る。[代数照合: §1.4]
4. Paper 1 は同一 Figure 4 設定の noise-off bias trajectory を示さない。[不在検索: figure captions / `p=0` / bias]
5. Paper 2 は preactivation shift の負側を death、正側を linearization と関連づけ、eNTK の low-rank 化も測る。[本文: §3.2]
6. Paper 2 は hidden preactivation shift を hidden bias と input/weight channel に帰属・分解しない。[不在検索: §2.2 記載範囲]
7. Paper 2 の per-unit/per-sample mean 比較に対応する実図は本文参照 Figure 21 ではなく、arXiv Figure 25／PMLR Figure 27 である。[本文・図の目視読み: Appendix F と各 figure captions]
8. Paper 2 の “additive” は介入 package の本文表現であり、factorial interaction contrast は報告しない。[本文: §4, Figure 5][不在検索: interaction/factorial/subadditive]
9. Paper 2 の WD が bias と LN affine を含むかは、公開本文と同定できた公式資材からは確定できない。[不在検索: PDF / TeX / PMLR page / 公式コード探索]

## 5. 不在検索ログと参照資材

**Paper 1.** arXiv v1 PDF と TeX source を対象に、`bias`, `monotonic`, `weight decay`, `L2`, `restore`, `recovery`, `reset`, `freeze`, `continual`, `lifelong`, `plasticity`, `catastrophic`, Paper 2 の題名・ID・著者名を検索し、該当節、全 theorem/corollary、全 figure caption、references を確認した。[不在検索]

**Paper 2.** arXiv v1 PDF/TeX、PMLR PDF/landing page を対象に、`bias`, `biases`, `biased`, `affine`, `gamma`, `beta`, `weight decay`, `L2`, `exclude`, `parameter group`, `preactivation`, `mean`, `covariance`, `dead`, `linear`, `threshold`, `additive`, `interaction`, `factorial`, `subadditive` を検索し、全出現周辺、Figure 2/4/5/15/21/25（arXiv）、Figure 17/21/27（PMLR）、Appendix F を確認した。[不在検索]

**コード探索。** PMLR landing page、両 PDF、題名・arXiv ID・著者名・図ファイル名による web/GitHub 検索から公式 repository を同定できなかったため、Paper 2 の optimizer parameter groups と LayerNorm construction には [コード] 格を付けていない。[不在検索]

本メモでは実験を実行せず、既存 result files を参照・変更していない。
