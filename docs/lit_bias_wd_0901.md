# lit_bias_wd_0901: bias 専用 weight decay の先行確認（段階 B・走なし）

作成: 2026-09-01 / 親: `HANDOFF bias 専用 weight decay` §3 / 状態: **記録のみ・判定なし**

本書は文献と公式実装の**事実の記録**である。実験の解釈・判定はここでは行わない。
各行に **格**（本文の数値 / 表の数値 / 図の目視読み / コードの読み取り）を明記する。

---

## 0. 出典の同定（先に確認したこと）

| # | 事実 | 格 |
|---|---|---|
| 0.1 | 依頼書は「Dohare et al., *Loss of Plasticity in Deep Continual Learning*, Nature 2024（arXiv:2306.13812v2）」と同定しているが、**arXiv:2306.13812 の v2 は Nature 版ではない**。v2 は 2023-08-18 提出・PDF 表題 `Loss of Plasticity in Deep Continual Learning`・脚注 "Preprint submitted to Neural Networks"・全 44 ページ | PDF メタデータと本文の読み取り |
| 0.2 | 同 arXiv には **v3** が存在し、表題が `Maintaining Plasticity in Deep Continual Learning` に変わっている。v3 も Neural Networks 体裁のプレプリントで、本件に関わる本文・付録の内容は v2 と実質同一（`pdftotext` 全文 1736 行 vs v2 1739 行、L2 / Appendix B の該当行はすべて一致） | PDF の読み取り |
| 0.3 | したがって **Nature 632, 768–774 (2024) の刊行版そのものは arXiv 上にない**。以下の記述はすべて **arXiv v2**（必要箇所は v3 でも同一であることを確認済み）に基づく。Nature 版の Extended Data / Supplementary の図番号は本書の番号と対応しない可能性がある | 上記からの帰結 |
| 0.4 | 公式実装は `github.com/shibhansh/loss-of-plasticity`（本文脚注 1 が指すリポジトリ。本 repo の `src/nets.py` docstring が「[D] 公式実装 (bp.py)」と呼んでいるもの） | 本文の脚注 + コードの読み取り |

> **注記**: 0.1–0.3 は依頼書 §3 の指定（v2 = Nature 2024）からの逸脱ではなく、指定された v2 をそのまま読んだ結果の同定である。追加で v3 も確認した。

---

## 1. L2 正則化 / weight decay の腕は存在するか。どのパラメータに掛けているか

### 1.1 腕の存在（本文）

| # | 事実 | 格 |
|---|---|---|
| 1.1a | 存在する。abstract に「loss of plasticity ... was substantially eased by $L^2$-regularization, particularly when combined with weight perturbation」 | 本文 |
| 1.1b | 比較対象として **Online Permuted MNIST**（3 隠れ層 × 2000 ユニット・800 タスク）と **Continual ImageNet**（5000 タスク）で L2 腕を走らせている | 本文 |
| 1.1c | 連続 RL（Slippery Ant, Appendix C/E）にも `PPO+L2` 腕がある | 本文 |
| 1.1d | **Slowly-Changing Regression（SCR = 本 repo の condA に対応する問題）については、論文中に L2 腕の図・表・数値が一切ない。** 付録 B は活性化関数の比較のみ | 本文（不在の確認） |

### 1.2 どのパラメータに掛けているか（コード）

**これが本件で最も重要な行。**

| # | 事実 | 格 |
|---|---|---|
| 1.2a | `lop/algos/bp.py` の `Backprop.__init__`: `self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)`。Adam / AdamW 分岐も同じく `self.net.parameters()` に `weight_decay` を渡す | コードの読み取り |
| 1.2b | `lop/algos/cbp.py`（Continual Backprop）: `optim.SGD(self.net.parameters(), ..., weight_decay=weight_decay)` / `AdamGnT(self.net.parameters(), ..., weight_decay=weight_decay)` | コードの読み取り |
| 1.2c | `lop/algos/convCBP.py`: 同上（`self.net.parameters()`） | コードの読み取り |
| 1.2d | `lop/algos/rl/ppo.py`: `Opt(list(self.pol.parameters()) + list(self.vf.parameters()), lr=lr, weight_decay=wd, ...)` | コードの読み取り |
| 1.2e | **公式実装のどこにも `param_groups` を分ける記述はなく、bias を除外する記述もない。** `torch.optim.SGD` の `weight_decay` は param group 内の**全パラメータ**の勾配に $\lambda p$ を加えるので、**weight と bias の両方が同一の $\lambda$ で減衰する** | コードの読み取り + PyTorch の仕様 |
| 1.2f | SCR の学習器 `lop/nets/ffnn.py::FFNN` は `nn.Linear(input_size, num_features)` と `nn.Linear(num_features, num_outputs)` の既定（`bias=True`）を使い、`layers[0].bias.data.fill_(0.0)` / `layers[-1].bias.data.fill_(0.0)` で初期化する。**隠れ層 bias $b$ と出力 bias $c$ は明示パラメータとして存在する**（本 repo の `kaiming_mlp_params` と同形） | コードの読み取り |
| 1.2g | 本文側は一貫して「a penalty ... proportional to the $\ell_2$-norm of **the weights of the network**」としか書かず、weight と bias を区別する記述はない | 本文 |

**結論（事実として）: 先行に存在するのは「全パラメータ L2」だけであり、「bias を除外した L2」も「bias だけの L2」も、論文にもコードにも無い。**

---

## 2. $\lambda$ の値と探索範囲

| # | 問題 | $\lambda$（weight decay）の探索範囲 | 選ばれた値 | 格 |
|---|---|---|---|---|
| 2.1 | Continual ImageNet（Table A.2） | {3e-5, 1e-5, **3e-6**, 1e-6}（step size は {0.1, **0.03**, 0.01, 0.003}） | **3e-6**（表で太字＝best-performing） | 表の数値（太字判定はページ画像の目視） |
| 2.2 | Online Permuted MNIST（Figure A.8 左上パネル） | {1e-3, **1e-4**, 1e-5} の 3 点を提示（step size は全手法 0.003 固定） | **1e-4**（ラベル横の■＝本文で使用した設定） | 図の目視読み |
| 2.3 | Slippery Ant / PPO+L2（Appendix E） | {1e-3, 1e-4, 1e-5, 1e-6} | 本文は「chose the best weight decay」とのみ記載、値は未記載 | 本文 |
| 2.4 | **SCR（= condA と同型の問題）**: `lop/slowly_changing_regression/cfg/sgd/bp/relu_wd.json` | **{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0}** | 論文に結果の記載なし | コードの読み取り |
| 2.5 | 同 `elu_wd.json` | {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} | 同上 | コードの読み取り |
| 2.6 | 2.4 / 2.5 の走の設定 | `num_inputs=20`, `num_flipping_bits=15`, `num_target_features=100`, `flip_after=10000`, `beta=0.7`, `opt=sgd`, `step_size=0.01`, `num_features=5`（学習器幅）, `num_data_points=1e6`, `num_runs=100` | — | コードの読み取り |

> **2.4–2.6 の含意（事実の並置のみ）**: 公式実装は、本 repo の condA と同一の問題パラメータ（m=20・f=15・教師幅100・T=10,000・β=0.7・SGD・lr=0.01）に対して weight decay を **1e-6 から 1e-0 まで 1 桁刻み**で掃く config を持っている。本件パイロット（依頼書 §4）のグリッド {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} はこの範囲の内側に完全に収まる。ただし公式側は**学習器幅 5・1M サンプル・全パラメータ減衰**であり、本件は**幅 100・5M・bias のみ減衰**なので、同一条件ではない。

---

## 3. L2 腕の LoP 緩和の効き

### 3.1 本文の言明

| # | 事実 | 格 |
|---|---|---|
| 3.1a | 「$L^2$-regularization does not fully mitigate the loss of plasticity」 | 本文 |
| 3.1b | 「While $L^2$-regularization reduces the average weight magnitude of the network, **it increases the percentage of dead units and decreases the effective rank**」 | 本文 |
| 3.1c | 「$L^2$-regularization on its own does not mitigate the loss of plasticity」（shrink-and-perturb の 2 成分の議論の中で） | 本文 |
| 3.1d | 「both shrink-and-perturb and $L^2$-regularization are **very sensitive to hyperparameter values**. They only reduce the loss of plasticity for a very small range of parameters, while for other hyperparameter values, they make the loss of plasticity worse」 | 本文 |
| 3.1e | Figure 4 caption: 「Only $L^2$-Regularization and shrink-and-perturb have higher accuracy than backpropagation after learning 800 tasks」 | 本文（caption） |
| 3.1f | Continual ImageNet では continual backprop が L2 と S&P の両方を上回る、とのみ記述。L2 の数値は本文に無い | 本文 |

### 3.2 Figure 4 の目視読み（Online Permuted MNIST・30 runs・タスク 0–800）

**すべて「図の目視読み」。数値は読み取り誤差 ±0.1（精度）／±1〜2 ポイント（dead 率・eff rank）を含む。**

| パネル | 系列 | 序盤 | task 800 |
|---|---|---|---|
| 4a `Percent Correct on MNIST` | L2 ($\lambda$=1e-4) | ≈95.2 | ≈94.3 |
| 4a | Backpropagation | ≈95.3 | ≈94.0 |
| 4a | Shrink & Perturb | ≈95.2 | ≈95.0（ほぼ平坦） |
| 4a | Dropout | ≈95 | ≈92.4 |
| 4a | Online Norm | ≈95.6（task 50 付近が峰） | ≈93.4 |
| 4a | Adam | ≈94.5 | 50 タスク以内に軸下限 91 を割って脱落 |
| 4b 左 `Percent of Dead Units` | **L2** | ≈0 | **≈23%（Adam を除く全手法で最悪）** |
| 4b 左 | Backpropagation | ≈0 | ≈13% |
| 4b 左 | Online Norm | ≈0 | ≈15% |
| 4b 左 | Dropout | ≈0 | ≈10% |
| 4b 左 | Shrink & Perturb | ≈0 | ≈9% |
| 4b 左 | Adam | — | ≈41% |
| 4b 中 `Weight Magnitude` | **L2** | ≈0.005 | **≈0.005（全手法で最小・ほぼ平坦）** |
| 4b 中 | Backpropagation | ≈0.025 | ≈0.035 |
| 4b 右 `Effective Rank`（[0,100] に規格化） | **L2** | ≈45 | **≈11（Adam を除き最低）** |
| 4b 右 | Backpropagation | ≈45 | ≈28 |
| 4b 右 | Dropout | — | ≈30 |
| 4b 右 | Shrink & Perturb | — | ≈22 |
| 4b 右 | Adam | — | ≈2 |

**この 3 枚の並置が本件にとって最も近い先行事実である**: 全パラメータ L2 は、重みノルムを止めて精度上の LoP を**弱めた**が、その代償として **dead 率を 13% → 23% に上げ、effective rank を 28 → 11 に下げた**。すなわち先行では **L2 は「死」と「ランク」を同時に悪化させている**。

---

## 4. 付録 B / 図 B.10 の設定（本件とは別件・`gate_dose_0830` の未解決項目）

### 4.1 Table B.3（**表の数値**・確定）

| 区分 | パラメータ | 値 |
|---|---|---|
| 問題 | m（入力ビット数） | **21**（うち最後の 1 ビットは定数 1 のバイアスビット） |
| 問題 | f（flipping bits） | 15 |
| 問題 | n（**教師**の隠れユニット数） | 100（LTU） |
| 問題 | T（bit flip 間隔） | 10,000 time steps |
| 問題 | Bias（入力層・出力層にバイアス項を含むか） | True |
| 問題 | $\theta_i$（LTU 閾値） | $(m+1)\beta - S_i$ |
| 問題 | $\beta$ | 0.7 |
| **学習器** | 隠れ層数 | **1** |
| **学習器** | 各隠れ層のユニット数 | **5** |

> 依頼書の「学習器の幅（5 ユニット？）」は **5 で確定**。本文にも「the learner has just five hidden units」とある（本文）。
> 実装側の SCR config は `num_inputs: 20` + `nn.Linear` の明示 bias であり、Table B.3 の m=21（定数ビット込み）と等価な書き方（コードの読み取り）。本 repo の condA（m=20 + 明示 $b$）とも等価。

### 4.2 走の条件（**本文**・確定）

| 項目 | 値 |
|---|---|
| ホライズン | **3M examples** |
| 反復数 | 活性化 × step size の各組み合わせで **100 independent runs** |
| 乱数の統制 | 100 本の例列を先に生成し、全活性化・全 step size で**同一の例列**を使う |
| 初期化 | uniform Kaiming、$b = \text{gain}\sqrt{3/\text{num\_inputs}}$。gain は tanh 5/3・sigmoid 1・ReLU $\sqrt2$・leaky-ReLU $\sqrt{2/(1+\alpha^2)}$・ELU/Swish $\sqrt2$ |
| step size | 0.01 / 0.003 / 0.001 の 3 水準 + Linear baseline |

### 4.3 図の構成と縦軸（**図の目視読み**）

| 項目 | 内容 |
|---|---|
| 図の構成 | **2 行 × 3 列の 6 パネル**（上段 Tanh / Sigmoid / ELU、下段 ReLU / Leaky-ReLU / Swish）。**a/b/c のサブパネル分けは実在しない** |
| 縦軸（共有ラベル） | **`Squared Error (Bins of 40k)`** — 真の目標と予測の差の二乗を 40k サンプルごとに平均したもの。網掛けは binned error の標準誤差 |
| 横軸 | `Example Number`、0 → 3M |
| 縦軸目盛 | 上段 3 パネル 0.2–1.0、下段 3 パネル 0.4–1.2 |
| Linear baseline | 全パネルで ≈0.8 の平坦線 |

> 本文は「In Figure B.10c, the squared error is presented in bins of 40k examples」と書くが、**図に c パネルは無い**。図全体が binned squared error である（本文とキャプションの不整合）。

### 4.4 各活性化の終端値（**図の目視読み**・読み取り誤差 ±0.05）

| 活性化 | step 0.01 | step 0.003 | step 0.001 | Linear ≈0.8 との関係（3M 時点） |
|---|---|---|---|---|
| ReLU | 0.65 → **>1.2（軸外）** | 0.6 → ≈1.0–1.1 | 0.6 → ≈0.85 | **明確に上回る（重症）** |
| Leaky-ReLU | 0.55 → ≈1.0 | 0.6 → ≈1.1 | 0.55 → ≈0.8 | **明確に上回る（ReLU と同程度に重症）** |
| Tanh | 0.45 → ≈0.85 | 0.55 → ≈0.85 | 0.6 → ≈0.85 | わずかに上回る |
| Sigmoid | 0.4 → ≈0.9 | 0.6 → ≈0.85 | 0.7 → ≈0.85 | 上回る |
| ELU | 0.45 → ≈0.75–0.9（振動） | 0.65 → ≈0.7 | 0.55 → ≈0.7 | **同程度〜やや下回る（軽症だが上昇はする）** |
| Swish | 0.6 → ≈1.05 | 0.55 → ≈0.75 | 0.6 → ≈0.75 | step 0.01 のみ上回る |

本文の要約（本文）: 「for some activations like ReLU and tanh, loss of plasticity is severe, and the error increases to the level of the linear baseline. While for other activations like ELU, loss of plasticity is less severe, but still there is a significant loss of plasticity」。

### 4.5 `gate_dose_0830` との緊張（事実の並置のみ・解釈しない）

`results/gate_dose_0830/verdict.csv` の `n_onset_5m` 列（5M・seed 0–9）:
`R_off/R_933/R_1216`（ReLU）は 10/10、`E_off/E_933/E_1216`（ELU）は 0/10、`LR_off/LR_933/LR_1216`（leaky）は 0/10。

原典 図 B.10 との差分は少なくとも次の 4 点ある（すべて上記の格つき事実から）:

1. **学習器幅**: 原典 5 / 本 repo 100
2. **ホライズン**: 原典 3M / 本 repo 5M
3. **指標**: 原典 `squared error`（40k bin・生スケール、Linear baseline との相対で読む）/ 本 repo `unfit`（= `residual_var / signal_var`）と onset 判定
4. **leaky-ReLU の位置づけ**: 原典では leaky-ReLU は**軽症ではなく ReLU 並に重症**。`gate_dose_0830` では leaky は ELU と同じく 0/10。したがって緊張は「ELU だけ」ではなく **leaky も含む 2 活性化**にある

---

## 5. 依頼書 §3 の分岐に対する事実の当てはめ

| 依頼書の分岐条件 | 本調査の事実 |
|---|---|
| 「bias を除外した L2」が先行にあるか | **無い**（§1.2e）。論文にもコードにも、param group を分ける記述自体が存在しない |
| 「全パラメータ L2」しかないか | **そのとおり**（§1.2a–d）。SGD/Adam/AdamW/PPO の 4 経路すべてが `.parameters()` 全体に単一 $\lambda$ を掛ける |

したがって依頼書の想定どおり、**本件は「$b$ だけの減衰で足りるか」の分離実験として立つ**。予定どおり段階 A へ進む。

ただし、新規性の主張を書くときに**必ず併記すべき先行事実**が 2 つある:

1. **$\lambda$ の桁は新規ではない**。公式実装は同型の SCR 問題に対し 1e-6〜1e-0 の sweep config を持っている（§2.4）。主張できるのは「どのパラメータを減衰させるか」の分離であって、「その桁を見つけたこと」ではない
2. **先行の全パラメータ L2 は dead と effective rank を悪化させている**（§3.1b・§3.2）。本件の予測（$b$ 減衰で死・飽和・ランク崩壊が同時に止まる）は、**先行の観測と符号が逆**である。したがって本件の判定 (a)（死の抑制）が通った場合、それは「L2 一般の再現」ではなく「全パラメータ L2 との**乖離**」として報告しなければならない

---

## 6. 参照した実体

- arXiv:2306.13812v2 PDF（44 ページ、`pdftotext -layout` 全文 + p.16 / p.34 / p.35 / p.37 / p.38 のページ画像）
- arXiv:2306.13812v3 PDF（表題変更の確認と、L2 / Appendix B 該当行の同一性確認のみ）
- `github.com/shibhansh/loss-of-plasticity` @ `main`: `lop/algos/bp.py`, `lop/algos/cbp.py`, `lop/algos/convCBP.py`, `lop/algos/rl/ppo.py`, `lop/nets/ffnn.py`, `lop/nets/linear.py`, `lop/slowly_changing_regression/expr.py`, `lop/slowly_changing_regression/cfg/sgd/bp/{relu,relu_wd,elu_wd,5m}.json`, および `git/trees/main?recursive=1` のツリー一覧（198 エントリ）
