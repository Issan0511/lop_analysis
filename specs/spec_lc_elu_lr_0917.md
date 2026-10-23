# Lillo & Cheney の箱で ELU の α と学習率を分ける — Random Label MNIST（400 epoch × 50 タスク）の 2×2（4 腕 × 5 seed、lr 1e−4 の 2 腕は 150 タスクまで延長）

状態: **事前登録**（本ファイルの commit が登録。本走のデータはまだ一つも無い。走ったのは §5 の検査の短い走だけ）
作成: 2026-09-17 / 起草: Claude / 依頼: Issa（別セッションの文献照合「Lillo & Cheney と V10 の位置づけ」を貼付。その末尾の 2×2 案を、事実確認で直してから実行する）
親: obsidian-research `可塑性喪失/主張/中心主張v10草案_0916.md` §1.2–1.3a・`可塑性喪失/論点/先行研究/ELUとLoPの文献照合_rsl_rl_0914.md`・`results/act_chimera_0913/rlmnist/`
実装: `src/lc_elu_lr_0917.py`（箱と走行）・`analysis/lc_elu_lr_0917/{checks.py,verdict.py,launch.sh}`
branch: `claude/lc_elu_lr_0917`

## 0. 一行と問い

Lillo & Cheney（ICLR 2026・arXiv 2509.22562v4）の Random Label MNIST で ELU は生涯 online 84.23 ± 0.70 % と生き残る。ただしその設定は **α = 3.6・lr = 1e−4**（表 E2）。当方の箱（α = 1・lr = 1e−3）の ELU は崩壊する（400 epoch 箱で t7–11、80 epoch 箱で t7–10）。**彼らの箱をそのまま使い、α と lr だけを振って、生存を決めているのがどちらかを分ける。**

- **Q1（主）**: α ∈ {1, 3.6} × lr ∈ {1e−3, 1e−4}。窓 t31–50 で床に落ちるかを分けるのは lr か α か。
- **Q2（再現の錨）**: 彼らの設定（α 3.6・lr 1e−4）の生涯 online 平均は 84.23 を再現するか。
- **Q3（輸送）**: lr 1e−4 の 2 腕で、t31–50 にも第2層の輸送（‖µ₂‖ の伸びと z̄₂ の沈み）が続いているか。
- **Q4（延長）**: lr 1e−4 の 2 腕は 150 タスクまでに床に落ちるか。lr は崩壊を遅らせただけか、止めたか。

## 1. なぜ（確認済みの事実・REPORT_ONLY）

### 1.1 彼らの箱（論文 v4 の表 E1・E2・B2、コード `lute47lillo/activations_plasticity` の iclr2026 `bdce354`）

- **表 E1 とコード**: 先頭 1200 枚の MNIST 訓練画像（/255）、タスクごとに一様乱数ラベル（訓練 1200 と検証 500 を交互に引く）、784–100–100–10（`nn.Linear` の既定の初期化）、`torch.optim.Adam(lr)` の既定値を 1 回だけ作る（moment はタスクをまたぐ）、batch 16、**400 epoch = 30,000 更新/タスク**、50 タスク、WD なし。**毎エポック同じ順に並べる（並べ替えなし）**。online は更新前のバッチ精度。表 2 の値は 50 タスク全更新の平均（Total Average Online Task Accuracy）で、5 run の平均 ± SD。
- **表 E2 の Random Label MNIST 列**: ELU (3.6, 1e−4)・CeLU (3.6, 1e−4)・SeLU (3.0, 1e−4)・**ReLU (−, 1e−4)**・Tanh (−, 1e−4)・Rational (…, 1e−4)。**lr 1e−4 を選んだのは ELU だけではない。** ReLU は lr 1e−4 でも 20.03 ± 2.46。Leaky-ReLU は傾き **0.8**・lr 1e−3 で 91.53。Swish は β = 0.01（x·σ(0.01x)、|x| ≪ 100 でほぼ x/2）、GeLU は gelu(0.8x)。
- **lr のグリッド**: 本文に明記は無い。表 B2 の構成数（ReLU 2、ELU 64 = α 32 値 × 2）と表 E2 の値から {1e−3, 1e−4} と読める。ELU では 32 の α × 2 の lr のうち (3.6, 1e−4) が最良 → **lr 1e−3 のどの α も生涯平均で 84.23 を下回った**、までは言える。崩壊したかどうかは分からない。
- **ELU の微分**: 彼らは `nn.ELU(alpha)`（`F.elu`）。torch 2.13 の CPU float32 autograd でその微分は $\alpha e^z$ で、z = −100 でも 0 でない（当方の宿主の expm1 実装は $z<-16.64$ で厳密に 0）。§5 S2 で検査する。
- **当方の宿主 `pmnist_rlmnist_0906` との違い**: 毎エポックの並べ替え（当方あり）、1200 枚の選び方（当方は seed ごとの一様抽出）、ELU の微分の実装（expm1）、指標（当方はタスク別と窓）。貼られた照合の「違いは 2 つだけ」は、この 4 点を落としている。

### 1.2 当方の 400 epoch 箱（`act_chimera_0913` rlmnist・α = 1・lr 1e−3・expm1・毎エポック並べ替え・GCP AMD CPU）

| 腕 | online < 0.5 の最初 | < 0.15 が続く最初 | 凍結 | 生涯 online（t1–50） | t31–50 の $O_t-F_t$ |
|---|---|---|---|---|---|
| ELU1 s0/s1/s2 | t9 / t9 / t7 | t10 / t11 / t9 | t24 / t18 / t30 | 0.228 / 0.232 / 0.203 | −0.014 / −0.014 / −0.013 |
| SMINH（深部の微分 e^z）s0/s1/s2 | t11 / t13 / t9 | t17 / t18 / t12 | t46 / なし / t43 | 0.273 / 0.307 / 0.238 | −0.011 / −0.011 / −0.011 |
| LR（leaky 0.1） | — | — | — | 0.821–0.827 | +0.685–0.688 |

（`per_task.csv` から本 spec の起草時に計算。$F_t$ は `act_chimera_rlmnist_0913.majority_ratio`。）同じ 30,000 更新/タスクでも α = 1・lr 1e−3 の ELU は生涯 0.2 台で、彼らの 84.23 と 60 pt 違う。差の候補は α・lr・並べ替え・微分の実装。**本走は彼らの箱を再実装して α と lr だけを振る**ので、並べ替えと微分の実装は全腕で彼らのものに揃う。

### 1.3 V10 の読み（貼られた照合の主張）

「lr は輸送の速さ。lr を 1/10 にすれば W の成長と ‖µ₂‖ の伸びが 1/10 になり、50 タスクでは裾に届かない。彼らの活性化の順位表は、輸送が地平線内に裾へ届くかどうかの表として読める。」Q1 はこの読みの必要条件（lr が生存を分ける）、Q3・Q4 と時間の尺度合わせ（§4.6）は「遅くしただけで止めてはいない」の部分を見る。

## 2. 設計

### 2.1 箱（`src/lc_elu_lr_0917.py`・彼らの random_label_mnist の経路の再実装。リポジトリにライセンスが無いのでコードは持ち込まない）

- 画像: `data/mnist/train-images-idx3-ubyte.gz` の先頭 1200 枚、float32 / 255。
- 乱数: `random.seed(s)`・`np.random.seed(s)`・`torch.manual_seed(s)` の直後に、50 タスク分のラベルを「訓練 1200 → 検証 500」の順に大域の生成器から引き、その後に網を作る（bench_main の順）。**t51 以降のラベルは別の生成器**（sha256("lc_elu_lr_0917|ext_labels|s")）から同じ順に引く。したがって t1–50 は `--tasks` に依らず bit 一致（S3）。
- 網: `nn.Linear(784,100)`・`nn.Linear(100,100)`・`nn.Linear(100,10)` をこの順に作り、活性化は層ごとに別のインスタンス。
- 最適化: `torch.optim.Adam(model.parameters(), lr)`（β (0.9, 0.999)・ε 1e−8・WD なし）を 1 回だけ作る。損失は `CrossEntropyLoss()`（平均）。
- 1 タスク: 400 epoch × 75 更新。各エポックで画像を**同じ順**に 16 枚ずつ。各バッチの更新前の正解数を数える。
- CPU 1 スレッド、`torch.set_flush_denormal` は既定（False、彼らの既定）。タスク終端ごとに checkpoint を書き、中断したら同じ状態から再開する（S5）。
- 彼らの 5 run と bit 一致は目指さない（seed の決め方と装置が違う）。一致させるのは手順（S1 で彼らの `MLPBenchmarks` と彼らの学習ループを写したものに対して bit 一致を確認する）。

### 2.2 腕（各 seed 0–4）

| 腕 | 活性化 | lr | タスク数 | 役割 |
|---|---|---|---|---|
| E1_lr1e3 | ELU α=1 | 1e−3 | 50 | 2×2（当方の設定） |
| E36_lr1e3 | ELU α=3.6 | 1e−3 | 50 | 2×2 |
| E1_lr1e4 | ELU α=1 | 1e−4 | **150** | 2×2 |
| E36_lr1e4 | ELU α=3.6 | 1e−4 | **150** | 2×2（彼らの設定） |
| R_lr1e4 | ReLU | 1e−4 | 50 | 錨・報告のみ（表 2: 20.03 ± 2.46） |
| LK08_lr1e3 | Leaky 0.8 | 1e−3 | 50 | 錨・報告のみ（表 2: 91.53 ± 0.18） |

錨は 2×2 の 20 run の後に回す。時間が足りなければ落とし、落としたことを結果に書く。

### 2.3 延長

lr 1e−4 の 2 腕だけ 150 タスクまで回す（Q4）。t1–50 の行は延長しても変わらない（S3）ので、Q1–Q3 は 20 run すべてが t50 を終えた時点で一度だけ判定する。Q4 は 10 run すべてが t150 を終えてから判定する。

## 3. 測る量（`per_task.csv`・`units.npz`。各タスクの終端で、1200 枚と現在のラベルの上で）

- `online`（正解数 ÷ 480,000）・`floor`（現在のラベルの最多クラスの割合 $F_t$）・`memo_acc`・`loss_end`。`units.npz` の `online_block` は 50 epoch ごとの online（8 個/タスク）。
- 層 l = 1, 2 のユニット別（`units.npz`）とその全ユニット平均（`per_task.csv`）:
  - `zbar`（画像平均の前活性 z̄）・`sigma`・`U`（画像での最大）・`pplus`（z > 0 の割合）
  - `gtr`（**訓練が使う微分**: 同じ float32 の z に同じ活性化モジュールを autograd で通した値の画像平均）・`zero`（その微分が厳密に 0 の画像の割合）・`dead`（`gtr` が 0 のユニットの割合）
  - `neffT` = $(\sum_x g)^2/(n\sum_x g^2)$（訓練の微分 g。全画像で 0 のユニットは 0）・`neff` と `loggmean`（関数としての微分 $\log\phi'$ から対数空間で。float64 でも下溢れしない）
  - `wnorm`（行ノルム）・`b`・`wmu`（$w_i\cdot\mu_l$）
  - `mu_norm_l2` = ‖µ₂‖（µ₂ = 1200 枚での第1層出力 $a_1$ の平均）・`q_l2`（$w_{2,i}\cdot\mu_2/\|\mu_2\|$）・`cos_l2`
  - `adam_small`（その層の W の座標のうち $\sqrt{\hat v}<\varepsilon$ の割合）
- 恒等式 $\bar z_{2,i}=w_{2,i}\cdot\mu_2+b_{2,i}$ は float32 の前向きの丸めの範囲で成り立つ（S4）。

## 4. endpoint と判定（`analysis/lc_elu_lr_0917/verdict.py`）

### 4.1 床（全問共通）

$O_t$ = タスク t の online、$F_t$ = §3 の floor。窓 W の平均を $O_W, F_W$ とする。

- **seed の判定**: $O_W \le F_W + m_W$ なら FLOOR、そうでなければ ABOVE。
- **$m_W = z\,s_c/\sqrt{20} = 0.00124$**。$z=\Phi^{-1}(1-0.05/20)=2.807$（2×2 の 20 run で Bonferroni）、$s_c=0.001967$ は act_chimera_0913 の 400 epoch 箱で崩れた 6 系列（ELU1・SMINH × seed 0–2）の t31–50 における $O_t-F_t$ の標本 sd（ddof 1）の二乗平均の平方根（§1.2 の表と同じデータ）。崩れた網は $F_t$ より 0.011–0.014 下にいるので、この幅は判定をほとんど動かさない。$F_W$ を上回るのは入力を使って当てている網だけ。
- **腕の判定**: 5 seed 中 4 以上が FLOOR → `COLLAPSED`、4 以上が ABOVE → `ABOVE`、それ以外 `SPLIT`。
- 発散した run（損失が非有限）は発散したタスク以降を FLOOR として数え、結果に明記する。

### 4.2 Q1（主・窓 t31–50）

| ラベル | 条件 |
|---|---|
| `LR_DOMINATES` | E1_lr1e3・E36_lr1e3 が COLLAPSED、E1_lr1e4・E36_lr1e4 が ABOVE |
| `ALPHA_DOMINATES` | E1_lr1e3・E1_lr1e4 が COLLAPSED、E36_lr1e3・E36_lr1e4 が ABOVE |
| `JOINT_ONLY` | E36_lr1e4 だけ ABOVE、他の 3 腕が COLLAPSED |
| `ALL_COLLAPSE` | 4 腕とも COLLAPSED |
| `NONE_COLLAPSE` | 4 腕とも ABOVE |
| `OTHER(…)` | 上のどれでもない（4 腕の判定を併記） |

### 4.3 Q2（再現の錨）

$O^{\rm life}_s$ = seed s の t1–50 の $O_t$ の平均（全更新の平均と同じ。彼らの表 2 の量）。$\Delta = 100\,\overline{O^{\rm life}} - 84.23$ [pt]、$SE=\sqrt{s^2/5+0.70^2/5}$（s は E36_lr1e4 の 5 seed の標本 sd [pt]、0.70 は彼らの SD）。

- `REPRODUCED` ⇔ $|\Delta|\le 3\,SE$。それ以外は `HIGHER`（Δ > 0）/ `LOWER`（Δ < 0）。
- 錨の 2 腕（報告のみ）も同じ式で表 2 と並べる。

### 4.4 Q3（lr 1e−4 の輸送・t1–50 の中で）

seed ごとに $\Delta\mu_s=\overline{\|\mu_2\|}_{31\text{–}50}-\overline{\|\mu_2\|}_{11\text{–}30}$、$\Delta z_s=\overline{\bar z_2}_{31\text{–}50}-\overline{\bar z_2}_{11\text{–}30}$（$\bar z_2$ は `zbar_l2`＝全ユニット平均）。

- seed: $\Delta\mu_s>0$ かつ $\Delta z_s<0$ なら RUNNING。
- 腕: 4 以上 RUNNING → `RUNNING`、1 以下 → `STOPPED`、それ以外 `MIXED`。
- Q3 のラベルは E1_lr1e4 と E36_lr1e4 の腕判定の組（例 `E1=RUNNING, E36=RUNNING`）。

### 4.5 Q4（延長・窓 t131–150）

§4.1 の判定を窓 t131–150 で（$m_W$ は同じ 20 タスク窓の値）。腕: `COLLAPSED_BY_150` / `ABOVE_AT_150` / `SPLIT`。

### 4.6 報告のみ（登録判定ではない）

- **崩壊の時刻**: $T_{1/2}$ = $O_t\le(O_1+F_t)/2$ となる最初のタスク、$T_{\rm floor}$ = $O_t\le F_t+m_1$（$m_1=z\,s_c=0.0055$）となる最初のタスク。lr 1e−4 と 1e−3 の比（打ち切りは「> 地平線」）。
- **時間の尺度合わせ**: lr × 更新数を揃えた点（lr 1e−4 の t = 10k と lr 1e−3 の t = k）で、‖µ₂‖・z̄₂・`gtr_l2`・`neffT_l2` の比。「輸送の速さが lr に比例する（ドリフト）」なら比は 1 付近、「√ で効く（拡散）」なら lr 1e−4 側が小さい。
- 錨 2 腕の生涯平均と表 2。
- 各腕の第1層・第2層の `neffT`・`gtr`・`dead`・`adam_small` の推移、第2層の分解 $\bar z_2=q_2\|\mu_2\|+\bar b_2$。
- act_chimera_0913 の ELU1 との差（並べ替えと微分の実装が違う箱）。

## 5. 検査（`analysis/lc_elu_lr_0917/checks.py`・本走前に all_pass。変異は検査対象ファイルへのちょうど 1 回の文字列置換で、同じ検査を落とさなければならない）

- **S1 手順の一致**: 彼らの `bench_models.MLPBenchmarks`（取得した iclr2026 `bdce354` のファイル、sha256 を記録。`rational.torch` はスタブ）と、彼らの `bench_main.py` L184–203 の学習ループと `bench_data_handlers.py` のラベルの引き方を写した参照に対し、本 runner の `build`・`train_task` が、α 3.6・lr 1e−4 と α 1・lr 1e−3 の seed 0 で、3 タスク × 3 epoch の各タスク終端の全パラメータと online を bit 一致で再現する。変異: 並べ替え（逆順）・α を無視・Adam の ε・ラベルの引き順（検証を先）・画像の順・ラベルを引くタスク数（初期化がずれる）・online の分母。
- **S2 訓練の微分**: `train_dphi(nn.ELU(α))` が z ∈ {−1, −16, −17, −20, −50, −100} で $\alpha e^z$（float32）と 2 ulp 以内で一致し 0 でない（α = 1, 3.6）。ReLU・leaky は {0, 1}・{0.8, 1}。変異: 活性化を expm1 実装に替える。
- **S3 延長の接頭**: `build(…, 50)` と `build(…, 150)` で初期値と t1–50 のラベルが bit 一致し、`build` 後の大域の生成器の状態も一致する。変異: 延長のラベルを大域の生成器から引く。
- **S4 読み出し**: $|\bar z_{2,i}-(w_{2,i}\cdot\mu_2+b_{2,i})|\le 785\,\varepsilon_{32}\,\overline{(\sum_j|w_{ij}x_j|+|b_i|)}$（float32 の和の丸めの上界。第1層も同様）、`neff` と `neffT` が z > −40 の網で相対 $4\varepsilon_{32}$ 以内、`per_task.csv` のユニット平均が `units.npz` の平均と一致。変異: µ₂ を $a_2$ から取る・`neff_of` の n を列数にする・平均を中央値にする。
- **S5 再開と決定性**: 3 タスクを通しで 2 回、2 タスクで止めて再開して 3 タスク、の 3 通りが `sec` 以外すべて一致。変異: 再開で Adam の状態を読まない。
- **S6 判定器**: 合成の `per_task.csv` で §4 の各ラベルが出る。変異: 腕の判定の seed 数 4 → 3・窓の始まり 31 → 30・$m_W$ の符号・Q2 の 3 SE → 2 SE・Q3 の比較窓。
- **S7 provenance**: CLI の走が status COMPLETE、`threads` = 1（全 import の後に設定）、開始時の git hash と dirty、ラベルの sha256 が max(50, タスク数) 個。
- **S-cost**: E1_lr1e3 の 1 タスク（400 epoch）の時間と peak RSS。並列数は min(6, (MemAvailable − 4 GiB) / (1.2 × RSS))、4 未満なら起動しない。崩壊後の腕は非正規化数で遅くなるので 3 倍で見積もる。

## 6. 予測（本 spec の commit の時点。Q1–Q4 の判定器も同時に commit）

### Claude

- **Q1 = `LR_DOMINATES`**（確信 50%）。腕ごと: E1_lr1e3 COLLAPSED（90%）・E36_lr1e3 COLLAPSED（70%）・E1_lr1e4 ABOVE（70%）・E36_lr1e4 ABOVE（85%）。
- E36_lr1e3 の $T_{1/2}$ は E1_lr1e3 以下（60%。α は深部の微分を $\ln 3.6=1.28$ 浅く見せるだけで、飽和した第1層の出力 −α が ‖µ₂‖ を大きくする側が効く）。
- **Q2 = `REPRODUCED`**（55%）。
- **Q3 = `E1=RUNNING, E36=RUNNING`**（60%）。
- **Q4 = E1_lr1e4 `ABOVE_AT_150`（55%）・E36_lr1e4 `ABOVE_AT_150`（65%）**。W̃ の伸びが √t（perm_cycle_0909）なら同じ伸びに要る更新数は $1/\mathrm{lr}^2$ で 100 倍。崩れた seed があれば $T_{1/2}$ の比は 15 より大きい（報告のみ）。
- 錨: R_lr1e4 は生涯 0.15–0.30、LK08_lr1e3 は 0.88–0.93。

### 貼られた照合（別セッション）の予測

本文にある表から: (α 1, lr 1e−3) 崩壊、(α 1, lr 1e−4) 生存、(α 3.6, lr 1e−3) 崩壊（時刻は α で前後）、(α 3.6, lr 1e−4) 84.2 の再現 → **Q1 `LR_DOMINATES`・Q2 `REPRODUCED`**。「lr を 1/10 にすれば W の成長と ‖µ₂‖ の伸びが 1/10」からの含意（Claude が読んだもの・本人の登録ではない）: Q3 は両腕 RUNNING、崩壊の時刻は 10 倍（E1_lr1e3 の $T_{1/2}$ が 7–10 なら 70–100）で **Q4 は E1_lr1e4 が `COLLAPSED_BY_150`**。

### Issa

未記入（本走の結果を見る前に受け付けたら、時刻つきで追記する）。

## 7. 実行計画と費用

- 検査 → commit → push → `analysis/lc_elu_lr_0917/launch.sh`（checks.json が all_pass・コードと spec が commit 済み・HEAD が push 済みでなければ起動しない）。
- 待ち行列: seed 0–4 の順に 2×2 の 4 腕、その後に錨 R_lr1e4 × 5・LK08_lr1e3 × 5。1 run = 1 プロセス・1 スレッド、`nice 10`、`setsid nohup` で切り離す。
- 見積もり（検査の S-cost で更新）: 健康な網 0.6–0.7 ms/更新 → 約 20 s/タスク。E*_lr1e3 は崩壊後 3 倍で約 45 分、E*_lr1e4 は 150 タスクで約 50–70 分、錨は 15–50 分。計 20–25 CPU 時間、4–6 並列で 4–6 時間。
- 生データ（`units.npz`・checkpoint・ログ）は git に入れず、終了後に `~/Projects/obsidian-research-data/lc_elu_lr_0917/` へ移して `backup_manifest.json` を commit する。取得した Lillo & Cheney のファイルと論文 PDF も同じ場所（`lillo_cheney_src/`）に置く（リポジトリは公開なので持ち込まない）。

## 8. 読み方の注意

- `LR_DOMINATES` は「この箱で生存を分けたのは lr」まで。「lr が輸送の速さを決める」は Q3・Q4 と §4.6 の尺度合わせで読む。
- 彼らの表 2 の他の活性化を「輸送が届いたかの表」と読み替えるのは、この走では検定しない（ReLU は lr 1e−4 でも 20.03 で、輸送を要しない死に方がある）。
- 本箱の ELU の微分には −16.64 の床が無い。凍結の有無を expm1 系統（act_chimera・mucap_ee・resp_ee）と比べるときは系統が違うことを書く。
- 並べ替えなしの効果は本走では分離しない（全腕で彼らの手順）。
