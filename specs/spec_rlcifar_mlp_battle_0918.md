# rlcifar_mlp_battle_0918 — RL-CIFAR × MLP で適応 Snake・kunekune 3 種と leaky / Smooth-Leaky / ELU / SiLU / GELU を戦わせる

書いた時刻: 2026-09-18 02:20（**実装の前・本走の前**）。Claude（Opus 5）。
経緯: RL-CIFAR × MLP で SNA が 50 タスク通して上がり続けた試走（§9 に開示）を見て、Issa「leaky（傾きも数点）、Smooth-Leaky（Lillo & Cheney の提案）ELU、Swish・GELU とバトルさせるか」。
その後 Issa が新しい活性化 kunekune（Snake の 1 周期だけ残して外を直線にする）を起案し、「この 3 つ（kunekune・縦に潰した版・裾の傾き 1 の版）も入れて spec を書いて」。
親: `pmnist_rlcifar_0907`（箱の定義）、`swish_battle_0917`（同じ箱の MLP で SNA が固定 Snake に勝った参照）、`rl-cifar-literature-0917`（先行研究の照合: この箱で非 ReLU の数値表は無い）。

## 0. 一行

RL-CIFAR × MLP（3072-100-100-10、1200 枚、50 タスク × 400 epoch、Adam 1e−3、WD なし）で 13 腕 × 入力 2 条件 × 10 seed を回し、
**(A)** 窓（t31–50）で SNA が全員に勝つか、**(B)** kunekune 3 種が SNA と並ぶか（周期性・縦スケール・裾の傾きのどれが効くか）を決める。

## 1. なぜ

- 試走（§9）で SNA は 50 タスクで窓 0.955、低下 −0.028（下がらず上がる）。ReLU は WD なしでも t1 で全滅、LR(0.1) は t3 で下がり始めた。固定 α の Snake は W が 5〜37 まで育つので谷が平均されて消える（mob≈1.00）。
- この箱で非 ReLU 活性化の数値表を出した先行研究は無い（Rohani 2025 は 2×100 MLP で tanh/GELU を図のみ、Lillo & Cheney の 17 活性化表は CNN）。
- SNA でイキるには (i) 強い相手（Smooth-Leaky は L&C の CNN で 98.4）、(ii) 調整の手間の平等（全腕を既定値のまま）、(iii) 入力の標準化条件（L&C は CIFAR を標準化している。当方の試走は /255 だけ）が要る。この spec はその 3 つを塞ぐ。
- kunekune: 試走の前活性を位相 θ = 2αz に写すと、t10 以降は座席が谷（−π/2）にあり、幅は 1.2（2αW の釘付け）。傾き 2 の山と山の間 [−3π/2, π/2] に質量の 93%（第 1 層）/ 97%（第 2 層）があり、右の山の外に 5% / 1.5%、左の山の外に 2% / 1.3%。−2 周期目と +2 周期目は事実上誰もいない（α は中央値で写した近似。§9）。
  Issa の読み「−2 周期目と 1 周期以降はいらない」はこのデータと合う。残る問いは「周期性（裾で再びゲートが下がること）は要るのか、谷 1 つと帯だけで足りるのか」。

## 2. 箱

宿主 `src/pmnist_rlcifar_0907.py` の手順をそのまま使う。**宿主は無改変**だが、`run_one` は 1 プロセス 1 走なので、この実験は走を束ねるエンジン `src/rlcifar_mlp_battle_0918.py` で回す（§2.2）。

### 2.1 手順（宿主と同一）
- 網 3072-100-100-10。init は PyTorch の nn.Linear 既定 U(±1/√fan_in)、`H.init_params(seed, device, DIMS)`（`init` stream）。
- データ: CIFAR-10 訓練 50,000 枚から seed ごとに 1200 枚（`rlc_subset` stream）。タスクごとに一様乱数ラベル（`rlc_labels` stream）、バッチ順は epoch ごとに randperm（`rlc_batch` stream）。画像はタスクをまたいで固定。
- 50 タスク × 400 epoch × 75 step（batch 16）= 30,000 step/タスク、cross-entropy、Adam（lr 1e−3、β 0.9/0.999、ε 1e−8、moment はタスクをまたいで保持）、WD なし、float32。
- 読み出しは宿主の `evaluate_rl`（memo_acc・dead_frac・zeroout・zbar・zsd・zbar_min・mob・eff_rank・w_norm・SNA の α 統計）と `preact_hist`（z のヒストグラム [−512, 256]、幅 0.1、ユニット平均 m1/m2）を各タスクの終わりに 1200 枚で取る。online_acc は更新前の argmax の平均（宿主と同じ）。
- 発散（loss が非有限・重みが非有限）は宿主どおり打ち切り、救わない。束ねた中では発散した走のスロットだけ止め、他の走には触れない（S-diverge）。
- seed 0–9（登録用）。試走・検査は seed 100 以上。

### 2.2 束ねるエンジン
- 1 プロセスに R 本の走を積み、`torch.baddbmm` で順伝播する（前例: `layer_chimera_rl_0914` / `relu_gelu_silu_rl_0914`）。逆伝播は autograd。活性化は要素ごとなので走スロットに沿って当てる。Adam は要素ごと。SNA 系の α の統計は走ごと（形 (R, 100)）。
- 理由: この網は行列積が一瞬で 1 step の時間がカーネル起動の回数で決まる。1 プロセス 1 走では GPU 4.5 ms/step（CPU 1.7 ms/step より遅い）、束ねてもカーネルの回数は同じなので R=20 で 1 本 2.5 時間相当が 20 本になる（S-cost で確定）。AMP は効かず数値も変わるので使わない。
- 宿主との一致（S-stack）: logits は bit 一致、勾配は backward の足し算順の違いで相対 1e−7（2026-09-18 02:00 に seed 100–102・LR・1 step で測定）。30,000 step ではカオス的に離れるので、軌道の bit 一致は求めず、1 step の一致（≤1e−6）と試走との水準一致（S-reuse）で保証する。
- 1 プロセス = 1 腕 × {raw, std} × seed 0–9 = R 20（S-cost で 2 腕 40 本が 1.5 倍以内なら 2 腕束ねてよい。選んだ R は provenance に書く）。
- 装置: GPU（RTX 5060 Ti、torch 2.13.0+cu130）。`CUBLAS_WORKSPACE_CONFIG=:4096:8`。同じ束ね方・同じ装置なら再現する。

### 2.3 入力の 2 条件
- `raw`: 宿主のまま x = u8/255（チャネル順は R 1024・G 1024・B 1024）。
- `std`: x = (u8/255 − mean_c)/std_c、mean = (0.4914, 0.4822, 0.4465)、std = (0.2470, 0.2435, 0.2616)（Lillo & Cheney の RL-CIFAR handler の Normalize と同じ値）。seed の 1200 枚に一度だけ当て、学習も評価も同じ x を使う。
- 同じ seed の raw と std は画像・ラベル列・バッチ順・init が同一なので、seed 対応の対になる。

## 3. 腕（13）。**全腕とも既定値のまま。この実験のために値を調整しない。**

θ = 2αz、Snake の 1 周期 = 2π。谷（φ′=0）は θ = −π/2、傾き 2 の山は θ = −3π/2 と +π/2。

| 腕 | 定義 | φ′ |
|---|---|---|
| `SNA` | z + sin²(αz)/α、α_i = clip(c/W_i, 0.005, 3)、W_i = √EMA_β var(z_i)、c=0.6、β=0.01（宿主の既定。`H.AdaptiveSnake` と同じ式・同じ演算順） | 1 + sin θ |
| `KKA` | kunekune。θ ∈ [−3π/2, π/2] で Snake、外は傾き 2 の直線: z<−3π/(4α) で 2z + 3π/(4α) + 1/(2α)、z>π/(4α) で 2z − π/(4α) + 1/(2α)。α は SNA と同じ則 | 帯の中 1 + sin θ、外 2。つなぎ目は C²（φ″ = 2α cos θ = 0） |
| `KKA23` | (2/3)·KKA(z)。α は z から SNA と同じ則 | (2/3)(1 + sin θ)、外 4/3 |
| `KKT1` | 裾の傾き 1 の kunekune。θ ∈ [−2π, π] で Snake（谷 1 つ＋両側の山まるごと）、外は恒等: z≤−π/α で z、z≥π/(2α) で z + 1/α | 帯の中 1 + sin θ、外 1。つなぎ目は C¹ |
| `R` | ReLU | 1[z>0] |
| `LK001` | leaky、負側の傾き 0.01 | |
| `LR` | leaky 0.1（宿主の腕） | |
| `LK03` | leaky 0.3 | |
| `SL` | Smooth-Leaky（L&C）: a·z + (1−a)·z·σ(k z)、a=0.1、k=c/p=5/3（c=5、p=3） | a + (1−a)(σ + kz σ(1−σ)) |
| `RSL` | Rand. Smooth-Leaky（L&C）: 学習時は r を要素ごとに U(0.125, 0.333) から毎 step 引き a→r、評価時は r=(0.125+0.333)/2=0.229 | 同上（評価は r 固定） |
| `ELU` | F.elu(z, 1)。訓練の微分は autograd のもの（負側 elu(z)+1。float32 では z<−16.64 で 0） | 1[z>0] + 1[z≤0](elu(z)+1) |
| `SILU` | z·σ(z) | σ + zσ(1−σ) |
| `GELU` | z·Φ(z)（erf の厳密形） | Φ(z) + z·N(z) |

- `KKA23` について: Adam・WD なしのこの箱では、活性化の一様な k 倍は「次の層の重みの init と lr を k 倍する」のと厳密に同じ（W″ = kW′ と置けば関数も勾配の向きも同じで、Adam の歩幅だけ k·lr になる）。ゲートの形は変わらない。差が出ればそれは lr・init の効果であることを、判定の名前に含める（§5 の K2）。
- 固定 α の Snake は入れない（試走で mob≈1、W が育つと消える。§9）。SN3・SN06 の参照は `swish_battle_0917` の MLP 箱にある。
- RSL の乱数: 1 プロセスに 1 本の cuda generator（seed は sha256("rsl_noise", 腕, seed の組)）。同じ束ね方なら再現するが、束ね方を変えると r の列は変わる。開示する。
- ELU の微分は F.elu の autograd（expm1 系統）を使い、Lillo & Cheney の nn.ELU に合わせる。手書き e^z（床なし）ではない。

## 4. 検査（`src/rlcifar_mlp_battle_0918_checks.py`。本走の前に all_pass を commit する。閾値は算術から出す）

- **S-act**: 13 腕の解析 φ′ が autograd と一致（相対 1e−5、格子 z ∈ [−60, 60]。ELU は z<−16.64 で両方 0 になることを含む）。SNA 系は α を (100,) の乱数にして per-unit の当て方も見る。
  kunekune 3 種はつなぎ目で φ が連続（1e−5）、φ′ が連続（KKA は φ″ も 0）。変異: 直線の定数を 1/(2α) だけずらすと連続性が落ちる。leaky の傾きを入れ替えると S-act が落ちる。
- **S-stack**: R=3（seed 100–102、raw）で、束ねたエンジンの 1 step の logits と勾配を宿主 `H.forward` + autograd の走ごとの値と比べる。logits は bit 一致、勾配は相対 ≤ 1e−6。LR と SNA（α 更新 1 回後の α を含む）で行う。
  ラベル列・バッチ順・init が走スロットごとに宿主の同 seed の stream と bit 一致。変異: スロットの params を入れ替えると落ちる。
- **S-reuse**（水準の一致）: 束ねた SNA・LR の seed 100・raw・3 タスクを、0917 の試走の行（CPU、`a83c5c0e…/scratchpad/mlpcifar_probe/`。SNA: online 0.851/0.932/0.941、memo 1.0；LR: 0.857/0.860/0.803、memo 1.0/1.0/0.866）と比べる。t1 の online は ±0.03、memo は SNA が 1.0。bit 一致は求めない（装置・足し算順が違う）。
- **S-rsl-mode**: 学習の r が要素ごとに [0.125, 0.333] にあり、続く 2 step で異なる。評価は r=0.229 固定で、2 回の評価が bit 一致。変異: 評価で r を引くと 2 回が食い違う。
- **S-std**: 訓練 50,000 枚全部に変換を当てると、平面ごとの平均が |·|<0.01、標準偏差が 1±0.01。変異: R の定数を G 平面に当てると平均が 0.01 を超える。raw と std で t1 の online が食い違う（S-switch）。
- **S-diverge**: 1 スロットを人為的に NaN にした束と、そのスロットを除いた束で、残りの走の重みが bit 一致。NaN の走は発散タスクを記録し以後の行を出さない。
- **S-cost**: R=20 と R=40 で 1 タスク × 20 epoch の ms/step を測り、1 プロセスの見込み時間と、2 腕束ねの可否を決める。
- **S-eval**: 束ねた評価（acc・dead_frac・zbar・zsd・mob・eff_rank・w_norm・α 統計）が、同じ重みで宿主 `evaluate_rl` を走ごとに呼んだ値と一致（1e−5）。
- 走の開始時に git の状態（HEAD・src/・analysis/ の未 commit 変更）を provenance に記録する。本走中に src/・analysis/ を編集するときは STOP ファイルで launcher を止めてから commit する。

## 5. 読み出し（登録）

窓 = タスク 31–50 の online_acc の平均（走ごと）。早期 = 1–10、後期 = 41–50、低下 = 早期 − 後期。崩壊 = 窓 < 0.5（走ごと）、腕の崩壊 = 10 seed の中央値の窓 < 0.5。
対は seed 対応の符号検定（宿主の `sign`、両側・正確）。「勝ち」= 負けが 1 seed 以下かつ p<0.05。発散した走は対から外し、数を報告する。

- **A（条件ごと）**: SNA 対 12 腕の窓。12 腕すべてに勝てば `SNA_TOP`、誰にも負けないが全勝でなければ `SNA_TIED_TOP`、1 腕でも SNA に勝てば `SNA_BEATEN`（勝った腕を書く）。raw と std で別々に出す。
- **主判定**: 両条件で `SNA_TOP` なら `SNA_WINS_BOTH`。
- **K1（周期性）**: `KKA` − `SNA`（窓、条件ごと）。KKA の勝ち `PERIOD_HURTS`、SNA の勝ち `PERIOD_HELPS`、どちらでもなければ `PERIOD_FREE`。
- **K2（縦スケール = 下流の init・lr）**: `KKA23` − `KKA`。勝ち負けがあれば `SCALE_MATTERS`（向きを書く）、なければ `SCALE_FREE`。
- **K3（裾の傾き）**: `KKT1` − `KKA`。KKT1 の勝ち `TAIL1_BETTER`、KKA の勝ち `TAIL2_BETTER`、なければ `TAIL_FREE`。
- 副（登録）:
  - 腕ごとの std − raw（同 seed 対）: `STD_HELPS` / `STD_HURTS` / `TIE`。
  - 腕ごとの崩壊の有無と、崩壊した seed 数。
  - 腕ごとの低下（10 seed の中央値）と、SNA との低下の対比較。
  - memo_acc の最小値（腕ごと・seed 中央値）。
- 記述（判定なし）: SNA 系 4 腕の位相ヒストグラム（θ = 2α_i z を per-unit の α で、幅 0.02 rad、[−6π, 6π]、各タスク・各層）と、帯の外にある質量の割合の時間経過。座席 2α_med·z̄、mob、W、dead_frac、eff_rank、w_norm の時間経過。発散の一覧。
  この位相ヒストグラムが、§1 の「α の中央値で写した近似」を per-unit の α で置き換える。

## 6. 予測（実装の前）

### Claude（Opus 5）、2026-09-18 02:20
- raw: `SNA_TOP` 80%。`R` は t1 で崩壊 95%。崩壊する確率: `LK001` 75%、`LR` 65%、`LK03` 50%、`SL` 60%、`RSL` 50%、`ELU` 70%、`SILU` 70%、`GELU` 70%。
- std: `SNA_TOP` 45%（leaky 系か SL が並ぶか上回りうる）。`R` は std でも崩壊 60%。
- `SNA_WINS_BOTH` 40%。
- K1: `PERIOD_FREE` 60%、`PERIOD_HELPS` 30%、`PERIOD_HURTS` 10%（質量の 93〜97% が帯の中にあるので、SNA と KKA の差は第 1 層の右裾 5% でしか生まれない）。
- K2: `SCALE_FREE` 60%、`SCALE_MATTERS` 40%（向きは KKA23 が下: lr 2/3 で 30,000 step の記憶が遅れる）。
- K3: `TAIL_FREE` 50%、`TAIL1_BETTER` 20%、`TAIL2_BETTER` 30%。
- 腕の順位（raw、窓の中央値）: SNA ≈ KKA ≈ KKT1 > KKA23 > SL ≈ RSL > LK03 > LR > LK001 > SILU ≈ GELU > ELU > R。
- std で最も得をするのは leaky 系（`STD_HELPS`）、SNA は `TIE`。

### Issa（2026-09-18 13:30 記入。本走は走行中で、13 腕のうち 6 腕が完走していたが、**Claude・Issa とも結果は 1 行も読んでいない**。
記入の直前に Claude が §5 の判定の定義だけを口頭で説明した。確率は付けず、ラベルだけ。）

- **A**: `SNA_TIED_TOP`（両条件とも。SNA は誰にも負けないが、全勝はしない）
- **K1**: `PERIOD_FREE` または `PERIOD_HURTS`（＝ KKA は SNA に劣らない。`PERIOD_HELPS` が出たら外れ）
- **K2**: `SCALE_FREE`
- **K3**: `TAIL2_BETTER`（裾の傾きは 2 のほうがよい）

## 7. 実行計画
1. この spec を commit（実装の前）。
2. エンジン・検査・launcher・report を書く。検査 all_pass を commit。
3. launcher `analysis/rlcifar_mlp_battle_0918/launch.py`: 1 腕 1 プロセス（R=20）を 13 本。起動順は SNA・KKA・R・LR・KKT1・KKA23・SL・RSL・LK001・LK03・ELU・SILU・GELU。
   同時起動の上限: nvidia-smi の compute プロセス（≥500 MiB、swish の CNN を含む）が 10 未満、かつ MemAvailable が「1 プロセスの RSS（S-cost で測る。単走で 1.7 GB）＋ 2.5 GB」以上のときだけ次を起動。自分の本数は 6 まで。STOP ファイルで起動を止める。
   見込み: 1 プロセス 2.5 時間、4〜6 並列で全体 7〜9 時間（swish の CNN が 9/18 午前に終わると並列を増やせる）。
4. 出力: `results/rlcifar_mlp_battle_0918/<腕>/{per_task.csv, provenance.json, hist/<腕>_<cond>_seed<k>.npz}`。csv は宿主の列 ＋ `cond` 列。npz は宿主の h1/h2/m1/m2/oob ＋ SNA 系は alpha1/alpha2 (50,100)・theta1/theta2 の位相ヒストグラム。
5. report・figures → summary.md・verdict.json を commit → §4 の片付け（生データの退避・main へ）。

## 8. 答えないこと
- 傾き・c・帯幅の調整（全腕既定値）。kunekune の c を変える実験は別 spec。
- CNN 箱・PM 箱・WD あり。Lillo & Cheney の数値との直接比較（彼らの指標は全タスク平均の online、当方は窓）。
- KKA23 で差が出たときに、それが init と lr のどちらか。

## 9. 開示（この spec を書く前に見たもの）
- 0917 の試走（seed 100、3 タスク、CPU）: SNA・SN06・SN3・R・LR。SNA 0.85→0.94、SN06/SN3 は mob≈1 で 0.62〜0.67、R は t1 で全滅、LR 0.86→0.80。
- 0917–18 の SNA 50 タスク（seed 100・101、CPU）: 窓 0.955/0.954、低下 −0.028/−0.029、memo 最小 0.980/0.951。
- 上の 2 本の hist から α の中央値で写した位相の分布（§1 の数字。2026-09-18 01:40）。
- 2026-09-18 02:00 の速度測定（seed 100、1 タスク × 20 epoch、GPU、単走 4.5 ms/step、4 並列 6.0 ms/step、RSS 1.7 GB）と、bmm と宿主の 1 step の一致（seed 100–102、LR）。
- α の下限 0.005 は、0907 の試走で下限 0.05 が当たったのを見て widen した値（`pmnist_rlcifar_0907` spec §2.5-b）。
- seed 0–9 の結果はどの腕もまだ無い。

---

## 追補 1（2026-09-18 03:45、実装のあと・本走の前）: 決めたこと、実装でわかったこと、検査

Issa は「眠すぎるので spec のおすすめもお願い」と言って離席した。以下の判断は Claude が決めた。結果はまだ 1 本も見ていない（seed 0–9 の走はまだ無い）。

### 1. Issa の依頼で足したもの

- **重みと前活性の保存**（Issa「GPU だった場合も最低限 W・b と前活性は保存できるよね」）:
  `results/rlcifar_mlp_battle_0918/<腕>/snap/<腕>_<条件>_seed<k>/tNN.npz` に、初期値（t00）と各タスクの終わりの
  W1・b1・W2・b2・W3・b3（float32）、SNA 系は V1・V2（α の統計）、さらに 1200 枚ぶんの z1・z2（float16、画像は subset の順）を置く。
  1 走 89 MB、260 走で約 23 GB。float32 の z は `replay_stack()`（走と同じスロットの並びで呼ぶ）で bit 単位に再現できる（S-snap）。
  float16 にしたのは、θ = 2αz の誤差が 6e−4 rad に収まり（α ≤ 0.08、|z| ≤ 200）、正確な値は上の再現で取れるため。
- **途中再開**: タスクごとに `ckpt.pt`（重み・Adam の moment と step 数・α の統計・全生成器の状態・行）を上書きする。
  プロセスが落ちても続きから走り、止めずに回した走と bit 一致する（S-resume）。50 タスクの先へ延長するときもこれを使う。

### 2. おすすめで決めたこと

- 腕は §3 の 13 本のまま。1 プロセス = 1 腕 × {raw, std} × seed 0–9（R=20）。§2.2 の「2 腕を束ねてよい」案は捨てた（活性化の種類ぶんカーネルが増えるだけで、束ねても速くならない）。
- launcher の上限: 500 MiB 以上の CUDA プロセス（swish の CNN を含む）が 11 未満、自分は 4 本まで（`plan.json` で変えられる。swish が終わったら 6 に上げる）、MemAvailable − 2.0 GB ≥ 2.5 GB。落ちた走は checkpoint から 2 回まで再開する。
- Issa の予測欄（§6）は空のまま本走に入る。あとで書く場合は時刻を付け、結果を見る前だったかを明記する。

### 3. 実装でわかったこと（spec の本文を訂正する点）

- **ELU の微分に床は無い**。torch 2.13 の `F.elu`（inplace でない）の autograd は負側を e^z で計算する。0 になるのは e^z がアンダーフローする z < −103.97 だけで、§3 に書いた「float32 では z<−16.64 で 0」は誤り。
  −16.64（CPU）・−16.98（CUDA）で 0 になるのは「出力 +1」の形（elu(z)+1）のほうで、これは inplace 版や手書きの系統。指標のゲートも e^z に合わせた（CUDA で autograd と bit 一致）。§6 の ELU の予測は書き換えない。
- Smooth-Leaky は原典の式そのまま `alpha*x + (1−alpha)*x*sigmoid(c*(x/p))`。RSL の学習時の r は、原典が `empty_like(x).uniform_(l,u)`、当方は `l + (u−l)·rand`（同じ分布・同じ範囲、乱数の並びは別）。
- **束ねたエンジンの数値**: 学習 step（batch 16）の logits は宿主と bit 一致、勾配は相対 1.6e−7。評価（1200 枚）の forward は束ね方で丸めが変わり、宿主と最大 8.4e−5 ずれる（|z| ≤ 45）。指標はそこから導いた幅に収まる（S-eval、最悪で幅の 4%）。
- **軌道はカオス的**: W1 を 1e−7 動かすと 75 step で指標が 2e−3 ずれ、エンジンと宿主の差と同じ速さで広がる（SNA）。
  さらに、**同じプロセスに積んだ 8 本の実現値のばらつき（LR の t1 で sd 0.0012）は、宿主自身の実現値のばらつき（同じ 1e−7 の揺らぎ、sd 0.0030）より小さい**（同じプロセスのスロットは相関が残る）。
  そこで S-reuse は「宿主自身の実現値の平均と、エンジンの実現値の平均を、宿主の sd で較正して比べる」形にした（§4 の「t1 の online は ±0.03」を置き換える）。0917 の試走行は記述として併記する。
- **速さ**: 1 step ごとの CPU–GPU 同期（宿主の書き方をそのまま引き継いでいた）が、Swish の CNN との時分割で 1 step 14.5 ms（R=20）まで遅くしていた。
  1 step を CUDA グラフに記録して再生する形に変え、6.5 ms/step（1 腕 2.7 時間）になった。eager 版と bit 一致する（S-graph）。
  Adam のバイアス補正は、`x / c`（Python の float）と `x * float32(1/c)` が bit 一致することを確かめてから GPU 上の定数にした。
- 位相ヒストグラム（§5 の記述用）は θ = 2α_i z を per-unit の α で、[−6π, 6π]・0.02 rad（1885 ビン）。hist の npz に th1・th2・alpha1・alpha2 を足した。

### 4. 検査の一覧（`results/_checks_rlcifar_mlp_battle_0918/checks.json`）

| 検査 | 何を見たか |
|---|---|
| S-act | 13 腕の解析ゲート = autograd の微分（float64 で 4e−16）。kunekune の継ぎ目は φ も φ′ も連続。ELU は CUDA で autograd と bit 一致。変異 3 種（leaky の傾き取り違え・KKT1 の帯の取り違え・KKA の直線の定数ずらし）はすべて検出 |
| S-std | 平面ごとの平均が 1.3e−6〜1.7e−4、標準偏差 1±1.3e−4。平面の順序を入れ替える変異は検出。raw と std は違う入力 |
| S-stack | ラベル列・バッチ順・init は宿主と bit 一致。step 0 の logits も bit 一致、勾配は 1.6e−7、SNA の α 統計は 1 回更新後も一致 |
| S-eval | 10 腕で、束ねた指標とヒストグラムが宿主の `evaluate_rl`・`preact_hist` と、forward のずれから導いた幅の中（最悪で幅の 4%） |
| S-rsl-mode | 学習の r は要素ごとに [0.125, 0.333]、評価は固定 r で決定的。同じ鍵の 2 回の走は一致 |
| S-diverge | NaN のスロットは発散として記録され、以後の行を出さない。他のスロットは NaN なしの走と一致 |
| S-snap | 走と同じスロットの並びで再現すると、記録した memo_acc・zbar・mob が 10 桁一致し、保存した float16 の z も一致。条件を取り違えると一致しない |
| S-graph | 13 腕すべてで、CUDA グラフ版と eager 版の行・hist・重みが bit 一致 |
| S-resume | タスク 1 で止めて再開した走は、止めずに回した走と bit 一致（行・hist・スナップショット）。再開の経路を通ったことを provenance で確認 |
| S-reuse | エンジンと宿主の実現値の平均が、宿主の sd で較正した幅の中（§3）。変異（SNA の β=0、LR を傾き 0 に）は外れる |
| S-cost | R=20 で 6.5 ms/step、1 腕 2.7 時間、RSS 1.9 GB、GPU 1.0 GB、1 タスクあたりの npz 58 MB |

### 5. 読み出しの補足（§5 の運用）

- 符号検定の対からは、発散した走だけを外す。崩壊した走（窓 < 0.5）は外さず、その窓の値を使う。
- 崩壊が早くて 50 タスクに届かない腕でも、打ち切りは発散のときだけ（宿主の規則）。
