# controls_cifar5p1_1008 — 5+1 CIFAR × MLP で V12 の 2 主張を強くする対照（C1 場の中身か数か／C2 α の追随か）

状態: **登録（この commit）。実装・検査・本走はこの commit より後。**

作成: 2026-10-08 / 起草・実装: Claude（Opus 5.5、実験担当サブエージェント）/ 起点 `origin/main=49ca6ade`。
run: `controls_cifar5p1_1008` / branch: `claude/controls_cifar5p1_1008` / worktree: `wt/controls_cifar5p1_1008`。
親: 依頼元（論文 V12 の主張を査読の問いに耐えさせる対照）。腕・主比較・ラベル・親の事前確率は依頼文のとおりで、起案者は選び直していない。起案者が足したもの（場の作り方の細部、`−` 符号の扱い、R_rand の定義、fresh 対照の定義、検査、Claude の予測）は本文で「追加」と明記する。
型: [resp_cifar5p1_1007](spec_resp_cifar5p1_1007.md)（§2.3 固定場と出力一致・§7 検査）と [cifar5p1_mlp_0920](spec_cifar5p1_mlp_0920.md)（箱・後期窓）。

## 0. 一行

**C1**: 劣化網（task 28 末）の第 2 層に入れて task 29 の online を .404 → .617 に戻した「健康網（task 2 末）の場」を、学んでいない場（新しく初期化した網の場）・開いた数だけ揃えた受け手自身の場・unit を入れ替えた場に替えても同じだけ戻るかを問う。**C2**: 適応 Snake（SNA）とくねくね（KKT1）の α 更新だけを止め（初期値 α = c = 0.6 で固定／課題 1 末の α で固定）、後期窓の優位が α の幅への追随によるかを問う。どちらも実ラベルの 5+1 CIFAR × MLP（std・Adam 1e−4・1 走 20 秒）、seed 0–9。

## 1. 既知情報と登録前に走らせたもの

### 1.1 既知（設計に使った。独立な予測成功として数えない）

- `resp_cifar5p1_1007`（登録 `4ed6a3d5`・結果 `results/resp_cifar5p1_1007/`）: ReLU/std、seed 0–9。task 29 の online E の seed 平均は N_c **0.4042**、R_c←h **0.6174**、fresh(t29) **0.6259**。P1 = +0.2132（97.5% [+0.1835, +0.2429]）。分岐直後の第 2 層 G2 は N_c 0.147、R_c←h 0.569。R_c←h の G2 は更新 78 で 0.319 に下がる（場は固定でも、ずらした引数 z + d は学習で動く）。対応差 R_c←h − N_c の seed 間 SD は約 0.035。
- `cifar5p1_mlp_0920`（seed 0–9、std、lr 1e−4）: 後期窓（hard 21–29 の online、seed 中央値）SNA **.6710**、KKT1 **.6667**、R **.4241**、R+l2init(1e−3) **.6649**。SNA の課題末の第 1/2 層の z の幅（zsd 中央値）は task 1 で 4.1/3.7、task 29 で 7.6/5.7、α の中央値は task 1 末 0.154/0.174 → task 29 末 0.081/0.109、2αW の中央値はどの課題でも 1.20、clip 率 0（KKT1 もほぼ同じ）。**幅は task 1 の中ですでに初期値（第 1 層 ≈ 0.58）の約 7 倍になる**ので、α = 0.6 固定では 2αW は task 1 末で約 5、後期で約 7〜9 になる（予測の根拠に使った）。
- `swish_battle_0917`: 乱数ラベル MNIST で固定 α の Snake が適応に −3.6〜−8.6 pt 負けた。同じ記録の CNN 箱では c = 0.6 の適応が固定 α に負けた（親の背景説明と起案者の確認）。

### 1.2 登録前に走らせたもの（介入腕なし・本走 seed の学習なし）

(a) **環境の確認**（`analysis/controls_cifar5p1_1008/pre_registration_envcheck.py`、出力 `results/controls_cifar5p1_1008/pre_registration/envcheck.{json,log}`）: 無改変の宿主 `cifar5p1_mlp_0920.run('SNA'/'KKT1', seeds 10–19, std, lr 1e−4, 30 課題, fresh)` を現環境（torch 2.13.0+**cu130**、RTX 5060 Ti）で走らせ、cu126 の委任済み記録 `results/cifar5p1_mlp_0920/{SNA,KKT1}_s10-19/` と比べた。**per_task.csv（eff_rank を含む全列・300 行）と fresh_control.csv がどちらも byte 一致**。peak RSS 2.40 GB、peak CUDA 1.81 GB。R/std の同種の照合（resp_cifar5p1_1007 §1.2）では eff_rank_l2 だけが相対 1e−7 で食い違ったので、§4.4 の S2-host(b) は resp と同じく eff_rank を除く全列の byte 一致を関門にし、eff_rank は報告にする。

(b) **場の設計統計**（`analysis/controls_cifar5p1_1008/pre_registration_design.py`、出力 `.../pre_registration/design_stats.json`）: resp の分岐状態 t02/t28 の退避ファイル（sha256 を resp の `backup_manifest.json` と照合）を読み、task 29 の画像（2500 枚 × 10 seed）の第 2 層 z を **CPU float32 で**評価して、場ごとの開いた割合（z ≥ 0 の pair の割合）を数えた。学習はしていない。seed 平均:

| 場（task 29 の画像） | 開いた割合 | 全画像で閉じた unit | 常に開いた unit | unit 別 z の SD（中央値） | unit 別 z の平均（中央値） |
|---|---:|---:|---:|---:|---:|
| 受け手 t28 自身（= N_c） | 0.147 | 32.2 | 3.7 | 5.42 | −7.79 |
| 健康網 t02（= R_h・R_perm・R_rand の値の集合） | 0.569 | 0.1 | 12.0 | 2.35 | +0.41 |
| 新しい網（init seed 1000+s、= R_fresh） | 0.500 | 0 | 0 | **0.18** | 0.00 |
| 受け手 + 一様シフト c（= R_match） | 0.569 | **0** | 8.0 | 5.42 | −0.82 |

R_match のシフトは c = +5.63〜+8.54（seed による）。**シフトだけで受け手の全画像で閉じた unit は 0 になる**（死んだ unit の z の最大値も −c より上にある）。R_fresh の場は z の尺度が R_h の約 1/13 で、学習による z の変化 Δz がすぐ場を上回る（§3.6 の読みの上限に書く）。GPU の値は §3.2 の定義で本走時に作り直し、この表の値は使わない。

seed 0–9 の介入腕・C2 の新しい腕は、登録前に一切走らせていない。

## 2. 固定する箱（両対照共通。`src/cifar5p1_mlp_0920.py` の `run()` そのもの）

- CIFAR-100 train（`cifar-100-python.tar.gz` sha256 `85cd44d0…77a7`、`/home/issan/Projects/claude/proj_004_drift/data/cifar100/`）。30 課題、奇数が hard（5 クラス × 500 = 2500 枚）、偶数が easy（1 クラス 500 枚）、クラスは `c51_classes`、バッチは `c51_batch` stream。780 更新/課題、batch 32。
- 3072–100–100–100、bias あり、入力 std、初期化は `init` stream の U(±1/√fan_in)。Adam lr 1e−4、β = (0.9, 0.999)、ε = 1e−8、WD なし、Adam の時刻 tc は課題をまたいで累積。
- seed 0–9 を R=10 で束ねる。CUDA float32・`baddbmm`・CUDA graph（warmup 3 step を巻き戻す）、`torch.use_deterministic_algorithms(True)`、TF32 off、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、CPU 2 thread。
- online = 課題の 780 バッチの更新前当たり率の平均（宿主 `acc_sum/780`）。
- 検査・計時は seed 100–109（R=10）と合成入力だけ。

## 3. C1: 場の中身か、開いたゲートの数か

### 3.1 分岐

- 接頭部を新しい走で task 1–29 学び、宿主と同じ fresh 対照（P を初期値、m/v/tc を 0、task 29 と同じバッチ列）を走らせる。`resp_cifar5p1_1007` の `Engine`・`natural_prefix` をそのまま import して使う（無改変）。
- 保存点 t02（donor）、t28（受け手）、t29（S1-prefix 用）。分岐は **t_c = task 28 末に固定**、継続課題は task 29（hard）、1 課題 780 更新。
- 全腕は t28 の独立な clone から始め、task 29 の画像・ラベル・780 バッチの行番号を共有する（clone の generator から宿主の手続きで生成）。**全腕 Adam 継続**（m・v・tc を保つ）、対象層は第 2 層、凍結 P0 = t28 の P。

### 3.2 場の定義（d は task 29 の画像だけで定義、外は NaN）

受け手の t28 の第 2 層前活性を z_br、場の目標値を T とし、すべて task 29 の全画像（B = 2500）の 1 回の束ねた評価 forward（GPU float32）で計算する。訓練中の層は resp と同じ

`a2 = φ(z0) + [φ(z + d) − φ(z0 + d)]`、φ(z) = clamp(z, min=0)、z0 は凍結 P0 から現在の minibatch で再計算

（resp の `Field`・`AnchorAdd` を無改変で使う）。分岐時は z = z0 なので括弧は厳密 0 で、全腕の分岐時の出力は自然腕と bit 一致する。

| 腕 | 目標値 T（float32、(R, 2500, 100)） | d | 開いた数 |
|---|---|---|---|
| `N_c` | — | なし（宿主の forward） | 0.147（受け手自身） |
| `R_h` | 健康網 t02 の z2（= resp の R_c←h そのもの） | fl32(f64(T) − f64(z_br)) | 0.569 |
| **`R_fresh`** | 新しく初期化した網の z2。init は宿主の `init_params('R', 1000 + s)`（受け手の init は seed s なので別 seed） | 同上 | 約 0.50（揃えない） |
| **`R_match`** | z_br + c_s（模様は受け手自身） | **定数 c_s**（float32） | R_h と同じ数に揃える |
| `R_perm` | t02 の z2 の unit を置換: T[r, x, i] = z_src[r, x, π_r(i)]、π_r は不動点のない置換 | fl32(f64(T) − f64(z_br)) | R_h と同じ（unit 別の開き具合の集合も同じ） |
| `R_rand`（任意・追加の定義） | t02 の z2 の値を seed 内の全 (画像, unit) pair で一様に並べ替えた値 | 同上 | R_h と同じ |

- **c_s の決め方（R_match）**: K_s = R_h の分岐時の開いた pair の数 = #{(x, i): fl32(z_br + d_h) ≥ 0}（d_h は R_h の d、≥ 0 は clamp の native autograd の 1 の側）。z_br の 250,000 個の値を float64 で降順に並べた v_1 ≥ v_2 ≥ … に対し、**c_s = fl32(−(v_K + v_{K+1})/2)**。実際に開いた数 K' = #{fl32(z_br + c_s) ≥ 0} と K との差を報告する（同値の値や丸めで数個ずれうる。合否の閾値は置かない）。
- **π_r（R_perm）**: `H.stream('ctrl1008_perm', s)` の CPU generator で `randperm(100)` を引き、不動点がなくなるまで引き直した最初の置換。
- **並べ替え（R_rand）**: `H.stream('ctrl1008_rand', s)` で `randperm(250000)` を 1 回引き、t02 の z2（画像 × unit を行優先で平らにしたもの）をその順に並べ替える。
- 依頼文の R_rand は「画像ごとに無作為な d（開いた割合を R_h に揃える）」。起案者は、値の集合（開いた数と |z| の分布）を R_h と同じに保ったまま画像と unit の構造だけを壊す上の定義を選んだ。
- R_fresh の新しい網の d は z の尺度が小さい（§1.2(b)）。尺度も開いた数も揃えない（依頼文の「新しく初期化した網の場」を文字どおりに取る）。

### 3.3 読み出し

- **主 E**: task 29 の online。
- memo、online CE、memo CE、resp と同じ診断（更新 0/78/390/780 に task 29 の全画像で G1・G2・Q・閉じた unit の率・n_eff・z の平均/SD・ずらした引数の平均）。
- 場の検査値（引数誤差/上界の最大比、曖昧帯の pair 数、R_match の K と K'）を seed 別に保存する。

### 3.4 C1 の判定

**適用条件**（先に判定）: (1) 6 腕 × 10 seed が揃い有限、入力・commit・hash が整合し、必須検査が PASS（でなければ `INCOMPLETE` / `CHECK_FAILED` / `DIVERGED`、主ラベルを出さない。seed を落として n を縮めない）。(2) **再現**: `N_c` と `R_h` の行（online_acc・online_ce・memo_acc・memo_ce・label_hash・batch_hash）が `results/resp_cifar5p1_1007/arms/{N_c,R_ch}/rows.json` と一致（でなければ `CHECK_FAILED`）。(3) 復元 P1 = E(R_h) − E(N_c) の 97.5% 区間の下端 > 0（再現が成り立てば既知の値と同じ。満たさなければ `NOT_REPRODUCED`）。

**主比較**（依頼文どおり）: `D_fresh = E(R_h) − E(R_fresh)`、`D_match = E(R_h) − E(R_match)`。n = 10 の対応差、df = 9、**各 97.5% 両側 t 区間**（2 比較の Bonferroni）。符号は下端 > 0 で +、上端 < 0 で −、他は 0（端点 = 0 は 0）。SD = 0 なら区間を同一点とし `DEGENERATE_SD` を併記。

| D_fresh | D_match | 主ラベル |
|---|---|---|
| + | + | `LEARNED_FIELD_MATTERS` |
| + | 0 か − | `PARTIAL`（`fresh_differs`） |
| 0 か − | + | `PARTIAL`（`match_differs`） |
| 0 か − | 0 か − | `OPEN_COUNT_SUFFICES` |

追加（起案者）: 依頼文のラベルは − を扱っていないので、**− は「対照が R_h 以上に戻した」なので「学んだ場が要る」の側に数えない**（上の表）。− が出たらラベルに `CONTROL_EXCEEDS:<腕>` を併記する。0 は「差を示せない」で、同等性の証明ではない（区間の幅を併記する）。

**副比較**（すべて 95%・REPORT_ONLY）: E(R_h) − E(R_perm)、E(R_h) − E(R_rand)、各介入腕 − N_c（R_h・R_fresh・R_match・R_perm・R_rand）、各腕 − fresh(t29)。**復元の割合** ρ_X = mean(E(X) − E(N_c)) / mean(E(R_h) − E(N_c))（報告のみ、丸めない）。

### 3.5 読みの上限

- 主ラベルが `OPEN_COUNT_SUFFICES` なら「この箱・この固定場の介入で、復元は開いたゲートの数（と学習で動く引数）で説明でき、健康網が学んだ模様を要しない」と読む。`LEARNED_FIELD_MATTERS` なら「同じ数を開いても健康網の模様の方がよく戻す」と読む。どちらも 1 課題 780 更新の固定場の人工的な介入で、自然な喪失の全媒介や恒久的な救済は示さない。
- R_fresh は開いた数（約 0.50）も z の尺度（R_h の約 1/13）も揃っていない。D_fresh は「学んでいない網の場」全体の比較で、数だけの比較ではない。数だけの比較は D_match。
- R_match の模様は受け手自身の z の並び。「数を揃えると受け手自身の模様で足りるか」であって、任意の模様で足りるかではない（それは R_rand が副として見る）。

## 4. C2: 適応 α の追随か

### 4.1 腕（seed 0–9、std、lr 1e−4、30 課題 × 780 更新、fresh 対照つき）

宿主 `src/cifar5p1_mlp_0920.py` の `SnakeFamily`（`rlcifar_mlp_battle_0918`）を無改変で使い、α の更新（V の EMA）だけを止める。α_i = clip(c/√V_i, lo, hi)、c = 0.6、lo = 0.005、hi = 3.0、V の初期値 1、β = 0.01。

| 腕 | 形 | α（V）の扱い | 用途 |
|---|---|---|---|
| `SNA` / `KKT1` | snake / kkt1 | 毎 step の EMA（宿主そのもの） | 既存の再現（記録と bit 一致） |
| **`SNA_fix0`** / **`KKT1_fix0`** | 同上 | **V ≡ 1（α ≡ 0.6）、EMA を一度も行わない** | 主比較 |
| `SNA_fixT1` / `KKT1_fixT1` | 同上 | task 1 は宿主どおり適応、**task 1 の最後の更新の後の V で凍結**し、task 2 以降は EMA を行わない | 副（幅に一度だけ合わせる） |
| `SNA_pus`（任意） | u = z / W_i、φ = u + sin²(c·u)/c、W_i = c/α_i = clip(√V_i, c/hi, c/lo) | V は SNA と同じ EMA（適応） | 副（正規化そのもの） |

- SNA の φ は W·S_c(z/W)（S_c(u) = u + sin²(cu)/c、W = c/α）と書けるので、`SNA_pus` は**数式上 SNA の出力（と勾配）を W_i で割ったもの**。正規化した z に固定 α の Snake を当てる腕で、SNA との違いは出力の尺度 W_i だけ（追加の記述）。
- `SNA_fix0` は宿主の `run(..., beta=0)` と同じ計算になる（`update` が β = 0 で何もしない）。検査ではこれを独立の参照に使う。
- **fresh 対照**（REPORT_ONLY、追加の定義）: 宿主どおり P を初期値、m/v/tc を 0 にして task 29 のバッチ列を学ぶ。V は、適応腕（SNA・KKT1・SNA_pus）では 1 に戻して適応、fix0 では 1 のまま凍結、fixT1 では**その走の凍結値 V_T1 のまま凍結**（「同じ活性化関数で新しい重み」）。

### 4.2 読み出し

- **主 E**: 後期窓 = hard 課題 21, 23, 25, 27, 29 の online の平均（seed ごと）。0920 と同じ定義。
- 早期窓（hard 1, 3, 5, 7, 9）、低下（早期 − 後期）、fresh gap、宿主の per_task 列（α の統計・zsd・mob・eff_rank・w_norm ほか）。
- R と R+l2init(1e−3) は委任済み記録 `results/cifar5p1_mlp_0920/R_std_lr0.0001/`・`R_std_lr0.0001_l2init1e-3/` の値（seed 0–9）を使う（依頼文の「既存の値」。新しく走らせない）。

### 4.3 C2 の判定

**適用条件**: (1) 7 腕 × 10 seed × 30 課題が揃い有限、必須検査 PASS。(2) **再現**: SNA と KKT1 の per_task.csv が委任済み記録 `results/cifar5p1_mlp_0920/{SNA,KKT1}_std_lr0.0001/` と **eff_rank_l1/l2 を除く全列で byte 一致**、fresh_control.csv が byte 一致（でなければ `CHECK_FAILED`）。

**主比較**（依頼文どおり）: `A_S = E(SNA) − E(SNA_fix0)`、`A_K = E(KKT1) − E(KKT1_fix0)`、各 97.5% 両側の対応 t 区間（Bonferroni 2）。符号は C1 と同じ。

| A_S | A_K | 主ラベル |
|---|---|---|
| + | + | `ADAPTIVITY_MATTERS` |
| + と（0 か −）のどちらか一方 | | `PARTIAL`（+ の側を併記） |
| 0 か − | 0 か − | `FIXED_ALPHA_SUFFICES`（− があれば `FIXED_EXCEEDS:<腕>` を併記） |

**副比較**（95%・REPORT_ONLY）: SNA − SNA_fixT1、KKT1 − KKT1_fixT1；各 fix 腕（4 本）− R；各 fix 腕 − R+l2init(1e−3)；SNA − SNA_pus、SNA_pus − R。**fixT1 の同等性の読み**: SNA − SNA_fixT1 の 95% 区間が [−0.005, +0.005] に収まれば `FIXT1_EQUIVALENT_0.005`（帯 0.005 は依頼文の値で、0920 §15 E1 の Snake 族内の同等性の帯と同じ）。KKT1 も同様に報告する。

### 4.4 読みの上限

- `ADAPTIVITY_MATTERS` は「この箱で、α を幅に追随させないと後期窓が下がる」と読む。何が下がるのか（表現・可塑性）は fresh gap と早期窓で補足するが、機構の判定はしない。
- fixT1 が SNA と同等なら「幅に一度合わせれば足りる（追随し続ける必要はない）」と読む。SNA_pus は正規化の一形態にすぎず、正規化一般の判定ではない。
- 0 は差を示せないで、同等性ではない（区間の幅を併記）。

## 5. 統計（共通）

- 区間は `analysis/resp_cifar_ee_0920/stats.py` の `interval`（10 個すべて有限を要求、df = 9、t 分位点は登録済みの連分数実装）。主は 97.5%、副は 95%。各対照の主比較 2 本が 1 つの族（C1 と C2 は別の族）。
- 対応は seed（同じ seed は同じクラス計画・同じバッチ列・同じ初期重み）。対応なしの比較はしない。

## 6. 予測（実装・本走の前に記入）

主ラベルは成立時の分布（合計 1）。採点は最大確率ラベルの一致と多クラス Brier、他は真偽と binary Brier。適用外・undefined は採点しない。

| 項目 | 親の事前確率 | 実装担当（Claude） |
|---|---:|---:|
| **C1** `OPEN_COUNT_SUFFICES` | 0.45 | 0.35 |
| C1 `PARTIAL` | 0.35 | 0.45 |
| C1 `LEARNED_FIELD_MATTERS` | 0.20 | 0.20 |
| C1 R_fresh − N_c の 95% 下端 > 0 | 0.85 | 0.92 |
| C1 R_match − N_c の 95% 下端 > 0 | — | 0.92 |
| C1 R_h − R_rand の 95% 下端 > 0（構造のない場は戻しが少ない） | — | 0.65 |
| C1 R_h − R_perm の 95% 区間が 0 を含む | — | 0.55 |
| **C2** `ADAPTIVITY_MATTERS` | 0.55 | 0.75 |
| C2 `PARTIAL` | 0.30 | 0.17 |
| C2 `FIXED_ALPHA_SUFFICES` | 0.15 | 0.08 |
| C2 `FIXT1_EQUIVALENT_0.005`（SNA − SNA_fixT1 の 95% 区間 ⊂ ±0.005） | 0.35 | 0.20 |
| C2 SNA − SNA_fixT1 の 95% 下端 > 0 | — | 0.45 |
| C2 SNA_fix0 − R の 95% 下端 > 0 | — | 0.90 |
| C2 R+l2init(1e−3) − SNA_fix0 の 95% 下端 > 0 | — | 0.80 |
| C2 SNA − SNA_pus の 95% 下端 > 0 | — | 0.55 |

Claude の理由（短く）:
- C1: どの場も受け手の閉じた unit（32 個）に勾配を通すので、R_fresh・R_match は N_c より大きく戻る。R_fresh の場は尺度が小さく、数十更新で引数が Δz に支配されて「凍結値 + relu(Δz)」の新しい unit のように振る舞うので、R_h に劣らない見込みが高い。R_match は受け手の z の尺度（SD 5.4）で門が固く、学習中に門の模様が動きにくいので R_h より少し劣る目がある。よって片方だけ差が出る PARTIAL を最も高くした。R_rand は画像の構造のない門で、初見の画像（1 周目）で汎化しにくい。
- C2: 幅は task 1 の中で約 7 倍になり、α = 0.6 固定では 2αW が後期に 7〜9（何周期も振動する領域）になる。KKT1 は固定 α では Snake の窓 θ ∈ [−2π, π]（z ∈ [−5.2, 2.6]）が幅の一部しか覆わず、窓の外は恒等写像（右側は 1/α の平行移動）なので、非線形が幅のごく一部に限られる。どちらも表現が線形に近づくので後期窓は SNA より数 pt 下がると見る。fixT1 は後期の 2αW が約 2 に留まり差は小さいが、±0.005 の帯には入りにくい（対応差の SD が 0.005 程度なら、n = 10 の 95% 区間の半幅は約 0.0036 で、平均差が約 0.0014 以内でないと帯に入らない）。

## 7. 検査（本走の前に全部実行し、本物で PASS・列挙した変異で FAIL）

宿主／無改変の resp エンジン／独立に組んだ参照／合成 fixture の期待値で照合し、自己参照にしない。検査は seed 100–109（R=10）と合成入力。全必須検査・列挙変異を機械可読な一覧から集め、未実行・空・古い PASS を失敗扱いにする（`results/_checks_controls_cifar5p1_1008/checks.json`、source hash を記録）。

| 検査 | 独立の参照・要求 | 落とすべき変異 |
|---|---|---|
| S-src | import して使う無改変の宿主・resp のファイルの sha256 が、resp の admission checks（`results/resp_cifar5p1_1007/admission_checks.json`）の記録と一致 | — |
| S1-host | 本実装の C1 接頭部（30 課題＋fresh）が現環境の無改変宿主 `run('R')` と per_task.csv・fresh_control.csv の全列 byte 一致 | 接頭部の保存点の取り違え（t27 を t28 として保存） |
| S1-prefix | t28 から再開した N_c の終端 P/m/v/tc/generator・バッチ列・行が接頭部の task 29 末と bit 一致 | m/v の欠落、tc の 0 戻し、generator の未復元 |
| S1-wiring | 本実装の場の腕の経路で作った `R_h` が、resp の `run_arm('R_ch')` と行・終端状態・診断・場のすべてで bit 一致 | donor を t03 に、Adam を reset、対象層を第 1 層に |
| S1-branch | 場のある新しい 4 腕で、分岐時の z1/a1/z2/a2/logit/CE が自然腕と bit 一致（全画像一括・780 訓練バッチ・32 枚刻み） | 凍結補正を落とす、P0 を現在の P に alias |
| S1-field | 各腕の d を独立の参照（CPU float64 の並べ替え・置換・中点）から作り直して全要素 bit 一致、task 外は NaN、引数の誤差上界（resp §2.3 の式、T を目標に）と上界外の全 pair でゲート一致、R_perm/R_rand の開いた数が R_h と厳密に同じ、R_match の d が定数 c_s で c_s が中点の定義どおり、R_fresh の donor が `init_params('R', 1000+s)` | R_fresh の donor を受け手の init（seed s）に、R_perm を恒等置換・全 seed で同じ置換・受け手側に置換、R_rand を並べ替えなし・unit の中だけの並べ替え、R_match を c = 0・unit ごとのシフト・K を R_fresh の数に、d の符号反転、他 seed の d |
| S1-isolation | 腕の順を反転して全腕の結果が bit 一致、分岐状態・donor の hash 不変 | 状態の浅い copy |
| S1-graph | eager と graph が bit 一致（R_fresh・R_match） | — |
| S2-host | 本実装の C2 エンジンの SNA・KKT1 が現環境の無改変宿主 `run('SNA'/'KKT1')` と、SNA_fix0・KKT1_fix0 が無改変宿主 `run(..., beta=0)` と per_task.csv・fresh_control.csv の全列 byte 一致（腕名の列を除く） | lr 2e−4、tc + 100、β 0.02、fix0 で EMA を行う、capture の warmup で V を戻さない |
| S2-freeze | fixT1: task 1 の行が適応腕と byte 一致、task 2 以降の各課題末の V が V_T1 と bit 一致、α を直接持つ独立の実装（`FixedAlpha`、凍結 α を 1 回だけ計算して保持）で task 2–4 を学ぶと行と終端状態が bit 一致 | task 2 末で凍結、最初から凍結（= fix0）、凍結時に V を 1 に戻す、task 3 だけ EMA を再開 |
| S2-pus | φ と autograd の微分が独立な float64 の式と γ_n × 絶対値の和の上界内、φ_pus · W と φ_SNA の差が丸めの上界内（「SNA を W で割ったもの」の確認） | W で割らない、√V でなく V で割る、clip を外す（clip に掛かる fixture で） |
| S2-graph | eager と graph が bit 一致（SNA_fixT1 の task 1–3、SNA_pus の task 1–2） | — |
| S-verdict | 両対照の全ラベル（− を含む）・端点 0・SD 0・欠測・非有限を独立 fixture で確認、t 分位点を独立の Simpson 積分で確認 | 対応なし、Bonferroni 忘れ（主に 95%）、欠測 seed、欠測腕、非有限 |
| S-resume | 腕の境界での STOP → 再開が中断なしの走と bit 一致、完了 marker は identity と hash を照合 | 偽の完了 marker、壊れた hash、異なる identity |
| S-cost/CLI | 検査 seed で CLI（`--check-mode`）が両対照の全腕を走らせ、時間・RAM・VRAM を実測 | 腕名の取り違え、課題の更新数の変更 |

bit 一致以外の算術比較は u32 = 2^−24・u64 = 2^−53・γ_n = nu/(1−nu) と和の絶対量で許容を定め、固定の相対許容・倍率を置かない。期待差が非零の fixture で誤った実装を検出できることを確かめる。`resp_cifar5p1_1007` の `Field`・`AnchorAdd`・`Engine`（Adam・graph・reset・診断）は同 run の検査（S-grad・S-reset・S-graph・S-resume ほか、変異 42 本）を通った無改変のコードで、S-src がその同一性を確かめる（照合するのはコードの 11 ファイル。resp の spec は結果の追記で hash が変わっているので除く）。

## 8. 実行計画

登録 commit → 実装 → 検査 seed での全検査 → 実装 commit → 本走 → 判定・予測採点・summary.md → 結果 commit → git 外の生データを `/home/issan/Projects/obsidian-research-data/controls_cifar5p1_1008/` へ退避し `results/controls_cifar5p1_1008/backup_manifest.json` を commit → `git fetch origin && git merge origin/main && git push origin HEAD:main`。worktree とブランチは消さない。

- 本走は `analysis/controls_cifar5p1_1008/launch.sh` が `src/controls_cifar5p1_1008.py` を起動する。実装の commit 済み・検査の all_pass と source hash の一致・入力の hash を確かめてから走る。順序: C1（接頭部 → S-host(b) 相当の 0920 記録照合 → N_c → S1-prefix → R_h → 再現の照合 → 残り 4 腕）→ C2（SNA → KKT1 → 記録照合 → 残り 5 腕）。照合が落ちたらその対照の残りの腕を走らせず `CHECK_FAILED`。
- GPU は 1 プロセス・腕は直列。共有ロック `/tmp/lop_analysis_gpu.lock`（blocking flock）を取り、MemAvailable ≥ 実測 peak RSS + 6 GB と GPU 空き ≥ 5 GB を待ってから走る。
- STOP は `results/controls_cifar5p1_1008/STOP`（腕の境界で止まる）。再開は同じ入力・spec・実装・環境でだけ許す。`provenance_start.json` に git hash・source hash・環境・argv・UTC/JST・PID を保存する。
- 途中の効果量を見て腕・窓・seed・予測を変えない。

## 9. 開示・範囲

- 独立監査なし。実装・検査は実装担当（Claude）による。
- C1 は既知の分岐状態（resp）、C2 は既知の 0920 の記録を参照して設計した確認的な対照で、完全盲検ではない。§1.2 の環境確認（seed 10–19、既存の腕）と設計統計（学習なし）は登録前に行った。
- 対象外: 第 1 層の場、Adam reset 版の対照、他の分岐点、raw 入力、c・β の掃引、他の活性化の固定 α。

## 10. 出力・片付け

`results/controls_cifar5p1_1008/` に summary.md、verdict.json/csv、paired.csv、c1_arm_table.csv、c2_arm_table.csv、c1_per_seed.csv、c2_per_seed.csv、c2 の各腕の per_task.csv/fresh_control.csv、c1 の prefix の照合表、field_stats.csv、predictions.csv、admission_checks.json、provenance_start/end.json、input_manifest.json、pre_registration/。状態（.pt）・場・unit 配列（.npz）・ログは git 外とし、退避の manifest に source・backup・bytes・sha256 を残す。
