# drive_rlmnist_1007 — 課題間の第2層駆動を Adam の実更新で測る（A2 の第 2 の箱への移植）

状態: **事前登録**（この spec の commit が登録。本走は未実行。本番 seed 0–9 の観測は一つも無い）
作成: 2026-10-07 / 実装・起草: Claude（Opus、実験担当）/ 依頼: 親（Fable）。目的は論文 V12 の表 5「証拠表（柱 × 箱）」の柱「駆動源の向き」× 箱「乱数ラベル MNIST × ELU→ELU」のセルを「condA 事後」から「登録」へ上げること。
起点 main `91ec9f2e`。run `drive_rlmnist_1007`、branch `claude/drive_rlmnist_1007`、worktree `wt/drive_rlmnist_1007`。
型: [drive_cifar_c_0920](spec_drive_cifar_c_0920.md)（A2・背骨で登録済み）。宿主: `src/mucap_el_run_0916.py` の `run_one(act2_name="ELU1")`（[l2cap_ee_0917](spec_l2cap_ee_0917.md) §2.1 の ref 腕）。

## 0. 一行

**乱数ラベル MNIST × ELU→ELU の ref 腕で、課題を切り替えた後の第 2 層自身の Adam 更新について、沈降の十分条件が上昇の十分条件より多く成立するかを、A2 と同じ定義で測る。** 上流（第 1 層）も動く実際の全移動と区別し、境界付近・課題内の継続・上向き/下向きの量を保存する。観測であり、介入による原因同定ではない。

## 1. 既知情報と、A2 から変える点

### 1.1 起案時に知っていたこと

- **A2（CIFAR・C 腕・ReLU）の結果**: M1x **NOT_SUPPORTED**（主 5 seed とも上向き条件 17–22% > 下向き 7–9%）、M2x CERTIFICATE_CONSISTENT、M3x **BOUNDARY_ENRICHED**（較正窓は最初の 1 epoch = 75 更新、境界の下向き条件 50–57% 対 後続 6–9%）、自己移動 S の主窓総和 5/5 負、上流 U の総和 5/5 負、後続区間の S 総和は 4/5 で正。
- **この箱の課題単位の記録**（`results/l2cap_ee_0917/runs/ref_s{0..9}/per_task.csv`、task 1–10 の行だけを読んだ）: 主 seed 0–4 は task 2–5 で memo_acc ≥ 0.999（各課題を覚え切る）。第 2 層の中央値 $\bar z_2$ は task 1 終端 0.1–0.36 → task 5 終端 −6.6〜−9.1、第 2 層の訓練の微分が厳密 0 の画像の割合（中央値）は task 5 で 0.06–0.14、task 10 で 0.98 前後（task 10 までに第 2 層が死ぬ）。$q_2=w_2\cdot e_2$ の中央値は task 1 の正から task 3 で負に変わり task 5 で −0.39〜−0.50、行ノルムは 1.6 → 3.0。較正 seed 5 は task 5 で memo 0.84 と早く崩れる。
- **開いていないもの**: 同じ記録の `units.npz` にある課題内の診断点（task 2–10 の 0/75/375/1500/3000/6000 更新）。これは窓ごとの全移動 Δm の和を決めるので、登録前には開かず、本走後の照合（§5.5）にだけ使う。
- **登録前の試作**（scratchpad）: 無改変宿主が現環境（venv Python 3.12.3・torch 2.13.0+cu130・機械 white-san）で l2cap ref seed 0 の task 1–2 の per_task 全列・units 全配列・task 1 終端 sha256 を bit で再現すること、検査 seed 100 の task 1–10 で A2 の誤差帯が ELU 宿主で fail 旗・非有限・証明違反 0 で通ること、費用（観測込みで 1 seed 10 課題 約 270 秒）と peak RSS（約 1.06 GB）だけを見た。**証明の件数・S・U・Δm の値は一切表示していない。**
- 独立監査なし。

### 1.2 A2 から変える点（定義は変えない）

1. **宿主**: CPU・float32・1 スレッド・`torch.set_flush_denormal(True)`・`H.setup('cpu')`（決定的アルゴリズム）。前向きは `forward2`（`x @ W.T + b`）、両隠れ層 ELU(α=1)（`src/elu_growth_0909.py` の ELU）。Adam は宿主の形 `p -= lr*(m/c1)/((v/c2).sqrt()+eps)`（c1, c2 は Python の倍精度、A2 は逆数の積）。moment と時刻 tc は課題をまたいで継続。
2. **φ′ は訓練の微分そのもの**: 第 2 層の各画像の φ′_is は、学習 batch の同じ float32 の z2 に宿主の φ を autograd で通した値（z>0 で 1、z≤0 で fl(expm1(z)+1)。float32 では z<−16.64 で厳密 0）。解析的な e^z で代用しない。φ′≥0 なので A2 §4.1 の不等式 |K φ′ e| ≤ |K| φ′ εL はそのまま成り立つ。
3. **大きさ**: 1 課題 6,000 更新（80 epoch × 75）。主窓 N = 4 × 6,000 × 100 = 2,400,000 unit×更新/seed。M3x の BIC は n=80。
4. **seed ごとに独立のプロセス**（宿主が seed ごとの CPU 走）。A2 の R10 の束ねは無い。seed 0–9 を全て走らせ、主 0–4、較正 5–9 を事前に固定。
5. **task 1–10 を走らせる**。主窓 task 2–5、task 6–10 は REPORT_ONLY（第 2 層が死ぬ過程の記録）。task 1 は状態形成と検査用で分母に入れない。
6. **第 1 層の S は REPORT_ONLY**: 入力 1200 枚は固定なので第 1 層の上流 U₁ ≡ 0、Δm₁ = S₁ = ΔW₁·x̄ + Δb₁。証明は付けない。
7. 全 1200 枚の特徴平均は、更新 n の後の値を更新 n+1 の前の値として再利用する（途中で P は変わらない。A2-self-total で再計算と bit 一致を検査）。

コードの正本: `analysis/drive_cifar_c_0920/numerics.py` の `measure`・`pack`・`KEYS`・`gamma`・`dot`・`augmented`、`analysis/drive_cifar_c_0920/stats.py` の `directional`・`certificate_status` を**無改変で import** する。`full_features`（宿主の前向き）と `decompose`（訓練の φ′）だけを `analysis/drive_rlmnist_1007/numerics.py` に移植し、`window` は n を引数にした版を `analysis/drive_rlmnist_1007/stats.py` に書く（A2 の規則で 400 → 80 以外は同じ）。

## 2. 箱・データ・状態（変えない）

乱数ラベル MNIST（seed ごとに固定 1200 枚 `train_x[RL.subset_idx(seed)]`・画素/255・中心化なし・課題ごとに `RL.task_labels(g_lab)` の一様乱数ラベル）、784–100–100–10、bias あり、宿主の初期化 `H.init_params(seed)`、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8・WD なし、batch 16、80 epoch × 75 = 6,000 更新/課題、標本順は `H.stream("rl_batch", seed)` の `randperm(1200)` を毎 epoch。学習腕は ref だけ（上限・bias 固定・帳簿なし）。

**学習軌道は l2cap の ref shard と bit 一致しなければならない**: 各 seed の task 1–5 について、per_task の全列（`results/l2cap_ee_0917/runs/ref_s<seed>/per_task.csv`）・units の全配列（`obsidian-research-data/l2cap_ee_0917/.../ref_s<seed>/units.npz` の該当行）・task 1 終端 state sha256（同 provenance）・無改変宿主（`run_one` を debug capture 付きで新たに走らせたもの）の task 1–5 終端の state sha256。task 6–10 の記録との一致は REPORT_ONLY。

観測は学習の P/m/v/tc/RNG/順列/ラベルに書き込まない。float64 で読むだけ。観測なし走との bit 一致を検査する（§7 A2-host）。

## 3. 評価する量（A2 §3 と同じ）

task t≥2 の各画像について、o は前課題 t−1 の実際のラベル、n は現在課題 t のラベル（同じ 1200 枚にラベルが引き直される）。o を argmax や現在ラベルへ差し替えない。o=n の画像も除外せず d=0 として残す。task 1 は o=n（分母に入れない）。

同一更新前状態・学習 batch で、h_s = a1(x_s)（batch の native float32）、z_is = z2(x_s)、J_ci = W3_ci（更新前の読み出し）、p_s = softmax(logit_s)（float32 の logit を float64 で）、φ′_is = §1.2-2 の訓練の微分。

```
d_is = J_{o_s,i} - J_{n_s,i}
e_is = sum_c p_sc J_ci - J_{o_s,i}
u_is = d_is + e_is              (一標本損失の a2 微分。batch 平均の 1/B を二重に入れない)
epsilon_os = 1 - p_{s,o_s}
L_is = max_{c != o_s} |J_ci - J_{o_s,i}|
|e_is| <= epsilon_os * L_is
```

## 4. 読み出しと十分条件

### 4.1 Adam 版（主・A2 §4.1 と同じ）

theta_i = (w2_i, b2_i)、k = (mu2_old, 1)、h̃_s = (h_s, 1)。M_prev は更新前の第 2 層の一次 moment、V_new は今回の native 勾配を入れた二次 moment。時刻 q は全履歴、c_q = 1−β1^q、D_q = diag(1/(sqrt(V_new/c2_q)+eps))。

```
K^D_is = k^T D_i h̃_s
T^D_i  = mean_batch[K^D_is * phi'_is * d_is]
R^D_i  = mean_batch[|K^D_is| * phi'_is * epsilon_os * L_is]
H^D_i  = k^T D_i M_prev_i
Qminus_i = beta1*H^D_i + (1-beta1)*(T^D_i - R^D_i)
Qplus_i  = beta1*H^D_i + (1-beta1)*(T^D_i + R^D_i)
```

実数では Qminus>0 なら自己移動 S_i<0、Qplus<0 なら S_i>0（S_i = −(lr/c1) k^T D M_new、M_new = β1 M_prev + (1−β1) g、k^T D g = T + mean K φ′ e ∈ [T−R, T+R]）。実装では §4.3 の誤差を差し引いて**符号まで保証できる**ものを CERT_DOWN/CERT_UP とし、どちらでもなければ UNRESOLVED、同じ unit×更新で両方成立なら CHECK_FAILED。素の件数（float64 で Qminus>0 / Qplus<0）と証明できなかった件数も併記する。旧課題の残差上界が課題内で大きくなり条件が成立しなくなることは科学的な結果として残す。

### 4.2 自己移動・上流・全移動（A2 §4.2 と同じ）

```
S_i = (w_new - w_old)^T mu_old + (b_new - b_old)
U_i = w_new^T (mu_new - mu_old)
Delta_m_i = S_i + U_i = w_new^T mu_new + b_new - w_old^T mu_old - b_old
```

mu_old/mu_new は固定 1200 枚の現在の a1 の float64 平均（更新前後）。S を M2x の符号保証の相手とする。native の mean(z2_new) − mean(z2_old) との閉包は別に検査する。S<0 でも U が勝って Δm>0 なら理論違反ではない。S の負の上界が U の正の上界を上回るときだけ CERT_TOTAL_DOWN（副）。

### 4.3 誤差・境界（A2 §4.3・実装追補と同じ式。正本は A2 の `measure`）

成分ごとの Ball(value, error)。u32 = 2⁻²⁴、u64 = 2⁻⁵³、γ_n(u) = nu/(1−nu)、tiny32 = float32 の最小正規数（flush された非正規化数も覆う）、tiny64 同様。

| 量 | 帯（独立の算術） | 由来 |
|---|---|---|
| 再構成勾配 g と native 勾配 | γ₁₂₈(u32)·mean_s[(Σ_c p_sc\|J_ci\| + \|J_{n_s,i}\|) φ′_is \|h̃_s\|] + 128 tiny32（A2 と同じ） | log-softmax と backward の exp（クラス c の相対誤差 ≤ (Δ_c+5)u32、Δ_c = max logit − logit_c、float32 の正規数の p では Δ_c ≤ 87.3）、10 項の読み出し積、φ′ との積 1 回、16 項の batch 和、1/16 は厳密。概算 ≤ 120 u32 < γ₁₂₈。p が正規数を下回るクラスの絶対誤差 ≤ tiny32·\|J_c\| は \|J_n\| の項が覆う（毎更新の検査で確かめる） |
| e の float64 計算 | γ₆₄(u64)(Σ p\|J\| + \|J_o\| + εL + L) + tiny64 | softmax・積和・1−p_o（p_o が 1 に丸まり他クラスに質量が残る境界を含む） |
| 一次 moment | γ₄(u32)(0.9\|M_prev\| + 0.1\|g\|) + 4 tiny32 | 0.9 と 0.1 の float32 化、積 2 回、和 1 回 |
| Adam の実パラメータ差 | γ₁₂(u32)(\|θ_old\| + \|Δθ_ideal\|) + 12 tiny32 | lr・c1・c2・eps の float32 化、m/c1・v/c2・sqrt・+eps・×lr・÷、θ からの減算（計 ≤ 11 回） |
| 1200 枚の平均 µ | γ₁₂₀₂(u64) mean\|a1\| + tiny64 | float64 の和と除算 |
| native の z2 平均 と µ の affine | γ₁₀₃(u32)(Σ_j mean_s\|a1_sj\|\|W2_ij\| + \|b2_i\|) + γ₁₂₀₂(u64) mean\|z2\| + 103 tiny32 | float32 の 100 項内積（任意の和順・FMA）＋ bias 加算、float64 平均 |
| float64 の内積 | γ_{2n+2}(u64) Σ\|a\|\|b\| + tiny64 | 長さ 101（第 2 層の拡大重み）・16（batch） |
| Q の誤差 qerr | (γ_{2·101+2·16+128}(u64) + γ₁₂(u32))·qscale + He + 0.1·mean(\|K\|φ′·eb) + Σ pre(0.1\|g_native−g\| + \|M_new−M_model\|)\|k\| + Σ pre(0.9\|M_prev\| + 0.1\|g\|)·err(k) | A2 の式そのまま |
| 証明 | CERT_DOWN: −lr·c1⁻¹·Qminus + move_error + err(S) < 0。CERT_UP: −lr·c1⁻¹·Qplus − move_error − err(S) > 0。move_error = lr c1⁻¹ qerr + Σ\|k\|\|Δθ_actual − Δθ_ideal\| | 等号・0 を跨ぐ区間は UNRESOLVED |

**測った欠陥は帯を通ってから証明の誤差へ射影する**（自己参照の許容にしない）: native 勾配と再構成の差、moment の差、実パラメータ差と理想 Adam 差はそれぞれ上の独立の帯を超えたら fail 旗。超えなければ、その**測った差**を qerr・move_error に足す。帯は本走の残差に合わせて広げない。fail 旗（勾配・e・moment・Adam・閉包・native 閉包・証明違反・成分閉包）と非有限は毎更新記録する。

**失敗時の扱い**: task 1–5 のどこかで fail 旗または非有限が 1 件でも出たら、その更新の前後の全状態・batch の画像 id・旧新ラベル・全項を `first_failure.pt` に保存してその seed を止め、全登録判定を CHECK_FAILED とする。task 6–10（REPORT_ONLY）では最初の失敗の fixture を保存し件数を数えて走り続け、その seed の REPORT_ONLY 部分に旗を立てる。

### 4.4 conf/label/history と SGD（副・A2 §4.4 と同じ）

CE = L_conf + L_label、π_c = 0.1。課題開始時の moment を M_start として M_hist = β1^s M_start、M_conf は 0 起点の conf 勾配 EMA、M_label = M_actual − M_hist − M_conf。共通の実 V_new と全履歴 bias 補正で S_conf + S_label + S_hist と S の加法閉包を検査する。SGD 形の K = h·µ + 1 での T/R も同じ点で保存するが仮想量と明記する。d/e 分解と conf/label 分解は名前・列を混ぜない。

### 4.5 縮約（A2 §8 と同じ）

100 unit × 全更新の全テンソルは保存しない。各 (seed, task, epoch) について 100 unit ごとに A2 の `KEYS`（件数: cert_down/cert_up/uncertain/cert_total_down/raw_down/raw_up/fail/certificate_violation/nonfinite、輸送: S/U/total/S_conf/S_label/S_hist の和・正部分・負部分・正の件数・負の件数、診断: Qminus/Qplus/T/R/H/move_error/S_error/native_closure/closure/grad_defect/grad_bound/adam_defect/component_defect/SGD_T/SGD_R）の 75 更新の和と |値| の最大を float64 で保存する。第 1 層は S₁ の和・正部分・負部分・正負の件数。

## 5. 登録判定

### 5.1 M1x（唯一の主科学判定）

主窓 task 2 先頭〜task 5 末、全 unit・全更新。seed s ごとに p_down = N_CERT_DOWN/N、p_up = N_CERT_UP/N、D_s = p_down − p_up、N = 2,400,000。主 seed 0–4。中央値への集計変更をしない。片側正確符号検定（非零 D の正の個数 k、非零数 n、p = Σ_{j≥k} C(n,j)/2ⁿ）。

| 状態 | ラベル |
|---|---|
| 全 5 seed で D>0（p = 1/32） | CROSS_TASK_SINK_CONDITION_HOLDS |
| 全 5 seed で D<0 | NOT_SUPPORTED（逆方向が揃ったという記述。両側 5% 検定とは称さない） |
| その他（tie・両条件 0・4/5 を含む） | UNRESOLVED |

HOLDS は**証明可能な方向の下向き優勢**であり、過半の更新が証明された・沈降量の過半を説明した・正味沈降が証明された、という意味ではない。4/5 を繰り上げない。

### 5.2 M2x（保証と実装の整合）

CERT_DOWN で S<0、CERT_UP で S>0 を独立再計算（実際の float32 パラメータ差と µ の float64 内積と誤差）で照合する。証明マージンが数値誤差を越えた反例が 1 件でもあれば CHECK_FAILED。違反 0 で CERTIFICATE_CONSISTENT、主窓の証明件数 0 なら NO_CERTIFIABLE_EVENTS。判定は主窓（seed 0–4 × task 2–5）。全 seed・全課題の違反数も併記する。

### 5.3 M3x（境界と後続、副）

較正 seed 5–9 の task 2–5 だけから、epoch ごとの平均全移動 v_e（全 unit 平均、seed・task 等重み、75 更新で割った 1 更新あたり値）を 80 点に縮約し、この較正データだけを先に開く。一定モデルと 1 変化点の 2 水準モデル（break b = 1..79）を最小二乗で当て、

BIC0 = 80·log(RSS0/80) + 1·log 80、BIC1 = 80·log(RSS1/80) + 3·log 80。

同率なら単純モデル、break 同率なら最小 b、全一定は単純モデル、RSS1 のみ 0 は 2 水準。2 水準の BIC が低く、先頭水準 < 0 かつ先頭水準 < 後続水準なら boundary = 各課題の更新 1..75b、later = 75b+1..6000。それ以外は WINDOW_NOT_IDENTIFIED。BIC は操作的な分割規則で、有意性や真の時定数の検定としない。**窓・較正 input の hash を主 seed の条件頻度を開く前に `window_calibration.json` として commit する。** 主 M1x は窓に依存せず常に全 task 2–5。

窓が定義された場合、主 seed ごとに 4 課題それぞれの p_down(boundary) − p_down(later)（各窓の unit×更新数で割る）を等重み平均し、同じ片側正確符号検定で 5/5 正なら BOUNDARY_ENRICHED、5/5 負なら LATER_ENRICHED（記述）、他は UNRESOLVED。後続区間を「定常状態」と呼ばない。

### 5.4 M4x と完全性

S/U/Δm の正部分・負部分・総和・1 更新あたり率・conf/label/history 成分を、課題別・主窓全体・（定義できれば）境界/後続・最初の 1 epoch/残り 79 epoch に保存する。seed 別の表: U の主窓総和の符号（5/5 負か）、S の主窓総和の符号（5/5 負か）、最初の 1 epoch の S 総和 / 主窓 S 総和。上流の動きや条件不成立の輸送を捨てない。

欠落 seed/task/epoch/必須列/偽 marker は INCOMPLETE、非有限は DIVERGED。主 seed を減らさず全科学判定を抑止する。少ない証明件数や不支持の結果は中断理由にしない。

### 5.5 適用条件（全部満たさなければ登録判定を出さない）

- **(A) 宿主の bit 一致**: 本走の全 seed 0–9 の task 1–5 が §2 の bit 一致を満たす。満たさなければ全ラベル CHECK_FAILED。
- **(B) 完全性**: seed 0–9 × task 1–10 × 80 epoch の縮約がすべて揃い、主窓・較正窓が有限。満たさなければ INCOMPLETE / DIVERGED。
- **(C) 数値検査**: 全 seed の task 1–5 で fail 旗 0・非有限 0。満たさなければ CHECK_FAILED。
- **(D) 縮約と記録の照合（REPORT_ONLY の妥当性検査、判定は変えない。窓の commit 後の report 段で行う）**: task 2–10 の課題内の診断点 0/75/375/1500/3000/6000 更新の区間ごとに、unit 別の Σ total と、同じ走の診断点の $\bar z_2$（= 記録。(A) で bit 一致）の差が |Σ total − Δ$\bar z_2$| ≤ (1+γ_{N+2}(u64))·Σ native_closure + γ_{N+2}(u64)·Σ|total| + N·tiny64 に入る（N はその区間の更新数。更新 n 後の native 平均を更新 n+1 前の値に再利用するので native 平均の差は望遠鏡和になる）。外れたら原因を記録して報告する。

## 6. 予測（走る前・この commit が正本）

親（Fable）の事前確率は A2 の結果（NOT_SUPPORTED・BOUNDARY_ENRICHED・U 5/5 負・S 5/5 負）を知ったうえでのもの。実装者（Claude）の確率は §1.1 の情報（A2 の結果・この箱の課題単位の記録・push と戻りの 2 相の読み）で、課題内の診断点・本走・試作の証明件数を見る前に記録した（scratchpad の記録 2026-10-07T21:38+09:00）。

| 項目 | 事象 | 親（Fable） | 実装者（Claude） |
|---|---|---:|---:|
| M1x | NOT_SUPPORTED | .60 | .50 |
| M1x | CROSS_TASK_SINK_CONDITION_HOLDS | .25 | .25 |
| M1x | UNRESOLVED | .15 | .25 |
| 窓の同定 | WINDOW_NOT_IDENTIFIED 以外 | .70 | .80 |
| M3x（窓が定義された場合） | BOUNDARY_ENRICHED | .75 | .85 |
| U | 主窓の U 総和が 5/5 seed で負 | .70 | .80 |
| S | 主窓の S 総和が 5/5 seed で負 | .60 | .75 |
| 境界の S | 最初の 1 epoch（各課題の更新 1–75）の S 総和 / 主窓 S 総和 > 1/2 が 5/5 seed | .55 | .60 |
| M2x | CERTIFICATE_CONSISTENT | 記入なし | .95 |

境界の S の比は seed ごとに B_s / T_s（B_s = Σ_{t=2..5} Σ_i Σ_{更新 1..75} S、T_s = 主窓の S 総和、T_s = 0 なら不成立）。採点: M1x は 3 クラスの multiclass Brier、他は binary Brier。適用条件を満たさない・検査失敗・M3x の窓未定義は理由付きで未採点。M2x は実装の予測なので科学の的中数に足さない。Issa 本人の予測は未記入（親や私の確率を本人のものとして転記しない）。

## 7. 必須検査（実装 commit の前に実行。本物で PASS、列挙した変異で FAIL）

collector は期待 ID・変異一覧と実 source の sha256 を持ち、未実行・空・別 source の PASS を拒否する。許容はすべて §4.3 の算術から導き、固定値・固定倍率は使わない。変異は検査対象のファイルの一箇所置換（exec）か入力の改変で作り、理論側と観測側を同時に同じ誤りへ置き換える空虚な一致検査をしない。

| ID | 独立の照合 | 必ず落とす変異 |
|---|---|---|
| A2-host | 検査 seed 100・101 の 5 課題 × 80 epoch で、無改変の `run_one(ref, ELU1)` と本 runner（観測あり・なし）が per_task 全列・units 全配列・info の hash・課題終端の全状態（P・m・v・tc）・終了時の両 RNG 状態で bit 一致（観測なしとの一致 = 観測が RNG・学習を進めない）。本番 seed 0–9 の観測なし runner が l2cap ref shard の task 1–5 と §2 の意味で bit 一致し、無改変宿主の task 1–5 終端 sha256 を記録 | 観測が g_batch を 1 回消費、Adam の演算順の入替、第 2 層を leaky に、観測が m に書く |
| A2-input | 1200 枚 = train_x[subset_idx(seed)]（画素/255・中心化なし）、課題ラベル列、task t の旧ラベル = task t−1 の実ラベル（task 1 は o=n） | 旧ラベルを argmax、旧ラベルを 2 課題前、別 seed の画像、入力の中心化 |
| A2-CE | 固定した native の h・z・logit・φ′ で、float64 の独立 autograd（CE → 読み出し → φ′）の勾配と再構成 g の一致、u=d+e、\|e\| ≤ εL + eb、native g が gb の中、autograd の φ′ が fl(expm1(z)+1) と bit 一致し z<−16.64 で 0 | batch 平均の二重（g/16）、conf/label の入替、d の符号反転、解析的 e^z の φ′（z<−16.64 の unit で gb を超える） |
| A2-Adam | 捕えた (M_prev, V_prev, g, tc) から math の scalar 計算で Adam 1 回を独立に再計算し、実 P 差・M・V が §4.3 の帯の中。課題をまたいで tc 継続 | 現勾配のみ（履歴なし）、V_prev を使う、課題で tc を 0 に戻す |
| A2-self-total | 同じ固定 1200 枚の前後 P から S/U/Δm を math.fsum で独立再計算、native 平均との閉包、再利用した µ と再計算した µ が bit 一致 | U の省略、U に W_old、別 seed の µ |
| A2-certificate | 判定式の厳密境界・丸め 0・成立/不成立を独立 fixture で照合 | 誤差を無視、Qplus を down に、K=1 |
| A2-nonvacuous | 検査 seed の task 2 の実更新で µ2≠0・W2 更新≠0・上流更新≠0・S≠Δm・証明件数>0 | 理論側だけ µ を差し替え（観測 S は固定）すると S/閉包が変わる |
| A2-confhist | t≥2 で M_conf + M_hist + M_label = 実 M、S 成分和の閉包、conf 勾配の独立計算 | history なし、conf を label 扱い、moment の上書き |
| A2-window | 合成 1/2 水準（n=80）・同率・定数・逆向き。窓の関数は較正 seed 以外を受け付けない | 主 seed での窓、固定 200 更新 |
| A2-verdict | 3⁵ 符号列の全ラベル・5/5 と 4/5・tie・全 0・未定義・欠測 | unit を独立標本、4/5 を成功、空分母の M2x を PASS |
| A2-manifest | run/seed/task/source/input/hash/marker の明示照合 | 途中の report、別 run の混入、偽の完了 marker |
| A2-cost | 検査 seed で task 1 と task 2 以降を全 1200 枚の毎更新観測込みで計時・peak RSS | batch16 だけの計時、観測を間引いた見積り |

## 8. 実行計画・費用

採用（この commit）→ push → 実装・検査 → 実装 commit/push → 本走 seed 0–9（task 1–10）→ 完全性・宿主一致の確認 → 較正 seed だけの窓生成・commit → 主 report → 予測採点 → 結果 commit → 片付け（§10）。

CPU 1 スレッド（`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）× seed ごとに 1 プロセス、**同時 4 本まで**。本数は A2-cost の peak RSS と `free -g` の空きから決め、空きメモリを常に 6 GB 以上残す（他の 4 エージェントと共有）。`nohup setsid … > results/drive_rlmnist_1007/logs/<名前>.log 2>&1 &`（名前に `/` を入れない）。監視は約 5 分おき、途中の科学成績は開かない。各 seed の縮約は課題ごとに atomic に書く（途中の進み具合の確認用）。1 seed は試作で約 5 分なので途中再開は作らない。止まった seed は出力を別名へ退避して最初から走り直す（同じ source/input なら bit 一致するはず。退避したものも manifest に残す）。

**固定監査区間**（結果に依存しない固定選択）: 各 seed の task 2 と task 5 の最初と最後の epoch について、開始時の全状態（P/m/v/tc・両 RNG・旧新ラベル・観測の状態）・順列・終了時の全状態・その epoch の縮約を保存し、本走後に再計算して全状態と縮約が bit 一致することを確かめる。

## 9. 解釈の範囲

十分条件の成立は原因の除去実験ではない。自己移動・全移動・重みのノルム成長・機能的 LoP を分ける。conf は定義だけで侵食、label は定義だけで回復とはしない。乱数ラベル MNIST・ELU→ELU・ref 腕・10 課題・既知 seed の観測で、他の箱・未使用 seed・長期極限には外挿しない。証明可能な頻度が少なければそのまま報告し、条件を緩めない。

## 10. 出力・片付け

`src/drive_rlmnist_1007.py`、`analysis/drive_rlmnist_1007/{numerics.py,stats.py,checks.py,report.py,launch.sh}`、`results/drive_rlmnist_1007/` に checks.json・host_identity.json・window_calibration.json・report/（summary.md・verdict.json・per_seed.csv・per_task.csv・transport_by_window.csv・per_unit.npz・predictions.json）。seed ごとの縮約 shard・失敗 fixture・監査区間・ログ・検査の試行は git 外。CLAUDE.md §4 に従い git 外の生データ（`__pycache__` 以外、`data` の symlink は辿らない）を `~/Projects/obsidian-research-data/drive_rlmnist_1007/` に移し、`results/drive_rlmnist_1007/backup_manifest.json`（source・backup・bytes・sha256）を commit、main へ merge して push する。worktree とブランチは親が確認してから消す。
