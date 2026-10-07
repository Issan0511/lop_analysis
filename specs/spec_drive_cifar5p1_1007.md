# drive_cifar5p1_1007 — A2 の自己移動・上流移動・全移動を 5+1 CIFAR × MLP（R/std）へ移植する

状態: **登録（この spec の commit が登録）。実装・検査・本走は未実施。科学量（S・U・Δm・J の値）は 1 つも見ていない。**

2026-10-07 / Claude（Fable の依頼による subagent）/ 起点 main `91ec9f2e`。run `drive_cifar5p1_1007`、branch `claude/drive_cifar5p1_1007`、worktree `wt/drive_cifar5p1_1007`。
型: [A2 drive_cifar_c_0920](spec_drive_cifar_c_0920.md)（§4.2・§4.4・§5 M3x/M4x・§7）。箱: [cifar5p1_mlp_0920](spec_cifar5p1_mlp_0920.md)（R/std・lr 1e−4）。

## 0. 一行

**5+1 CIFAR × MLP の既定セル R/std で、毎更新の前後に「現在の課題の全画像」での第 2 層の前活性平均の変化を、自己移動 S・上流移動 U・全移動 Δm = S + U に分け（課題の切替で画像集合が入れ替わる分 J は別項）、境界（課題の先頭）への濃縮・上流の向き・easy と hard の差・自己移動の正味を、較正 seed だけで決めた窓と主 seed 5 本の 5/5 符号で判定する。** 観測であり介入ではない。学習は 1 bit も変えない。

## 1. 位置づけと既知

- V12 表 5「証拠表（柱 × 箱）」の柱「駆動源の向き」× 箱「5+1 CIFAR × MLP（実ラベル）」のセルは「事後」。根拠は spec_cifar5p1_mlp_0920 §8.3 の事後読み「殺しているのは easy タスク（1 クラス 500 枚を 50 epoch）」で、t28（easy）の死 .64 → t29（hard）の .37 という**課題末の値**に基づく。課題末の値は課題の中の移動と、画像集合の入れ替わり（easy は 1 クラスで均質）が混ざっていて分かれていない。この run はそれを分けて登録する。
- 型 A2（ReLU・C 腕・CIFAR-10 乱数ラベル・5 課題 × 30,000 更新）の結果を知ったうえで起案している: M1x **NOT_SUPPORTED**、M3x **BOUNDARY_ENRICHED**（較正窓は最初の 1 epoch = 75 更新）、S 総和 5/5 負、U 総和 5/5 負、後続区間の S 総和は 4/5 で正、S_conf 総和 5/5 負（共通分母で約 10^10 単位の相殺）。
- 箱の既知（R/std・lr 1e−4・seed 0–9・`results/cifar5p1_mlp_0920/R_std_lr0.0001/`）: 後期窓 .424、fresh gap +.228、死 l2 .38、mob l2 .002、z̄ l2 −8.82。課題末 z̄ l2 の seed 平均は t1 +1.58 → t30 −8.46。第 1 層は 30 課題を通して死者 0。
- **実装前に行ったこと（科学量は見ていない）**: 無改変の宿主 `cifar5p1_mlp_0920.run('R', seeds 0–9, 'std', 30, fresh=True, graph=True)` を現環境（torch 2.13.0+cu130、記録時は +cu126）・2 スレッドで再走し、記録の per_task.csv と比べた。**eff_rank_l2 の 10 桁目だけが 300 行中 111 行で異なり、他の全列と fresh_control.csv はバイト一致した。** 6 課題で 1/2/4/8/20 スレッドを試し、どのスレッド数でも記録の eff_rank_l2 は再現しない（2 スレッドで 60 行中 10 行、他は 20–23 行が差）。eff_rank_l1 は全スレッド数で一致。学習軌道に依存する列（online・train・test・dead・zbar・mob・w_norm）はすべて一致しているので、差は評価専用の float64 Gram 行列（GPU の cuBLAS）または LAPACK の環境差と読む。§9 S-host の基準をこれに合わせる。

## 2. A2 の M1x/M2x がこの箱で定義できない理由

A2 の十分条件 CERT_DOWN/CERT_UP は、同じ画像に旧ラベル o と新ラベル n が付く乱数ラベル箱の d = J_o − J_n（J は最終読み出し W3 の行）に依る。5+1 CIFAR ではクラスが再訪せず、課題ごとに画像も入れ替わるので、旧ラベル o が存在しない。**したがって条件の成立頻度（M1x）と証明と実移動の整合（M2x）は登録しない。** 登録するのは A2 §4.2 の S/U/Δm、§4.4 の conf 成分（副）、§5 M3x の境界濃縮、M4x の収支に当たる量。

## 3. 箱（変えない）

`src/cifar5p1_mlp_0920.py` の `run()` の既定セルを無改変で使う: CIFAR-100（sha256 `85cd44d0…77a7`）、`std`（チャネル平均・標準偏差で正規化）、3072–100–100–100、両隠れ層 ReLU（`torch.clamp(z, min=0)`）、bias あり、PyTorch 既定の一様初期化（stream `init`）。30 課題・hard と easy が交代（奇数課題が hard = 5 クラス 2,500 枚、偶数課題が easy = 1 クラス 500 枚、クラスは再訪しない、stream `c51_classes`）。各課題 780 更新 × batch 32（stream `c51_batch`、順列の連結を 32 ずつ切る）。Adam lr 1e−4、β=(.9,.999)、eps 1e−8、WD なし、moment と時刻 tc は課題境界で継続。R=10 束ね（seed 0–9 を昇順 slot）、1 更新を CUDA graph で捕獲して replay。fresh 対照（最後の hard 課題 29 を初期値から同じ batch で再学習）も宿主どおり走らせる。学習腕は R だけで介入しない。TF32 off、決定的アルゴリズム、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、torch 2 スレッド（eff_rank の LAPACK のため宿主 CLI の既定に合わせる）。

本走は seed 0–9 の R10。**主 seed 0–4、較正 seed 5–9**（窓の較正専用で主判定に入れない）。検査は seed 100–109 の R10 を使い、seed 0–9 の観測は本走以外で開かない。

観測は学習の P/m/v/RNG/順列・CUDA graph の静的テンソルに書き込まない。graph の replay の前後に eager で読むだけ（float32 の保存状態を float64 に上げて計算する）。100 unit × 全更新のテンソルを CPU に転送せず、GPU 上で課題内の bin ごとに縮約する。

## 4. 測るもの

### 4.1 記号

slot r（seed）、課題 t = 1..30、課題内の更新 j = 1..780。X_t は課題 t の画像集合（宿主の評価と同じ行順 `cat(class_rows[q] for q in classes)`、hard N=2,500、easy N=500）。h(P; x) = clamp(W1 x + b1, 0)（float32 の宿主 forward）。
µ(P; X_t) = (1/N) Σ_n h(P; x_n) を float64 で（float32 の h を float64 に上げて和を取る）。第 2 層 unit i の平均前活性 m2_i(P; X_t) = w2_i·µ + b2_i（float64、w2_i は W2 の行）。

### 4.2 第 2 層の自己移動・上流移動・全移動（主）

更新 j の前後の実際の float32 パラメータ P_old/P_new（native 学習が書いた値）と、同じ X_t での µ_old = µ(P_old; X_t)、µ_new = µ(P_new; X_t) から

```
S_i  = (w2_i,new − w2_i,old)·µ_old + (b2_i,new − b2_i,old)     自己移動
U_i  = w2_i,new·(µ_new − µ_old)                                 上流移動
Δm_i = S_i + U_i = m2_i(P_new) − m2_i(P_old)                   全移動
```

µ_new は次の更新の µ_old としてそのまま使う（同じ P・同じ画像・同じ演算で決定的）。負は「沈める側」。閉包は 2 つ独立に検査する（§9 S-self-total）: (i) float64 の代数的閉包 |(m2(P_new) − m2(P_old)) − (S + U)|、(ii) native の float32 前活性 z2 = baddbmm(b2, h, W2ᵀ) の画像平均の差 mean(z2_new) − mean(z2_old) との閉包。許容はどちらも §9 の丸め上界（γ_n、u32 = 2⁻²⁴、u64 = 2⁻⁵³）から導き、固定値・固定倍率は使わない。

### 4.3 集合の入れ替わり J（課題ごと）

課題 t ≥ 2 の最初の更新の前に、同じ P（課題 t−1 の終状態）で

```
J_i(t) = w2_i·(µ(P; X_t) − µ(P; X_{t−1}))
```

を保存する。µ(P; X_{t−1}) は課題 t−1 の最後の更新の µ_new そのもの。**J は U に入れない。** 恒等式として m2_i の課題末 → 次の課題の先頭の差が J に等しく、課題の中の Σ_j Δm が課題末 − 課題先頭に等しい（§9 S-self-total (c) で検査）。課題末と課題先頭の m2（unit 別）も保存する。

### 4.4 第 1 層（REPORT_ONLY）

入力 x は課題の中で固定なので、µ_x(t) = (1/N) Σ_n x_n（float64）は課題内で一定。S1_i = (w1_i,new − w1_i,old)·µ_x + (b1_i,new − b1_i,old)。**U1 ≡ 0（構成上）、Δm1 = S1。** 切替の J1_i(t) = w1_i·(µ_x(t) − µ_x(t−1))。判定には使わない。

### 4.5 conf/label/history の分解（副・A2 §4.4 の移植）

CE = L_conf + L_label、L_conf = mean(logsumexp(z3) − πᵀz3)、L_label = −mean((e_y − π)ᵀz3)、π_c = 1/100。logit 勾配は conf: (p − π)/B、label: (π − e_y)/B（B = 32）。第 2 層へは δ2 = (δ3 W3) ⊙ 1[z2 ≥ 0]（宿主の clamp の native 微分は z = 0 でも 1）、g_W2 = Σ_s δ2_s a1_sᵀ、g_b2 = Σ_s δ2_s。これを同じ batch・更新前状態で float64 で作る。

課題の開始時の実 moment を M_start として、s 更新後 M_hist = b1f^s·M_start、M_conf = conf 勾配の EMA（0 起点）、**M_label = M_actual − M_hist − M_conf（残差）**。b1f, a1f は宿主の float32 演算で実際に使われる係数 fl32(0.9)、fl32(1 − 0.9)。実分母 D = 1/(sqrt(v_new · inv_c2) + eps) と実際の inv_c1・inv_c2（float32 の値）を 3 成分で共有し、S_comp,i = (−lr·inv_c1·D ⊙ M_comp)_i·(µ_old, 1)。S_conf + S_label + S_hist と実 S の差は実パラメータの float32 丸めだけで、これを §9 S-conf の上界で検査する。別に label 勾配の EMA（0 起点）を直接積算し、残差 M_label と累積上界の中で一致することを検査する（history の欠落・conf/label の入れ替えを独立に落とすため）。

**共通分母の相殺に注意する**: 当てはめ後は conf 勾配と label 勾配がそれぞれ O(1) で打ち消し合い、実勾配は小さく D は大きい。S_conf と S_label は実 S より桁で大きい逆符号の項になりうる（A2 で約 10^10 対 10^3）。**寄与率・割合として読まない。独立介入の効果とも呼ばない。**

### 4.6 bin への縮約

課題内を 26 更新ごとの 30 bin（bin b = 更新 26(b−1)+1 .. 26b、b = 1..30）に縮約する。hard は 1 epoch ≈ 78 更新なので約 3 bin/epoch、easy は 1 epoch ≈ 15.6 更新。hard も easy も 780 更新 = 30 bin。bin ごと・slot ごと・課題ごとに、S・U・Δm・S_conf・S_label・S_hist・S1 の unit 和（全更新の和）・正部分の和・負部分の和・正の件数・負の件数、観測した更新数、§9 の検査の違反件数と最大比（誤差/上界）を GPU で積算する。S・U・Δm は unit 別 × bin の和も保存する。**1 更新あたり率**は unit 平均・更新平均（和 / (100 × 更新数)）と定義する。総和は unit 和（A2 と同じ単位）で報告する。

### 4.7 課題末の列

online・memo（train_acc）・dead/mob/z̄（l1, l2）は宿主の per_task.csv の列をそのまま使う。G2（第 2 層の訓練の微分 1[z2 ≥ 0] の画像平均 → unit 平均）を課題末の全画像で別に記録する。いずれも REPORT_ONLY。

## 5. 窓の較正（A2 M3x の規則の移植・較正 seed 5–9 だけ）

主 seed 0–4 の値を開く前に、較正 seed 5–9 の課題 2–30 だけから、hard（課題 3, 5, …, 29 の 14 課題）と easy（課題 2, 4, …, 30 の 15 課題）で**別々に**窓を決める。

v_b（b = 1..30）= 各（較正 seed, 課題）について [bin b の Δm の unit 和 / (26 × 100)] を作り、全 seed・全課題を等重みで平均した値（1 更新あたりの unit 平均の全移動）。一定モデルと、break b = 1..29（先頭 = bin 1..b、後続 = bin b+1..30）の 1 変化点 2 水準モデルを最小二乗で当てる。

```
BIC0 = 30·log(RSS0/30) + 1·log 30
BIC1 = 30·log(RSS1/30) + 3·log 30      （2 水準 + break）
```

同率は単純モデル、break の同率は最小の b。RSS0 = 0（全一定）は単純モデル、RSS1 だけ 0 は 2 水準（BIC1 = −∞）。**BIC1 < BIC0 かつ 先頭水準 < 0 かつ 先頭水準 < 後続水準**なら boundary = 更新 1..26b、later = 更新 26b+1..780（hard_window・easy_window）。それ以外は WINDOW_NOT_IDENTIFIED。これは自己相関を含む系列の**操作的な分割規則**であり、BIC 差を有意性・真の時定数の検定と呼ばない。規則とコードは本走前に固定し、得られた窓・較正入力の sha256 を `results/drive_cifar5p1_1007/window_calibration.json` として**主 seed を開く前に commit する**。主判定は窓に依存しない部分（D2・D3・D4）を含め、窓の選択を主 seed で変えない。

## 6. 登録判定（主 seed 0–4・課題 2–30・片側正確符号検定は 5/5 のみ）

5 seed の符号検定は 5/5 だけを検出とする（片側 Bin(5, 1/2) で P(K=5) = 1/32、P(K≥4) = 6/32 なので 4/5 を繰り上げない）。0 は非零に数えない。独立標本単位は seed（unit・更新・課題を標本にしない）。頻度（件数）と移動量（和）を混ぜて読まない。D1–D4 はすべて移動量で判定する。

- **D1（主）BOUNDARY_ENRICHED**: seed ごとに、課題 t = 2..30 それぞれで [boundary の 1 更新あたり S（unit 平均）] − [later の 1 更新あたり S] を取り（課題 t の種類の窓を使う）、29 課題を等重みで平均する。**5/5 負 → BOUNDARY_ENRICHED**（境界のほうが沈める側）、5/5 正 → LATER_ENRICHED、他 → UNRESOLVED。hard_window・easy_window の**両方**が同定されたときだけ判定し、片方でも WINDOW_NOT_IDENTIFIED なら D1 は NOT_JUDGED。副として D1-hard（課題 3..29 の奇数だけ）・D1-easy（課題 2..30 の偶数だけ）を、その種類の窓が同定されたときに同じ規則で出す。
- **D2 UPSTREAM_DOWN**: 課題 2–30 の U の総和（unit 和・全更新）が 5/5 負 → UPSTREAM_DOWN、5/5 正 → UPSTREAM_UP、他 → UNRESOLVED。
- **D3 EASY_SINKS（事後の読みの登録）**: seed ごとに [easy 課題 2, 4, …, 30 の 1 更新あたり Δm（unit 平均・課題等重み）] − [hard 課題 3, 5, …, 29 の同じ量]。**5/5 負 → EASY_SINKS**、5/5 正 → HARD_SINKS、他 → UNRESOLVED。副として S だけ・U だけの版を同じ規則で出す。J は含まない（課題の中の移動だけ）。
- **D4 SELF_NET_DOWN**: 課題 2–30 の S の総和が 5/5 負 → SELF_NET_DOWN、5/5 正 → SELF_NET_UP、他 → UNRESOLVED。
- **D5（副・記述）**: (a) seed ごとの境界占有率 = Σ_{t=2..30} [boundary の S の和] / Σ_{t=2..30} [課題全体の S の和]（両方 unit 和、分母 0 は未定義）が 0.5 を超える seed の数（A2 の「境界が正味のほぼ全部」の型。D1 と同じく両方の窓が要る。片方だけ同定なら同定された種類の課題だけで計算して明示する）。(b) 課題 2–30 の S_conf 総和の符号（5/5 負か）。(c) 課題 1（hard 1・初期値から）の最初の 78 更新（bin 1–3）の S の和（erosion_race の「切替の衝撃」に当たる量）。
- **REPORT_ONLY**: J の課題別・切替の向き別（hard→easy／easy→hard）の和、S・U・Δm の課題別・bin 別・正負別、第 1 層 S1/J1、S < 0 の件数（頻度）、S_label・S_hist、課題末の m2・G2。

完全性: 欠落 seed/課題/bin/必須列・偽の完了 marker は **INCOMPLETE**、非有限は **DIVERGED**、bit 不一致・閉包違反は **CHECK_FAILED**。どれも全科学判定を抑止する。主 seed を減らさない。

## 7. 適用条件（判定が意味を持つための条件・本走後に検査）

- A1 宿主同一性: 本走の per_task.csv が、同じ環境で無改変宿主を seed 0–9 で再走した per_task.csv と全列バイト一致し、かつ記録 `results/cifar5p1_mlp_0920/R_std_lr0.0001/per_task.csv` と eff_rank_l1/eff_rank_l2 以外の全列でバイト一致する。fresh_control.csv は記録とバイト一致する。eff_rank の差は件数と最大差を報告する（§1 の環境差）。
- A2 完全性: 10 slot × 30 課題 × 780 更新の全観測、全 shard の sha256、宿主行 300・fresh 10。
- A3 数値: 本走の全観測で §9 の閉包・勾配・Adam 再現・成分・label EMA・望遠鏡和の違反 0、非有限 0。
- A4 窓: D1 と D5(a) は両方の窓の同定を要する（§6）。

## 8. 予測（本走の前に記録。結果を見てから変えない）

親（Fable）の事前確率は A2 の結果（NOT_SUPPORTED・BOUNDARY_ENRICHED・U 総和 5/5 負・S 総和 5/5 負）を知ったうえでのもの。Claude の確率は本 spec の起案者のもので、同じ情報と §1 の箱の既知（課題末の列）を見たうえでのもの。互いを転記しない。

| 項目 | ラベル | 親（Fable） | Claude |
|---|---|---:|---:|
| 窓 hard | WINDOW_IDENTIFIED | .65 | .75 |
| 窓 easy | WINDOW_IDENTIFIED | .55 | .70 |
| D1（判定された場合） | BOUNDARY_ENRICHED / UNRESOLVED / LATER_ENRICHED | .60 / .30 / .10 | .75 / .20 / .05 |
| D2 | UPSTREAM_DOWN（Claude: / UNRESOLVED / UPSTREAM_UP） | .60 | .60 / .30 / .10 |
| D3 | EASY_SINKS（Claude: / UNRESOLVED / HARD_SINKS） | .70 | .45 / .30 / .25 |
| D4 | SELF_NET_DOWN（Claude: / UNRESOLVED / SELF_NET_UP） | .55 | .70 / .25 / .05 |
| D5(a) | 境界占有率 > .5 が 5/5 | .40 | .50 |
| D5(b) | S_conf 総和 5/5 負 | — | .80 |

Claude の根拠（短く）: 切替直後は旧クラスを支える unit を押し下げる向きが標本間でそろう（新クラスの W3 行は非目標として押し下げられてきた）ので境界の S は強く負、Adam の v（β2 の記憶 ≈ 1000 更新）が境界の大きな勾配を覚えて後続の歩幅を縮める → 窓は同定され D1 は境界濃縮。U は第 1 層の出力の伸び × 負の相対位置で負に寄る（z̄ l2 → −8.8）。D3 は両向きの切替がどちらも強い押し下げを持ち、hard は後続も学習が続いて U を通じた沈みが続きうるので、課題末の dead の差（集合の均質さが混ざる）ほどには easy に傾けない。

採点: D1 は 3 値の multiclass Brier（判定されなかった場合は理由付きで未採点）、他は主ラベルの binary Brier（Claude の 3 値分布は副として multiclass Brier も出す）。窓は同定の成否で binary Brier（必ず採点）。検査失敗・不完全なら全項目を理由付き未採点。

## 9. 検査（登録前に列挙・実装 commit の前に実行し、本物で pass・列挙した変異で fail）

collector（`analysis/drive_cifar5p1_1007/checks.py`）は期待する ID と変異の一覧、実 source の sha256 を持ち、未実行・空・別 source の PASS を拒否する。本走の CLI は checks.json の source hash と全 PASS を確かめてからでないと走らない。許容は算術から導く（γ_n(u) = n·u/(1 − n·u)、演算数は下記）。bit 一致の検査は許容 0。

| ID | 本物で確かめること | 必ず落とす変異 |
|---|---|---|
| S-host | (a) 無改変宿主 seed 0–9 の現環境再走が記録と eff_rank 以外の全列・fresh でバイト一致（環境資格）。(b) 観測付きエンジン（graph・seed 100–109・30 課題 + fresh）の per_task.csv・fresh_control.csv が同じ環境の無改変宿主とバイト一致、全 30 課題末の P（6 テンソル × 10 slot）が bit 一致、batch 行列が一致、観測の前後で torch の CPU/CUDA 大域 RNG 状態が不変。(c) 本走で §7 A1 | 観測が c51_batch を 1 回引く／観測が b2 に 2⁻²⁰ を足す／Adam の bias 補正を外す（学習側の変異で比較の感度を示す）。R5 は REPORT_ONLY |
| S-self-total | (a) 検査走の固定 fixture（全 slot、課題 2 の更新 1・780、課題 3 の更新 1、課題 30 の更新 780）で、保存した float32 の P_old/P_new と計画から作り直した画像集合から、CPU の float64・math.fsum で S・U・Δm・J・S1 を独立に再計算し、GPU 値と導出上界の中で一致。(b) 全観測更新で代数的閉包と native 閉包の違反 0。(c) 課題ごとの望遠鏡和（Σ_j Δm = 課題末 − 先頭、先頭 − 前課題末 = J）。(d) 本走でも同じ位置の fixture を保存し、本走後に (a) の独立再計算を行う（違反は CHECK_FAILED） | U を省く（Δm := S）／U に W_old を使う／別課題（前課題）の画像の µ／課題の最初の更新で J を U に混ぜる |
| S-bin | bin = 26 更新 × 30、hard/easy とも 780 更新、bin ごとの観測更新数 26・unit 件数 2,600、bin の和 = 課題の和 | 27 更新の bin／bin 番号の off-by-one（j//26） |
| S-conf | (a) 保存 logit の float64 autograd による独立 CE 微分で conf + label = ∂CE/∂z3、再構成した第 2 層勾配と native 勾配の差が γ_{2C+B+16}(u32) の上界内（C = 100、B = 32）、eager 再計算の勾配から m・v・P の更新を bit 再現。(b) 残差 M_label と直接積算の label EMA が累積上界内。(c) S_conf + S_label + S_hist と実 S の差が実パラメータ丸めの上界内 | conf/label の入れ替え／history なし（M_hist := 0）／π = 1/10／batch 平均を二重に掛ける／conf moment を毎更新リセット |
| S-window | 合成系列: 一定 → 未同定、先頭負の 2 水準 → 正しい b で同定、逆向き（先頭が正または後続より高い）→ 未同定、RSS の同率 → 小さい b、RSS1 = 0 → 2 水準。較正は較正 seed の shard だけを開く（主 seed の shard を壊しても窓が同じ）。hard/easy を別に較正 | 主 seed の混入／b を固定（例 3）／同率で大きい b／逆向きを同定 |
| S-verdict | 3⁵ の符号パタン全列挙で 5/5 だけがラベル、tie・全 0 は UNRESOLVED、NaN は DIVERGED、長さ ≠ 5 は INCOMPLETE、窓未定義で D1 は NOT_JUDGED | 4/5 を検出／unit を標本に数える／seed 欠落を通す |
| S-graph | 観測付きエンジンの graph と eager（seed 100–109・課題 1–3）で各課題末の P/m/v/カウンタと観測配列が bit 一致。捕獲時の warmup 3 回を全巻き戻し（捕獲後の状態 = 初期状態） | warmup の巻き戻しを省く／replay 前に batch 行を更新しない |
| S-manifest | report が run_id・seed・課題・source・入力・shard の sha256・完了 marker を照合 | shard 欠落／別 run_id／偽の marker（checksum 不一致）／source hash 不一致 |
| S-cost | 検査走（30 課題・観測あり）の時間・最大 CUDA メモリ・最大 RSS・書き込み量。観測した更新数 = 23,400、画像数 hard 2,500・easy 500 の記録 | batch だけ観測（N = 32）／1 更新おきに観測 |

上界の導出（実装 commit の検査コードで確定し、本走の残差に合わせて広げない）:
- µ の float64 平均: γ_{N+2}(u64)·mean|h| + tiny64（h ≥ 0 なので mean|h| = µ）。
- native z2 の画像平均: 要素ごとに float32 の内積 K = 100 + bias 加算 + alpha/beta で γ_{K+3}(u32)·(|w|·|h| + |b|) + (K+3)·tiny32、画像平均は線形なので |w|·µ + |b| に置き換え、さらに float64 平均の γ_{N+2}(u64)·mean|z2|。
- float64 の内積・差: γ_{2(K+1)+2}(u64)·Σ|a||b| + tiny64、µ の誤差は Σ|w|·µe で伝播。代数的閉包は float64 丸めだけ（µ の誤差は恒等式に入らない）、native 閉包は µ の誤差と native の誤差を足す。
- Adam の実パラメータ差と float64 の理想更新: 7 回の float32 演算（m·c1、lr·、v·c2、sqrt、+eps、÷、p −）で γ_7(u32)·(|ideal| + |p_new|) + 7·tiny32（lr・eps は fl32 に丸めた値で理想を作る）。
- 勾配の再構成: log_softmax の和 C、δ3·W3 の和 C、batch の和 B、exp/log/除算/減算の余裕 16 で γ_{2C+B+16}(u32)·(1/B)Σ_s [Σ_c (p_sc + 1/C)|W3_ci| + |W3_{y_s,i}|]·1[z2 ≥ 0]·|(a1, 1)| + (2C+B+16)·tiny32（p·|log p| ≤ 1/e の丸めの増幅を 1/C 項で覆う）。
- label EMA の累積: E_s = b1f·E_{s−1} + a1f·gb_s + γ_2(u32)·(b1f|m_old| + a1f|g|) + 2·tiny32 + γ_4(u64)·(|M_conf| + |M_hist| + |M_ldir| + |m_new| + |M_label|)。

## 10. 実行計画・資源

登録 commit → 実装（`src/drive_cifar5p1_1007.py`、`analysis/drive_cifar5p1_1007/{checks.py, report.py, launch.sh}`）→ 検査（seed 100–109 と無改変宿主 seed 0–9 だけ）→ checks.json を含めて実装 commit → 本走（`launch.sh`、seed 0–9、30 課題 + fresh）→ 完全性と A1 の確認 → 較正 seed だけで `window_calibration.json` を作って commit → 主 report・予測採点・summary.md → 結果 commit → git 外の生データを `~/Projects/obsidian-research-data/drive_cifar5p1_1007/` へ退避して backup_manifest.json を commit → main へ merge・push。worktree と branch は消さない（親が確認してから）。

GPU プロセスは同時 1 本まで（`/tmp/lop_analysis_gpu.lock` を flock で待つ）。GPU の空きメモリを 6 GB 以上残す。観測は 1 更新ごとに全画像の第 1 層 forward（hard で約 15 GFLOP）を足すので宿主の 20 秒より重い。本走前に S-cost で実測する。生の shard は `results/drive_cifar5p1_1007/raw/`（git 外）、主 seed と較正 seed を別ファイル（`primary/`・`calibration/`）に書く。走行中の表示は宿主の行（online・test）だけで、S/U/Δm の値は出さない。

## 11. 解釈の範囲

観測であって原因の除去実験ではない。S・U・Δm は「現在の課題の画像での平均前活性」の変化であり、別の画像集合（次の課題・テスト）での変化ではない。J を分けたので、課題末の dead/z̄ の課題種の差（集合の均質さが混ざる）とは別の量である。conf/label の分解は共通の実分母での形式的な分解で、寄与率にしない。R/std・lr 1e−4・seed 0–9・30 課題の観測で、他の腕・他の箱・長時間極限へ外挿しない。5 seed の 5/5 規則は検出力が低い（各 seed の効果が同符号でも 1 seed の揺れで UNRESOLVED になる）ことを結果に残す。

## 12. 出力・片付け

`results/drive_cifar5p1_1007/`: checks.json、window_calibration.json、verdict.json、per_seed.csv、per_task.csv（観測の課題別）、per_bin.npz（seed × 課題 × bin の unit 和）、per_unit_task.npz、host_identity.json、predictions.json、summary.md、production の provenance/input_manifest/complete/cost、backup_manifest.json。raw（shard・fixture・ログ・検査走の出力）は git 外で、CLAUDE.md §4 どおり退避し sha256 付きで manifest にする。既存の src・analysis・他 run の結果・他の worktree は変更しない。vault には書かない。
