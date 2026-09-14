# Snake の位相を MNIST へ移す（箱 B 主・Random Label 予備走・SGD 橋）spec 0914

状態: **非登録の設計草案（登録版は specs/spec_snake_phase_mnist_0914.md。食い違いは登録版が優先）**・旧状態 草案・未登録（実装前・走る前・Issa 未確認・Issa の予測欄は未記入・repo にも vault にも未 commit）/ 更新: 2026-09-14 / 起草: Claude（設計 3 案と批評 9 本を統合、指摘 41 件を検証して改訂。末尾「草案の検証記録」）
親: [[W増大メカニズム_0909]] §5.4（活性化による ‖W̃ᵢ‖ 成長の違い）/ [[零点への復元はW増大を抑えるか_検証設計_0913]] §4C 第 3 段階（引き取る。逸脱は §14.3 に登録）
姉妹: [[零点復元と重み収縮_学習実験_spec_0913]]（CondA 側）/ repo `wt/wcap_rlmnist_0914/specs/spec_wcap_rlmnist_0914.md`（W病理チャット・Random Label・走行中）
発端: Issa（2026-09-14）「CondA で snake の議論がうまくできないので、これらの活性化を RandomLabel とか Permuted の MNIST でやるべきだよね」
run id: `snake_phase_mnist_0914`（worktree `wt/snake_phase_mnist_0914`・branch `claude/snake_phase_mnist_0914`）

> 数値の等級: **登録** = results/*/summary.md・verdict.csv から写したもの。**事後** = 本セッション（2026-09-14）で既存ログから計算したもの（未登録）。**推論** = 算術・推論。
> 事後の計算スクリプト: scratchpad `synth_check.py`・`synth_pair.py`（本 spec の起草時）、批評時の `feas_timing*.py`・`s3real.py`・`statlens/`、改訂時の `verify_rev1.py`（実 c0 での成長比・三分位比・置換相関・位相振幅）・`verify_rev2.py`（χ²・t 分位・n_main・交互作用の検出力）・`verify_rev3.py`（分解能マージン h_E・単腕 D の SD）。

---

## 0. 問い（1 文）

**Snake 自身が時間劣化を示す MNIST の箱で、CondA の Snake 3 位相（normal / peak / valley）と定数オフセットは、時間劣化・fresh gap（水準とは分けて測る）と第 1 層の個体別中心化ノルム ‖W̃ᵢ‖ の成長を変えるか。変えるなら、それは「初期関数（起点）」「根の位置と活性値の水準（オフセット）」「位相の構造そのもの」のどれから来るか。**

この問いを 4 つに分けて答える。
- (a) 固定 α=0.6（箱 B で Snake の劣化が最大・bit 錨あり）での位相と、その厳密な分解（起点 × オフセット）。答え方は §6.6 の対応表で登録する。
- (b) 適応 α（SNA 型）での位相。後期まで位相の効き目が残る唯一の族。初期関数を揃えた角（SNAi_P・SNAi_V）を置き、「位相の構造」を起点から切り離す。
- (c) CondA 移植族（§6.7、無条件の登録ラベル）: CondA で見えた peak の成長抑制・群除去の偏り・根への着座を、LoP のある箱で ‖W̃ᵢ‖ について答える。CondA の自由重み u に当たる量は MNIST に無い（§17）。
- (d) 予備: Random Label MNIST は Snake の議論の器になるか（400 epoch・既存ランナー無改変）。CondA の個体の分岐は SGD 固有か（SGD 橋・α=1）。

**確認的な見出し（headline）は E1（時間劣化の対差、§6.1）のラベルだけ**にする。E2・E3・§6.4 の総合クラス・CT 族・M ゲート・FS 族は副次で、族をまたぐ多重性は制御しない（§6.1 末尾）。

## 1. なぜ CondA ではだめか

- **Snake が課題を解き切り、劣化が無い。** CondA 後期（task451–500）の MSE は normal q0 4.09e−11・peak q0 6.41e−9・valley q0 2.93e−11。early（task2–20）は 4.17e−5・3.55e−4・8.93e−5 で、3 腕とも late < early（登録の副量、`zero_attraction_analysis_0913/results/zero_attraction_learning_0913/summary.md` の早期・後期損失表）。比べるべき時間劣化が存在しない。
- 同じ穴は 0903–0904 にもあった。合成箱で α≥1 は EXACT_FIT（U≈1e−12）になり、逃れたのは α=0.5 だけ（[[前活性の力学_事後_0904]] §8.1）。
- 推論: CondA の学習器は幅 100 で、元の Bit-Flipping 問題（CBP 2021）の学習器（隠れ 5 ユニット）の 20 倍ある。解き切りは Snake の性質ではなく容量の効果かもしれない。
- CondA で見えた現象（peak の自由重み成長 −9.4%、比 .906・CI .894–.917、`verdict.csv` log 差 −0.09877・CI [−0.11186, −0.08658]、DIRECTIONAL_PHASE_SUPPRESSION／peak 70.8% の個体が初期値より縮む／群除去）は **SGD・MSE・32 点支持**のもの。LoP の有無と結びつけて検定できない。

## 2. 箱の選択と理由

**主: Permuted MNIST 箱 B**（gate_shape_0911 と同一: 784–100–100–10・CE・batch 16・10,000 枚層化抽出 = 625 更新/タスク・手書き Adam lr 1e−3・β .9/.999・eps 1e−8・moment とステップ数はタスクをまたいで持ち越し・正則化なし・t1–120・CPU float32・1 スレッド・決定的アルゴリズム・torch 2.13.0+cu130・white-san のみ）。

理由:
1. **固定 α=0.6 の Snake が LIN より明確に劣化する。**
   - 登録（`results/gate_shape_0911/summary.md` §1。L = タスク終端 acc の中央値 t16–20 − 中央値 t101–120・seed 中央値）: SN06 **2.74**、LIN 0.63、LR 2.28、SNA 0.71、SN02 0.80、SN15 1.05。
   - seed 別 L（同 rows.csv から再計算・登録と一致）: SN06 2.74/2.89/2.25、LIN 0.70/0.63/−0.01、SNA 0.71/0.70/1.32、LR 2.53/2.28/2.07。
   - 事後（窓平均 D = mean t16–30 − mean t101–120）: SN06 2.42/2.26/2.14、LIN 0.23/0.36/0.17。seed 対の D_pair(SN06−LIN) は 2.20/1.90/1.97。
2. **劣化は幅に媒介されている。** SN06 の wclamp で R_c 0.02・ρ_w 0.98（登録、同 summary §1・窓 t61–100）。
3. **安い。** gate_shape_0911 の 13 腕 × 3 seed の 120 タスク走は **52–124 s**（provenance wall_seconds・2–8 並列下）。seed 1–2 は 52–101 s。seed 0 は 57–124 s で、錨を持つ SN02・SN06・SN15・SNA・LIN の seed 0 は、最初の錨タスクまで G1 と測定の対照を追加で回すため 81–124 s（非 seed 0 の約 1.5 倍）。SN06 は 64–96 s。事後: P コアで 0.53 s/タスク、E コアで 1.32 s/タスク。
4. **bit 錨がある。** gate_shape_0911 の SN06・SNA・LR・LIN（s0–2・per-unit 16 配列 × 120 タスク・git 追跡）。複製ループが t1–3 で 16 配列すべて max|diff| = 0.0、P コアと E コアとも一致（事後）。

**Random Label MNIST は予備走だけにする（段階 R）。**
- 400 epoch・50 タスクで SNA は天井に張り付いて平坦。登録: online_acc t31–50 = 0.9859 ± 0.0003、memo_acc 1.0000（`results/shell_l2_rlmnist_0913/summary.md`）。事後（origin/main の per_task.csv、SNA none、online_acc）: t11–20 0.9857 対 t41–50 0.9861、後期傾き（t31–50 の OLS）+0.0003/10 タスク（段階 R の定義 t11–50 では +0.0001/10 タスク）、どちらの窓でも 10 seed 中 0 が負。
- 固定 α の Snake は RL で一度も走っておらず、**反証ではなく未検証**。CondA の失敗を繰り返すかどうかを、既存ランナー無改変で先に確かめる。
- RL は W病理チャットの箱（wcap_rlmnist_0914 が 02:07 から走行中）なので、主にはしない。

**α の選び方。**
- 固定 α=0.6 を主にする。一方で、α=0.6 では **後期窓までに位相の効き目が洗い流される**。事後（gate_shape_0911 units、中央値ᵢ exp(−2α²sdcurᵢ²)、seed 0/1/2）:
  - SN06: t1 0.64/0.65/0.67 → t20 0.43/0.43/0.40 → t60 0.25 → t120 0.09/0.10/0.09。|ḡᵢ−1| の中央値は t120 で 0.087/0.068/0.077。
  - SN15 は t20 以降 0.00 だが、それでも L 1.05（登録）なので、固定 α の Snake の劣化に位相構造は要らない。
  - SNA は |ḡᵢ−1| 中央値が t120 でも 0.308/0.282/0.315（2αᵢWᵢ≈1.2 に固定されるため）。
- したがって (a) の後期窓の対比は「位相」ではなく「起点とオフセット」の効果として読む。位相構造の問いは (b) 適応 α 族で、初期関数を揃えた角との対比として問う（§6.1 F2・§6.5 M1）。
- α=1（CondA の値）は箱 B に錨も登録の劣化もないので、SGD 橋でだけ使う。SGD 箱の SN1 は L 1.43 ± 0.24 対 LIN 0.23 ± 0.21（seed 中央値 ± SD・10 seed・事後・未追跡 `results/pmnist_adapt_0905/boxB`。この boxB は **200 タスク・CUDA** の走で、値はその t1–120 部分から計算した）。

**19 条件の扱い。**
- 移すもの: Snake q0 × 3 位相。
- Snake q=±.5 の 6 条件: 恒等式（§3.2）により「バイアスの平行移動 + オフセット」の点にすぎない。代わりに分解腕で K 軸を 2 水準（枝 k=0, −1）だけ取る。
- Leaky: a=.1 q0（錨・陽性対照）、a=.1 に K_P と同じオフセット（q=−2.142）と、符号を反転したオフセット（q=+2.142）。Leaky ではオフセットはバイアスの平行移動と等価ではない（leaky(z+s) は leaky(z)+K と一致しない）ので、Snake の恒等式による ±.5 の省略は Leaky には効かない。そこで符号の 2 水準だけを Adam 箱に置く。大きさ |q|=.5 と a=.3/.7 への依存は Adam では検定しない（§17）。CondA で分岐が最大だった a=.7 q−.5 は SGD 橋へ。
- linear: 床として移す。

## 3. 活性化の定義

### 3.1 式（隠れ層 2 層とも同じ活性化・宿主の慣例）

位相族: ψ_θ(z) = z + [cos θ − cos(2αz + θ)]/(2α)、ψ_θ′(z) = 1 + sin(2αz + θ)。

| 名 | θ | φ(z) | φ′(z) | φ(0) | φ′(0) | 実装（float32） |
|---|---|---|---|---|---|---|
| normal | 0 | z + sin²(αz)/α | 1 + sin(2αz) | 0 | 1 | 宿主の式そのまま `z + torch.sin(a * z) ** 2 / a`（a は Python float）。`+0.0` を足さない |
| peak | +π/2 | z + sin(2αz)/(2α) | 1 + cos(2αz) | 0（sin 0 = 0 で厳密） | **2**（cos 0 = 1 で厳密） | 閉形式 `z + torch.sin(2.0*a*z) / (2.0*a)`。sin²(αz+π/4) は使わない（float32 で sin²(π/4) ≠ 0.5） |
| valley | −π/2 | z − sin(2αz)/(2α) | 1 − cos(2αz) ≥ 0 | 0 | **0**（二重零点。原点近傍で φ ≈ (2/3)α²z³、φ′ ≈ 2α²z²） | 閉形式 |
| offset Snake | — | ψ_0(z) + q | ψ_0′ | q | 1 | q ≠ 0 のときだけ `+ q` |
| offset leaky | — | where(z>0, z, a·z) + q | where(z>0, 1, a) | q | — | 宿主 LR の式に、q ≠ 0 のときだけ `+ q` |
| 適応位相 | 0/±π/2 | 上の式の α を αᵢ(t) に置換（層・ユニット別） | 同 | 0 | 1/2/0（どの αᵢ でも） | **`GS.H.AdaptiveSnake`（= `src.pmnist_boundary_host_0908.AdaptiveSnake`）を継承**する。箱 B のループ（GS/C/G/B 経由）の isinstance 分岐は GS.H のクラスを見るので、V 更新と層引数はこの継承でだけ通る。`src.pmnist_0905.AdaptiveSnake` はファイルがバイト一致（sha 53e2c102）でも別のモジュールオブジェクトで、isinstance が互いに通らない（事後に確認: `GS.H is pmnist_0905` → False、`isinstance(pmnist_0905.AdaptiveSnake(...), GS.H.AdaptiveSnake)` → False）。そちらから継承すると V 更新も層引数も黙って落ちる。peak は `z + sin(2 aᵢ z) * (0.5 * aᵢ.reciprocal())`。αᵢ = clip(0.6/√Vᵢ, .05, 3)、V は β=.01 の EMA で初期値 1（t=0 で αᵢ=0.6） |

CondA の式（`origin/codex/zero-attraction-0913:src/nets.py` の SNAKE_PHASE_0913、commit 89521b3）は **値として写す**。import しない。docstring に出所を書く。

### 3.2 恒等式（設計の土台・S4 で検査）

固定 α で、すべての整数 k について

**ψ_θ(z) = ψ_0(z + s_θ + kπ/α) + K_θ − kπ/α**、s_θ = θ/(2α)、K_θ = (cos θ − 1 − θ)/(2α)。

- 事後の検算（float64、z∈[−15,15]、20 万点）: k=0, ±1 で max 誤差 3.6e−15〜7.1e−15（`synth_check.py`）。
- α=0.6 の値（表示は 5–6 桁。float64 で式から算出し、丸めた K に π/α を足すことはしない）:
  - peak: (s, K) = (+1.30900, −2.14233)。k=−1 では (−3.92699, +3.09366)。
  - valley: (−1.30900, +0.47566)。k=−1 では (−6.54498, +5.71165)。
  - π/α = 5.23599。
  - 検算: ψ₀(1.309) = 2.14233、ψ₀(−3.92699) = −3.09366、ψ₀(−1.309) = −0.47566。
  - **config の 17 桁は、この表から写さない。** s = θ/(2α) + kπ/α、K = (cos θ − 1 − θ)/(2α) − kπ/α を float64 で評価して生成し、生成に使った式の文字列を config に並べて置く。
- α=1 の値: peak (0.78540, −1.28540)、valley (−0.78540, +0.28540)。
- 学習可能な bias があり、Adam と SGD は bias の平行移動に対して不変。したがって **固定 α の位相腕は、「bias 初期値の平行移動 s」と「一定のオフセット K」を持つ normal Snake と、力学系として同一**になる。
- **ただし (s, K) の組は一意ではない（枝 k）。** 「peak の固有オフセットは K_P」とは言えない。
  - 枝に依らない量: 根の位置がゲート最大（peak）・ゲート零（valley）にあること。
  - 枝で変わる量: 活性値の水準 K。
  - これを分ける 1 点検査として P06c_k1 を置く（§4）。
- 根と傾き（normal 座標 u）:
  - normal + K_P の根は u = +s_P で、傾き 2（ψ_0(1.309) = 1.309 + 0.5/0.6 = 2.1423）。
  - normal + (K_P + π/α) の根は u = −3.927 で、傾き 2。
  - normal + K_V の根は u = s_V = −1.309 で、傾き 0（ψ_0(−1.309) = −1.309 + 0.8333 = −0.4757）。
  - LR + K_P の根は z = +2.142 で、傾き 1。

### 3.3 初期出力の交絡とその扱い

位相は初期の関数を変える。初期ゲートは層ごとに違う（推論）。
- **層 1**: W1 ~ U(±1/28)、b1 ~ U(±1/28)、MNIST/255 の ‖x‖² ≈ 88 なので、前活性の分散は (88+1)/2352 で sd ≈ 0.19（約 0.2）。
  - N06 ≈ 1。
  - P06 ≈ 2（層あたり利得 2）。
  - V06 ≈ 2α²E[z²] = 0.72 × 0.04 ≈ 0.03。
- **層 2**: 入力は a1（100 ユニット）、W2・b2 ~ U(±0.1)。
  - N06: 前活性 sd ≈ 0.13（100 × (0.01/3) × E[a1²] + 0.01/3 から）。ゲート ≈ 1。
  - P06: ゲート ≈ 2。
  - V06: a1 ≈ (2/3)α²z³ ≈ 0 なので z2 ≈ b2（sd ≈ 0.058）。ゲート ≈ 0.72 × 0.0033 ≈ **0.002**。
- V06 は前向きの信号がほぼ死んだ状態で始まる。事後の測定では、V06 の初期 |g_W1| 中央値が 1.3–1.6e−7（批評時 `s3real.py`）。

扱い:
1. **初期関数を揃えた角（init map）。** 初期関数が N06 と等しい P06c / V06c / P06c_k1、初期関数が P06（V06）と等しい P06i（V06i）を置く。起点の効果とオフセットの効果を、同じ初期関数どうしの対比で分ける（§4 の 2×2）。
   - 適応族にも同じ型の角を置く: **SNAi_P / SNAi_V** = 適応 normal（宿主 AdaptiveSnake(.6,.01)）に to_phase(s, K)（α=.6 の値）を掛けたもの。宿主の AdaptiveSnake は V=1 から始まるので t=0 で αᵢ=0.6 になり、SNAi_P の初期関数は SNAP（= P06）と実数で等しい。V の更新は var(z) だけを使い、var は平行移動で不変なので、1 更新目の V も両者で実数で等しい。t>0 で αᵢ が動くと恒等式は崩れる（s・K は α に依存する）。この崩れこそが「位相の構造」の効果になる。
2. **腕の中の時間差を主にする**（E1）。水準差は腕の中で打ち消される。水準は E2 に分けて出す。
3. **base 窓を t16–30 にする**（初期の立ち上がりを除く）。起点の過渡が base 窓に残るかは M3 で 3 状態（CLEARED／TRANSIENT_IN_BASE／TRANSIENT_UNRESOLVED）に判定し、CLEARED のときだけ S 因子を base 窓込みで読む。
4. **fresh gap はその腕自身の θ₀ から取る**（手法ごとの定義）。fresh の水準差が有意なら FRESH_LEVEL_CONFOUNDED を付け、読みを初期関数を揃えた角に移す。
5. step 0 の CE、logit RMS、A(t1)、task 1 の ce0→ce20 を水準の共変量として記録する。出力を補正する介入（固定の出力補正）は本走に混ぜない（設計ノート 0913）。

init map（H.init_params(seed) の直後に float64 で計算し、1 回だけ float32 に落とす。W は全腕共通）:
- comp(q): b2 ← b2 − q·(W2·1)、b3 ← b3 − q·(W3·1)。
- to_phase(s, K): b1 ← b1 + s、b2 ← b2 + s + K·(W2·1)、b3 ← b3 + K·(W3·1)。
- twin(s): b1 ← b1 + s、b2 ← b2 + s（補償なし。normal + K で位相腕と同じ関数を別の演算経路で作る）。

## 4. 腕の表

### 4.1 段階 3（本走・箱 B Adam）

固定 α=0.6 の 2×2（normal 座標で読む）:

| | 水準 0（根 u=0・傾き 1） | 水準 K（根がゲート最大／ゲート零） |
|---|---|---|
| 起点 N06（初期関数 = normal） | **N06** | **P06c**（K_P=−2.142・根 u=+1.309・傾き 2）／**P06c_k1**（+3.094・根 u=−3.927・傾き 2）／**V06c**（K_V=+0.476・根 u=−1.309・傾き 0） |
| 起点 = 位相腕の初期関数 | **P06i** ／ **V06i** | **P06** ≡ P06tw ／ **V06** ≡ V06tw |

| 腕 | 活性化 | init map | 初期関数 | 目的 | seed |
|---|---|---|---|---|---|
| N06 | normal α=.6 | なし | N06 | 基準。G1 錨（gate_shape_0911 SN06） | n_main |
| P06 | peak 閉形式 | なし | P06 | 位相の総効果（CondA peak q0 の移植） | n_main |
| V06 | valley 閉形式 | なし | V06 | 同（CondA valley q0）。死んだ起点 | n_main |
| P06c | offset Snake q=K_P | comp(K_P) | N06 | K 因子（起点は N06） | n_main |
| P06c_k1 | offset Snake q=K_P+π/α=+3.09366 | comp(q) | N06 | 枝の検査（根の幾何は同じで水準の符号と大きさが違う） | n_main |
| P06i | normal（q=0） | to_phase(s_P, K_P) | P06 | S 因子（水準 0） | n_main |
| V06c | offset Snake q=K_V | comp(K_V) | N06 | K 因子（valley） | n_main |
| V06i | normal | to_phase(s_V, K_V) | V06 | S 因子（valley） | n_main |
| P06tw | offset Snake q=K_P | twin(s_P) | P06 | 実装の対照（P06 と実数で同一の軌道。float の分岐の大きさ） | 0–9 |
| V06tw | offset Snake q=K_V | twin(s_V) | V06 | 同 | 0–9 |
| SNA | 適応 normal（宿主 AdaptiveSnake(.6,.01)） | なし | N06（実数で） | 適応族の基準。G1 錨（gate_shape_0911 SNA） | n_main |
| SNAP | 適応 peak | なし | P06（t=0 で αᵢ=.6） | 後期まで位相の効き目が残る族での位相（総効果） | n_main |
| SNAV | 適応 valley | なし | V06 | 同。原点のゲート零は z/Wᵢ 座標で不変 | n_main |
| SNAi_P | 適応 normal | to_phase(s_P, K_P)（α=.6） | P06 | 適応族の起点の角。SNAP − SNAi_P が同じ初期関数での位相構造の効果 | n_main |
| SNAi_V | 適応 normal | to_phase(s_V, K_V)（α=.6） | V06 | 同（valley） | n_main |
| LIN | 恒等 | なし | — | 劣化の床。G1 錨 | n_main |
| LR | leaky a=.1 | なし | — | 陽性対照・G1 錨 | n_main |
| LR_qKp | leaky a=.1 + q=K_P（−2.14233） | comp(K_P) | LR | 単調ゲートでのオフセット（K 効果は Snake 特有か） | n_main |
| LR_qKpn | leaky a=.1 + q=−K_P（+2.14233） | comp(−K_P) | LR | 同・符号反転（Leaky ではオフセットがバイアス移動と等価でないため符号を分ける） | n_main |

**19 腕**: 17 腕 × n_main seed + 2 腕（P06tw・V06tw）× 10 seed。

### 4.2 段階 R（Random Label 予備走）

| 腕 | 活性化 | ランナー |
|---|---|---|
| N06-RL / P06-RL / V06-RL | §3.1（初期関数は揃えない。予備走なので総効果だけを見る） | `src/pmnist_rlmnist_0906.py`（origin/main・**ファイルは無改変**）を、宿主 `src.pmnist_0905.ARMS` に PhaseSnake を差し込むラッパーから呼ぶ。1200 枚・400 epoch = 30,000 更新/タスク・50 タスク・Adam 1e−3・**CPU**・seed 0–9 |

ラッパーの呼び出しは次のとおり固定する（無改変の main() の既定は `--device auto` で white-san では CUDA を選び、出力先の既定は origin/main で追跡されている `results/pmnist_rlmnist_0906/<arm>` の中になるため）。
- **1 プロセス = 1 腕 × 1 seed**: `--arms <arm> --seeds <s> --device cpu --out results/snake_phase_mnist_0914/rl/<arm>_s<s>`。無改変の main は per_task.csv を (arm, seed) の 1 走（14–20 分）が終わってから書くので、途中で殺した seed はタスク単位では残らない。**殺した seed は丸ごと走り直す**と宣言する。
- import 前に `OMP_NUM_THREADS=MKL_NUM_THREADS=1`、import 後に `torch.set_num_threads(1)`（H.setup はスレッド数を設定しない）。
- 無改変の main が書く provenance.json は run_id が `pmnist_rlmnist_0906` で活性化の定義を含まない。ラッパーが隣に `sidecar_provenance.json`（run id `snake_phase_mnist_0914/rl`、acts の sha256、PhaseSnake の θ・α、ラッパーの sha256、ランナーと宿主の blob sha、device、スレッド数、git_hash）を書く。
- **ユニット別 ‖W̃ᵢ‖ の記録**: 無改変ランナーが記録するのは非中心化の行ノルムの中央値（`w_norm_*`）だけなので、ラッパーが `RL.evaluate_rl` を包み（shell_l2_rlmnist_0913 の S4 と同じ手法。run_one はモジュール大域の名前で呼ぶので包みが効く）、タスク末ごとに層 1・2 の cnormᵢ・mᵢ・bᵢ を float64 の複製から読んで `units.npz` に書く。読むだけで params に触れないことを S18 で検査する。

陽性対照（LR）: **自前で LR × 10 seed を同じラッパーで走らせ、予算に数える**（§12.3）。wcap_rlmnist_0914 の LR_ref は、次の両方が満たされたときだけ代わりに使ってよい。
- wcap の判定（verdict）が commit・push されている。それまで wcap の per_task.csv を読まない。
- 本 spec の S18 で、wcap のランナー（`src/wcap_rlmnist_0914.py`、wcap の git_head の blob）の LR ref と無改変の 0906 `run_one` が、CPU・seed 0・1 タスク × 2 epoch で bit 一致する。既存の検査はこの一致を直接は示していない: wcap S4 は LR ref を 0913 ランナー（shell_l2_rlmnist_0913）と照合し、shell_l2 S4 は 0913 ランナー（none）と 0906 run_one を R と SNA だけで照合している。連鎖には LR の経路が抜けている。

### 4.3 段階 S（SGD 橋・Issa の判断 §16-1 待ち）

SGD 箱のプロトコル: optimizer・lr・epoch・再歩行は pmnist_adapt_0905 boxB と同一（plain SGD lr .02、1 タスク = 同じ 625 batch 順を 4 周 = 2,500 更新）。**タスク数（本件 120、boxB 200）とデバイス（本件 CPU、boxB CUDA）は異なり、boxB とは bit の錨を持たない**（boxB の provenance: optimizer sgd・lrs [0.02]・epochs_per_task 4・steps_per_task 2500・n_tasks 200・device cuda）。宿主・ストリーム・測定は箱 B と同じ。

| 腕 | 活性化 | 目的 |
|---|---|---|
| N1 | normal α=1（宿主 SN1） | CondA の α での基準 |
| P1 / V1 | peak・valley α=1 | CondA の位相を CE MNIST の SGD で見る |
| LIN-S | 恒等 | CondA で分岐最大（92.7% 収縮）。床 |
| LR07q-S | leaky a=.7 + q=−.5、comp(q) | CondA の leaky で分岐最大（82.6%） |

**5 腕 × seed 0–9**。CondA の値（LIN 92.7%・LR a=.7 q−.5 82.6%・peak q0 70.8%・normal q0 27.1%）は、`zero_attraction_analysis_0913/results/zero_attraction_learning_0913/group_summary.csv` の **ALL 行の shrink_count/n の seed 平均**を事後に集計したもの。縮みの判定は許容幅なしの ‖u(end)‖² < ‖u(0)‖²（`zero_attraction_report_0913.py:105–108` の `n2[-1] < n2[0]`）で、**eps に依存しない**（eps = .05/.1/.2 は原点近傍群 |μ_final − z0| ≤ eps を選ぶ半径で、ALL 行の割合は 3 つの eps ですべて同じ）。

### 4.4 後続実験 M（媒介・別名の実験として別 spec）

§11 に概要だけ登録する。

## 5. seed・タスク・予算

- 箱 B・SGD 箱: 120 タスク。1 タスク終端ごとに test 1 万枚の精度と per-unit 測定。
- **n_main**（段階 3 の seed 数）は、段階 2 の較正で算術により決める（§7.2）。範囲は 10〜30。事後の見込みは 14–31 と幅が広い（使う対差の池で変わる）ので、予算は上限 30 で立てる。
  - seed 0–2 は錨の seed。seed 3 以降は同じ role 付きストリームから新たに引く。
  - 腕は RNG に入らない（sha256('pmnist_0905|role|seed')）ので、すべての対比は seed 対でタスク列が同一。
- pilot seed は 100–104（SGD の安定性だけに使う）。endpoint には入れない。
- 追加ストリーム（arm 名を含めない）: `sp0914_probe_perm|data|batch`（PD 用の未見 8 置換。120 タスクのどの置換とも異なることを S8 で検査）、`sp0914_cap_perm`、`sp0914_cap_reinit`、`sp0914_group`（RAND 群と CT2 の無作為群）。
- fresh probe は t1（自己検査）、t16–20、t101–120 の 26 本。逐次タスクと byte 同一の (perm, idx, order) を使い、625 更新（SGD 箱は 2,500 更新）。
- snapshot（C.snapshot: params・Adam m/v/t・SNA V・generator 状態）を t20・t60・t120 で torch.save し、sha256 を provenance に記録する。t0 は H.init_params と init map から再計算できる。
- 予算は全腕で同じ: 625 更新/タスク、lr は調整しない（Kumar の選択交絡を避ける）。

## 6. 主 endpoint と判定

### 6.1 E1 時間劣化（主・唯一の確認的な見出し）: 同一タスク列の対差

acc は正答数/10000。計算は整数で行う（同点処理のため）。seed k、腕 θ、基準 ref:

**D_pair,k(θ, ref) = mean_{t∈16..30}[acc_θ(t) − acc_ref(t)] − mean_{t∈101..120}[acc_θ(t) − acc_ref(t)]**（pt）

正なら θ の方が多く劣化している。

族（族内で Holm、α=.05 両側）:

| 族 | 対比 | m |
|---|---|---|
| **F1** 固定α（基準 N06） | C1 P06−N06（位相の総効果）、C2 V06−N06、C3 P06c−N06（K 因子・peak）、C4 P06i−N06（S 因子・peak）、C5 V06c−N06、C6 V06i−N06 | 6 |
| **F1x** 分解 | I_P = (P06−P06i) − (P06c−N06)、I_V（同）、B_P = P06c_k1 − P06c（枝） | 3 |
| **F2** 適応位相（基準 SNA と起点の角） | C7 SNAP−SNA（総効果）、C8 SNAV−SNA、**C7s SNAP−SNAi_P**（同じ初期関数での位相構造）、**C8s SNAV−SNAi_V** | 4 |
| **F3** 単調ゲート（基準 LR） | C9 LR_qKp − LR、C9n LR_qKpn − LR | 2 |

- **F1x の交互作用 I は加法性の判定にだけ使う**。ラベルは `NON_ADDITIVE`（Holm で有意）／`ADDITIVE`（TOST: (1 − 2·.05/3) 反転 CI ⊂ [−h_D, +h_D]）／`ADDITIVE_INCONCLUSIVE`（それ以外）。事後の算術（正規近似）: 対差の SD .27 を腕ごとの独立成分に割ると I の SD ≈ .38 で、n=20 では反転 CI の半幅 ≈ .20、真の I=0 でも ADDITIVE になる確率は δ=.24 のときですら約 .40（両位相とも約 .16）、n=30 で約 .77。h_D ≈ .19 ではさらに低い。**したがって n_main は I の等価性に検出力を持たせない**と宣言し、ADDITIVE が出ないことを前提に読み方を決める（§6.6）。
- 主効果 S_P = ½[(P06i−N06) + (P06−P06c)] と K_P = ½[(P06c−N06) + (P06−P06i)]（valley も同様）は、**I のラベルに関係なく**反転 95% CI 付きで出す（REPORT・Holm なし）。I が ADDITIVE でないとき、主効果は「2 つの起点で平均した効果」としてだけ書き、単純効果（C3・C4・C5・C6）と並べる。
- **双子の軌道の検査**（実装の零対比。どの族にも入れない）: T_P = P06 − P06tw、T_V = V06 − V06tw の D の差は REPORT（n=10 で検出力が低く、判定に使わない）。判定は初期の軌道で行う。
  - δ_tw = maxᵢ |cnormᵢ^P06 − cnormᵢ^P06tw| / medianᵢ cnormᵢ^P06 を **task 1 の 20 更新目**のパラメータで測る（層 1・2）。20 更新目なら浮動小数の分岐はまだ小さく、O(1) の実装差とは桁が分かれる見込み。
  - 下の参照（浮動小数の経路差の実測）: δ_ref = max(δ_ulp, δ_path)。δ_ulp は同じ腕の W1[0,0] を 1 ulp 動かした走、δ_path は P06 を数学的に同一の別の式（`z + torch.sin(2.0*a*z) * (0.5 / a)`）で走らせた走との差。段階 0 に seed 0–2 で測る。
  - 上の参照（実装差の実測）: δ_noK = 補償 K を落とした双子（P06tw_noK。初期 logits が O(1) ずれる）との差。段階 0 に seed 0–2 で測る。
  - `IDENTITY_TRAJECTORY_BREAK`: どれかの seed で log δ_tw > ½[log max_seed δ_ref + log min_seed δ_noK]（2 つの実測の参照の対数の中点。固定倍率は使わない）。発火したら段階 3 の前に実装を調べる（§11 段階 2 の停止規則）。
  - 識別できない場合: min_seed δ_noK ≤ max_seed δ_ref なら、この検査は `TWIN_CHECK_NOT_IDENTIFIABLE` として記録し、停止には使わない（S23 で宣言どおりに分岐することを検査）。V06/V06tw も同じ（V06tw_noK）。

検定と CI:
- **正確な符号反転検定**（n ≤ 20 は 2ⁿ を全列挙、n > 20 は乱数 10⁶ 回・rng `default_rng([20260914, 族番号, 対比番号])`・p = (b+1)/(M+1)）。
- **CI は符号反転検定を位置ずれについて反転**したもの（percentile bootstrap は n=10–30 で被覆不足のため使わない）。

ラベル（上から順に判定し、最初に当てはまるものを採る）:
1. `INCOMPLETE`: 完全な seed 対の数 n_c < n_min(m)（§7.3）。
2. `NOT_TESTABLE_REF_FLAT`: F1・F1x で TESTABLE_FIXED が不成立、F3 で TESTABLE_LEAKY が不成立。
3. `MORE_DECLINE` / `LESS_DECLINE`: Holm で有意、符号で決める。(1 − 2·.05/m) 反転 CI が [−h_D, +h_D] に収まれば接尾 `_WITHIN_RESOLUTION`（接尾は基のラベルを変えない。§6.4 の総合クラスでは基のラベル MORE/LESS として扱い、クラス名に同じ接尾を引き継ぐ）。
4. `EQUIVALENT`: TOST。(1 − 2·.05/m) 反転 CI ⊂ [−h_D, +h_D]（h_D は §7.1 の分解能マージン）。**意味は「単走のタスク標本の分解能 h_D より小さい」であって、「実用上効果が無い」ではない**。結果ノートでもそう書く。
5. `INCONCLUSIVE`: それ以外。反転 CI の半幅を分解能として必ず併記する。

修飾（ラベルは変えない）:
- `WINDOW_SENSITIVE`: 登録互換の L（中央値 t16–20 − 中央値 t101–120）で作った対差と D_pair の、seed 別差の反転 95% CI が 0 を含まない。符号が同じでも発火する。
- `TRANSIENT_IN_BASE` / `TRANSIENT_UNRESOLVED`: M3（§6.5）。S 因子の対比（C4・C6、F1x の I）で、M3 が CLEARED でないときに付ける。
- `REF_FLAT`: F2 で D_pair(SNA, LIN) の CI が 0 を含む。F2 の基準は平坦でもよい（位相腕が劣化を作ること自体が答え）が、明示する。
- `UNANCHORED`: G1 が 120 タスクで崩れた。登録値との比較は止めるが、内部の対比ラベルは出す。
- `BROKEN_k/n`・`DIVERGED_k/n`: §6.5。
- `FRESH_LEVEL_CONFOUNDED`: §6.3。
- `UNDERPOWERED_FOR_RESOLUTION`: §7.2 の式が 30 を超えた。

**書き方の規則**:
- F1 の後期窓の効果は、M1 が PHASE_ACTIVE でない限り「起点」「根の位置・活性値の水準」の効果として書く。
- 「位相構造が LoP を変えた」と書けるのは、**C7s または C8s**（初期関数を揃えた対比）が MORE/LESS で、かつ M1 がその対で t16–30 と t101–120 の両方 PHASE_ACTIVE のときだけ。C7・C8（総効果）だけが有意なら「適応族で位相腕は（起点を含めて）劣化を変えた」とだけ書く。
- 固定 α の起点の対比（C4・C6）が EQUIVALENT でも INCONCLUSIVE でも、C7s・C8s の読みは変えない（適応族の起点は SNAi で直接揃えている）。

**多重性と見出し**:
- 確認的な見出しは **E1 のラベル**（F1・F2・F3 の各族で Holm）。族をまたぐ多重性（4 族 × E1–E3、M ゲート、CT 族、FS 族）は制御しないと宣言する。
- E2・E3・§6.4 の総合クラス・§6.6 の答え・§6.7 の CT 族は副次。見出しの主張（「位相／起点／オフセットが時間劣化を変えた」）は E1 のラベルからだけ書き、副次のラベルだけを根拠に見出しを書かない。

### 6.2 E2 水準（E1 と分ける）

- A_late = mean acc t101–120、A_base = mean acc t16–30。同じ族・同じ検定で ΔA_late・ΔA_base に Holm を掛け、`HIGHER` / `LOWER` / `EQUIVALENT` / `INCONCLUSIVE` を付ける。EQUIVALENT は TOST で (1 − 2·.05/m) 反転 CI ⊂ [−h_A, +h_A]（h_A は窓ごとの分解能マージン h_Alate・h_Abase、§7.1）。
- 併記（ラベルなし）: A(t1)、全タスク平均、step 0 の CE、設計ノートの窓（task1 / 2–20 / 21–40 / 41–100 / 101–119 / 120）。

### 6.3 E3 fresh gap（副次・main axis の LoPGap。見出しは E1 だけ、§6.1 末尾）

- A_fresh,θ(t): θ 自身の θ₀（H.init_params(seed) に init map を適用）から、**新しい Adam**（m=v=0・step 0・SNA は V=1）で、逐次タスク t と byte 同一の (perm, idx, order) を 625 更新学習した後の test 精度。
- **Gap_late(θ) = mean_{t∈101..120}[A_fresh,θ(t) − A_seq,θ(t)]**。ΔGap_late = Gap_late(θ) − Gap_late(ref) を同じ族で Holm 検定し、`GAP_LARGER` / `GAP_SMALLER` / `EQUIVALENT`（CI ⊂ [−h_Gap, +h_Gap]、§7.1）/ `INCONCLUSIVE` を付ける。
- 修飾 `FRESH_LEVEL_CONFOUNDED`: ΔA_fresh（t101–120 の A_fresh 平均）が Holm で有意。この場合、V06・P06 の読みは初期関数を揃えた角（V06c・P06c）に移す。
- Gap_early（t16–20 の 5 本）は記述だけ。
- 注: 1 本目（t1）の fresh は逐次の task 1 と同じ計算なので、S6 の自己検査に使う。

### 6.4 E1×E2×E3 の総合クラス

**上から順に判定し、最初に当てはまるものを採る**（クラスは互いに排他になる）。「E1 MORE_DECLINE」は接尾 `_WITHIN_RESOLUTION` 付きも含み、その場合クラス名に接尾を引き継ぐ。前提として E1 が INCOMPLETE・NOT_TESTABLE の対比はクラスを付けない。

| 順 | クラス | 条件 |
|---|---|---|
| 1 | `CONFLICT` | E1 が有意（MORE/LESS）で、E3 が逆向きに有意（MORE_DECLINE と GAP_SMALLER、または LESS_DECLINE と GAP_LARGER）。ΔA_late が E1 と逆向き（例: MORE_DECLINE と HIGHER）なのは「高く始まって多く落ちた」で矛盾ではないので、ここには入れない |
| 2 | `TEMPORAL_LOP_MORE` | E1 MORE_DECLINE、かつ（ΔA_late LOWER または GAP_LARGER） |
| 3 | `TEMPORAL_LOP_LESS` | E1 LESS_DECLINE、かつ（ΔA_late HIGHER または GAP_SMALLER） |
| 4 | `BASE_SHIFT` | E1 有意、ΔA_base が E1 の符号を生む向きに有意、ΔA_late と E3 はともに有意でない |
| 5 | `NO_DIFFERENCE_RESOLVED` | E1 EQUIVALENT、ΔA_late EQUIVALENT、E3 EQUIVALENT |
| 6 | `LEVEL_ONLY` | E1 EQUIVALENT、ΔA_late 有意 |
| 7 | `LEVEL_DIFF_DECLINE_INCONCLUSIVE` | E1 INCONCLUSIVE、ΔA_late 有意 |
| 8 | `INCONCLUSIVE` | それ以外 |

S14 の合成シャードで 8 クラスすべてと、2 つのクラスの条件を同時に満たす組（例: MORE_DECLINE・LOWER・GAP_SMALLER → CONFLICT）が期待どおりになることを検査する。

### 6.5 操作確認とゲート（E1 のラベルを読む前に評価する）

**M1 位相の効き目 PHASE_ACTIVE**（層別・窓 t16–30 と t101–120）

- 各ユニットの位相の梃子を Aᵢ = |mean_x exp(i·2αᵢ zᵢ(x))| とする（probe 512 枚・現置換。Gaussian 近似は使わない）。
- 任意の位相 θ, θ′ で |ḡᵢ^θ − ḡᵢ^θ′| ≤ 2|sin((θ−θ′)/2)|·Aᵢ が成り立つ（ḡᵢ − 1 = Im(e^{iθ} E[e^{i2αz}]) からの算術）。
- seed ごとの統計量 Φ = √2 · medianᵢ Aᵢ(θ 腕) − IQRᵢ(ḡᵢ^ref)（窓内のタスク平均。peak 対 valley なら √2 を 2 に）。
- 判定: 反転 95% CI > 0 なら `PHASE_ACTIVE`、< 0 なら `PHASE_WASHED`、それ以外は `PHASE_BORDERLINE`。
- 意味: 位相を動かしても、ユニットの平均ゲートが基準腕自身のユニット間の散らばりより動かせないなら、位相は基準の分布の外へ出られない。

**M2 平均活性の座り直し SEATING**（オフセット角 P06c・P06c_k1・V06c 対 N06、LR_qKp・LR_qKpn 対 LR。層別・窓 t1・t16–30・t101–120）

- r_a = [meanᵢ āᵢ^θ − meanᵢ āᵢ^ref] / q、āᵢ = mean_probe φ(zᵢ)（オフセット込み）。
- r_a = 1: 前活性が動かず、オフセットが活性値に残る。r_a = 0: 前活性が座り直して、平均活性が基準の水準に戻る。
- 反転 95% CI で判定:
  - `MEAN_RESEATED`: CI が 0 を含み 1 を含まない。
  - `NO_RESEATING`: CI が 1 を含み 0 を含まない。
  - `PARTIAL`: CI ⊂ (0,1)。
  - `OVERSHOOT`: CI が区間 [0,1] の外。
  - `INCONCLUSIVE`: CI が 0 と 1 をともに含む。
- 点ごとの根への近さ |α(zcurᵢ − z₀)| は記述だけ。帯の占有率は LIN で飽和するので判定に使わない（事後: 箱 B で |zcur| ≤ sdcur の割合は LIN 0.99）。

**M3 起点の過渡**（S 因子の対: P06 対 P06c、P06i 対 N06、V06 対 V06c、V06i 対 N06）

- Δu(t) = medianᵢ uᵢ^θ − medianᵢ uᵢ^ref。u は normal 座標の zcur で、peak の閉形式腕は +s、valley は −s を足す。P06i・V06i は normal の経路なのでそのまま。
- 初期値は Δu(0) = ±1.309。
- 統計量: seed ごとの e_k = mean_{16–30} Δu − mean_{101–120} Δu と、その反転 95% CI。
- **マージン Δu_m（算術と実測）**: normal 座標で前活性を Δu だけ動かしたときの平均ゲートの変化は、ユニットごとに |ḡᵢ(z+Δu) − ḡᵢ(z)| = |Im[(e^{i2αΔu} − 1)·e^{iθ}E e^{i2αz}]| ≤ 2|sin(αΔu)|·Aᵢ。これが基準腕のユニット間の散らばり IQRᵢ(ḡᵢ^ref) を超えない最大の Δu を Δu_m とする: Δu_m = arcsin(min(1, IQR/(2·medianᵢ Aᵢ^ref)))/α。seed ごとに基準腕の t16–30 の値から計算し、seed の中央値を使う（M1 と同じ比較の型）。
- 判定（3 状態）:
  - `CLEARED`: e の反転 95% CI ⊂ [−Δu_m, +Δu_m]（base 窓に残る起点のずれは、ゲートを基準の散らばりより動かせない）。
  - `TRANSIENT_IN_BASE`: CI が 0 を含まない。
  - `TRANSIENT_UNRESOLVED`: それ以外（「有意でない」を「過渡が無い」と読まない）。
- 読み方: S 因子の対比（C4・C6）の E1 を base 窓込みで読むのは **CLEARED のときだけ**。TRANSIENT_IN_BASE と TRANSIENT_UNRESOLVED では、S 因子の効果は E2 の ΔA_late と E3 だけで読む（base 窓を使わない）。
- 事後の目安（SN06 s0 の zcur 中央値。腕間の Δu ではなく単腕の軌道）: t1 +0.11、t2 −0.35、t5 −0.58、t10 −0.73、t20 −0.68、t120 −1.02。0 から自身の吸引点までに 5–10 タスクかかるが、t20 から t120 にもまだ動く。

**可検定性**
- `TESTABLE_FIXED`: D_pair(N06, LIN) の反転 95% CI_lo > 0。事後の見込みは 2.20/1.90/1.97。
- `TESTABLE_LEAKY`: D_pair(LR, LIN) の反転 95% CI_lo > 0。事後の見込みは 1.78/1.57/1.77。

**壊れた走**
- `BROKEN`（run 単位）: t101–120 で ce20 < ce0 のタスク数が正確二項の片側 .05 を超えない（< 15/20）。ce0 はタスク開始前、ce20 は 20 更新後の probe CE で、gate_shape_0911 の rows には無いのでループに足す。
- 腕ラベル `BROKEN_k/n`。壊れた seed は対比から落とし、INCOMPLETE 規則を当てる。
- 注: この規則が検出するのは全壊だけ。部分的な学習不全は A_late と A_fresh の水準として出す。

**発散**: 非有限の loss またはパラメータで `DIVERGED(task, step)`。その run は欠測にする（try/finally で直前のタスクまでの出力を書いてから止める）。

### 6.6 §0 の問いへの答え方（登録の対応表・副次）

位相ごと（peak・valley）に、上から順に最初に当てはまる答えを採る。入力は E1 のラベル（C1–C6、C7s/C8s）、F1x の I と B_P、M1、M3。

| 順 | 答え | 条件 |
|---|---|---|
| 1 | `NOT_ANSWERABLE` | C1 または C2 が INCOMPLETE・NOT_TESTABLE、または IDENTITY_TRAJECTORY_BREAK が未解決 |
| 2 | `NON_ADDITIVE` | その位相の I が NON_ADDITIVE。起点とオフセットに分けて書かず、単純効果（C3/C5 = 起点 N06 でのオフセット、C4/C6 = 水準 0 での起点）を並べて書く |
| 3 | `BOTH` | 単純効果の K（C3/C5）と S（C4/C6）がともに MORE/LESS。S は M3 が CLEARED のとき E1 で、そうでなければ ΔA_late と E3 で判定 |
| 4 | `LEVEL` | K が MORE/LESS、S が EQUIVALENT（M3 CLEARED）または S の ΔA_late と E3 がともに EQUIVALENT |
| 5 | `ORIGIN` | S が MORE/LESS（上と同じ窓の規則）、K が EQUIVALENT |
| 6 | `NONE` | 総効果（C1/C2）・K・S がすべて EQUIVALENT |
| 7 | `INCONCLUSIVE` | それ以外（I が ADDITIVE_INCONCLUSIVE のときも、単純効果が 3–6 を満たせばそれを採る） |

- `LEVEL` の細分（peak だけ）: B_P が EQUIVALENT なら `LEVEL_ROOT_GEOMETRY`（枝に依らない根の位置で決まる）、MORE/LESS なら `LEVEL_ACTIVATION_VALUE`（活性値の水準と符号で決まる）、それ以外は `LEVEL`。
- **位相の構造**（適応族）: `STRUCTURE` = C7s（または C8s）が MORE/LESS、かつ M1 がその対で両窓 PHASE_ACTIVE。`STRUCTURE_NOT_SHOWN` = C7s（C8s）が EQUIVALENT。それ以外は `STRUCTURE_INCONCLUSIVE`。固定 α の答えとは独立に出す。
- 固定 α の答えに M1 が PHASE_WASHED（後期）なら、答えの文言は「起点」「根の位置と活性値の水準」に限る（§15-1）。
- S14 の合成シャードで表のすべての行と、I が ADDITIVE_INCONCLUSIVE で単純効果が BOTH を満たす組を検査する。

### 6.7 CondA 移植族 CT（無条件・副次の登録ラベル）

CondA で見えた現象が LoP のある箱にあるかを、**E1・E3 の有意性に関係なく**答える。対象の対比は C1（P06−N06）・C2（V06−N06）・C7（SNAP−SNA）・C8（SNAV−SNA）。n_min(4) = 8。

**CT1 成長（CondA の見出し「peak は normal より成長が小さい」に対応）**
- logN_ℓ = ½ log meanᵢ,t∈101..120 cnormᵢ²（層 ℓ = 1, 2）。全腕で W の初期値が共通（init map は bias だけを動かす）なので、腕の差 ΔlogN_ℓ = logN_ℓ(θ) − logN_ℓ(ref) は成長比の対数の差と同じ。
- 検定: seed 対の正確な符号反転検定、層ごとに 4 対比で Holm。
- ラベル（層ごと、順に判定）: `GROWTH_SUPPRESSED`（Holm で有意・負）／`GROWTH_ENHANCED`（有意・正）／`GROWTH_BELOW_CONDA_SCALE`（TOST: (1 − 2·.05/4) 反転 CI ⊂ [−0.09877, +0.09877]）／`INCONCLUSIVE`。有意なラベルにも、CI がこの幅に収まれば接尾 `_BELOW_CONDA_SCALE` を付ける。
- マージンの出所: CondA の登録値（`verdict.csv`）の log 差 −0.09877（peak q0 対 normal q0、task451–500）。意味は「CondA で見えた効果より小さい」。u（自由重み）と ‖W̃ᵢ‖ は別の量なので、同じ大きさを期待する根拠ではなく、比較の物差しとして使う（§15-3）。
- t16–30 の同じ量と、固定 α の分解の対比（C3–C6）の ΔlogN は REPORT。
- 注: 成長の差は W 病理の証拠ではない（腕間の比較。§14.1）。

**CT2 群除去の偏り（CondA「peak の育った群を除くと定数予測器に落ちる」に対応）**
- 対象: N06・P06・V06・SNA・SNAP・SNAV の t120 snapshot、層 1（層 2 と t20 は REPORT）。§8-3 の論理補償つき除去。
- 群: HIGH・LOW（log cnormᵢ(0) で残差化した cnormᵢ(t120) の上位・下位 33）と、`sp0914_group` から引く **無作為な 33 ユニット群 B=64 個**（その最初の 2 つが RAND1・RAND2）。除去は評価だけ（test 10,000 枚の前向き計算）。
- 量: eₖ(X) = ΔCEₖ(X) − meanⱼ ΔCEₖ(Rⱼ)（無作為群の除去に対する超過）。帯 b_θ = 無作為群どうしの ΔCE の差 |ΔCEₖ(Rⱼ) − ΔCEₖ(Rⱼ′)| の、seed と 32 組の対にわたる中央値（その腕で測った散らばり）。
- ラベル（腕ごと、X ∈ {HIGH, LOW}、腕内で m=2 の Holm）: `CARRIES_MORE_THAN_RANDOM`（有意・正）／`CARRIES_LESS_THAN_RANDOM`（有意・負）／`WITHIN_RANDOM_BAND`（(1 − 2·.05/2) 反転 CI ⊂ [−b_θ, +b_θ]）／`INCONCLUSIVE`。
- 位相の比較（P06 対 N06 の e(HIGH) の差など）は REPORT。

**CT3 根への着座（CondA「中心群」に対応。相対の比較だけ）**
- sₖ(θ) = medianᵢ |zcurᵢ − z₀| / sdcurᵢ（層 1、probe 512 枚・現置換、t101–120 のタスク平均）。q0 の腕はどれも根が z₀ = 0 に 1 つだけ（φ′ ≥ 0 の単調関数、normal は z ∈ [−1/α, 0] に他の根が無いことを算術で確認）。
- 対比 C1・C2・C7・C8 の Δs に Holm（m=4）。ラベル: `CLOSER_TO_ROOT`（有意・負）／`FARTHER_FROM_ROOT`（有意・正）／`SEATING_UNRESOLVED`。**等価ラベルは置かない**（導出できるマージンが無いため）。
- 絶対の着座（「根に吸い寄せられている」）は判定しない。LIN を零にすると帯の占有率が飽和し（事後 0.99）、位相を一様にずらす零分布はゲートの位相への揃いを検定するだけで、根そのものへの着座と区別できない（§17）。

**CT4 分岐（Adam）**: §8-2 の量は REPORT。分岐のラベルは SGD 橋（段階 S の FORK_*）だけに置く。

**CondA の効果の分類（設計ノート 0913 §6）との対応**（§14.3 の逸脱 8）:
- 「初期から実際に縮む」→ Adam では f_shrink（REPORT・算術上ほぼ自明）、SGD では FORK_*。
- 「初期からは育つが対照より成長が小さい」→ CT1 の GROWTH_SUPPRESSED。
- 「平均だけ零点へ寄り、幅は育つ」→ CT3 の CLOSER_TO_ROOT かつ CT1 が GROWTH_SUPPRESSED でない。
- 「読み出し v が消えて参加しなくなる」→ 出力側の列ノルムが初期値より小さいユニットの割合（REPORT）と CT2。
- 「全勾配までは縮むが optimizer 後に反転する」→ 本 spec では測らない（1 更新の診断を落としたため。逸脱 10）。
- 「形状間・課題間で符号が割れる」→ CT1–CT3 の C1/C2/C7/C8 と SGD 橋の符号の並び。

## 7. 閾値の導出

### 7.1 分解能マージン h_E（旧 δ_D を置き換える）

旧草案の δ_D = 0.24 pt は、leaky の腕内の ω 則（1.05 pt/倍）を Snake の α 梯子の**腕間の**幅の範囲に当てて作っていた。この箱の登録結果は腕間で幅が劣化を並べないこと（gate_shape_0911 A6: Spearman(N, L) = +0.01、`WIDTH_NOT_BETWEEN_ARMS`）を示しており、Snake 族の中では符号まで逆になる（SN02 は N 6.76 と SN06 の 5.78 より広いのに L 0.80 対 2.74 と劣化が小さい）。ω 則の傾きの族依存も揺れている（現在地の 9/12 の記録: tau_clamp_0912 では leaky −1.06 対 ELU −1.59、band_omega_0912 では 625 更新固定なら −1.06 対 −1.00、turn_budget_0912 で ω 則は更新数固定の範囲でしか成り立たない）。傾きの選び方だけで旧 δ_D は 0.24–0.36 pt に動く。**腕間の幅を前提にしたマージンは使わない。**

代わりに、科学的な最小効果量（SESOI）の意味を捨て、**単走の分解能**をマージンにする。
- 定義: 端点 E ∈ {D, A_late, A_base, Gap_late} ごとに、1 seed の対差系列 d(t) = acc_θ(t) − acc_ref(t)（Gap は A_fresh − A_seq の対差）について、窓ごとにタスクの線形傾向を除いた残差の分散からタスク標本の SE を出す。D なら SE_D = √(var(res_{16–30})/15 + var(res_{101–120})/20)（ddof 2）、A_late なら √(var(res_{101–120})/20)。
- **h_E = 段階 2 の盲検の対（S19 の対）× seed でプールした SE の二乗平均の平方根**。点推定をそのまま使い、上側限界は取らない（マージンを広げる向きの調整をしないため）。
- 意味: 「腕の差が、1 本の走のタスク標本のゆらぎで分解できる大きさより小さい」。結果ノートで EQUIVALENT を「効果なし」と書かない。
- 事後の見込み（gate_shape_0911 rows、N06→SN06 の読み替え、SN06−LIN と LR−LIN × seed 0–2、`verify_rev3.py`）: h_D ≈ 0.19（0.18–0.20）、h_Alate ≈ 0.11（0.09–0.13）、h_Abase ≈ 0.16（0.13–0.18）。h_Gap は既存ログに fresh が無いので段階 2 で初めて出る。
- 参考: 事後の SN06 の過剰劣化 D_pair(SN06−LIN) の平均は 2.02 pt。

### 7.2 n_main（段階 2 で決め、addendum を commit してから段階 3）

- n_main = n の最小値で、n ≥ ((t_{n−1, 1−.05/12} + t_{n−1, .8}) · σ̂_U / h_D)² を満たすもの（F1 の Holm 第 1 段で、真の差 h_D を検出力 80% で検出）。範囲は [10, 30] に丸める。
- 式が 30 を超えたら n_main = 30 とし、全ラベルに `UNDERPOWERED_FOR_RESOLUTION` を付ける。
- F1x の I の等価性には検出力を持たせない（§6.1。I の SD は単純対比の約 √2 倍）。
- σ̂_U: 段階 2 の 3 つの池の最大値。池 1・2 は 80% 片側上側信頼限界 √(df·s²/χ²_{df, .20})。
  - 池 1: 参照の対差のうち seed ごとに独立な 2 対比 {N06−LIN, LR−LIN}、各 10 seed（**df 18**）。N06−LR = (N06−LIN) − (LR−LIN) は線形従属なので池に入れない（入れると df 27 と数えて係数 √(27/χ²_{27,.20}) = 1.142 になり、正しい √(18/χ²_{18,.20}) = 1.183 より小さく出る）。
  - 池 2: 双子の対差 {P06−P06tw, V06−V06tw}、各 10 seed（df 18）。双子が非相関化しなければ 0 に近く、池 1 と 3 が上界を担う。
  - 池 3（効果に盲目な上界）: 位相腕が絡む対比は、単腕の D の seed 間 SD から sd(θ − ref) ≤ sd_θ + sd_ref で抑える。P06・V06（s0–9）と N06（s0–9）の単腕 SD だけを出し、平均は出力しない（S19 の AST 検査で、gate0 の出力が SD の欄だけであることを確かめる）。池 3 の値は max(sd_P06, sd_V06) + sd_N06（点推定。三角不等式がすでに上界なので信頼限界を重ねない）。
- 事後の見込み: gate_shape_0911 の 9 腕 36 対でプールした対差 SD は 0.24 pt（各 3 seed）、S19 型の 2 対（SN06−LIN、LR−LIN）だけなら 0.15・0.12、単腕 SD は SN06 0.14・LIN 0.10・LR 0.04（池 3 の型の上界 SN06−LIN で 0.24）。σ̂_U = 0.284（0.24 × 1.183）と h_D = 0.19 なら式は n = 31 で上限 30 に当たり、σ̂_U が 0.15 × 1.183 ≈ 0.18 なら n = 14（`verify_rev2.py` の nmain）。**見込みは 10–30 の全域にまたがる**ので、予算は n_main=30 で立てる（§12.3）。

### 7.3 その他の閾値

- **n_min(m)**: 最小の両側 p は 2/2ⁿ で、これが .05/m 以下でなければならない。よって n ≥ log₂(2m/.05)。m=6 → 8、m=4 → 8、m=3 → 7、m=2 → 7、m=1 → 6。
- **Holm 族の m は事前に固定**する。ゲートで対比が落ちても m は変えない。
- **二項 15/20**: P(X≥15 | 20, .5) = .0207 ≤ .05、P(X≥14) = .0577。
- **M1 の比較**: 位相で動かせるゲート量の算術上界（2|sin(Δθ/2)|·A）と、基準腕の測定済みの IQR を比べる。固定の定数は使わない。
- **M2 の錨 0 と 1**: 2 つの仮説が算術で予言する値。
- **M3 のマージン Δu_m**: 同じ算術上界（2|sin(αΔu)|·A）と基準腕の IQR から（§6.5）。
- **CT1 の ±0.09877**: CondA の登録の log 差（§6.7）。**CT2 の帯 b_θ**: その腕の無作為群どうしの ΔCE の差の中央値（実測）。
- **双子の検査の境界**: 2 つの実測の参照（浮動小数の経路差・K を落とした実装差）の対数の中点（§6.1）。
- **分岐（SGD 橋）の 1/100**: 100 ユニット層の分解能（1 seed あたり 1 ユニット）。
- **RL ゲート**: 0 と陽性対照 LR の測定分布から決める（§11 段階 R）。
- **段階 2 の停止規則の k**: 完全な seed 対の数が n_min(6)=8 を下回る確率から（§11 段階 2）。
- **メモリの余白**: 実測の RSS と、実測したデスクトップ側の MemAvailable の落ち込み（§12.2）。4 GiB の定数は使わない。
- **S 検査の許容**:
  - float64 の丸め伝播上界（Higham の γ_n × 項の和 × 層ごとの ‖W‖∞ と max|φ′|）を実行時に計算する。
  - float32 に丸めた init map は、bias ごとの半 ulp の伝播上界。
  - 変異の検出幅は、変異の解析的なずれ（例: K を落とすと |K|·|W3·1|）の 1/2 以上とし、許容の固定倍率は使わない。
- **並列数**: §12.2（実測の RSS から決める）。

## 8. 副次診断（ラベルなし・結果前に列挙。ラベルを持つものは §6.7 の CT 族と段階 S の FORK_* だけ）

1. **ユニットの運命**（毎タスク、両隠れ層）:
   - 記録する配列: cnormᵢ = ‖Wᵢ − mean(Wᵢ)‖、mᵢ、bᵢ、zbarᵢ・sdᵢ（8 参照置換、C.measure）、zcurᵢ・sdcurᵢ、āᵢ、ḡᵢ、Aᵢ、offᵢ = P(φ′<.25)、q10ᵢ、出力側の列ノルム（層 1: ‖W2[:,i]‖、層 2: ‖W3[:,i]‖。CondA の |v| の類似物）、SNA 族の αᵢ（両層）、t0 の値。
   - task 1 からの運命: 生の Spearman(cnormᵢ(t1), cnormᵢ(t120))。比 g = c(t)/c(0) どうしの相関は分母を共有して見かけの相関が出るので使わない（事後: c0 をランダムに置換して Spearman(c1/c0′, c120/c0′) を 20 回平均すると、腕と seed によって約 0.0–0.4 になる。SN06 .07/.12/.13、SNA .02/.13/.22、SN02 .06/.07/.03、LR .09/.05/.15、LIN .08/.30/.41。c0 は H.init_params(seed) の層 1 の中心化行ノルム、c1・c120 は gate_shape_0911 units。付録 A の `verify_rev1.py`）。
   - t120 の LOW / HIGH 三分位群（log cnormᵢ(0) で残差化した cnormᵢ(t120) で並べる）の後ろ向き軌道を t0/1/5/20/60/120 で追う。
2. **分岐の記述（Adam）**:
   - CondA と同じ f_shrink も出す。定義は CondA と同じ **strict な縮み ‖W̃ᵢ(end)‖² < ‖W̃ᵢ(0)‖²（許容幅なし）**のユニットの割合（CondA の `n2[-1] < n2[0]`。eps は使わない）。**Adam では算術上ほぼ自明**と明記する。事後: gate_shape_0911 の SN06/SNA/SN02/LR/LIN で、t1 終端の cnorm の最小値が初期値 √(1/3) の 1.41–1.70 倍。
   - 使う量:
     - 成長比の対数の seed 内 SD: 層 1 で log(cnormᵢ(t120)/cnormᵢ(0)) のユニット間 SD（ddof 1）。cnormᵢ(0) は **H.init_params(seed) から計算した各ユニットの実際の初期値**（共通の名目値 √(1/3) で割ると実質 SD(log c120) になり、定義と違う量になる）。腕の値は n_main seed の平均。事後（t120、seed 0–2、実 c0、`verify_rev1.py`）: SN06 .032/.032/.031、SNA .038/.033/.028、SN02 .029/.027/.032、SN15 .023/.025/.029、LR .041/.041/.039、LIN .037/.042/.042 → **Snake・LR・LIN で 0.023–0.042**、GELU .131–.136（**0.13**）、ELU1 .20–.28、ReLU .42–.54。
     - Sarle の双峰係数（一様分布の値 5/9 と比べる）。
     - 上位 10% のユニットが持つ Σcnorm² の割合を基準腕と比べる。
     - 出力側の列ノルムが初期値より小さいユニットの割合（CondA の peak で v→0 だったことの類似物）。
3. **論理補償つきの群除去**（t20・t120 の snapshot、全腕）:
   - 層 1 の群 G: z2′ = z2 − W2[:,G](a1_G − mean_x a1_G)（z2 の平均を厳密に保つ）。
   - 層 2 の群 G: logits′ = logits − W3[:,G](a2_G − mean_x a2_G)（logit の平均を厳密に保つ）。
   - 群: **LOW（33）・MID（34）・HIGH（33）が 100 ユニットを分割**する（log cnormᵢ(0) で残差化した cnormᵢ(t120) の順位）。RAND1・RAND2（各 33）は `sp0914_group` から引く互いに素な無作為群で、CT2 ではさらに無作為群を B=64 個まで引く（§6.7）。SGD 橋では CondA の符号群 S = {g<1}、G = {g≥1}（g = ‖W̃ᵢ(end)‖/‖W̃ᵢ(0)‖）も使う。これは CondA の strict な縮みと同じ定義で、CondA の eps（原点近傍群を選ぶ半径）とは関係がない。
   - 報告: 無傷との Δacc・ΔCE、クラス事前確率による予測器（test のクラス頻度の CE ≈ ln 10）との比較。層 1・t120 の HIGH/LOW の超過は CT2 のラベル（§6.7）、それ以外は REPORT。
4. **群の将来学習**（t120 のみ。N06・P06・V06・SNA・SNAP・SNAV・LR・LIN と SGD 橋の全腕。未見の 2 置換。**全変種で新しい Adam**（m=v=0, t=0）を使い、optimizer 状態の交絡を除く。SGD 箱は 2,500 更新）:
   - 変種:
     - FROZEN（W1・b1 を凍結）。
     - ALL。
     - X ∈ {LOW, HIGH, RAND1, RAND2}（層 1 では X の行だけ学習、上位層は学習）。
     - X_RE（X の行を `sp0914_cap_reinit` から再初期化し、出ていく W2 列は保つ）。
     - X_NR（HIGH と RAND1 だけ。W̃ を初期ノルムへ縮尺し、向き・行平均・bias は保つ）。
   - 利得 C(X) = [A(X,end) − A(X,0)] − [A(FROZEN,end) − A(FROZEN,0)]。
   - 記述だけ。帯は腕ごとの RAND1−RAND2 零分布。
   - 限定: Adam 下の自然な群間のノルム差は **約 5–8%**（事後: t120 で log c0 について残差化した cnorm の上位と下位の三分位の RMS ノルム比 − 1。LIN 4.6–5.0%、SN06 5.4–6.7%、SNA 5.8–7.3%、LR 7.3–8.1%。実 c0 は H.init_params(seed)、`verify_rev1.py`）と小さいので、差が出ないのは算術上の見込み。
5. **帳簿**（層 1・層 2）:
   - タスク単位で厳密に取る: タスク開始時と終了時の W̃ から N_s, N_e, D = ‖ΔW̃‖。
     - G = N_e² − N_s² = D² + 2⟨W̃_s, ΔW̃⟩。
     - c = −⟨W̃_s, ΔW̃⟩/(N_s·D)。N_s か D が float64 の床以下なら NaN。
     - ユニットの帳簿を先に合計してから集約する。
   - 1 更新ごとの Q = Σ‖ΔW̃‖² は、t16–20 と t101–120（25 タスク × 625 = 15,625 更新、SGD 箱は 62,500 更新）だけ取る → ρ = D²/Q。
     - **実装は簡素版に固定する**: 層 1・層 2 について、更新の直前と直後の W の差を行平均で中心化し、ユニット別の二乗和だけを float64 で積む（回転角・内積の分解・変位の累積は取らない）。
     - 費用の上界: `bench_ledger.py` の簡素版（層 1 だけで I・K・Q・変位を float64 で積む）が +0.57 ms/更新なので、Q だけの 2 層でも 2 × 0.57 ms を上界とし、15,625 更新で **P コア約 18 s**（E コア約 45 s）。重い版（elu_turn_0912 の計装、LR 135–138 s 対 gate_shape_0911 LR 60–62 s の差 75 s を LATE=(61,120) の 37,500 更新で割って約 2.0 ms/更新）は使わない。重い版なら層 1 だけで約 31 s になる。
     - 全タスクでは取らない。
   - 帳簿であって介入ではない。self/rest に分解しない。
6. **幅とゲートの記述（すべて REPORT・ラベルなし・腕間の比較）**（全対比。CI 付き・Holm なし）:
   - **これらは腕をまたぐ幅の比較であり、W 病理（腕の中で ‖W̃ᵢ‖ の成長が時間劣化の病理か）の証拠ではない**と、verdict.csv と結果ノートの両方に書く。この箱の登録結果は腕間で幅が劣化を並べないこと（gate_shape_0911 A6、Spearman(N, L) = +0.01）を示している。W の因果は後続実験 M の腕内の幅操作だけで問う。
   - 層 1 の幅: logN1 = ½ log meanᵢ,t101–120 cnormᵢ²（CT1 と同じ量）。
     - ω 残差 r_k = D_pair,k − 1.05·log₂(N1_θ,k/N1_ref,k) とその CI を出す。旧草案の OMEGA_ACCOUNTS / OMEGA_PARTIAL / OMEGA_UNRESOLVED のラベルは**付けない**（leaky の腕内の当てはめを Snake の腕間に当てる標本外の読みで、Snake 族の中では符号まで合わないため）。
     - log₂ω が当てはめの範囲内かを併記する。
   - 層 2 の幅 N2。沈下の利得 |m2ᵢ|·|Σⱼ a1ⱼ|（オフセットは層 2 の行平均チャネルに掛かる）。zbar2 のずれ。
   - 出力層 W3 の行平均と列ノルム（出力層であり主軸の変数ではない、と明記）。
   - ゲート: Ḡ・NL_x・Cov（層別）。
   - 旧草案の W_CONSISTENT / W_OPPOSITE / W_SILENT / W_EQUIVALENT のラベルは**付けない**。本 spec は「W に媒介される／されない」を書かない（§15-6）。
7. **PD（同一の未見タスクで学習力の時間差を見る）**:
   - `sp0914_probe_*` の 8 置換で、t20 と t120 の snapshot から（Adam 状態を持ち越して）625 更新、θ₀ から新しい Adam で 625 更新。
   - PD = meanₚ[A@t20 − A@t120]、LoPGap@t20・@t120。記述だけ。twin 腕では取らない。
   - 8 置換は 120 タスクのどの置換とも異なること（未見）を S8 で検査する。
8. **登録互換の量**: L（中央値 t16–20 − 中央値 t101–120）、D21 = mean t21–30 − mean t101–120、G = A(1) − A(t101–120)、後期傾き t61–120（pt/100 タスク）。
9. **φ′ の零点と根からの位相**: uᵢ = frac((2αzbarᵢ + θ + π/2)/2π)。零分布は各 seed の zbar の実際の散らばりに一様な位相ずらしを掛けて作る（一様円の零分布は使わない。SN02 の R≈0.9 は周期に比べて散らばりが小さいための自明な値でありうる）。
10. **資源**: タスクごとに ru_maxrss、/proc/self/status の VmRSS、CPU コア番号（os.sched_getaffinity と psutil の cpu_num）、経過秒。

## 9. S 検査（すべて変異対照つき。run を殺す assert は出力を書いた後）

| # | 検査 | 変異対照（検出されなければその検査は空虚として fail） |
|---|---|---|
| S1 | **G1 錨**: N06・LR・LIN・SNA の s0–2 を gate_shape_0911 の per-unit 16 配列 × 120 タスク（1,920 鍵、件数ガード付き）と照合し、acc は文字列一致。配列は 2 群に分ける。**(i) 学習状態の配列**（cnorm・m・bias）: **max\|diff\| = 0.0**。**(ii) float64 で集計した派生量**（zbar・sd・zcur・sdcur・gbar・gvar・off*・q10・hard・star）: 許容は Higham の γ_n × 項の和の実行時上界（追加した測定コードで float64 の集計の順序が変わりうるため。gate_shape_0911 自身の G1 でも、別の実験の錨と集計順が違う派生量には 2.2e−16〜1.8e−15 の差があった）。事後: SN06 s0 の t1–3 で P/E コアとも 16 配列 0.0。**G1 の commit の前に、4 腕すべての s0 × t1–3 で同じ照合を行い、その結果で P2 の確率を書く**。120 タスクで崩れたら、最初にずれたタスクで分ける（(i) が t1 でずれたら CODE_MISMATCH で停止、(i) が後期にずれたら UNANCHORED、(ii) だけが上界を超えたら MEASURE_MISMATCH で測定コードを調べる） | init で W1[0,0] += 1e−3 → (i) の差 > 0／集計の片側だけ W2 の 1 行を 1e−3 動かす → (ii) が上界を超える |
| S2 | **活性化の形**: float32 と float64 で φ(0) == 0.0（Snake 族）、φ′(0) が厳密に 1.0 / 2.0 / 0.0。float64 の格子（[−6/α, 6/α]、φ′ の零点 z = (−π/2 − θ + 2kπ)/(2α) と根を明示的に挿入）で、解析 φ′ と autograd の差 ≤ γ_n·Σ\|項\|（n は式ごとの演算数）。valley の零点付近は絶対床 eps64 | (a) peak が normal に落ちる → φ′(0)=2 の検査が落ちる、(b) sin 項の符号反転（peak↔valley）、(c) 別の位相の解析 φ′ を正しい φ と組み合わせる → 不一致の最大は 2\|sin(Δθ/2)\|。解析値: normal↔peak・normal↔valley（Δθ=π/2）で √2、peak↔valley（Δθ=π）で 2。変異ごとにこの解析値の 1/2 以上の検出を要求する |
| S3 | **q=0・θ=0 の bit 同一**: PhaseSnake(θ=0, α=.6, q=0) と OffsetLeaky(.1, 0) の前向き・逆向きが、宿主 SN06・LR と torch.equal。適応 normal 経路（AdaptivePhaseSnake(θ=0)）が宿主 `GS.H.AdaptiveSnake` と、**箱 B のループと同じ `GS.H.forward` を通して** torch.equal。あわせて `isinstance(AdaptivePhaseSnake(...), GS.H.AdaptiveSnake)` が True | 文字列置換で `z + torch.sin(a*z + 1e-6)**2 / a`、`+ 1e-6`（q）を入れる → equal が False／継承元を `src.pmnist_0905.AdaptiveSnake` に替える → isinstance が False になり、forward の V 更新が走らないことを検出。宣言: 常に `+0.0` を足す変異は符号付き零しか変えず検出不能・無害 |
| S4 | **恒等式（float64・代数とコード）**: P06(W,b) と normal+K_P(W, b+s) の logits と 6 テンソルの勾配。枝 k=−1、valley、twin 経路も同様。状態は θ₀・3 タスク煙走の終状態・重みを 1/4/8 倍した乱数状態。許容は float64 伝播上界。**float32 の Adam 更新の比較は REPORT のみ**（事後: V06 の θ₀ で 1 更新の差が設計許容の 168–262 倍。死んだ起点で eps 1e−8 が丸めを増幅するため。軌道として両方を走らせることはないので問題ない） | K を落とす／s の符号反転／K_V と +s の組 → logits が O(1) ずれる（解析値を事前に計算し、その 1/2 以上） |
| S5 | **init map**: float32 に丸めた実パラメータを float64 に上げて前向き計算し、P06c・V06c・P06c_k1 の logits が N06、P06i と P06tw と SNAi_P（t=0）が P06、V06i と V06tw と SNAi_V（t=0）が V06、LR_qKp と LR_qKpn が LR と、半 ulp の伝播上界以内で一致。W1・b1 の勾配も一致。W2 の勾配差は q·δ2·1ᵀ、W3 は q·δ3·1ᵀ（上界以内） | b3 の補償を外す（ずれは解析的に \|q\|·\|W3·1\|）／補償を 2 回かける／補償なしで q だけ足す → W1 勾配の恒等式が落ちる |
| S6 | **fresh probe**: t1 の fresh が逐次の task 1 の acc と終状態を bit 再現する（事後に可能と確認済み: acc 0.9196・torch.equal True）。全 probe の (perm, idx, order) の sha256 が S8 のタスク別 hash と一致 | (a) 開始点を task 1 終状態にする、(b) task 2 の置換を使う、(c) **task 1 後**の Adam 状態（step 625）を持ち込む、(d) SNA 族で逐次の走の V（task 1 後の値）を fresh probe に持ち込む（V=1 から始めない）→ SNA の t1 fresh の再現が崩れることを検出 |
| S7 | **非侵襲**: 5 タスクの煙走で、probe（t3 に強制）・PD・capacity・除去・帳簿・層 2/3 の記録を入れた場合と入れない場合で、rows・units・タスクごとの全パラメータの sha256 が同一 | (a) probe が main の batch 生成器から 1 回だけ引く → **t4 以降**の hash が変わる、(b) 追加記録で W1 に 1e−9 足す |
| S8 | **ストリームの腕独立と未見性**: タスクごとの perm/idx/order と `sp0914_*` の sha256 が全腕で同一、seed 間では異なる。PD の 8 置換と capacity の 2 置換が、同じ seed の 120 タスクのどの置換とも異なる（assert は出力を書いた後） | role 文字列に腕名を混ぜる／seed s+1 に s を使う／PD の置換をタスクの置換生成器から引く → task 1–8 の置換と一致して検出 |
| S9 | **追加測定の正しさ**: 層 2/3 の per-unit 配列、Aᵢ、āᵢ、Φ、r_a、Δu_m、CT1 の logN、CT3 の s、ω 残差（REPORT）、成長比の対数の SD（実 c0）を、独立な float64 の再実装と上界以内で照合 | 独立再実装の側だけ W2 の 1 行を 1e−3 動かす／ω 残差の傾き 1.05 を 1 にする／成長比の分母を √(1/3) にする → 解析的なずれの 1/2 以上で検出 |
| S10 | **帳簿**: G = D² + 2⟨W̃_s, ΔW̃⟩ が float64 上界以内。窓内の Q が 1 更新ごとの ΔW̃ の二乗和と一致 | 内積の符号反転／1 タスクだけ中心化を省く |
| S11 | **M1–M3 の読み出し**: 合成配列で、オフセットを保つ → r_a=1、座り直す → r_a=0、LIN → Aᵢ の上界とゲート差がともに 0、同一 run 同士 → Φ の差 0 | q の符号を分母で反転 → ラベルが入れ替わる／ゲート計算の θ を入れ替える → Φ が非零 |
| S12 | **群除去**: z2・logits の平均が float64 で厳密に不変、G=∅ で変化 0、出力列を 0 にしたユニットの除去で変化 0 | 補償項を外す → 平均が W[:,G]·mean a_G ずれる／G の補集合を除去する |
| S13 | **capacity の mask**（コードの検査・結果の証拠には数えない。**トートロジーと宣言**）: 凍結した行が bit 不変、FROZEN で層 1 が不変、**LOW・MID・HIGH が 100 ユニットを分割し、RAND1 と RAND2（と CT2 の B 個の無作為群のうち同じ番号の対）が互いに素**、X_RE が再初期化系列と一致、全変種の Adam が m=v=0・t=0 から開始 | 0 行目を飛ばす mask／本流の init 系列を再利用する再初期化／Adam 状態を持ち越す変種／MID を 33 にする（1 ユニットが漏れる） |
| S14 | **判定コード（verdict.py）**: 合成シャードで、**すべてのラベルと修飾**が期待どおりになる。E1: MORE/LESS（10/10 同符号）・`_WITHIN_RESOLUTION`・EQUIVALENT（真値 0・雑音 0.1）・INCOMPLETE（seed を欠く）・NOT_TESTABLE_REF_FLAT。修飾: WINDOW_SENSITIVE、TRANSIENT_IN_BASE／TRANSIENT_UNRESOLVED、REF_FLAT、UNANCHORED、BROKEN_k/n、DIVERGED_k/n、FRESH_LEVEL_CONFOUNDED、UNDERPOWERED_FOR_RESOLUTION。E2 の HIGHER/LOWER/EQUIVALENT、E3 の 4 ラベル。§6.4 の 8 クラスと、条件が重なる組の優先順位。F1x の NON_ADDITIVE/ADDITIVE/ADDITIVE_INCONCLUSIVE。§6.6 の対応表の全行と LEVEL の細分・STRUCTURE。M1–M3 の全状態。n_main の規則（σ̂_U の 3 池の最大、30 の上限、df 18）。 | 腕の列を入れ替える → 符号が反転／Holm の段を外す → 境界の対比数が変わる／seed の対を崩す → CI が広がる／§6.4 の順を入れ替えて CONFLICT を後ろにする → 重なる組のクラスが変わる／M3 を 2 状態に戻す（非有意を CLEARED にする）→ TRANSIENT_UNRESOLVED の合成が CLEARED になって検出／池 1 に N06−LR を足して df 27 と数える → σ̂_U が変わって検出／h_D を定数 0.24 に戻す → 植えた等価の境界例のラベルが変わる |
| S15 | **発散と壊れた走**: loss に NaN を注入（task 2, step 10）→ `DIVERGED(2,10)`、task 1 の rows と units がディスクに残る。lr=0 の 5 タスク煙走 → BROKEN。14/20 と 15/20 の合成配列が境界どおりに判定される | NaN を注入しても flush しない版 → task 1 の出力が無いことを検出／15/20 を 14/20 と判定する off-by-one 版 |
| S16 | **SGD プロトコルの橋**（段階 S）: 新ループ（sgd・4 epoch）と `src.pmnist_0905.run_one(optimizer='sgd', epochs=4, lr=.02, device cpu)` が、LR・LIN・SN1 の seed 0 × 3 タスクで acc と w_norm/zbar/zsd/mob を完全一致（この 3 腕は isinstance 分岐を通らない固定の Activation なので、2 つの宿主モジュールの違いは効かない） | epochs=3／lr=.0201／epoch ごとの再シャッフル |
| S17 | **資源の測定とメモリの見張り**: VmRSS が、**埋めた** 512 MiB テンソル（torch.ones）の確保前後で 512 MiB − ページ粒度以上増える。launch.sh は偽の peak RSS 20 GiB を与えると P=0 で起動を拒む。wcap・act_chimera のプロセス（スクリプトパスで照合）がいれば拒む。見張り（`memwatch.py`、§12.2）は、偽の /proc/meminfo（MemAvailable を境界の上下に置いた列）を与えると、境界を下回った時点で最新の shard に SIGSTOP を、復帰の境界を上回った時点で SIGCONT を送る（ダミーの sleep プロセスで確認）。SwapFree が RSS_peak を下回る偽の列では新規起動を止める | torch.empty（ページを触らない）で増えないことを示す／pgrep の照合パターンを外す／見張りの比較を ≥ と ≤ で入れ替える → 停止すべき列で停止しないことを検出／最古の shard を止める版 → 止まる PID が違うことを検出 |
| S18 | **RL ラッパー**: `src/pmnist_rlmnist_0906.py` と `src/pmnist_0905.py` の sha256 が origin/main の blob と一致（無改変）。差し込んだ θ=0 PhaseSnake が `pmnist_0905.ARMS['SN06']` と、1 タスク × 2 epoch の per_task 行で完全一致。ランナーと宿主の `.kind` 分岐を grep し、新しい kind が汎用経路だけを通ることを確認。**呼び出しの固定**: 起動したプロセスの provenance.json の device が `cpu`、出力先が `results/snake_phase_mnist_0914/rl/<arm>_s<seed>` の下、`torch.get_num_threads() == 1`、sidecar_provenance.json がある。**evaluate_rl の包みの非侵襲**: 包みあり・なしで 1 タスク × 2 epoch の per_task.csv とタスク末の全パラメータの sha256 が一致し、units.npz の cnormᵢ が float64 の独立な再計算と一致。**LR の直接照合**: wcap のランナー（wcap の git_head の blob）の LR ref と、無改変の 0906 `run_one` の LR が、CPU・seed 0・1 タスク × 2 epoch でタスク末の重み bit 一致（wcap の LR_ref を陽性対照に使う場合だけ必要） | 名前 SN06 のまま P06 を差し込む → 行がずれる／`--device` を渡さない版 → provenance が cuda になって検出（CUDA が無い機械では device の既定値の文字列を検査）／`--out` を渡さない版 → 出力先が results/pmnist_rlmnist_0906 になって検出／包みの中で W1 に 1e−9 足す → sha が変わって検出／LR の照合で l2 の係数を 2→1 にした runner を使う（l2 腕）→ 不一致を検出 |
| S19 | **盲検**: 段階 2 の `gate0.py` が読む腕の対は {N06−LIN, LR−LIN, P06−P06tw, V06−V06tw} と、単腕 SD 用の P06・V06・N06 だけ（AST で検査）。gate0 の出力（`gate0.json`）の欄は SD・h_E・σ̂_U・n_main・TESTABLE・BROKEN の数・双子の検査だけで、平均の差の欄が無い | P06−N06 を読む行を足す → 検出／gate0.json に単腕 D の平均を書く版 → 欄の検査で検出 |
| S20 | **RL の判定（rl_gate.py）**: 合成の per_task（腕 × 10 seed × 50 タスク）で、RL_GO_FRESHGAP・RL_GO_PROGRESSIVE・RL_NO_GO_AT_400EP・RL_INCONCLUSIVE、陽性対照の不成立（LR_CONTROL_FAILED）がそれぞれ期待どおり。G・P・D・傾きが wcap の定義（G = A(1) − A(31–50) など）と数値で一致 | 窓を 41–50 にずらす／P の符号を反転する／RL_GO_PROGRESSIVE から傾きの条件を落とす／G の経路を消す（P だけで判定）→ 植えた「早期に崩れて平坦」の腕が NO_GO になって検出 |
| S21 | **分岐の判定（段階 S の FORK_*）**: 合成のユニット別の初期値と終値で、FORK_PRESENT・FORK_ABSENT・FORK_INCONCLUSIVE が期待どおり。f_shrink が strict な縮み（‖w(end)‖² < ‖w(0)‖²）で数えられている | strict な < を ≤ 0.95·c0 の許容幅に置き換える → 植えた境界ユニットの数え方が変わって検出／1/100 の境界を 2/100 にする（off-by-one）→ ちょうど 1/100 の合成で検出／単峰超過の項を落とす → 単峰で縮みの多い合成が PRESENT になって検出 |
| S22 | **CT 族の判定**: 合成で CT1 の 4 ラベルと接尾、CT2 の 4 ラベル（帯 b_θ の算出を含む）、CT3 の 3 ラベルが期待どおり | CT1 のマージンを ±0.09877 から ±ln 1.170 に替える → 境界例で検出／CT2 の超過を無作為群の平均でなく RAND1 だけで取る → 帯の合成で検出／CT3 の分母 sdcur を落とす → 尺度を変えた合成でラベルが変わって検出 |
| S23 | **双子の軌道の検査**（§6.1）: 段階 0 に seed 0–2 で P06・P06tw・P06tw_noK・P06(1 ulp)・P06(別式) と V06 の同じ組を task 1 の 20 更新まで走らせ、δ_ref < 境界 < δ_noK なら、P06tw_noK が `IDENTITY_TRAJECTORY_BREAK` を発火し、P06tw が発火しない。min δ_noK ≤ max δ_ref なら `TWIN_CHECK_NOT_IDENTIFIABLE` になる（合成の δ で分岐を確認） | 境界を δ_ref の最大値そのものにする版 → 合成の境界例で検出／noK の走を P06tw と同じ設定にする（変異が効かない）→ δ_noK が δ_ref 以下になり NOT_IDENTIFIABLE の分岐に入ることを確認 |
| S24 | **段階 2 の停止規則**（§11）: 合成の gate0 入力で、TESTABLE_FIXED 不成立 → F1・F1x の shard を段階 3 の表から外す、TESTABLE_LEAKY 不成立 → F3 を外す、BROKEN/DIVERGED の seed 数が k 以上 → その腕を外す、IDENTITY_TRAJECTORY_BREAK → 段階 3 の表を出さない | k を 1 つ小さくする → 境界の合成で検出／TESTABLE の CI を 90% にする → 境界の合成で検出 |
| S25 | **煙走と本走の分離**: launch.sh は、shard 表の出力先が `results/_smoke_snake_phase_mnist_0914/` を指す行、または probe の強制フラグ（force_probe_tasks 等）を持つ行があれば起動を拒む。煙走・資源の走は必ず `_smoke` の下に書く（act_chimera_0913 の spec にある「出力先を共有するとスモークと本走を取り違えうる」への対策） | 本走の表に `_smoke` の行を 1 つ混ぜる → 拒否を検出／強制フラグの検査を外す → 検出 |

## 10. 事前予測（記名・走る前）

**Claude**（確率・反証条件）:

| # | 予測 | 確率 | 反証 |
|---|---|---|---|
| P1 | S4・S5 の恒等式が float64 の上界以内で成り立ち、変異はすべて検出される | .98 | どれか 1 つでも上界を超える、または変異を見逃す |
| P2 | N06・LR・LIN・SNA の s0–2 が 120 タスクで、学習状態の配列（cnorm・m・bias）が 0.0 で一致し、派生量が上界以内（G1、S1 の 2 群）。確率は、G1 の commit の前に 4 腕の s0 × t1–3 の照合を行ってから書き直す（現在の値は SN06 s0 の t1–3 だけに基づく） | .90 | 学習状態の配列に非零の差、または派生量が上界を超える |
| P3 | TESTABLE_FIXED と TESTABLE_LEAKY がともに成立（事後の見込み: D_pair(SN06−LIN) 2.20/1.90/1.97、D_pair(LR−LIN) 1.78/1.57/1.77、gate_shape_0911 rows、窓 t16–30 / t101–120） | .95 | どちらかの CI_lo ≤ 0 |
| P4 | N06 の n_main seed の D の平均が、3 seed からの予測区間に入る。**丸め前の D**（SN06 rows の acc から t16–30 平均 − t101–120 平均）= 2.4243/2.2630/2.1442、平均 2.2772、SD 0.1406。n=20 なら 2.2772 ± (17/20)·4.3027·0.1406·√(1/3+1/17) = 2.2772 ± 0.3221 = **[1.955, 2.599]**。n_main が違えば同じ式で、**丸め前の D から**計算し直して addendum に 3 桁で書く | .85 | 区間外 |
| P5 | 位相の洗い流し: 固定 α の P06 対 N06、V06 対 N06 は、層 1 の t101–120 で PHASE_WASHED（.75）。t16–30 では少なくとも V06 対 N06 が PHASE_ACTIVE（.65）。SNAP 対 SNA、SNAV 対 SNA は t101–120 で PHASE_ACTIVE（.80）。根拠は §2 の事後の振幅（SN06 t120 で 0.09–0.10、SNA の \|ḡ−1\| 0.28–0.32） | 左記 | 固定 α の後期で ACTIVE、または適応族の後期で WASHED |
| P6 | 位相の総効果は α の梯子全体より小さい: C1・C2 の 95% 反転 CI がともに ±0.8 pt に収まる。根拠: 事後の α 梯子の対差 D_pair(SN06−SN02) 1.24–1.88、(SN06−SNA) 1.16–1.87。固定 α の位相の振幅（SN06 の medianᵢ exp(−2α²sdcurᵢ²)）は t1 0.64–0.67 から t60 で 0.25（t1 の約 4 割）、t120 で 0.09–0.10（約 1.5 割）まで減衰し、後期窓では小さい | .75 | どちらかの CI が ±0.8 をはみ出す |
| P7 | 分解: 両位相で、K 因子主効果の点推定の絶対値 > S 因子主効果の点推定の絶対値。S 因子（C4・C6）は MORE/LESS にならない。根拠: 起点のずれは行平均・bias のずれで 5–15 タスクのうちに緩む（M3 の目安）が、オフセットはすべての W2・W3 勾配に残り続ける | .55 | S 因子が Holm で有意、かつ seed 別の \|S\|−\|K\| の CI が正 |
| P8 | オフセット角の平均活性は層 1 で t101–120 に完全には残らない: P06c・V06c・LR_qKp の r_a が MEAN_RESEATED か PARTIAL（NO_RESEATING ではない） | .55 | 3 つのうち 2 つ以上が NO_RESEATING |
| P9 | 枝: B_P = P06c_k1 − P06c は EQUIVALENT にならない（根の幾何だけでなく活性値の水準と符号が効く） | .55 | EQUIVALENT |
| P10 | オフセット効果は Snake 特有ではない: C9（LR_qKp−LR）と C3（P06c−N06）の点推定が同符号。C9n（LR_qKpn−LR）は C9 と点推定の符号が逆（.55） | .60 | C9 と C3 が両方 Holm で有意かつ逆符号／C9n と C9 が両方有意で同符号 |
| P11 | 適応位相: C7・C8 の少なくとも一方が MORE_DECLINE。SNAV の後期 Cov（P[φ′<.25]）が 3 腕中最大（.75）。初期関数を揃えた C7s・C8s の少なくとも一方も MORE_DECLINE で、§6.6 の答えが STRUCTURE（.35）。根拠: valley のゲート零は z/Wᵢ 座標で原点に固定。登録: 13 腕で Cov が L を順序付ける（ρ +0.70・gate_shape_0911 summary §2） | .50 | C7・C8 がともに EQUIVALENT か LESS／SNAV の Cov が最大でない／C7s・C8s がともに EQUIVALENT（STRUCTURE の予測だけの反証） |
| P12 | （REPORT の予測・ラベルなし）Holm で有意になった F1・F2 の対比の ΔlogN1 の点推定は、D_pair と同符号（劣化が多い腕ほど層 1 が広い）が過半数。腕間の比較なので W 病理の証拠には数えない | .55 | 過半数が逆符号 |
| P13 | E1 と E3 の整合: Holm で有意な E1 の対比に CONFLICT は出ない | .80 | CONFLICT が 1 つでも出る |
| P14 | Adam では CondA 型の分岐が無い: 全 Snake 位相腕（固定・適応）で、層 1 の成長比の対数（実 c0 基準、t120）のユニット間 SD の n_main seed 平均 < 0.13（事後の箱内 GELU の値 .131–.136。Snake・LR・LIN は 0.023–0.042、§8-2）。初期値より strict に縮むユニットは 0（.97・算術上ほぼ自明で、証拠には数えない） | .85 | どれか 1 腕でも ≥ 0.13 |
| P15 | （段階 S を走らせる場合）SGD でも CondA そのままの分岐は現れない: N1・P1・V1・LIN-S・LR07q-S のすべてで FORK_PRESENT にならない（§11 段階 S）。もし現れるなら f_shrink 最大は LIN-S か LR07q-S（条件付き .6） | .55 | どれか 1 腕で FORK_PRESENT |
| P16 | RL 予備走: N06-RL は RL_GO_FRESHGAP にも RL_GO_PROGRESSIVE にもならない。根拠: SNA は天井で平坦（§2）、固定 1200 枚の正の転移で G < 0（wcap の事後: SNA none の G −1.32）。P06-RL・V06-RL も同様（.55）。V06-RL は死んだ起点のため task 1 の水準が低く、G の点推定が 3 腕で最も小さい（.5） | .60 | N06-RL がどちらかの RL_GO |
| P17 | 群の将来学習（記述）: Adam の全腕で C(LOW)−C(HIGH) が腕ごとの RAND1−RAND2 帯に入る | .65 | 帯外の腕がある |
| P18 | **CT1 層 1**: C1（P06−N06）は GROWTH_SUPPRESSED にならない（CondA の向きは固定 α=0.6・Adam の箱で再現しない）。根拠: 位相の振幅は後期窓で t1 の約 1.5 割、Adam は勾配の大きさを正規化する | .60 | C1 の層 1 が GROWTH_SUPPRESSED |
| P19 | **CT1 層 2 とオフセット**: 層 2 の logN2 の seed 平均が \|K\| の順に小さい: P06c_k1（\|K\|=3.09）< P06c（2.14）< V06c（0.48）< N06。根拠（推論）: 層 2 の勾配 δ2(a1 + K)ᵀ のうち K の項は列に一定なので中心化で消えるが、Adam の分母 v には K²δ2² として入り、中心化した成分の歩幅を \|a1\|/\|K\| の程度に縮める。C1 の層 2 は GROWTH_SUPPRESSED（.55） | .50 | 順序が 1 箇所でも崩れる／C1 の層 2 が GROWTH_SUPPRESSED でない（後者は括弧の予測だけの反証） |
| P20 | **CT2**: Adam の 6 腕すべてで HIGH・LOW とも CARRIES_MORE/LESS_THAN_RANDOM にならない（WITHIN_RANDOM_BAND か INCONCLUSIVE）。根拠: 群間のノルム差が 5–8% と小さい（§8-4） | .60 | どれか 1 腕の HIGH か LOW が有意 |
| P21 | **CT3**: C2（V06−N06）が CLOSER_TO_ROOT。根拠（推論）: valley は根でゲートが 0 になり、根の近くのユニットは勾配を受けにくく動きにくい | .40 | C2 が FARTHER_FROM_ROOT か SEATING_UNRESOLVED |
| P22 | **E2・E3 の向き（死んだ起点）**: C2（V06−N06）で ΔA_late が LOWER（.55）、V06 に FRESH_LEVEL_CONFOUNDED が付く（.50）。C1（P06−N06）の ΔA_late は LOWER でも HIGHER でもない（.60）。F3 の C9・C9n の E3 はともに GAP_LARGER でない（.60） | 左記 | 各括弧の予測ごとに逆の結果 |
| P23 | **壊れた走**: 段階 2 の V06 s0–9 で BROKEN または DIVERGED は 0 本（.80）。P06 も 0 本（.90） | 左記 | 1 本以上 |
| P24 | **M3**: P06i−N06 と V06i−N06 の少なくとも一方が CLEARED にならない（TRANSIENT_IN_BASE か TRANSIENT_UNRESOLVED）。根拠: 単腕の zcur 中央値が t20 から t120 にも動く（§6.5） | .60 | 両方 CLEARED |
| P25 | **n_main**: 段階 2 の式が上限 30 を超え、UNDERPOWERED_FOR_RESOLUTION が付く。根拠: 36 対でプールした SD 0.24 なら n=31、S19 型の 2 対だけなら n=14（§7.2） | .45 | 式が 30 以下で UNDERPOWERED_FOR_RESOLUTION が付かない |
| P26 | **FS 族**（段階 S を走らせる場合）: P1−N1、V1−N1 の E1 はともに MORE/LESS にならない | .55 | どちらかが Holm で有意 |
| P27 | **RL の陽性対照**: LR の P の CI_lo > 0 かつ G の CI_lo > 0（0906 CUDA の事後: P +1.74、10/10 正、G +16.32）。CPU でも同じ向き | .90 | どちらかの CI_lo ≤ 0（LR_CONTROL_FAILED） |

**Issa**: 未記入。**G1 の事前登録 commit に含める**（段階 0 の煙走より前。段階 1 の錨と段階 2 の較正の結果は段階 3 より前に読まれるので、「結果を読む前」ではなく「最初の走より前」を締切にする）。本表の Claude の予測も同じ commit で確定し、以後は addendum でしか変えない（P2 の確率の書き直しも G1 の commit の前に行う）。
> P5（位相は後期に洗い流されるか）: ／ C1 P06−N06: ／ C2 V06−N06: ／ K 対 S の大きさ: ／ C7・C8（適応位相の総効果）: ／ C7s・C8s（初期関数を揃えた位相構造）: ／ CT1（peak は normal より育たないか、層 1・層 2）: ／ CT2・CT3: ／ SGD の分岐: ／ RL で Snake は劣化するか（G と P）:

## 11. 段階とゲート

- **G0 調整**（走る前）:
  - Issa の承認。
  - wcap_rlmnist_0914 の本走終了を確認する（02:07 開始・見込み 3.43 h・`results/wcap_rlmnist_0914/checks.json` S-cost）。
  - act_chimera_0913（未 commit の spec、箱 B 51 走 + RL 15 走）との順序を合意する。
  - Codex の zero-attraction チャットに、設計ノート 0913 §4C の第 3 段階を引き取ることを伝える。
  - `git worktree list` の残骸（lop_analysis_shell0913 は main に取り込み済みで残存、最上位の zero_attraction_0913・zero_attraction_analysis_0913・elu_* は §1 の配置外）を Issa に報告する。触らない。
- **G1 事前登録**:
  - `git -C proj_004_drift fetch origin` の後、`wt/snake_phase_mnist_0914` を origin/main から作る。
  - `specs/spec_snake_phase_mnist_0914.md` と `configs/snake_phase_mnist_0914.yaml` を commit・push。config には腕・定数（s・K・π/α を 17 桁。§3.2 の式から float64 で生成し、式の文字列を並べる）・窓・族・m・rng・h_E の算出規則・n_main の規則・停止規則の式・CT のマージンを入れる。
  - **Issa の予測欄と Claude の予測表をこの commit に含める**（§10。締切は段階 0 の煙走より前）。P2 の確率は、この commit の前に 4 腕 × s0 × t1–3 の照合をしてから書く。
  - vault の spec/ に同文を置き、現在地に 1 行リンク。
- **段階 0 検査**:
  - S1（3 タスク分）・S2–S15・S17–S25 がすべて通り、変異がすべて検出される（`checks.json` all_pass）。S23 の双子の参照（δ_ref・δ_noK）もここで測る。
  - 全腕の 3 タスク煙走。**煙走・資源の走・双子の参照の走は、すべて `results/_smoke_snake_phase_mnist_0914/` の下に書く**（S25）。
  - 実装と煙走を commit・push。その hash が provenance の git_hash になる。
- **段階 0 資源**:
  - 最重量の腕（SNAV、probe・PD・capacity・snapshot を t3・t20・t120 に強制）で **120 タスクの煙走を 1 本**、`results/_smoke_snake_phase_mnist_0914/resources/` に走らせ、ru_maxrss を t5・t20・t60・t120 で、秒/タスクをコア番号付きで記録する（外挿しない）。
  - 同じ時間帯に `memwatch.py` を記録だけのモードで回し、MemAvailable・SwapFree・自分のプロセスの RSS の合計を 2 秒ごとに記録する（デスクトップ側の落ち込み ΔM_desk の測定、§12.2）。
  - RL のラッパーは別に、1 腕 × 1 seed × 2 タスク × 400 epoch を `_smoke` の下に走らせて、peak RSS と秒/100 epoch を測る（wcap の値を借りない）。
  - §12.2 の式で P を決める。§12.3 の見積りを実測値で置き換え、承認された壁時計を超えるなら §12.3 の縮小規則を当ててから段階 1 へ。
- **段階 1 錨**（最初の門）:
  - N06・LR・LIN・SNA の s0–2（12 走）を先に走らせる。
  - S1 を 120 タスクで判定: CODE_MISMATCH なら全停止、MEASURE_MISMATCH なら測定コードを直すまで停止（学習状態は一致しているので走り直しは測定だけ）、UNANCHORED なら続行して記録。
- **段階 2 較正（盲検）**:
  - N06・LR・LIN の s3–9（21 走）と、P06・P06tw・V06・V06tw の s0–9（40 走）。
  - `gate0.py` は S19 の対と単腕 SD だけを読み、h_E・σ̂_U・n_main・TESTABLE・双子の検査・BROKEN/DIVERGED の数を出す。
  - **停止規則（登録）**。どれも gate0 の出力だけで決まり、対比の平均を読まない（S24 で検査）:
    1. TESTABLE_FIXED（D_pair(N06, LIN) の 10 seed の反転 95% CI_lo > 0）が不成立 → F1・F1x の対比に要る腕（P06c・P06c_k1・P06i・V06c・V06i）の段階 3 を走らせず、F1・F1x を `NOT_TESTABLE_REF_FLAT` と報告する。F2・F3・CT は続ける。
    2. TESTABLE_LEAKY が不成立 → LR_qKp・LR_qKpn の段階 3 を走らせず、F3 を NOT_TESTABLE_REF_FLAT と報告する。
    3. 腕ごとの BROKEN・DIVERGED の seed 数 k_obs（s0–9 のうち）: Clopper–Pearson の 95% 上側限界 π_U を出し、段階 3 で n_main seed を走らせたときに完全な seed 数が n_min(6)=8 を下回る確率 P(Binomial(n_main, 1 − π_U) < 8) が .05 を超えるなら、その腕の段階 3 を止めて `NOT_TESTABLE_BROKEN` と報告する（k は n_main で決まるので、addendum に k の値を書く）。
    4. IDENTITY_TRAJECTORY_BREAK → 段階 3 の shard 表を出さず、実装を調べる（CODE 調査）。原因を直したら段階 0 からやり直す。TWIN_CHECK_NOT_IDENTIFIABLE は停止しない。
  - addendum（`gate0.json`・h_E・n_main・停止規則の適用結果・段階 3 の shard 表・実測に基づく費用）を commit・push してから段階 3 へ。
- **段階 3 本走**:
  - 残りの shard を n_main まで。seed ごとに腕を交互に並べる（途中で止めても対が残るように）。
  - `memwatch.py` を常駐させる（§12.2。MemAvailable が境界を下回ったら最新の shard を SIGSTOP、戻れば SIGCONT、止めた時刻と PID をログに書く）。
  - rows は毎タスク flush、units はタスクごとに追記保存、provenance は最後。
  - 対照は本ループの前に走らせ、結果は記録だけで assert は最後。
  - 全 shard が揃うまで F1–F3 の対比は計算しない。
- **段階 3 判定**:
  - `verdict.py`（S14 で検証済み）。
  - summary.md の数値は verdict.csv から写し、窓ラベルを付け、事後の項目は「事後」と書く。
  - vault の 測定/ に結果ノートを置き、現在地に 1 行。
- **段階 R（RL 予備走）**:
  - 別の時間帯に、W病理チャットの RL プロセスがいないことを確認してから。
  - N06-RL・P06-RL・V06-RL × seed 0–9、CPU（事後: CUDA は負荷下で 1.07–1.79 ms/更新、CPU は 0.42–0.48 ms/更新で CPU の方が速い）。
  - 陽性対照 LR（§4.2 の規則。既定は自前の LR × 10 seed）。
  - `memwatch.py` を常駐させる。並列数は段階 0 資源で測ったラッパーの peak RSS から §12.2 の式で決める。
  - 指標: W病理の定義に揃えて **G = A(1) − A(31–50)（fresh gap。RL ではタスクが交換可能なので task 1 がその手法の fresh 学習そのもの、wcap spec §1 の読み 1）**、P = A(11–20) − A(41–50)、D = A(2–6) − A(31–50)、傾き t11–50（online_acc）。memo_acc は別に出す。G は「固定 1200 枚の正の転移の利益 − 可塑性の損失」の正味である（wcap spec §1 の読み 2）ことを併記する。ユニット別の cnormᵢ（層 1・2、§4.2 の包み）は REPORT。
  - ラベル（腕ごと・反転 95% CI、上から順）:
    - `LR_CONTROL_FAILED`（陽性対照に付ける）: LR の G の CI_lo ≤ 0 かつ P の CI_lo ≤ 0。このとき Snake 腕の NO_GO は出さず、RL_INCONCLUSIVE にする。
    - `RL_GO_FRESHGAP`: G の CI_lo > 0（早く崩れて平坦になる腕も、器として拾う）。
    - `RL_GO_PROGRESSIVE`: P の CI_lo > 0、かつ傾きの CI_hi < 0。
    - `RL_NO_GO_AT_400EP`: G の CI_hi ≤ 0、かつ P の CI が 0 を含み、かつ P の CI_hi < LR の P の CI_lo（陽性対照より有意に小さい）。
    - `RL_INCONCLUSIVE`: それ以外。
    - 事後の参照: 0906 CUDA の LR は P +1.74（seed 範囲 0.92–2.29、10/10 正）・G +16.32、SNA none（0913）は P −0.03・G −1.32（wcap spec §1 の表）。
  - NO_GO は「400 epoch・この腕」に限定する。80 epoch の予備走は Issa の判断（§16-4）。
  - **RL_GO が出たときに起きること（登録）**: 本 spec の中では何も追加で走らせない。GO の腕と経路（FRESHGAP／PROGRESSIVE）を結果ノートに書き、Issa の承認を得て**別の spec**（run id `snake_phase_rl_<MMDD>`、branch `claude/snake_phase_rl_<MMDD>` を本実験の main 取り込み後に origin/main から切る）を起こす。その spec の最低条件: ユニット別の中心化ノルム ‖W̃ᵢ‖（層 1・2）を毎タスク記録するランナー、固定 α の 2×2（N06・P06・P06c・P06i と valley の同型）、W病理チャットの wcap の定義（G・P・D）との一致の検査、wcap との衝突回避の合意。
- **段階 S（SGD 橋・Issa の判断 §16-1）**:
  - S16 が通ること。
  - 安定性の pilot: seed 100–104 × 5 腕 × t1–40。どれかが非有限なら、箱全体の lr を .02 × minθ (rms φ′_N1 / rms φ′_θ)²（初期値の probe での 2 層分の積）に下げ、新しい箱として記録して 1 回だけ再 pilot。それでも発散したら `SGD_UNSTABLE` で中止。
  - 本走 5 腕 × 10 seed。
  - 族 FS（基準 N1、m=2）: P1−N1、V1−N1 に E1–E3 と同じ規則。n_min(2) = 7。
  - 分岐のラベル: f_shrink = 層 1 で **strict に縮んだ** ユニット（‖W̃ᵢ(t120)‖² < ‖W̃ᵢ(0)‖²、許容幅なし。CondA の `n2[-1] < n2[0]` と同じで eps に依存しない）の割合。`FORK_PRESENT` は f_shrink の CI_lo > 1/100、かつ単峰超過 f_obs − Φ(−μ̂/σ̂)（μ̂・σ̂ は log 成長比のユニット間の平均と SD）の CI_lo > 0。`FORK_ABSENT` は f_shrink の CI_hi < 1/100。それ以外は `FORK_INCONCLUSIVE`（S21 で検査）。
  - CondA の比較値は §4.3 の ALL 行の割合（eps に依存しない）だけを使う。
- **後続実験 M（媒介・別名・別 spec・本 spec では走らせない）**:
  - 発火条件: F1・F2・F3 のどれかで TEMPORAL_LOP_MORE/LESS。
  - **新しい seed**（勝者の呪いを避ける）で段階 3 の対を再走し、d_ref の符号が再現しなければ `NOT_REPLICATED`。
  - 主な設計は **幅合わせ射影（WMATCH）**: θ 腕の各ユニットの ‖W̃ᵢ‖ を、毎更新、同じ seed の基準腕の順位対応の軌道（タスク終端値をタスク内で線形補間）へ射影する。
    - 零対照: 基準腕を自分自身の軌道へ射影し、その劣化が基準と EQUIVALENT であること。そうでなければ `WMATCH_INVASIVE`。
  - 通常の wclamp（自腕の t20 値に固定）は「クランプが劣化を消すか」の確認に留める。SN06 の R_c 0.02（登録）のため、腕間差がクランプ下で 0 に潰れる（床による圧縮）ので、腕間の媒介の証拠には使わない。
  - 層を限定したオフセット（層 1 のみ・層 2 のみ）もここに入れる。
- **ブランチの扱い**（CLAUDE.md §3「1 ブランチ = 1 実験」）:
  - 段階 3（箱 B）・段階 R（RL 予備走）・段階 S（SGD 橋）は、**同じ spec・同じ run id の登録された部分段階**として 1 つの実験に数え、`results/snake_phase_mnist_0914/{boxb, rl, sgd}` に置いて一緒に main に入れる。
  - ただし段階 S は **G1 の commit までに Issa が承認した場合だけ**この実験に含める。G1 の後に承認された場合は、本実験の main 取り込みと片付けが済んでから、origin/main から `claude/snake_phase_sgd_<MMDD>` を別に切り、別の spec で登録する（本 spec の段階 S の節をその spec に写す）。
  - RL_GO の後続（上記）と後続実験 M も、それぞれ別の branch と spec にする。
- **片付け**（CLAUDE.md §4）:
  - 段階 3・R（・S）の結果 commit が済んだその日に行う。
  - git 外の snapshot・units・log を `~/Projects/obsidian-research-data/snake_phase_mnist_0914/` へ移し、`results/snake_phase_mnist_0914/backup_manifest.json` を commit。
  - worktree 内で merge origin/main し、HEAD:main へ push。
  - is-ancestor を確認してから worktree・branch を削除する。
  - 後続実験 M は origin/main から別名で切り、t20 snapshot は manifest の sha256 で照合して読む。

## 12. 費用見積りと並列数

### 12.1 実測値（出所）

| 量 | 値 | 出所・条件 |
|---|---|---|
| 箱 B 120 タスク 1 走 | **52–124 s**（seed 1–2 は 52–101 s。seed 0 は 57–124 s で、錨の対照を回す SN02・SN06・SN15・SNA・LIN の seed 0 は 81–124 s。SN06 64–96 s、SNA 73–124 s、R 101–103 s） | gate_shape_0911 の 13 腕 × 3 seed の provenance wall_seconds、2–8 並列下 |
| 箱 B 1 タスク（GS.train の複製） | P コア（cpu1）**0.53 s**（更新 0.396・測定 0.073・gate 0.011・抽出 0.047）／E コア（cpu22）**1.32 s**／固定なし・負荷下 1.41 s | 事後、批評時 `feas_timing.py`・`feas_timing2.py`、2026-09-14 02:13–02:17、SN06 s0 |
| fresh probe 625 更新 + test 評価 | P 0.41 s／E 1.11 s | 同上 |
| snapshot 1 個 | 1,096,420 bytes | 同上 |
| ru_maxrss（箱 B） | MNIST 読込後 1043 MiB、3 タスク後 1076 MiB | 同上（更新なしの別測定でも 1.04 GiB） |
| 1 更新ごとの W̃ 帳簿 | 重い版 約 2.0 ms/更新（層 1）／簡素版 +0.57 ms/更新（層 1、I・K・Q・変位を float64 で積む） | 事後: elu_turn_0912 LR 135.1–137.9 s 対 gate_shape_0911 LR 60.5–62.1 s、計装は LATE=(61,120) の 37,500 更新だけ（`src/elu_turn_0912.py:26`）なので 75 s / 37,500 ≈ 2.0 ms。簡素版は `bench_ledger.py`。本 spec は簡素版の Q だけを 2 層で取る（§8-5） |
| wclamp 継続 t21–100 | ref+wclamp で 147–194 s | gate_wclamp_0911 provenance |
| RL CPU | LR_ref 4.23 s/100 epoch（8 並列下）→ 1 走 14.3 分、peak RSS 1.047 GiB、slots 10（MemAvailable 17.5 GiB から） | `wt/wcap_rlmnist_0914/results/wcap_rlmnist_0914/checks.json` S-cost、02:06 |
| CPU 1 更新（RL 形状） | Snake 0.437・peak 0.423・leaky 0.479 ms（CUDA 1.07–1.79 ms） | 事後、批評時 `steptime.py`、wcap 10 並列の負荷下 |
| 機械の状態 | MemAvailable **13.2 GiB**、swap 5.9/8.0 GiB 使用、wcap_rlmnist_0914 が 10 プロセス（RSS 各約 0.90 GiB）で走行中 | `free -m`・`swapon`・`ps`、本セッション（起草時） |

### 12.2 並列数の決め方

**P = min(C_free, ⌊(MemAvailable − ΔM_desk) / RSS_peak⌋)**。swap は余裕に数えない。

- RSS_peak: 段階 0 資源の 120 タスク煙走で **t120 に直接測った** 最重量腕の peak（probe・snapshot の一時確保を含む）。外挿も固定倍率も使わない。RL は段階 0 資源で測ったラッパーの peak（wcap の値を借りない）。
- **ΔM_desk（旧草案の 4 GiB の定数を置き換える実測の余白）**: `memwatch.py` の記録から、M(t) = MemAvailable(t) + Σ（自分の shard の RSS）(t) を作り、「直前の窓 w の中の M の最大値 − M(t)」の、記録全体にわたる最大値。窓 w は E コアで測った最重量の 1 走の壁時計（その間に起きる落ち込みに、走り始めた shard が耐える必要があるため）。記録は段階 0 の検査・煙走・資源の走の間（1 時間程度）から始め、本走中も更新し続ける（本走中に大きな落ち込みが出れば、それ以後の起動と見張りの境界が自動的に厳しくなる）。常駐セッションの RSS は MemAvailable にすでに入っているので別に足さない。
- C_free = 論理 28 − `ps` で数えた CPU 張り付きのプロセス数。物理 20 コアを超えない。
- launch.sh は起動の直前に MemAvailable と SwapFree を読み直す。P < 1、SwapFree < RSS_peak（あと 1 shard が swap に押し出されたら受け止められない）、または wcap・act_chimera のプロセスがいれば起動しない。OMP/MKL=1、nice、explicit shard 表で xargs -P P、provenance.json がある shard は飛ばす。`_smoke` の行や強制フラグの行があれば拒む（S25）。
- **走行中の見張り `memwatch.py`**（2 秒ごと、S17 で検査）:
  - 境界 1（MemAvailable < RSS_peak + ΔM_desk）: 新しい shard を起動しない。最新の shard に SIGSTOP を送る。**SIGSTOP はメモリを解放しない**（止めた shard の一時確保の増加を防ぎ、カーネルがページを追い出せるようにするだけ）と明記する。
  - 境界 2（MemAvailable < RSS_peak。1 shard 分の余地も無い）: 最新の shard に SIGTERM を送り、その部分出力を `_killed/<時刻>/` へ移す。その shard は後で丸ごと走り直す。
  - 復帰（MemAvailable ≥ 2·RSS_peak + ΔM_desk。止めた shard の一時確保 1 回分と境界 1 の余白を足した算術）: 止めた順と逆に SIGCONT。
  - 止めた・殺した時刻、PID、shard 名、そのときの MemAvailable と SwapFree をログに書き、summary の資源の節に写す。
- 例（推論）: ΔM_desk が 2 GiB と測られ、MemAvailable が 12.8 GiB なら ⌊10.8/1.1⌋ = 9。wcap 終了後に 20 GiB なら ⌊18/1.1⌋ = 16 → C_free で頭打ち。

### 12.3 見積り（推論）

**予算の出所は段階 0 資源の直接測定だけ**にする。以下は、その測定までの推論の見積り（E コアは P コアの約 2.5 倍、§12.1 の 0.53 s 対 1.32 s から）。

1 走の見積り（箱 B、P コア／E コア）:
- 学習と測定 120 × 0.53 s = 64 s
- fresh 26 × 0.41 s = 11 s
- 窓内の帳簿（簡素版の Q、2 層、15,625 更新 × 2 × 0.57 ms を上界）≈ 18 s
- 層 2/3 の per-unit 記録（C.measure と同じ程度、120 × 0.073 s）≈ 9 s
- PD 24 × 0.41 s = 10 s（twin 以外）
- capacity 24 × 0.41 s = 10 s と CT2 の群除去（2 snapshot × 2 層 × 約 66 群の test 評価）≈ 8 s（capacity 8 腕 = N06・P06・V06・SNA・SNAP・SNAV・LR・LIN。CT2 はそのうち 6 腕）
- → **capacity 腕 約 130 s／325 s、その他の腕 約 112 s／280 s、twin 約 102 s／255 s**

SGD 箱の 1 走: 1 タスク = 4 × 0.396 + 0.073 + 0.011 + 0.047 = 1.715 s → 120 タスクで 206 s。2,500 更新の probe は 1 本 ≈ 4 × 0.41 = 1.64 s で、fresh 26 本 43 s・PD 24 本 39 s・capacity 24 本 39 s。窓内の帳簿は 62,500 更新 × 2 × 0.57 ms ≈ 71 s。層 2/3 の記録 9 s。→ **約 400 s P／1,000 s E**（旧草案の 250 s／620 s は probe を 625 更新で数えていた）。

走数（19 腕）: 17 腕 × n_main + 2 twin × 10。n_main=20 で 360 走、n_main=30 で 530 走。段階 1 と 2 の 73 走を含む。

| 段階 | 走数 | core-h（P–E） | 壁時計（P=8–12） |
|---|---|---|---|
| 0 検査・煙・資源 | — | ≈0.5 | ≈1–1.5 h（人手を含む） |
| 1 錨 | 12（capacity 腕） | 0.43–1.1 | ≈5–10 分 |
| 2 較正 | 61（capacity 腕 41 + twin 20） | 2.1–5.1 | ≈10–40 分 |
| 3 本走（n_main=20） | 残り 287（capacity 107 + その他 180） | 9.5–23.7 | ≈0.8–3.0 h |
| 3 本走（**n_main=30、予算の基準**） | 残り 457（capacity 187 + その他 270） | 15.2–37.9 | ≈1.3–4.7 h |
| R RL 予備走 | 40（Snake 30 + 自前の LR 10。LR は条件付きでなく予算に数える） | 9.3–13.3（1 走 14–20 分） | ≈0.9–1.7 h |
| S SGD 橋 | pilot 25（40 タスク・probe なし）+ 本走 50 | 0.5–1.2 + 5.6–13.9 | ≈0.5–1.9 h |
| **合計（n_main=30、S なし）** | **≈570 走** | **≈27–58** | **≈3.5–8.7 h**（2–3 晩に分ける） |
| **合計（n_main=30、S あり）** | **≈645 走** | **≈33–73** | **≈4–10.6 h** |
| 参考: 合計（n_main=20、S なし） | ≈400 走 | ≈22–44 | ≈3–7 h |

**縮小規則（登録）**: 段階 0 資源の実測で、段階 3（n_main=30）の壁時計の見積りが Issa の承認した壁時計（§16-5）を超えるなら、腕と seed には触れず、REPORT だけの項目を次の順に削ってから段階 1 に進み、削った項目を addendum に書く: (1) capacity の X_NR 変種、(2) capacity 全体、(3) PD、(4) 窓内の帳簿を t101–120 だけにする。登録ラベルの入力（E1–E3、M1–M3、CT1–CT3、G1）は削らない。それでも超えるなら段階 1 の前に Issa に諮る（seed や腕を自動では減らさない）。

ディスク: 1 走あたり per-unit 2 層で 2.5–3 MB、snapshot 3 個で 3.3 MB → 530 走で約 3.2 GB（360 走で約 2.2 GB）。git には判定の入力（seed 単位の小さな派生ファイル・1 走数十 KB）と、G1 錨腕の完全な配列だけを入れる。残りは obsidian-research-data に置き、manifest を commit する。

## 13. 実装計画

worktree: `wt/snake_phase_mnist_0914`、branch: `claude/snake_phase_mnist_0914`（origin/main から。`proj_004_drift` 自体は main のまま）。data/mnist は gitignore されているので symlink を張り、`T.data_dir` で sha256 を照合する（boundary_groups_0908 が H.DATA_DIR を固定しているので、解決したパスを provenance に書く）。

**committed の module は 1 行も編集しない**（宿主 sha 53e2c102… を provenance に固定）。次のものは使わない:
- `GS.run`: SPEC の場所が 実行済み/ に移ったため、provenance を書くところで FileNotFoundError になる。
- `EG.hdefect`・boundary_groups の credits: θ=0・h=0 を決め打ちしている。
- `GS.check_dphi`: 内部で `GS.make_act(arm)` を呼ぶので、新しい腕名（N06 など）は make_act に無く KeyError になり、そもそも実行できない（`src/gate_shape_0911.py:115–127`）。仮に腕名を通しても、変異対照は腕名が SN06/SNA 以外なら Snake(.6) になり、N06 と同じ関数なので空虚になって、`GS.run` の assert `g2_dphi_mutctl > 1e-3`（l.307）で止まる。本 spec の S2 で置き換える。
- `T.finite_guard` の assert 経路・`C._assert`: 出力を書く前に止める。

| 追加するファイル | 内容 |
|---|---|
| `specs/spec_snake_phase_mnist_0914.md`・`configs/snake_phase_mnist_0914.yaml` | 本 spec と定数。commit 件名 `[snake_phase_mnist_0914] spec（実装前・走る前）: Snake の位相と起点/オフセットは箱 B の時間劣化と幅の成長を変えるか`。末尾に帰属の行 |
| `src/snake_phase_acts_0914.py` | PhaseSnake（θ ∈ {0, +π/2, −π/2}・α・q）。S3 の変異用に `phase_shift` 引数を持つ。OffsetSnake、OffsetLeaky、AdaptivePhaseSnake（**`GS.H.AdaptiveSnake` = `src.pmnist_boundary_host_0908.AdaptiveSnake` の継承**。`src.pmnist_0905` からは継承しない）、init map（comp・to_phase・twin を float64 で計算して 1 回だけ cast。SNAi_P/V は適応 normal に to_phase）、定数 s・K・π/α（§3.2 の式から float64 で生成）。ローカルの make_act と restore。RL のラッパー用の固定 α の PhaseSnake は宿主のクラスに依存しないので、どちらの宿主にも差し込める |
| `src/snake_phase_mnist_0914.py` | GS.train（`src/gate_shape_0911.py:184`）の hook 付き複製。optimizer は adam/sgd と epochs（SGD の再歩行は `src/pmnist_0905.py` run_one から写す）。測定は C.measure（`width_sink_clamp_0909.py:167–229`）＋ gate_block の複製（`gate_shape_0911.py:137–168`。GS の古い大域変数を provenance に漏らさない）＋ 層 2/3 のブロック（Aᵢ・āᵢ・出力列ノルム・ce0/ce20）。ほかに fresh probe（`pmnist_boundary_host_0908.py:431–485` の _train_steps を移植し、予算を箱に合わせる）、PD、capacity、群除去（`unit_triage_0911` と CondA の群除去の型）、タスク単位の帳簿と窓内の Q、snapshot（t20/t60/t120 と sha256）。rows は毎タスク flush、units は追記、try/finally で書き出し、provenance が最後（git_hash・宿主 sha・spec sha・acts sha・torch・CPU 型・コア番号の系列・RSS の系列・ストリームの hash） |
| `src/snake_phase_rl_pilot_0914.py` | `src.pmnist_0905.ARMS` に PhaseSnake を差し込み、無改変の `pmnist_rlmnist_0906.main` を **`--arms <arm> --seeds <s> --device cpu --out results/snake_phase_mnist_0914/rl/<arm>_s<s>`** で呼ぶラッパー（1 プロセス = 1 腕 × 1 seed）。import 前に OMP/MKL=1、import 後に `torch.set_num_threads(1)`。`RL.evaluate_rl` を包んでタスク末ごとに層 1・2 の cnormᵢ・mᵢ・bᵢ を units.npz へ書く（読むだけ）。`sidecar_provenance.json`（run id、acts の sha、θ・α、ラッパーの sha、ランナーと宿主の blob sha、device、スレッド数、git_hash）を書く。殺された seed は丸ごと走り直す（無改変の main は seed の走が終わるまで per_task.csv を書かない） |
| `analysis/snake_phase_mnist_0914/checks.py` | S1–S25。`analysis/shell_l2_rlmnist_0913/checks.py:70–126` の変異の枠組み（アンカー文字列がちょうど 1 回現れる置換、run_check）を写す。`checks.json` を出す |
| `analysis/snake_phase_mnist_0914/gate0.py` | 段階 2 の盲検の較正。h_E（分解能マージン）、σ̂_U（3 池）、n_main、TESTABLE、BROKEN/DIVERGED の数と停止規則の適用、双子の軌道の検査の結果、双子の非相関の報告（後期のタスク別残差の相関が、同じ seed の別腕間の相関（事後 0.15–0.75・中央値 0.50）以下か）。出力の欄は S19 で検査 |
| `analysis/snake_phase_mnist_0914/verdict.py` | E1–E3、M1–M3、F1x の加法性、§6.4 の総合クラス（優先順位つき）、§6.6 の答え、§6.7 の CT 族、族別 Holm、正確な符号反転検定とその反転 CI、S14・S22 の合成モード。`--src` で煙走のディレクトリだけを指せる。本走ディレクトリに `--partial` は使わない |
| `analysis/snake_phase_mnist_0914/rl_gate.py` | 段階 R のラベル（S20 で検証） |
| `analysis/snake_phase_mnist_0914/memwatch.py` | §12.2 の見張り（記録モードと制御モード。S17 で検証） |
| `analysis/snake_phase_mnist_0914/{launch.sh, launch_rl.sh, smoke_resources.sh}` | `bash -n`、起動条件（checks all_pass・commit 済み・push 済み・資源の式・SwapFree・他実験の不在・`_smoke` の行と強制フラグの拒否）、date、プロセス数の確認、平らなログ名。smoke_resources.sh の出力先は `results/_smoke_snake_phase_mnist_0914/` に固定 |

vault: `spec/Snakeの位相をMNISTへ_spec_0914.md`（MCP、明示したファイルだけを commit。作業ツリーには無関係な変更済み spec が 3 つあるので巻き込まない。編集前に git log を見る）、現在地に 1 行リンク、結果は `測定/` に別ノート。

## 14. 主軸との関係・衝突回避・逸脱

### 14.1 閉じる問い

- **主**: [[W増大メカニズム_0909]] §5.4（成長指数の活性化依存）の Snake 位相族について答える。
  - 原点の幾何（傾き 1/2/0）と定数オフセットは、CE・Adam で第 1 層の ‖W̃ᵢ‖ の成長と時間劣化を変えるか。
  - 変えるなら、起点（初期関数）と根の位置・活性値の水準のどちらか（厳密な分解）。
  - 位相の効き目が残る適応族で、位相の構造は効くか。
- あわせて、CondA で見えた Snake の現象（peak の成長抑制・群除去の偏り・根への着座）が、**LoP のある箱で存在するか**を、§6.7 の CT 族の登録ラベルで答える（E1・E3 の有意性に関係なく出る）。答えられる範囲は ‖W̃ᵢ‖ についてであり、CondA の見出しの量（自由重み u の成長比）そのものは MNIST に類似物が無いので検定しない（§17）。個体の分岐は Adam では REPORT、SGD 橋でだけ FORK_* のラベルを持つ。「決着」とは書かず、「CT 族のラベルが出た範囲で答えた」と書く。
- [[幅の規制は可塑性を買うか_0910]] と W病理（‖W̃ᵢ‖ の成長が時間劣化の病理か）には、**後続実験 M の腕内の幅操作**でだけつなぐ。本 spec の腕間の幅比較（CT1、§8-6 の REPORT）は W病理の証拠にしない。根拠: 腕間では幅が損失を並べない。`gate_scale_invariance_0909` の 8 腕（t120）で Spearman +0.07・Pearson +0.01、幅 7.4 に抑えた leaky（wcap2）は幅 8.7 の Snake より 1 pt 悪い（[[幅の規制は可塑性を買うか_0910]] §1c′ l.44）。gate_shape_0911 の 13 腕でも Spearman(N, L) = +0.01（summary §2 A6、`WIDTH_NOT_BETWEEN_ARMS`）で、GELU は N 6.39 と Snake 族（5.78–6.76）並みの幅で L 4.33 と 13 腕中最大の劣化（summary §1）。
- 閉じないもの: 駆動源問題、φφ′ 均衡、自己項による復元、SGD/MSE での CondA の分岐の原因、L2 系。

### 14.2 W病理チャット（wcap_rlmnist_0914）と act_chimera_0913 との衝突回避

- 主の箱は Permuted 箱 B。W病理は Random Label の 8 腕（LR × ref/l2/l2init/capT1/cap2、R × ref/l2/cap2。wcap の checks.json の S-cost と S1 の腕）。
- RL の予備走は Snake 3 腕と、陽性対照の LR（既定は自前の 10 seed、§4.2）。wcap の LR_ref を代わりに使うのは、wcap の verdict が commit・push され、かつ S18 の LR の直接照合が通ったときだけ。**wcap の per_task.csv は、その verdict の commit まで読まない**（他のチャットの走行中で判定前の結果を、こちらの判定の入力にしないため）。l2・cap・ReLU の腕は置かない。
- RL の指標は W病理の定義（G・P・D・傾き）に揃える。**RL の fresh gap は wcap と同じ G = A(1) − A(31–50)** で、段階 R のラベルは G の経路（RL_GO_FRESHGAP）を持つ（§11）。箱 B の E3（新しい Adam での fresh probe）は箱 B 用の定義で、RL の G とは別の量。
- white-san の時間は、wcap の本走が終わってから使う。act_chimera の段階 1（箱 B・RL）とは Issa の決めた順に（§16-2）。活性化を差し込む module の重複は、act_chimera の spec が commit された時点で共有するか判断する。

### 14.3 [[零点への復元はW増大を抑えるか_検証設計_0913]] §4C からの逸脱（登録）

1. 第 2 段階（同じ MSE 課題で SGD/full-batch/Adam）を飛ばす。CondA の Snake は MSE で天井のため LoP を検定できない。代わりに CE MNIST の中で optimizer だけを変える SGD 橋（段階 S）を置く。
2. α=1 ではなく固定 α=0.6（箱 B で Snake の劣化が最大・錨あり）と、適応 α 族。α=1 は SGD 橋だけ。
3. 代表 6 条件ではなく、Snake 3 位相・固定 α の 2×2 分解（枝 1 点を含む）・適応位相 3 と起点の角 2・対照 4（LIN・LR・LR のオフセット 2 符号）。Leaky−.5（a=.7）は LR07q-S として SGD 橋へ。
4. L2 の対照は置かない（ノルム抑制の別経路は W病理チャットの領分。本件では後続実験 M の WMATCH で扱う）。
5. seed 0–9 ではなく、算術で決める n_main（10–30）。
6. 主な窓は箱 B の慣例に合わせて t16–30 と t101–120。設計ノートの窓（task1 / 2–20 / 21–40 / 41–100 / 101–119 / 120）はすべて副次として出す。
7. **帳簿の粒度**: 設計ノートは「各 1 step で W̃・全 W・bias について Δ‖w‖² = 2⟨w, Δw⟩ + ‖Δw‖² を直接記録する」と定めるが、本 spec はタスク単位の厳密な帳簿（N・D・c）と、t16–20・t101–120 の窓内の 1 更新ごとの Q だけを取る。理由: 1 更新ごとの完全な帳簿は重い版で約 2 ms/更新（§12.1）で、全 120 タスク × 625 更新では 1 走に約 150 s 余計にかかり、走数（最大 530）に掛けると予算が倍になる。step 単位で「自己項は内向きでもタスク全体では外向き」を調べる問いは本 spec では答えない。
8. **主判定の連鎖と効果の分類の置き換え**: 設計ノートの主判定（連鎖 1: 中心の局所復元係数が正で複数 ε で符号が安定、2: 同一ユニットの動径項が内向き、3: peak でも同じ連鎖、4: 縮小が出力消失だけでないこと）は、LoP の端点（E1–E3）と CT 族に置き換える。理由: 連鎖 1–2 は CondA の 1 層・スカラー出力・MSE・SGD で「中心」と「自由重み」が定義できることに依っており、CE・Adam・2 層の箱には同じ量が無い（Adam の 1 更新は勾配の大きさを正規化し、中心の局所係数が更新の向きを決めない）。連鎖 4 の「出力消失だけでないこと」は CT2（群除去）と出力側の列ノルムの REPORT で部分的に受ける。6 つの効果の分類の対応は §6.7 末尾に登録した。
9. **固定の出力補正による機序の比較を落とす**: 設計ノートの「任意入力で計算する固定関数の補正（人工的な応答介入）」は本走に混ぜない（§3.3-5）。理由: 初期関数の交絡は init map で厳密に揃えられ、補正関数は学習中に活性化と一緒に変わらないので、補正そのものが新しい交絡になる。必要なら後続実験 M に入れる。
10. **Adam の 1 更新の診断を落とす**: 設計ノート第 2 段階の「通常更新」「m 履歴なし」「実現分母固定」の 1 step 診断は行わない。理由: 第 2 段階（MSE 課題での optimizer 比較）自体を飛ばしたため（逸脱 1）で、CE の箱で同じ診断をするには snapshot からの分岐計算を全腕に足す必要があり、本 spec の問い（時間劣化と ‖W̃ᵢ‖ の成長）の判定には使わない。そのため効果の分類の「全勾配までは縮むが optimizer 後に反転する」は測らない。

## 15. 引用禁止・限定（結果前に明記）

1. 固定 α=0.6 の後期窓の結果から、「位相（零点の構造）は LoP に効く／効かない」と書かない（M1 が PHASE_WASHED のとき）。書けるのは「起点」「根の位置と活性値の水準」の効果だけ。
2. 「peak の固有オフセットは K_P = −2.142」「peak = normal + オフセット」と、(s, K) を一意に書かない（枝 k で変わる）。
3. CondA の数値（70.8% の収縮・比 .906・99.97%・群除去）は condA/w100/T=10⁴/batch=1 に限った**仮説**（[[引用禁止]] E）。CondA の自由重み u と MNIST の ‖W̃ᵢ‖ を同じ量として比べない。CT1 のマージン ±0.09877 は物差しとして借りるだけで、「CondA と同じ大きさの効果が MNIST にある／ない」とは書かない。
4. Adam 下の「初期値より縮んだ個体 0%」は算術上ほぼ自明なので、分岐が無い証拠に数えない。
5. ω 則（1.05 pt/倍）は leaky の 15 点・ω 4 倍幅の当てはめ。位相腕と D 窓への適用は標本外。本 spec の閾値にはもう使わない（§7.1）。
6. 腕間の幅で損失を並べない。CT1 と §8-6 の幅の差を「W に媒介された」「W に媒介されない」と書かない。
13. EQUIVALENT は「単走の分解能 h_E より小さい」であり、「効果が無い」「実用上同等」と書かない。
7. 本件の「valley Snake」（φ′ ≥ 0）は、[[谷の符号梯子_結果_0912]] の VALLEY_CAUSAL（GELU/SiLU の φ′ < 0）と別物。
8. F_self + F_rest の分解、φφ′ 均衡、自己項による復元を根拠にしない（自己項は rest と打ち消し合う: CondA natural trace）。
9. mob（Ḡ）が LoP を決める、と書かない（[[引用禁止]] B）。「Snake のユニットは φ′ の零点に位相ロックする」と書かない（[[PermutedMNIST_追加診断_0905]] §5）。
10. RL の NO_GO は「400 epoch・この 3 腕」に限る。「Random Label では Snake は劣化しない」と一般化しない。
11. 群の将来学習（§8-4）は記述だけ。「小さい個体は学習力を保つ／失う」を主張しない。
12. 本 spec 中の「事後」の数値は登録結果ではない。結果ノートでは verdict.csv の値だけを窓ラベル付きで引く。

## 16. Issa に決めてもらう点

1. **段階 S（SGD 橋・α=1・5 腕 × 10 seed・6–15 core-h）を本 spec に含めるか。G1 の commit までに決める**（後から承認するなら別 branch・別 spec、§11 ブランチの扱い）。含めないと、Adam で分岐が見えなかったことを optimizer のせいにできない（CondA の分岐は SGD の結果）。含めると、損失とデータを変えたまま optimizer だけを比べられる。
2. **white-san の順番。** act_chimera_0913（箱 B 51 走 + RL 15 走・spec 未 commit）と本 spec のどちらを先に回すか。wcap は早くても 05:30 ごろ終わる見込み。本 spec は箱 B で 1.3–4.7 h（段階 3、n_main=30）、RL で 0.9–1.7 h の壁時計を使う。
3. **Codex の設計ノート 0913 §4C 第 3 段階を本 spec が引き取ってよいか**（逸脱 10 点は §14.3）。よければ Codex チャットに伝える。
4. **RL の範囲。** 400 epoch の予備走で NO_GO のとき、80 epoch の予算で追加の予備走（新しい RL の箱で、W病理チャットの領分と重なる）を許すか。許さなければ、RL の答えは「400 epoch では器にならない（この 3 腕）」で止まる。RL_GO のときの後続 spec（§11）を起こしてよいか。
5. **n_main の上限 30 と承認する壁時計**（段階 3 の最悪で約 38 core-h、壁時計 1.3–4.7 h、S なしの合計 27–58 core-h・3.5–8.7 h）。上限を下げると、h_D での EQUIVALENT が出にくくなる（UNDERPOWERED_FOR_RESOLUTION）。事後の見込みでは n_main は 14–31 のどこにも落ちうる（§7.2）。承認した壁時計を段階 0 の実測が超えたら §12.3 の縮小規則を当てる。
6. **腕の追加 3 本（SNAi_P・SNAi_V・LR_qKpn）**。n_main=30 で +90 走・約 3–7 core-h。SNAi がないと「位相の構造」を起点から切り離せない（F2 の総効果だけになる）。LR_qKpn がないと Leaky のオフセットは負の 1 符号だけになる。

## 17. 限界（結果前に明記）

- CondA から損失（MSE→CE）、データ、深さ（1 層→2 層）を同時に変えている。SGD 橋が分けるのは MNIST の中の optimizer だけ。
- 活性化は隠れ層 2 層とも。層 1 だけの版は後続実験 M。
- 120 タスクまで。Snake の遅い動き（t301–400）は見ない。
- 適応位相族では**厳密な分解（起点 × オフセットの恒等式）は存在しない**（t>0 で αᵢ が動くと s・K が合わなくなる）。一方で**初期関数を揃えることはできる**（SNAi_P・SNAi_V。宿主 AdaptiveSnake は V=1 から始まり t=0 で αᵢ=0.6、V の更新は平行移動で不変な var(z) だけを使う）。SNAP − SNAi_P は「同じ初期関数から出発したときの位相構造の効果」で、固定 α の K 因子と S 因子への分解ではない。
- 枝は peak の k=0, −1 だけ。valley の他の枝（水準 +5.71）は走らせない。
- h_E は単走の分解能で、科学的な最小効果量ではない。M1・M3 の比較（位相の上界 対 基準の IQR）は選んだ比較で、損失への帰結から導いたものではない。メモリの余白 ΔM_desk は段階 0 以降の記録の範囲の落ち込みしか見ておらず、それより大きい落ち込みには見張りの境界 2（SIGTERM）でしか備えられない。
- F1x の交互作用 I の等価性には検出力が無い（§6.1）。「起点とオフセットは加法的」とはほぼ書けない前提で §6.6 の答え方を決めている。
- 双子（P06tw・V06tw）が 120 タスクで非相関化する保証はない。σ の較正と実装の検査にだけ使う。双子の軌道の検査は task 1 の 20 更新目で、浮動小数の分岐と実装差の桁が分かれない場合は識別できない（TWIN_CHECK_NOT_IDENTIFIABLE）。
- **CondA の自由重み u に当たる量は MNIST に無い**。CondA の u は「課題に情報を持たないが 0 でない入力（ランダムビット）」への重みで、勾配の雑音で育つ。MNIST で候補になる「全タスク画像で 0 の画素への重み」は勾配が 0 なので、SGD では凍結し、Adam では持ち越した moment で動くだけで、雑音で育つ量ではない。雑音画素を足すと箱 B が変わり錨を失う。したがって CondA の見出し（自由重みの成長比 .906）そのものは検定しない。CT1 は同じ向きの問いを ‖W̃ᵢ‖ について立てるだけ。
- **根への絶対の着座は判定しない**（CT3 は腕の相対比較だけ）。LIN を零にすると帯の占有率が飽和し、位相を一様にずらす零分布はゲートの位相への揃いを検定するだけで根への着座と区別できない。
- **Leaky のオフセットは Adam の箱で a=.1・\|q\|=2.142 の 2 符号だけ**。CondA の大きさ \|q\|=.5 と a=.3/.7 への依存は Adam では検定しない（a=.7 q−.5 は SGD 橋だけ）。
- probe の 512 枚は test 集合と共有（箱 B から継承）。
- 群の将来学習は 1 タスク予算・入力の可塑性だけを見ていて、ラベルの可塑性は測らない。
- 後続実験 M の幅合わせ射影は未実装・未検証で、非侵襲かどうか分からない。

---

### 付録 A: 本 spec 起草時の事後計算（再現用）

- `synth_check.py`:
  - SN06・SN02・SN15・SNA の位相振幅と |ḡᵢ−1|（t1/20/60/120）。
  - L と窓平均 D の seed 別の値。
  - 恒等式と枝の検算。
- `synth_pair.py`:
  - 9 腕（SN06・SN02・SN15・SNA・LR・LR03・LR001・LIN・ELU1）× 3 seed の D_pair、seed 内のタスク標本 SE 0.14–0.23 pt。
  - 36 対でプールした SD 0.24 pt。
  - 後期残差 SD 0.40–0.91 pt、腕間の残差相関 0.15–0.75（中央値 0.50）。
- 追加の確認（本セッション）:
  - SN06・SNA・SN02・LR・LIN の t1 終端の cnorm 最小値 / √(1/3) = 1.41–1.70。
  - zcur 中央値の軌道（SN06 s0: t1 +0.11 → t120 −1.02、SNA s0: +0.11 → −1.21、LR s0: +0.60 → −4.42、LIN s0: ≈0 → +0.11）。
- `verify_rev1.py`（改訂時。gate_shape_0911 units と `src.pmnist_0905.init_params(seed)` の層 1 の中心化行ノルム c0。MNIST は読まない）:
  - 成長比の対数のユニット間 SD（t120、実 c0 基準。旧草案の 0.023–0.040 は名目値 √(1/3) で割った値で、実質 SD(log c120)）: SN06 .032/.032/.031、SNA .038/.033/.028、SN02 .029/.027/.032、SN15 .023/.025/.029、LR .041/.041/.039、LIN .037/.042/.042、GELU .131/.135/.136、ELU1 .284/.201/.250、R .543/.422/.458。計算: `np.log(c120 / c0).std(ddof=1)`。
  - 残差化三分位の RMS ノルム比 − 1: log c120 を log c0 に 1 次で回帰した残差の下位 33 と上位 33 で √(mean c120²_HIGH / mean c120²_LOW) − 1。SN06 .066/.067/.054、SNA .073/.067/.058、LR .081/.079/.073、LIN .047/.046/.050。
  - 置換した c0 での比どうしの Spearman（20 回平均）: SN06 .07/.12/.13、SNA .02/.13/.22、SN02 .06/.07/.03、SN15 −.01/.25/.11、LR .09/.05/.15、LIN .08/.30/.41。
  - 位相の振幅 medianᵢ exp(−2α²sdcurᵢ²)（SN06）: t1 .644/.647/.666、t60 .248/.254/.246、t120 .092/.099/.091。
- `verify_rev2.py`（改訂時。mpmath で χ²・t の分位）: χ²_{18,.20} = 12.857（係数 1.183）、χ²_{27,.20} = 20.703（係数 1.142）。n_main（δ=.24）は σ̂_U .284 で 21、.274 で 20。F1x の I の SD ≈ .38、n=20 の反転 CI の半幅 .196、真の I=0 での ADDITIVE の確率 .40（δ=.24）、n=30 で .77。
- `verify_rev3.py`（改訂時。gate_shape_0911 rows）: 分解能 h_D の RMS 0.192（0.182–0.204）、h_Alate 0.111、h_Abase 0.157（SN06−LIN・LR−LIN × seed 0–2）。対差 D の SD: SN06−LIN 0.154、LR−LIN 0.118。単腕 D の SD: SN06 0.141、LR 0.038、LIN 0.097。n_main: σ̂_U .284・h .19 で 31、σ̂_U .18・h .19 で 14。
- 改訂時の単発の確認（スクリプトは残していない）: P4 の丸め前の D（SN06 rows）、RL の SNA none の窓平均と傾き（origin/main の per_task.csv）、gate_shape_0911 の wall_seconds、CondA group_summary.csv の ALL 行の縮み割合（eps 別）、α=.6 の s・K の float64 値、2 つの宿主モジュールの isinstance、φ = z + sin²(αz)/α（α=.2/.6/1）が z<0 に根を持たないこと（z∈[−20, 0) の格子）。

---

## 草案の検証記録（2026-09-14、指摘 41 件）

各指摘をその証拠に当たって確かめてから直した。番号は受け取った指摘の順。「一部却下」は、指摘の結論は採ったが根拠か修正案の一部が誤っていたもの。

### 直したもの

1. **CondA の eps は縮みの許容幅ではない**（major）: 確認した。`zero_attraction_report_0913.py:105–108` で eps は原点近傍群を選ぶ半径で、縮みは `n2[-1] < n2[0]`。group_summary.csv の ALL 行の割合は eps=.05/.1/.2 で同じ（LIN .927、LR a.7 q−.5 .826、peak q0 .708、normal q0 .271）。f_shrink を strict な縮みで定義し直し、§4.3・§8-2・§8-3・§11 段階 S から「eps .05」を削った。S21 を足した。
2. **§2・§12.1 の走時間の範囲**（minor）: 全体の範囲が内訳と矛盾していた。52–124 s に直した。修正案の「seed 0 は対照込みで 81–124 s」は一部却下（下記）。
3. **α=0.6 の K の丸め**（minor）: 確認した（K_P −2.142330、K_P+π/α +3.093657、K_V +0.475664）。§3.2・§4.1 を直し、config の 17 桁は式から生成すると明記した。valley の k=−1 の (s, K) も書き足した。
4. **2 つの宿主モジュール**（minor）: 確認した（`GS.H is pmnist_0905` → False、isinstance → False、ファイルはバイト一致）。継承元を `GS.H.AdaptiveSnake` と明記し、S3 を `GS.H.forward` 経由の比較と継承元の変異に直した。
5. **LR_ref の bit 一致の検査は存在しない**（minor）: 確認した（wcap S4 は 0913 ランナーとの照合、shell_l2 S4 は R と SNA だけ）。指摘 34 と合わせて、既定を自前の LR × 10 seed にして予算に数え、wcap の LR_ref は verdict の commit と S18 の直接照合が揃ったときだけ使う規則にした。
6. **成長比の対数の SD の分母**（minor）: 確認した。実 c0 で再計算（Snake・LR・LIN 0.023–0.042、GELU 0.131–0.136）し、計算方法を付録 A に書いた。P14 の閾値 0.13 は変わらない。
7. **P4 の区間**（minor）: 確認した。丸め前の D で [1.955, 2.599] に直し、addendum でも丸め前の D を使うと明記した。
8. **群間のノルム差 4–6%**（minor）: 確認した（LIN 4.6–5.0%、SN06 5.4–6.7%、SNA 5.8–7.3%、LR 7.3–8.1%）。「約 5–8%」に直した。
9. **RL の傾きの窓**（minor）: 確認した（t31–50 で +0.00031、t11–50 で +0.00011、どちらも 0/10 が負）。窓を明記した。
10. **§14.1 の GELU 反例の出所**（minor）: 確認した（l.44 は Spearman +0.07 と leaky wcap2 対 Snake。GELU は l.60 だけ）。出所を l.44 の内容と gate_shape_0911 summary §2 A6・§1（GELU N 6.39、L 4.33）に分けて書いた。
11. **池 1 の自由度**（minor）: 確認した（N06−LR は線形従属、係数 1.142 対 1.183）。池 1 を独立な 2 対比・df 18 にした。σ̂_U .274 → .284 で、δ=.24 なら n_main は 20 → 21。
12. **費用見積り**（minor）: 確認した（窓内の帳簿は重い版なら層 1 だけで約 31 s、段階 S は probe を 2,500 更新で数えると 1 走 300–340 s 以上、n_main=30 の壁時計の下端 0.9 h）。帳簿を簡素版の Q・2 層に固定して 1 走を数え直し（capacity 腕 130 s／325 s など）、段階 S は帳簿も含めて約 400 s／1,000 s、腕の追加も含めて表を作り直した。指摘 40 と合わせて、実測だけを予算の出所にし、縮小規則を足した。
13. **§14.2 の wcap の腕**（minor）: 確認した（S-cost のキーは 8 腕）。LR 5 腕・R 3 腕と書いた。
14. **P6 の「t60 までに洗い流される」**（minor）: 確認した（SN06 の振幅 t1 .64–.67、t60 .25、t120 .09–.10）。「t60 で t1 の約 4 割、t120 で約 1.5 割」に直した。
15. **S2 の変異 (c) の √2**（minor）: 確認した（2|sin(Δθ/2)|）。変異ごとの解析値（√2 または 2）を書いた。
16. **33 ユニットの 4 群は分割にならない**（minor）: 確認した。LOW(33)・MID(34)・HIGH(33) が分割し、RAND1・RAND2 は互いに素と直し、S13 に MID を 33 にする変異を足した。
17. **boxB は 200 タスク・CUDA**（minor）: 確認した（provenance）。§4.3 に lr・epoch・再歩行だけが同一でタスク数とデバイスが違い、bit の錨を持たないと書き、§2 の SN1 の値の出所にも書いた。
18. **RL ラッパーの device と出力先**（minor）: 確認した（`--device auto`、既定の出力先は追跡されている `results/pmnist_rlmnist_0906/` の中）。`--device cpu --out results/snake_phase_mnist_0914/rl/<arm>_s<s>` を固定し、S18 で検査する。
19. **層 2 の初期ゲート**（minor）: 確認した（層 2 の V06 は z2 ≈ b2、ゲート ≈ 0.002）。§3.3 を層別に書いた。
20. **置換した c0 の相関 0.09–0.26**（minor）: 確認した（再計算で約 0.0–0.4）。値を差し替え、スクリプト `verify_rev1.py` を付録 A に挙げた。
21. **GS.check_dphi を使わない理由**（minor）: 確認した（make_act に新しい腕が無く KeyError、通しても変異対照が空虚で l.307 の assert で止まる）。書き直した。
22. **CondA の現象に登録ラベルが無い**（major）: 確認した。§6.7 に無条件の CT 族を足した。CT1 成長（ΔlogN、層 1・2、C1/C2/C7/C8、Holm、マージンは CondA の登録の log 差 ±0.09877）、CT2 群除去（無作為群 B=64 に対する超過と、その腕で測った帯）、CT3 根への着座（相対の比較だけ）。CondA の効果の分類 6 つとの対応を書き、P18–P21 を足した。一部却下（下記）。
23. **δ_D が腕間の幅を前提にしている**（major）: 確認した（A6 の Spearman +0.01、SN02 は広いのに劣化が小さい）。修正案 (b) を採り、δ_D を単走の分解能マージン h_E（段階 2 の盲検の対のタスク標本 SE）に置き換え、EQUIVALENT の意味を「分解能未満」に限った。OMEGA_* と W_* のラベルは REPORT に下げ、腕間の比較で W 病理の証拠ではないと明記した。修正案 (a) は却下（下記）。
24. **F1x の交互作用の検出力と答えの経路**（major）: 確認した（I の SD ≈ .38、n=20 で ADDITIVE の確率 ≈ .40 at δ=.24）。修正案の 2 つ目を採り、主効果は I に関係なく CI 付きで出し、I は NON_ADDITIVE／ADDITIVE／ADDITIVE_INCONCLUSIVE とし、§6.6 に §0 の問いへの対応表（NOT_ANSWERABLE／NON_ADDITIVE／BOTH／LEVEL（細分つき）／ORIGIN／NONE／INCONCLUSIVE、適応族の STRUCTURE）を登録した。
25. **F2 は初期関数と交絡**（major）: 確認した（宿主 AdaptiveSnake は V=1 で始まり、V の更新は var(z) だけ）。SNAi_P・SNAi_V を足し、F2 を m=4（C7s・C8s）にし、「位相構造」の書き方の規則を C7s・C8s に移し、§17 を「厳密な分解は無いが初期関数は揃えられる」に書き直した。
26. **M3 の 2 状態**（major）: 確認した。CLEARED（算術上界と基準の IQR から導いたマージン Δu_m に CI が収まる）／TRANSIENT_IN_BASE／TRANSIENT_UNRESOLVED の 3 状態にし、S 因子を base 窓込みで読むのは CLEARED のときだけにした。
27. **RL のゲートが G を使わない**（major）: 確認した（wcap は G を主、P を REPORT にしている）。RL_GO_FRESHGAP の経路と LR_CONTROL_FAILED を足し、NO_GO に G の条件を足し、RL_GO の後に起きること（別 spec・別 branch・最低条件）を登録した。ユニット別の cnorm は `RL.evaluate_rl` の包みで記録し、S18 で非侵襲を検査する。
28. **判定の経路に S 検査が無い**（major）: 確認した。S14 をすべてのラベル・修飾・総合クラス・対応表・n_main の規則に広げ、S20（rl_gate）・S21（FORK）・S22（CT）・S23（双子の軌道）・S24（段階 2 の停止規則）・S25（煙走の分離）を足し、S6 (d)（SNA の V の持ち込み）と S8（PD の置換の未見性）を足した。双子の検査は有意性検定から、実測の 2 つの参照（浮動小数の経路差と K を落とした実装差）の対数の中点による task 1・20 更新目の判定に変えた。変異の一部は却下（下記）。
29. **予測の締切と抜け**（major）: 確認した。Issa と Claude の予測を G1 の commit に含める（段階 0 より前）と書き、P22–P27（E2・E3 の向き、壊れた走、M3、n_main、FS 族、RL の陽性対照）を足した。CT の予測は P18–P21。
30. **停止規則とメモリ**（major）: 確認した。段階 2 の停止規則 4 つ（TESTABLE の不成立、Clopper–Pearson と n_min から決める BROKEN の k、双子の検査）を登録し、4 GiB を実測の ΔM_desk に置き換え、swap を余裕に数えない規則と、走行中の見張り（境界 1 で SIGSTOP、境界 2 で SIGTERM、復帰の境界）を足した。修正案の「SIGSTOP でメモリを守る」は、SIGSTOP がメモリを解放しないので境界 2 の SIGTERM を足して補った。RL のラッパーの RSS は自前の煙走で測る。
31. **§4C からの逸脱の抜け**（major）: 確認した（設計ノートの 1 step の帳簿、主判定の連鎖 1–4 と効果の分類、固定の出力補正、Adam の 1 step 診断）。逸脱 7–10 を理由つきで足し、効果の分類は §6.7 の CT 族に対応づけた。
32. **総合クラスの排他性**（minor）: 確認した。優先順位（CONFLICT が先）、接尾の扱い、E2 の EQUIVALENT、NO_DIFFERENCE_RESOLVED に ΔA_late EQUIVALENT を要求、を足した。
33. **σ̂_U の池が位相腕を見ていない**（minor）: 確認した。単腕 D の SD から sd_θ + sd_ref で抑える効果に盲目な池 3 を足し、3 池の最大を取る。
34. **LR 陽性対照の条件**（minor）: 指摘 5 と同じ内容として確認し、合わせて直した。wcap の per_task.csv を verdict の commit まで読まない規則も足した。
35. **RL ラッパーの provenance・スレッド・途中停止**（minor）: 確認した（run_id が 0906、per_task.csv は seed の走の後にだけ書かれる、H.setup はスレッドを設定しない）。1 プロセス = 1 腕 × 1 seed、sidecar provenance、スレッド 1、殺した seed は丸ごと走り直す、を書いた。
36. **S1 の完全一致**（minor）: 結論を採った。S1 を学習状態の配列（0.0）と float64 の派生量（γ_n の上界）に分け、4 腕の s0 × t1–3 の照合を G1 の commit と P2 の確率の前に行うと書いた。根拠の一部は却下（下記）。
37. **煙走の出力先**（minor）: 確認した。`results/_smoke_snake_phase_mnist_0914/` に固定し、launch.sh の拒否を S25 で検査する。
38. **族をまたぐ多重性と見出し**（minor）: 確認した。見出しを E1 のラベルに限り、族をまたぐ多重性は制御しないと宣言した。
39. **Leaky のオフセットの符号と大きさ**（minor）: 確認した（Leaky ではオフセットがバイアス移動と等価でない）。LR_qKpn（q=+2.142）を足して F3 を m=2 にし、|q|=.5 と a=.3/.7 への依存は Adam では検定しないと §17 に書いた。
40. **帳簿の費用と縮小規則**（minor）: 指摘 12 と合わせて直した（予算の出所を段階 0 の実測に限り、縮小規則を登録）。最悪の core-h は腕の追加も含めて数え直した（段階 3 で最大約 38 core-h、S なしの合計 27–58 core-h）。
41. **ブランチの扱い**（minor）: 確認した。段階 3・R・S を同じ実験の登録された部分段階として一緒に main に入れ、段階 S は G1 の commit までに承認された場合だけ含め、それ以後の承認や RL_GO の後続は別 branch・別 spec にすると書いた。

### 却下したもの・一部却下したもの

- **指摘 2 の修正案の内訳**: 「seed 0 は対照込みで 81–124 s」は誤り。LR・LR03・LR001・ELU1・ELU03・GELU・SILU の seed 0 は 56.6–72.3 s で、81 s を超えるのは錨の対照を最初の錨タスクまで回す SN02・SN06・SN15・SNA・LIN の seed 0 だけ（gate_shape_0911 provenance）。全体 52–124 s と seed 1–2 の 52–101 s は正しいので、それを採り、seed 0 の内訳を正しく書いた。
- **指摘 22 (c) の「位相をずらす零分布に対する絶対の着座」**: 採らなかった。一様な位相ずらしの零分布はゲートの周期のどこに揃うかを検定するだけで、単調な Snake の根（z=0 に 1 つだけ）への着座と区別できない。LIN を零にすると帯の占有率が飽和する（事後 0.99）。CT3 は腕の相対比較だけを登録し、絶対の着座を判定しないことを §17 に書いた。
- **指摘 22 (d) の「全タスク画像で 0 の画素への重み」を自由重みの類似物にする案**: 採らなかった。CondA の u は 0 でない雑音入力への重みで勾配の雑音で育つが、0 画素への重みは勾配が 0 で、SGD では凍結し、Adam では持ち越した moment で動くだけなので、同じ機序の量にならない。修正案のもう一方（類似物は無く CondA の見出しは検定しない、と §17 に書く）を採った。
- **指摘 22 (4) の「P14 には判定規則が無い」**: 一部却下。P14 には反証条件（どれか 1 腕でも ≥ 0.13）があり、予測としては判定できる。量の定義（実 c0 基準・t120・層 1・n_main seed の平均）が曖昧だった点は採り、§8-2 と P14 に書いた。
- **指摘 23 の修正案 (a)**（α 梯子の劣化差 1.05/1.93/2.33 pt からマージンを作る）: 採らなかった。マージンが 1 pt 以上になり、EQUIVALENT がほぼ何でも通って情報を持たなくなる。修正案 (b) を採った。
- **指摘 28 の変異のうち「FORK の eps .05→.1」と「W のマージン ln1.170→ln1.17²」**: 指摘 1 で FORK が eps を使わなくなり、指摘 23 で W_* のラベルを廃止したので、変異の対象が無い。代わりに S21 で「strict な < を 0.95·c0 の許容幅に替える」変異、S9 で「ω 残差の傾き 1.05→1」「成長比の分母を √(1/3) にする」変異を置いた。
- **指摘 28 (e) の修正案（K を落とした双子で有意性検定の IDENTITY_TRAJECTORY_BREAK を発火させる）**: 一部却下。D の有意性検定は n=10 で検出力が低く、変異が発火しても通常の双子の検出力は上がらない。判定を task 1 の 20 更新目の軌道の差に移し、K を落とした双子を上の参照として使う形にした。
- **指摘 36 の根拠**: 一部却下。gate_shape_0911 の G1 に出た 4e−16〜1.8e−15 の差は、集計順の違う**別の実験の錨**（elu_growth_0909 など）との比較で出たもので（`gate_shape_0911.py` のコメント「zbar_i / sd_i are accumulated in a different order by elu_growth_0909」）、本 spec の S1 が照合する gate_shape_0911 自身の配列との比較ではない。ただし追加の測定コードで float64 の集計順が変わる危険は本物なので、修正案（2 群への分割と 4 腕での事前照合）は採った。
- **指摘 30 の修正案の「SIGSTOP で守る」**: 一部却下。SIGSTOP はメモリを解放しないので OOM は防げない。境界 2 で SIGTERM を送る段を足した。

### 検証で新たに分かり、あわせて直したこと

- ω 則の傾きの族依存は vault の現在地の 9/12 の記録の中でも揺れている（tau_clamp_0912 の −1.06 対 −1.59 のあと、band_omega_0912 で 625 更新固定なら −1.06 対 −1.00、turn_budget_0912 で ω 則は更新数固定の範囲だけ）。§7.1 に書いた。
- 新しい n_main の規則（h_D ≈ 0.19）では、事後の見込みが 14–31 に広がる。予算を上限 30 で立て直した（§7.2・§12.3・§16-5）。
- RL の SNA は固定 1200 枚の正の転移で G < 0 になる（wcap spec §1 の事後）ので、G の経路を足しても Snake の RL_GO は出にくい見込み。P16 の根拠に書いた。
- 旧草案の §2 は M1 を「§6.4」と参照していた。§6.5 に直した。
