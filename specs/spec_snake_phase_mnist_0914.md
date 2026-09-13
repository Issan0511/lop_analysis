# snake_phase_mnist_0914 spec：Snake の位相と定数オフセットを Permuted MNIST 箱 B へ移す（登録版・第 1 段）

状態: **事前登録（実装前・走る前）** / 更新: 2026-09-14 03:40 JST / 起草: Claude（Opus 5）
親: vault [[W増大メカニズム_0909]] §5.4 / 姉妹: [[零点復元と重み収縮_学習実験_spec_0913]]（CondA 側）
発端: Issa（2026-09-14 01:3x）「CondA で snake の議論がうまくできないので、これらの活性化を RandomLabel とか Permuted の MNIST でやるべきだよね」→「寝るので測定は全自動でお願い」「実行もして」
run id: `snake_phase_mnist_0914` / worktree `wt/snake_phase_mnist_0914` / branch `claude/snake_phase_mnist_0914`（origin/main 94ed49a から）

> **この文書の位置づけ。** 設計ワークフロー（読み手 6・設計 3 案・批評 9・統合・検証 2・修正、指摘 41 件を反映）が出した全体草案 `specs/design_draft_snake_phase_mnist_0914_full.md`（**非登録**、150 KB）から、Issa 就寝中の一晩で実装・検査・完走できる核だけを切り出して登録する。削った部分（§9）は捨てたのではなく、別 spec の後段として残す。導出の詳細（恒等式の検算、事後の SD、費用の実測）は全体草案の同名節を参照し、本文では結論だけを書く。**本文の判定規則は全体草案と食い違う場合、本文が優先する。**
>
> 数値の等級: **登録** = results/*/summary.md・verdict.csv から写したもの。**事後** = 2026-09-14 の設計ワークフローが既存ログから計算したもの（未登録）。**推論** = 算術。

---

## 0. 問い

**Snake 自身が時間劣化を示す箱 B（Permuted MNIST・Adam）で、CondA の Snake 3 位相（normal / peak / valley）と定数オフセットは、(E1) 時間劣化、(E2) 水準、(E3) fresh gap、(CT1) 隠れ層の個体別中心化ノルム ‖W̃ᵢ‖ の成長を変えるか。変えるなら、それは初期関数（起点 S）と、根の位置・活性値の水準（オフセット K）のどちらから来るか。後期まで位相の効き目が残る適応 α 族で、初期関数を揃えても位相の構造そのものが効くか。**

確認的な見出しは **E1 のラベルだけ**。E2・E3・CT1・M1・総合クラスは副次で、族をまたぐ多重性は制御しない。

## 1. なぜ CondA ではだめか

CondA（1 層・MSE・SGD・32 点支持）では Snake が課題を解き切り、比べるべき時間劣化が無い。後期（task451–500）MSE は normal q0 4.09e−11・peak q0 6.41e−9・valley q0 2.93e−11（登録、`zero_attraction_analysis_0913/results/zero_attraction_learning_0913/summary.md`）。CondA で見えた peak の自由重み成長 −9.4%（`DIRECTIONAL_PHASE_SUPPRESSION`）・peak の 70.8% の個体の縮小・群除去の偏りは、LoP の有無と結びつけて検定できなかった。

## 2. 箱

**箱 B = gate_shape_0911 と同一のプロトコル**: 784–100–100–10、CE、batch 16、1 タスク = 10,000 枚（625 更新）、手書き Adam lr 1e−3・β .9/.999・eps 1e−8、moment と step はタスクをまたいで持ち越し、正則化なし、**t1–120**、test 10,000 枚で評価、CPU float32・1 スレッド、white-san のみ。データ列・初期化は宿主 `src/pmnist_boundary_host_0908.py` の role 付きストリーム（腕は RNG に入らない）なので、同じ seed の腕どうしでタスク列と初期 W は同一。

選んだ理由（登録、`results/gate_shape_0911/summary.md` §1、L = タスク終端 acc の中央値 t16–20 − 中央値 t101–120、seed 中央値）: SN06 **2.74**・SNA 0.71・LR 2.28・LIN 0.63。固定 α=0.6 の Snake が LIN より明確に劣化する。SN06 の R_c 0.02・ρ_w 0.98（幅クランプで劣化が消える）。事後の窓平均 D の seed 対 D_pair(SN06−LIN) は 2.20/1.90/1.97。

**Random Label MNIST は本 spec では走らせない**（§9）。SNA は 400 epoch・50 タスクで天井に張り付き平坦（登録 `results/shell_l2_rlmnist_0913/summary.md`: online_acc t31–50 = 0.9859）で、しかも今夜は W病理チャットの `wcap_rlmnist_0914` が同じ機械で RL を 10 プロセス走らせている。

## 3. 活性化

隠れ 2 層とも同じ活性化（宿主の慣例）。α = 0.6。

位相族 ψ_θ(z) = z + [cos θ − cos(2αz + θ)]/(2α)、ψ_θ′(z) = 1 + sin(2αz + θ)。

| 名 | 式（実装・float32） | φ(0) | φ′(0) |
|---|---|---|---|
| normal | `z + torch.sin(a*z)**2 / a`（宿主 SN06 と同じ演算順） | 0 | 1 |
| peak | `z + torch.sin(2.0*a*z) / (2.0*a)` | 0 | 2 |
| valley | `z - torch.sin(2.0*a*z) / (2.0*a)` | 0 | 0（二重零点） |
| offset Snake | normal + q（q ≠ 0 のときだけ `+ q`） | q | 1 |
| offset leaky | 宿主 LR（a=.1）の式 + q | q | — |
| 適応位相 | 上の式の α を αᵢ(t) に置換。**`GS.H.AdaptiveSnake`（= `src.pmnist_boundary_host_0908.AdaptiveSnake`）を継承**（`src.pmnist_0905.AdaptiveSnake` からの継承は isinstance が通らず V 更新が黙って落ちる。事後に確認済み）。αᵢ = clip(0.6/√Vᵢ, .05, 3)、V は β=.01 の EMA で初期 1 | 0 | 1/2/0 |

**恒等式（設計の土台）**: 固定 α で任意の整数 k について ψ_θ(z) = ψ_0(z + s + kπ/α) + K − kπ/α、s = θ/(2α)、K = (cos θ − 1 − θ)/(2α)。bias は学習可能で Adam は bias の平行移動に不変なので、**固定 α の位相腕は「bias 初期値を s ずらし、定数オフセット K を足した normal Snake」と力学系として同一**。ただし (s, K) は枝 k で一意でない。

定数（float64 で式から生成して config に 17 桁で置く。本表は表示用）:
- peak (k=0): s = +1.3089969389957472、K_P = −2.1423302723290805
- peak (k=−1): s = −3.9269908169872414、K = +3.0936574836539084
- valley (k=0): s = −1.3089969389957472、K_V = +0.4756636056624139
- π/α = 5.235987755982989

**init map**（H.init_params(seed) の直後に float64 で計算し、1 回だけ float32 に落とす。W は全腕共通、bias だけ動く）:
- comp(q): b2 ← b2 − q·(W2·1)、b3 ← b3 − q·(W3·1)（出力に足された q を次層で打ち消す → 初期関数が q=0 の腕と一致）
- to_phase(s, K): b1 ← b1 + s、b2 ← b2 + s + K·(W2·1)、b3 ← b3 + K·(W3·1)（normal の経路で位相腕と同じ初期関数を作る）

## 4. 腕（17）

固定 α=0.6 の 2×2（normal 座標で読む）:

| | 水準 0 | 水準 K |
|---|---|---|
| 起点 = normal の初期関数 | **N06** | **P06c**（K_P）／**P06c_k1**（K=+3.0937、同じ根の幾何・逆符号の水準）／**V06c**（K_V） |
| 起点 = 位相腕の初期関数 | **P06i** ／ **V06i** | **P06** ／ **V06** |

| 腕 | 活性化 | init map | 初期関数 | 目的 |
|---|---|---|---|---|
| N06 | normal | — | N06 | 基準。G1 錨（gate_shape_0911 SN06） |
| P06 | peak | — | P06 | 位相の総効果（CondA peak q0 の移植） |
| V06 | valley | — | V06 | 同（valley q0） |
| P06c | normal + q=K_P | comp(K_P) | N06 | K 因子（peak） |
| P06c_k1 | normal + q=+3.0937 | comp(q) | N06 | 枝の検査 |
| P06i | normal | to_phase(s_P, K_P) | P06 | S 因子（peak） |
| V06c | normal + q=K_V | comp(K_V) | N06 | K 因子（valley） |
| V06i | normal | to_phase(s_V, K_V) | V06 | S 因子（valley） |
| SNA | 適応 normal（宿主 AdaptiveSnake(.6,.01)） | — | N06 | 適応族の基準。G1 錨 |
| SNAP | 適応 peak | — | P06 | 位相の総効果（適応族） |
| SNAV | 適応 valley | — | V06 | 同 |
| SNAi_P | 適応 normal | to_phase(s_P, K_P) | P06 | 同じ初期関数での位相構造（SNAP−SNAi_P） |
| SNAi_V | 適応 normal | to_phase(s_V, K_V) | V06 | 同（valley） |
| LIN | 恒等 | — | — | 劣化の床。G1 錨 |
| LR | leaky a=.1 | — | — | 陽性対照。G1 錨 |
| LR_qKp | leaky + q=K_P | comp(K_P) | LR | 単調ゲートでのオフセット |
| LR_qKpn | leaky + q=−K_P | comp(−K_P) | LR | 同・符号反転 |

**seed 0–19（n = 20 固定）**。seed 0–2 は gate_shape_0911 と同じ系列、3–19 は同じ role 付きストリームから引く。**走る順は seed 昇順で腕を交互に並べる**（途中で止めても対が残る）。

## 5. 1 走で記録するもの

出力先 `results/snake_phase_mnist_0914/runs/<arm>_s<seed>/`:
- `rows.csv`（毎タスク flush）: task、acc_seq（正答数/10000）、ce0（タスク開始前の probe CE）、ce20（20 更新後の probe CE）、acc_fresh（fresh probe のタスクだけ、他は空）、秒、VmRSS、ru_maxrss、コア番号、および gate_shape_0911 rows と同じ層 1 の測定列（G1 用）。
- `units.npz`（t0 と毎タスク終端、層 1・2 × 100 ユニット）: cnorm（行平均を引いた行ノルム）、m（行平均）、b、zcur・sdcur（probe 512 枚・現置換の前活性の平均と SD）、gbar（mean φ′）、A（位相の梃子 |mean exp(i·2αᵢ zᵢ)|）、abar（mean φ、オフセット込み）、outcol（次層の列ノルム）、alpha（適応族）。加えて gate_shape_0911 units と同じ層 1 の配列（G1 用）。
- `snap_t020.pt`・`snap_t060.pt`・`snap_t120.pt`（params・Adam m/v/t・SNA の V、sha256 を provenance に）。群除去・着座・将来学習の後段解析（§9）の入力。
- `provenance.json`（最後に書く）: git_hash、宿主と acts の sha256、spec・config の sha256、torch 版、CPU、コア番号の系列、RSS の系列、ストリームの hash、壁時計。

**fresh probe**: t ∈ {1, 16–20, 101–120} の 26 本。その腕自身の θ₀（H.init_params(seed) に init map を適用）から、**新しい Adam**（m=v=0・step 0・SNA 族は V=1）で、逐次タスク t と byte 同一の (perm, idx, order) を 625 更新学習し、test 精度を測る。逐次の状態・RNG には触れない（S7）。

## 6. 主 endpoint と判定

### 6.1 E1 時間劣化（確認的な見出し）

acc は正答数/10000、差は pt。seed k の対差:

**D_pair,k(θ, ref) = mean_{t=16..30}[acc_θ(t) − acc_ref(t)] − mean_{t=101..120}[acc_θ(t) − acc_ref(t)]**（正なら θ の方が多く劣化）

族（族内 Holm、α=.05 両側、m は事前固定でゲートで対比が落ちても変えない）:

| 族 | 対比 | m |
|---|---|---|
| F1 固定 α | C1 P06−N06、C2 V06−N06、C3 P06c−N06（K・peak）、C4 P06i−N06（S・peak）、C5 V06c−N06（K・valley）、C6 V06i−N06（S・valley） | 6 |
| F1x 分解 | I_P = (P06−P06i) − (P06c−N06)、I_V（同型）、B_P = P06c_k1 − P06c | 3 |
| F2 適応 | C7 SNAP−SNA、C8 SNAV−SNA、C7s SNAP−SNAi_P、C8s SNAV−SNAi_V | 4 |
| F3 単調 | C9 LR_qKp−LR、C9n LR_qKpn−LR | 2 |

検定: **正確な符号反転検定**（n=20 は 2²⁰ を全列挙）。CI は符号反転検定を位置ずれ δ について反転したもの（格子 0.001 pt）。

ラベル（上から最初に当てはまるもの）:
1. `INCOMPLETE`: 完全な seed 対の数 n_c < n_min(m) = ⌈log₂(2m/.05)⌉（m=6,4 → 8、m=3,2 → 7）。
2. `NOT_TESTABLE_REF_FLAT`: F1・F1x で TESTABLE_FIXED 不成立、F3 で TESTABLE_LEAKY 不成立（§6.5）。
3. `MORE_DECLINE` / `LESS_DECLINE`: Holm で有意、符号で決める。(1 − 2·.05/m) 反転 CI ⊂ [−h_D, +h_D] なら接尾 `_WITHIN_RESOLUTION`。
4. `EQUIVALENT`: TOST、(1 − 2·.05/m) 反転 CI ⊂ [−h_D, +h_D]。**意味は「単走のタスク標本の分解能より小さい」であって「効果が無い」ではない。**
5. `INCONCLUSIVE`: それ以外。反転 CI の半幅を併記する。

F1x の I は加法性の判定だけに使い、`NON_ADDITIVE`（Holm 有意）／`ADDITIVE`（TOST）／`ADDITIVE_INCONCLUSIVE`。n=20 では I の等価性に検出力がほぼ無いことを先に宣言する（全体草案 §6.1 の算術: 真の I=0 でも ADDITIVE になる確率は約 .16–.40）。主効果 S_P = ½[(P06i−N06) + (P06−P06c)]、K_P = ½[(P06c−N06) + (P06−P06i)]（valley も同型）は反転 95% CI 付きで REPORT。

**書き方の規則**: 固定 α の後期窓の効果は、M1 が PHASE_ACTIVE でない限り「起点」「根の位置・活性値の水準」の効果として書き、「位相」とは書かない。「位相構造が LoP を変えた」と書けるのは C7s または C8s が MORE/LESS で、かつ M1 がその対で t16–30 と t101–120 の両方 PHASE_ACTIVE のときだけ。

### 6.2 E2 水準

A_late = mean acc t101–120、A_base = mean acc t16–30。ΔA_late・ΔA_base に E1 と同じ族・検定・Holm を当て、`HIGHER` / `LOWER` / `EQUIVALENT`（CI ⊂ [−h_Alate, +h_Alate] など）/ `INCONCLUSIVE`。併記（ラベルなし）: A(t1)、全タスク平均、step 0 の CE。

### 6.3 E3 fresh gap

Gap_late(θ) = mean_{t=101..120}[A_fresh,θ(t) − A_seq,θ(t)]。ΔGap_late に同じ族・検定で `GAP_LARGER` / `GAP_SMALLER` / `EQUIVALENT`（± h_Gap）/ `INCONCLUSIVE`。修飾 `FRESH_LEVEL_CONFOUNDED`: t101–120 の A_fresh 平均の差が Holm で有意。

### 6.4 総合クラス（副次・上から最初に当てはまるもの。E1 が INCOMPLETE・NOT_TESTABLE の対比には付けない）

1 `CONFLICT`: E1 有意で E3 が逆向きに有意（MORE_DECLINE と GAP_SMALLER、LESS_DECLINE と GAP_LARGER）
2 `TEMPORAL_LOP_MORE`: E1 MORE_DECLINE かつ（ΔA_late LOWER または GAP_LARGER）
3 `TEMPORAL_LOP_LESS`: E1 LESS_DECLINE かつ（ΔA_late HIGHER または GAP_SMALLER）
4 `BASE_SHIFT`: E1 有意、ΔA_base が E1 の符号を生む向きに有意、ΔA_late と E3 はともに有意でない
5 `NO_DIFFERENCE_RESOLVED`: E1・ΔA_late・E3 がすべて EQUIVALENT
6 `LEVEL_ONLY`: E1 EQUIVALENT、ΔA_late 有意
7 `LEVEL_DIFF_DECLINE_INCONCLUSIVE`: E1 INCONCLUSIVE、ΔA_late 有意
8 `INCONCLUSIVE`: それ以外

### 6.5 可検定性・操作確認

- `TESTABLE_FIXED`: D_pair(N06, LIN) の反転 95% CI_lo > 0。`TESTABLE_LEAKY`: D_pair(LR, LIN) の反転 95% CI_lo > 0。
- **M1 PHASE_ACTIVE**（層 1・2、窓 t16–30 と t101–120、対 C1 C2 C7 C8 C7s C8s）: 任意の位相 θ, θ′ で |ḡᵢ^θ − ḡᵢ^θ′| ≤ 2|sin((θ−θ′)/2)|·Aᵢ（ḡ − 1 = Im(e^{iθ}E e^{i2αz}) からの算術）。seed ごとの Φ = 2|sin(Δθ/2)|·medianᵢ Aᵢ(θ 腕) − IQRᵢ(ḡᵢ^ref)（窓内タスク平均。normal 対 peak・valley は Δθ=π/2 で係数 √2）。反転 95% CI > 0 → `PHASE_ACTIVE`、< 0 → `PHASE_WASHED`、他 → `PHASE_BORDERLINE`。意味: 位相を動かしても平均ゲートが基準腕自身のユニット間の散らばりより動かせないなら、位相は基準の分布の外に出られない。
- `BROKEN`（run 単位）: t101–120 で ce20 < ce0 のタスク数 < 15/20（片側正確二項 P(X≥15|20,.5)=.0207）。壊れた seed は対比から落とす。
- `DIVERGED(task, step)`: 非有限の loss・パラメータ。その run は欠測（直前のタスクまでの出力は書いてから止める）。

### 6.6 §0 への答え方（副次・位相ごと・上から最初）

1 `NOT_ANSWERABLE`: C1 または C2 が INCOMPLETE・NOT_TESTABLE
2 `NON_ADDITIVE`: その位相の I が NON_ADDITIVE（単純効果 C3/C5・C4/C6 を並べて書く）
3 `BOTH`: K（C3/C5）と S（C4/C6）がともに MORE/LESS
4 `LEVEL`: K が MORE/LESS、S が EQUIVALENT
5 `ORIGIN`: S が MORE/LESS、K が EQUIVALENT
6 `NONE`: C1/C2・K・S がすべて EQUIVALENT
7 `INCONCLUSIVE`: それ以外

S の単純効果（C4/C6）を MORE/LESS と数えるのは、E1 のラベルに加えて **ΔA_late か E3 のどちらかが同じ向きに有意**なときだけ（起点の過渡が base 窓 t16–30 に残って E1 だけを動かす場合を拾わないため。全体草案の M3 を簡略化した規則）。peak の `LEVEL` は B_P が EQUIVALENT なら `LEVEL_ROOT_GEOMETRY`、MORE/LESS なら `LEVEL_ACTIVATION_VALUE`。
適応族: `STRUCTURE` = C7s（C8s）が MORE/LESS かつ M1 がその対で両窓 PHASE_ACTIVE、`STRUCTURE_NOT_SHOWN` = EQUIVALENT、他は `STRUCTURE_INCONCLUSIVE`。

### 6.7 CT1 成長（CondA の見出し「peak は normal より育たない」の移植）

logN_ℓ = ½ log meanᵢ,t∈101..120 cnormᵢ²（層 ℓ=1,2）。W の初期値は全腕共通なので ΔlogN は成長比の対数差と同じ。対比 C1・C2・C7・C8、層ごとに Holm m=4、正確な符号反転検定。ラベル: `GROWTH_SUPPRESSED`（有意・負）／`GROWTH_ENHANCED`（有意・正）／`GROWTH_BELOW_CONDA_SCALE`（(1−2·.05/4) 反転 CI ⊂ [−0.09877, +0.09877]）／`INCONCLUSIVE`。マージンは CondA の登録 log 差 −0.09877（`zero_attraction_learning_0913/verdict.csv`、peak q0 対 normal q0）を物差しとして借りたもので、u と ‖W̃ᵢ‖ は別量（同じ大きさを期待する根拠ではない）。**腕間の幅の差は W 病理の証拠にしない**（箱 B では腕間で幅が劣化を並べない: gate_shape_0911 A6 Spearman(N, L) = +0.01、登録）。

REPORT（ラベルなし）: 固定 α の分解対比の ΔlogN、層 1 の成長比の対数のユニット間 SD（実 c0 基準）と strict な縮み ‖W̃ᵢ(t120)‖² < ‖W̃ᵢ(0)‖² の割合（Adam では算術上ほぼ 0 と明記）、Spearman(cnormᵢ(t1), cnormᵢ(t120))、M3 型の起点の過渡 Δu(t)、M2 型の平均活性の座り直し r_a、登録互換の L（中央値 t16–20 − 中央値 t101–120）。

## 7. 閾値の導出

- **分解能マージン h_E**（E ∈ {D, A_late, A_base, Gap_late}）: 基準対 {N06−LIN, LR−LIN} の各 seed の対差系列 d(t) について、窓ごとにタスクの線形傾向を除いた残差分散からタスク標本の SE を出し（D なら √(var(res₁₆₋₃₀)/15 + var(res₁₀₁₋₁₂₀)/20)、ddof 2）、2 対 × 20 seed で二乗平均の平方根を取る。**位相・オフセットの対比の値は読まない**（盲検）。事後の見込み（gate_shape_0911 の 3 seed）: h_D ≈ 0.19、h_Alate ≈ 0.11、h_Abase ≈ 0.16。固定の効果量（SESOI）は置かない。
- n_min(m) = ⌈log₂(2m/.05)⌉（最小の両側 p 2/2ⁿ が .05/m 以下）。
- 二項 15/20: P(X≥15|20,.5) = .0207 ≤ .05、P(X≥14) = .0577。
- M1: 位相で動かせるゲート量の算術上界と基準腕の実測 IQR の比較。
- CT1: CondA の登録 log 差。
- S 検査の許容: float64 の丸め伝播上界（n·eps·Σ|項| を実行時に計算）。変異の検出幅は変異の解析的なずれの 1/2 以上。固定倍率は使わない。
- 並列数: §8。

**n=20 固定の理由**（全体草案の n_main 較正を削った）: 事後の対差 SD 0.12–0.28 pt（3 seed、池により異なる）なら、n=20 の反転 95% CI の半幅は約 0.06–0.13 pt。h_D ≈ 0.19 の分解能での EQUIVALENT と、0.3 pt 以上の差の検出には足りる見込み。全体草案の式では n 14–31。上限に届く場合の不足は `INCONCLUSIVE` として出す。

## 8. 資源と並列数

実測（事後、全体草案 §12.1）: 箱 B 1 タスク P コア 0.53 s／E コア 1.32 s、fresh probe 1 本 P 0.41 s／E 1.11 s、ru_maxrss 約 1.08 GiB、snapshot 1.1 MB。1 走の推論: P コア 約 90 s、E コア 約 230 s（層 2 の測定を含む）。**17 腕 × 20 seed = 340 走 ≈ 8.5–22 core-h**。

**並列数 P = min(C_free, ⌊(MemAvailable − ΔM_desk) / RSS_peak⌋)**。swap は余裕に数えない。
- RSS_peak: 段階 0 の資源走（最重量の適応腕 1 本・probe と snapshot を含む・120 タスク）で直接測る。
- ΔM_desk: 03:26 から 2 秒ごとに記録している MemAvailable と実験プロセスの RSS の合計 M(t) について、「直前の窓 w（最重量の 1 走の壁時計）の M の最大値 − M(t)」の記録全体での最大値。本走中も更新する。
- C_free = 論理 28 − CPU 張り付きのプロセス数、物理 20 を超えない。
- 起動の直前に MemAvailable と SwapFree を読み直し、P < 1 または SwapFree < RSS_peak なら起動しない。W病理チャットの wcap は**止めない・待たない**（上の式で分け合う）。
- 見張り `memwatch.py`（2 秒ごと）: MemAvailable < RSS_peak + ΔM_desk で新規起動を止め最新の shard に SIGSTOP（SIGSTOP はメモリを解放しないと明記）、MemAvailable < RSS_peak で最新の shard に SIGTERM（その shard は後で丸ごと走り直す）、MemAvailable ≥ 2·RSS_peak + ΔM_desk で逆順に SIGCONT。時刻・PID・shard 名をログに書く。

## 9. 本 spec で走らせないもの（全体草案から削った。後段の別 spec）

削った理由は一晩の実装・検証の上限。**登録ラベルの入力（E1–E3・M1・CT1・G1）は削っていない。**
1. **段階 R（Random Label 予備走）**: 固定 α の Snake は RL で未検証（反証ではない）。W病理チャットの RL と機械・箱が重なる。
2. **段階 S（SGD 橋、α=1）**: CondA 型の個体の分岐を optimizer に帰す検査。
3. **CT2 群除去（論理補償つき）・CT3 根への着座・群の将来学習（capacity）・PD**: snapshot t20/t60/t120 と units を保存するので、後段の spec で既存 snapshot から行える（その場合は本 spec の結果を見た後の登録になることを明記する）。
4. **双子の軌道検査（P06tw・V06tw）・n_main の盲検較正・M3 のラベル化・更新ごとの帳簿**。
5. **後続実験 M（腕内の幅合わせ射影 WMATCH）**: 本 spec の対比で TEMPORAL_LOP_MORE/LESS が出たときの媒介検定。

## 10. S 検査（すべて変異対照つき。run を殺す assert は出力を書いた後）

変異が検出されなければその検査は空虚として fail。`analysis/snake_phase_mnist_0914/checks.py` → `checks.json`。

| # | 検査 | 変異対照 |
|---|---|---|
| S1 | **G1 錨**: N06・LR・LIN・SNA の s0 を、gate_shape_0911 の rows・units と t1–3 で照合（本走前）。学習状態の配列（cnorm・行平均・bias）は max\|diff\| = 0.0、float64 集計の派生量は丸め伝播上界以内、acc は一致。本走では s0–2 × 120 タスクで同じ照合を行い、崩れたら最初にずれたタスクで `CODE_MISMATCH`（学習状態が t1 でずれる → 本走停止）／`UNANCHORED`（後期にずれる → 続行・記録）／`MEASURE_MISMATCH` に分ける | init で W1[0,0] += 1e−3 → 学習状態の差 > 0 を検出 |
| S2 | **活性化の形**: float32 と float64 で φ(0) == 0（Snake 族）、φ′(0) が厳密に 1/2/0。float64 の格子（φ′ の零点と根を挿入）で解析 φ′ と autograd の差が上界以内 | peak を normal に落とす／sin 項の符号反転／別位相の解析 φ′ と組む（不一致の解析値 √2 または 2 の 1/2 以上で検出） |
| S3 | **q=0・θ=0 の bit 同一**: PhaseSnake(θ=0,q=0) と OffsetLeaky(q=0) が宿主 SN06・LR と、適応 normal が宿主 `GS.H.AdaptiveSnake` と、**箱 B のループと同じ forward 経路**で torch.equal。isinstance が True | `+1e−6` を入れる／継承元を `src.pmnist_0905.AdaptiveSnake` に替える → isinstance False と V 更新の欠落を検出 |
| S4 | **恒等式（float64）**: P06(W,b) と normal+K_P(W, b+s) の logits と全勾配、枝 k=−1、valley。状態は θ₀・3 タスク煙走の終状態・重みを 4 倍した状態 | K を落とす／s の符号反転 → logits が O(1) ずれる（解析値の 1/2 以上） |
| S5 | **init map**: float32 に丸めた実パラメータを float64 で前向き計算し、P06c・V06c・P06c_k1 の logits が N06 と、P06i・SNAi_P（t=0）が P06 と、V06i・SNAi_V が V06 と、LR_qKp・LR_qKpn が LR と、半 ulp の伝播上界以内で一致 | b3 の補償を外す（ずれ \|q\|·\|W3·1\|）／補償を 2 回かける |
| S6 | **fresh probe**: t1 の fresh が逐次の task 1 の acc と終状態を bit 再現する。probe の (perm, idx, order) の sha256 が逐次タスクと一致 | 開始点を task 1 終状態にする／task 2 の置換を使う／task 1 後の Adam 状態を持ち込む／SNA 族で逐次の V を持ち込む |
| S7 | **非侵襲**: 5 タスクの煙走で、fresh probe（t3 に強制）・層 2 測定・snapshot を入れた場合と入れない場合で、rows の acc 列と毎タスクの全パラメータの sha256 が同一 | probe が main の batch 生成器から 1 回だけ引く／追加測定で W1 に 1e−9 足す |
| S8 | **ストリームの腕独立**: タスクごとの perm/idx/order の sha256 が全腕で同一、seed 間では異なる | role 文字列に腕名を混ぜる／seed s+1 に s を使う |
| S9 | **追加測定**: 層 2 の per-unit 配列、Aᵢ、abar、M1 の Φ、CT1 の logN を独立な float64 再実装と上界以内で照合 | 再実装側だけ W2 の 1 行を 1e−3 動かす／A の位相係数 2α を α にする |
| S14 | **判定コード**: 合成シャードで、E1 の全ラベル（MORE/LESS、`_WITHIN_RESOLUTION`、EQUIVALENT、INCOMPLETE、NOT_TESTABLE_REF_FLAT）、E2・E3 のラベル、§6.4 の 8 クラスと優先順位、F1x の 3 ラベル、§6.6 の全行と細分、M1 の 3 状態、CT1 の 4 ラベル、BROKEN・DIVERGED が期待どおり | 腕の列を入れ替える → 符号反転／Holm を外す／seed の対を崩す／§6.4 の順を入れ替える／h_D を定数にする |
| S15 | **発散と資源**: loss に NaN を注入（task 2, step 10）→ `DIVERGED(2,10)` で task 1 の出力がディスクに残る。memwatch に偽の MemAvailable 列を与え、境界で SIGSTOP/SIGTERM/SIGCONT が最新の shard に送られる（ダミー sleep プロセス）。launch は偽の RSS 20 GiB で P=0 として起動を拒む | flush しない版／比較の不等号を入れ替えた版／最古の shard を止める版 |

**committed の module は 1 行も編集しない**（宿主 sha を provenance に固定）。`GS.run`・`GS.check_dphi`・出力前に止める assert 経路は使わない（全体草案 §13 の理由）。

## 11. 段階とゲート（全自動）

1. **G1 事前登録**: 本 spec・`configs/snake_phase_mnist_0914.yaml`・全体草案（非登録）を commit・push。vault の spec/ に同文を置き、現在地に 1 行。
2. **段階 0 検査**: 実装 → S1（t1–3）・S2–S9・S14・S15 がすべて通り、変異がすべて検出される（`checks.json` all_pass）。実装を commit・push（その hash が provenance の git_hash）。1 回で通らなければ実装を直して再検査してよい（検査の内容・変異は変えない。変えたら addendum に書く）。
3. **段階 0 資源**: 最重量の適応腕（SNAV・seed 0）の 120 タスク走を `results/_smoke_snake_phase_mnist_0914/` に 1 本。RSS_peak・秒/タスクを記録し P を決める。
4. **段階 1 錨**: N06・LR・LIN・SNA の s0–2 を先に走らせ、S1 を 120 タスクで判定。CODE_MISMATCH なら全停止。
5. **段階 3 本走**: 残りの shard を seed 昇順・腕交互で、memwatch 常駐。全 340 shard が揃うまで対比を計算しない。1 shard が 2 回失敗したらその seed を欠測にし、INCOMPLETE 規則を当てる。
6. **判定**: `verdict.py`（S14 で検証済み）→ `verdict.csv`・`summary.md`（数値は verdict.csv から写し窓ラベルを付ける）。vault 測定/ に結果ノート、現在地に 1 行。
7. **片付け**（CLAUDE.md §4）: git 外の units・snapshot・log を `~/Projects/obsidian-research-data/snake_phase_mnist_0914/` へ移し manifest を commit、merge origin/main、HEAD:main へ push、is-ancestor 確認後に worktree・branch を削除。git には rows.csv・provenance・判定入力の小さな派生ファイルと G1 錨腕の units だけを入れる。

## 12. 事前予測（記名・走る前）

**Claude**（全体草案の予測のうち、本 spec に残った endpoint に対応するもの。番号は全体草案のまま）:

| # | 予測 | 確率 | 反証 |
|---|---|---|---|
| P1 | S4・S5 の恒等式が float64 の上界以内で成り立ち、変異はすべて検出される | .98 | 1 つでも上界超え・変異の見逃し |
| P2 | N06・LR・LIN・SNA の s0–2 が 120 タスクで、学習状態の配列が 0.0 で一致（G1） | .90 | 非零の差 |
| P3 | TESTABLE_FIXED と TESTABLE_LEAKY がともに成立 | .95 | どちらかの CI_lo ≤ 0 |
| P4 | N06 の 20 seed の D（mean t16–30 − mean t101–120）の平均が [1.955, 2.599]（3 seed の丸め前の D 2.4243/2.2630/2.1442 からの予測区間） | .85 | 区間外 |
| P5 | 固定 α の P06 対 N06・V06 対 N06 は層 1 の t101–120 で PHASE_WASHED（.75）。SNAP 対 SNA・SNAV 対 SNA は t101–120 で PHASE_ACTIVE（.80） | 左記 | 固定 α の後期で ACTIVE、または適応族の後期で WASHED |
| P6 | C1・C2 の 95% 反転 CI がともに ±0.8 pt に収まる（位相の総効果は α 梯子の差 1.2–1.9 pt より小さい） | .75 | どちらかがはみ出す |
| P7 | 両位相で \|K 主効果\| > \|S 主効果\|（点推定）。C4・C6 は MORE/LESS にならない | .55 | S が Holm 有意かつ \|S\|−\|K\| の CI が正 |
| P9 | B_P = P06c_k1 − P06c は EQUIVALENT にならない | .55 | EQUIVALENT |
| P10 | C9 と C3 の点推定が同符号。C9n は C9 と逆符号（.55） | .60 | C9 と C3 が両方有意で逆符号 |
| P11 | C7・C8 の少なくとも一方が MORE_DECLINE。C7s・C8s の少なくとも一方も MORE_DECLINE で STRUCTURE（.35） | .50 | C7・C8 がともに EQUIVALENT か LESS |
| P13 | Holm で有意な E1 の対比に CONFLICT は出ない | .80 | 1 つでも出る |
| P14 | 全 Snake 位相腕で、層 1 の成長比の対数のユニット間 SD の seed 平均 < 0.13（Adam では CondA 型の分岐が無い。REPORT の予測） | .85 | どれか 1 腕で ≥ 0.13 |
| P18 | CT1 層 1 の C1（P06−N06）は GROWTH_SUPPRESSED にならない | .60 | GROWTH_SUPPRESSED |
| P19 | 層 2 の logN2 の seed 平均が \|K\| の順に小さい: P06c_k1 < P06c < V06c < N06（Adam の分母に K²δ² が入り中心化成分の歩幅を縮める、推論） | .50 | 順序が 1 箇所でも崩れる |
| P22 | C2（V06−N06）の ΔA_late が LOWER（.55）、V06 に FRESH_LEVEL_CONFOUNDED（.50）。C1 の ΔA_late は LOWER でも HIGHER でもない（.60） | 左記 | 逆 |
| P23 | V06 の 20 seed で BROKEN・DIVERGED は 0 本（.80）。P06 も 0 本（.90） | 左記 | 1 本以上 |

**Issa**: 就寝中のため未記入（「全自動で」の指示により記入を待たずに登録する）。起床後に書く予測は addendum として区別する。

## 13. Claude が Issa の代わりに決めた点（就寝中・全自動の指示による）

1. 範囲を箱 B の核（本文）に絞り、RL 予備走・SGD 橋・CT2/CT3・将来学習・PD・双子・n_main 較正を後段に回した（§9）。
2. n = 20 固定（全体草案の n_main 10–30 の較正を削った）。
3. W病理チャットの wcap_rlmnist_0914 は止めず、メモリの式で機械を分け合う。act_chimera_0913（spec 未 commit・未起動）より先に回す。
4. Codex の 検証設計_0913 §4C 第 3 段階を「引き取る」とは書かない（本 spec はその一部の移植にとどまる）。
5. 課金のある GCP は使わない。

## 14. 引用上の限定

- 箱 B（Permuted MNIST・Adam・625 更新・t1–120・幅 100・隠れ 2 層）限定。RL・CIFAR・SGD へ外挿しない。
- 位相腕は活性化を 2 層ともに変える。層を限定したオフセットは未検定。
- EQUIVALENT は「単走のタスク標本の分解能より小さい」。
- 腕間の幅（CT1）の差は W 病理の証拠ではない。
- CondA の自由重み u（5 bit への入射）に当たる量は MNIST に無い。CT1 は ‖W̃ᵢ‖ についての答えで、CondA の見出しの量そのものの再現ではない。
- 固定 α=0.6 では位相の効き目が後期までに洗い流される見込み（事後: SN06 の中央値ᵢ exp(−2α²sdcurᵢ²) が t1 0.64–0.67 → t120 0.09–0.10）。後期の効果は M1 に従って言葉を選ぶ。
