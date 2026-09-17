# 残り 0.11 は移植先の「逃げ」か — t2 の網の動く余地を塞いで respdyn の主腕を回し直す（RL・ELU→ELU・12 腕 × 10 seed）

状態: **事前登録**（本ファイルの commit が登録。上限・固定をかけた腕はまだ一つも記録として走っていない）
作成: 2026-09-17 / 起草: Claude / 依頼: Issa「（t2 の網に上限を置いて respdyn の主腕を回し直す）を実装して」
前の走: `specs/spec_swap_ee_0917.md`（`REMAINDER_IN_BOTH`・事後の読み「残り＝移植先が登れた量」）・`specs/spec_respdyn_ee_0917.md`（R = +0.112）
実装: `src/escape_ee_0917.py`・`analysis/escape_ee_0917/{checks.py,verdict.py,launch.sh}`
branch: `claude/escape_ee_0917`

## 0. 一行と問い

respdyn の主腕 S2dyn_10r（健康な t2 の網に、崩れていく t10 の網の動く第2層の場を渡す・Adam 初期化）は、応答の床 S2u30r より 0.112 上で学ぶ。swap_ee_0917 では、同じ場を渡しても伸びない網（cap12）の残りは 0.037 で、事後に見ると残りの大きさは「移植先の網自身の第2層が場の下で登れた量」と並んだ（t2 +3.5、上限なし cap12 +1.6、上限つき cap12 +1.0）。

**同じ t2 の網のまま、動く余地だけを塞ぐ。** 各 Adam 更新の直後に、t2 の分岐状態で測った値で「保持」をかける（分岐点では何も動かない）。

| 保持 | 中身 |
|---|---|
| none | なし（respdyn の腕そのもの。bit で再現） |
| c1 | 第1層の $\|q_i\|$ と $\|v_i\|$ に上限（mucap_el_0916 の上限・seed の平均画像の軸） |
| c2 | 第2層の行ノルムに上限（l2cap_ee_0917 の上限） |
| c12 | c1 と c2 の両方 |
| c12b | c12 に加えて $b_1,b_2$ を t2 の値へ書き戻す |
| frz | $W_1,b_1,W_2,b_2$ を t2 の値へ書き戻す（逃げは 0。特徴が床と同じになるので、この腕は床と bit で一致する。較正用） |

- **Q1（主）**: 余地を塞ぐ（c12）と、床の上の残りは 0.112 から下がるか。0 まで下がるか。
- **Q2（並び）**: 塞ぎ方を変えた 5 腕（none・c1・c2・c12・c12b）で、網自身の登りの順位と学習能力の順位は揃うか。
- **Q3（固定した場）**: 固定した場（S2_10r・床の上 0.253）の残りも、塞ぐと下がるか。
- **対照**: 保持そのものが健康な t2 の網の学習を削らないか（N2r と N2r_c12）。

## 1. なぜ（既存データ・REPORT_ONLY）

- respdyn の記録（10 seed・継続 1 タスク目）: S2dyn_10r の E 0.324・網自身の第2層の平均前活性の登り +3.50、S2_10r 0.464・+4.95、S2u30r 0.212・0.00、N2r 0.724・−0.80。
- swap_ee_0917: 押せない cap12 の網に同じ動く場を渡すと、上限あり 0.197（登り +1.00、床 0.160 → 残り 0.037）、上限なし 0.210（+1.63）。cap12 の網は上限を外すと第2層の行が 1.60 → 2.00、$\|\mu_2\|$ が 9 → 17 に伸びた。
- l2cap の上限の検査（S2c）と mucap の上限の検査（S1–S3）は済んでいる。上限は本 runner でも同じ関数を使う。

## 2. 設計

### 2.1 箱（resp_ee / respdyn / swap と同一）

Random Label MNIST（1200 枚固定）、784–100–100–10、両隠れ層 ELU(α=1)、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8、batch 16、6000 更新/タスク、seed 0–9、float32、CPU 1 スレッド、`torch.set_flush_denormal(True)`。

### 2.2 1 seed の流れ

1. ref の自然軌道 task 1–12（resp_ee の `run_prefix`）。t2 と t10 で分岐状態を保存。
2. 12 腕。すべて t2 の網から、Adam 初期化で、t2 の網自身のラベルと標本順で 2 タスク続ける。前向きは resp_ee の固定補正つき。動く場の影は N10（t10 の網の自然な継続・上限なし）で、respdyn と同じ。

| 腕 | 場 | 保持 | 役割 |
|---|---|---|---|
| **S2dyn_10r** | 動く | none | **respdyn の主腕の再現** |
| S2dyn_c1 | 動く | c1 | 第1層だけ塞ぐ |
| S2dyn_c2 | 動く | c2 | 第2層だけ塞ぐ |
| **S2dyn_c12** | 動く | c12 | **主** |
| S2dyn_c12b | 動く | c12b | bias も塞ぐ |
| S2dyn_frz | 動く | frz | 逃げ 0（較正） |
| **S2u30r** | 一様 −30 | none | **床**（resp_ee・respdyn の再現） |
| S2u30r_c12 | 一様 −30 | c12 | 床に保持（第2層の微分が全部 0 なので第1・2層に勾配が届かず、床と bit で一致するはず） |
| S2_10r | 固定 | none | 固定した場（再現） |
| S2_10r_c12 | 固定 | c12 | 固定した場に保持 |
| N2r | — | none | 自然（再現） |
| N2r_c12 | — | c12 | 保持だけの効果（対照） |

## 3. 測る量

- **E** = 継続 1 タスク目の online 精度。継続 2 タスク目は副。
- **登り** climb = 網自身の第2層の平均前活性（1200 枚・全ユニット）のタスク終わり − タスク始め（respdyn の probe）。
- 各腕: resp_ee の `unit_summary`、respdyn の軌跡の probe（750 更新ごと）、第1・2層の行ノルムの平均、$b_2$ の平均、保持の外れ（上限の超過・書き戻す値からのずれ）、上限が書いた行数、書き戻した要素数、状態 sha256。
- 再現の記録（seed ごと）: 自然軌道の各タスクの状態 hash = resp_ee、S2dyn_10r = respdyn、S2u30r・S2_10r・N2r = respdyn と resp_ee（状態 hash・2 タスク）、6 本の動く腕の影 = N10（t11・t12）、全腕の分岐点の logits、動く腕の $d(0)$ = 固定した場、S2u30r_c12 の状態 = S2u30r、S2dyn_frz の online = S2u30r。

## 4. endpoint

- $R_{\rm none}=E(\text{S2dyn\_10r})-E(\text{S2u30r})$、$R_{c12}=E(\text{S2dyn\_c12})-E(\text{S2u30r\_c12})$、**$\Delta R=R_{\rm none}-R_{c12}$**。
- 操作の確認: $\Delta{\rm climb}={\rm climb}(\text{S2dyn\_10r})-{\rm climb}(\text{S2dyn\_c12})$。
- 対照: $I=E(\text{N2r})-E(\text{N2r\_c12})$。
- 固定した場: $\Delta R_{\rm fix}=(E(\text{S2\_10r})-E(\text{S2u30r}))-(E(\text{S2\_10r\_c12})-E(\text{S2u30r\_c12}))$。
- 並び: seed ごとに 5 腕（none・c1・c2・c12・c12b）の climb と E の Spearman 順位相関 $\rho$。
- 報告: $R_{c1},R_{c2},R_{c12b},R_{\rm frz}$（床は S2u30r）、$R_{c12}-R_{c12b}$、継続 2 タスク目の同じ量。

## 5. 判定

### 5.1 帯

有効 seed の差の平均・SD・対応差 t 区間。**主の 2 量（$\Delta R$ と $R_{c12}$）は 97.5%**、他は 95%。符号は区間の下端 > 0 で +、上端 < 0 で −、それ以外 0。

### 5.2 適用条件

- **(A)** seed ごとに §3 の再現の記録がすべて真（S2u30r_c12 と S2dyn_frz の一致は報告のみで条件にしない）、12 腕 × 2 タスクの E と climb が有限、保持をかけた腕は各タスク終端で上限の超過が 1e−4 以下（4·eps32·R、R ≤ 200）で、書き戻す値からのずれも 1e−4 以下、S2dyn_c12 は各タスクで上限が行を書いた。有効 seed が 8 未満なら INAPPLICABLE。
- **(B)** $R_{\rm none}$ の 95% 区間の下端 > 0（respdyn の残りの再現）。満たさなければ NOT_REPRODUCED。
- **(C)** $\Delta{\rm climb}$ の 95% 区間の下端 > 0（保持が実際に登りを減らした）。満たさなければ ESCAPE_NOT_MANIPULATED（逃げを操作できていないので、この走からは問いに答えない）。

### 5.3 ラベル

**主**（97.5%）:

| $s(\Delta R)$ | $s(R_{c12})$ | ラベル |
|---|---|---|
| + | 0 / − | **REMAINDER_REMOVED_BY_HOLD** |
| + | + | **REMAINDER_REDUCED_BY_HOLD** |
| 0 | — | HOLD_LEAVES_REMAINDER |
| − | — | HOLD_RAISES_REMAINDER |

**並び**: 10 seed の $\rho$ の符号で正確な両側符号検定。p < 0.05 で正が多ければ CLIMB_TRACKS、負が多ければ CLIMB_OPPOSES、それ以外 CLIMB_UNRESOLVED。

**固定した場**（95%）: $\Delta R_{\rm fix}$ が + なら FIX_REMAINDER_REDUCED、0 なら FIX_REMAINDER_UNCHANGED、− なら FIX_REMAINDER_RAISED。

**読みのフラグ**: $I$ が + で、かつ $I$ の平均が $\Delta R$ の平均以上なら HOLD_IMPAIRS（残りの減少を「逃げを塞いだ」と読めない）。

### 5.4 読み方（登録）

- **REMOVED / REDUCED** → 同じ網・同じ場のまま動く余地を塞ぐと残りが減る。「respdyn の残りは移植先が自分で逃げられる量」が介入で支持される。REDUCED なら、c12 では塞ぎきれない逃げ道がある（$R_{c12b}$ と c1・c2 の比較で、どこが残ったかを書く）。
- **HOLD_LEAVES_REMAINDER** → 登りを減らしても残りは変わらない。残りは逃げの量ではない。swap の事後の読みは撤回する。
- **CLIMB_TRACKS** → 塞ぎ方の違う 5 腕で、登った量の順に学べる。
- **HOLD_IMPAIRS** → 保持そのものが学習を削っているので、主のラベルを「逃げ」の証拠にしない。
- $R_{\rm frz}$ は定義上 0 になる（網の特徴が床と同じ）。逃げ 0 の端点として並べるだけで、証拠には数えない。
- すべて「この ELU→ELU・乱数ラベル・6000 更新・t2 の網・t10 の動く場・Adam 初期化・継続 2 タスク・t2 の値での保持」に限る。

## 6. 検査（登録前に開発実行。許容は導出から）

`python3 analysis/escape_ee_0917/checks.py` → `results/escape_ee_0917/checks.json`。本走そのものが §3 の再現を全 seed で記録し、判定は (A) でそれを要求する。**登録前の開発実行（`--only`）で S1・S8・S9 は pass、変異 20/20 を検出**（S1 の seed 0 で、S2dyn_10r と S2u30r の状態 hash が記録と一致、S2dyn_frz の online が床と一致、S2u30r_c12 の状態が床と一致）。全検査（S-cost を含む）は本 spec の commit の後、runner に登録 commit を書き込んでから 1 回通しで実行する。

| 検査 | 要求 | 変異 |
|---|---|---|
| S1 腕（seed 0・80 epoch・自然軌道は 1 回） | 6 腕で: 分岐点の logits、S2dyn_10r = respdyn・S2u30r = resp_ee と respdyn（hash・2 タスク）、影 = N10 と $d(0)$、S2dyn_c12 と N2r_c12 は 3 列とも超過 ≤ 1e−4 で第2層の行を毎タスク書く、S2dyn_frz は第1・2層が t2 の値のまま（ずれ 0）で online が床と一致、S2u30r_c12 の状態 = S2u30r。**E は出力しない** | 4（c12 が第2層を塞がない・frz が $b_2$ を動かす・保持の値を t10 から取る・動く場が N2 に追随） |
| S8 判定 | 合成 seed 16 場面でラベルと推定値（1e−9）、t 分位点 | 13 |
| S9 CLI | provenance（実験名・spec・flush・分岐 2・元 10・12 タスク・各モジュールの hash）、定数（12 腕・6 保持・80 epoch） | 3 |
| S-cost | 4 プロセス同時（腕を 3 本）で exit 0・4 スロット以上。1 seed = 自然軌道 + 素の腕 6 本 + 影つきの腕 6 本で投影 | — |

## 7. 事前予測（Claude・保持をかけた腕を回す前）

| 項目 | 予測 |
|---|---|
| (A)(B) | 満たす 95%（(B) は bit 再現なので 99%） |
| (C) 保持が登りを減らす | 75%（swap の cap12 は上限で登りが 1.6 → 1.0） |
| **主** | **REMAINDER_REDUCED_BY_HOLD 45%** / REMOVED 15% / HOLD_LEAVES 25% / RAISES 5% / ESCAPE_NOT_MANIPULATED 10% |
| $\Delta R$ | +0.04（+0.01〜+0.08）、$R_{c12}$ +0.07 |
| $R_{c12b}<R_{c12}$ | 55% |
| E(c2) < E(c1)（第2層を塞ぐ方が効く） | 55% |
| 並び | CLIMB_TRACKS 55% |
| 固定した場 | FIX_REMAINDER_REDUCED 70% |
| 読みのフラグ | 立たない 75%（保持は t2 の網を救う向きに働き、$I$ は負になりうる） |
| $R_{\rm frz}=0$ と S2u30r_c12 = 床 | 99% |

**Issa の予測**（2026-09-17・登録 commit 106b28f の後、本走の前にチャットの選択式で記入）: 主 **REMAINDER_REDUCED_BY_HOLD**、並び **CLIMB_TRACKS**、固定した場 **FIX_REMAINDER_REDUCED**、対照 N2r_c12 は N2r より**良くなる**（$I<0$）。ラベルだけを記録し、Issa 自身の確率は付けない。Claude の最頻ラベルと全項目で一致。

## 8. 費用と実行

- 1 seed = 自然軌道 12 タスク + 素の腕 6 本 × 2 タスク + 影つきの腕 6 本 × 2 タスク（swap の実測から 8〜10 分）。S-cost で投影する。
- 並列数は min(4, S-cost のスロット, 起動時の空きメモリ)。別セッションのジョブと並走するので上限 4。seed 順。`provenance.json` のある seed は飛ばす。
- 起動は本 spec の登録 commit を runner に書き込み、全検査を 1 回通して all_pass になってから。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- `results/escape_ee_0917/runs/s<seed>/`（`prefix.csv`・`arms.csv`・`traj.csv`・`provenance.json` は git、`units.npz` は退避）。判定: `analysis/escape_ee_0917/verdict.py` → `results/escape_ee_0917/{summary.md,verdict.csv,paired.csv,provenance.json}`。
- `units.npz` と `results/_checks_escape_ee_0917/` は `obsidian-research-data/escape_ee_0917/` へ退避し `backup_manifest.json` を commit。結果 commit の後、その日のうちに main へ統合し worktree と branch を消す。

## 10. 実行記録（本走後の追記・計画は変えていない）

- 本走: 2026-09-17 18:32–18:58、commit 52c99b7、最大 4 並列、10/10 seed 完走・失敗 0。1 seed 7.1–9.2 分。判定スクリプトは本走後に変更していない。
- 再現: 全 seed で自然軌道 12/12 タスクが resp_ee と一致、S2dyn_10r・S2u30r・S2_10r・N2r は respdyn と resp_ee の記録と 2 タスクとも一致、影・分岐点の logits・$d(0)$ もすべて一致（無効 seed なし）。報告のみの記録: S2dyn_frz の online は 10/10 seed で床と一致。S2u30r_c12 の状態は 9/10 seed で床と一致し、seed 4 は開始時に微分が 0 でない (ユニット, 画像) が 0.000009 残っていて上限が第1層を書いた（online は 2 タスクとも床と同値）。
- 判定: 主 **ESCAPE_NOT_MANIPULATED**（適用条件 (C) 不成立: $\Delta{\rm climb}$ = +0.54 [−0.50, +1.58]・5/10）。主の 2 量そのものは $\Delta R$ = +0.046 [+0.031, +0.061]・$R_{c12}$ = +0.066 [+0.041, +0.091]（97.5%・ともに 10/10）。並び **CLIMB_UNRESOLVED**（$\rho$ 平均 +0.23・7 対 3・p 0.34）、固定した場 **FIX_REMAINDER_REDUCED**（+0.074・10/10）、読みのフラグ **NO_IMPAIRMENT_FLAG**（$I$ = −0.017）。
- 独立検算（verdict.py を使わず arms.csv から）: 主な 6 量の平均と SD の差は 0。
