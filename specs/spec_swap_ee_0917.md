# 残り 0.11 は「自分の重みの押し」か — cap12 の網と ref の網で第2層の場を入れ替える（RL・ELU→ELU・14 腕 × 10 seed）

状態: **事前登録**（本ファイルの commit が登録。入れ替えの腕はまだ一つも走っていない）
作成: 2026-09-17 / 起草: Claude / 依頼: Issa「残り 0.11 を問う『cap12 の網と ref の網で場を入れ替える』」
前の走: `specs/spec_respdyn_ee_0917.md`（動く場の移植・`PARTIAL`・R = +0.112）・`specs/spec_l2cap_ee_0917.md`（cap12 `RESCUED`）・`specs/spec_resp_ee_0917.md`（固定した場の移植）
設計の出所: obsidian-research `可塑性喪失/測定/動く場の移植_応答低下からLoPへ_結果_0917.md` §10-2、`可塑性喪失/主張/中心主張v10作業リスト_0916.md` の格上げ計画「上流と下流の一本化」
実装: `src/swap_ee_0917.py`・`analysis/swap_ee_0917/{checks.py,verdict.py,launch.sh}`
branch: `claude/swap_ee_0917`

## 0. 一行と問い

respdyn_ee_0917 では、健康な t2 の網に「崩壊していく t10 の網の動く場」を渡しても、応答の床（一様 −30）より **0.112 上**で学べた（R = S2dyn_10r − S2u30r）。同じ走は、この残りを「自然な網では自分の重みが押し下げるが、移植先の網は押さずに登る（+3.5）」と読んだ（未検証）。

l2cap_ee_0917 の cap12 の網は、**自分の重みで押せない網**である。第2層の平均前活性は $\bar z_2\approx\|w_2\|\cos(w_2,e_2)\|\mu_2\|+b_2$ で、てこ $\|w_2\|\,\|\mu_2\|$ は ref の t10 で 270、cap12 の t10 で 9（t2 の ref は 11）。**cap12 の網と ref の網の t10 で、第2層の場を入れ替える。**

- **Q1（主・残り）**: 押せない網（cap12）に ref の動く場を渡したとき、床の上に残りは出るか。押す網（ref 自身）では残りは出ないか。
- **Q2（媒介）**: cap12 の救命（W の成長を止める → 学べる）は、第2層の応答の場を介しているか。「沈める」（cap12 の網に ref の場）と「戻す」（ref の網に cap12 の場）の両方向で見る。
- **Q3（成長のスイッチ）**: 同じ網・同じ場のまま、上限を外す／掛ける（重みの成長を許す／止める）と学習は変わるか。

## 1. なぜ（既存データ・REPORT_ONLY）

### 1.1 respdyn_ee_0917 の分解（t2 の網・Adam 初期化・10 seed）

| 腕 | E（継続 1 タスク目の online 精度） |
|---|---|
| N2r（自然） | 0.724 |
| S2_10r（t10 の固定した場） | 0.464 |
| S2dyn_10r（t10 の場が自然な継続に追随） | 0.324 |
| S2u30r（応答の床） | 0.212 |
| N10r（崩壊した網そのもの） | 0.169 |

- R = +0.112 [+0.081, +0.143]（10/10）。事後: S2dyn_10r の網自身の第2層は 750 更新で +0.57、終わりに +3.5 登った。自然な N10 は −15.1 → −8.9 と自分で沈んだ。
- resp_ee_0917: 崩壊した t10 の網に t2 の固定した場を戻すと、Adam 初期化でも E = 0.911（R2_10r）。自然な t10 の網の t10 の第2層は、訓練の微分が 0 の (ユニット, 画像) が 0.939。

### 1.2 cap12 と ref の t10（l2cap_ee_0917・10 seed 平均）

| | $\|w_2\|$ | $\|\mu_2\|$ | $\cos(w_2,e_2)$ | $\bar z_2$ | 微分 0 の割合 | online |
|---|---|---|---|---|---|---|
| ref t10 | 3.88 | 69.6 | −0.195 | −50.9 | 0.94 | 0.158 |
| cap12 t10 | 1.61 | 5.5 | −0.223 | −2.4 | 1e−4 | 0.866 |

- cap12 の網は ref と task 1 を共有し、task 2 から上限（第1層の $|q|$・$\|v\|$ と第2層の行ノルム、半径は task 1 終端値）で育った。ユニットの番号は両網で同じ初期化から来ている。
- **本 runner で作った cap12 の軌道は、l2cap_ee_0917 の記録と bit 一致する**（task 1–3 の `EL.unit_arrays` 39 配列、S0c と本走の全 seed・全 12 タスクで記録）。

## 2. 設計

### 2.1 箱（resp_ee / respdyn / l2cap と同一）

Random Label MNIST（1200 枚固定）、784–100–100–10、両隠れ層 ELU(α=1)、Adam lr 1e−3・β (0.9, 0.999)・ε 1e−8、batch 16、6000 更新/タスク、seed 0–9、float32、CPU 1 スレッド、`torch.set_flush_denormal(True)`。

### 2.2 1 seed の流れ

1. **ref の自然軌道** task 1–12（resp_ee の `run_prefix`）。t2 と t10 の終端で分岐状態（重み・Adam・ラベルと標本順の乱数・両層の前活性の場）を保存。
2. **cap12 の自然軌道** task 1–12（respdyn の `Stepper` に、`run_one` と同じ順で各更新の直後に上限を掛けたもの）。t10 で分岐状態を保存。
3. **14 腕**。各腕は分岐状態から、その網の自分のラベルと標本順で 2 タスク続ける。**全腕 Adam 初期化**（respdyn の主腕と同じ）。前向きは resp_ee の固定補正つき $a_2=\phi(z^0_2)+[\phi(z_2+d)-\phi(z^0_2+d)]$（分岐点で logits は自然な網と bit 一致し、訓練の微分は $\phi'(z_2+d)$）。

### 2.3 腕

場: **固定** $d=z_2^{(\rm 元)}-z_2^{(\rm 網)}$（float64 で引いて float32）、**動く** $d(s)=z_2^{(\rm 影)}(s)-z_2^{(\rm 網)}$（影は元の網の自然な継続そのもの。Adam 継続・自分のラベルと標本順・**元の網の上限つき**。respdyn の `DynField`）、**一様** $d=-30$。上限: **own** = cap12 の task 1 終端の半径、**here** = 分岐状態の半径（分岐点では何も動かさず、以後の成長だけを止める）、なし。

| 腕 | 網 | 場 | 上限 | 役割 |
|---|---|---|---|---|
| NC_r | cap12 t10 | — | own | cap12 の自然（健康） |
| NC_free_r | cap12 t10 | — | なし | 上限を外した自然 |
| SC_fix_r | cap12 t10 | ref t10（固定） | own | 沈める（固定） |
| **SC_dyn_r** | cap12 t10 | ref t10（動く） | own | **沈める（動く）** |
| SC_dyn_free_r | cap12 t10 | ref t10（動く） | なし | 沈める・成長を許す |
| **SC_u30_r** | cap12 t10 | 一様 −30 | own | **cap12 の網の床** |
| **NR_r** | ref t10 | — | なし | **ref の自然（崩壊）**。resp_ee の N10r と同一 |
| NR_cap_r | ref t10 | — | here | 崩壊後に成長を止める |
| RR_fix_r | ref t10 | cap12 t10（固定） | なし | 戻す（固定） |
| **RR_dyn_r** | ref t10 | cap12 t10（動く） | なし | **戻す（動く）** |
| RR_dyn_cap_r | ref t10 | cap12 t10（動く） | here | 戻す・成長を止める |
| **RR_u30_r** | ref t10 | 一様 −30 | なし | **ref の網の床** |
| S2dyn_10r | ref t2 | ref t10（動く） | なし | respdyn の主腕の再走 |
| S2u30r | ref t2 | 一様 −30 | なし | resp_ee の床の再走 |

- 「沈める」と「戻す」の場は、同じ seed の 2 つの網の t10 の第2層の前活性の差（ユニット番号の対応は共有した初期化による）。
- SC_* の網は ref の場で訓練の微分がほぼ 0 から始まり、RR_* の網は cap12 の場でほぼ全部が生きた状態から始まる（S1 で確認）。

## 3. 測る量

- **E** = 継続 1 タスク目の online 精度（6000 更新の更新前の正解率の平均）。継続 2 タスク目は副。
- 各腕: resp_ee の `unit_summary`（開始と各タスク終端: 訓練の微分・0 の割合・$n_{\rm eff}$・$\bar z$・$\|\mu_2\|$ など）、respdyn の軌跡の probe（750 更新ごと: 網自身の $\bar z_2$、場の平均、実効引数、0 の割合、$n_{\rm eff}$）、第2層の行ノルムの平均、上限からの超過（own の半径に対して・腕の上限に対して）、上限が書いた行数、状態 sha256、影の状態 sha256。
- **再現の記録**（seed ごと）: ref の軌道の各タスクの状態 hash = resp_ee の記録、cap12 の軌道の各タスク終端の `EL.unit_arrays` = l2cap_ee_0917 の退避した units（sha256 照合済みのときだけ）、NR_r と S2u30r = resp_ee の N10r と S2u30r（状態 hash）、S2dyn_10r と S2u30r = respdyn の記録、各影 = その網の自然な継続（t11・t12 の状態 hash）、分岐点の logits の一致、動く場の $d(0)$ = 固定した場。

## 4. endpoint（seed ごとに作って seed 間で要約）

### 4.1 Q1（主）

- $R_C=E(\text{SC\_dyn\_r})-E(\text{SC\_u30\_r})$: 押せない網の、ref の動く場のもとでの床の上の残り。
- $R_R=E(\text{NR\_r})-E(\text{RR\_u30\_r})$: 押す網（自分の崩れていく場）の床の上の残り。
- $D=R_C-R_R$。
- 報告: $R_{T2}=E(\text{S2dyn\_10r})-E(\text{S2u30r})$（respdyn の R の再走）と $R_C-R_{T2}$。

### 4.2 Q2（媒介）

$TE=E(\text{NC\_r})-E(\text{NR\_r})$、沈める $L_S=E(\text{NC\_r})-E(\text{SC\_dyn\_r})$（固定版 $L_S^{\rm fix}$）、戻す $G_R=E(\text{RR\_dyn\_r})-E(\text{NR\_r})$（固定版 $G_R^{\rm fix}$）、動きの寄与 $M_C=E(\text{SC\_fix\_r})-E(\text{SC\_dyn\_r})$。比 $L_S/TE$・$G_R/TE$ は平均の比で報告。

### 4.3 Q3（成長のスイッチ）

$P_C=E(\text{SC\_dyn\_r})-E(\text{SC\_dyn\_free\_r})$、$P_R=E(\text{RR\_dyn\_cap\_r})-E(\text{RR\_dyn\_r})$、$P_N=E(\text{NR\_cap\_r})-E(\text{NR\_r})$、$P_F=E(\text{NC\_r})-E(\text{NC\_free\_r})$。正なら「上限を掛けた方が学べる（成長が学習を削る）」。

### 4.4 報告のみ

各腕の軌跡（網自身の $\bar z_2$ の動き・場の動き・実効引数・0 の割合・$\bar n_{\rm eff}$）、継続 2 タスク目の同じ量、上限を外した腕が cap12 の半径をどれだけ超えたか、記憶率。

## 5. 判定

### 5.1 帯

有効 seed の差の平均・SD・対応差 t 区間。**Q1 は 97.5%**（$R_C$ と $D$ の 2 量の Bonferroni）、Q2・Q3 は 95%。符号は区間の下端 > 0 で +、上端 < 0 で −、それ以外 0。t 分位点は 0916 の実装。

### 5.2 適用条件

- **(A) 再現と完全性**: seed ごとに §3 の再現の記録がすべて真（ref の 12 タスク、cap12 の 12 タスク、NR_r・S2u30r・S2dyn_10r の 2 タスク、4 本の入れ替えの動く腕の影の 2 タスク、S2dyn_10r の影、全腕の分岐点の logits、動く腕の $d(0)$）、14 腕 × 2 タスクが有限、上限を掛けた腕（NC_r・SC_fix_r・SC_dyn_r・SC_u30_r・NR_cap_r・RR_dyn_cap_r）は各タスク終端で半径の超過が 1e−4 以下（float32 の射影の丸め 4·eps32·R を R ≤ 200 で押さえた値）、NC_r・SC_dyn_r・NR_cap_r・RR_dyn_cap_r は各タスクで上限が行を書いた。有効 seed が 8 未満なら全ラベル INAPPLICABLE。
- **(B) 現象がある**: $TE$ と $R_{T2}$ の 95% 区間の下端 > 0。満たさなければ Q1・Q2 は NOT_REPRODUCED。

### 5.3 ラベル

**Q1**（97.5%）:

| $s(R_C)$ | $s(D)$ | ラベル |
|---|---|---|
| + | + | **REMAINDER_FOLLOWS_HOST** |
| + | 0 | REMAINDER_IN_BOTH |
| + | − | REMAINDER_LARGER_IN_PUSHING_HOST |
| 0 / − | — | NO_REMAINDER_IN_NONPUSHING_HOST |

**Q2**（95%）: $L_S$ と $G_R$ がともに + なら **BOTH_WAYS**、片方だけなら SINK_ONLY / RESTORE_ONLY、どちらも + でなければ NEITHER。

**Q3**（95%・4 本それぞれ）: + なら GROWTH_COSTS、− なら GROWTH_HELPS、0 なら NO_EFFECT。

### 5.4 読み方（登録）

- **REMAINDER_FOLLOWS_HOST** → 残りは「押せない網」に現れ、「押す網」には現れない。respdyn の読み（残り = 自分の重みの押しが無いこと）と整合する。Q3 の $P_C$ が + なら、押しの正体は重みの成長だと言える。$P_C$ が 0 なら、押せないのは成長の有無ではなく、てこ（すでに育った $\|w_2\|\|\mu_2\|$）の小ささによると書く。
- **REMAINDER_IN_BOTH** → 残りは押す網にもある。「自分の重みの押し」の読みは支持されない。
- **NO_REMAINDER_IN_NONPUSHING_HOST** → 押せない網でも床まで落ちる。respdyn の残りは t2 の網に特有（押さないことでは説明できない）。
- **Q2 の BOTH_WAYS** → cap12 の救命は第2層の応答の場を介している。$L_S/TE$ と $G_R/TE$ で媒介の割合を書く（1 を超えうる）。これで同じ箱・同じ seed で「W の成長 → 輸送 → 応答 → 学習能力」の 4 本の矢印がすべて介入でつながる。
- **$P_N$ が 0** → 崩壊した網は、成長を止めても自力で戻れない。+ なら崩壊後も押しが学習を削っている。
- 入れ替えは人工的な関数変更（固定補正）で、影が運ぶのは第2層の前活性の動きだけ。前向きの値は網自身のまま。
- すべて「この ELU→ELU・乱数ラベル・6000 更新・t10 の分岐・Adam 初期化・継続 2 タスク」に限る。

## 6. 検査（登録前に実行。許容は導出から）

`python3 analysis/swap_ee_0917/checks.py` → `results/swap_ee_0917/checks.json`。本走そのものが §3 の再現を全 seed で記録し、判定はそれを適用条件 (A) で要求する。**登録前の開発実行（検査ごとに `--only`）で S0c・S1・S8・S9 はすべて pass、変異 24/24 を検出**。S1 の seed 0 では、t10 の第2層の訓練の微分が 0 の割合は ref の場で 0.964、cap12 の場で 0.0002 だった。S1 の出力には腕の E を載せていない（予測の前に入れ替えの腕の結果を見ないため）。全検査（S-cost を含む）は本 spec の commit の後、runner に登録 commit を書き込んでから 1 回通しで実行し、その `checks.json` を本走の条件にする。

| 検査 | 要求 | 変異 |
|---|---|---|
| S0c cap12 の軌道 | 80 epoch・task 1–2 の `EL.unit_arrays` の全配列が l2cap_ee_0917 の cap12_s0 の退避（sha256 照合）と bit 一致、task 2 で上限が行を書く | 4（task 1 から上限・初期化の半径・直交の上限を抜く・行ノルム上限を抜く） |
| S1 腕（seed 0・80 epoch・prefix は 1 回） | 8 腕で: 分岐点の logits が一致、入れ替えが本物（ref の場の cap12 の網は開始時に 0 の割合 ≥ 0.9、cap12 の場の ref の網は ≤ 0.01、自然な腕はそれぞれ自分の値）、動く場の $d(0)$ = 固定した場、両方の影が自然な継続（t11・t12 の hash）、上限の腕は超過 ≤ 1e−4 かつ行を書く、上限を外した cap12 は cap12 の半径を超える、NR_r = resp_ee の N10r | 4（cap12 の影が上限なし・own の上限を掛けない・戻す腕が ref 自身の場・固定補正なし） |
| S8 判定 | 合成 seed 14 場面でラベルと推定値（1e−9）、t 分位点 | 13 |
| S9 CLI | provenance（実験名・spec・flush・分岐 10・継続 2・各モジュールの hash）、定数（分岐 10・80 epoch・12 タスク・14 腕） | 3 |
| S-cost | 4 プロセス同時（腕を 3 本に減らした 1 seed）で exit 0・メモリから 4 スロット以上。1 seed = prefix + 素の腕 9 本 + 影つきの腕 5 本で投影 | — |

## 7. 事前予測（Claude・入れ替えの腕を回す前）

| 項目 | 予測 |
|---|---|
| (A)(B) | 満たす 90% |
| **Q1** | **REMAINDER_FOLLOWS_HOST 40%** / REMAINDER_IN_BOTH 25% / NO_REMAINDER_IN_NONPUSHING_HOST 25% / LARGER_IN_PUSHING 5% / その他 5% |
| $R_C$ | +0.08（0.03〜0.15）。cap12 の網は b₂ と回転で登れる（てこ 9 でも ±18 まで動かせる）。床は cap12 の特徴の読み出しだけで 0.2〜0.3 |
| $R_R$ | +0.02（−0.01〜+0.05）。ref の t10 の特徴はほぼ −1 の定数で、読み出しの床は 0.12〜0.16 |
| $R_{T2}$ | +0.112（記録と bit 一致で同じ値）99% |
| **Q2** | **BOTH_WAYS 80%**。$L_S\approx0.55$、$G_R\approx0.70$、$TE\approx0.68$（$G_R/TE$ は 1 を超えうる: resp_ee の R2_10r は 0.911） |
| $P_C$ | NO_EFFECT 50% / GROWTH_COSTS 35% / GROWTH_HELPS 15% |
| $P_R$ | GROWTH_COSTS 45% / NO_EFFECT 45% / GROWTH_HELPS 10% |
| $P_N$ | NO_EFFECT 70%（崩壊後に成長を止めても戻れない） |
| $P_F$ | NO_EFFECT 55% / GROWTH_COSTS 35%（上限を外しても 2 タスクでは崩れきらない。ref は t1 から T½ 7） |

**Issa の予測**（2026-09-17・登録 commit 64adf3b の後、本走の前にチャットの選択式で記入）: Q1 **REMAINDER_FOLLOWS_HOST**、Q2 **BOTH_WAYS**、$P_N$ **NO_EFFECT**、$P_C$ **NO_EFFECT**。$P_R$・$P_F$ は未記入。ラベルだけを記録し、Issa 自身の確率は付けない。Claude の最頻ラベルと全項目で一致。

## 8. 費用と実行

- S-cost の投影で 1 seed 10〜15 分を想定（prefix 24 タスク＋腕 28 タスク＋影 10 タスク）。
- 並列数は min(4, S-cost のスロット, 起動時の空きメモリで収まる数)。登録時点で別セッションのジョブが走っているので上限を 4 にする。seed 順。`provenance.json` のある seed は飛ばす。
- 起動は本 spec の登録 commit を runner に書き込み、全検査を再実行して all_pass になってから。

## 9. 結果の置き場所と後片付け（CLAUDE.md §4）

- `results/swap_ee_0917/runs/s<seed>/`（`prefix.csv`・`arms.csv`・`traj.csv`・`provenance.json` は git、`units.npz` は退避）。
- 判定: `analysis/swap_ee_0917/verdict.py` → `results/swap_ee_0917/{summary.md,verdict.csv,paired.csv,provenance.json}`。
- `units.npz` と `results/_checks_swap_ee_0917/` は `obsidian-research-data/swap_ee_0917/` へ退避し `backup_manifest.json` を commit。結果 commit の後、その日のうちに main へ統合し worktree と branch を消す。
