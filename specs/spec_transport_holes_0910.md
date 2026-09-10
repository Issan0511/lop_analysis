# 輸送問題の穴（ReLU・GELU・SiLU）を箱 B で塞ぐ spec 0910
状態: 検証中 / 更新: 2026-09-10
親: [[中心主張v9草案_0910]] §11（穴の一覧の正本） / [[輸送問題の穴を塞ぐ_引き継ぎ_0910]] / [[長い地平線で沈下と幅を分ける_結果_0910]]
run id: `transport_holes_0910`（4 つの副走 A / B / C / D をまとめて 1 回登録する）

## 0. 何を決めるか

[[中心主張v9草案_0910]] §11 の監査で、輸送問題は **leaky・ELU の箱 B に限って**しか決着していないことが分かった。残る穴は 3 つ:

- **ReLU**: 吸収の静的条件は箱 A で厳密。だが **「死 → LoP」は箱 B で未検定**。

  V9 §11 は「箱 B の t120 精度は R −2.7/0.0/−2.6 pt で leaky −2.9 と同程度」と書くが、**これは腕ごとに違う seed を 1 つずつ引いた単点差で、窓も揃っていない**。本 spec の窓（base = t16–20 の中央値）で committed データから 3 seed 揃えて引き直すと、**t96–100 の低下は**

  | 腕 | seed 0 | seed 1 | seed 2 | 出典 |
  |---|---|---|---|---|
  | R | 1.79 | 2.00 | 2.12 | `leak_ladder_force_posthoc_0910` |
  | LR (leaky 0.1) | 2.21 | 1.96 | 2.08 | `width_sink_clamp_0909` の ref |
  | LR001 (leaky 0.01) | 2.32 | 2.52 | 2.16 | `leak_ladder_force_posthoc_0910` |
  | ELU1 | 0.89 | 1.56 | 1.90 | `width_sink_clamp_0909` の ref |

  **ReLU は leaky と区別が付かない（むしろわずかに良い）。**42–49/100 のユニットが死んでいる（実測・§7）のに、である。「死んでいても余分には失っていない」は**この形でなら引ける**。
- **GELU・SiLU**: **箱 B にハーネス実装すら無かった**（`valley_acts_0910` で実装済み・未走）。谷の向こうで φ′<0 だから「増やせ」が逃走になる、は**導出であって測定でない**。
- **駆動源の符号**: 負側の e·φ′ の符号は leaky・Snake でしか測っていない。

`clamp_horizon_0910` は leaky・ELU について「**幅を止めると累積損失の 82–91% が消える／深さを止めても 3–9% しか戻らない**」を出した。**同じ介入を ReLU と谷越え型に掛けて、活性化ごとに lever が違うのかを決める。**

## 1. 箱を混ぜない

本 spec は**すべて箱 B**（pmnist・784→100→100→10・Adam・batch 16・lr 0.001）である。箱 A（合成教師・1 層・SGD）の ReLU 吸収・谷越えの結果は**引用するが、根拠には使わない**。箱をまたぐ駆動源の同一性（V9 §11 の X1）は本 spec の範囲外。

## 2. 走

既 commit のファイルは**編集しない**（committed 結果の provenance の sha256 が動くため）。新しい自己完結モジュールが `clamp_horizon_0910` / `width_sink_clamp_0909` / `why_down_posthoc_0910` を import して腕だけ足す。`clamp_horizon_0910` が `width_sink_clamp_0909` に対して取ったのと同じやり方。

| 副走 | run id | 腕 | 地平線 | 本数 | 実測見積 |
|---|---|---|---|---|---|
| **A** 谷越え型の参照軌道 | `long_horizon_acts_0910` | GELU・SILU × seed 0–2 | t1–400・介入なし | 6 | 約 4 分/本 |
| **B** ReLU のクランプ | `clamp_horizon_acts_0910` | R × seed 0–2 × {ref, dclamp, wclamp, wcap2} | t21–400 | 3 | 約 15 分/本 |
| **C** 谷越え型のクランプ | `clamp_horizon_acts_0910` | GELU・SILU × seed 0–2 × 同 4 クランプ | t21–400 | 6 | 約 15 分/本 |
| **D** 駆動源の符号 | `why_down_acts_0910` | R・ELU1・GELU・SILU（seed 0） | t1–100・2 窓 | 4 | 約 4 分/本 |

**順序**: A → （B と C と D）。C と D の GELU/SiLU は A が作る参照軌道を検算の錨に使うので、A が先でなければならない。B は A に依存しないので A と並走してよい。

クランプの定義・分岐点・測定は `clamp_horizon_0910` §1 と同一（init から t1–20 を 1 回、bit 一致の t20 状態から 4 分岐、毎更新クランプ、測定はタスク終端のみ）。

### 2.1 谷越え型に固有の測定（新規・ラベル無しの副測定ではなく判定量）

`dead_hard`（全プローブ標本で φ′ < 1e−6）と `sat`（φ′ < 0.05 の対の割合）は **ReLU と leaky を念頭に定義されていて、符号付きのゲートに閾値を掛けている**。谷越え型では φ′ が**負**になりうるので、この 2 つは「死」と「反転」を混ぜる。しかも谷の向こうの最小ゲートは **GELU −0.1289・SiLU −0.0998** で、**leaky の床 0.1 より絶対値が大きい**。つまり反転ユニットは動かないのではなく**逆向きに、leaky より速く動く**。既存の counter をそのまま読むと**真実の反対を書いた表になる**。分けて記録する:

- **`gate_neg_frac`** = プローブの (標本 × ユニット) 対のうち φ′(z) < 0 の割合
- **`inv_units`** = プローブ標本の 50% 超で φ′(z) < 0 となるユニット数（＝谷の向こうに座ったユニット）
- **`beyond_frac`** = z < z_c の対の割合（z_c は谷底。GELU −0.7517915239 / SiLU −1.2784645428）
- **`immobile_units`** = **|φ′|** < 1e−6 が全プローブ標本で成り立つユニット数（＝本当に動かない）。ReLU ではこれが `dead_hard` と一致し、谷越え型では 0 に近いはず。

**非谷型（R・LR・ELU）では `gate_neg_frac`・`inv_units`・`beyond_frac` の 3 つは恒等的に 0 でなければならない**（φ′ ≥ 0）。これを検査として持つ（恒真ではない: **谷型では `gate_neg_frac` > 0.05 になることを同時に要求する**）。

## 3. 検算（G1 系）— **副走ごとに強さが違う。それを明記する**

| 副走 | 錨 | 比べるもの | 強さ |
|---|---|---|---|
| **B**（R） | `results/leak_ladder_force_posthoc_0910/R_s{seed}_units.npz` | `ref` 腕の t1–100・per-unit 6 配列（zbar_i・sd_i・cnorm_i・m_i・bias_i・star_i）・**600 件**・maxabs ≤ 1e−10 | **強い**。ただし 2 点の留保: (1) **`ref` 腕のみ** —— R には `dclamp`/`wclamp` の committed 参照が無いので被覆は LR/ELU1（3 クランプ × 80 タスク × 5 量）より弱い。(2) **錨は登録走ではなく事後走**（`leak_ladder_force_posthoc_0910` は自分の docstring で post-hoc・unregistered と宣言していて、provenance に `spec_sha256` も `host_sha256` も持たない）。錨自体は `gate_scale_invariance_0909` と 0.0 で一致しているので値は正しいが、**「登録走の再現」ではない** |
| **A**（GELU/SiLU） | **無い**（初走） | — | **G1 は存在しない。「G1 0.0」と書いてはならない。**代わりに §3.1 の 3 つ |
| **C**（GELU/SiLU） | A が書き出す `results/long_horizon_acts_0910/{arm}_none_s{seed}_units.npz` | `ref` 腕の t1–100・per-unit 6 配列・600 件・maxabs ≤ 1e−10 | **中**。A と C は**別のコード経路**（A は `C.loop`・MEAS=[625]／C は `clamp_horizon_0910.loop`）なので再現は非自明。ただし錨は同じ走の産物であって独立の登録走ではない |
| **D**（R） | 同 B | 各窓終端の ‖W̃ᵢ‖・maxabs ≤ 1e−10 | 強い |
| **D**（ELU1） | `results/elu_growth_0909/ELU1_none_s0_units.npz` | 同上（既存 `why_down_posthoc_0910` と同じ） | 強い |
| **D**（GELU/SiLU） | A の出力 | 同上 | 中 |

### 3.1 A の検算（committed 参照が無いときに何をするか）

1. **測定の非侵襲**: 毎タスク終端で測定する走と、t1 と t400 でだけ測定する走で、**t400 の per-unit 6 配列が bit 一致**。変異対照: 測定のたびに W1 に +1e−9 を足すと落ちる（`measure_mut`）。
2. **ストリームの腕非依存**: t1–5 で引かれる `perm`・`idx`・`order` が **LR の走と bit 一致**（腕は乱数系列に入らない、という S-init の主張の直接検査）。変異対照: seed を 1 ずらすと落ちる。
3. **`valley_acts_0910._selftest`**: φ が torch のカーネルと 8.9e−16、dφ が autograd と 2.2e−16、谷底 z_c が dφ の根、符号構造 φ′>0 (z>z_c) / φ′<0 (z<z_c)、φ の最小が z_c、変異対照（z_c+0.05 で根の検査が発火）。provenance に値を書く。

**A の provenance には `g1` というキーを置かない。**`no_committed_reference: true` と書く。

## 4. 判定（事前登録）

### 4.A 谷越え型は死が飽和するか（副走 A）

窓 late = t301–400、base = t16–20。3 seed 一致でのみラベル。ELU1 の実測は `dead_hard`(t400) = 3/6/9 で `DEATH_SATURATES`。

- **A1 反転の蓄積**: `inv_units`(t400)。3 seed とも ≥ 20 → **`INVERSION_ACCUMULATES`**／3 seed とも ≤ 10 → **`INVERSION_SATURATES`**／他 `PARTIAL`。
- **A2 精度**: late 窓の acc 傾き（pt/100task）。`long_horizon_0910` の LR −0.30/−0.41/−0.60・ELU1（`ELU_DECELERATES`）と同じ帯に入るか。3 seed とも ELU1 の late 傾きより 0.3 pt/100task 以上負 → **`VALLEY_FALLS_FASTER`**／3 seed とも同等（差 < 0.3）→ **`VALLEY_SAME_BAND`**／他 `PARTIAL`。
- **A3 累積損失と ‖W̃‖ の順**: `long_horizon_0910` は「累積損失は ‖W̃‖ の順」（leaky 15.2/4.3pt・ELU 13.7/3.1pt・Snake 9.0/0.9pt）だった。谷越え型 2 腕を加えて **5 点でこの順序が保たれるか**。保たれる → **`WIDTH_ORDER_HOLDS`**／破れる → **`WIDTH_ORDER_BREAKS`**（V9 の幹に対する反証）。

### 4.B / 4.C クランプ（副走 B・C）

**ρ・窓・ゲート・ラベルは `clamp_horizon_0910` の §2 を追補 1 で改訂した形をそのまま使う**（再定義しない）。**正本は追補 1 であって §2 の表ではない** —— §2 の表には廃止済みの `WIDTH_DOSE_MONOTONE` が残っており、害の帯も `LEVEL_SHIFT_ONLY` も D の傾き比も入っていない。§2 だけを写すと**同じ設計ミスを 2 回目**として作ることになる。すなわち late = t301–400、base = t16–20、

$$L_{\rm ref}=\mathrm{acc_{base}}-\mathrm{acc_{ref}(late)},\qquad \rho_C=\frac{\mathrm{acc}_C(\rm late)-\mathrm{acc_{ref}(late)}}{L_{\rm ref}}$$

ゲート G0（L_ref ≥ 2.0 pt）・G3（late 窓のタスク内 CE 改善が 90% 以上のタスクで正。**精度でゲートしない** —— 精度が判定量なので循環する）・G4（dclamp が深さを保つ）・G5（wclamp が沈下を止めていない）、害の帯 ρ ≤ −0.2 → `HARMS`、無効果 −0.2 < ρ ≤ 0.2、`REMOVES` は **ρ ≥ 0.5 かつ slope_late(C) ≥ slope_late(ref) + 0.3 pt/100task**（満たさなければ `LEVEL_SHIFT_ONLY`）、D の第 2 項は **late 窓の σ_inv の傾き比** `b_S(dclamp)/b_S(ref) ≥ 0.75`（‖W̃‖ の水準比ではない）、用量 E は `DOSE_GRADED` / `DOSE_THRESHOLD` / `CAP_BEATS_CLAMP` / `DOSE_PARTIAL`（B が `REMOVES` でなければ `NOT_TESTABLE_NO_WIDTH_EFFECT`）、ゲート → ラベルの対応表も追補 1 R5 のまま。

**`MIXED`**（spec 本文に定義が無く実装だけが付けていたラベル）をここで明示的に定義する: **可検定な seed が 3 つ揃っているのに 3/3 一致しなかったとき**に付く。「効果が無かった」ではなく「seed 間で割れた」の意味であり、**どちらの記名予測も支持しない**。

**C が測れない見込みへの備え**: leaky で C が `NOT_TESTABLE_DEPTH_ALSO_STOPPED` になったのは「`wclamp` が沈下も止めてしまう（z̄ −2.90 対 ref −8.29）ので G5 が 3 seed とも落ちる」という**構造的**理由だった。負側に非零ゲートを持つ GELU/SiLU も同型のリスクがある。**C は落ちうる。決め手は C ではなく 4.C の A（ρ_dclamp）なので、C が落ちても走の価値は落ちない。**

追加（この spec に固有）:

- **F 死と精度の分離（R のみ）**: `dclamp`（沈下を止める）が `dead_hard`(t400) を ref より 3 seed とも 20 以上減らす → **`DEPTH_DRIVES_DEATH`**／3 seed とも 5 未満しか減らさない → **`DEATH_NOT_FROM_DEPTH`**／他 `PARTIAL`。**これは A（精度の ρ）と独立に読む**: 沈下を止めて死が減っても精度が戻らなければ「死は LoP の原因ではない」が箱 B で立つ。
- **G 反転と精度の分離（GELU/SiLU のみ）**: 同じ形で `inv_units`(t400) について。`DEPTH_DRIVES_INVERSION` / `INVERSION_NOT_FROM_DEPTH` / `PARTIAL`。

**G0 が通らない見込みへの備え**: R の t120 精度低下は leaky と同程度（−2.7 pt/100task 相当）なので L_ref ≥ 2.0 pt は通る見込み。谷越え型は未知。通らなければその腕は `NOT_TESTABLE_NO_LOSS` で、**それ自体が「谷越え型は箱 B で LoP しない」という所見**である（V9 §11 の穴 2 に対する答えになる）。

### 4.D 駆動源の符号（副走 D）

`why_down_posthoc_0910` の分解 $\partial L/\partial m_i=\sum_x\varphi'(z_i)e_i(x)S(x)$ を 4 活性化で取る。窓 (21,25) と (96,100)、区間 1–20 / 21–100 / 101–625。

**負側（z ≤ 0）の $\sum \varphi' e S$ の符号**を主判定にする。3 活性化の比較で、両窓・両区間（21–100・101–625）で符号が一致したときだけラベル:

- leaky・ELU: **負**（＝浮き。既発表と一致すること。一致しなければ実装の誤り）
- **GELU・SiLU: 正**（＝沈み）→ **`VALLEY_SIGN_FLIPS`**
- **R: 恒等的に 0**（負側の φ′ = 0）→ 検査。0 でなければ実装の誤り
- GELU/SiLU が負のまま → **`VALLEY_SIGN_HOLDS`**（導出が外れ・V9 §2 段階 2 の反証）
- 窓や区間で符号が割れる → **`VALLEY_SIGN_MIXED`**

副（ラベル無し）: 谷の向こうの標本だけに絞った $\sum \varphi' e S$、正側との比、誤答/正答の内訳。

## 5. 事前予測（記名・**走る前**）

### Claude

- **A1 `INVERSION_ACCUMULATES`**。谷の向こうでは φ′<0 なので、負側標本の「もっと活性化しろ」（e<0）が**沈み**になる。leaky（φ′=+a）・ELU（+e^z）が持っていた**復元の regime が無い**ので、深くなるほど押しが強まる正のフィードバックになり飽和しない。t6 の時点で既に z < z_c の対が GELU 0.57 / SiLU 0.40 ある（実測・§7）。
- **A2 `VALLEY_FALLS_FASTER`**、**A3 `WIDTH_ORDER_HOLDS`**。
- **B（R）: A `DEPTH_NO_EFFECT`（ρ_dclamp ≤ 0.2）、B `WIDTH_REMOVES_LOSS`（ρ_wclamp 0.5–0.8）。** ReLU でも lever は幅で、死は随伴。**F は `DEPTH_DRIVES_DEATH`**（沈下を止めれば死は減る。だが精度は戻らない）。**この組み合わせ（死は減るのに精度は戻らない）が立てば「死 → LoP」は箱 B で棄却される。**
- **C（GELU/SiLU）: A `DEPTH_REMOVES_LOSS`（ρ_dclamp ≥ 0.5）。**これが本 spec の**決め手**である。谷越え型でだけ沈下が LoP を作るなら、V9 の二分（ゲートが正のまま残る型／消える型／**反転する型**）が介入で確定する。B `WIDTH_REMOVES_LOSS` も立つ（両方立てば「分離できていない」＝ 追補 1 R7 の「両者外れ」行）。
- **D `VALLEY_SIGN_FLIPS`。**

### Issa（**本人から直接取得・2026-09-10・走る前**）

- **B（R）: `DEPTH_NO_EFFECT` / `WIDTH_REMOVES_LOSS`。**「戻さない（幅が lever）」。Claude と同じ。
- **C（GELU/SiLU）: `DEPTH_REMOVES_LOSS`。**「戻す（谷越え型は別型）」。Claude と同じ。
- **A1: `INVERSION_SATURATES`（≤ 10/100）。****ここだけ Claude と割れる。**

（V9 §10 の「谷越え型は復元の regime が無く死が頭打ちにならない側の候補」は Claude の読みであって Issa の予測ではなかった。本人確認で `INVERSION_SATURATES` に確定。）

### 分岐点と反証の形

- **Issa と Claude が割れるのは A1 だけ。** Issa `INVERSION_SATURATES` / Claude `INVERSION_ACCUMULATES`。
  この割れ方には意味がある: Issa は **C（dclamp が戻す）と A1（反転は飽和する）を同時に主張している**。つまり「谷越え型で沈下が LoP を作るのは、反転ユニットの**数**が増えるからではない」。Claude の導出（正のフィードバックで反転が増え続ける）は、C が立って A1 が `SATURATES` だと**機構の説明としては外れる**（結論だけ当たる）。
- **決め手は 4.C の A（ρ_dclamp）。** ≥ 0.5 なら「活性化によって lever が違う」が立ち、両者が当たる。≤ 0.2 なら **leaky・ELU・GELU・SiLU の 4 型すべてで lever は幅**で、V9 の幹が強くなり**両者外れ**。
- **4.B の F と 4.B の A が同時に立つ**（死は減るが精度は戻らない）と、箱 A の「ReLU は死へ直結 → LoP」が箱 B で**否定される**。
- D が `VALLEY_SIGN_HOLDS` なら V9 §2 段階 2 の導出そのものが外れ、C の予測の根拠が両者とも消える。

**通算**: Issa 8/8 前後、Claude 2/6（`long_horizon_0910` で Claude は 4/4 外し）。**重みは Issa 側に置く。**

## 6. 変異対照（反空虚・**通算 7 回目の空虚な検査を出さないため**）

`clamp_horizon_0910` の対照 10 本をそのまま継承する。加えて:

- **対照は本ループの前に走らせる**（引き継ぎノート §4 の宿題）。落ちたときに 15 分を捨てないため。下限は対照が実際に走る状態（t20）の床 `f32_rowmean_bound_t20` で評価する（追補 1 R1）。**本ループの後で、観測値が対照値の 1/100 以下であることを改めて assert する**（前倒しで失われる「観測値も上回る」性質を戻す）。
- **G1 の件数ガード**: B は 600 件、C は 600 件を要求する（キー 0 一致で maxabs 0.0 が通るのを防ぐ）。
- **谷検査の変異対照**: `gate_neg_frac` を非谷腕（R）で測ると 0、谷腕で測ると > 0.05。両方を要求する（片方だけなら恒真になりうる）。
- **A の非侵襲対照**: `measure_mut=True`（測定のたびに W1 += 1e−9）で t400 の bit 一致が落ちる。
- **D の R 検査の変異対照**: ReLU の負側寄与が恒等的に 0 であることは恒真に見えるが、φ′ を `dphi(z)+1e−12` に差し替えると落ちる。

## 7. 走る前に実測した数（この spec を書く時点で既知・判定量ではない）

- ReLU 参照軌道は `clamp_horizon_0910.loop` で再走すると `leak_ladder_force_posthoc_0910` の R と **t1–24 で 6 配列とも maxabs 0.0**（変異対照 6.32）。**B の G1 は成立する。**
- `wcap2` の上限（2×t20 の ‖W̃ᵢ‖）に最初に触れるタスク: **R は t27–28**（LR は t57–67・ELU1 は t52–64・SNA は t80–94）。`cap_bind_steps > 0` の assert は R で安全に通る。t100 での最大比は R 4.0–11.0 に対し LR 2.4–2.7 で、**ReLU は生き残った行だけが速く伸びる**。
- 6 タスクの試走（判定窓の外）: GELU acc 0.933・SiLU 0.930・ELU1 0.926・R 0.920。`z < z_c` の対の割合は GELU 0.57・SiLU 0.40。**谷の向こうの regime は実際に占有されている。**
- 本 spec の走はすべて **lab**（i9-13900KF・torch 2.12.0+cu126・numpy 2.4.6）で行う。**white-san で作られた committed 参照との bit 一致は実測で確認済み**（`clamp_horizon_0910` の LR seed0 スモークで G1 rows/units とも 0.0・12 行 60 配列・変異対照 4.71）。[[machines-and-network]] の「参照を作ったマシンから動かせない」は、この 2 台の間では**成り立たない**。

## 7.5 実装上の罠（走る前の監査で潰したもの・6 レンズ 31 件から）

既 commit のモジュールを import して腕だけ足す設計には、**黙って壊れる経路が 8 つ**ある。すべて新モジュール側で閉じる。

1. **`H.DATA_DIR` が別マシンの絶対パスに固定されている。** `src/boundary_groups_0908.py:10` が `/home/issan/Projects/claude/proj_004_drift/data/mnist` を代入する（import 連鎖に必ず入る）。lab には無い。新モジュールが **import 後に自分で解決し直す**。committed ファイルは編集しない。データの sha256 は committed provenance と 4 本とも一致することを確認済み。
2. **`clamp_horizon_0910` の `TASKS`・`CLAMP_FROM`・`OUT` は def 時のデフォルト引数**（L169・L286・L298）。モジュール変数を差し替えても**効かない**。逆に `REF`・`CAP`・`G1_LAST` の差し替えは**プロセス全体に即座に効く**。この非対称は逆に覚えやすい。→ 新モジュールは自前の `run` を持つ。
3. **`C.restore` はモジュール大域の `make_act` を呼ぶ**（`width_sink_clamp_0909.py:310`）。GELU/SILU は `C.make_act` が `KeyError`。→ 自前の `restore` を書く（12 行）。**`R` は要らない**（`C.make_act('R')` は既に `H.ARMS['R']` を返す）。
4. **`_controls` は run() の `clamp_from` を無視してモジュール大域の `CLAMP_FROM` を読む**（L218・L223・L256・L277・L279）。→ 自前の `_controls` は引数で受ける。
5. **`g1_check` の件数ガードは 4 クランプの width_sink 参照に決め打ち**（`3*(...)` 行・`15*(...)` 配列）。ref 腕のみの錨では 80 行・600 配列なので、そのまま使うと**軌道が完全一致していても落ちる**。→ 自前の `g1_check`。
6. **`src/long_horizon_0910.py` は import しただけで `C.MEAS = [625]` を代入し、`C.measure` を try/finally 無しで monkeypatch する**（L7・L31–33）。例外が出ると `C.measure` が包まれたまま残る。→ **import しない。**必要な 40 行は写す。
7. **NaN に対する番人が無い。** 発散すると `(cn0 > CAP*c).any()` が False になって「拘束されなかったステップ」として数えられ、1000 秒後に「checks never recorded」で落ちる。→ 毎タスク終端で `acc`・`rowmean`・`cnorm` の有限性を検査する。
8. **`TIME_CAP` の assert は CSV/npz/provenance を書いた後**（L341）。超過しても**完全に見える出力ディレクトリが残る**。→ 終了コードで判定する。ファイルの存在で判定しない。

加えて **報告モジュールは腕を決め打ちしている**（`clamp_horizon_report_0910.py:14` `ARMS=['LR','ELU1']`・`long_horizon_report_0910.py:7` `ARMS=['LR','ELU1','SNA']`）うえ、**既存の結果ディレクトリに `verdict.csv` を書く**。新走は**新しい出力ディレクトリと新しい報告モジュール**を使う（既存 run の出力を上書きしない）。

## 8. 限界（結果前に明記）

- 3 seed・pmnist・第 1 層のみ。保持（後ろ向き）は測らない。
- **B の G1 は `ref` 腕のみ**（R には committed のクランプ参照が無い）。
- **A・C・D の GELU/SiLU に独立の committed 参照は無い。**C・D の錨は A 自身の産物である。
- 谷越え型の z_c は固定（GELU/SiLU はパラメータを振らない）。用量梯子は無い。
- 箱 A との同一性（V9 §11 の X1）は測らない。
- `dead_hard` / `sat` は谷越え型では「死」と「反転」を混ぜる。§2.1 の 4 量で分けるが、**既発表の ELU/leaky の値と直接は比べられない**。
- **C（`WIDTH_WITHOUT_DEPTH`）は構造的に落ちうる**（§4.B/C の備え）。落ちても A で決まる。
- **B の G0 は際どい。** R の t96–100 の損失は 1.79/2.00/2.12 pt で、G0 の下限 2.0 pt に接している。t301–400 まで伸ばせば leaky が 2.2 → 4.4 pt に増えたのと同様に増える見込みだが、**増えなければ R の全ラベルが `NOT_TESTABLE_NO_LOSS` になる**。そのときは「ReLU は箱 B では 400 タスクでも大きく失わない」こと自体が V9 §11 の穴 1 への答えになる（死が LoP を作らない、の最も強い形）。
- 本 spec の run id は 4 つの副走をまとめた登録である。**副走ごとに独立に判定し、片方が落ちても他方は生きる。**
