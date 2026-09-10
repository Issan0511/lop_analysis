# 長い地平線で沈下と幅を分ける spec 0910
状態: 検証中 / 更新: 2026-09-10
親: [[長い地平線でのleakyとELUの分岐_結果_0910]] / [[幅か沈下か_クランプ介入_結果_0909]] / [[幅の規制は可塑性を買うか_0910]]
run id: `clamp_horizon_0910`

## 0. 何を決めるか

Issa の指摘（2026-09-10）: 「Snake に比べて leaky も ELU も同程度に悪いのは、沈下が原因と言えるのではないか」。
`long_horizon_0910` の t400 で、累積精度損失は leaky −4.3 / ELU −3.1 / Snake −0.9 pt。この順序は **‖W̃‖（15.2 / 13.7 / 9.0）でも z̄（−8.3 / −8.3 / −1.6）でも同じように付く**。**n=3 の腕間相関では沈下と幅を分けられない。**

分けられるのは介入である。`width_sink_clamp_0909`（t21–100）では:

| 腕 | 状態 | 精度差 vs ref |
|---|---|---|
| leaky `dclamp` | 沈下を止め、幅は 75–97% 伸びる | +0.4 pt |
| leaky `wclamp` | 幅を止める（結果として沈下もほぼ止まる） | +1.3 |
| ELU `dclamp` | 沈下を止め、幅は 29–66% 伸びる | −0.2 |
| **ELU `wclamp`** | **幅を止めるが沈下は続く（−6.9、ref −6.5 より深い）** | **+1.1** |

ELU の 2 腕が「浅いのに戻らない／深いのに戻る」の両方を出しているが、**これは t61–100 の窓で、損失が出るのは t301–400 である**。**同じ介入を t400 まで伸ばして、長い地平線でも同じ乖離が成り立つかを決める。**

## 1. 走

開始状態・分岐は `width_sink_clamp_0909` と同一（init から t1–20 を 1 回、そこから bit 一致の t20 状態で分岐、毎更新クランプ）。**継続を t21–400 に伸ばす。**

| 腕 | 固定するもの | 実装 |
|---|---|---|
| `ref` | — | 無介入 |
| `dclamp` | m̄（沈下の実体） | 毎更新後、全行に同じ定数を足して m̄ = m̄(t20) |
| `wclamp` | ‖W̃ᵢ‖（中心化ノルム）を t20 の値 c_i に固定 | W̃ᵢ ← W̃ᵢ·c_i/‖W̃ᵢ‖ |
| `wcap2` | ‖W̃ᵢ‖ の**上限**を 2c_i（超えたときだけ射影） | ‖W̃ᵢ‖ > 2c_i のとき W̃ᵢ ← W̃ᵢ·2c_i/‖W̃ᵢ‖ |

`wcap2` は「t20 の水準に留め置く」と「成長を止める」を分けるための用量点。‖W̃‖ ∝ √t なので 2c_i には t ≈ 80 で達し、そこから拘束が効く。

腕: **LR・ELU1 × 4 クランプ × seed 0–2 = 24 継続。** Snake は沈下も幅も小さく `dclamp` がほぼ no-op なので入れない（対照は `long_horizon_0910` の ref を引く）。`mclamp` は t21–100 で精度に効かなかったので落とす。
測定はタスク終端のみ（`long_horizon_0910` と同じ）。1 継続 ≈ 10 分、2 プロセスで **合計 ≈ 30 分**。

## 2. 判定（事前登録）

窓は固定: **late = t301–400**、**base = t16–20**（クランプ開始時の水準・その seed の共有前置き）。精度は窓内タスク終端の中央値。

各 seed について
$$L_{\rm ref}=\mathrm{acc_{base}}-\mathrm{acc_{ref}(late)},\qquad
\rho_C=\frac{\mathrm{acc}_C(\rm late)-\mathrm{acc_{ref}(late)}}{L_{\rm ref}}$$

$\rho_C$ は「その腕が ref の累積損失の何割を取り戻したか」。$\rho=1$ で全部、$\rho=0$ で無効果。

**可検定性ゲート G0**: $L_{\rm ref}\ge 2.0$ pt。下回る seed はその腕の比を `NOT_TESTABLE_NO_LOSS` とする（`long_horizon_0910` の実測は leaky 4.4・ELU 4.0 pt なので通る見込み）。

**主判定**（腕ごとに、可検定な 3 seed が 3/3 一致したときだけ）

| 判定 | 規則 |
|---|---|
| **A 深さ** | ρ_dclamp ≥ 0.5 → `DEPTH_REMOVES_LOSS`／≤ 0.2 → `DEPTH_NO_EFFECT`／他 `DEPTH_PARTIAL` |
| **B 幅** | ρ_wclamp ≥ 0.5 → `WIDTH_REMOVES_LOSS`／≤ 0.2 → `WIDTH_NO_EFFECT`／他 `WIDTH_PARTIAL` |
| **C 乖離（ELU）** | ρ_wclamp ≥ 0.5 **かつ** z̄_wclamp(late) ≤ z̄_ref(late) + 1.0（沈下が止まっていない）→ `WIDTH_WITHOUT_DEPTH` |
| **D 乖離（両腕）** | ρ_dclamp ≥ 0.5 **かつ** ‖W̃‖_dclamp(late)/‖W̃‖_ref(late) ≥ 0.75（幅は伸びている）→ `DEPTH_WITHOUT_WIDTH` |
| **E 用量** | 3 seed とも −0.1 ≤ ρ_wcap2 ≤ ρ_wclamp + 0.1 → `WIDTH_DOSE_MONOTONE`／3 seed とも ρ_wcap2 > ρ_wclamp + 0.1 → `CAP_BEATS_CLAMP`／他 `DOSE_PARTIAL` |

**C と D は排他ではない。両方立てば「どちらを止めても戻る」＝ 分離できていない**（そのときは §5 の副測定で読む）。

**進捗ゲート**
- **G1（再現）**: `ref`・`dclamp`・`wclamp` の t21–100 のタスク終端が `width_sink_clamp_0909` の committed 出力（`zbar_inv`・`sigma_inv`・`star_sd`・`acc`・per-unit npz）と **maxabs ≤ 1e−10**。1 つでも外れたら全腕を無効として停止する。
- **G2（クランプの実効性）**: `width_sink_clamp_0909` と同じ検査（毎ステップ現状態・float32 の証明可能上界で床を張る）。`wcap2` は「‖W̃ᵢ‖ ≤ 2c_i·(1+1e−6) が全ステップで成立」かつ「拘束が効いていないステップでは W1 が差 0」。
- **G3（学習が壊れていない）**: late 窓のタスク内 CE 改善（更新 20 → 625）が 90% 以上のタスクで正。**精度でゲートしない**（精度が判定量なので循環する）。外れた腕seed は `NOT_TESTABLE_LEARNING_BROKEN`。
- **G4（`dclamp` が深さを保っている）**: |z̄_dclamp(late) − z̄_dclamp(t20)| < 0.25 × |z̄_ref(late) − z̄_ref(t20)|。
- **G5（`wclamp` が沈下を止めていない＝C の前提）**: ELU の `wclamp` で z̄(late) ≤ z̄_ref(late) + 1.0。満たさなければ C は `NOT_TESTABLE_DEPTH_ALSO_STOPPED`。

## 3. 事前予測（記名・走る前）

- **Issa（沈下側・Claude による読み替え。本人の確認は結果を読む前に取る）**: **A `DEPTH_REMOVES_LOSS`**。沈下を止めれば損失は消える。ELU の C `WIDTH_WITHOUT_DEPTH` は**立たない**（t61–100 で見えた乖離は短い窓の産物で、t400 では沈下の効果が勝つ）。
- **Claude（幅側）**: **A `DEPTH_NO_EFFECT`**（ρ_dclamp ≤ 0.2）、**B `WIDTH_REMOVES_LOSS`**（ρ_wclamp 0.5–0.8）、**C `WIDTH_WITHOUT_DEPTH` が立つ**、D は立たない、E `WIDTH_DOSE_MONOTONE`（ρ_wcap2 ≈ 0.3）。
- **分岐点は A と C。** A が `REMOVES` なら Issa、`NO_EFFECT` なら Claude。C が立てば幅側が決定的。
- **両者外れの形**: A・B とも `REMOVES`（どちらを止めても戻る＝介入では分離できない）か、A・B とも `NO_EFFECT`（損失の原因は第 1 層の外）。
- 通算: Issa 9/11 前後、Claude 3.5/11（`long_horizon_0910` で Claude は 4/4 外し、Issa は B が惜しい）。**重みは Issa 側に置く。**

## 4. 検算と変異対照

`width_sink_clamp_0909` の検査群をそのまま使う（毎ステップの目標検査・不変量検査・変異対照 10 本・float32 の証明可能上界）。追加:
- `wcap2` の上限検査（G2）と、その変異対照（上限を 4c_i にすると t400 で ‖W̃ᵢ‖ が 2c_i を超える）。
- G1 の変異対照: 初期値 W1[0,0] に +1e−3 で t21–100 の再現が 1e−4 を超えて外れる。
- 測定の非侵襲: `ref` は測定込みで G1 を通す。

## 5. 副測定（ラベル無し・結果前に列挙）

late 窓での z̄・σ_inv・star_sd・‖W̃ᵢ‖・p⁺・‖W2col‖・dead_hard・dead_soft・飽和対・CE、および各腕の精度時系列の late 傾き。
**C と D が両方立った場合の読み方**: `wclamp` と `dclamp` が共に戻すなら、共通の下流量（‖W2col‖ か実効ランク）を見る。どちらの腕でも同じ量が ref から離れていれば、それが真の lever の候補になる。

## 6. 限界（結果前に明記）

- `wclamp` は「成長を止める」だけでなく「t20 の水準に留め置く」。`wcap2` で 1 点だけ分けるが、用量は 2 点（1×・2×）しかない。
- t20 の一点からの介入。t1 から掛けた場合は未検定。
- 保持（後ろ向き）は測らない。現タスク精度のみ。
- 第 1 層のみ。W2・W3 は自由に伸びる（副測定で記録）。
- pmnist・784→100→100→10・3 seed。
