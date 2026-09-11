# 勾配の揃いを直接動かす spec 0911

状態: **未実行（事前登録）** / 更新: 2026-09-11 / 起草: Claude（Issa の指示「勾配の揃い方を直接動かす腕を図って」）
親: [[個体別3分表_結果_0911]] §2b（ゲート → 生勾配 → 歩幅の 1 本道） / [[ゲート被覆か非線形性か_結果_0911]] §3b（mob はレバーではない） / [[幅の規制は可塑性を買うか_0910]]（主軸 B） / [[タスク内経路の持続_結果_0910]] §「次は勾配ノイズへの介入（バッチサイズ）」
run id: `grad_coherence_0911`
実装: lop_analysis `collective_kick_0908` / `specs/spec_grad_coherence_0911.md` / `src/grad_coherence_0911.py`・`src/grad_coherence_report_0911.py`

## 0. 何を決めるか

0911 未明の 4 走で、ゲートは**歩幅**を通して効くと分かった（[[個体別3分表_結果_0911]]）。ただし媒介変数 graw² は「勾配の大きさ」と「勾配の断続性」を分けていない。ユニット間ではこの 2 つが一緒に動くので、**観測では分離できない**。

同時に、mob（入力平均ゲート）は腕の間の目印にはなるが、腕の中で状態として動かしても効かないと分かった（[[ゲート被覆か非線形性か_結果_0911]] §3b）。**残る候補は「勾配がどれだけ揃っているか」そのもの。** これを介入で動かす。

**装置**: Adam に入る前の第 1 層の勾配に、**不偏のガウスノイズ**を注ぐ。行 $i$ ごとに
$$g'_i = g_i + c\,s_i\,\xi_i,\qquad s_i=\lVert g_i\rVert/\sqrt{784},\ \ \xi_i\sim\mathcal N(0,I_{784})$$
期待勾配は厳密に不変（$\mathbb E[g']=g$）、更新回数・データ・バッチ順・活性化・他層も不変。Adam の側では $\hat v\to \hat v(1+c^2)$ なので、**正規化後の歩幅が $1/\sqrt{1+c^2}$ に縮み、向きが乱れる。** 注入はユニットごとに**自分の勾配に比例**させるので、低ゲートのユニットだけを狙い撃ちしない（相対 SNR の劣化は全ユニット共通）。

**交絡**: ノイズは「揃い」と同時に「実効歩幅」を縮める。これを分けるため、**ノイズ無しで lr を $1/\sqrt{1+c^2}$ 倍した腕**を置く。歩幅は同じで向きは乱れていないので、この 2 腕の差が「揃いそのもの」の効果である。

**同時に V9 の幹を試す**: ノイズは 1 歩を短くし向きを乱すので、堆積 $\lVert\Delta\widetilde W_{\rm task}\rVert^2=\rho\sum_s\lVert d\widetilde W_s\rVert^2$ の**両方の因子を下げる**。つまり $\lVert\widetilde W\rVert$ は**遅く**育つはずである。もし**幅が縮むのに精度が落ちる**なら、「LoP の幹は $\lVert\widetilde W_1\rVert$」（V9）は必要条件としては残るが**十分ではない**ことになる。これは §3 の E3 で判定する。

## 1. 走

箱 B（pmnist 784→100→100→10・Adam lr=1e−3・batch 16・625 更新/タスク・CPU 1 スレッド）。init・置換列・抽出・バッチ順は seed で決まり腕に依らない。**ノイズは専用の乱数系列 `stream('gradnoise', seed)` から引き、c=0 のときは 1 回も引かない**（＝ c=0 は committed 軌道と bit 一致）。t1–120、seed 0–2。

| 腕 | 活性化 | 介入 | 役割 |
|---|---|---|---|
| `N0` | leaky a=0.1 | 無し | 参照・**G1 の錨**（`elu_growth_0909` の LR を bit 再現） |
| `N05` | leaky | W1 勾配へ c=0.5 | 用量 |
| `N1` | leaky | c=1 | 用量（主） |
| `N2` | leaky | c=2 | 用量 |
| `LRh` | leaky | ノイズ無し・**lr ×1/√2** | `N1` の歩幅対照 |
| `LRq` | leaky | ノイズ無し・**lr ×1/√5** | `N2` の歩幅対照 |
| `N1L23` | leaky | c=1 を **W2・W3 だけ**へ | 層の特異性 |
| `N0w` | leaky | 無し ＋ **wclamp**（t21 から ‖W̃ᵢ‖ を自分の t20 値へ射影） | E4 の対照 |
| `N1w` | leaky | c=1 ＋ wclamp | **E4（決定的）** |
| `SN0` | 適応 α Snake | 無し | 一般性の錨 |
| `SN1` | 適応 α Snake | c=1 | 一般性 |

11 腕 × 3 seed = **33 走**。1 走 ≈ 2 分、8 並列で 10 分程度。

**測定**（毎タスク終端・[[ゲート被覆か非線形性か_結果_0911]] と同じプローブ 512 枚・現置換）: test 精度、ゲート（ḡᵢ・Cov・mob）、`zbar_i`・`sd_i`・`cnorm_i`（幅）、および late 窓 t61–120 では毎更新の **κ2**（Adam 正規化歩幅 $(\hat m/(\sqrt{\hat v}+\varepsilon))^2$ の W1 平均）・**S²**（実現した中心化歩幅の予算）・**ρ = D²/S²**（タスク内の経路持続）・graw²（生勾配。注入**前**）。タスク内 CE 改善（更新 20 → 625）。

## 2. 量（登録）

窓: **base = t16–20**、**late = t101–120**、seed は中央値。

- **L = acc(base) − acc(late)** [pt]（可塑性の目減り）
- **level = acc(late)**、**acc1 = acc(t1–5)**（学習能力そのもの）
- **slope** = late 窓の精度の傾き [pt/100task]
- **κ2**・**S²**・**ρ**・**N = ‖W̃ᵢ‖ のユニット平均**（すべて late 窓）
- 腕 X の**損傷** $D_X = L(X) - L(\texttt{N0})$

## 3. 判定（事前登録）

**E0 操作チェック**: κ2 が c に単調減、かつ κ2(N2)/κ2(N0) ≤ 0.6（予測は 1/√5 = 0.45）。3/3 seed で成立 → `MANIPULATION_OK`。不成立なら E1–E5 は `NOT_TESTABLE_NO_COHERENCE_CHANGE`。

**E1 用量反応**（3/3 seed）: L(N0) < L(N05) < L(N1) < L(N2) → **`NOISE_COSTS_PLASTICITY`**／逆順 → `NOISE_HELPS`／|L(N2) − L(N0)| < 0.5 pt → `NOISE_NEUTRAL`／他 → `DOSE_PARTIAL`。

**E2 可塑性か最適化か**: ノイズは学習そのものも遅くする。`N2` と `N0` について
- Δacc1 = acc1(N2) − acc1(N0)（学習能力の差）と Δslope = slope(N2) − slope(N0)（劣化の速さの差）を出す。
- Δslope ≤ −0.3 pt/100task（3/3） → **`PLASTICITY_SPECIFIC`**（ノイズは劣化を速める）
- |Δslope| < 0.3 かつ Δacc1 ≤ −1.0 pt → **`OPTIMIZATION_ONLY`**（水準が下がっただけ）
- 他 → `E2_PARTIAL`

**E3 幅を通るか**（V9 の幹の検定）: r = N(N2)/N(N0)（late）。
- r ≥ 1.15（3/3） → `NOISE_ROUTES_VIA_WIDTH`
- r ≤ 0.95（3/3） → **`NOISE_SHRINKS_WIDTH`**。このとき **E1 が `NOISE_COSTS_PLASTICITY` なら、「幅が小さいほど良い」は腕の中でも破れる**（`WIDTH_NOT_SUFFICIENT` を併記）
- 他 → `WIDTH_UNCHANGED`

**E4 決定的 — wclamp はノイズ損傷を消すか**: $D_{N1}=L(\texttt{N1})-L(\texttt{N0})$、$D_{N1w}=L(\texttt{N1w})-L(\texttt{N0w})$。$D_{N1}\ge 0.5$ pt のときだけ可検定（そうでなければ `NOT_TESTABLE_NO_DAMAGE`）。
- $D_{N1w} \le 0.3\,D_{N1}$ → **`NOISE_DAMAGE_IS_WIDTH`**（幅が媒介・揃いは随伴）
- $D_{N1w} \ge 0.7\,D_{N1}$ → **`NOISE_DAMAGE_SURVIVES_CLAMP`**（揃いは幅と独立な第 2 チャネル）
- 他 → `E4_PARTIAL`

**E5 揃いか歩幅か**（本 spec の主眼）: 歩幅を揃えた対照との差 $\Delta_1=L(\texttt{N1})-L(\texttt{LRh})$、$\Delta_2=L(\texttt{N2})-L(\texttt{LRq})$。まず歩幅が実際に揃っているかを確認（κ2·lr² の比が 1±0.15 に入る。外れたら `NOT_TESTABLE_STEP_NOT_MATCHED`）。
- $\Delta_1,\Delta_2 \ge +0.5$ pt（3/3） → **`COHERENCE_COSTS_BEYOND_STEP`**（揃いそのものが効く）
- $|\Delta| < 0.5$（3/3） → **`STEP_SIZE_EXPLAINS_ALL`**（ノイズの害は歩幅が縮んだことだけ）
- $\Delta \le -0.5$ → `NOISE_BEATS_SMALL_LR`

**E6 層の特異性**: L(N1L23) − L(N0) ≤ 0.3·D_{N1}（3/3） → `FIRST_LAYER_SPECIFIC`／≥ 0.7·D_{N1} → `NOT_LAYER_SPECIFIC`／他 `E6_PARTIAL`。

**E7 一般性（Snake）**: D_SN = L(SN1) − L(SN0)。sign(D_SN) = sign(D_N1) かつ |D_SN| ≥ 0.3·|D_N1|（3/3） → `GENERALISES_TO_SNAKE`／|D_SN| ≤ 0.3·|D_N1| → `LEAKY_SPECIFIC`／他 `E7_PARTIAL`。

**可検定性**: 発散した腕seed は `DIVERGED` として落とす。タスク内 CE 改善が late 窓の 90% 未満の腕は `LEARNING_BROKEN`（E1–E7 から除外）。

## 4. 進捗ゲートと変異対照

- **G1（再現）**: `N0` の per-unit 3 配列 × t1–120 が `elu_growth_0909` の LR と maxabs ≤ 1e−10（`cnorm_i` は bit 一致 0.0）。`SN0` も同様（SNA）。**c=0 の腕でノイズ生成器を一度も引かないことの証拠**でもある。件数ガード付き。
- **G1 変異対照**: 初期値 W1[0,0] += 1e−3 で t1 の再現が 1e−8 を超えて外れる。
- **G2（ノイズが不偏）**: 走の前に、固定した勾配行列に対し注入を 2000 回繰り返し、(i) 平均が元の勾配と相対 ≤ 3/√2000 ≈ 0.067 以内、(ii) 分散が $c^2s_i^2$ と相対 10% 以内。**変異対照**: 平均に $+0.1\,s_i$ の偏りを足した版が (i) を破る。
- **G3（ノイズが他層・他系列を汚さない）**: `N1L23` で W1 の勾配が注入前後で厳密に一致（0.0）、`N1` で W2・W3 が厳密に一致（0.0）。`gradnoise` 系列を 1000 回引いた後でも `perm`/`data`/`batch` の状態が不変（厳密比較）。
- **G4（歩幅対照の整合）**: `LRh`・`LRq` の lr が 1e−3/√2・1e−3/√5 に一致（相対 1e−12）。E5 の前提として κ2·lr² の比を記録。
- **G5（wclamp の実効性）**: committed の `c3_cnorm_rel` ≤ 1e−6・`c3_m_absdiff` ≤ float32 の証明可能床・`clamp_tail_absdiff` = 0 と、skip / rowscale / tail の変異対照が各 100 倍で発火。
- **G6（帳簿）**: 1 ≤ ρ ≤ 625(1+1e−9)、Σ_s dW̃_s = W̃(終端) − W̃(開始) が 1e−10。
- 時間上限 900 s / 走。

## 5. 事前予測（記名・走る前）

**Claude**（今回は選言を書かない。各判定に 1 点だけ張る）:

| 判定 | 予測 | 根拠 |
|---|---|---|
| E0 | `MANIPULATION_OK`・κ2(N2)/κ2(N0) ≈ **0.45** | $\hat v\to\hat v(1+c^2)$ から機械的 |
| E1 | **`NOISE_COSTS_PLASTICITY`**・L(N2) − L(N0) ≈ **+1.5 pt** | 揃い → 歩幅 → 追随の鎖 |
| E2 | **`PLASTICITY_SPECIFIC`**（Δslope ≤ −0.3） | 鎖が正しいなら劣化が速まる |
| E3 | **`NOISE_SHRINKS_WIDTH`**（r ≈ 0.85） | 堆積 = ρ·ΣS² で両因子が下がる |
| E4 | **`NOISE_DAMAGE_SURVIVES_CLAMP`**（$D_{N1w}\ge0.7D_{N1}$） | 揃いは幅と独立なチャネル |
| E5 | **`COHERENCE_COSTS_BEYOND_STEP`**（Δ ≈ +0.8 pt） | 向きの乱れは歩幅の縮小以上に効く |
| E6 | **`FIRST_LAYER_SPECIFIC`** | V9 の幹は第 1 層 |
| E7 | **`GENERALISES_TO_SNAKE`** | 機構が活性化に依らないなら |

**E3 と E1 が同時に立つと V9 の幹が弱まる**（幅が縮むのに損失が増える＝ `WIDTH_NOT_SUFFICIENT`）。**E5 が `STEP_SIZE_EXPLAINS_ALL` なら、「揃い」は独立な量ではなく実効 lr の言い換えであり、0911 未明の読みを書き換える。** どちらも自分の主張が負ける形で、分岐点として登録する。

**Issa**: 未記入。結果を読む前に書く欄:
> E1: ／ E2: ／ E3: ／ E4: ／ E5: ／ E7:

## 6. 副測定（ラベル無し・結果前に列挙）

late 窓の Cov・mob・z̄・σ・p⁺・dead・‖W2col‖、ρ と S² の用量依存、graw²（注入前）の用量依存、精度時系列、タスク内 CE 改善の曲線、per-unit の κ2 とゲートの相関が用量で変わるか。

## 7. 限界（結果前に明記）

- ノイズは等方（行ごとに同じ分散）。実際の勾配ノイズは異方でありデータ由来。**「バッチサイズを下げたのと同じ」ではない**（バッチはノイズと同時に更新回数あたりの情報量も変える）。
- 用量 3 点、t120、3 seed、leaky が主で Snake が 1 点。
- 歩幅対照 `LRh`/`LRq` は lr を定数倍するだけで、Adam の $\hat v$ の時定数は変えない。κ2·lr² が 1±0.15 に入らなければ E5 は取り下げる。
- `N1w` の wclamp はその腕自身の t20 状態を基準にする（`N0w` と基準が違う）。E4 は差の差なので成立するが、基準の違いは副測定に記録する。
- 第 1 層のみ（`N1L23` を除く）。保持（後ろ向き）は測らない。
