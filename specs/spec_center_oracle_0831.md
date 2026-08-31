# オラクル中心化（境界降下の B1/B2 分離）

親: [[centered死因分解_posthoc_spec_0831]] §6 / 関連: [[容量不足レジームの機構_0830]] §11・[[中心主張v4.2草案_0830]] 柱1・[[論点マップ]]（★可識別性の交絡）
状態: **spec（事前登録・未実行）** / 作成 2026-08-31 / run id: `center_oracle_0831`

> **本 spec は [[centered死因分解_posthoc_spec_0831]] の E3 が `BOUNDARY_CARRIES_DESCENT` または `BOUNDARY_DOMINANT` を出したときにだけ回す。** 出なければ問い自体が消えるので起票を取り下げる。

---

## 0. 問い

centered 腕で $\beta$ の降下がタスク境界窓に局在するとして、その原因は

- **B1 EMA の遅れ**: `center_alpha = 0.01` の時定数 $1/\alpha = 100$ step のあいだ `running_mean` が旧タスクの flip を引き続けるので、境界直後だけ µ が実質的に復活する
- **B2 切替ショック**: 教師が変わって残差 $\delta$ が大きい窓であり、$b$ への自己税は入力分布によらず片側符号なので、$\lvert\delta\rvert$ が大きいほど $b$ が深く沈む

のどちらか。**遅れを厳密に 0 にした腕を 1 本足せば分離できる。**

## 1. 介入

`src/mlp2_phase1.py` の `_layer_stats` には既に `mean_source ∈ {"ema", "support"}` の分岐があり、`"support"` は当該タスクの厳密な支持平均（$\alpha\to0$ の理想化）を引く。**現在は S-taut の検査にしか使っていない。これを学習の前向きにも通す。**

- 新腕 `L1w100_Aexact`: 隠れ1層×100、第1層に**支持平均をオラクルで**引く。タスク境界で即座に切り替わるので遅れ 0
- 支持平均はタスクが決まった時点で閉形式（$\mu=[\text{flip}\,\|\,0.5\cdot\mathbf 1_5]$）。**サンプルから推定しない**（推定にすると遅れが復活する）
- 乱数を消費しない（`consumes_rng: false` を維持）。よって既存 `L1w100_A1` と init・教師・入力列・flip 軌道が bit 一致する

> **step 1 以降は軌道が分岐する。**「同一個体の追跡」ではなく、init・教師・入力実現までのペアリング（[[現在地]] 運用メモ）。

## 2. 腕

| 腕 | 中心化 | 遅れ | 走 |
|---|---|---|---|
| `L1w100_A1` | EMA $\alpha=0.01$ | 100 step | **既存ログを参照**（`mlp2_phase1_0829`）。再走しない |
| `L1w100_Aexact` | 支持平均オラクル | 0 | **新規** |

### 2.1 追補（任意・用量反応）

主判定は上の 2 腕で閉じる。$\alpha$ を振れば B1 に**用量反応**が付く。回すなら:

| 腕 | $\alpha$ | 遅れ $1/\alpha$ |
|---|---:|---:|
| `L1w100_A1_a100` | 0.1 | 10 step |
| `L1w100_A1_a001` | 0.001 | 1000 step |

$\alpha$ は config のみで実装不要。**追補を回すかは主判定と独立に判断してよいが、回すなら本 spec と同時に凍結する**（後から足すと事後になる）。$\alpha=1$ は batch=1 では直前 1 サンプル平均になり分散が別物なので**採らない**。

条件は他すべて `mlp2_phase1_0829` を継承: condA・w100・$T=10^4$・batch=1・lr=0.01・5M・seed 0–9。

## 3. Endpoint と判定基準（**実行前に凍結**）

$\Delta\beta^{\rm bnd}$・$\Delta\beta^{\rm int}$ の定義は [[centered死因分解_posthoc_spec_0831]] §2 E3 をそのまま継承する。不確かさも同じ（seed クラスタ bootstrap・$B=10{,}000$・percentile・bootstrap seed `20260829` 継承）。

### P1 主判定 — 遅れを 0 にすると境界降下は消えるか

$$R := \frac{\lvert \Delta\beta^{\rm bnd}(\text{Aexact})\rvert}{\lvert \Delta\beta^{\rm bnd}(\text{A1})\rvert}$$

seed 別中央値の比ではなく、**seed ごとに比を取ってから中央値**を出す（腕は同 seed でペア）。

- 95% CI の**上限 < 0.2** → **`EMA_LAG_IS_THE_CAUSE`**（B1）
- 95% CI の**下限 > 0.8** → **`SWITCH_SHOCK_IS_THE_CAUSE`**（B2）
- CI が $[0.2, 0.8]$ にかかる → **`BOTH_CONTRIBUTE`**
- Aexact の $\Delta\beta^{\rm bnd}$ 自体の CI が 0 を含む → **`BOUNDARY_DESCENT_ELIMINATED`**（B1 の最強形）

### P2 副判定 — 水準に効くか

- 5M 時点の layer 1 `strict_dead_frac`、および直近 100 タスク連続死率（[[centered死因分解_posthoc_spec_0831]] E6 と同定義）
- 判定: Aexact − A1 の paired 差の CI 上限 < 0 なら **`ORACLE_REDUCES_DEATH`**、CI が 0 を含めば **`NO_LEVEL_EFFECT`**
- **P1 と P2 は独立に読む。** 境界降下が消えても水準が動かない可能性がある（内部の回復が減って相殺する）。その場合は「機構は同定したが水準には効かない」と書く

### P3 追補（$\alpha$ スイープを回した場合のみ）

$\Delta\beta^{\rm bnd}$ を $\log(1/\alpha)$ に回帰。傾きの CI が 0 を跨がず、Aexact（$1/\alpha=0$）が外挿線上にあれば **`LAG_DOSE_RESPONSE`**。

## 4. サニティ

| id | 内容 | 基準 |
|---|---|---|
| **S0'** | Aexact の step 0 の全記録量が `L1w100_A1` の step 0 と bit 一致（init・教師・入力実現の共有） | 完全一致 |
| **S-taut** | Aexact では第1層の $M \equiv 0$ が**厳密**（支持平均をそのまま引くため） | $\max\lvert M\rvert = 0$。**恒真なので判定に使わない**（[[運用ルール]] §5 教訓②・⑩） |
| **S1** | $\hat p=0 \Rightarrow \beta\le-1$、$\beta\le-\sqrt5 \Rightarrow \hat p=0$ | 違反 0 件 |
| **S2** | 32 パターン厳密恒等式（`mlp2_phase1_0829` の S1/S2 を継承） | PASS |
| **S3** | `OMP_NUM_THREADS=1` | PASS |
| **S7** | 数値発散検出器（`L2_A2` の前例）。発散したら救済せず `NUMERIC_DIVERGENCE` で停止 | 実装の存在 |
| **S8** | flip 変化が `step % 10000 == 0` の遷移でのみ起き、seed あたり 499 回 | 完全一致 |

## 5. 交絡（**必ず summary.md に書く**）

**オラクル中心化は µ を消すと同時に、タスク可識別性を完全に消す。** condA では $\mu\approx[\text{flip}\,\|\,0.5\cdot\mathbf 1]$ なので µ ＝ タスク識別子であり、支持平均を厳密に引くと学習器の入力は**毎タスク同じ 32 パターン**になる（[[論点マップ]] の ★交絡）。

EMA 版は境界直後の $\sim1/\alpha$ step だけ可識別性が残るので、**Aexact − A1 の差には「遅れの差」と「可識別性の差」が同居する**。したがって本走の帰結は

- 書いてよい: 「境界降下は EMA の遅れ窓に起因する／しない」
- 書いてはいけない: 「µ の効果を測った」「centering を改善すれば LoP が防げる」

$\alpha$ スイープ（§2.1）はこの交絡を**弱めない**（どの $\alpha$ でも定常部では可識別性が消えている）ため、交絡の分離には別設計（flip を別経路で明示入力する腕）が要る。**本 spec の対象外。**

## 6. スコープ・引用上の注意

- condA・w100・隠れ1層・$T=10^4$・batch=1・lr=0.01・5M・seed 0–9
- **深さ 2 へ外挿しない。** 第2層の入力は第1層の活性で、支持平均の閉形式が使えない
- `L1w100_A1` 側は再走ではなく既存ログの参照。commit 済み成果物である旨を provenance に記録する
- `strict_dead` は当該タスク支持域のラベル（[[centered死因分解_posthoc_spec_0831]] E6 の留保を継承）
- **0/10 は「観測しなかった」の強さ**。「起きない」と書かない

## 7. コスト

新規は 1 腕 × 10 seed × 5M（追補を採ると 3 腕）。`mlp2_phase1_0829` は 4 腕 5M を完走しているので、単純比で **その 1/4**。実時間の見積もりは `mlp2_phase1_0829/provenance.json` の実測から起こすこと。
