# relu_gelu_silu_rl_0914 —— ReLU・GELU・SiLU は RL-MNIST で死ぬか

親: `spec_layer_chimera_rl_0914.md`（Q1 `FLOOR_IN_DEEP_LAYER`: ELU を殺すのは第 2 層に φ′ の床が無いことだけ）
関連: 箱 B の符号梯子 0912（GELU/SiLU の谷の反転が PM 系の損失 2.4–3.1 pt の原因）

## 0. 問い

`layer_chimera_rl_0914` で、RL-MNIST の崩壊は「第 2 層の φ′ に床が無いこと」で決まった。
ReLU（負側 φ′ ≡ 0）・GELU・SiLU（φ′ → 0、しかも谷の向こうで符号反転）は**どれも床を持たない**。
床の説が正しければ 3 つとも死ぬ。Issa の事前予想も「全部死ぬ」。

## 1. 箱（`layer_chimera_rl_0914` / `elu_environment_0913` と同一）

1200 枚固定・784-100-100-10・Adam(lr 1e−3, β 0.9/0.999, ε 1e−8)・batch 16・6000 更新/タスク（80 epoch）・
50 タスク・seed 0–2・GPU float32・TF32 無し・決定論的。環境は RL（毎タスク新しい一様乱数ラベル）と
PM（毎タスク画素置換・正ラベル）。**両層に同じ活性化**。

| 腕 | φ(z) | φ′(z), z<0 |
|---|---|---|
| `LR` | leaky 0.1 | 0.1（床あり・既知 ABOVE） |
| `ELU1` | ELU α=1 | e^z（床なし・既知 AT_FLOOR） |
| `R` | ReLU | 0 |
| `GELU` | z·Φ(z)（erf 形） | Φ(z)+zφ(z)、z<−0.752 で負 |
| `SILU` | z·σ(z) | σ(1+z(1−σ))、z<−1.278 で負 |

5 腕 × 2 環境 × 3 seed = 30 モデル。

## 2. 判定

**床の判定は `layer_chimera_rl_0914` と同一**（閾値はこの走のラベルから導く）: seed ごとに、
late 窓 t41–50 の online 精度の平均が F + 3s/√10 以下なら `AT_FLOOR`、超えれば `ABOVE_FLOOR`。
F と s（ddof=1）は同じ窓の各タスクの多数派ラベル割合の平均と標準偏差。腕の集約は 3/3 一致で
`AT_FLOOR` / `ABOVE_FLOOR`、割れれば `DISAGREEMENT`。

**Q1**（RL の R・GELU・SILU）:
- 3 腕とも `AT_FLOOR` → **`ALL_DIE`**
- 3 腕とも `ABOVE_FLOOR` → **`NONE_DIE`**
- 混在 → **`SURVIVORS:<生き残った腕>`**
- どれかが `DISAGREEMENT` → **`UNRESOLVED`**

## 3. ゲート

- **G0（再現）**: RL で `ELU1` が `AT_FLOOR`、`LR` が `ABOVE_FLOOR`（どちらも 3/3）。外れたら Q1 は `NOT_TESTABLE`。
- **G1（錨・診断）**: RL の `LR`・`ELU1` の online 精度を `layer_chimera_rl_0914` の `LL`・`EE`（結果 `58c1819`）と
  タスクごとに比べ max|Δ| を報告。0.01 超で `ANCHOR_DRIFT`（Q1 は変えない）。バッチのモデル数が 24 → 30 に
  変わるので bit 一致は保証しない。
- **活性化の自己検査**: float64 で φ を torch の kernel（`relu`・`gelu`・`silu`）と、φ′ を autograd と照合。
  変異対照（φ を 1e−3 ずらすと検査が落ちること）つき。

## 4. 報告のみ（ラベルにしない）

- PM の late online 精度（各腕が学習できる活性化かの確認。PM でも床なら RL 固有の死ではない）
- 崩壊タスク: その seed の閾値以下に入り、以後 t50 まで出ないタスク
- 第 2 層の `tinygate`（|φ′| < 1e−8 = Adam の ε）の割合・ゲート平均・z̄₂ の推移

## 5. 記名予測（走行前に固定）

- **Issa: `ALL_DIE`**
- **Claude: `ALL_DIE`（55%）**。次点は ReLU だけ生き残る（20%: φ′ が厳密に 0 なので死んだユニットは旅をせず
  その場で止まり、新しいラベルのたびに一部が活性のまま残りうる）、GELU/SiLU が生き残る（15%: 箱 B で谷の
  向こうは復元域だった）、`UNRESOLVED` 10%。
- **Claude の副予測（報告のみ・導出から）**: 崩壊タスクは GELU < ELU1 ≈ SILU。ゲートが ε=1e−8 を割る深さは
  GELU で |z|≈6.3、ELU で 18.4、SiLU で 21 なので、沈む速さが同程度なら GELU が約 3 倍早く凍る。
