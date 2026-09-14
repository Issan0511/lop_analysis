# 層別キメラで RL の ELU 崩壊の所在を決める（`layer_chimera_rl_0914`）

事前登録 / 起案: Claude（2026-09-14・Issa 指示「回して」）/ 実装・実行: Codex（GPT Sol）
親: [[ELUの沈下と学習余力を分ける_結果_0913]] §C（`elu_environment_0913`）／[[act-chimera-0913]]

## 0. 問い

Random Label MNIST（RL）で ELU(α=1) は t10 前後で online 10% の床に落ちるのに、Permuted MNIST（PM）では leaky と同水準（99%）を保つ。

事後観察（`elu_environment_0913` の `units.npz` の再集計・未登録）では、差は**第 2 層に局在**する。t50 の第 2 層前活性の中央値は RL-ELU −153（sd 45・上端 z̄+3sd = −19・ゲート<.05 が 99.997%）に対し PM-ELU −19.7（sd 10.5・上端 +12 で t2–t50 ほぼ不変）。leaky は RL のほうがむしろ浅い（−7.5・上端 +3）。第 1 層は環境で大差がなく、ELU 第 1 層は RL でも死なない。RL-ELU の第 2 層の沈下は t2→t8 で毎タスク倍増し、相対速度 dz/sd は −0.8/タスクで一定、**sd が毎タスク ×1.45 で育つ**（LR ×1.10・PM-ELU ×1.07）。

**この崩壊が「ELU がどちらの層にあるか」で決まるかを、層別に活性化を混ぜて決める。**

読みの対立:

| 読み | 主張 | Q1 の予言 |
|---|---|---|
| **育つ入力の層**（Claude の事後の読み） | RL の記憶圧が ELU 第 1 層の正側出力を肥大させ、それが第 2 層の前活性幅を指数的に広げ、幅に比例して沈む第 2 層が上端ごと折れ目の下に落ちる。第 1 層の入力は画素で有界なので同じことは起きない | EL・LE とも生存。EE だけ崩壊 |
| **床の無い深い層**（単純な床説） | 深部で φ′→0 の層はどこにあっても RL では死ぬ。生死を分けるのは第 2 層に床があるかどうか | LE も崩壊。EL は生存 |

## 1. 箱（`elu_environment_0913` と同一・宿主は無改変で vendoring 済み `ef13afc`）

MNIST 1200 枚/seed 固定・784-100-100-10 MLP・Adam lr 1e−3・batch 16・**6000 更新/タスク（80 epoch）**・50 タスク・3 seed。
RL = 画像固定・毎タスク一様乱数ラベル。PM = 毎タスク画素置換・真ラベル。**介入なし（`ref` のみ・W1 クランプは使わない）。**
初期化・タスク系列・ラベル draw の RNG は宿主の `stream(role,seed)`・`initial(seed)` をそのまま使う。

## 2. 腕（2×2×2 = 24 モデル）

`act1` ∈ {ELU1, LR} × `act2` ∈ {ELU1, LR} × `env` ∈ {RL, PM} × `seed` ∈ {0,1,2}

| 略号 | 第 1 層 | 第 2 層 | 位置づけ |
|---|---|---|---|
| **EE** | ELU1 | ELU1 | 親（既存 RL-ELU の再現） |
| **EL** | ELU1 | leaky .1 | キメラ |
| **LE** | leaky .1 | ELU1 | キメラ |
| **LL** | leaky .1 | leaky .1 | 親（既存 RL-LR の再現） |

第 3 層（読み出し）は線形で全腕不変。前向き φ と逆向き φ′ は層ごとに一致させる（`act_chimera_0913` のような前向き/逆向きの分離はしない）。

## 3. 測る量（毎タスク終端）

- `online_acc`・`train_acc`（宿主と同じ定義）
- 層別のユニット別 `zmean`・`zstd`・`gate_mean`・`lowgate`（φ′<.05 の入力割合）
- **`a1_rms`**: 第 1 層出力 a₁ の RMS（1200 入力 × 100 ユニットをまたぐ）— **新規・読みの操作チェック**
- `w_norm_l1/l2/l3`・`cnorm_l1`

## 4. 窓と閾値（すべて算術から導く）

- **late 窓** = t41–50 の `online_acc` 平均。
- **床 F**: 各タスクのラベル draw に対する最良定数予測器の精度 max_c n_c/1200 を窓の 10 タスクで平均。
  **閾値 thr = F + 3·s/√10**（s は窓 10 タスクの同量の標準偏差）。
  `AT_FLOOR` ⇔ late ≤ thr、`ABOVE_FLOOR` ⇔ late > thr。
- **第 2 層の幅の成長率 g** = (sd₂(t8)/sd₂(t2))^(1/6)（sd₂ はユニット中央値・seed 平均）。
  高低の境は**この走の両親の幾何中点** √(g_EE·g_LL)。以上を HIGH、未満を LOW。
  （既存の走での参考値は g_EE ≈ 1.45・g_LL ≈ 1.10 だが、**閾値はこの走の両親から取り直す**。）

## 5. 登録判定

**Q1（主）** RL の EL と LE の床判定:

| 結果 | ラベル |
|---|---|
| EL・LE とも `ABOVE_FLOOR` | **`GROWING_INPUT_LAYER`** |
| LE のみ `AT_FLOOR` | **`FLOOR_IN_DEEP_LAYER`** |
| EL・LE とも `AT_FLOOR` | **`EITHER_ELU_KILLS`** |
| EL のみ `AT_FLOOR` | **`FLOOR_IN_SHALLOW_LAYER`** |
| EE が `ABOVE_FLOOR` | **`NOT_REPRODUCED`**（他はすべて `NOT_TESTABLE`） |

**Q2（操作チェック）** 第 2 層の幅の成長率 g はどちらの層の活性化で決まるか:

| 結果 | ラベル |
|---|---|
| g_EL HIGH かつ g_LE LOW | **`WIDTH_SET_BY_L1`** |
| g_EL LOW かつ g_LE HIGH | **`WIDTH_SET_BY_L2`** |
| その他 | **`WIDTH_MIXED`** |

**Q3（REPORT のみ）** PM の 4 腕の late 水準。PM は天井（99%）の想定で、崩壊しない対照。

**G1（アンカー）** RL の EE・LL は既存 `elu_environment_0913` の RL/ELU1/ref・RL/LR/ref を再現するか。
全 50 タスクの `online_acc` の max|Δ| を報告する。**bit 一致は要求しない**（バッチ構成が変わると GPU の縮約順が変わりうる）。
max|Δ| > 0.01 なら `ANCHOR_DRIFT` を立てて Q1/Q2 のラベルに併記する。

## 6. 予測（結果を見る前）

**Claude**:

1. Q1 = **`GROWING_INPUT_LAYER`**（確信 0.5）。次点 `FLOOR_IN_DEEP_LAYER`（0.3）・`EITHER_ELU_KILLS`（0.15）・その他（0.05）
2. Q2 = **`WIDTH_SET_BY_L1`**（0.6）
3. LE の late は LL（≈0.48）を下回るが床より上（0.20–0.45）
4. EL の late は LL と同程度かやや下（0.40–0.50）
5. 崩壊する腕では第 2 層の `lowgate` が late で >0.95 になり、崩壊しない腕では <0.5
6. **最もあり得る外れ方**: LE も崩壊して `EITHER_ELU_KILLS` になること。ELU の第 2 層は入力が育たなくても、RL の記憶圧だけで復元圏外まで沈むかもしれない

**Issa**: （結果を見る前に記入）

## 7. 引用制限

- 3 seed・1 箱・**80 epoch**。400 epoch 箱（`act_chimera_0913` の RL）とは別の走で、そちらの ELU1/SMINH の崩壊とは水準が違う。
- Q2 は**記述的な同時変化**。幅の成長を止める介入はしていないので、幅 → 沈下 → 崩壊の因果は決まらない。
- PM は天井なので「PM ではキメラが無害」を一般化しない。
- 前向きと逆向きを層内で分けていないので、「効いているのはゲートか前向きの値か」はこの走では決まらない（`act_chimera_0913` の分離腕の役目）。
- 崩壊/生存の二値は late 窓の床判定であって、生存腕どうしの水準差は別に読む。
