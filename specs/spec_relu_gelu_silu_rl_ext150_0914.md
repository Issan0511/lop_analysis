# relu_gelu_silu_rl_ext150_0914 —— GELU と SiLU seed 2 は 150 タスクで死ぬか

親: `spec_relu_gelu_silu_rl_0914.md`（結果 `7ce3b17`・登録判定 `UNRESOLVED`）
50 タスクでは GELU が 3/3 で床の上（late 0.221・傾き −0.009〜−0.013/10 タスク）、SiLU seed 2 も床の上（0.348・−0.029/10 タスク）。
他の床を持たない腕（R 3/3・SiLU s0/s1）は床。延長して決着させる。

## 1. 箱と腕

親と完全に同一（30 モデル: LR/ELU1/R/GELU/SILU × RL/PM × seed 0–2・両層同一活性化・1200 枚・80 epoch/タスク）。
**タスク数だけ 50 → 150**。乱数ストリームは親の続きをそのまま消費する（t1–50 は親と同一になるはず）。
エンジン・自己検査・活性化は親の `src/relu_gelu_silu_rl_0914.py` を import してそのまま使う。

## 2. 判定（窓を t141–150 に移すだけで、規則は親と同一）

モデルごとに late 窓 **t141–150** の online 精度平均が F + 3s/√10 以下なら `AT_FLOOR`（F・s は同じ窓のラベルの多数派割合）。
腕の集約は 3/3 一致。

- **Q1 GELU**: 3/3 `AT_FLOOR` → **`GELU_DIES_BY_150`** ／ 3/3 `ABOVE_FLOOR` → **`GELU_SURVIVES_150`** ／ 他 → **`GELU_SPLIT`**
- **Q2 SiLU seed 2**: `AT_FLOOR` → **`SILU_S2_DIES_BY_150`** ／ `ABOVE_FLOOR` → **`SILU_S2_SURVIVES_150`**
- **Q3 まとめ**: RL の R・GELU・SILU の 9 モデルすべてが t141–150 で `AT_FLOOR` → **`ALL_DIE_BY_150`**、そうでなければ生き残ったモデルを列挙

## 3. ゲート

- **G0（対照）**: RL の LR が t141–150 で 3/3 `ABOVE_FLOOR`。落ちたら「箱全体が 150 で死ぬ」ので Q1–Q3 に `CONTROL_ALSO_DIES` を付記（ラベルは出す）。
- **G1（再現）**: 30 モデルの t1–50 の `online_acc`・`train_ce` を親の `7ce3b17:results/relu_gelu_silu_rl_0914/rows.csv` と照合し max|Δ| を報告（件数も）。0 でなければ明記。
- 親の自己検査（φ・φ′・手書き逆伝播・変異対照）を走の先頭で再実行。

## 4. 報告のみ

- 最初に床を割ったタスク（t141–150 の閾値を全タスクに当てる）
- GELU と SiLU s2: median z̄₂ が谷底 z_c（GELU −0.752・SiLU −1.278）を**初めて割ったタスク**、第 2 層の φ′<0・|φ′|<1e−8 の割合（t50/t100/t150）
- LR の傾き

## 5. 記名予測（走行前に固定）

線形外挿の算術（親の t31–50 の傾き）: GELU は床に t≈125（s0）/156（s1）/153（s2）、SiLU s2 は t≈124、LR は t≈346 以上。
GELU の median z̄₂ は −0.09〜−0.10/10 タスクで下がっており、谷底 −0.752 を t≈126–130 で割る。

- **Issa**: 親の `ALL_DIE` を引き継ぐ（延長について新しく表明はしていない）→ Q3 `ALL_DIE_BY_150`
- **Claude**: Q1 `GELU_DIES_BY_150` 40% ／ `GELU_SPLIT` 35% ／ `GELU_SURVIVES_150` 25%。Q2 `SILU_S2_DIES_BY_150` 55%。G0 は LR 生存 90%。
- **Claude の機構予測（報告のみ）**: GELU が死ぬ seed では、**median z̄₂ が z_c を割るタスクが、最初に床を割るタスクより前**に来る
  （SiLU s0/s1 と同じく、谷の向こうへ入ってから死ぬ）。逆順（床が先）なら、GELU の死は反転域と無関係な緩やかな劣化。
