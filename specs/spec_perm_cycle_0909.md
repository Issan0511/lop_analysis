# 置換循環による幅成長の反実仮想 spec 0909
状態: 検証中 / 更新: 2026-09-09
親: [[幅と沈下の介入_結果_0909]] / [[Leak距離補償仮説_説明と検証方針_0908]]
事後解析（0909 夜・未登録）で素LR の第1層中心化重み ‖W̃‖ が **t^0.46–0.48（√t に収束）** で伸び、‖W̃‖² のタスクあたり増分が t20–120 で一定（+0.57）、各タスクの更新 ΔW̃ と累積 W̃ の cos が −0.05（sd 0.033、R^783 のランダム方向の帰無 sd 0.036 と同じ）で全期間一定、増分同士の自己相関 −0.02、全 100 ユニットで指数 0.43–0.48、行平均 |m|/‖W̃‖ = 5e−3 で固定、第1層分散の現タスク由来割合 42%→25%（l2 は 87% 不変）を確認した。読みは「**各タスクが直交する化石フィルタを 1 層ずつ堆積し、WD 無しでは侵食されないので二乗が足し算になる**」。
**問い**: この直交はランダム置換というベンチマーク設計が作っているのか。置換を使い回せば増分が整列し、成長が止まるはずである。本 spec はその反実仮想。保存済み task20 状態からの決定的再生のみで、GPU 走はしない。

## 条件と継続
開始状態は task20 終端（`boundary_tasks20_40_0909/source/{ARM}_{iv}_s{seed}_states.pt` の boundaries[0] `before`・`adam_before`・V）。task21..40 を 625 更新×20 タスク継続する。**標本の層化抽出とバッチ順は各境界の保存 `rng_after_perm` をそのまま使う**ので、条件間で違うのは画素の置換だけ。
- `FRESH`: 保存 `perm`（task21..40 の本来の置換）。参照腕。保存 `after_625` 20 個を params・Adam・V とも差 0 で再現しなければならない（G1）。
- `CYCLE1`: task21 の perm を 20 タスクすべてに使う。
- `CYCLE2`: task21, 22 の perm を交互。
- `CYCLE5`: task21..25 の perm を循環。
腕: LR/none（主）、SNA/none、LR/l2（WD 対照: 循環しても幅は動かないはず）。seed0..2。3 腕 × 3 seed × 4 条件 = 36 継続。

## 測定
各タスク終端で: ‖W̃‖²（中心化行ノルムの二乗のユニット平均、params から厳密）、ΔW̃ の cos（累積 W̃ との）、‖ΔW̃‖²、z̄_inv・σ_inv（固定参照 perm 8 本・seed 20260909）、行平均、bias、現タスクの test 精度。
継続終了後に 20 個の増分 ΔW̃_t（t=21..40）の**全対の cos 行列**（ユニット平均）を保存し、「同じ perm の対」と「違う perm の対」に分ける。

## 判定（事前登録）
seed ごとに計算し、3/3 一致でラベル。
- 成長率 g_X = task K+1..40 での Δ‖W̃‖² のタスク平均（K は循環長。FRESH は K=0）。
  **`GROWTH_IS_PERMUTATION_DRIVEN`**: CYCLE1・2・5 のすべてで g_X < 0.25·g_FRESH。
  **`GROWTH_PERSISTS_UNDER_CYCLING`**: いずれかで g_X > 0.75·g_FRESH。それ以外 `PARTIAL`。
- 整列: CYCLE2・5 の増分対で、同じ perm の対の cos 平均 a_same、違う perm の対の cos 平均 a_diff。
  **`SAME_PERM_ALIGNED`**: a_same > +0.2 かつ |a_diff| < 0.05。**`NO_ALIGNMENT`**: a_same < 0.05。それ以外 `PARTIAL`。
- 深さ追随: CYCLE5 の task26..40 での z̄_inv の変化 Δz_C と FRESH の同区間 Δz_F。
  **`DEPTH_STOPS_WITH_WIDTH`**: |Δz_C| < 0.25·|Δz_F|。**`DEPTH_KEEPS_SINKING`**: |Δz_C| > 0.75·|Δz_F| かつ同符号。それ以外 `PARTIAL`。
- l2 対照: 全条件で g_X の絶対値が 0.05 未満なら `WD_PINS_REGARDLESS`。

## 事前予測（記名）
- Claude: `GROWTH_IS_PERMUTATION_DRIVEN`・`SAME_PERM_ALIGNED`・`DEPTH_STOPS_WITH_WIDTH`・`WD_PINS_REGARDLESS`。CYCLE1 は解けたタスクの勾配が消えて Adam の雑音歩行だけになり Δ‖W̃‖² ≈ 0.04–0.1/task。CYCLE5 は最初の 5 タスクで 5 層堆積して以後ほぼ頭打ち。
- 反証の形: 循環でも Δ‖W̃‖² が +0.4 以上で伸び続ければ、直交は置換由来ではなく学習側の性質。

## 検算と変異対照
1. FRESH が保存 after_625 20 個を差 0 で再現。対照: W1[0,0] に +1e−3 で >1e−4。
2. CYCLE 条件で実際に使った perm が指定どおり（保存 perm との一致・不一致を明示的に検算）。対照: CYCLE1 の task22 の perm が保存 task22 perm と**一致しない**こと。
3. ‖W̃‖² の増分恒等式 Δ‖W̃‖² = ‖ΔW̃‖² + 2⟨W̃,ΔW̃⟩ が各タスクで 1e−10 以内。対照: 符号を反転した式が外れる。
4. 測定が学習を変えない（検査 1 が兼ねる）。対照: 測定側で W1 に +1e−9。

## 運用
CPU 1 thread・最大 2 process・1 継続 120 秒上限。spec とコードを実行前に commit/push。CSV・verdict・summary・図を commit、cos 行列 npz は SHA 照合付きで `obsidian-research-data/perm_cycle_0909` へ。
限界: 20 タスクの短い継続。循環条件はタスクが易しくなる（既習）ので現タスク精度を条件間で比較しない。可塑性の読み出し（新規 perm への学習）は本 spec に含めない。元 GPU 軌道ではなく CPU 再生軌道。
