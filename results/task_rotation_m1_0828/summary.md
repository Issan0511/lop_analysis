# task_rotation_m1_0828

## 主判定

- **INCONCLUSIVE_GUARD**
- 有効4 seed 内の記述的 Spearman 平均（主判定に使用しない）: -0.118788
- 95% seed-cluster bootstrap CI: **未算出（登録ガード未通過）**
- 有効 seed: 4/10
- ガード: seed 内有効境界50本以上、かつ有効 seed 8/10以上

## 構造確認

- 隣接タスクは全境界で 1-bit flip
- 先生提示の一般余弦式と直接内積は機械精度内で一致
- `dead2path_0821` の再分類死イベント集合と完全一致

## スコープ

condA・w100・T=10,000 の既存ログにおける観察的な境界別関連であり、因果効果ではない。
