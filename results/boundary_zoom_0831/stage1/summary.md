# boundary_zoom_0831 — 段階1

判定: **TIMESCALES_NOT_SEPARABLE**

- tau_fit 中央値: 230.0
- 回復を観測した境界: 16 / 19
- S0'（既存1000-step点のbit一致）: PASS
- S1/S2 exact recorder: PASS
- S2 duplicate probe: PASS
- S3 OMP_NUM_THREADS=1: PASS
- S8 flip timing: PASS
- 実行時間: 37.0 sec

## 実装上の固定

tau_fit は residual_var を用い、各境界の offset [-300,0] の中央値を境界前定常水準とした。境界後の最大絶対偏差から 1/e 以内へ初めて戻る20-step格子点を tau_fit とする。これは段階1の実行前に固定した。

seed 0 の乱数列を既存runとbit一致させるため、内部では元runと同じ10 seedベクトルを進めた。保存・判定対象はseed 0のみで、他9 seedはRNG paddingである。

段階1の tau_fit は seed 0・19境界の記述統計であり、水準として引用しない。
