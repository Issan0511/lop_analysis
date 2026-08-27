# ceiling_t0_0828

## 主判定

- 構造・統計 sanity: **PASS**
- T0_root: **NO_RESOLVED_SEPARATION**
- `g_A(40)`: -0.93249204 [-0.98345895, +0.56248494]
- 同時根存在率: 1.0000
- T0_horizon: **NO_RESOLVED_HORIZON_SHIFT**
- `P_g=g_A(40)-g_A(1)`: -0.54246679 [-0.62640007, +0.4924704]
- 4根同時存在率: 1.0000

## 1-step sanity

- `S_one=E[D_1-F_gate,start]`: +1.7432354e-06 [-5.606468e-06, +9.8322587e-06]
- 95% CI が 0 を含む: **True**

## 適用範囲

condA・w100・T=10,000・batch=1・std の境界前窓。同一開始コホートの k<=40 に限定する。
bulk Δ1000 は `bulk_reference.csv` の別領域参考値であり、主判定には含めない。
