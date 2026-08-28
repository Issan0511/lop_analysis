# ceiling_t0b_C_0829

腕C（pre-registered (new seeds)）。凍結 spec: vault `b44078c` `可塑性喪失/spec/天井T0b_spec_0828.md`。

## 主判定

- preflight: **PASS**（`preflight.csv`）/ 構造アサーション B1–B10: **PASS**（`assertions.csv`）
- ベースラインゲート `M(1)`: -1.0492021e-05 [-4.448832e-05, +2.7740721e-05] → **PASS**
- ガード: 寄与 seed **10/10**、窓内ペア 157,248
- **T0b_window: H_POSITIVE**
- `M(40)`: +6.381574e-05 [+1.3617671e-05, +0.00010602302]（seed クラスタ bootstrap B=10000, seed=20260829, studentized CI）

## 副次量（判定に使わない）

- 経路分解 k=40: `F_path-F_0` = +7.4483922e-05 [+4.5578299e-05, +0.00013397737] / `D-F_path` = -1.0668182e-05 [-3.9557565e-05, +2.2355897e-05]
- ブリップ寄与（帯 [0.6,0.7) 除外 − 窓全体）: -3.1680362e-06（相対 -0.0496）
- 窓内 strict off: k 別 {1: 0, 2: 0, 5: 0, 10: 0, 20: 0, 40: 0}
- CI: studentized（spec §4 改訂 vault `7f6b7d7`）。§5-7 の等調零点のみ percentile（位置量・判定外）
- 等調零点 k=40（**記述であり判定していない**）: `z̃_F` = +0.155025, `z̃_D` = nan, `g̃` = nan [nan, nan]、同時存在率 0.4138
  - 交点座標は準 max 型であり、`F` のゼロ近傍を横切る脆さは isotonic では消えない。`g̃` の CI が 0 を外しても「分離が示された」とは書かない（spec §5-7）。
  - no descending crossing inside the window: the smoothed F is mixed and the smoothed D is all_negative on the guarded window bands。すなわち上側窓の内側で `g̃` は定義されない。
- k プロファイル・感度は `window_curve.csv` / `sensitivity.csv`。

## 適用範囲（spec §10）

condA・w100・T=10,000・batch=1・std の境界前窓、上側窓 `[+0.1,+0.9)`、同一開始コホートの k<=40 に限定する。
力のゼロ点と変位のゼロ点の**位置**の主張、天井が `|F_self|/|F_rest|=1` であること、Δ1000 の符号逆転の説明、選抜と道のりの因果的二分、Q17 の `F_rest` 対称性、centered・他の w・T・batch・データセットへの一般化、真の介入効果はいずれも本結果からは言えない。
