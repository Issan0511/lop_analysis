# spec_function_blind_m2_0828: 直接機能量群の開口量変化

proj_004 / 作成 2026-08-28 / **M2 実行前固定**

Obsidian 正本: `可塑性喪失/spec/直接機能量と開口量M2_spec_0828.md`、事前登録 commit `1096be3`。

## 0. 問い

`function_blind_direct_0823_confirm` で固定済みの `utility_nmse`
low/mid/high 群について、

```text
S_i(t) = mean_32 relu(W_i(t) x_in + b_i(t))
delta_S = S_i(t1) - S_i(t0)
```

を比べる。高 Delta-L 群の凍結率が低いのは、起点 `S0` が大きいだけか、
`delta_S` の分布自体が異なるかを分ける。

## 1. 構造前提

- 既存ログは32 preactivation、W、Sを保存しておらず、Sは復元不能
- 元走と同じ condA/w100/T10000/std/batch1/lr0.01、seed 0..19、
  `generator_offset=20260830`、810,000 step を CPU・`OMP_NUM_THREADS=1` で再生する
- `t0=B+1`, `t1=B+10000`, B=200k..800k の61ペア
- リスク集合、`cell_id`, `utility_nmse_group`, `primary_cell_valid` は既存
  `exposures.csv` から固定し、再分位化しない
- S は開口数で割らず常に32で割る。`S=0 iff p_hat=0`
- r-swap は実行済みなので、M2 は独立判定

起草時に S、delta_S、群別分布、主推定量は未計算。

## 2. 再生一致 gate

正本 input:

```text
results/function_blind_direct_0823_confirm/
```

- `exposures.csv` SHA-256
  `2edc9aa82185843d8fd7f9663380b60590cd75027b27601f19546b39ef7b126b`
- `instrumentation_meta.json` SHA-256
  `a191e440fb9da5ed7a61c3491100911c3f0c09848fffa088160df4d96c6cd8b3`
- 20 NPZ は元 `meta.json.input_sha256` と一致

新 probe は読み取り専用。次をすべて必須にする。

1. 最終 complete state hash 辞書が元走と完全一致
2. 全122記録点の既存 `UNIT_KEYS` / `RUN_KEYS` が `np.array_equal`
3. S が finite/nonnegative、`S=0 iff p_hat=0`
4. 決定論的20例の scalar 32-support 平均との誤差 `<1e-12`

不一致なら `REPLAY_MISMATCH` で停止し、判定しない。

## 3. 主推定量

作業6の有効幾何セル c 内で high H と low L を比べる。Y に対し

```text
PS_c(Y) = mean_{h,l}[I(Y_h>Y_l) + 0.5 I(Y_h=Y_l)]
A_c(Y)  = PS_c(Y) - 0.5
w_c     = min(n_H,c, n_L,c)
A_Y     = sum(w_c A_c) / sum(w_c)
```

主は `A_deltaS`。`A_S0` を起点差に使う。raw 平均差も同じセル重みで
必ず出すが、主判定に差し替えない。mid は分布記述のみ。

## 4. bootstrap と判定

- 20 seed block bootstrap、B=10000、`default_rng(20260901)`
- 全推定量は同じ resample index
- セル/群ラベルは固定。群が消えたセルは当該 replicate で重み0
- 優越確率の等価域: `[-0.05,+0.05]`
- 非有限 replicate が1件でもあれば sanity FAIL

`M2_dynamics`:

1. CI 全体が等価域内: `EQUIV_DYNAMICS`
2. CI 下端 > +0.05: `HIGH_LESS_PUSHED`
3. CI 上端 < -0.05: `HIGH_MORE_PUSHED`
4. 他: `INCONCLUSIVE`

`M2_baseline`:

1. CI 下端 > +0.05: `HIGH_STARTS_MORE_OPEN`
2. CI 上端 < -0.05: `HIGH_STARTS_LESS_OPEN`
3. CI 全体が等価域内: `EQUIV_BASELINE`
4. 他: `INCONCLUSIVE_BASELINE`

`EQUIV_DYNAMICS + HIGH_STARTS_MORE_OPEN` のときだけ、「起点がより開いていただけ」を
M2 の操作的範囲で支持する。

## 5. sanity と出力

追加 gate:

- 15,582 risk exposure、6002/2839 cell、high/low行数と重みが元走と一致
- `(seed,unit,t0,t1)` で1:1 join
- risk rows で `S0>0`, `S1=0 iff end_strict_dead=1`
- 同一入力/commit/RNGの解析2回で全CSV byte一致

出力は `results/function_blind_m2_0828/`。元ディレクトリを上書きしない。

## 6. 解釈上限

- 観察的関連であり、Delta-L の因果効果と呼ばない
- `INCONCLUSIVE` を差なしと呼ばない
- 作業6 PROTECTIVE や r-swap SPECIFIC を事後的に差し替えない
- condB、他幅、他教師、因果介入へ外挿しない
- 同一軌道の再計装を独立 replication と呼ばない
