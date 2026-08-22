# spec_q3_covariance_axis_posthoc_0822: centered 後の共分散主空間仮説

proj_004 / 作成 2026-08-22 / **結果観察後の探索解析（事前登録ではない）**

## 0. 問い

`ratchet_centered_0822` で `cos(u, mu)` 上の消灯領域が消えた理由として、次の仮説を既存成果物だけで数値検証する。

1. centered ではユニットの重み方向が、入力共分散 `Sigma` の最大固有空間へ移る。
2. タスク境界直後の残差平均 `mu = E[x] - running_mean` はその最大固有空間とほぼ直交する。
3. その結果、ゲート率は `cos(u, mu)` 単独より、バイアスと分散を含む規格化マージン
   `beta = (w^T mu + b) / sqrt(w^T Sigma w)` でよく説明される。

これは結果を見た後に立てた機構仮説であり、以下の数値は確認的証拠として扱わない。新規学習走、ハイパーパラメータ変更、seed 除外は行わない。

## 1. 幾何

condA の周期内入力は、先頭15個が固定 flip bit、末尾5個が独立 `Bernoulli(0.5)` である。定数を引く centered 操作は共分散を変えないため、両アームとも

`Sigma = diag(0 x 15, 0.25 x 5)`。

最大固有値は5重に縮退する。したがって固有ベクトル1本との cosine は定義せず、最大固有空間 `S_max` への射影率

`q(v) = ||P_max v||^2 / ||v||^2`

を使う。等方ランダム方向の期待値は `dim(S_max)/d = 5/20 = 0.25`。

## 2. 固定する解析

入力は std / centered の step 0 と step 1,000,000 checkpoint、および centered の既存 probe log だけとする。seed 0--9 を全て使う。

### H1: W の最大固有空間への移動

- unit ごとの `q(w_i)` を全unitと alive unit（厳密32パターンで `p_hat >= 0.05`）に分けて集計する。
- seed ごとの平均を実験単位とし、(a) centered final - centered initial、(b) centered final - std final を対応あり seed bootstrap（10,000回、seed=20260822）の平均差と95% percentile CIで出す。
- 集団整列軸は W の第1右特異ベクトルとする。raw W と行を単位化した U の両方、全unit / alive unit の両方で `q(e1)` と第1特異値エネルギー比を報告する。
- H1 は主に全unitの平均射影差で判断し、主軸指標は分布の集中を記述する補助量とする。

### H2: 境界直後の mu と最大固有空間

- 元の入力generatorとEMA更新を再生し、20,901個の既存記録時点で `mu` を復元する。
- checkpoint の `running_mean`、probe log の `flip_state` と `mu_norm` に一致することを検査する。
- `q(mu)` と、ベクトルから部分空間への主角 `acos(sqrt(q(mu)))` を seed ごとに集計する。
- 境界直後は probe の時刻規約に合わせて `t mod T = 1..100`、境界前は `T-100..T-1` および更新前の `t mod T = 0`、残りを bulk とする。
- H2 は境界直後の角度を連続量として報告する。恣意的な PASS 閾値は置かない。

### H3: ゲート座標

- 各 checkpoint で32パターンから `p_hat`、`cos(u,mu)`、`beta` を厳密計算する。
- seed ごとに unit 間 Spearman 相関 `rho(p_hat, cos)` と `rho(p_hat, beta)` を計算する。
- centered final での差 `rho_beta - rho_cos` を対応あり bootstrap する。95% CI が0より上なら「この checkpoint では beta の順位説明が優位」と記述する。
- `beta` は5ビット離散分布の完全な十分統計ではないので、機構の完全証明とは呼ばない。

## 3. 出力と留保

実装は `analysis/q3_covariance_axis.py`。出力は `results/ratchet_centered_0822/exploratory_covaxis/` に置く。

- `checkpoint_unit_metrics.csv`
- `checkpoint_seed_metrics.csv`
- `comparisons.csv`
- `mu_geometry_seed_summary.csv`
- `mu_geometry_by_offset.csv`
- `summary.md`
- `fig_covariance_axis.png`

最終 checkpoint しか W の全方向を保存していないため、境界直後の contemporaneous な W--mu 角は直接測れない。最大固有空間と mu の角度は各時点で厳密だが、最終 W 主軸を過去の mu に当てる計算を行う場合は「final-axis proxy」と明記する。主張範囲は condA・w100・T=10,000・alpha=0.01・1M step に限定する。
