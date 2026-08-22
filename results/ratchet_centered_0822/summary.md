# ratchet_centered_0822: centered消灯点の判定

仕様: `specs/spec_ratchet_centered_0822.md`。事前登録コミット: `a08645c`。集計コードコミット: `ec5bc34`。

## 0. 一行

C1 **不可比** / C2 **不可比（C2を実施しない）** / C3 **不可比のため未実施**。

## 1. 主判定

| arm | theta_med | 95% CI | theta_all | theta_med_strat |
|---|---:|---:|---:|---:|
| std | -0.15 | [-0.15, -0.15] | -0.55 | -0.15 |
| centered | NA | [NA, NA] | NA | NA |

paired Δtheta = centered − std = **NA** [95% CI NA, NA]。

| id | question | result | theta_centered | theta_std | delta_theta | ci_lo | ci_hi | detail |
|---|---|---|---|---|---|---|---|---|
| C1 | centered曲線はstdと可比か | 不可比 | NA | -0.15 | NA | NA | NA | centered theta_med finite=False, cos<0有効ビン=12 |
| C2 | 消灯点は動いたか（主判定） | 不可比（C2を実施しない） | NA | -0.15 | NA | NA | NA | paired seed bootstrap 95%CI; 分解能0.05 |
| C3 | 四分位層別で一貫するか | 不可比のため未実施 | NA | NA | NA | NA | NA | 報告のみ。C2を覆さない |

## 2. ||w|| 四分位

四分位境界はarm内のpooled ||w||から別々に計算した（相対層）。

| arm | min | Q1 | Q2 | Q3 | max |
|---|---:|---:|---:|---:|---:|
| std | 0.8717 | 1.7023 | 2.2980 | 3.0719 | 7.7249 |
| centered | 0.8489 | 1.4139 | 1.6383 | 1.9865 | 4.0070 |

| scope | delta_theta_med | ci_lo | ci_hi | bootstrap_finite |
|---|---|---|---|---|
| all | NA | NA | NA | 0 |
| w_q1 | NA | NA | NA | 0 |
| w_q2 | NA | NA | NA | 0 |
| w_q3 | NA | NA | NA | 0 |
| w_q4 | NA | NA | NA | 0 |

## 3. 時間半割

| arm | scope | estimate | point | ci_lo | ci_hi | bootstrap_finite | theta_med_strat | nonzero_samples_below_theta_med |
|---|---|---|---|---|---|---|---|---|
| std | t_lt_500k | theta_med_time_half | -0.15 | -0.15 | -0.15 | 10000 | NA | NA |
| std | t_ge_500k | theta_med_time_half | -0.15 | -0.15 | -0.15 | 10000 | NA | NA |
| centered | t_lt_500k | theta_med_time_half | NA | NA | NA | 0 | NA | NA |
| centered | t_ge_500k | theta_med_time_half | NA | NA | NA | 0 | NA | NA |

## 4. E1（報告のみ）

- **std mu_norm**: median 3.041381, IQR [2.692582, 3.201562]。境界offset 0 / +1 / +100 の中央値は 3.041381 / 3.041381 / 3.041381。
- **std final dead_frac**: pooled 0.9470、seed中央値 0.9500、seed IQR [0.9500, 0.9500]。
- **centered mu_norm**: median 0.116858, IQR [0.071724, 0.590684]。境界offset 0 / +1 / +100 の中央値は 0.073082 / 0.992698 / 0.373327。
- **centered final dead_frac**: pooled 0.3240、seed中央値 0.3250、seed IQR [0.2900, 0.3700]。

境界offset 0はprobe順序上flip前、変更後のflip_stateが最初に見えるのは+1。dead_fracは判定に使用していない。

## 5. 集計規約とスコープ

- cosビンは [-0.60, 0.60)、幅 0.05。有効ビンはpooled n >= 1000。範囲外件数は `gate_curve.csv` に記録。
- bootstrapはseed束ね B=10,000、`np.random.default_rng(20260822)`。同じseed復元抽出を両armへ適用したpaired比較。
- 主判定はtheta_med全体のみ。theta_all、四分位、時間半割、theta_med_strat、mu_norm、dead_fracは副次または報告のみ。
- スコープはcondA・w100・T=1e4・batch=1・center_alpha=0.01。condBおよびalpha依存性へ外挿しない。

## 6. 出力

- `verdict.csv`: C1–C3
- `gate_curve.csv`: 全arm・全層・全ビンの曲線とCI
- `theta_estimates.csv`: theta_med/all・時間半割
- `per_seed_metrics.csv`: final dead_frac / mu_norm
- `analysis_meta.json`: provenance・bootstrap設定
- `figures/fig_q3_gate_curves.png`, `fig_q3_mu_norm_boundary.png`
