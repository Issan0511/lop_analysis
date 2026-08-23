# spec_mu_titration_0823: µ タイトレーション

proj_004 / 作成 2026-08-23 / 実行前事前登録

位置づけ: `中心主張v3作業リスト_0823.md` 作業5。作業リスト上は作業3・4の後だが、2026-08-23 の明示依頼「作業5を実装」を Go として先行する。本実験は新規学習走を伴う。**本 spec、canonical config、走行コードを commit するまで、新規アームの p_hat、cos、strict_dead を見ない。**

## 0. 一行

condA A_w100 のオンライン中心化 EMA 更新率 `center_alpha` を8点で振り、各点10 seedの実現 `mu_norm` と、観測消灯点 `theta_med`、厳密壁 `cos_crit`、bias逃走成分を同時に測る。「µが壁を手前に引く」と「µがbの逃走自由度を奪う」を、同じ用量曲線上の別量として分離する。

## 1. 介入の意味

現コードの `center_alpha` は平均を引く割合ではなく EMA 更新率である。

```text
running_mean <- (1-alpha) running_mean + alpha x_raw
x_in         = x_raw - running_mean
```

`enc=centered` では running_mean を常に100%引く。alpha は追随時定数とサンプリング雑音床の両方を変えるため、`alpha -> mu_norm` の単調性は仮定しない。介入ラベルは alpha、科学的な用量軸は各走行で実測する `mu_norm = ||E[x_in]||` とする。「部分減算率」とは書かない。

## 2. 設計

### 2.1 固定条件

- condA、m=20、f=15、target_hidden=100、width=100、T=10,000
- batch=1、lr=0.01、ReLU、method=none、1,000,000 step
- seed 0--9、CPU、`OMP_NUM_THREADS=1`
- 教師は生入力、学習器だけが centered 入力を見る
- 介入以外は `ratchet_log_0819` / `ratchet_centered_0822` と同じ

### 2.2 alphaグリッド

```text
0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-2
```

8点をすべて同じ commit から再走する。alpha=0 は学習器にとって既存 std と同一、alpha=0.01 は既存 centered と同一である。両端も新loggerで再走し、既存ログとの再現性と新しい十分統計を得る。

各 alpha は**別の R=10 invocation**とする。R=80の一括ベクトル化はgeneratorのshape依存な割当てでseed対応を壊す。各 invocation でgeneratorをresetし、初期値・教師・生入力・flip軌道をalpha間でseed単位に対応させる。

### 2.3 入力だけのPhase 0校正

学習器を走らせず、同じ入力乱数列だけを1M step再生した。既存20,901記録点での pooled median `mu_norm` は次のとおり。この値はグリッド被覆の確認だけに使い、判定には本走の実測値を使う。

| alpha | 0 | 1e-6 | 3e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median mu_norm | 3.041 | 2.329 | 1.878 | 1.513 | 1.068 | 0.629 | 0.166 | 0.117 |

alpha=1e-3では0.039まで下がる一方、alpha=0.01では0.117へ戻る。これは予想済みのEMA雑音床である。alpha=0.01をalpha順の単調判定には使わず、全用量反応は実測mu_norm順で読む。

## 3. 追跡可能な十分統計

各記録点で32パターンをfloat64で厳密列挙する。`delta_x := x_in - mu`、`resid := yhat-y` とする。

```text
M               = max_delta_x w·delta_x = ||w_free||_1 / 2
s               = w·mu + b
b_plus_M        = b + M
cos_crit        = -(b+M) / (||w|| ||mu||)
delta_b_field   = -2 lr v E[resid gate]
delta_wmu_field = -2 lr v E[resid gate (x_in·mu)]
```

NPZに保存する主要列:

- run: `step, seed, center_alpha, mu_norm, eval_loss_exact, flip_state`
- unit: `cos_u_mu, p_hat, w_norm, b, M, s, b_plus_M, cos_crit, delta_b_field, delta_wmu_field`

主解析はgit管理外checkpointに依存せず、Mを事後のkappa近似で置き換えない。

## 4. 記録グリッドと時間重み

既存と同じく境界±100 stepを毎step、その他を1000 stepごとに記録する（20,901点）。ただし生グリッドは記録点の約95%を境界窓が占め、実時間を代表しない。

- **主集計 bulk**: `step % 1000 == 0` かつ最寄り境界から100 step超
- **boundary**: 実現境界のoffset `[-100,+100]`
- **all-recorded**: 既存報告との互換用。判定には使わない

alphaは境界後の追随速度を変えるため、bulkとboundaryを混ぜない。時間半割と固定phase offset +5000を副次で報告する。

## 5. 推定量

### 5.1 観測消灯点 theta

- cosビン幅0.05、範囲 `[-1,1)` の40ビン
- 有効ビンは対象scopeでn>=1000
- 曲線値はビン内 `p_hat` の中央値
- `theta_med` は、低cos側から連続してビン中央値 `p_hat=0` となる最大ビン上端
- `theta_all` は参考値。最小値統計なので主判定に使わない
- 主scopeはbulk。boundary offset別、時間半割、固定phase、all-recordedは副次

`theta_med=NA` かつexact `cos_crit < -1` の場合は「壁なし」ではなく、cos定義域外への左打ち切りと記録する。

### 5.2 exact wall（経路1）

`q := (b+M)/||w|| > 0` の点だけを、µが大きいほど死領域が広がる壁レジームとする。

```text
-cos_crit = q / mu_norm
log(-cos_crit) = log(q) - log(mu_norm)
```

各 alpha × seed × scope で `mean log(-cos_crit)`、`mean log(q)`、`mean log(mu_norm)` を**同一マスク**で出し、加法恒等式誤差を保存する。用量点間のwall傾きを、直接の分母成分 -1 と、学習後分子qの変化に分解する。

`frac(cos_crit<-1)`、`frac(|cos_crit|>1)`、`frac(b+M<=0)` も報告する。`b+M<=0` は理論上向きが反転する別レジームなので、q>0のwall傾きに混ぜない。

### 5.3 bias逃走（経路2）

現在のmuを固定した32パターン期待勾配から、次を出す。

```text
bias_share_field = |delta_b_field| /
                   (|delta_b_field| + |delta_wmu_field|)
```

`p_hat>0` かつ分母 `>1e-12` のunit-recordだけを対象とする。arm × seedでは、外れた微小更新をunit数で過重しないよう、分子の絶対値和を分子・両成分の絶対値和を分母にした集約比も主値として保存する。両成分の同符号率を併記する。

この固定mu分解はrunning_mean自体の次step変化 `w·Delta mu` を混ぜない。構造参照 `1/(1+mu_norm^2)` も併記するが、`resid*gate` とxの相関を無視するため数値一致は要求しない。この一軸スイープで言うのは経路別中間量の分離までであり、strict_deadへの独立な因果寄与率ではない。

### 5.4 表現型

- 最終 `strict_dead = mean(p_hat==0)`
- 最終 `near_off = mean(0<p_hat<0.05)`
- 互換用 `dead_0.05 = mean(p_hat<0.05)`
- 最終 exact eval loss

修飾なしの `dead` は使わない。

## 6. 統計

- seed束ねpaired bootstrap、B=10,000、`np.random.default_rng(20260823)`
- 8 alphaの全推定で同じseed復元抽出重みを使う
- percentile 95% CI
- 用量回帰のxはalphaでなく、各bootstrap標本内の実測mu_norm
- theta回帰は有限thetaのarmだけを使い、`theta_med = a + b / mu_norm` を当てる
- exact wall回帰は `mean log(-cos_crit)` 対 `mean log(mu_norm)`
- escape回帰は集約 `bias_share_field` 対 `1/(1+mu_norm^2)`
- thetaの同値、NA、打ち切りを補間で埋めない。有限bootstrap標本数を必ず報告する

## 7. 事前判定

| ID | 問い | PASS | その他 |
|---|---|---|---|
| C0 | 用量とthetaを識別できたか | bulk arm median mu_normの最大/最小比>=10、異なるmu_normが6水準以上、有限thetaが4水準以上 | 満たさなければtheta主傾きはVOID |
| C1 | thetaはmuで動いたか（主） | `theta = a + b/mu_norm` の b<0、paired bootstrap 95% CI上端<0、かつseed別Spearman rho(mu,theta)中央値>=0.6 | 逆向きCI確定はFAIL、それ以外INCONCLUSIVE |
| W1 | exact wallは整合するか | q>0のlog wall対log mu傾きの95% CI上端<0。theta=NAの低doseはmedian cos_crit<-1で盤外化 | C1の機構照合。単独でC1をPASSにしない |
| C2 | bias逃走は戻ったか | bias share対`1/(1+mu_norm^2)`傾きの95% CI下端>0、かつseed別Spearman中央値>=0.6 | 逆向きCI確定はFAIL、それ以外INCONCLUSIVE |
| P1 | 表現型も同方向か | final strict_dead対mu_normの傾きとCIを報告 | 報告のみ。C1/C2を覆さない |

有限theta間の順序違反率も報告する。実測mu_normが大きいほどthetaが浅い、という順序に反するarm pairが20%を超えた場合、C1が傾き基準をPASSしても「非単調」を明記する。

総合判定:

- **FULL_PASS**: C0 PASS + C1 PASS + W1 PASS + C2 PASS
- **PARTIAL**: C0 PASSでC1/C2の一方だけPASS
- **VOID**: C0 FAIL
- **FAIL**: C1とC2が両方とも逆向きに確定
- **INCONCLUSIVE**: 上記以外

W1の直接分母成分 -1 とq成分の傾きを必ず併記するが、「全効果の何%が経路1/2」という因果媒介比率は計算しない。

### 7.1 事前予測

- alpha 0--3e-4でmu_normは単調に下がる。alpha=.01は雑音床で3e-4よりやや大きくなりうる
- mu_normが小さいほどq>0 unitのcos_critは負側へ深くなり、低doseでは-1を越えてthetaがNAになる
- mu_normが小さいほどbias shareは大きくなる
- final strict_deadは大doseほど高い

この予測を新規学習走の前に本specとともにcommitする。

## 8. Sanity / Phase 0

- **S1**: `OMP_NUM_THREADS=1`、torch thread=1
- **S2**: probeあり/なし100,000 stepでnet/env/running_meanがbit一致（各alpha）
- **S3**: 3記録点でexact p_hatとN=2000経験ゲート率が既存二項ゆらぎ基準をPASS
- **S4**: flipは実現境界だけで1ビット
- **S5a**: `M=||w_free||_1/2` と32 supportのmaxのmax abs error<=1e-12（float64）
- **S5b**: 全probeで `p_hat==0 <=> s+M<=0` の不一致0。有限分母上のcos_crit式誤差<=1e-12
- **S5c**: `delta_b_field + delta_wmu_field` と同じfull-support勾配から直接計算したDelta sの誤差<=1e-12
- **S6a**: alpha間でstep0 net/teacher/envと全flip軌道がseed単位にbit一致
- **S6b**: alpha=0は既存std、alpha=.01は既存centeredと共通列が数値一致。alpha=0のrunning_meanだけは定義上対象外
- **S7**: 10 seed、20,901記録点、予期しないNaNなし、`p_hat*32`は整数

S2--S7のどれかがFAILしたarmを黙って除外しない。全体を止め、修正をcommitし、同じcommitから全armを再走する。

## 9. 実行順序

1. 本spec、config、runner/loggerをcommit（事前登録）
2. 主集計を見ない短走Phase 0（shape、恒等式、無擾乱だけ）
3. clean HEADから8 alphaを実行。各alphaは別R=10、並列実行可
4. S2--S7を確認
5. 解析コードをcommitしてから解析
6. 同一commit・同一入力から別outdirに再解析し、CSV/JSON/Markdownのbyte一致を確認
7. raw/derived SHA256、summary、verdict、figures、metaをcommit
8. その後にのみvaultへ数値を転記

Canonical commands:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m src.mu_titration --config configs/mu_titration_0823.yaml --alpha GRID_VALUE --s2-steps 100000
OMP_NUM_THREADS=1 .venv/bin/python -m analysis.mu_titration.mu_titration --config configs/mu_titration_0823.yaml
```

## 10. 出力

```text
results/mu_titration_0823/
  arms/alpha_*/config_used.yaml, meta.json, logs/seed*.npz
  arm_manifest.csv
  raw_sha256.csv
  gate_curve.csv
  theta_estimates.csv
  dose_response.csv
  path_decomposition.csv
  per_seed_metrics.csv
  verdict.csv
  summary.md
  analysis_meta.json
  determinism_check.md
  figures/
```

## 11. 禁止事項・留保

- condB、他width/T/batch/lrへ外挿しない
- alphaを部分減算率と呼ばない。実測mu_normがdose
- all-recorded poolを実時間平均と呼ばない
- `p_hat<0.05`を不可逆な凍結と呼ばない
- theta=NAを壁の不在と即断しない
- C1/C2をstrict_deadへの独立な因果寄与率へ変換しない
- 結果後にbin、閾値、scope、判定表を本spec内で上書きしない。必要なら事後addendumを先にcommitする
