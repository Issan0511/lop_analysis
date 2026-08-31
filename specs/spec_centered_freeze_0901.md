# centered_freeze_0901: condA・centered × freeze_bias（P-1）

状態: **事前登録・未実行** / 作成: 2026-09-01 / run id: `centered_freeze_0901`

親: `中心主張v5計画_0831` §5 P-1 / 参照: `mlp2_phase1_0829` の
`L1w100_A1`、`bias_margin_0814` PB-1、`condA_freeze_0815`

## 1. 問い

condA の入力をオンライン EMA で中心化して µ 経路を弱めた 5M 走では、1 隠れ層・
幅100の `strict_dead` が再び増える。この死が bias $b$ の沈下を輸送チャネルとするなら、
学習軌道の乱数実現を変えずに $b\equiv0$ へ凍結すると終盤の死はほぼ消えるはずである。

既存 `condA_freeze_0815` は std・1M であり、µ 経路が支配するため本問いの対照ではない。

## 2. 設計

- condA: `m=20`, `f=15`, teacher width 100, `T=10,000`, batch 1
- learner: 1 隠れ層・幅100・ReLU、plain SGD、`lr=0.01`
- centering: 学習器入力だけを `center_alpha=0.01` の EMA で中心化。教師は生入力を見る
- seed 0--9、5,000,000 step、CPU、`OMP_NUM_THREADS=1`
- 新規走は `freeze_bias=true` の1腕だけ。free 腕は committed 済みの
  `results/mlp2_phase1_0829/logs/L1w100_A1_seed{0..9}.npz` を使う
- `freeze_bias` は `VecMLP.sgd_step` の bias 更新だけを止める。勾配計算、入力・教師・
  初期化の乱数消費は変えない
- 32 パターン全支持を task end（0, 10k, ..., 5M）で読み取り専用列挙し、
  `strict_dead := p_hat == 0`、`unfit := Var(yhat-y)/Var(y)` を厳密計算する

## 3. 実行前 sanity

- **S0**: 同じ harness の `freeze_bias=false` 30k replay が既存 `L1w100_A1` と、
  1k 格子の `p_hat`、`eval_loss_exact`、`unfit` で一致すること。失敗時は本走禁止
- **S1**: seed、step grid、初期条件が free 参照と一致する
- **S2**: frozen 腕の全記録点で `max|b| == 0.0`
- **S3**: `p_hat` は 1/32 格子、全主指標は有限、32 点列挙は学習状態と RNG を変えない

## 4. 事前登録判定

実験単位は seed。主窓は task 451--500 の task-end 50 点で、seed 内平均を一値にする。
差は `frozen - free` の paired percentile bootstrap 95% CI（10,000回、seed 20260901）。

### P1（主判定: b 経路の必要性）

`BIAS_ROUTE_DECISIVE` は次の3条件をすべて満たす場合:

1. frozen の終盤 `strict_dead_frac` grand mean $\le 0.05$
2. 減少率 `(free - frozen) / free` $\ge 0.80$
3. paired 差 `frozen - free` の 95% CI 上端 $<0$

3のみ満たすが1または2を満たさない場合は `BIAS_ROUTE_PARTIAL`。3を満たさない場合は
`BIAS_ROUTE_NOT_SUPPORTED`（frozen が有意に悪化なら `BIAS_FREEZE_INCREASES_DEATH`）。
最終 step=5M の値も同じ形式で報告するが、主判定は終盤窓から動かさない。

### P2（副次: 機能）

終盤 `unfit` の paired 差を `FROZEN_BETTER / FROZEN_WORSE / NULL` で報告する。
ただし bias 凍結は仮説空間も変えるため、P2 単独から「dead の機能コスト」は因果同定しない。

## 5. 帰結の境界

- P1 が decisive なら、**condA・centered・本設定の終盤死**について b 経路が必要である。
  condB、std、多層、他幅、他 optimizer へ外挿しない
- EMA centered は µ を厳密ゼロにしない。frozen に残る死は EMA 残差 µ 経路を含み得る
- `strict_dead` と LoP は同義ではない。機能は P2 を別に報告する
