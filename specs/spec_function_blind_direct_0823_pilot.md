# spec_function_blind_direct_0823_pilot: 直接機能量 ΔL の計装パイロット

proj_004 / 作成 2026-08-23 / 対象リポジトリ: lop_analysis / **パイロット専用・主張への昇格禁止**

> **先行登録**: Obsidian `中心主張v3作業リスト_0823.md` の作業6を commit `66c4f3f` で登録済み。本 spec を計装実装・パイロット走の前に commit する。
>
> **既知情報**: 作業2で300k any-hitの天井効果とendpoint Hの未決着を確認済み。さらに本spec作成前に、既存 `ratchet_log_0819` の p̂ だけを用いて同一タスク内候補を探索し、タスク切替+1から次境界までの `strict_dead` 率が、200k〜800kで pooled 0.2545（1,889/7,423）、seed等重み 0.2681 と確認した。この確認には ΔL を使っていないが、追跡窓の選択は既存結果を見た**パイロット選択**である。

---

## 0. 目的と禁止

condA・w100 の各ランドマークで、各ユニットを一つだけ消音した厳密損失増分 `ΔL_i` を読み取り専用 probe で記録できることを確認し、独立確認走の seed 数・主解析・等価幅を決める材料を作る。

本パイロットの効果量・CI・符号を、機能盲目性の支持・否定・確認結果として本文、vault の結論、命題リストへ転記してはいけない。確認結果は、パイロット後に別 commit で固定する confirmation spec と独立乱数走だけから出す。

## 1. 条件と乱数

- condA・w100・T=10,000・std・batch=1・lr=0.01
- 学習 step: 0〜810,000
- pilot R=10、run label `seed=0..9`
- `generator_offset=0`。これは `ratchet_log_0819` と同じ乱数系列を再生するためで、独立データではない
- `common.generator_offset` を `setup_group → make_gens` に渡す。未指定時の既定0は既存全実験と bit 一致でなければならない
- confirmation は0以外の固定 offsetを使う。seedラベルだけを10..へ変えることは禁止する。現実装のseedラベルはR軸の表示名で、乱数seedそのものではないためである

## 2. ランドマーク・転帰

- task switch `B ∈ {200000, 210000, ..., 800000}` の61個
- 起点 `t0=B+1`。学習ループのprobeはstep先頭に走るため、Bで起きる `env.step()` 内のflipと1回のSGD更新を経た新タスク状態である
- 終点 `t1=B+10000`。次のflipを行う直前の、同じタスク末尾状態である
- 主リスク集合: t0で `p_hat >= 0.05`
- 主転帰: t1で `strict_dead := (p_hat == 0)`
- 副転帰: t1で `dead_0.05 := (p_hat < 0.05)`
- 中間時点のhitは使わない。同一タスク中に `p_hat=0` ならそのタスクの32入力上で勾配が恒等0なので、endpointで十分である
- 同一unitが複数taskに入る反復曝露を許す。unitを独立標本としてCIを作らない

## 3. 直接機能量

各記録点で現在タスクの32入力を全列挙し、float64で教師と学習器を評価する。

- `q_i(x) := v_i relu(W_i x_in + b_i)`
- `δ(x) := yhat(x)-y(x)`
- unit i 消音後の残差は `δ_{-i}=δ-q_i`
- `delta_mse_i := E[(δ-q_i)^2] - E[δ^2] = E[q_i^2 - 2δq_i]`
- 主機能量 `utility_nmse_i := delta_mse_i / Var(y)`
- 副機能量 `utility_raw_i := delta_mse_i`

`utility_nmse>0` はその時点・その現タスクで消音すると損失が増える正の限界寄与、0付近は冗長、負値は消音で損失が下がることを表す。unitの普遍的価値、将来タスク価値、Shapley値ではない。

同じprobeで `p_hat`, `x=w·mu_hat`, `r=sqrt(||w||^2-x^2)`, `w_norm`, `b`, `v`, `eval_nmse`, `Var(y)` を保存する。

## 4. パイロットで確認するもの

1. `delta_mse` のベクトル式を、seed×step×unitから決定的に選ぶ少なくとも20例について、実際にunitを一つずつ消音して再forwardした値と最大絶対誤差 `<1e-12` で照合
2. p̂ がk/32に量子化し、`x^2+r^2=||w||^2` の最大相対誤差 `<1e-10`
3. pilotの offset=0 の p̂・x・r が、共通記録点で `ratchet_log_0819` と保存float32の丸め許容内で一致
4. probeあり/なしの同一短走で、net・env・teacher・running_mean・全generator stateがbit一致
5. utilityの有限性、符号分布、seed×taskごとのリスク数・転帰率、p̂/x/rとの相関
6. utility群の切り方、幾何統制法、seed数の候補。ただしどの候補の結果も確認結果として扱わない

## 5. パイロット後に固定する confirmation spec

confirmationデータを見る前に、少なくとも以下を別spec・別commitで固定する。

- `generator_offset` とR
- 主utilityの群分けまたは連続効果の定義
- p̂・x等の統制法と空セル処理
- 主効果、等価幅、bootstrap単位・反復数・RNG seed
- PASS / PROTECTIVE / HARMFUL / INCONCLUSIVE の規則
- 必要seed数。計算不能なら事前に固定した実行可能上限と、その場合の「検出力不足」表記
- confirmationの出力一覧とsanity

パイロット効果が好都合か不都合かで、確認主指標を差し替えない。confirmationを見てから追加した解析は追補と明記する。

## 6. 実装・出力予定

- config: `configs/function_blind_direct_0823_pilot.yaml`
- 計装: `src/function_blind_direct.py`
- 集計: `analysis/function_blind_direct/pilot.py`
- 出力: `results/function_blind_direct_0823_pilot/`
  - `logs/seed*.npz`: t0/t1のunit指標
  - `exposures.csv`: seed×unit×taskのpilot表
  - `pilot_diagnostics.csv`, `pilot_summary.md`, `meta.json`
  - probe無擾乱性と旧ログ一致のsanity

実行コマンドは実装後に `--help` とconfigへ固定し、`OMP_NUM_THREADS=1` を必須とする。入力成果物を上書きしない。

## 7. スコープ

condA・w100・T=10,000・std・batch=1・この教師族だけ。パイロットは旧乱数系列の再計装であり独立再現ではない。O（オラクル修復）は作業2で完了しているため再実行しない。
