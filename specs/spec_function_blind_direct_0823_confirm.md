# spec_function_blind_direct_0823_confirm: 直接機能量 ΔL と同一タスク内凍結の独立確認

proj_004 / 作成 2026-08-23 / 対象リポジトリ: lop_analysis / **confirmation実行前固定**

> **先行記録**: Obsidian作業6は `66c4f3f` で登録済み。計装pilot specは `aaf7c19` / `828fec1`、実装は `a975d6a`、pilot正本は `a8c4ffb`。
>
> **pilotで見たもの**: generator_offset=0の旧軌道再計装では、utility_nmseのt0三分位で endpoint strict_dead の high−low RD = −0.3224 [−0.3591, −0.2838]、p̂×壁マージン3×3調整で −0.2344 [−0.2749, −0.2022] だった。これは設計用の既知結果で、confirmationへ昇格しない。以下の群分け・統制・判定規則は独立乱数データを見る前に固定する。

---

## 0. 一行

現在タスクの32入力上で単独消音損失 `ΔL_i` が大きいユニットほど、同じ `p̂` と厳密壁マージンを揃えても、そのタスク末尾で `strict_dead` になりにくいかを独立乱数20系列で確認する。

## 1. 条件・独立性・実行量

- condA・w100・T=10,000・std・batch=1・lr=0.01・0〜810,000 step
- R=20、run label `seed=0..19`
- **`generator_offset=20260830`**。pilotの0と異なる乱数ストリームを、結果を見る前に固定する
- seedラベルは乱数seedではなくR軸の表示名なので、番号だけを20..へ変える設計は禁止
- `src/train.py` の未指定offset=0経路が従来と同一であること、実offsetの初期状態hashがoffset=0と異なることをsanityに入れる
- probe無擾乱比較は同じoffset・R=20で **100,000 step**。net / env / teacher / running_mean / eval_fixed / 全generator stateのhashが完全一致しなければ中止
- pilot旧ログとの数値一致は独立offsetでは定義不能なので行わない

R=20はpilotの独立cluster数10を2倍にする実行可能な固定数。pilotの3×3調整CI半幅は約0.036で、R=20なら同程度のcluster分散の下でさらに縮む。真のRDが0付近なら±0.05等価判定、pilot方向の大効果なら保護効果判定の双方に現実的な精度を与える。結果を見てRを追加しない。

## 2. ランドマーク・リスク集合・転帰

pilotから変更しない。

- task switch `B ∈ {200000, 210000, ..., 800000}` の61個
- 起点 `t0=B+1`: 新しいflip状態で1回のSGD更新を経た直後
- 終点 `t1=B+10000`: 次のflip直前、同一タスク末尾
- 主リスク集合: t0で `p_hat>=0.05`（condAでは2/32以上）
- 主転帰 `end_strict_dead`: t1で `p_hat==0`
- 副転帰 `end_dead_0.05`: t1で `p_hat<0.05`
- any-hit、300k転帰、複数taskをまたぐ転帰は使わない
- 同一unitの反復曝露を許すが、unit独立のCIを作らない

## 3. 直接機能量

pilotと同じ読み取り専用float64式を使う。

- `q_i(x)=v_i relu(W_i x_in+b_i)`
- `δ(x)=yhat(x)-y(x)`
- `delta_mse_i=E[(δ-q_i)^2]-E[δ^2]=E[q_i^2-2δq_i]`
- 主機能量 `utility_nmse_i=delta_mse_i/Var(y)`
- 副機能量 `utility_raw_i=delta_mse_i`

`utility_nmse>0` はその時点・現タスクでの正の単独限界寄与だけを意味する。普遍的価値、将来タスク価値、Shapley値ではない。

## 4. 主解析: exact p̂ × 壁マージン五分位

### 4.1 幾何セル

1. `p_count := round(32*p_hat)` とし、t0のリスク集合を **`t0 × p_count`** で完全層別する
2. 各 `t0 × p_count` 内で `pre_max=max_32(W_i x_in+b_i)` を五分位にする。cutは `np.quantile(values, [0.2,0.4,0.6,0.8])`、同値規則は `bin = searchsorted(cuts, value, side='left')`
3. 幾何セルは `t0 × p_count × margin5_bin`
4. 各幾何セル内で `utility_nmse` を三分位にし、同じquantile / `side='left'` 規則で low / mid / high を付ける
5. lowまたはhighが0件のセルは主効果から除外し、セル数・除外曝露数を報告する。補完、隣接セル結合、cut変更はしない

この設計は、pilotの粗い `p̂×margin` 3×3より厳しく、死に近いほどΔLが小さくなる自明な交絡を抑えるために採る。pilotの効果が大きくなる／小さくなる方を選んだのではなく、作業6の問い「同じ生存状態・同じ壁距離」を最も直接実装する。

### 4.2 効果量

- 各セル `c` で `RD_c = risk_high - risk_low`
- 重み `w_c=min(n_high,n_low)`
- 主効果 `RD_adj=sum_c(w_c RD_c)/sum_c w_c`
- 群ラベルとセルは元のconfirmation全データで一度だけ決め、bootstrap中は固定する
- high utilityが保護される場合、RDは負

### 4.3 seedブロックbootstrap

- 20 seedを復元抽出し、選ばれたseedの全unit・全task曝露を一塊で複製する
- `B=10000`, `np.random.default_rng(20260831)`
- 各resampleで固定済みセル内の件数・event数を再集計し、§4.2のRD_adjを計算
- percentile 95% CI。有限値でないresample数を報告し、1件でもあればsanity FAIL

## 5. 主判定

意味のある差の境界を作業2から継承して `δ=0.05` とする。以下は上から順に排他的に適用する。

1. **EQUIV**: CI全体が `[−0.05,+0.05]` 内
2. **PROTECTIVE**: CI上端 `<−0.05`。有用群の凍結率が意味のある幅で低い
3. **HARMFUL**: CI下端 `>+0.05`
4. **INCONCLUSIVE**: 上記以外。0を含まないだけ、または点推定が大きいだけでは結論にしない

主転帰 `end_strict_dead` × 主機能量 `utility_nmse` × §4の五分位設計だけで主判定する。

## 6. 固定した副次解析

主結果の差し替えに使わない。

1. 同じ主セルで副転帰 `end_dead_0.05`
2. `pre_max`を十分位にした感度（他は同じ）
3. t0ごとのutility三分位だけを使う未調整RD
4. 主セルで `utility_raw` 三分位
5. utilityの符号（negative / zero / positive）別の率。zeroの定義はfloat64で厳密0
6. epoch別（200–390k / 400–590k / 600–800k）の主RD。epochを選んで全体結果を置換しない

## 7. サニティ

- C-S1: R=20、offset=20260830、width=100、T=10000、全122記録点、全61 t0/t1ペア
- C-S2: 20例の逐一消音再forwardとベクトルΔLの最大絶対誤差 `<1e-12`
- C-S3: p̂量子化誤差 `<1e-12`、`x^2+r^2=||w||^2` 最大相対誤差 `<1e-10`
- C-S4: 全記録点で `p_hat==0 ⇔ pre_max<=0`、非有限値0
- C-S5: 100k probeあり／なしの完全state・全generator hash一致
- C-S6: offset=20260830の初期hashがoffset=0と異なる。必須キー `net.W`, `net.v`, `env.flip_state`, `teacher.W`, `teacher.v` の全てが異なる
- C-S7: 主解析の有効セル、low/high各件数、seed別リスク数・event数、反復曝露分布を出力
- C-S8: 同一commit・同一NPZ入力で全CSVがbyte一致

いずれかFAILなら主判定を出さない。

## 8. 実装・出力

- config: `configs/function_blind_direct_0823_confirm.yaml`
- 計装: `src/function_blind_direct.py`（pilotと共通、modeをconfirmationとしてmetaへ保存）
- 集計: `analysis/function_blind_direct/confirm.py`
- 出力: `results/function_blind_direct_0823_confirm/`
  - `logs/seed*.npz`, `instrumentation_meta.json`, `config_used.yaml`
  - `exposures.csv`, `primary_cells.csv`, `primary_rates.csv`, `verdict.csv`
  - `secondary_results.csv`, `repeat_exposure.csv`, `sanity.csv`
  - `summary.md`, `determinism_check.md`, `meta.json`

実行は `OMP_NUM_THREADS=1` 必須。confirmation configは `require_s2=true`, `s2_steps=100000`, `require_reference=false` とする。入力・pilot成果物を上書きしない。

## 9. 解釈禁止

- PROTECTIVEなら、作業6の操作的定義の下で「選抜は機能を見ない」は否定側。柱3全体、Oの容量診断、壁機構まで否定しない
- EQUIVなら、この条件・このΔL・この同一タスク転帰に限って機能盲目性を支持する
- 観察解析なので、機能を人工的に入れ替えた因果介入とは呼ばない
- ΔLが大きいunitが壁から遠いという交絡を、完全に消したとは主張しない。§4の層別範囲だけ
- condB、他幅、他教師、長期将来、個別unitの普遍価値へ外挿しない
- pilotとconfirmationを合算しない
