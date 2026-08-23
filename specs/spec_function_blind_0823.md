# spec_function_blind_0823: 死の機能盲目性（同時刻ハザード＋オラクル bias 修復）

proj_004 / 作成 2026-08-23 / 対象リポジトリ: lop_analysis / **再学習なし**

> **承認**: 中心主張 v3 は 2026-08-23 に承認済み。本 commit を解析実装前の仕様固定点とし、commit 後に実装・実行する。
>
> **盲検性の毀損**: 本解析に対応する使い捨て事後計算は既に実行され、概数も vault の `中心主張v3草案_0823.md` と `中心主張v3作業リスト_0823.md` に記録されている（ハザード 0.837 / 0.839 / 0.850、修復 unfit 0.143 → 0.009、対照 0.0001 / 0.070 / 0.081 / 0.196）。したがって本 spec は盲検事前登録ではなく、**コード・判定規則・出力を固定した再現解析**である。独立な確認とは呼ばない。

---

## 0. 一行

同じ時刻・同じ生存条件で入力応答重み `r` が大きいユニットほど死ににくいかを測り、さらに死者の `W` を固定したまま bias と出力オフセットだけで損失を回復できるかを測る。前者は選抜則が機能代理 `r` を参照するか、後者は落とされた表現が壊れていたかを問う。

## 1. 入力（read-only）

### H: 同時刻ハザード

- `results/ratchet_log_0819/logs/seed{0..9}.npz`
- condA・w100・T=10,000・batch=1・1M step
- 各 seed の `step`, `cos_u_mu`, `w_norm`, `p_hat`, `flip_state`
- 記録グリッドは bulk 1000 step と境界 ±100 の毎 stepが混在する。転帰は**保存された記録点上で初めて条件を満たしたこと**として定義し、記録点間を補間しない

### O: オラクル bias 修復

- `results/posreset_0819/snapshots/A_w100_cont_step500000.pt`
- 10 seed を並列に持つ condA w100 のスナップショット
- `net.{W,b,v,c}`, `env.flip_state`, `teacher.{W,b,v,cout,tau}` のみを読む
- 現タスクの入力は `[flip_state(15) | rnd(5)]` の 32 パターンを全列挙し、教師・学習器ともサンプリング誤差なしで評価する

入力ファイルは一切書き換えない。

## 2. 共通する死の用語（8/23 裁定を採用）

- **凍結（`strict_dead`）**: `p_hat == 0`
- **ほぼ消灯（`near_off`）**: `0 < p_hat < 0.05`。condA では `p_hat = 1/32`
- **旧作業指標（`dead_0.05`）**: `p_hat < 0.05`。凍結とほぼ消灯の和

旧事後値の再現対象は `dead_0.05` だが、不可逆性を含意しない。`strict_dead` を必ず副次解析として併記する。

## 3. H: 同時刻ハザード層別

### 3.1 起点・共変量・転帰

- 起点: `t0 in {200000, 300000, 400000, 500000, 600000}`
- 主リスク集合: t0 で `p_hat >= 0.05` の seed × unit
- 追跡窓: `(t0, t0 + 300000]`
- 主転帰 `event_dead_0.05`: 追跡窓内の保存記録点で一度でも `p_hat < 0.05`
- 副転帰 `event_strict_dead`: 同じ主リスク集合が一度でも `p_hat == 0`
- `x = w_norm * cos_u_mu`
- `r = w_norm * sqrt(max(0, 1 - cos_u_mu**2))`

同一 seed × unit が複数 t0 に入ることを許す。これは5つのランドマーク時点での反復曝露であり、独立標本として SE を計算してはいけない。

### 3.2 r 三分位

- 各 t0 の主リスク集合を10 seedでプールし、`r` を `low / mid / high` の三分位に分ける
- `r` は連続値なので通常の 1/3, 2/3 quantile cut を使う。同値が cut に乗った場合は `low: r <= q1`, `mid: q1 < r <= q2`, `high: r > q2` とする
- 粗死亡率は全 t0・seed・unit 曝露をプールして `low / mid / high` の順で報告し、既知の 0.837 / 0.839 / 0.850 の再現対象とする

### 3.3 依存を尊重した不確実性

- 点推定: 各 seed 内で5起点をプールした群別死亡率を計算し、10 seed を等重み平均
- 主効果: `RD_high-low = risk(high) - risk(low)`
- seed 単位の paired bootstrap。10 seed を復元抽出し、選ばれた seed の全 unit・全 t0 をブロックとして複製する
- `B=10000`, `np.random.default_rng(20260823)`, percentile 95% CI
- **機能盲目性の等価基準**: `RD_high-low` の 95% CI 全体が `[-0.05, +0.05]` に入れば **EQUIV**。CI が 0 を含むだけなら **INCONCLUSIVE** であり「無相関」と書かない
- CI 上端 `< 0` なら `r` の保護効果（PROTECTIVE）、CI 下端 `> 0` なら逆向き（HIGHER_R_HIGHER_HAZARD）

主判定は `event_dead_0.05`。`event_strict_dead` に同じ手続きを適用するが副次で、主判定の差し替えには使わない。

### 3.4 p_hat・x の 3×3 統制

- 各 t0 の主リスク集合内で、`p_hat` と `x` をそれぞれ三分位に分け、`p_bin × x_bin` の9セルを作る
- 各セルについて r 三分位別の `n_exposure`, `n_event`, `risk` を報告する（3×3セル × r 3群）
- 各セルの `RD_high-low` を報告する。low または high が0件のセルは NA とし補完しない
- 調整要約は、各セルの low/high の小さい方の標本数を重みとする加重平均 `RD_adj`。seed ブロック bootstrap で95% CIを付ける
- `RD_adj` は補強解析であり、H の PASS/FAIL は §3.3 の主効果だけで決める。セル内の一部だけを選んで主張しない

## 4. O: オラクル bias 修復

### 4.1 厳密評価と unfit

- 学習器: `yhat = sum_i v_i * relu(W_i x + b_i) + c`
- 教師: `h_j = 1[(W^T_j x + b^T_j) >= tau_j]`, `y = sum_j v^T_j h_j + cout`
- 各 seed の32パターン上で評価する
- 主指標 `unfit_var = Var(yhat - y, ddof=0) / Var(y, ddof=0)`
- 副指標 `nmse = Mean((yhat-y)^2) / Var(y, ddof=0)`
- seed ごとの値と10 seed中央値を報告する

### 4.2 主修復（局所スイッチ診断）

- 主対象: スナップショットで `dead_0.05` のユニット
- 各対象ユニットの32パターン上の現在の preactivation 最大値を `zmax_i` とする
- primary kick `k=0.5`: `b_i <- b_i + (k - zmax_i)`。これにより対象の最良パターンで preactivation が厳密に k になる
- 対象外の b は現在値を保持する
- その後、**全 hidden bias b と出力 offset c のみ**を最適化する。`W` と `v` は固定し、hash 不変を検査する
- 感度: `k in {0.1, 0.25, 0.5, 1.0}`。k=0.5だけが主判定で、他は結果を見て差し替えない
- 副次: `strict_dead` のみを蹴る版を同じ k 集合で実行する

### 4.3 W 対照4本（容量診断）

各対照は `b=0`, `c=mean(y)` から開始し、§4.4 と同じく b と c だけを最適化する。`v` はスナップショットのまま固定する。

1. **learned**: 学習済み W
2. **fresh_he**: W の全要素を `U(-sqrt(6/20), +sqrt(6/20))` から新規生成
3. **row_shuffle**: seed ごとに W の100行を置換なしで並べ替え、v は元の unit 順のまま
4. **rnd_randomized**: W の flip 15列は保持し、rnd 5列だけを同じ He 分布から新規生成

乱数は `np.random.default_rng(20260823 + seed)` を使い、各 seed で `fresh_he → row_shuffle → rnd_randomized` の順に消費する。一つの決定的 realization を主結果とする。これは手法比較ではなく、学習済み W の構造を破壊する負対照である。

### 4.4 最適化（固定）

- torch float64、CPU、全32パターン、乱数ミニバッチなし
- Adam、`lr=0.03`, `steps=20000`
- seedごとの `MSE / Var(y)` を等重み平均した目的関数
- 各 seed について全 step 中の最良パラメータを保持し、最終出力には最良値を使う
- 非有限値が1つでも出たらそのアームを中止し FAIL_SANITY
- 初期値より最良目的が悪い場合は FAIL_SANITY
- 追加 restart、学習率変更、早期終了による結果の差し替えは禁止。収束診断として 5,000 / 10,000 / 20,000 step の指標を保存する

### 4.5 判定

- **O1 RECOVER**: primary repair（dead_0.05, k=0.5）の10 seed中央値で `1 - unfit_repair/unfit_current >= 0.90`
- **O2 INFORMATIVE_W**: paired seed bootstrapで `unfit_learned - unfit_control` の95% CI上端が、3つの破壊対照すべてに対して `< 0`
- bootstrapは seed単位、`B=10000`, RNG seed `20260824`
- O1 と O2 がともに通れば「スイッチを開けば利用可能な学習済み W が残る」と書いてよい
- 「dead unit単体が教師に有用」「この修復が学習手法として有効」「再開後も改善が持続する」とは書かない

## 5. サニティ

### H

- H-S1: 10 seed、各 width=100、step 0..1M、全 t0 と t0+300k が記録点に存在
- H-S2: `p_hat` が k/32 に量子化（最大誤差 < 1e-7）
- H-S3: `r` が有限かつ非負、`x**2 + r**2 == w_norm**2` の最大相対誤差 < 1e-6
- H-S4: 同一個体の反復曝露数分布と、seed別・t0別のリスク集合数を出力

### O

- O-S1: snapshot step=500000、R=10、h=100、d=20、flip=15、32パターン
- O-S2: 手実装の current yhat が `VecMLP` の式と一致（最大絶対誤差 < 1e-10）
- O-S3: current `p_hat` が k/32 に量子化
- O-S4: 最適化前後で W と v の byte hash が不変
- O-S5: kick 後、対象ユニットの `max(pre)` と k の最大誤差 < 1e-10
- O-S6: 同一コマンドの再実行で CSV 数値が全桁一致

## 6. 実装・出力

- 実装: `analysis/function_blind/function_blind.py`
- 実行:

      OMP_NUM_THREADS=1 .venv/bin/python -m analysis.function_blind.function_blind \
        --logs results/ratchet_log_0819/logs \
        --snapshot results/posreset_0819/snapshots/A_w100_cont_step500000.pt \
        --outdir results/function_blind_0823

- 出力:
  - `meta.json`: git hash、入力 sha256、環境、RNG、経過時間、sanity
  - `hazard_exposures.csv`: seed × unit × t0 の曝露、共変量、群、転帰
  - `hazard_rates.csv`: r群別の粗率・seed等重み率・CI
  - `hazard_cells_3x3.csv`: p_hat × x セル × r群
  - `hazard_verdict.csv`: 主・副転帰の RD、CI、判定
  - `oracle_per_seed.csv`: current、修復k感度、4対照の指標
  - `oracle_trace.csv`: 0 / 5000 / 10000 / 20000 step の収束診断
  - `oracle_verdict.csv`: O1・O2
  - `summary.md`: sanity → H → 3×3 → O → 留保 → 禁止事項
  - `figures/`: ハザード率、3×3 RD、修復感度、対照 unfit

## 7. スコープ・禁止事項

- condA・w100・T=1e4・batch=1・10 seed・既存の2データ源だけ。condB、他時刻、他幅へ外挿しない
- H は保存記録点上の300k累積転帰であり、連続時間ハザードではない
- H は一時的な消灯を含む。境界を越えた再点灯があるため `event_dead_0.05` を不可逆死と呼ばない
- r は入力応答重みの大きさの代理であり、教師への因果的寄与そのものではない
- O はオラクル容量診断。動的再学習を含まず、手法提案ではない
- null差を「同一」と書くには §3.3 の EQUIV が必要
- 既知の事後概数に合わない結果もそのまま保存し、specや実装を結果に合わせて変更しない。実装バグを直した場合は commit と逸脱欄に記録する
