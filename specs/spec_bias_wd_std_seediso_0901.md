# bias_wd_std_seediso_0901: std b-WD の seed 単位隔離再走

状態: **事前登録・未実行** / 作成: 2026-09-01 / run id: `bias_wd_std_seediso_0901`

親: `bias_wd_std_0901`（commit `af70722`）
理由: 前走では `S_main` seed 7 と `S_sub` seed 2 が数値発散し、腕単位停止のため
主 endpoint を計算できなかった。本再走では、非有限 seed のみを独立停止・独立除外し、
残りの対応 seed で同じ反証テストを完遂する。

---

## 1. 据え置く設計

以下は `specs/spec_bias_wd_std_0901.md` から変更しない。

- condA: `m=20`, `f=15`, teacher width 100, beta 0.7
- learner: ReLU、hidden `[100,100]`、plain SGD、`lr=0.01`、batch 1
- encoding: std、`centered_layers=[]`
- 5,000,000 step = 500 task、seed 0–9、CPU、`OMP_NUM_THREADS=1`
- task末の32パターン厳密記録、非有限 probe は1,000 stepごと
- 腕と lambda:

| 腕 | `wd_b` | 役割 |
|---|---:|---|
| `S_none` | 0 | 対照 |
| `S_main` | 1e-3 | 主判定 |
| `S_sub` | 1e-1 | REPORT_ONLY |

- B02 = task 51–100、B10 = task 451–500
- 主 endpoint: 床 `1e-23` を各点に当てた `mean(log10 unfit)` の B10−B02
- 判定境界:
  - 劣化比の95% CI下端 >= 0.5 → `LOP_PERSISTS`
  - 劣化比の95% CI上端 <= 0.1 → `LOP_REMOVED`
  - それ以外 → `INCONCLUSIVE_PARTIAL`
- paired percentile bootstrap、B=20000。新しい bootstrap seed は
  `20260904`（前走の `20260903` を再利用しない）
- dead、M/B台帳、`S_sub`、B10静的差は REPORT_ONLY

---

## 2. seed 単位隔離

### 2.1 検出と停止

各腕は従来どおり10 seedを対応づけたベクトル軌道で走らせる。1,000 stepごとの
probeで seed ごとに `W1/W2/b1/b2/v/c` の有限性を調べる。

ある seed に非有限が見つかった場合:

1. 当該 seed だけを `NUMERIC_DIVERGENCE` として停止する
2. 検出 step・task・非有限 tensor を `seed_status/<arm>_seed<seed>.json` に記録する
3. 当該 seed の全時点データをその腕の解析から除外する（発散前の部分軌道も使わない）
4. 他 seed の学習・記録は継続する

乱数対応を崩さないため、隔離後も環境入力の10行分の乱数消費は継続する。停止 seed の
学習率を0にし、学習器状態を有限なゼロへ quarantine する。学習演算は seed 次元間で
縮約しないため、非停止 seed の軌道は隔離なしの同一腕と bitwise 同一でなければならない
（S-isoで検査）。停止 seed の quarantine 後の値は解析・ログに使用しない。

### 2.2 除外上限

- **各腕2/10 seedまで除外可**
- 3本目の seed が非有限になった時点で腕を停止し、
  `ARM_INVALID_EXCLUSION_LIMIT` とする
- 有効腕は完走 seed が8本以上あることを要する
- 主比較は `S_none` と `S_main` の完走 seed の**共通集合**だけを使う
- 共通集合が8本未満なら `CONTRAST_INVALID_TOO_FEW_PAIRED` とし、主判定を出さない
- `S_sub` の記述比較も `S_none` との完走 seed 共通集合を使う

除外 seed の補充・seed追加・途中再開・rescue・値のwinsorizeは行わない。

---

## 3. 集計

seed j・腕 a の劣化を

`d[a,j] = mean_log10_unfit[a,j,B10] - mean_log10_unfit[a,j,B02]`

とする。主比較の有効な共通 seed 集合を J とし、seed対応比

`r[j] = d[S_main,j] / d[S_none,j]`, j in J

の平均に paired percentile bootstrap CI を付ける。前走と同じく
`abs(d[S_none,j]) < 1.0 dex` を小分母として記録し、対応劣化差
`d[S_main,j] - d[S_none,j]` のCIを常に併記する。

腕別のdead・台帳・静的水準は、その腕の完走 seed 全体で集計する。腕間差は対応する
完走 seed 共通集合を使う。

---

## 4. サニティ

| ID | 内容 | 失敗時 |
|---|---|---|
| S0 | 隔離runnerの `S_none` を committed `L2_none` と30k・1k格子で比較。除外0、`unfit`・`eval_loss_exact`・dead一致 | 本走禁止 |
| S1/S2 | lambda=0 identity、WDが隠れ層biasだけを触る | 本走禁止 |
| S-iso | seed 1へ人工的に非有限を注入し、隔離後100 stepで他9 seedの全学習器状態・入力状態が対照runnerとbitwise一致 | 本走禁止 |
| S-cap | 人工的に3 seedを非有限化し、3本目で `ARM_INVALID_EXCLUSION_LIMIT` になる | 本走禁止 |
| S3 | 壁恒等式、1/32量子化、第1層kappa閉形式、独立実装一致。betaはスケール正規化尺度 | 本走禁止 |
| S4-iso | 実走のseed別除外イベントと腕ごとの除外数 | 規則どおり隔離または腕無効 |

---

## 5. 成果物・規律

```
specs/spec_bias_wd_std_seediso_0901.md
configs/bias_wd_std_seediso_0901.yaml
src/bias_wd_std_seediso_0901.py
results/_gate_bias_wd_std_seediso_0901/
results/bias_wd_std_seediso_0901/
    verdict.csv  summary.md  paired_endpoints.csv  exclusions.csv
    task_end_metrics.csv  block_levels.csv  run_sanity.json
    provenance.json  fig_bias_wd_std_seediso.png
```

commit は **spec単独 → config+実装 → 結果** の3段。各段で
`git ls-remote origin refs/heads/main` によりpushを確認する。

数値は `verdict.csv` / `summary.md` からのみ報告する。前走の主判定不能を、除外seedの
部分軌道で埋めない。
