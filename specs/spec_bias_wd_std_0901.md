# bias_wd_std_0901: std 腕に bias 専用 weight decay を入れる反証テスト

状態: **事前登録・未実行** / 作成: 2026-09-01 / run id: `bias_wd_std_0901`

親: `可塑性喪失/spec/HANDOFF_bias_wd_std_0901.md`
前件: `bias_wd_0901`（commit `da22465`、centered で `BIAS_WD_PROTECTS`）

---

## 1. 問いと予言

centered の死と機能劣化が bias $b$ の自走で、$\mu/\Sigma$ 駆動の本体 LoP とは
別の病気なら、std（無中心化）腕へ同じ bias 専用 weight decay（b-WD）を入れても
本体の LoP はほぼ消えないはずである。

std では

$$s = w\cdot\mu + b$$

の $\mu$ チャネルが残るので、$b$ を減衰させても壁には $M=w\cdot\mu/\sigma$
側から到達できる。また flip 15 座標と $b$ が 1 自由度に縮退する台帳の逃げ道が
あり、実効 bias を flip 側へ移せる。

**予言 P**: 主腕 `S_main` の `mean(log10 unfit)` の B10−B02 劣化は、対照
`S_none` の 50%以上残る。std でも LoP が消えた場合はこの予言の反証であり、
$\mu$ 駆動説の裁定は本実験では行わず Issa に返す。

---

## 2. 設計

- 問題: condA、`m=20`, `f=15`, teacher width 100, $\beta=0.7$
- task 長 $T=10{,}000$、batch 1、500 task = 5,000,000 step
- learner: ReLU、hidden `[100,100]`、plain SGD、`lr=0.01`、Kaiming uniform 初期化
- encoding: **std**。`centered_layers=[]` とし、中心化は一切入れない
- seed 0–9、CPU、`OMP_NUM_THREADS=1`
- task 末（10,000 step ごと、step 0 を含む 501 点）に32パターン厳密列挙
- 非有限ガードは 1,000 step ごと。checkpoint は step 0 / 5M
- `bias_wd_0901` の `setup_arm_p1` / `train_arm_p1` / レコーダを流用し、
  構築後に `VecMLPL.set_weight_decay_b` で `wd_b` だけを差し込む

### 2.1 更新式

$$b \leftarrow b - \eta\,(g_b + \lambda b)$$

decoupled WD ではなく素の L2 勾配である。両隠れ層の bias $b$ のみに掛け、
$W$・$v$・出力 bias $c$ には掛けない。分岐を置かず常に
`gb + wd_b * b` を計算する。`freeze_bias=true` と `wd_b>0` の併用は禁止する。

### 2.2 腕

新しい $\lambda$ パイロットは行わない。centered で使った主値と最強値をそのまま
使う。

| 腕 | hidden | encoding / centered_layers | `wd_b` | 役割 |
|---|---|---|---:|---|
| `S_none` | `[100,100]` | std / `[]` | 0 | 対照。既存 `L2_none` の複製 |
| `S_main` | `[100,100]` | std / `[]` | 1e-3 | **主判定** |
| `S_sub` | `[100,100]` | std / `[]` | 1e-1 | 用量反応 REPORT_ONLY |

3腕は init・教師・入力列・flip 軌道が seed 内で同一なので paired とする。

---

## 3. 記録

task 末の厳密サポートで、run ごとに `unfit`, `eval_loss_exact`、各層で次を記録する。

- `strict_dead_frac`
- alive 中央の生の $b$、$M=(w\cdot\mu)/\sigma$、$B=b/\sigma$、
  $\beta=M+B$、$\kappa$
- alive の `p_hat` 中央、`p_hat<=8/32` 率、`p_hat>=30/32` 率
- `eff_rank`, `eff_rank_W`, `w_norm_median`, `wcos_mean`

台帳移動の副解析には、第1層・第2層それぞれの alive 中央 `M_median_alive` と
`B_median_alive` の B02→B10 変化を使う。予言は b-WD 下でも M 側の沈下が続くこと。

---

## 4. 窓・床・集計

- 50 task 刻み。B02 = task 51–100、B10 = task 451–500
- 実験単位は seed。各ブロック内50 task末を seed 内で平均して一値にする
- 主 endpoint は **`mean(log10 unfit)`**。各点に深さ2系の床 `1e-23` を当ててから
  log10 を取り、ブロック内平均する
- `log10(mean unfit)` も REPORT_ONLY で出すが判定には使わない
- CI は seed 水準 paired percentile bootstrap、`B=20000`、
  **`bootstrap_seed=20260903`**
- studentized は併算し、退化検出を記録するが主 CI には用いない

seed $j$・腕 $a$ の劣化を

$$d_{a,j}=\overline{\log_{10}{\rm unfit}}_{a,j,B10}
            -\overline{\log_{10}{\rm unfit}}_{a,j,B02}$$

とし、主な劣化比を $r_j=d_{S\_main,j}/d_{S\_none,j}$ とする。分母が小さい seed
（`abs(d_S_none)<1.0 dex`）が1本でもあれば、その事実を記録し、比に加えて
対応劣化差 $d_{S\_main,j}-d_{S\_none,j}$ の CI を必ず併記する。分母の大小に
かかわらず差は常に `verdict.csv` に出す。

---

## 5. 事前登録判定

主判定に使うのは `S_main` と `S_none` のみ。判定は seed 対応の劣化比 $r_j$ の
paired percentile bootstrap CI で、次の順に一意に決める。

1. CI 下端 $\ge 0.5$ → **`LOP_PERSISTS`**（予言どおり）
2. CI 上端 $\le 0.1$ → **`LOP_REMOVED`**（予言の反証）
3. それ以外 → **`INCONCLUSIVE_PARTIAL`**

静的水準 `mean(log10 unfit)` B10 の `S_main-S_none` 対応差は REPORT_ONLY。

以下もすべて REPORT_ONLY で、主判定には使わない。

1. B10 の `strict_dead_frac`
2. 各層の alive 中央 M/B チャネルの B02→B10 推移
3. `S_sub` の劣化比・劣化差・M/B 推移
4. `mean(log10 unfit)` B10 の腕間差

---

## 6. サニティ

| ID | 内容 | 失敗時 |
|---|---|---|
| S0 | `S_none` を committed `mlp2_phase1_0829/L2_none` に対して30k step・1k格子で replay。`unfit`・`eval_loss_exact`・各層 `strict_dead_frac` が一致 | 本走禁止 |
| S1/S2 | 前件の代数テストを流用。$\lambda=0$ は無WD経路と bitwise 一致し、$\lambda>0$ が触るのは隠れ層 $b$ のみ | 本走禁止 |
| S3 | 壁恒等式、1/32量子化、第1層 $\kappa$ 閉形式、有限性、独立実装一致。$\beta$ は前件で修正済みの `max|a-b|/max|b_ref|` 尺度（許容 1e-10） | 本走禁止 |
| S4 | 1,000 step ごとの非有限ガード。発散した腕だけ停止し、他腕は継続 | 当該腕のみ停止 |

std では高 $\lambda$ でも $\mu$ 経由で dead になれるため、centered 前件の S5
恒真ガードは置かない。

---

## 7. 引いてはいけない線

1. `LOP_PERSISTS` でも「b-WD は無意味」と書かない。centered では効いており、
   regime が違う
2. `LOP_REMOVED` でも $\mu$ 駆動説の棄却まで飛ばない。裁定は Issa に返す
3. `strict_dead_frac` の変化を機能改善・悪化と読み替えない
4. `S_sub` を見て主 $\lambda$ を選び直さない

---

## 8. 成果物と commit 規律

```
specs/spec_bias_wd_std_0901.md
configs/bias_wd_std_0901.yaml
src/bias_wd_std_0901.py
results/_gate_bias_wd_std_0901/
results/bias_wd_std_0901/
    verdict.csv  summary.md  paired_endpoints.csv  task_end_metrics.csv
    block_levels.csv  run_sanity.json  provenance.json  fig_bias_wd_std.png
```

commit は **spec 単独 → config+実装 → 結果** の3段。各段で
`git ls-remote origin refs/heads/main` により push を確認する。数値は
`verdict.csv` / `summary.md` からのみ報告し、窓を必ず明記する。
