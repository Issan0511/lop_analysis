# shell_l2_rlmnist_0913 — 判定（Random Label MNIST・Shell 正則化で L2-Init の利得を分解）

> 自動生成: `analysis/shell_l2_rlmnist_0913/verdict.py`。spec: `specs/spec_shell_l2_rlmnist_0913.md`。数値はこのファイルと `verdict.csv` から転記する。

## 0. 実行したもの

- **本走の commit**: `f83d2469aeeb42a264752cb4a14a058fb02d5bc7`（run 時の code の未 commit 変更: True）
- 集計時の HEAD: `f83d2469aeeb42a264752cb4a14a058fb02d5bc7`（branch `exp/shell_l2_rlmnist_0913`）
- 環境: host `claude-shell0913`・GCP `c2d-standard-32`（`asia-northeast1-b`）・CPU AMD EPYC 7B13・Python 3.12.14・torch 2.13.0+cpu・numpy 2.5.1・pandas 3.0.5・各 run 1 スレッド（`torch.set_num_threads(1)`・`OMP_NUM_THREADS=1`）・CPU
- run: **16/80 揃い**・欠損 64・発散 0・1 run の壁時計 中央値 0.0 分（最大 0.0 分）
- 検査: `checks.json` all_pass = **True**

## 1. 結論（活性化ごとに独立・spec §4.4 の表を上から適用）

| act | ラベル | 理由コード | Bonferroni 版（α=0.025） | 結論 |
|---|---|---|---|---|
| **R** | **D** | INCOMPLETE | D INCOMPLETE | 方向効果は未同定。性能比較のみ確定 |
| **SNA** | **D** | INCOMPLETE | D INCOMPLETE | 方向効果は未同定。性能比較のみ確定 |

限定: **Random Label MNIST・784–100–100–10・Adam lr=1e−3・λ=1e−3・50 タスク・seed 0–9・CPU の、この箱の中だけの結論**。Shell は tensor ごとの半径の拘束で、ユニットごとのノルムや hard projection については何も言わない。

- **活性化を跨ぐ文（「R でも SNA でも〜」）**: **書いてよい**（両ラベル一致かつ Bonferroni 版でも同じ）

- R の INCOMPLETE の理由: R_none_s0: window tasks 31-50 incomplete or non-finite; R_none_s0: layer_metrics window rows 0 != 120; R_none_s1: window tasks 31-50 incomplete or non-finite; R_none_s1: layer_metrics window rows 0 != 120; R_none_s2: missing; R_none_s3: missing; R_none_s4: missing; R_none_s5: missing; R_none_s6: missing; R_none_s7: missing; R_none_s8: missing; R_none_s9: missing …
- SNA の INCOMPLETE の理由: SNA_none_s0: window tasks 31-50 incomplete or non-finite; SNA_none_s0: layer_metrics window rows 0 != 120; SNA_none_s1: window tasks 31-50 incomplete or non-finite; SNA_none_s1: layer_metrics window rows 0 != 120; SNA_none_s2: missing; SNA_none_s3: missing; SNA_none_s4: missing; SNA_none_s5: missing; SNA_none_s6: missing; SNA_none_s7: missing; SNA_none_s8: missing; SNA_none_s9: missing …

## 2. 4 腕の成績（窓 = タスク 31–50 の `online_acc`・seed ごとの窓平均を seed 間で要約）

### R

| reg | n | 平均 ± SD | 中央値 | 最小–最大 | memo_acc 平均 |
|---|---|---|---|---|---|
| `none` | 0 | — ± — | — | —–— | — |
| `l2` | 0 | — ± — | — | —–— | — |
| `l2init` | 0 | — ± — | — | —–— | — |
| `shell` | 0 | — ± — | — | —–— | — |

### SNA

| reg | n | 平均 ± SD | 中央値 | 最小–最大 | memo_acc 平均 |
|---|---|---|---|---|---|
| `none` | 0 | — ± — | — | —–— | — |
| `l2` | 0 | — ± — | — | —–— | — |
| `l2init` | 0 | — ± — | — | —–— | — |
| `shell` | 0 | — ± — | — | —–— | — |

## 3. 対応差（seed 対応・bootstrap は seed の復元抽出 10,000 回）

### R

| 対比 | 平均 ± SE | 95% CI | 97.5% CI | 勝ち/n | 両側 sign p | 分類（α=0.05） | 分類（Bonf） |
|---|---|---|---|---|---|---|---|
| d1 = `shell` − `l2` | — ± — | [—, —] | [—, —] | 0/0 | nan | UNRESOLVED | UNRESOLVED |
| d2 = `shell` − `l2init` | — ± — | [—, —] | [—, —] | 0/0 | nan | AMBIG | AMBIG |
| d3 = `l2init` − `l2` | — ± — | [—, —] | [—, —] | 0/0 | nan | UNRESOLVED | UNRESOLVED |

同等性帯 δ = ±0.005（d2 のみ）。A には d3 の 95% CI 下限 ≥ 0.010 も要る（spec §4.4 の解釈 2）。

### SNA

| 対比 | 平均 ± SE | 95% CI | 97.5% CI | 勝ち/n | 両側 sign p | 分類（α=0.05） | 分類（Bonf） |
|---|---|---|---|---|---|---|---|
| d1 = `shell` − `l2` | — ± — | [—, —] | [—, —] | 0/0 | nan | UNRESOLVED | UNRESOLVED |
| d2 = `shell` − `l2init` | — ± — | [—, —] | [—, —] | 0/0 | nan | AMBIG | AMBIG |
| d3 = `l2init` − `l2` | — ± — | [—, —] | [—, —] | 0/0 | nan | UNRESOLVED | UNRESOLVED |

同等性帯 δ = ±0.005（d2 のみ）。A には d3 の 95% CI 下限 ≥ 0.010 も要る（spec §4.4 の解釈 2）。

## 4. norm-match ガード（W1–W3・タスク 31–50 × seed 0–9 のタスク末 ‖p‖/‖p0‖ の中央値の比 ∈ [0.9, 1.1]）

### R: **FAIL**

| tensor | M(shell) | M(l2init) | 比 | 判定 |
|---|---|---|---|---|
| W1 | — | — | — | FAIL |
| W2 | — | — | — | FAIL |
| W3 | — | — | — | FAIL |
| b1（REPORT） | — | — | — | (out) |
| b2（REPORT） | — | — | — | (out) |
| b3（REPORT） | — | — | — | (out) |

### SNA: **FAIL**

| tensor | M(shell) | M(l2init) | 比 | 判定 |
|---|---|---|---|---|
| W1 | — | — | — | FAIL |
| W2 | — | — | — | FAIL |
| W3 | — | — | — | FAIL |
| b1（REPORT） | — | — | — | (out) |
| b2（REPORT） | — | — | — | (out) |
| b3（REPORT） | — | — | — | (out) |

## 5. REPORT_ONLY（判定に使わない）

### 5.1 `none` との対比

| act | 対比 | 平均 ± SE | 95% CI | 勝ち/n | p |
|---|---|---|---|---|---|
| R | `l2` − `none` | — ± — | [—, —] | 0/0 | nan |
| R | `l2init` − `none` | — ± — | [—, —] | 0/0 | nan |
| R | `shell` − `none` | — ± — | [—, —] | 0/0 | nan |
| SNA | `l2` − `none` | — ± — | [—, —] | 0/0 | nan |
| SNA | `l2init` − `none` | — ± — | [—, —] | 0/0 | nan |
| SNA | `shell` − `none` | — ± — | [—, —] | 0/0 | nan |

### 5.2 shell が取り戻した L2-Init の利得の割合 f = d1 / d3

- R: f = —（95% CI [—, —]・d3 が小さいと不安定）
- SNA: f = —（95% CI [—, —]・d3 が小さいと不安定）

### 5.3 tensor ごとの窓内中央値（seed × タスク 31–50 をまとめた中央値）

| act | reg | tensor | ‖p‖/‖p0‖ | cos(p,p0) | 自腕の罰則 | L2-Init 罰則に占める半径成分 |
|---|---|---|---|---|---|---|

### 5.4 ε の不活性（全記録点の sq_norm の最小 > 3.36e-05 なら ε² は 1 bit も効いていない）

- R `none`: min S = 0.02433（b3・task 1・seed 1）→ 不活性
- R `l2`: min S = 0.02299（b3・task 2・seed 1）→ 不活性
- R `l2init`: min S = 0.02155（b3・task 2・seed 1）→ 不活性
- R `shell`: min S = 0.0218（b3・task 2・seed 1）→ 不活性
- SNA `none`: min S = 0.02248（b3・task 2・seed 1）→ 不活性
- SNA `l2`: min S = 0.02082（b3・task 2・seed 1）→ 不活性
- SNA `l2init`: min S = 0.02162（b3・task 2・seed 1）→ 不活性
- SNA `shell`: min S = 0.02242（b3・task 2・seed 1）→ 不活性

### 5.5 0906（CUDA・別の走）の窓平均 — 参考。bit 一致は求めず、主判定に使わない

| act | reg | 0906 平均（n） | 本走 平均（n） |
|---|---|---|---|
| R | `none` | 0.1140（10） | —（0） |
| R | `l2` | 0.9410（10） | —（0） |
| R | `l2init` | 0.9581（10） | —（0） |
| SNA | `none` | 0.9859（10） | —（0） |
| SNA | `l2` | 0.9643（10） | —（0） |
| SNA | `l2init` | 0.9668（10） | —（0） |

## 6. 検査（`checks.json`）

- **S1_S-identical**: PASS・mutation 4/4 検出
- **S2_S-shell-zero-init**: PASS・mutation 1/1 検出
- **S3_S-shell-grad-parallel**: PASS・mutation 4/4 検出
- **S4_S-l2-unchanged**: PASS・mutation 2/2 検出
- **S5_S-repro**: PASS・mutation 1/1 検出
- **S6_S-zero0-rule**: PASS・mutation 2/2 検出
- **S7_S-eps-inert**: PASS・mutation 1/1 検出
- **S8_S-shell-wiring**: PASS・mutation 2/2 検出
- **S9_S-log**: PASS・mutation 3/3 検出
- **S10_S-verdict-synthetic**: PASS・mutation 4/4 検出
