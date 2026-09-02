# spec_valley_off_0903 — 谷の逃走・走 A（オラクルなしの自然な condA で GELU・SiLU は逃げるか）

Obsidian 側の正本: `可塑性喪失/spec/谷の逃走_走A_spec_0903.md`（v1・2026-09-03・Kubo 起案）。
親主張: `可塑性喪失/主張/到達と離脱_統合主張_0903.md` §4。
本ファイルは **実装より先に config と一緒に単独 commit する** repo 側正本であり、
vault の §1・§2・§4 の逐語の写しに実装上の決めを足したものである。段 2（走 B `G2_b1` /
`S2_b1`・2 層）は **本 spec の対象外**（Issa の例外扱いが要る。vault §3）。

## 1. 問い

現在 GELU・SiLU の逃走（負側の谷まで沈み、そこから戻れない）は **オラクル用量 12.16 の
1 点**（`gate_dial_0902`）でしか見ていない。「大きな平均シフトを外から掛けたから谷まで
押し込まれただけだ」という反論を退けられない。**オラクルを外した自然な condA
（`gate_dose_0830` の `_off` 腕の機構）でも同じことが起きるか**を測る。

走 A が通らなければ、統合主張 §4 の目玉は「強い平均シフトの下で」に限定される。

## 2. 設計

* 腕（新規 2 本）: `G_off`（GELU・ダイヤル β=1）・`S_off`（SiLU・ダイヤル β=1）。
* 機構: `gate_dose_0830` の `_off`（`target_dose: null`・`target_mu_norm: null`・
  `centered_layers: []`・オラクルなし。互換 EMA の running mean は更新されるが
  入力からは引かれない）に、`gate_dial_0902` の閉形式 SiLU/GELU（真の導関数）を載せる。
* 1 層・幅 100・5M step・seed 0–9・batch 1・lr 0.01・CPU。
* `generator_offset: 0`（明示）。親走と同一系列で init・教師・入力実現・flip が bit 一致
  すること（S-pair）が設計の土台なので、ここで系列を切ってはいけない。
* ロガーは `gate_dial_0902` の `DialRecorder`（`mob` / `absmob` / `zmax` / `zmean` /
  `v_unit`）を継承。`_off` 腕は `centered_layers` が空なので、記録される `z` は
  **中心化前の生の $z$** である（vault §2-4 の疑い (3)）。
* 対照は再走しない: `gate_dose_0830` の committed 出力（`R_off` / `E_off` / `LR_off`）。
  主 endpoint $U$ は `verdict.csv` から**転記**し、ユニット別量だけ `logs/*.npz` を読む。
  対照 logs には `mob` / `zmax` 列が無いので、沈下は `p_hat == 0`、凍結は
  `p_hat < 1e-6`（ReLU では $\mathbb E_x\varphi' = \hat p$ なので同じ量）で作り、
  **出所と代用であることを必ず添える**。

## 3. 記号・窓・床

vault §1 と `gate_dose_0830` §5 の逐語継承。

* 窓は**タスク終端の記録点のみ**（`step % 10000 == 0`）で 10 点。1M 窓 = tasks 91–100、
  5M 窓（末尾窓）= tasks 491–500、early 窓 = tasks 2–11。
* $U$ = 窓内の `unfit` 平均。床 $10^{-16}$（`dose_const_5m_0830` で較正済み・再較正しない）。
* 発症 = $U \ge 0.05$。
* 沈下 = $\max_x z_i \le 0$、深さ = $-\bar z_i$、
  凍結 = $\lvert\mathbb E_x\varphi'(z_i)\rvert < 10^{-6}$（**本走のロガー `layer1_mob`**。
  `u_fr` を使う `gate_dial_0902` の定義とは別物なので、引用時に出所を添える）。
* 谷底 $z_c = -u^\ast/\beta$: GELU β=1 で $-0.7519$、SiLU β=1 で $-1.2785$。
  谷の向こう = $\max_x z_i \le -u^\ast$。
* CI: percentile ブートストラップ（B=10000・`bootstrap_seed: 20260912`・未使用日付）。

## 4. 既知の対照値（`gate_dose_0830`・2026-09-03 に正本で照合）

| 腕 | 5M 発症 | 5M median $\log_{10}U$ | 1M 発症 | 1M median $\log_{10}U$ |
| --- | --- | --- | --- | --- |
| `R_off` | 10/10 | −0.0930 | 9/10 | −0.6748 |
| `E_off` | 0/10 | −1.9704 | 0/10 | −2.1641 |
| `LR_off` | 0/10 | −2.2755 | 0/10 | −3.2451 |

**5M では ReLU が天井（median $U$ 0.81）にいるので「ReLU より悪い」は 5M では検定できない。
1M 窓で検定する。**

## 5. 事前予測（Kubo・走の前に固定・vault §2-3 の逐語）

| # | 予測 | 自信 |
| --- | --- | --- |
| A1 | `G_off` は 5M で 10/10、1M で ≥8/10 | 高 |
| A2 | **1M 窓**の paired $\Delta\log_{10}U$（`G_off` − `R_off`）が ≥ +0.2 dex、符号 ≥ 8:2。5M は天井のため報告のみ | 中 |
| A3 | `G_off` の 5M 末尾窓: 沈下率 ≥ 0.99、深さ中央値 ≥ 8（`R_off` より深い・paired 10:0）、凍結率 ≥ 0.9 | 高 |
| A4 | `S_off` は 5M で ≥5/10、深さ中央値 ≥ 5、沈下率 ≥ 0.95 | 低 |
| A5 | `G_off` でも谷の向こうからの同一タスク内脱出が起きる（率 $10^{-3}$ 台/記録） | 中 |

A1 の 1M 側（≥8/10）は `R_off` 自身が 1M で 9/10 なので **ReLU との差を含意しない**。
ReLU との差は A2（paired 差）だけが検定する。

## 6. 判定ラベル（vault §2-4）

`G_off` について、**registered order** で最初に当たった行を主ラベルとし、
満たした行はすべて `co_satisfied` に残す（`bwd_leak_0902` 追補 7 の慣行）。

| 条件 | ラベル |
| --- | --- |
| `G_off` の 5M 発症 ≤ 2/10 | `FLIGHT_NEEDS_OFFSET` |
| A1 ∧ A3 ∧ ¬A2 | `WORSE_UNTESTABLE_AT_CEILING` |
| A1 ∧ A3 | `FLIGHT_WITHOUT_ORACLE` |
| A3 ∧ 5M 発症 3–7/10 | `FLIGHT_SLOW` |
| それ以外 | `PARTIAL` |

registered order = `[FLIGHT_NEEDS_OFFSET, WORSE_UNTESTABLE_AT_CEILING,
FLIGHT_WITHOUT_ORACLE, FLIGHT_SLOW, PARTIAL]`。
A4・A5 と `S_off` の全量は REPORT_ONLY でラベルに入らない。

**外れたときに第一に疑うもの**（順に）: (1) 自然な condA の用量が谷底に届く圧力を持たない
（`R_off` の 5M `layer1_dose` と深さ分布を先に見る）、(2) S-pair が `R_off` と切れている、
(3) ロガーの `zmax` の座標（`_off` は中心化しないので生の $z$）。

## 7. 前段チェック

| 名 | 内容 |
| --- | --- |
| S1 | `OMP_NUM_THREADS=1` |
| S-dial | 登録した $u^\ast$ / $u_{fr}$ が数値解と一致（相対 6%・登録値は 2 桁丸め） |
| S-fd | SiLU/GELU の閉形式後ろ向きと float64 中心差分（深い裾は erfc 参照形）。活性化は `gate_dial_0902` のダイヤル 1.0 と同一なので、同モジュールの検査を同じ config で回す |
| S-num | 全域で NaN/inf 無し・float32 で厳密 0 になる深さを記録（判定に使わない） |
| S-limit | β→大 で ReLU に寄る（許容つき）・$\varphi'(0)=1/2$ |
| S-mob | **本走の `_off` 経路で** `mob == p_hat`（ReLU 腕）・`zmean == (M+B)*denom`・`zmax >= zmean` |
| S-log-b | ユニット別ロガーの有無で既存全列が bit 一致（軌道中立） |
| S-pair | `G_off` / `S_off` / 参照 ReLU `_off` の init・教師・入力列・flip・seed hash が互いに、かつ親走 `R_off` の `flip_state` と bit 一致（30k） |
| S-ref | `gate_dose_0830/provenance.json` の `output_sha256` と読む出力が一致。ずれたら転記列が動いていないかを履歴から確認 |
| S-floor | 床 $10^{-16}$ が親走の `floor_calibration.csv` と一致・再較正しない |
| S-mask | $U^{(10)}_{100}$ / $U^{(10)}_{500}$ が親走の $U_k$ の作り方（終端 10 点）と一致 |
| S-cap | early 窓で $U<0.05$ の seed が 9/10 以上。満たさない腕は `CAPACITY_UNDEFINED` |
| S-ci | 縮退 CI の自己検査 |

## 8. 出力

`results/valley_off_0903/`: `verdict.csv`・`contrasts.csv`・`unit_summary.csv`・
`revival.csv`・`depth_hist.csv`・`summary.md`・`provenance.json`・`config_used.yaml`・
`logs/*.npz`（gitignore。保存先は本機）。

数値の引用は `verdict.csv` と `summary.md` からのみ。凍結率・沈下率は出所と窓を添える。
本 spec は外部文献を引かない。

## Log

- 2026-09-03 起票（v1）。vault `谷の逃走_走A_spec_0903` v1 の段 1 を repo 側 spec に写した。
