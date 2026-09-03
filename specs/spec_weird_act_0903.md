# spec_weird_act_0903 — 謎関数ダイヤル（戻り道を設計した活性化で「向き・幅・容量」を 1 つずつ分離する: 1 層・帯内用量固定・5M）

Obsidian 側の正本: `可塑性喪失/spec/謎関数ダイヤル_spec_0903.md`（v1・2026-09-03・Claude 起草／Issa 裁定済み）。
親 spec: `可塑性喪失/spec/ゲート硬さダイヤル_spec_0902.md`（実行済み・宿主 `gate_dial_0902`）。
親主張: `可塑性喪失/主張/到達と離脱_統合主張_0903.md` §2-5（容量・可動度・向きの 3 列）。
本ファイルは **実装より先に config と一緒に単独 commit する** repo 側正本であり、
vault の §2・§4・§5・§6・§7 の逐語の写しに実装上の決めを足したものである。

**段 1 と段 2 は分けて投入する**（2026-09-03 Issa 裁定・vault §4.1）。本 spec は
**両方の段を事前登録する**が、実行は段 1（5 腕）→ 段 2（6 腕）の順に分ける。
段 2 の腕表・母数・集計は **段 1 の結果を見て変更しない**。変えるときは本 spec を失効させる。

## 1. 問い

非 ReLU の LoP は「現象1 の押し下げ ＋ 活性化固有の戻り道」で書かれているが、既存の 4 族
（leaky・ELU・SiLU・GELU）は **容量（サポート上でアフィンか）・可動度 $\mathbb E_x[\varphi']$・
向きの符号 $\mathrm{sign}\,(\varphi^2)'$ の 3 列が同時に動く**ので、どれが病理を担うかを族の
比較では割れない。負側の形を **1 か所だけ設計した** 5 族を同一ハーネスで回して 1 列ずつ分離する。

- **V1** 「反転」の定義は $\varphi'<0$ か $\varphi\varphi'>0$ か（鏡 leaky）
- **V2** 吸収域の幅は支持幅 $\max_x z-\min_x z$ を越える必要があるか（段付き leaky の梯子）
- **V3** 可動度を一定に保ったまま分水嶺を置くと罠になるか（折り返し leaky）
- **V4** 逃走の先を非アフィンな井戸で止めると救われるか（櫛）

## 2. 関数の定義（5 族・活性化名 10 個）

すべて $z\ge0$ で恒等（`relu` / `leaky_relu` と同じ正側）。負側だけを設計する。
母数 $a$（櫛は振動数 $\alpha$）は既存の `act_alpha` に載せる。**第 2 母数（$d$・$\beta$）は
活性化名に載せ、`VALLEY_ZERO` と同じ形のクラス定数辞書で引く**（`set_activation` の署名を変えない）。
すべて **自分の forward の真の導関数**（折れ目は測度 0）。`bwd_leak_0902` の代替勾配とは違う。

| 族 | 活性化名 | forward（$z<0$） | backward（$z<0$） | 母数 |
| --- | --- | --- | --- | --- |
| 鏡 leaky `LRm` | `mirror_leaky` | `(0.0 - a)*z` | `0.0 - a` | $a$=0.1 |
| 折り返し leaky `LRv` | `fold_leaky_d2` / `fold_leaky_d1` | `where(z>-d, a*z, (0.0-a)*(z+2d))` | `where(z>-d, a, 0.0-a)` | $a$=0.1・$d\in\{2,1\}$ |
| 段付き leaky `RB` | `band_leaky_d0p5` / `_d1` / `_d2` / `_d4` | `where(z>-d, 0, a*(z+d))` | `where(z>-d, 0, a)` | $a$=0.1・$d\in\{0.5,1,2,4\}$ |
| 滑り出し leaky `LRq` | `ramp_leaky_d1` | `where(z>-d, -kappa*z*z, a*z + a*d/2)` | `where(z>-d, -2*kappa*z, a)` | $a$=0.1・$d$=1・$\kappa=a/(2d)$=0.05 |
| 櫛 `CB` | `comb_binf` / `comb_b5` | `-env*sin(alpha*z)**2` | `env*(sin(alpha*z)**2/beta - alpha*sin(2*alpha*z))` | $\alpha\in\{1,2\}$・$\beta\in\{\infty,5\}$・`env = exp(-z/beta)` |

実装上の決め:

* `0.0 - a` と書く（$a$=0 で $-0.0$ を作らない。`spec_phantom_wall_0902` 追補 9 と同じ理由）。
* `comb_binf` は `env = 1.0`・`1/beta = 0.0` を **定数で**書き、`exp` を呼ばない。
* `kappa = a/(2d)` は `set_activation` 内で 1 回だけ計算する。
* 辞書: `FOLD_DEPTH = {"fold_leaky_d2": 2.0, "fold_leaky_d1": 1.0}`、
  `BAND_WIDTH = {"band_leaky_d0p5": 0.5, "band_leaky_d1": 1.0, "band_leaky_d2": 2.0, "band_leaky_d4": 4.0}`、
  `RAMP_DEPTH = {"ramp_leaky_d1": 1.0}`、`COMB_ENVELOPE = {"comb_binf": inf, "comb_b5": 5.0}`。
* **既存の活性化分岐は 1 行も触らない。** 活性化名の追加とクラス定数辞書の追加は乱数を消費しない。

閉形式の幾何（**代入値・実験出力ではない・引用不可**。実装は同じ量を数値で解き直して照合する = S-dial）:
櫛 $\alpha$=1 の分水嶺（葉の $\varphi^2$ 最大）は $\beta=\infty$ で $u$=1.571・4.712・7.854、
$\beta$=5 で 1.670・4.812・7.954。井戸は $u$=3.142・6.283・9.425。
$\alpha$=2 は分水嶺 0.785・2.356・3.927、井戸 1.571・3.142・4.712。
折り返し $d$=2 の分水嶺は $u$=2、極小は $u$=4。

## 3. 腕と段

run id `weird_act_0903`。宿主は `gate_dial_0902`（1 層・オラクル用量 12.16 固定・5M・腕プロセス並列）。
`gate_dial_0902.validate_config` は腕表を逐語照合するので通さず、`p3_runs_0902` と同じく
**`gate_dial_0902._run_arm` を腕表だけ足して直接呼ぶ**。宿主のコードは触らない。

| 腕 | 活性化 | `act_alpha` | 割る問い | 段 |
| --- | --- | --- | --- | --- |
| `LRm_a0p1_1216` | `mirror_leaky` | 0.1 | **V1** 反転の定義 | 1 |
| `LRv_d2_1216` | `fold_leaky_d2` | 0.1 | **V3** 分水嶺の位置 | 1 |
| `RB_d1_1216` | `band_leaky_d1` | 0.1 | **V2** 吸収域の幅 | 1 |
| `CB_a1_1216` | `comb_binf` | 1.0 | **V4** 井戸の容量 | 1 |
| `LRv_d1_1216` | `fold_leaky_d1` | 0.1 | V3 の錨（REPORT_ONLY・下記） | 1 |
| `RB_d0p5_1216` | `band_leaky_d0p5` | 0.1 | V2 梯子 | 2 |
| `RB_d2_1216` | `band_leaky_d2` | 0.1 | V2 梯子 | 2 |
| `RB_d4_1216` | `band_leaky_d4` | 0.1 | V2 梯子 | 2 |
| `LRq_d1_1216` | `ramp_leaky_d1` | 0.1 | 錨: 壁の点での硬さ | 2 |
| `CB_a1_b5_1216` | `comb_b5` | 1.0 | 包絡（Issa 案「下がることを嫌う」） | 2 |
| `CB_a2_1216` | `comb_binf` | 2.0 | 井戸間隔 対 支持幅 | 2 |

**`LRv_d1` の格**: 段 1 に同梱して回すが、**V3 のラベルは `LRv_d2` のみで付ける**。
`LRv_d1` は REPORT_ONLY で、`LRv_d2` が `PARTIAL`（1–4）に落ちたときに限り V3′ として
別ラベルを立てる。**両腕の結果を見てから主判定腕を選ばない。**

対照（すべて committed・再走しない・別走の値であることを引用時に必ず書く）:
`R_1216` / `LR_1216` / `E_1216`（`gate_dose_0830`）、`LR_a0p01_1216` / `S_b1_1216` /
`G_b1_1216`（`gate_dial_0902`）、`Gc_b1_1216` / `Sc_b3_1216`（`valley_clamp_0902`・水準のみ）。
主 endpoint $U$ は対照走の `verdict.csv` から**転記**する。**本 spec に対照の数値を書かない**
（vault が引く 2 点だけ: `S_b1` は $n_{\rm onset}$ が `CAPACITY_UNDEFINED` で水準 1M −1.716・5M −0.753、
`Gc_b1` は $U$ 0.30）。

## 4. 記号・窓・床

`gate_dial_0902` §3・§5 の逐語継承。

* condA・幅 100・隠れ 1 層・$T=10^4$・batch=1・素の SGD・lr 0.01・seed 0–9・
  `total_steps = 5e6`・cpu・float32・**オラクル用量固定 12.16 のみ**・`eval_loss_exact` 併記・
  1M `state_hash`。
* **`generator_offset: 0`**（S-pair のため系列を切らない）。
* 床 $10^{-16}$（1 層系・継承元 `gate_dose_0830`。再較正しない）。
* 窓はタスク終端の記録点のみで 10 点。末尾窓 = tasks 491–500、1M 窓 = tasks 91–100、
  early 窓 = tasks 2–11。$U_k$ = 窓平均未フィット率。**発症 = $U_k \ge 0.05$**。
* CI: $n_{\rm onset}$ は Clopper–Pearson、差は seed クラスタ paired percentile bootstrap
  B=10000（studentized 併算＋退化ガード・percentile 主）。**等価限界 $\Delta$=0.15 dex 据え置き**。
  seed 別符号検定を REPORT で併記。`bootstrap_seed: 20260914`（未使用日付）。
* `submerged` := $\max_x z_i \le 0$。**`span` := $\max_x z_i - \min_x z_i$**（新規）。
* `frozen` := $\lvert\mathbb E_x\varphi'\rvert < 10^{-6}$、
  `frozen_abs` := $\mathbb E_x\lvert\varphi'\rvert < 10^{-6}$。
  **`LRv`・`CB` では支持が折れ目・井戸を跨ぐと $\varphi'$ の符号が混じるので `frozen` は
  凍結の指標にならない。** どちらも verdict には使わない。
* 位置指標（腕別・REPORT）: `RB` の `in_band` := $\max_x z_i\le0$ かつ $\min_x z_i\ge-d$／
  `LRv` の `at_sink` := $\lvert\bar z_i+2d\rvert\le0.5$／
  `CB` の `at_well` := $\min_{k\ge1}\lvert\bar z_i+k\pi/\alpha\rvert\le0.5$。

**ロガー（軌道中立・追加のみ）**: `gate_dial_0902` の 5 列（`layer1_mob`・`layer1_absmob`・
`layer1_zmax`・`layer1_zmean`・**`layer1_v_unit`**）を継承し、**`layer1_zmin` $=\min_x z_i(x)$ を
1 列足す**（`span` のため）。S-log-b で bit 一致を見る。列名は §10 追補 1 を見よ。

## 5. 事前予測

### 5.1 Issa（vault §7.1・repo commit で凍結）

**V1・V2 の $d^\ast$・V3・V4・`CB_a1_b5` の発散・外れたとき第一に疑うもの —— 6 項すべて「わからない」。**
段の切り方は「段 1 と段 2 を分ける／`LRv_d1` は段 1 へ」、$\Delta$=0.15 は据え置き。
**したがって本走に対する Issa 側の事前予測は存在しない。** 結果ノートで「当たった／外れた」と
書くときは誰の予測かを必ず添える。事前登録は §6 の判定表の凍結で成立している。

### 5.2 Claude（vault §7.3・実行前・3 列の代数だけから）

| 項目 | 予測 | 自信 |
| --- | --- | --- |
| V1 | `REVERSAL_IS_PRODUCT_SIGN`（`LRm` 0/10）。沈下深さ中央値は leaky と同じ ≈4 | 高 |
| V2 | `BAND_WIDTH_THRESHOLD`・$d^\ast$=2（$d$=0.5: 0、1: 0–2、2: ≥5、4: 10/10）。`span` 中央値 1.4–2 | 中 |
| V3 | `WATERSHED_TRAPS`（`LRv_d2` ≥5/10・`at_sink` ≥0.5・`frozen_abs` 0） | 低〜中 |
| V4 | `WELL_RESCUES`（`CB_a1` ≤2/10・対 `S_b1` の CI が $-\Delta$ の下・`at_well` ≥0.5） | 低〜中 |
| `LRq` | `LR_1216` と等価限界内または `INCONCLUSIVE_WIDE`・0/10 | 中〜高 |
| `CB_a1_b5` | `CB_a1` と同じ判定・発散 0–2 seed | 中 |
| `CB_a2` | `SLOW_FIT` または `CAPACITY_UNDEFINED`。フィットすれば 0/10 | 低 |
| C1 再現 | `LRm` 以外の全腕で $b$ 分担 0.09–0.10・沈下 30k–50k step | 中 |
| 3 レジーム | `RB_d2`・`RB_d4` の `in_band` が −0.13 前後、`LRm`・`LRq` の活性ユニットが −0.20 前後 | 中 |

### 5.3 先生（vault §7.2・**転記ではなく含意**。先生の予測ではない）

§2-3 の「帯へ向かうが既定、例外は谷の向こうだけ」⇒ `LRm` は帯へ、`LRv` の $(-2d,-d)$ と
`CB` の葉の深い半分は深い側へ。§2-1 補題 ⇒ `LRv` の極小と `RB` の死帯は容量 0、`CB` の井戸は容量あり。

## 6. 判定ラベル（vault §5・**判定順 V1 → V2 → V3 → V4・互いに独立・1 つの verdict に畳まない**）

E1（発症数）・E2（水準・P3′ 対 `R_1216`・P5′ 対 `LR_1216`・位置比 $\rho$）・
E3（発症時刻 $k^\ast$・打ち切り 500・KM 曲線）を全腕で出す。

### V1 — 反転の定義（`LRm`・段 1 で確定）

| `LRm` の $n_{\rm onset}$(5M) | ラベル |
| --- | --- |
| 0 | `REVERSAL_IS_PRODUCT_SIGN` |
| ≥5 | `REVERSAL_IS_DERIVATIVE_SIGN` |
| 1–4 | `PARTIAL`（E2 の P5′ で読む） |

### V2 — 吸収域の幅（`RB` 梯子・**段 2 の完了後にのみ付く**）

梯子: `LR_1216`（$d$=0・committed）→ `RB_d0p5` → `RB_d1` → `RB_d2` → `RB_d4` →
`R_1216`（$d=\infty$・committed）。段 1 の `RB_d1` 単独では下表のどの条件も評価できないので、
段 1 では $n_{\rm onset}$ と `span` を REPORT に置くだけにする。

| 条件 | ラベル |
| --- | --- |
| 新規 4 腕の $n_{\rm onset}$ が $d$ について非減少で、0 の腕と ≥5 の腕が両方ある | `BAND_WIDTH_THRESHOLD`（$d^\ast$ = ≥5 になる最小の $d$ を添える） |
| 新規 4 腕がすべて ≥5 | `ANY_BAND_ABSORBS` |
| 新規 4 腕がすべて 0 | `NO_BAND_ABSORBS_IN_RANGE` |
| $d$ を増やして $n_{\rm onset}$ が 3 以上減る、または隣接対比の CI が丸ごと $-\Delta$ の下 | `REVERSAL` |
| 上記以外 | `PARTIAL` |

REPORT（ラベルを作らない）: $d^\ast$ と 1M 窓の沈下ユニットの `span` seed 中央値を並べる
（`LR_1216` は列が無いので `RB_d0p5` を代理）。`in_band` 率は総和と率の両方（教訓⑰）。

### V3 — 分水嶺の位置（`LRv_d2`・段 1 で確定）

| `LRv_d2` の $n_{\rm onset}$(5M) | ラベル |
| --- | --- |
| ≥5 | `WATERSHED_TRAPS` |
| 0 | `MOBILITY_SUFFICES` |
| 1–4 | `PARTIAL` → `LRv_d1` に本表を適用して **V3′** を別に立てる |

### V4 — 井戸の容量（`CB_a1` 対 `S_b1`・`Gc_b1`・段 1 で確定）

| 条件 | ラベル |
| --- | --- |
| `CB_a1` の $n_{\rm onset}$(5M) ≤2 かつ paired（`CB_a1` − `S_b1`・5M）の CI が丸ごと $-\Delta$ の下 | `WELL_RESCUES` |
| ≥5 かつ末尾窓の沈下ユニット `at_well` 率 ≥0.5 | `WELL_TRAPS` |
| ≥5 かつ `at_well` 率 <0.5 | `COMB_FAILS_ELSEWHERE` |
| 上記以外 | `PARTIAL` |

### 副次（REPORT_ONLY・ラベルにしない）

C1 の再現（初回沈下までの $\Delta(w\cdot\mu)$・$\Delta b$・$b$ 分担・深さ・所要 step）、
境界回帰の 3 レジーム（per-unit slope の中央値と四分位。**診断スクリプトは本走で `src/` に置く**）、
錨 3 本（`LRq`−`LR_1216`／`CB_a1_b5`−`CB_a1`／`CB_a2`−`CB_a1`）、
`submerged` 率の時系列と復活率（境界／非境界別・総和と率）、末尾窓の深さ十分位、
$\lvert v_i\rvert$・`eff_rank`・$\lVert w\rVert$・$b$・`preact_sd`・`eval_loss_exact`、
`frozen` / `frozen_abs` / `in_band` / `at_sink` / `at_well` 率。

### 数値発散

`gate_dose_0830` §5.6 を逐語継承（`NUMERIC_DIVERGENCE` でセルを空にし、verdict が空セルを
要求すれば `INCONCLUSIVE_DIVERGENCE`）。**`CB_a1_b5` で発散が出たら当該腕だけ落とし、
V4 は `CB_a1` で判定する**（錨であって判定腕ではない）。発散 seed 数・最初の発散 step・
直前の最深 $\bar z$ を `summary.md` に書く。

## 7. 前段チェック

| 名 | 内容 |
| --- | --- |
| S1 | `OMP_NUM_THREADS=1` |
| S-pair | 新規腕の step 0 で init・教師・入力列・flip が `gate_dose_0830` の `logs/` と bit 一致（30k 短走） |
| S-limit | `band_leaky` $d$=0 が `leaky_relu` と、`fold_leaky` $d=10^6$ が `leaky_relu` と、`mirror_leaky` $a$=0 が `relu` と 30k 短走で bit 一致。**対象は `logs/*.npz` の全列と `state_hash`** |
| S-fd | 5 族の backward を float64 中心差分と照合（$z\in[-20,20]$ の 41 点＋折れ目 $\pm0.1$ の 21 点、折れ目 $\pm10^{-3}$ は除外、許容 $10^{-6}$）。`comb_b5` は包絡が $e^4$ まで伸びるので相対許容 |
| S-num | $z\in[-200,200]$（`comb_b5` だけ $[-100,100]$）で NaN・inf 無し。`comb_b5` の溢れる深さを `provenance.json` に書く |
| S-dial | 登録した分水嶺・井戸・極小の代入値を、実装が $\varphi'$ の零点と $(\varphi^2)'$ の零点を数値で解き直して相対許容 6% で照合 |
| S-const | config の `activation.*.second_param_value`・`arms[].second_param`・`nets.py` のクラス定数辞書の 3 者が一致（§10 追補 5） |
| S-mob | `layer1_mob` が `band_leaky` で $\hat p + a\Pr_x[z<-d]$、`mirror_leaky` で $\hat p-(1-\hat p)a$ と一致（30k・許容 $10^{-6}$）。`layer1_zmin ≤ layer1_zmean ≤ layer1_zmax` を全記録で検算 |
| S-log-b | ロガー有無の 2 走が 30k で bit 一致。不一致ならロガーを外す |
| S-dose | 実測 $\lVert\mu\rVert$ が目標と相対誤差 $10^{-10}$ 以内 |
| S-cap | **二段**。early 窓の $U<0.05$ が 9/10 以上。満たさない腕は 1M 窓で再判定し、通れば `SLOW_FIT` として V に残す。両方落ちれば `CAPACITY_UNDEFINED`。**`CB_a2` と `RB_d4` が最も怪しい** |
| S-taut | `LRv`・`CB` の `frozen` が構成上恒真／恒偽になっていないか（判定式に入れていないので影響なし） |
| S-mask | $U_k$ の作り方（タスク終端 10 点）が宿主と一致 |
| S-cover | §6 の各項目に実装の対応先を 1 対 1 で照合した表を `summary.md` の先頭に置く。**副次の C1 再現と 3 レジームは初版で落ちやすい** |

**本走前に §6 の集計はしない。**

## 8. 出力

`results/weird_act_0903/`: `verdict.csv`（腕・族・母数・$n_{\rm onset}$ 1M/5M・CP95・$U_k$ 10 値・
V1–V4・P3′・P5′・$\rho$・`SLOW_FIT` / `CAPACITY_UNDEFINED` / `NUMERIC_DIVERGENCE`）、
`summary.md`、`onset_times.csv`、`ladder_table.csv`（V2 の梯子と `span`）、
`position_table.csv`（`in_band`・`at_sink`・`at_well`・`frozen`・`frozen_abs`）、
`depth_hist.csv`、`c1_table.csv`、`regime_table.csv`、`layer_stats.csv`、`increments.csv`、
`provenance.json`（宿主 SHA・**回した段**・溢れ深さ）、`config_used.yaml`、
`logs/*.npz`（gitignore）、`arm_status/`。

**数値の引用は `verdict.csv` と `summary.md` からのみ。**

## 9. 引用上の注意

* 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。「起きない」と書かない。
* 対照は**別走の committed 値**。同一走の腕ではない。
* **用量 1 点（12.16）・1 層・5M・float32 の主張である。** 引くときは用量を添える。
* §2 の分水嶺・井戸・極小は**閉形式の代入値**。`span` の実測（`layer1_zmin`）が出るまで引かない。
* **5 族はすべて本走のための合成活性化。** 処方箋として一般化しない。
* `frozen` は `LRv`・`CB` で凍結の指標にならない。引くなら `frozen_abs` と出所・窓を添える。
* V1〜V4 は互いに独立。「3 列のどれが病理を担う」は 4 判定から人が読む裁定であってラベルではない。
* Issa 案の原形 $e^{-x/\alpha}\sin^2(-\alpha x)$ は $\alpha$ が振動数と包絡を兼ねる。
  本走の `CB` は分離版であり、**原形は回していない**。

本 spec は外部文献を引かない。


## 10. 実装段の追補

**§1〜§9 の登録済みの判定基準は 1 つも緩めていない。** 以下は vault が実装に委ねた分岐の決着と、
起草時に気づかれていなかった不整合への guard である（`spec_gate_dial_0902.md` §6.3 と同じ格）。

1. **★ 列名の訂正（vault §4.3 の字義）。** vault は宿主のユニット列を
   `layer1_mob` / `layer1_absmob` / `layer1_zmax` / `layer1_zmean` / **`layer1_v`** と書くが、
   実装の `NEW_UNIT_KEYS`（`src/gate_dial_0902.py:101`）は `("mob","absmob","zmax","zmean","v_unit")` で、
   npz の列名は **`layer1_v_unit`**。`layer1_v` という列は存在しない。**逐語継承が正なので実装は
   `layer1_v_unit` を採る。vault 側に追補が要る。**

2. **ロガーの入れ方（宿主を触らない解）。** `_run_arm` は `rec = DialRecorder(probes, st)` を
   **直書き**しており（`gate_dial_0902.py:530`）、recorder を差し替える引数が無い。一方
   `NEW_UNIT_KEYS` と `unit_extra_record` を書き換えると宿主自身のロガーが変わる。
   vault §4.3 の「宿主のコードは触らない」を守るため、本走は
   **`DialRecorder` を継承した `WeirdRecorder`（`zmin` を 1 列足すだけ）と、`_run_arm` の本体を
   写した `_run_arm_weird`（recorder の 1 行だけ差し替え）** を本モジュールに置く。
   `write_arm_logs_dial` は `rec.unit.items()` を総称で回すので変更不要。
   **`src/gate_dial_0902.py` と `src/nets.py` の既存の行は 1 行も変えない**（`nets.py` へは追記のみ）。

3. **`nets.py` への追記の形**（`silu_clamp` / `bwd_reflect` の先例に逐語で倣う）:
   `ACTIVATIONS` に 10 名を足す／クラス定数辞書 `FOLD_DEPTH`・`BAND_WIDTH`・`RAMP_DEPTH`・
   `COMB_ENVELOPE` を `VALLEY_ZERO` の隣に置く／`act_fn` は **ELU フォールスルーの直前**に、
   `act_grad` は **`activation_plus_alpha` の分岐の直前**に新分岐を足す。既存の `==` 比較より
   上には 1 つも挿入しない。**両方の連鎖にはガードが無く、名前だけ足して分岐を忘れると
   その活性化は黙って ELU になる**ので、10 名すべてについて forward・backward の両方を
   単体テストで閉じる。

4. **`act_alpha` のガード（新規タプル 2 本）。** `comb` は $\alpha$=2 を取るので、
   `[0,1]` を課す `SURROGATE_ACTIVATIONS` には入れられない。また 5 族はすべて
   **自分の forward の真の導関数**なので、代替勾配を意味する `SURROGATE_ACTIVATIONS` は
   意味論的にも誤り。よって新規タプル 2 本を足し、`set_activation` の
   `STEEPNESS_ACTIVATIONS` 分岐の**後ろ・`act_grad_form` 分岐の前**に検査を挿入する:
   `WEIRD_SLOPE_ACTIVATIONS`（leaky 由来 8 名・$a\in[0,1]$）と
   `WEIRD_FREQ_ACTIVATIONS`（櫛 2 名・有限正の $\alpha$）。**既存の名前をこれらに入れない。**

5. **S-const（新規の前段チェック）。** 第 2 母数（$d$・$\beta$）は活性化名に埋め込まれ、
   実体は `nets.py` のクラス定数辞書にある。config の `activation.*.second_param_value` と
   `arms[].second_param` は**その写し**なので、3 者の一致を検査する。写しがずれたまま走ると
   §2 の幾何（分水嶺・井戸）と実際に回った関数が食い違う。

6. **`u_star` / `u_fr` は 5 族とも登録しない。** 谷の理論を持たないので、宿主の V3 表
   （`dial_table.csv`）には載せない。S-dial が照合するのは §2 の**分水嶺・井戸・極小**である。

7. **テストの様式は `unittest`。** 直近 3 本の活性化テストは pytest だが、この checkout には
   `.venv` も `pytest` も無い（`python3` = miniconda・torch 2.12.0・numpy 2.4.6）。
   `src/test_weird_act_0903.py` は `OMP_NUM_THREADS=1 python3 -m unittest src.test_weird_act_0903 -v`
   で走る形にする（`test_gate_dial_0902.py` / `test_bwd_leak_0902.py` と同じ様式）。

8. **段 1 の出力に V2 のセルを作らない。** §6 の V2 は段 2 の完了後にのみ付く。
   段 1 の `verdict.csv` では V2 セルを空にし、`RB_d1_1216` の $n_{\rm onset}$ と `span` を
   REPORT 行として置く。`provenance.json` の `stages_run` に回した段を書く。

## Log

- 2026-09-03 起票（v1）。vault `謎関数ダイヤル_spec_0903` v1（段裁定・§7.1 記入まで済み）を
  repo 側 spec に写した。段 1（5 腕）と段 2（6 腕）の両方を事前登録し、実行だけを分ける。
