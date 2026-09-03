# spec_comb_isolate_0903 — 櫛の分離（ゲートなしの非線形性は LoP の代償を払わないか: 段 A = 1 層／段 B = 深さ 2）

Obsidian 側の正本: `可塑性喪失/spec/櫛の分離_spec_0903.md`（v1・2026-09-03・Claude 起草／Issa 依頼・§7.1 記入済み）。
親: `可塑性喪失/spec/実行済み/謎関数ダイヤル_spec_0903.md`（実行済み・V4 は機構未分離）／`可塑性喪失/測定/謎関数ダイヤル結果_0903.md` §8。
本ファイルは **実装より先に config と一緒に単独 commit する** repo 側正本であり、
vault の §0–§8 の逐語の写しに実装上の決めを足したものである。

**段 A と段 B の両方を本 commit で事前登録する**（2026-09-03 Issa 裁定）。実行は段 A → 段 B の順次。
段 A `comb_isolate_0903`（1 層・4 腕）と段 B `comb_mlp2_0903`（深さ 2・1 腕）は
**窓も床も違う**（§4）。**段 C（実ベンチ）は本 spec の対象外**（先生相談が先・vault §4.3）。

## 1. 問い

`weird_act_0903` の V4 は `CB_a1`（櫛・α=1・包絡なし）が 0/10 で `WELL_RESCUES` を取ったが、
その水準は median log₁₀U = −12.49 で対照の最良（`E_1216` −2.73）より 10 桁近く低い。
**「井戸が救った」のか「別の関数クラスへ逃げた（厳密補間した）」のかが割れていない。**

ゲート族（ReLU・leaky・ELU・GELU）は片側で $\varphi'$ を下げることで非線形性を作るので、
**非線形性と可動度の損失が同じつまみ**である。周期族（櫛・Snake）は振動から非線形性を作り
$\varphi'$ を削らない。もし `CB_a1` の 0/10 がこの違いで出ているなら、主張候補は
「**LoP はゲートで非線形性を買うことの代償であり、振動で買えば代償を払わない**」になる。
本走は**表現力を落として井戸だけ残す**腕でこれを割る。

- **V5** 井戸 1 個は救うか（段 A）
- **V6** 櫛の水準 1e−13 は多葉の表現力か井戸か（段 A）
- **V7** 深さ 2 で櫛が LoP を出さないか（段 B）

**「ゲートで非線形性を買う代償」は動機であってラベルではない。** V5–V7 から人が読む。
**処方箋ではない**（Snake を推奨として引かない）。

## 2. 関数の定義（活性化名 4 個）

すべて `act_alpha` は **振動数 $\alpha$**（`band_leaky_dpi` だけ傾き $a$）。$\alpha$=1 固定。
葉の端 $-\pi/\alpha=-\pi$。漏れ $a$=0.1 は `LR_1216` に揃えたクラス定数。
すべて **自分の forward の真の導関数**（折れ目は測度 0）。

| 族 | 活性化名 | forward（$z<0$） | backward（$z<0$） | 一言 |
| --- | --- | --- | --- | --- |
| 1 葉＋平坦 `CB1f` | `comb1_flat` | `-sin(az)**2`（$-\pi\le z<0$）／`0` | `-a*sin(2az)`／`0` | 井戸 1 個の先は ReLU の壁。**壁の硬さ $\varphi'(0^-)=0$ は `CB_a1` と同じ** |
| 1 葉＋leaky `CB1l` | `comb1_leaky` | 同上／`a_leak*(z+pi)` | 同上／`a_leak` | 井戸 1 個の先に戻り道 |
| 段付き leaky `RB_dpi` | `band_leaky_dpi` | `0`（$-\pi\le z<0$）／`a*(z+pi)` | `0`／`a` | `CB1l` の葉を平坦に置き換えた対照 |
| Snake（錨）`SN` | `snake` | **全域** `z + sin(az)**2/a` | `1 + sin(2az)` | **ゲートなし**・単調・負側の可動度平均 1 |

対照（committed・再走しない）: `comb_binf`（= `CB_a1`・`weird_act_0903`）。

実装上の決め:

* `comb1_*` は `math.pi / act_alpha` を分岐内で解き直す（`VALLEY_ZERO` と同じ流儀・キャッシュしない）。
* `comb1_leaky` の漏れは `COMB1_LEAK = {"comb1_leaky": 0.1}`。**`act_alpha` は $\alpha$ に使う。**
* `band_leaky_dpi` は **新しい分岐を書かない**。`act_fn`/`act_grad` が `if self.act in self.BAND_WIDTH:`
  で辞書引きするので、`BAND_WIDTH["band_leaky_dpi"] = math.pi` の 1 行で forward・backward とも通る（§10 追補 3）。
* `snake` は **正側も恒等ではない**。したがって `submerged`（$\max_x z\le0$）を定義しない。
* ガードのタプル: `band_leaky_dpi` → `WEIRD_SLOPE_ACTIVATIONS`（$a\in[0,1]$）、
  `comb1_flat` / `comb1_leaky` / `snake` → `WEIRD_FREQ_ACTIVATIONS`（有限正の $\alpha$）。

### 2.1 段 A の 2×2

| | 先が平坦 | 先が leaky |
| --- | --- | --- |
| **$[-\pi,0)$ が葉** | `CB1f` | `CB1l` |
| **$[-\pi,0)$ が平坦** | `R_1216`（committed） | `RB_dpi` |

行の差 = 井戸（葉）の効果、列の差 = 戻り道の効果。**`CB_a1` と `CB1l` の差 = 2 葉目以降（多葉の表現力）の効果。**

### 2.2 本走が答えないこと

包絡（$\beta<\infty$・発散域）・$\alpha\ne1$・Adam・実ベンチ・「Snake が良い活性化か」。

## 3. 腕と段

### 段 A `comb_isolate_0903`（1 層・4 腕新規）

宿主は `gate_dial_0902`。`validate_config` は通さず、`weird_act_0903._run_arm_weird`（宿主 `_run_arm` の写し）を使う。

| 腕 | 活性化 | `act_alpha` | 割る問い | 対 |
| --- | --- | --- | --- | --- |
| `CB1f_a1_1216` | `comb1_flat` | 1.0 | **V5** 井戸 1 個は救うか | `R_1216` |
| `CB1l_a1_1216` | `comb1_leaky` | 1.0 | V5・**V6** | `RB_dpi`・`CB_a1` |
| `RB_dpi_1216` | `band_leaky_dpi` | 0.1 | V5 の対照 | `CB1l` |
| `SN_a1_1216` | `snake` | 1.0 | 錨: ゲートなしの周期 | REPORT_ONLY |

対照（committed・転記）: `R_1216`・`LR_1216`・`E_1216`（`gate_dose_0830`）、
`CB_a1_1216`・`RB_d2_1216`・`RB_d4_1216`（`weird_act_0903`）、`S_b1_1216`（`gate_dial_0902`）。
**本 spec に対照の数値を書かない**（§1 の水準 2 つは動機の材料で、判定の対照値ではない）。

### 段 B `comb_mlp2_0903`（深さ 2・1 腕新規）

宿主は `lr_a1_0901`。1 腕 `CB_A1` = `comb_binf`（α=1）・`hidden [100,100]`・`centered_layers [1]`。
**中心化なし（`_none`）は回さない**（`E_none` の発散前例）。
対照 `L2_A1`（ReLU）・`LR_A1`（leaky）・`E_A1`（ELU）は committed・paired。
**段 A の結果を見て段 B の腕表を変えない。**

## 4. 記号・窓・床（**段 A と段 B で違う**）

| | 段 A `comb_isolate_0903` | 段 B `comb_mlp2_0903` |
| --- | --- | --- |
| 継承元 | `gate_dial_0902` §3・§5 | `lr_a1_0901` |
| 深さ・幅 | 1 層・100 | 2 層・[100,100] |
| 中心化 | `centered_layers: [1]` | 同 |
| 用量 | オラクル固定 **12.16** | オラクルなし（`A_layer_input_centering`） |
| 末尾窓 | タスク **491–500** | タスク **451–500** |
| 1M 窓 | 91–100 | 91–100（`EXACT_FIT` 用に同定義で作る） |
| early 窓 | 2–11 | —（S-cap は段 A のみ） |
| 床 | **1e−16** | **1e−23** |
| 発症 | $U_k\ge0.05$・Clopper–Pearson | 発症数は使わない（水準の paired 差のみ） |
| CI | paired percentile bootstrap B=10000・**Δ=0.15 dex** | 同 |
| bootstrap seed | **20260915** | 20260915（同一 spec の 2 段なので分けない） |
| `generator_offset` | 明示 0 | 明示 0 |

`submerged`・`span`・`frozen`・`frozen_abs` は `spec_weird_act_0903.md` §4 逐語。
`at_well`（$\lvert\bar z_i+\pi\rvert\le0.5$）を `comb1_*` で、`in_band` を `RB_dpi` で REPORT。
**`SN` は `submerged` を定義しない**（負側に壁がない）。

### ★ 新規ガード `EXACT_FIT`（両段・本 spec の新設）

**1M 窓の $U$ seed 中央値 ≤ 1e−8** なら `EXACT_FIT` を立てる。
**`EXACT_FIT` の腕には水準（log₁₀U の差）に基づくラベルを付けない**（機構の帰属に使えない）。
$n_{\rm onset}$ に基づくラベルは残す。`eval_loss_exact` を併記する。
閾値は float32 厳密補間の床（≈1e−13）と leaky の 1e−3 の中間で、**桁が 5 以上離れた値**。

## 5. 事前予測

### 5.1 Issa（vault §7.1・2026-09-03 記入）

| 項目 | 予測 |
| --- | --- |
| `CB1f` | ≥5/10 発症・`at_well` < 0.3 |
| `CB1l` | 0/10 |
| `RB_dpi` | 0–2/10 |
| V5 | `WELL_IRRELEVANT_RETURN_PATH_CARRIES` |
| V6 | `LEVEL_FROM_MULTILOBE` |
| V7 | `NO_LOP_DEPTH2`・`EXACT_FIT` 立つ・発散 0 seed |
| `SN` | 0/10・`EXACT_FIT` 立つ・発散 0 seed |

**★ この記入は vault §7.2（Claude）の 7 項と逐語で同一である。独立の予言ではない。**
Issa は起草側の候補値をそのまま採ったので、**照合の相手は 1 組しかない**。
`configs/*.yaml` の `preregistration.prediction_provenance` に
`draft_values_proposed_first_then_approved_by_Issa` として記録した（`lr_a1_0901` の同名フィールドと同じ格）。
**結果ノートに「Issa と Claude の両方が当てた／外した」と書かない。**
独立な予言が要るなら **先生の vault §7.3 を本走前に埋めるしかない**（現状は未記入・含意も引かない）。

### 5.2 外れたとき第一に疑うもの（vault §7.2・順に）

(i) `CB1l` が `EXACT_FIT` → 表現力の読みが誤り（1 葉＋戻り道で十分な関数クラスになる。V6 が拾う）。
(ii) `CB1f` が 0/10 → 沈下ユニットが葉の中に留まっている（`at_well` と深さ十分位を見る）。
壁の硬さ $\varphi'(0^-)=0$ のまま平坦で救われることになり、v6 の硬さ主張に直接効く。
(iii) `SN` が発症 → ゲートなしでも LoP が出る。「ゲートで非線形性を買う代償」という枠組み自体が倒れる。

## 6. 判定（**V5 → V6 → V7・互いに独立・1 つに畳まない**）

### V5 — 井戸 1 個は救うか（段 A）

| 条件 | ラベル |
| --- | --- |
| `CB1f` $n_{\rm onset}$(5M) ≤2 かつ paired（`CB1f` − `R_1216`）の CI が丸ごと $-\Delta$ の下 かつ `CB1f` が `EXACT_FIT` でない | `SINGLE_WELL_RESCUES` |
| `CB1f` ≥5 かつ `CB1l` ≤2 かつ paired（`CB1l` − `RB_dpi`）の CI が丸ごと $-\Delta$ の下 | `WELL_HELPS_ONLY_WITH_RETURN_PATH` |
| `CB1f` ≥5 かつ paired（`CB1l` − `RB_dpi`）が `EQUIV_SOFT` | `WELL_IRRELEVANT_RETURN_PATH_CARRIES` |
| `CB1f` ≥5 かつ `CB1l` ≥5 | `LOBE_DOES_NOT_RESCUE` |
| 上記以外 | `PARTIAL` |

`CB1l` または `RB_dpi` が `EXACT_FIT` なら 2・3 行目の水準条件は評価せず `INCONCLUSIVE_EXACT_FIT`。

### V6 — 櫛の水準の帰属（段 A・**`EXACT_FIT` の有無だけで付ける**）

| `CB_a1`（committed） | `CB1l` | `CB1f` | ラベル |
| --- | --- | --- | --- |
| 立つ | 立たない | 立たない | `LEVEL_FROM_MULTILOBE` |
| 立つ | 立つ | — | `LEVEL_FROM_SINGLE_LOBE` |
| 立たない | — | — | `GUARD_MISCALIBRATED`（閾値 1e−8 を事後に動かさない） |
| 上記以外 | | | `PARTIAL` |

`SN` の `EXACT_FIT` は REPORT。

### V7 — 深さ 2（段 B）

| 条件 | ラベル |
| --- | --- |
| paired（`CB_A1` − `L2_A1`・末尾窓 log₁₀U）の CI が丸ごと $-\Delta$ の下 | `NO_LOP_DEPTH2`（`EXACT_FIT` なら機構を読まない旗を添える） |
| 同 CI が $\pm\Delta$ 内または 0 を含む | `LOP_PERSISTS_DEPTH2` |
| `CB_A1` − `LR_A1` が `SHORT_OF_SOFT` | 併記 `WORSE_THAN_LEAKY_DEPTH2` |
| `NUMERIC_DIVERGENCE` | `INCONCLUSIVE_DIVERGENCE` |

### 副次（REPORT_ONLY）

C1 の再現（初回沈下までの $\Delta(w\cdot\mu)$・$\Delta b$・$b$ 分担・深さ・所要 step ——
**`weird_act_0903` で未実装のまま。本走で `src/` に置く**）／`at_well`・`in_band`・`frozen_abs`／
深さ十分位／**全ユニット** span と $\lvert v\rvert$（`CB_a1` の 1.12・0.19 が 1 葉でも出るか）／
`SN` の前活性 sd と $\lvert v\rvert$／段 B の第 2 層 dose・`eff_rank`・層別 `submerged`。

### 数値発散

`gate_dose_0830` §5.6 逐語継承。**`SN` は正側も非線形なので発散を許容する腕（錨）。**
段 B の `CB_A1` が発散したら `NUMERIC_DIVERGENCE` で V7 を空にする。

## 7. 前段チェック

`spec_weird_act_0903.md` §7 を逐語継承（S1・S-copy・S-const・S-dial・S-fd・S-num・S-mob・
S-log-b・S-pair・S-dose・S-cap・S-taut・S-mask・S-cover）。追加 3 本:

| 名 | 内容 |
| --- | --- |
| S-limit | `comb1_leaky` / `comb1_flat` が **開区間 $(-\pi,\infty)$ で** `comb_binf` と bit 一致（端点は §10 追補 1）／`band_leaky_d0` が `leaky_relu` と一致／`snake` が $\alpha=10^{-6}$ で恒等に相対 1e−4 以内（§10 追補 2） |
| **S-guard** | `EXACT_FIT` 閾値 1e−8 が committed 出力で `CB_a1_1216` で立ち、`LR_1216`・`E_1216`・`R_1216`・`S_b1_1216` で立たないことを**本走前に**確認。立たなければ閾値を動かさず `GUARD_MISCALIBRATED` |
| S-B | 段 B の `CB_A1` が step 0 で `LR_A1` と init・教師・入力列 bit 一致（`lr_a1_0901` の S0′ 様式） |

**本走前に §6 の集計はしない。**

## 8. 出力

段 A `results/comb_isolate_0903/`: `verdict.csv`・`summary.md`・`position_table.csv`・
`c1_table.csv`・`onset_times.csv`・`depth_hist.csv`・`layer_stats.csv`・`provenance.json`・
`config_used.yaml`・`logs/*.npz`（gitignore）・`arm_status/`。
段 B `results/comb_mlp2_0903/`: `lr_a1_0901` と同型（`verdict.csv`・`layer_stats.csv`・
`s_distribution.csv`・`summary.md`・`provenance.json`・`logs/*.npz`）。

**数値の引用は `verdict.csv` と `summary.md` からのみ。**

## 9. 引用上の注意

* 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。「起きない」と書かない
* 対照は**別走の committed 値**。`CB_a1`・`RB_d2`・`RB_d4` は `weird_act_0903`
* 段 A は用量 1 点（12.16）・1 層・5M・float32。段 B は深さ 2・第 1 層中心化・オラクルなし・素の SGD
* **`EXACT_FIT` の腕の水準差を機構として引かない。** $n_{\rm onset}$ は引ける
* 4 族すべて本走のための合成活性化。`SN`（Snake）は先行があるが**推奨として引かない**
* 「ゲートで非線形性を買う代償」は §1 の**動機**であってラベルではない
* 段 C は本 spec の外。段 A・B の結果から段 C を自動起案しない
* **§5.1 の Issa 予測は §7.2（Claude）と同一で独立の予言ではない**（§5.1）

## 10. 実装段の追補

**§1〜§9 の登録済みの判定基準は 1 つも緩めていない。** 以下は vault が実装に委ねた分岐の決着と、
起草時に気づかれていなかった不整合への guard である。

1. **★ S-limit の区間の訂正（vault §6 の字義）。** vault は「`comb1_leaky` と `comb_binf` が
   $z\ge-\pi$ で bit 一致」と書くが、**閉区間では成り立たない**。float64 で
   $\sin(-\pi) = -1.2246\times10^{-16}$ なので、`comb_binf` は $z=-\pi$ で
   $\varphi = -1.4998\times10^{-32}$・$\varphi' = -2.4493\times10^{-16}$ を返し、
   1 葉版の厳密 0 と一致しない（`comb1_leaky` はそこが登録済みの折れ目で $\varphi'=0.1$）。
   **一致は開区間 $(-\pi,\infty)$ で見る。** 端点 1 点は測度 0 で、S-fd も折れ目 $\pm10^{-3}$ を
   除外しているので**登録された検査は弱めていない**。実測: 開区間 20001 点で forward・backward とも
   `torch.equal` が真。
2. **★ `snake` の S-limit の $\alpha$ を確定（vault は「$\alpha\to$ 極小」で数値を書いていない）。**
   $\varphi-z=\sin^2(\alpha z)/\alpha\approx\alpha z^2$ なので、$\lvert z\rvert\le30$ で相対 1e−4 を
   満たすには $\alpha\le3\times10^{-6}$ が要る。実測は $\alpha$=1e−3 で 3.0e−2、1e−4 で 3.0e−3、
   1e−5 で 3.0e−4、**3e−6 で 9e−5、1e−6 で 3e−5**。**登録値を $\alpha=10^{-6}$・相対許容 1e−4 とする。**
3. **`band_leaky_dpi` に新しい分岐を書かない（§2）。** `act_fn`/`act_grad` は
   `if self.act in self.BAND_WIDTH:` で辞書引きするので、`BAND_WIDTH` への 1 エントリと
   `ACTIVATIONS`・`WEIRD_SLOPE_ACTIVATIONS` への名前追加だけで forward・backward とも正しい。
   実測で確認（帯の内側 $\varphi=\varphi'=0$、$z<-\pi$ で $\varphi'=a$、$z=-4$ で $\varphi=-0.0858$）。
   `BAND_WIDTH` への追加が `weird_act_0903` の committed 走・集計に触らないことも確認した
   （読み出しはすべて腕名を鍵にしている）。
4. **★ 段 B の対照は 3 腕とも `results/lr_a1_0901/verdict.csv` から取る。**
   `LR_A1` は `P2_log10_mean_unfit_level` の seed 別水準がそのまま入っているが、
   **`L2_A1` と `E_A1` の水準行は無い**。同ファイルの paired 差から seed ごとに再構成する:
   `L2_A1 = LR_A1 − P2_delta_log10_mean_unfit(LR_A1−L2_A1)`、
   `E_A1 = L2_A1 + P2_E_A1_reference_delta_log10_mean_unfit(E_A1−L2_A1)`。
   **seed ごとの引き算なので近似を入れない。** 検算: 再構成した U の中央値は
   `L2_A1` 0.02158・`E_A1` 0.00282・`LR_A1` 0.002621 で、vault §1 が引く
   0.022・0.0028・0.0026 と一致する（2026-09-03 確認）。
5. **段 B は宿主の 3 関数の写しを持つ。** `lr_a1_0901.setup_arm_lr` は `activation` の
   キーワードを持つが `_run_arm` がそれを渡さず、`write_arm_logs_lr` は
   `arm="LR_A1"`・`activation="leaky_relu"`・`negative_slope=SLOPE` を npz に**直書き**する。
   メタデータを正しく書くため、本走は `setup_arm_comb` / `_run_arm_comb` /
   `write_arm_logs_comb` を本モジュールに置く。**`src/lr_a1_0901.py` は 1 行も変えない。**
   写しであることは S-copy（`weird_act_0903` の様式）で機械的に検算する。
6. **段 B の config ブロック名は `comb_mlp2:`。** 宿主の `_p1_cfg` は `cfg["lr_a1"]` を
   `cfg["phase1"]` に写すので、本モジュールは同じ写しを `cfg["comb_mlp2"]` に対して持つ。
   宿主の `_arm(cfg)` は `name == "LR_A1"` を探すので使えない（本走の腕は `CB_A1`）。
7. **`preregistration` ブロックの 5 つの真は「この config を含む単独 commit そのもの」を指す**
   （`lr_a1_0901` と同じ扱い）。commit より前に実装・実行しない。
8. **テストは `unittest`**（この checkout に `.venv` も `pytest` も無い）。
   `src/test_comb_isolate_0903.py` を `python3 -m unittest src.test_comb_isolate_0903 -v` で回す。

## Log

- 2026-09-03 起票（v1）。vault `櫛の分離_spec_0903` v1（§7.1 記入済み・裁定 3 点決着）の
  段 A＋段 B を repo 側 spec に写した。**両段を本 commit で事前登録し、実行は順次。**
  §10 に実装段の追補 8 件（うち追補 1・2 は vault 字義の訂正、追補 4 は対照の転記路の確定）。
