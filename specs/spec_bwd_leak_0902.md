# spec_bwd_leak_0902: 順逆分離 × b-WD（forward と backward の片道性を別々に外す）

proj_004 / 作成 2026-09-02（vault v2 と同日） / run id: `bwd_leak_0902`
Obsidian 側の正本: `可塑性喪失/spec/逆伝播漏れ2x2_spec_0902.md`（v2・出典チャット `bwd_leak_0902`）
親: `specs/spec_gate_dose_0830.md`（実行済み・`GATE_LOAD_BEARING`） / 宿主実装: `src/gate_dose.py`
関連: `configs/bias_wd_0901.yaml`（b-WD の差し込み規約）／`src/channel_2x2_0901.py`（等価限界 D4 の実装先例）

> **状態: 凍結**（2026-09-02）。§7.1 の Issa 事前予測は vault 側で記入完了。この commit では **spec と config だけ**を追加する（運用ルール §2 / 9月運用_0901 §3-3 の 3 commit 構成）。
> 宿主は `gate_dose_0830`（1 層・オラクル用量固定・5M）をそのまま使い、**新規 8 腕**を段 1 / 段 2 に分けて足す。ReLU 腕・forward-leaky 腕は再走せず、`results/gate_dose_0830/verdict.csv` の seed 別 $U_k$ を対照に使う（§4.2）。
> **§6.3 は実装段の決定と追補である。** vault spec が実装段に委ねた分岐（S-log）と、起草時に気づかれていなかった不整合への guard をここに置く。**登録済みの判定基準は 1 つも緩めていない**（追加のみ）。

---

## 0. 一行

`gate_dose_0830` は「片道性が load-bearing」を示したが、片道を外す手段は **forward と backward を同時に変える**活性化（ELU・leaky）だった。沈んだ leaky ユニットは出力 $az$ を読み出し層に供給し続けるので、LoP が消えたのが「勾配で脱出できるから」なのか「沈んでも使えるから」なのかは現在の証拠では言えない（**★穴 1**）。本走は片道性を**別々に**外す: **順 ReLU／逆 leaky（`BL`）** は出力 0 のまま勾配だけ通し、**順 leaky／逆 ReLU（`FL`）** は出力 $az$ を供給するが勾配は通さない。これで担い手を決める（**V1**）。さらに `BL` に b-WD を掛けて、forward が 0 のときに消える self 復元力の明示的な代替になるかを見る（**V2**・折衷案）。

## 1. なぜ回すか

| 材料 | 出所 | 格 |
| --- | --- | --- |
| 帯内用量で ReLU 10/10、ELU・leaky 0/10。滑らかさは不要、片道でないことだけが効く | `results/gate_dose_0830/verdict.csv` | commit 済み出力 |
| 沈下は死ではない（ELU 0.36–0.45・leaky 0.63–0.67 でも LoP なし） | 同 | commit 済み出力 |
| **★穴 1**: 「落ちた先で**勾配**が消える」の証拠はすべて活性化関数ごと替えており、出力と勾配を同時に復活させている | vault v6 前半の穴レビュー（9/2） | 論証（未登録） |
| self 項の恒等式: 負側の上向き復元力は $\sigma\sigma' = a^2 z < 0$ から出る。**forward が厳密 0 なら $\sigma = 0$ で消える。** 同じ理由で $v_i$ への勾配 $\delta\cdot a_i$ も 0 | vault `selfterm_identity` §3 | 定理＋未検証予言 |
| b-WD（λ=1e-3）は centered で dead・機能劣化・b 暴走を同時に止めた | `results/bias_wd_0901/`（`BIAS_WD_PROTECTS`） | commit 済み出力（**2 層・centered 系**。1 層・std 系への移植は本走が初） |
| 先行: forward ReLU / backward 代替勾配は **SUGAR**（Horuz et al., arXiv:2505.22074, 2025-05）が提案済み | alphaxiv 要約（**原典未精読・数値は引かない**） | 外部・記憶級 |

**動機**: (i) ★穴 1 を埋める。`BL` 0/10・`FL` 10/10 なら v6 (i) は「活性化関数の話」から「逆伝播の話」に上がり、$m^- = \mathbb E[\varphi'\mid z<0]$ が主張から導かれた定義になる。逆なら (i) は「出力のゼロ化」の話で書き直し。(ii) `BL` が発症した場合に折衷案（b-WD）が成立するか。(iii) SUGAR に対し「LoP 文脈での機構分解（出力 vs 勾配、self vs rest）」が当方の差分。

## 2. 何を分けるか（未登録の推論。verdict には使わない）

$a=0.1$。沈んだユニット（$z\le0$）に残る経路:

| 経路 | `R`（順 ReLU／逆 ReLU） | `LR`（順 leaky／逆 leaky） | `BL`（順 ReLU／逆 leaky） | `FL`（順 leaky／逆 ReLU） |
| --- | --- | --- | --- | --- |
| rest 経路（$-2\eta v_i\,\mathrm{gate}\,G$、$w,b$ を動かす） | 0 | $a$ 倍 | $a$ 倍 | **0** |
| self 復元力（$\sigma\sigma'$ に比例） | 0 | 戻る | **0**（$\sigma=0$） | **0**（$\sigma'=0$） |
| $v_i$ の勾配 $\delta\cdot a_i$ | 0 | 戻る | **0（凍結）** | 戻る（$a_i=az$） |
| 読み出し層への供給 | 0 | $az$ | **0** | $az$ |

- `BL` が持つのは**勾配のみ**。`FL` が持つのは**出力のみ**（$az$ の供給と $v_i$ の学習。$w,b$ は吸収）
- **V1（担い手）**: `BL` 0／`FL` 発症 → 勾配が担い手。`BL` 発症／`FL` 0 → 出力が担い手。両方 0 → どちらでも足りる。両方発症 → 両方要る
- **V2（折衷）**: `BL` で発症し `BLW` で発症しなければ、失われた self 復元力を b への外的復元力が代替している

**本走が答えないこと**: $a$ の掃引・帯型代替勾配・Adam・多層・幅。無限時間の主張（0/10 は「5M までに観測しなかった」）。

## 3. スコープと前提

`gate_dose_0830` §3 を逐語継承: condA・幅 100・隠れ 1 層・$T=10^4$・batch=1・素の SGD・lr 0.01・seed 0–9・`total_steps=5e6`・device=cpu・オラクル用量固定・`eval_loss_exact` 併記・1M `state_hash`。床 **1e-16**（1 層系・再較正しない）。bootstrap seed **`20260905`**。

`generator_offset` は**明示的に 0**。本走は親走と同一 seed 集合・同一乱数系列を意図的に共有する設計（§4.2 の S-pair）なので、9月運用_0901 §3-1 の「新 seed 群は `generator_offset` で作る」は本走には適用しない。seed ラベルは乱数系列に入らないので、seed を `[10..19]` にしても同じ 10 実現が別ラベルで出るだけで、対応づけだけが壊れる。この非適用を `provenance.json` に記録する。

## 4. 設計

### 4.1 腕（新規 8 本 ＋ 対照 4 本は committed）

| 腕 | forward（$z\le0$） | backward（$z\le0$） | b-WD | 用量 | 出所 | 段 |
| --- | --- | --- | --- | --- | --- | --- |
| `R_933` / `R_1216` | 0 | 0 | なし | 9.33 / 12.16 | **committed** | — |
| `LR_933` / `LR_1216` | $az$ | $a$ | なし | 同 | **committed** | — |
| `BL_933` / `BL_1216` | 0 | **$a$（代替勾配）** | なし | 同 | **新規・V1 主対象** | **1** |
| `FL_933` / `FL_1216` | **$az$** | **0** | なし | 同 | **新規・V1 主対象** | **1** |
| `RW_933` / `RW_1216` | 0 | 0 | λ=1e-3 | 同 | **新規**（b-WD 単独の対照） | 2 |
| `BLW_933` / `BLW_1216` | 0 | $a$（代替勾配） | λ=1e-3 | 同 | **新規・折衷案** | 2 |

- **段**: 段 1 最小 = `BL_1216` + `FL_1216`（2 腕・約 15 分・9月運用_0901 §0-2 の 15 分級枠）／段 1 用量 2 点 = 4 腕・約 30 分／段 2 = 4 腕・約 30 分。**段の切り方と用量点数は §9 #0 で Issa が決める。回した段と用量は `provenance.json` に記録する**
- 用量 1 点なら **12.16** を推す（ReLU は 1M で既に 10/10 なので 1M 窓でも読める）
- $a=0.1$ は `gate_dose` の leaky と同値。**`BL`・`FL` と `LR` の差がそれぞれ「出力」「勾配」だけになる**ようにするため
- λ=1e-3 は `bias_wd_0901` の C4 値を継承。**1 層・std 系での較正はしない**

### 4.2 対応づけ

代替勾配・順逆の分離・b-WD はいずれも乱数を消費しないので、新規腕は `gate_dose_0830` の各腕と init・教師・入力列・flip 軌道が bit 一致する（S-pair）。**step 1 以降は軌道が分岐する**。paired 対比は `results/gate_dose_0830/verdict.csv` の seed 別 $U_k$（10 値）を相手に取る。引用時に「対照は別走の committed 値」と書く。

### 4.3 実装

- `VecMLPL` に 2 つの活性化を追加。閉形式のまま（autograd に切り替えない）。**ReLU 経路・leaky 経路は 1 行も触らない**
  - `act='bwd_leaky'`（`BL`）: 前向き `torch.relu(pre)`、後ろ向き `where(pre>0, 1, a)`
  - `act='fwd_leaky'`（`FL`）: 前向き `where(pre>0, pre, a*pre)`、後ろ向き `(pre>0)`
  - どちらも既存の `relu` / `leaky_relu` の式を**組み合わせるだけ**にし、新しい算術を書かない（S-limit・S-cross で検証）
- b-WD は `VecMLPL.set_weight_decay_b(1e-3)`（b のみ・W と v には掛けない）
- 用量固定は `dose_const_5m_0830` のオラクルモードそのまま

### 4.4 用語

- `BL` / `BLW` / `RW`: forward が ReLU なので `strict_dead`（$\hat p_i\equiv0$）がそのまま定義できる。`BL` / `BLW` は**吸収でない**（負側に勾配があるので復活しうる）
- `FL`: $\hat p_i$ は backward ゲートの開口率で、`strict_dead` は $w_i,b_i$ が**吸収された**ユニットを指す。**出力は $az$ で非ゼロ・$v_i$ は学習を続ける**
- `LR` も同様に「沈下 ≠ 死」。P7a の表で `R` 系と同じ列に並べるときは注記する
- どちらも verdict には使わない

## 5. 集計（事前登録）

窓・床・発症定義・CI は `gate_dose` §5 を逐語継承。**窓はタスク終端の記録点だけ**（5M 窓 = `step % 10000 == 0` かつ $4.9\times10^6 <$ step $\le 5\times10^6$ の 10 点。100 点でも 101 点でもない）。$U_k$ = 窓平均未フィット率、**発症 = $U_k\ge0.05$**、$n_{\rm onset}$ は Clopper–Pearson、差は seed クラスタ paired bootstrap B=10000・percentile 主・studentized 従。

### 5.1 主 endpoint — 発症数（5M）

各腕の onset 状態を 3 値に潰す: **zero**（回した用量すべてで $n_{\rm onset}=0$）／**present**（いずれかで $\ge5$）／**mid**（どちらでもない）。用量 1 点で回した場合はその 1 点で判定し、引用時に用量を添える。

**V1（担い手・★穴 1）**: `BL` × `FL`。

| `BL`（勾配のみ） | `FL`（出力のみ） | V1 |
| --- | --- | --- |
| zero | present | **`GRADIENT_CARRIES`** |
| present | zero | **`OUTPUT_CARRIES`** |
| zero | zero | **`EITHER_SUFFICES`** |
| present | present | **`BOTH_REQUIRED`** |
| 上記以外（mid が混じる） | | **`PARTIAL`** |

**V2（折衷・b-WD）**: `BL` が zero なら **`NOT_APPLICABLE`**（`RW` は C4 の移植確認として REPORT）。それ以外は `BLW` × `RW`。

| `BLW` | `RW` | V2 |
| --- | --- | --- |
| zero | present | **`RESTORING_FORCE_REQUIRED`** |
| — | zero | **`WD_B_SUFFICIENT_ALONE`** |
| present | — | **`COMPROMISE_FAILS`** |
| 上記以外 | | **`PARTIAL`** |

前提: committed の `R_*` は 10/10・`LR_*` は 0/10。実測がこれと違えば結果を読まない。**飽和ガード**: 二値が 0 に揃った場合は §5.2 を同格で報告する。

### 5.2 水準（主判定と同格）

各腕の seed 中央値 $\log_{10}U(5\mathrm{M})$・$\log_{10}U(1\mathrm{M})$。paired Δ $\log_{10}U(5\mathrm{M})$（用量ごと）:

- **P3'**: `BL`−`R`、`FL`−`R`、`BLW`−`R`、`RW`−`R`
- **P5（V1 の水準側）**: `BL`−`LR` と `FL`−`LR`。**等価限界 $\Delta=0.15$ dex**（`spec_channel_2x2_0901` D4 を継承）。**書かれた順に判定する**: CI が $[-0.15,+0.15]$ に収まれば **`*_EQUIV_FWD`**、CI が 0 を上に外せば（$\mathrm{ci\_lo}>0$）**`*_SHORT_OF_LR`**、どちらでもなければ **`INCONCLUSIVE_WIDE`**
  - 符号の向き: `BL`−`LR` $>0$ は $\log_{10}U(BL) > \log_{10}U(LR)$、すなわち `BL` の方が**合っていない** ＝ forward の非ゼロ出力に利がある
- **P6（交互作用）**: (`BLW`−`BL`) − (`RW`−`R`)。**機構の証拠には使わない**

### 5.3 副次（REPORT_ONLY）

- **復活数**: `BL` / `BLW` で $\hat p_i$ が 0 → 正へ戻ったユニット数・イベント数・率（教訓⑰: 総和と率の両方）。**`FL` は 0 のはず**（backward が片道）— 0 でなければ実装を疑う
- **凍結率**: `BL` で末尾窓中 $\Delta v_i = 0$ のユニット割合。`FL` で $\Delta w_i = \Delta b_i = 0$ のユニット割合。どちらも `strict_dead` 集合と一致するはず
- **`FL` の固定特徴の使われ方**: 沈んだユニットの $|v_i|$ の末尾窓中央値（`R` の dead ユニットと比較）
- `strict_dead` / `submerged_frac` / `alive` / `eff_rank` / `‖w‖` / `b_median` / `median_M` / `median_B` / `preact_sd_median` の時系列、`eval_loss_exact`。**`‖w‖` は主要共変量**
- 数値発散は `gate_dose` §5.6 を継承（→ §6.3 追補 6 で判定ごとの必要セルを書く）

### 5.4 副次 P7 — ユニット別 $s_i = M_i + B_i$ の分布（登録・verdict 不使用）

$M_i = w_i\cdot\mu/\mathrm{denom}_i$、$B_i = b_i/\mathrm{denom}_i$、$\mathrm{denom}_i = \sqrt{w_i^\top\Sigma w_i}$。用量は操作変数、$s_i$ は読み出し。

- 集約順（凍結・→ §6.3 追補 5 で時間軸を明示）: ユニット → **記録点ごとの seed 内中央値**（全ユニット／$\hat p_i>0$ のみ の 2 系列）→ **窓内記録点の平均** → seed 中央値。`median_s` と `median_M + median_B` は**別物として両方出す**（引用禁止 縮退 (ii)）。`denom` の中央値を併記（縮退 (iii)）
- **P7a**: 各腕の `median_s` / `median_M` / `median_B`
- **P7b（漏れが戻すチャネル）**: paired Δ（`BL` − `R`）の `median_M`・`median_B`。CI が 0 を外したチャネルを「戻したチャネル」と読む。**ラベル化はしない**
- **P7c（b-WD の係留）**: `RW` − `R`、`BLW` − `BL` の `median_B` と `median_M`
- **P7d**: `BLW` − `BL`、`BLW` − `RW`、`FL` − `BL`（新規腕のみで閉じる）
- $s_i$ は「位置」であって「病理」ではない。`BL` の位置が `R` と同じ深さでも復活数が正なら、差は位置ではなく可動度にある

## 6. 前段チェック・出力・コスト

- **S-pair（必須）**: 新規腕の step 0 で init・教師・入力列・flip が `gate_dose_0830` と bit 一致（30k 短走）。**対応は seed ごとの init ハッシュで取る**（位置合わせではない）
- **S-limit（必須）**: `bwd_leaky` と `fwd_leaky` で $a=0$ が ReLU 経路と 30k 短走で bit 一致。`set_weight_decay_b(0)` が無介入と bit 一致
- **S-cross（必須）**: `bwd_leaky` の前向きが `relu` の前向きと、後ろ向きが `leaky` の後ろ向きと、それぞれ同一入力で bit 一致。`fwd_leaky` は逆の組合せで同様
- **S-dose（必須）**: 実測 ‖µ‖ が目標と相対誤差 $10^{-10}$ 以内
- **S-bwd（必須）**: 有限差分照合は**不可**。§6.3 追補 2 の 5 点を単体テストで確認
- **S-wd（必須）**: step 1 で `RW` と `R` の差が b のみ、かつ $\Delta b = -\eta\lambda b$ に一致
- **S-taut**: 判定指標（未フィット率）が介入で定義上恒真になっていない
- **S-log（必須）**: → **分岐 A**（§6.2）
- **S-log-b（必須）**: 追加ロガーは軌道中立。30k 短走でロガー有無の 2 走が bit 一致
- **S-ref（追加）**: 対照として読む `results/gate_dose_0830/` の各ファイルが親 `provenance.json` の `output_sha256` と全数一致し、親 commit がリモートに在る（運用ルール §2 の push 監査）
- S1: `OMP_NUM_THREADS=1`。S6: 床は継承。**本走前に §5 の集計はしない**

出力: `results/bwd_leak_0902/stage{1,2}/` に `verdict.csv`・`summary.md`・`layer_stats.csv`・`s_distribution.csv`・`revival.csv`・`provenance.json`（親 commit SHA・**S-log の分岐と理由・回した段と用量**を記録）・`logs/`・`arm_status/`。**数値は `verdict.csv` と `summary.md` からのみ転記**（§6.3 追補 4 の carve-out を除く）。

コスト（1 腕 ≈ 8 分）: 段 1 最小 ≈ 15 分／段 1 用量 2 点 ≈ 30 分／段 2 ≈ 30 分／全部 ≈ 60–65 分。分岐 A なので対照の再走はゼロ。

### 6.1 S-log の分岐調査（作業表 #2b・実施済み 2026-09-02）

**結論: 分岐 A。走の追加なし。P7a–P7d すべて実施可能。**

`results/gate_dose_0830/logs/*.npz`（9 腕 × 10 seed = 90 本、欠けなし）に、腕水準の集約値だけでなく**ユニット別の配列がすでに保存されている**。`elu_swamp.exact_layer_record_elu` が `M` / `B` / `denom` / `p_hat` / `w_norm` / `zbar` / `dzbar` をユニット別に返し、`gate_dose.write_arm_logs_gate` が `layer1_*` として `(5001, 100)` float32 で書き出している。実測:

- 対照 `R_933` / `R_1216` / `LR_933` / `LR_1216` の 4 腕すべてで 6 配列が揃っている
- `M + B` と `zbar / denom` の一致は最大絶対差 1.4e-6（float32 の往復ノイズ）。float64 での恒等式照合は親 `provenance.json` の `sanity` に `wall` 2.1e-15〜2.9e-15 として記録済み
- NaN は 5001 記録点 × 100 ユニット × 10 seed で **0 件**（`denom` の最小値が 8.5e-2 で `sigma_tol=1e-8` から 6 桁以上の余裕）

新規腕は同じ `GateRecorder` 経路を通るので、**これらの列は追加のロガーなしに出る**。分岐 A でロガーの新規追加が要るのは §5.3 の凍結率のためだけ（`v_unit` / `b_unit`）。S-log-b はその 2 列だけを検査する。**既存列は 1 列も変えない・消さない。**

vault spec §5.4 の「`gate_dose` §5.5 に載っているのは腕水準の `median_M` / `median_B` であり、ユニット別列があるとは書かれていない」は spec 側の記述の不足であって、実装にはユニット別列がある。

### 6.2 実装段の決定と追補

**登録済みの判定基準は 1 つも緩めていない。以下はすべて追加である。**

**追補 1（P5 の等価限界と検出力）。** $\Delta=0.15$ dex は `channel_2x2_0901` の D4 からの継承で、そこでは **2 層・centered・床 1e-23** の系で `mean(log10 unfit)` に対して較正された値である。本走は **1 層・std・床 1e-16** で、統計量も `log10(mean U)` と集約順が逆。委任した監査の診断計算では、committed ログの末尾窓で `LR_933` の `median log10(mean U)` = −1.785 に対し `median mean(log10 U)` = −2.949 で、集約順だけで 1.16 dex 動く（**診断値。転記対象ではない**）。さらに committed の seed 別 $U_k$ から作った近い対比（`LR_1216`−`LR_933` など）の paired CI 幅は 0.97–1.54 dex で、$n=10$ で $\pm0.15$ に収めるには sd(diff) $\lesssim 0.21$ dex が要る。**`*_EQUIV_FWD` はこの $n$ ではほぼ到達不能である可能性が高い。** 対応:

- 登録どおり $\Delta=0.15$ で判定する（**較正し直さない**。vault spec は「Issa 上書き可」としているので、上書きするなら本走前に）
- `p5_margin_recalibrated_for_this_system: false` を config に明記し、`summary.md` に上の事情を書く
- **較正定数の要らない seed 別符号検定を REPORT_ONLY の併記として事前登録する**（現象3存在確認_P2b の G0 で採用済みの道具）。結果を見てから足すのは事後になるので、いま登録する
- CI が丸ごと 0 の**下**にある場合（その経路だけの方が `LR` より良い）は登録ラベルが無く `INCONCLUSIVE_WIDE` に落ちる。情報を落とさないため `p5_ci_below_zero` をフラグ列として出す

**追補 2（S-bwd の操作化）。** vault spec §6 の字義は `BL` について「$\Delta w,\Delta b = a\cdot$(leaky 腕の負側更新)」。**この字義どおりの等式は成立しない**: leaky 腕は forward も変わるので、同一状態・同一入力でも $\hat y$ が違い $\delta = \hat y - y$ が違う。比の基準として leaky 腕は使えない。確認したい代数は保ったまま、比の基準を「同じ `bwd_leaky` 経路の $a=1$（全開）」に取り直す。負側ユニットを 1 個手で置いて:

- **(a)** `BL`: $\Delta v_i = 0$ が厳密（負側は forward が 0 なので $g_{v_i} = 2\delta a_i = 0$）
- **(b)** `BL`: $\Delta b_i,\Delta w_i$ が閉形式 $-\eta\,2\delta\,v_i\,a\cdot(1, x)$ と bit 一致
- **(c)** `BL`: 負側更新が $a$ に**厳密比例**（forward が ReLU なので $\delta$ が $a$ に依存せず、$a=1$ の全開更新のちょうど $a$ 倍。$a \in \{0, 0.1, 0.2, 1\}$）
- **(d)** `FL`: $\Delta w_i = \Delta b_i = 0$ が厳密
- **(e)** `FL`: $\Delta v_i = -\eta\,2\delta\,az$ と bit 一致

**追補 3（窓の点数）。** 5M 窓は**タスク終端の 10 記録点**であって、窓内の全 100 記録点ではない。委任した監査が committed `verdict.csv` の $U_k$ をこの定義で最大相対誤差 3.1e-16 で再現し、記録点 100 点で取ると `LR_933` で 199 倍ずれることを実測した。新規腕の $U$ も**同一の 10 点定義**で作る（宿主の `_window_indices` をそのまま使う）。

**追補 4（対照を logs から読むことの carve-out）。** §6 の「数値は `verdict.csv` と `summary.md` からのみ転記」は主 endpoint への規則である。P7b / P7c と §5.3 の対照はユニット別 `M`/`B`/`p_hat` を要し、これは `results/gate_dose_0830/logs/{arm}_seed{k}.npz` にしかない。**この 1 経路を明示の例外として登録し**、引用時にファイル名・配列名・記録点の選び方・seed 数まで書く（運用ルール §5 教訓⑫ 8/30 追補）。なお当該 npz は `.gitignore` 対象で commit されていないため、fresh clone では P7b / P7c を再現できない。S-ref で親 `provenance.json` の `output_sha256` と全数照合したうえで使う。

**追補 5（P7 の時間軸）。** vault spec の集約順「ユニット → seed 内中央値 → seed 中央値」には時間軸が無い。記録点は 5001 点あるので縮約の順序が決まらない（教訓⑫: 窓の取り方で 1.39 → 0.74 が動いた前例）。**窓を末尾窓（タスク 491–500）に固定し、時間縮約を「記録点ごとに seed 内中央値 → 窓内記録点の平均」として集約順に挿入する。**

さらに:

- **$\hat p_i>0$ のみの系列は `R` 腕でほぼ推定不能**。委任した監査の実測（診断値）では、末尾窓の `R_1216` は 100 個の (seed, 記録点) セルのうち 20 個で $\hat p_i>0$ のユニットが 0 個、seed 3 は窓全体で 1 観測しかない。**seed あたり最小ユニット数 3 を登録し**（`q2_min_submerged_units_per_seed` の先例）、未満は `INSUFFICIENT_DATA` とする。全ユニット系列にはこの問題は無い
- 「alive」という語は `LR` / `FL` では死を意味しないので、系列名は **`p_hat_positive`** とする
- `median_M` は log 量ではないので、P7c の「等価限界 0.15 内」は **dex ではない**。`p7c_equivalence_margin_s_units: 0.15` として別名で登録し、`summary.md` に「P5 の 0.15 dex とは無関係の数・較正していない」と書く
- 正規化量だけで力学を書かないため、**生の `zbar` / $w\cdot\mu$ / `b` / `denom` を全系列に併記する**（現象3指標再定義 D-5・引用禁止 C）
- 対照の `median_M` / `median_B` は **committed の腕水準列を使わず、ユニット別配列から両系列とも計算し直す**（committed 列は全 100 ユニットの `nanquantile` なので、$\hat p_i>0$ 系列と集約 scope が混ざる）
- 縮退 (i) が非該当な理由は「中心化腕が無いから」ではない（config はどの帯内腕にも `centered_layers: [1]` を置く）。**介入が `oracle_fixed_mu_offset` で ‖µ‖ が用量に固定されており EMA 中心化ではない**からである。ただし**µ の向きはタスク境界ごとに回る**ので、$M_i$ は凍結したユニットでも動く。`median_M` の変化を「ユニットが動いた」と読まない

**追補 6（復活数の定義）。** vault spec §5.3 は「`gate_dose` の ReLU 腕は 0 が既知」と書くが、**記録点だけで数えるとこれは成り立たない**。タスク境界で 32 パターンの厳密支持が引き直されるので、凍結した dead ユニットが 1 ステップも動かないまま $\hat p_i>0$ になる。committed ログでの実測（**診断値。転記対象ではない**）:

| 腕 | within-task 復活 | 境界越え復活 |
| --- | ---: | ---: |
| `R_933` | 0 | 14801 |
| `R_1216` | 0 | 6584 |
| `LR_933` | 1855 | 40687 |
| `LR_1216` | 3959 | 43879 |

したがって「ReLU 腕は 0」が成り立つのは **within-task**（同一タスク内かつ `flip_state` 不変）の定義のみ。**主は within-task とし、境界越えは併記する。**

**追補 7（V1/V2 の判定表の実装）。** ワイルドカード「—」入りの表を順序規則で実装すると、`mid` を含むセルが `PARTIAL` を飛ばして先の行に当たりうる。**3 値 × 2 腕の 9 セルを config に全列挙して凍結し**、解析コードは順序を再実装せずその写像を引く。あわせて、当たった行以外に**条件を満たしていた行**も `verdict.csv` に残し（`co_satisfied_labels`）、生の onset 三つ組を `summary.md` に逐語で出してラベルを再導出可能にする。

**追補 8（P6 の扱い）。** vault spec は P6 に閾値も CI 規則も与えていない。加えて `R` 系のベースラインは天井付近（committed `R_933` の `median_log10_U_5m` = −0.215）なので、`RW`−`R` には改善余地が 2 dex 以上あり `BLW`−`BL` にはほとんど無く、**構成上すでに天井汚染される**（`channel_2x2_0901` §8.1 の制限 3 と同型）。対応: **P6 はラベルを付けず CI のみ REPORT_ONLY** とし、早期窓の腕間ベースライン広がりを測って閾値超えなら「P6 を単独で読むな」を `summary.md` に出す（S-ceiling 相当）。また P6 の向きは `channel_2x2_0901` の交互作用と**逆**なので、あちらのラベル語彙は 1 語も持ち込まない。

**追補 10（回した段と用量・9/2 Issa 裁定）。** §9 #0 は **全 8 腕・帯内 2 用量の一括**（約 64 分）と決まった。9月運用_0901 §0-2 の「現象1 の M 級新規走は今月回さない」に対する例外として Issa が認めたもので、`ident_mu_2x2_0901` と同じ扱い。§3-6「長い走ほど早く投入する」を適用する。`--stage all --doses both`。回した段と用量は `provenance.json` の `stage_run` / `doses_run` に記録される。

**追補 11（P5 の $\Delta$ 据え置き・9/2 Issa 裁定）。** 追補 1 の検出力の事情（`*_EQUIV_FWD` は $n=10$ ではほぼ到達不能で、`INCONCLUSIVE_WIDE` 予測はほぼ恒真になる）を提示したうえで、**$\Delta=0.15$ のまま回す**と Issa が裁定した（本走前）。符号検定の REPORT_ONLY 併記と `p5_ci_below_zero` フラグはそのまま置く。

**追補 9（`BLW`−`BL` の水準対比）。** §7.1 の Issa の水準側予測「`BLW`−`BL` の CI が 0 を下に外す」を照合するには、新規腕どうしの差の paired CI が要る（P3' の 2 本の CI からは作れない）。`p3prime_delta_contrasts` として登録する。

## 7. 事前予測

### 7.1 Issa（2026-09-02 記入・**記入完了**・vault が正本）

- **V1（`BL_*` の発症数）**: **0**（回した用量すべてで $n_{\rm onset}(5\mathrm{M})=0$）
- **V1（`FL_*` の発症数）**: **≥5**（発症する）
- **→ V1 ラベル: `GRADIENT_CARRIES`**
- **V2: `NOT_APPLICABLE`**（`BL` が 0 なので折衷は不要）
  - ただし水準では `BLW` が `BL` より良いと予想: **`BLW`−`BL` の CI が 0 を下に外す**（→ 追補 9）
- **P5（`BL`−`LR`）: `INCONCLUSIVE_WIDE`**
- **P5（`FL`−`LR`）: `FL_SHORT_OF_LR`**（`LR` の方が良い ＝ CI が 0 を上に外す）
- 外れたときに第一に疑うもの: 記入なし

> 9/2 の当初記入は V2 = `RESTORING_FORCE_REQUIRED` だったが、これは水準の予想を発症数の欄に書いたもので同日中に訂正（Issa）。訂正前の記録は vault §7.1 に一行で残っている。

### 7.2 Claude（2026-09-02 記入・実行前）

- **V1: `GRADIENT_CARRIES`**（`BL` 0/10・`FL` 10/10）。理由: `gate_dose` で沈下 0.63–0.67 の leaky が 0/10 なので、逃げるのに要るのは「落ちた先で勾配が消えないこと」であり rest 経路だけで足りる可能性が高い。`FL` は $w,b$ が吸収されるので固定特徴の線形読み出しにしかならず、flip が替わる condA では合わせ切れない
- P5（`BL`−`LR`）: **`BL_SHORT_OF_LR`**。理由: `BL` は $v_i$ が凍結するので、復帰したユニットの出力側が古いまま
- P5（`FL`−`LR`）: `FL_SHORT_OF_LR`。ただし `FL`−`R` は負
- V2: `NOT_APPLICABLE`
- `RW`: 2 用量とも 0。**`RW` が 0 でなければ** C4 のスコープ制限になる

### 7.3 外れたときに第一に疑うもの（Claude）

(i) S-bwd・S-cross（順逆の組合せの実装ミス。`FL` で within-task 復活数 > 0 ならここ）。(ii) λ=1e-3 の 1 層・std 系への非移植性。(iii) 発散。(iv) `OUTPUT_CARRIES` なら §5.3 の $|v_i|$ 中央値で「固定特徴として使われている」を確認。(v) `BOTH_REQUIRED` なら self 恒等式の負側項を `LR` の committed ログで実測。

## 8. 引用上の注意

- 0/10 は「5M までに観測しなかった」（片側 95% 上限 $p\le0.2589$）
- 対照 `R` / `LR` は**別走の committed 値**（§4.2）。同一走の腕ではない
- 用量 1 点で回した場合は**用量を必ず添える**。V1 は 1 点なら 1 点の主張
- `FL` の `strict_dead` は「$w,b$ の吸収」であって「出力ゼロ」ではない
- 復活数は **within-task** の定義でのみ「ReLU 腕は 0」（追補 6）
- P5 の $\Delta=0.15$ dex は**この系で較正し直していない**継承値（追補 1）。P7c の 0.15 は dex ではない別の数（追補 5）
- P6 と P7 は verdict に使わない
- SUGAR は原典精読まで数値を引かない
- スコープ: condA・1 層・w100・$T=10^4$・batch=1・lr 0.01・$a=0.1$・λ=1e-3・用量 9.33 / 12.16（回した分）

## 9. 作業表

| # | 作業 | 走 | 状態 |
| --- | --- | --- | --- |
| 0 | 段の切り方と用量点数を Issa が決める（§4.1） | なし | **9/2 決定 → 全 8 腕・帯内 2 用量を一括**（約 64 分・§6.2 追補 10） |
| 1 | §7.1 を Issa が記入 | なし | **9/2 記入完了**（V1 `GRADIENT_CARRIES` / P5 `FL_SHORT_OF_LR`） |
| 2 | 本 spec と config を repo へ単独 commit・push | なし | 本 commit |
| 2b | S-log の分岐調査 | なし | **完了 → 分岐 A**（§6.1） |
| 3 | `bwd_leaky`・`fwd_leaky` 追加・全 sanity | なし | **完了**（`75fa522`・preflight 15 gate PASS） |
| 4 | 段 1 本走 | **あり** | #0 の裁定により段 2 と一括 |
| 5 | 段 2 本走 | **あり** | 同上 |
| 6 | SUGAR 原典精読 | なし | 未着手 |
| 7 | V1 が出たら v6 (i) と §4 $m^-$ の文言を確定 | なし | 未着手 |
