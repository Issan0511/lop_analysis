# sna_cnn_cause_1009 — CNN で適応 Snake が固定 Snake に負ける原因

依頼（Issa、2026-10-09 /goal）: 「適応型 snake が普通の snake より悪かった原因を解明して」。
書いた時点（本走 A を起動した直後で、ログはまだ一行も読んでいない）で分かっていることと、問い・腕・判定・予測を登録する。

## 0. 既知（このノートの前に読んだもの）

- 箱は RL-CIFAR CNN（`rlcifar_cnn_0908`）。窓（t31–50 の online_acc）は SNA 0.946、SN3 0.964、SNAc3 0.962、SN06 0.958（各 10 seed、seed 0–9）。
- **負けの中身は可塑性の喪失**。早期（t1–10）は SNA 0.971 > SN3 0.965 で、SNA のほうが速く覚える。低下（早期 − t41–50）は SNA +0.028、SN06 +0.016、SN3 +0.002、SNAc3 +0.001。
- swish_battle_0917 では、c=3 で差の 88% が消えた（C_VALUE）。SNAc3 の conv の α は上限 3 に張り付き（＝固定 α=3）、fc は 2αW=6 になる。
- avgpool_cnn_0917 によれば、conv2 の谷越えは max-pool が起こす。ただし avg-pool でも SNA の低下は +0.021 で、谷越えは劣化の主因ではない。
- 既存の課題ごとの記録から見たこと:
  - SNA の前活性の幅（zsd）は、t1 の時点で c2・f1・f2 が SN3 の 3〜4 倍ある。
  - f3（読み出し）の行ノルムは SNA だけ伸び（1.14→2.31）、SN3（1.79→1.00）・SNAc3・SN06 では縮む。
  - c2 の 2αW は、劣化する腕（SNA 1.2・SN06 0.68→1.95）では 2 以下に留まる。劣化しない腕（SN3 1.7→5.1・SNAc3 1.7→3.4）では 3 以上に育つ。
  - c1 の 2αW は 4 腕とも 1 前後で差が無い。
- MLP の箱（RL-MNIST・RL-CIFAR MLP）では、SNA（c=0.6）は劣化しない。

## 1. 問い

- **Q1** 劣化を作るのは、どの層の c=0.6 か（conv か fc か）。
- **Q2** 劣化を作るのは、α の追随か、課題 1 で決まる状態か。追随とは、W が育つと α が縮み、2αW=1.2 が保たれることを指す。
- **Q3** チャネル単位の pooled 分散（位置ごとの平均のばらつきが W に混ざる）は効いているか。

## 2. 腕（`src/sna_cnn_cause_1009.py`、束ねエンジン、seed 10–19、50 課題 × 400 epoch、宿主のプロトコル）

| 腕 | c1 | c2 | f1 | f2 | 備考 |
|---|---|---|---|---|---|
| SNA | c 0.6 | c 0.6 | c 0.6 | c 0.6 | 宿主の SNA |
| SNAc3 | c 3 | c 3 | c 3 | c 3 | conv は上限 3 に張り付く |
| CV06FC3 | c 0.6 | c 0.6 | c 3 | c 3 | conv だけ c=0.6 |
| CV3FC06 | c 3 | c 3 | c 0.6 | c 0.6 | fc だけ c=0.6 |
| SNAfrz1 | c 0.6 | c 0.6 | c 0.6 | c 0.6 | 課題 1 の終わりで全チャネル・ユニットの α を凍結 |
| SNApp | c 0.6 | c 0.6 | c 0.6 | c 0.6 | conv の W を位置内分散（位置ごとのバッチ分散の位置平均）から作る |

- 束 A は SNA・SNAc3・CV06FC3・CV3FC06（40 本）、束 B は SNAfrz1・SNApp（20 本）。
- seed 10–19 は未使用の seed。宿主の seed 0–9 の値は「既知」で使ってしまったので、比較はすべて束ねエンジンの中で行う。
- 検査 8 本（`results/_checks_sna_cnn_cause_1009/checks.json`）は all_pass。宿主との 1 step の損失は完全一致、勾配の差は 1e-7。束ねた走どうしの独立は bit で確認した。CUDA グラフと eager は bit 一致する。

## 3. 量

- **窓** = t31–50 の online_acc の seed 平均。**低下** = t1–10 の平均 − t41–50 の平均。
- **差** gap = 窓(SNAc3) − 窓(SNA)。**回復率** q_X = (窓(X) − 窓(SNA)) / gap。
- 機構の記録は次のとおり。
  - 課題ごとに、課題頭の新ラベルでの CE（switch_ce）と、epoch ごとの学習曲線。
  - 層ごとに、W・α・2αW・座席 2αz̄・mob・mob_pool・eff_rank・行ノルム・f3 の Frobenius ノルム。
  - 走ごとのチェックポイント（t1,2,3,5,10,20,30,40,50 の重み・Adam・V）。

## 4. 判定（閾値は雑音から導く）

- 宿主の seed 間 SD は、窓で SNA 0.0046・SNAc3 0.0026。対の差の SE は宿主の SN3−SNA で 0.0015、gap は 0.016。したがって q の SE は約 0.1 で、0.75 と 0.25 は 1 と 0 からそれぞれ 2.5 SE 離れた位置になる。
- **E0 エンジン較正**: ENGINE_OK は次の 3 条件をすべて満たすとき。
  - 窓(SNA) が宿主 0.9458 から ±0.004 以内（2 群の平均差の 2 SE = 2·√(0.0046²/10·2)）。
  - 窓(SNAc3) が宿主 0.9618 から ±0.0025 以内。
  - gap ≥ 0.008（宿主の半分）。
  - 満たさなければ ENGINE_DIFFERS とし、Q1〜Q3 は束の中の比較としてだけ読む。
- **Q1**（q_conv = q(CV3FC06)、q_fc = q(CV06FC3)）:
  - CONV_LOCUS: q_conv ≤ 0.25 かつ q_fc ≥ 0.75（conv に 0.6 があれば劣化し、fc を 3 にしても救われない）。
  - FC_LOCUS: q_fc ≤ 0.25 かつ q_conv ≥ 0.75。
  - SUFFICIENT_EACH: 両方 ≤ 0.25（どちらの層の 0.6 も単独で劣化させる）。
  - JOINT: 両方 ≥ 0.75（両方そろって初めて劣化する）。
  - PARTIAL: それ以外。
- **Q2**: q(SNAfrz1) ≥ 0.75 なら FROZEN_RESCUES、≤ 0.25 なら FROZEN_NO、それ以外は FROZEN_PARTIAL。
- **Q3**:
  - 操作確認として、SNApp の c2 の pooled 2αW の窓平均が SNA の 1.2 から 10% 以上離れなければ PP_NOT_MANIPULATED とする（位置平均のばらつきが小さく、何も変わっていない）。
  - 操作が効いていれば、q(SNApp) ≥ 0.75 で POOLING_CAUSE、≤ 0.25 で POOLING_NOT、それ以外は POOLING_PARTIAL。
- どの判定でも、符号の一貫性（seed ごとの向き）を符号検定で併記する。

## 5. 予測（Claude、本走 A 起動直後・結果を読む前）

- E0 = ENGINE_OK: 0.85
- Q1: CONV_LOCUS 0.50 / FC_LOCUS 0.20 / SUFFICIENT_EACH 0.15 / JOINT 0.05 / PARTIAL 0.10。
  理由は二つある。
  - SN06（fc は洗われて mob≈1、conv は小さな 2αW）も劣化する。
  - MLP（fc だけ）の SNA は劣化しない。
- Q2: FROZEN_RESCUES 0.25 / FROZEN_PARTIAL 0.45 / FROZEN_NO 0.30。
  凍結しても c2 の W は 2 倍程度しか育たないので、c2 の 2αW は 1.2→2.3 止まりと読む。
- Q3: POOLING_CAUSE 0.25 / POOLING_PARTIAL 0.35 / POOLING_NOT 0.40。
  PP_NOT_MANIPULATED は 0.15（全体の中の割合で、上の 3 つはそれ以外の場合の配分）。
- 機構の予測（記述的、登録はしない）:
  - CV06FC3 では f3 の行ノルムは伸びない。
  - 劣化する腕では、t1 からの低下が switch_ce の増加と並ぶ。

Issa の予測は取っていない（/goal の依頼で、問いの登録の前に聞いていない）。

## 6. 次（結果しだい）

- CONV_LOCUS なら、c1 と c2 を分ける（S:c3-c0.6-c3-c3 と S:c0.6-c3-c3-c3）。機構はチェックポイントからの介入（α を差し替えて数課題回す）で詰める。
- FC_LOCUS なら、f1 と f2 を分ける。CNN の fc と MLP の fc の違い（入力が max-pool 後の正に偏った特徴であること）を測る。

## 7. 追補 1（2026-10-09 19:55、本走 A の t1–t10 を見たあと・束 B と fork の走の前）

### 7.1 見たもの（開示）

本走 A の t1–t10 の課題ごとの値を見た。online の seed 平均は次のとおり。

| 腕 | t1 | t5 | t10 |
|---|---|---|---|
| SNA | 0.977 | 0.970 | 0.968 |
| SNAc3 | 0.972 | 0.959 | 0.966 |
| CV06FC3 | 0.973 | 0.977 | 0.976 |
| CV3FC06 | 0.975 | 0.976 | 0.969 |

- 劣化は fc の c=0.6 についてきているように見える。§4 の Q1 の判定（窓 t31–50）は変えない。
- SNAc3 は、宿主と同じく t3–t9 に一時的に落ち込む。同時に mob_pool_c2 が 0.80→0.56 に下がる（conv を c=3 で上限に張り付けた c2 のゲート）。
- 課題 5 のチェックポイントでも、c2 の位置間分散の割合は 13%（α にして 7%）だった（`varsplit.py`、5 seed）。

### 7.2 計画の変更

- **SNApp は回さない**。理由は二つ。
  - 測定（t1・t5）で、c2 の √(pooled/位置内) は 1.07 だった。§4 の Q3 の操作確認（pooled 2αW が 10% 以上動く）に届かない見込みが高い。
  - conv は劣化の層ではない形勢である。
  - Q3 は測定で答える: 「チャネル単位の pooled 分散に混ざる位置平均の差は c2 の分散の 13%、α にして 7%」。
- **Q2 の判定を差し替える**（SNAfrz1 はまだ走っていない）。
  - 元の基準は窓 t31–50 で、SNAc3 を基準にした q だった。
  - SNAc3 の序盤の落ち込みが基準を濁すこと、束 B を 30 課題で回すことから、§7.4 の基準に替える。
- 束 B の腕は、すべて conv を c=0.6 に揃える（SNAc3 の序盤の落ち込みを避けるため）。

### 7.3 束 B（`results/sna_cnn_cause_1009/B`、seed 10–19、**30 課題**）

| 腕 | c1 | c2 | f1 | f2 | 備考 |
|---|---|---|---|---|---|
| F1ONLY | c 0.6 | c 0.6 | c 0.6 | c 3 | fc のうち f1 だけ 0.6 |
| F2ONLY | c 0.6 | c 0.6 | c 3 | c 0.6 | fc のうち f2 だけ 0.6 |
| SNAfrz1 | c 0.6 | c 0.6 | c 0.6 | c 0.6 | 課題 1 の終わりで α を凍結 |
| SNA+fixconv | c 0.6 | c 0.6 | c 0.6 | c 0.6 | conv は初期値のまま（Adam の歩幅を 0 に）、fc だけ学ぶ |
| CV06FC3+fixconv | c 0.6 | c 0.6 | c 3 | c 3 | 同上で fc は c=3 |

**fork**（`results/sna_cnn_cause_1009/F30`、束 A の t30 のチェックポイントから 10 課題、seed 10–19）:

| fork | 中身 |
|---|---|
| SNA>SNA | 対照 |
| SNA>CV06FC3 | fc の c を 3 に |
| SNA>CV3FC06 | conv の c を 3 に |
| SNA>SNA@f3x0.5 | 読み出しを 0.5 倍（logit を半分にする。argmax は不変） |
| CV3FC06>CV3FC06 | 対照 |
| CV3FC06>SNAc3 | fc の c を 3 に |

### 7.4 判定

基準は、束 A の SNA（劣化する）と CV06FC3（平ら）の同じ課題範囲の値とする。束をまたぐが、エンジン・seed は同じ。

- 窓 W21 = t21–30 の online の seed 平均。gap_B = W21(CV06FC3) − W21(SNA)。
- 損の担い度 s_X = (W21(CV06FC3) − W21(X)) / gap_B。1 で SNA 並みに悪く、0 で CV06FC3 並みに良い。
- 雑音の目安: 束 A の t10 時点の対の差の SE は約 0.002。gap_B は 0.01 以上と見込むので、s の SE は 0.2 以下、帯は 0.25 と 0.75 とする。
- **Q1b（fc のどちらか）**:
  - F2_CARRIES: s(F2ONLY) ≥ 0.75 かつ s(F1ONLY) ≤ 0.25。
  - F1_CARRIES: その逆。
  - EITHER_SUFFICES: 両方 ≥ 0.75。
  - NEEDS_BOTH: 両方 ≤ 0.25。
  - SPLIT: それ以外。
- **Q2′（α の凍結、§4 の Q2 を差し替え）**: s(SNAfrz1) ≤ 0.25 で FROZEN_RESCUES、≥ 0.75 で FROZEN_NO、それ以外は FROZEN_PARTIAL。
- **Q4（CNN 固有か）**:
  - 低下を drop5 = t1–5 の平均 − t26–30 の平均とする。
  - Δfix = drop5(SNA+fixconv) − drop5(CV06FC3+fixconv)、Δref = drop5(SNA) − drop5(CV06FC3)（束 A）、r = Δfix / Δref。
  - r ≥ 0.75 で FC_ALONE（固定した conv 特徴の上でも fc の c=0.6 が劣化を作る）、r ≤ 0.25 で NEEDS_MOVING_CONV、それ以外は FIX_PARTIAL。
- **F（状態か損傷か）**:
  - W31–40 を fork 後 10 課題の online の seed 平均とする。基準の差 g_F = W31–40(CV06FC3, 束 A) − W31–40(SNA, 束 A)。
  - 回復率 rec = (W31–40(SNA>CV06FC3) − W31–40(SNA>SNA)) / g_F。
  - rec ≥ 0.75 で REGIME（今の c で決まり、fc を 3 にすればすぐ戻る）、≤ 0.25 で DAMAGE（溜まった状態が残る）、それ以外は MIXED。
- **F-logit**:
  - 対象は t31–33 の平均で、(SNA>SNA@f3x0.5 − SNA>SNA) を g_F と同じ課題範囲の束 A の差で割った比。
  - 比 ≥ 0.5 で LOGIT_SCALE_MATTERS、それ以外は LOGIT_SCALE_NOT。
- **F の較正**: SNA>SNA の W31–40 が、束 A の SNA の W31–40 から ±0.004 以内（2 SE）であること。

### 7.5 予測（Claude、束 B・fork の前、束 A の t10 までを見た状態で）

- Q1b: F2_CARRIES 0.45 / SPLIT 0.25 / EITHER_SUFFICES 0.10 / F1_CARRIES 0.10 / NEEDS_BOTH 0.10。
  t1–t10 では f2 の座席が +0.9→−1.07 に沈み、mob_f2 は 1.37→0.72 に下がった。f1 の変化は小さい。
- Q2′: FROZEN_RESCUES 0.50 / FROZEN_PARTIAL 0.30 / FROZEN_NO 0.20。
  凍結すると、fc の W の成長（t1→t10 で 3〜5 倍）で 2αW が育ち、ゲートが洗われる。
- Q4: FC_ALONE 0.40 / FIX_PARTIAL 0.35 / NEEDS_MOVING_CONV 0.25。
- F: REGIME 0.55 / MIXED 0.30 / DAMAGE 0.15。F-logit: LOGIT_SCALE_MATTERS 0.25。
- Q1（§4、窓 t31–50）の予測は登録時のまま変えない（CONV_LOCUS 0.50。t10 までの形勢では外れそう）。
