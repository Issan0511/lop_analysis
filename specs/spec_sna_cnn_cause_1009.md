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
