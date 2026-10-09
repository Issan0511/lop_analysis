# sna_cnn_cause_1009 — 適応 Snake（c=0.6）が RL-CIFAR CNN で固定 α=3 に負ける原因

spec: `specs/spec_sna_cnn_cause_1009.md`（本文＋追補 1〜9、予測つき）。エンジン `src/sna_cnn_cause_1009.py`（束ね・CUDA グラフ）、fork `src/sna_cnn_cause_1009_fork.py`、検査 `results/_checks_sna_cnn_cause_1009/checks.json`（9 本 all_pass）。判定 `analysis/sna_cnn_cause_1009/verdict.py`（§4 → `summary.md`・`verdict.json`）と `verdict2.py`（追補 → `summary2.md`・`verdict2.json`）。登録外の探索は `explore_summary.md`。seed 10–19、online_acc の seed 平均。

| 束 | 腕 | 課題 | 判定 |
|---|---|---|---|
| A | SNA・SNAc3・CV06FC3・CV3FC06 | 50 | E0 ENGINE_OK（gap +0.0148・10/10）、Q1 FC_LOCUS（CV06FC3 の暴走が作ったラベル。CV3FC06 は t31–50 で SNAc3 と同水準） |
| C | SNA+fcamp0.7／0.9 | 30 | QF FLAT_POINT_NOT（平らな点を持ち上げると逆効果・暴走） |
| D | S:a3-c0.6-c3-c3・S:a3-c0.6-c0.6-c0.6+fcamp0.7 | 30 | QD1 ANCHOR_HOLDS・QD2 BOTH_EXPLAIN（c1 固定 α=3＋fc の持ち上げで SNA +1.4 pt、SNAc3 +0.7 pt） |
| B1 | SNAfrz1・SNA+fixconv・CV06FC3+fixconv | 30 | Q2′ FROZEN_RESCUES・Q4 NEEDS_MOVING_CONV |
| F30 | t30 からの fork 6 種 | 10 | F MIXED（fc→3 は暴走、conv→3 は即座に全回復 rec_conv +2.97）・F-logit MATTERS |
| B2 | F1ONLY・F2ONLY | 30 | Q1b SPLIT（fc の片方を 3 にすると平ら） |

**一文**: 進行性の劣化は c2・f1・f2 の 3 層が全部 2αW=1.2 の鋭いゲートのまま谷に沈み続けることで起き、どれか 1 層を洗う・α を凍結する・平らな点を持ち上げる（c1 の錨つき）のいずれでも消えて固定 α=3 を上回る。尺度の暴走は、鋭いゲートの fc が 1 層も残らず c1 の α が上限から外れたときに起きる。

生データ（チェックポイント・state・ログ）は `obsidian-research-data/sna_cnn_cause_1009/`（`backup_manifest.json`）。
