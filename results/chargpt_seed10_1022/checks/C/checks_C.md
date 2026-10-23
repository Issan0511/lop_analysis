# C の検査（手順書 §4 の C・spec §7・追補 1）— byakko（RTX 5070 Ti）、2026-10-22、compute_C

道具: `analysis/chargpt_seed10_1022/revival10.py`（写し元 intervene_revival.py @ a970f12c、blob f9b7ccde）、`revival10_plan.py`、`checks_C.py`。
数の正本 = `checks_C.json`（`checks_C.py summarize` の出力）と `plan_check.json`（開始点と窓の出し直し）。各走の出力 = `start/`・`replay1/`・`replay1b/`・`par2/` の下の per_task.csv・layer_stats.csv・extras.csv・compare_orig.csv・provenance.json。

| 検査 | 結果 | 数 |
| --- | --- | --- |
| 開始点と窓を元の layer_stats.csv から spec §5 の規則で出し直し、spec の表と照らす | PASS | C1: 10 seed すべて一致（seed 3 は R 無し・最小 0.674）、窓の合計 188。C3: 10 seed の S・R すべて一致、E = min(R+9,100) で窓の合計 242（spec §8 と同じ） |
| 開始状態（C1・C3 の S_k 末、C2 の課題 25 末）の一様 probe の L1 の死が元の layer_stats.csv（u=10000）と一致 | PASS | 29/29 で完全一致（差 0）。L0（uniform）29/29、L1（chain）29/29 も完全一致 |
| 開始状態の L1 の死が spec §5 の表と整合（≥ 0.9） | PASS | 29/29。C2 の課題 25 末は 0.959〜0.992（spec「0.96〜0.99」） |
| lnex1 の開始時の平均の分母が元と一致（定数の検査） | PASS | 19/19（C1 9・C3 10）。相対差の最大 6.7e−8（float32 の丸め） |
| control の 1 課題の再生が動き、元の記録との差を出せる | PASS | 4 本（C2 seed0・C1 seed0・C1 seed7・C3 seed0）。NaN 0 |
| control が元の走とビット一致するか（記録。追補 1 で一致しない見込み） | 不一致（落ちではない） | 4/4 で不一致。更新 0〜2 は一致し、更新 3 から離れる。online の差の最大 0.011〜0.034、課題平均の差 −0.0067〜+0.0048。L1 の死（uniform）の差: −0.0020・−0.0098・−0.0254・+0.0059 |
| 同じ台で 2 回回すとビット一致するか（byakko の control を対照にする前提） | PASS | 2/2（C2 seed0 control・C1 seed0 lnex1）で online と layer_stats.csv が完全一致 |
| ほかの腕が 1 課題動く | PASS | lnex1（C3-2・ReLUcap-2）、nocommon、gamma03f（固定した γ の動き 0.0）、gamma03 |
| 1 課題の時間 | — | GPU に他の走が無いとき 15.7〜16.4 秒。2 本同時で各 33.2 秒（合計の速さは 1 本と同じ）。他の担当と GPU を分けたとき 27〜47 秒 |
| GPU のメモリ | — | 1 本 0.58〜0.59 GB（max_memory_allocated） |

注:
- 「chain」は写し元のとおり、その課題の chain probe（元の layer_stats.csv の chain と同じもの）。spec §5 の「課題 S_k の chain probe」は `chain_start` として各課題末に別に測る。
- gamma03f の課題 26 末の L1 の死（seed 0、1 課題の検査）は 0.016、gamma03 は 0.19。判定は本走のあとで judge が行う。
