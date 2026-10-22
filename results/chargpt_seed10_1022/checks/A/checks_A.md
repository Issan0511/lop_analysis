# A（装填）の検査 — chargpt_seed10_1022（2026-10-22 22:29、byakko、compute_A）

script = `analysis/chargpt_seed10_1022/load10.py`（sha256 0205260f4df17cdc97e18f74753afcaaa8dc3abd9f8a540ddbe82a9faea34866。checks.json の provenance と一致）。CPU・float64・8 threads。入力の束は BUNDLE_OK（tar_sha256 一致・1459 file 不一致 0）の後に読んだ。

結果: **all_pass = true、35/35**（checks.json）。

| 検査 | 数 | 結果 |
| --- | --- | --- |
| (i) seed 1000 の M2/C3 の init と課題末 35（36 状態）を出し直し、`results/chargpt_load_fold_1021/load/table_head.csv`・`table_units.csv`（L1、p07 の列を含む）と照合 | 36 状態・不一致 0 | PASS。連続量の相対差の最大 = head 1.1e−12（t95 の null_q50）、unit 4.4e−14（t93 の own_abs_median_alive）。割合（own_sink・full_sink・self_agreement・full_own_agreement・uniform_own_sink・fixed_next_own_sink）の差は全部 0。n_alive などの数は全部一致。null_quantile は 1/1000 以内、identity_rel と closure_max は 1e−10 以下 |
| (ii) 補題の恒等式・位置の閉包・head 反転の変異・1 クラス目標の変異・G_own と G_full の有限差分（L0 と L1）・評価値の照合。対象 = M2/C3 seed 1000・M2/C3 seed 0・M2/ReLUcap seed 0 の課題 1 末 | 33 | 33 PASS。有限差分の相対誤差の最大 = G_own 1.7e−7、G_full 3.6e−10（閾 1e−4）。評価値は写し元の all_pass の基準（2e−6 × |記録| + .9g の丸め）で通る。.9g の丸めだけの基準（登録時の基準）では 3/3 とも通らない（相対差 1.1e−7〜7.7e−7。記録が float32 の評価のため。写し元でも同じ） |
| (iii) 同じ seed の C3-2 と ReLUcap-2 の init.pt・task1.pt（seed 0〜9、20 対） | 20 対 | PASS 20/20。共通の 15 tensor（重み・buffer）が bytes で一致、float64 の forward（一様 probe 8 系列）がビットで一致。ただし ReLUcap の state_dict には cap の帳簿の buffer（各層の V・cap・cap_on・v_rate）が足されていて、file の sha と state_dict 全体の一致は 0/20。課題 1 末までは cap_on = 0・cap = inf（cap は掛かっていない）を確かめて通した |

時間: checks の全体 55 秒、(i) の 36 状態で 43 秒（1 状態 約 1.2 秒）。`timing.json`: C3 seed 0 の 5 状態（init・1・10・50・100）で 1 状態 平均 1.46 秒（最大 1.97 秒）→ 本走 720 状態で約 0.3 時間（1 process・8 threads）。

途中の落ち（直した実装の誤り）:
1. 1 回目: (iii) が 0/20。state_dict の key の集合を比べたため、ReLUcap だけにある cap の帳簿で落ちた（`attempt1_iii_ii.log`）。比べ方を上の基準に直した。
2. 2 回目: (i) が 1 件落ちた（課題 1 末の kappa_ratio が NaN）。k1 を決める前に比を出していた。写し元と同じ順（課題 1 末で k1 を決めてから比）に直した。このときの log は 3 回目で上書きした。
