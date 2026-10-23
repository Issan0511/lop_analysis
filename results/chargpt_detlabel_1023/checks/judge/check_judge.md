# check_judge（judge.py の検査）

- 66/66 通過。judge.py sha256 a2e51af0d8b7704f461f3bd56139281a89ddccf479e9a648511ef86f178d77dd

| 検査 | 合否 | 詳細 |
| --- | --- | --- |
| a/C3: 10 seed が読める | 通過 | {} |
| a/C3/seed0: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (65, 65), 'Rk': (55, 55)} |
| a/C3/seed1: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (64, 64), 'Rk': (50, 50)} |
| a/C3/seed2: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (87, 87), 'Rk': (84, 84)} |
| a/C3/seed3: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (78, 78), 'Rk': (None, None)} |
| a/C3/seed4: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (None, None), 'Rk': (56, 56)} |
| a/C3/seed5: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (51, 51), 'Rk': (37, 37)} |
| a/C3/seed6: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (98, 98), 'Rk': (68, 68)} |
| a/C3/seed7: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (44, 44), 'Rk': (44, 44)} |
| a/C3/seed8: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (None, None), 'Rk': (65, 65)} |
| a/C3/seed9: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (True, True), 'c1': (True, True), 'onset': (57, 57), 'Rk': (57, 57)} |
| a/C3: 連続量の最大の差（a2_table.json との、全 seed・全量）≤ 1e-12 | 通過 | {'end1': '0', 'dE': '0', 'dE110': '0', 'g0': '0', 'g1': '0', 'L0d100': '0', 'L0d_91_100': '0', 'L0d50': '0', 'L1d10': '0', 'L1d50': '0', 'L1d100': '0', 'early0': '0', 'late0': '0', 'cont100_0': '0', 'fresh100_0': '0', 'early1': '0', 'late1': '0', 'cont100_1': '0', 'fresh100_1': '0', 'early2': '0', 'late2': '0', 'cont100_2': '0', 'fresh100_2': '0'} |
| a/C3: J1 の LOSS の数・g の中央値（§2.1） | 通過 | {'n_loss': 10, 'g': {'median': 15.13687250277442, 'min': 14.00998413726232, 'max': 16.661946458450082, 'n': 10}} |
| a/C3: L0 の死 0.87 [0.00, 0.96]・d ≥ 0.5 が 8/10・J5a 10/10・R_k 9/10 で 37〜84（中央値 56）・J1b 10/10 | 通過 | {'d': {'median': 0.865234375, 'min': 0.0, 'max': 0.958984375, 'n': 10}, 'n_ge05': 8, 'J5a': 10, 'Rk': [55, 50, 84, None, 56, 37, 68, 44, 65, 57]} |
| a/C3: 死の量 layer_stats.csv frac_p_lt005 と unit_stats.npz (1 − uniform_q) < 0.005 の差（全課題・両層） | 通過 | 最大の差 0 |
| a/C3: §2.2 の課題末どうしの 1 − cos（表の 2〜3 桁） | 通過 | {'2-10': (0.11632690210709548, 0.11632690210709548), '41-50': (0.030660497802671965, 0.03143043941232451), '91-100': (0.0009703306920183924, 0.011445653195451516)} |
| a/C3: §7.4 の net100 の参照 L0（a2_j4_fallback.txt の 3 桁） | 通過 | {'alive': {'median': 1.6632438168881123, 'min': 1.3703180915584763, 'max': 1.7789496129067635, 'n': 10}, 'all': {'median': 1.1654248558151814, 'min': 1.0847402453721695, 'max': 1.7787243894390263, 'n': 10}} |
| a/C3: §7.4 の net100 の参照 L1（a2_j4_fallback.txt の 3 桁） | 通過 | {'alive': {'median': 1.3589088967210738, 'min': 1.2212622192546703, 'max': 1.4838768815784755, 'n': 10}, 'all': {'median': 1.0281999314076802, 'min': 0.4622994712192405, 'max': 1.356173260635287, 'n': 10}} |
| a/ReLUcap: 10 seed が読める | 通過 | {} |
| a/ReLUcap/seed0: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (41, 41)} |
| a/ReLUcap/seed1: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (59, 59)} |
| a/ReLUcap/seed2: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (50, 50)} |
| a/ReLUcap/seed3: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (49, 49)} |
| a/ReLUcap/seed4: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (51, 51)} |
| a/ReLUcap/seed5: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (50, 50)} |
| a/ReLUcap/seed6: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (51, 51)} |
| a/ReLUcap/seed7: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (60, 60)} |
| a/ReLUcap/seed8: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (50, 50)} |
| a/ReLUcap/seed9: J1 の LOSS・相ごとの喪失・条件 1・L0 の死の始まり・R_k が一致 | 通過 | {'j1': (False, False), 'c1': (False, False), 'onset': (None, None), 'Rk': (49, 49)} |
| a/ReLUcap: 連続量の最大の差（a2_table.json との、全 seed・全量）≤ 1e-12 | 通過 | {'end1': '0', 'dE': '0', 'dE110': '0', 'g0': '0', 'g1': '0', 'L0d100': '0', 'L0d_91_100': '0', 'L0d50': '0', 'L1d10': '0', 'L1d50': '0', 'L1d100': '0', 'early0': '0', 'late0': '0', 'cont100_0': '0', 'fresh100_0': '0', 'early1': '0', 'late1': '0', 'cont100_1': '0', 'fresh100_1': '0', 'early2': '0', 'late2': '0', 'cont100_2': '0', 'fresh100_2': '0'} |
| a/ReLUcap: J1 の LOSS の数・g の中央値（§2.1） | 通過 | {'n_loss': 0, 'g': {'median': 0.1973771152535104, 'min': 0.17896588828508797, 'max': 0.4528334728356956, 'n': 10}} |
| a/J6 C3: §2.3 のラベル（OTHER_TERM_FLIPS 0/10・LN_HALVES 10/10） | 通過 | ('OTHER_TERM_FLIPS', 0, 'LN_HALVES', 10) |
| a/J6 C3: seed 内の窓の中央値が loadsign の windows.csv・components_windows.csv と一致（210 値） | 通過 | 最大の差 0 |
| a/J6 ReLUcap: §2.3 のラベル（OTHER_TERM_KEPT 10/10・LN_HALVES 10/10） | 通過 | ('OTHER_TERM_KEPT', 10, 'LN_HALVES', 10) |
| a/J6 ReLUcap: seed 内の窓の中央値が loadsign の windows.csv・components_windows.csv と一致（210 値） | 通過 | 最大の差 0 |
| a/A2: 再生の開始の checkpoint（重みと Adam）が 290 課題すべてに在る | 通過 | [] |
| a/A2: 更新 100 の checkpoint（u100_task*.pt）の数（記録） | 通過 | 0 個（byakko の展開には無い。net100 は prereg の値から照合） |
| b/sign: 9/10 → p = 0.021、8/10 → 0.109、引き分けを除く | 通過 |  |
| b/thr: ceil(k·n/10) | 通過 |  |
| b/label_count: n=10 の 8/7/3/2、n=9 の 8/7、n=8 の 7/6、n=7 は INSUFFICIENT | 通過 |  |
| b/label_ratio: ρ = 0.5 → SIMILAR、0.4999 → SMALLER、2 → SIMILAR、2.0001 → LARGER、NaN と n < 8 → INSUFFICIENT | 通過 |  |
| b/label_J5b: m = 5 → INSUFFICIENT、6 中 5 → REVIVES、10 中 8/3/2、使える 7 → INSUFFICIENT | 通過 |  |
| b/case same: ラベル | 通過 | ['LOSS_PERSISTS', 'L0_GROWTH_SIMILAR', 'L0_DEATH_PERSISTS', 'MOTION_SIMILAR', 'L1_EARLY_DEATH', 'L1_REVIVES', 'OTHER_TERM_FLIPS', 'LN_HALVES'] |
| b/case same: ΔE の引き分け 10 → 符号検定 n = 0・p = NaN、P1 は偽 | 通過 | {'n_pairs': 10, 'n_pos': 0, 'n_neg': 0, 'n_tie': 10, 'p': None} |
| b/case edge_hi: J1 8/10 → PERSISTS、ρ2 = 0.5 → SIMILAR、d = 半分ちょうど → PERSISTS、ρ4 = 1/3 → SMALLER、L1 0.9 ちょうど → EARLY、8/10 戻る → REVIVES、7/10 → KEPT・LN_HALVES | 通過 | (['LOSS_PERSISTS', 'L0_GROWTH_SIMILAR', 'L0_DEATH_PERSISTS', 'MOTION_SMALLER', 'L1_EARLY_DEATH', 'L1_REVIVES', 'OTHER_TERM_KEPT', 'LN_HALVES'], 0.5, 0.3333333333333333, {'median': 0.45, 'min': 0.45, 'max': 0.45, 'n': 10}, 0.45) |
| b/case edge_hi: P3（ρ4 = 1/3 ちょうど）は真、P6 真、P6-LN 真、P2 偽 | 通過 | {('P1', 'Fable'): False, ('P1', 'Opus'): False, ('P2', 'Fable'): False, ('P2', 'Opus'): False, ('P3', 'Fable'): True, ('P3', 'Opus'): True, ('P6', 'Fable'): True, ('P6-LN', 'Fable'): True} |
| b/case edge_lo: 7/10 → MIXED、ρ2 < 1/2 → SMALLER、d < 半分 → REDUCED、ρ4 > 2 → LARGER、7/10 → MIXED、6/10 → FLIPS・NOT_HALVES | 通過 | ['MIXED', 'L0_GROWTH_SMALLER', 'L0_DEATH_REDUCED', 'MOTION_LARGER', 'MIXED', 'OTHER_TERM_FLIPS', 'LN_NOT_HALVES'] |
| b/case gone: 2/10 → LOSS_GONE、3/10 → MIXED、早期の死 0 → NO_EARLY_DEATH・J5b INSUFFICIENT、J6 の入力なし → INSUFFICIENT | 通過 | ['LOSS_GONE', 'MIXED', 'NO_EARLY_DEATH', 'INSUFFICIENT', 'INSUFFICIENT'] |
| b/case J5b: 早期に死んだ seed 5 → INSUFFICIENT（J5a は MIXED）、6 中 5 戻る → REVIVES | 通過 | ['MIXED', 'INSUFFICIENT', 'L1_REVIVES', 5, 6] |
| b/case 欠けた seed: n = 9・7 LOSS → MIXED、n = 8・7 LOSS → PERSISTS、n = 8 の J6 6 → KEPT（境 ceil(5.6) = 6） | 通過 | ['MIXED', 8, 'LOSS_PERSISTS', 7, 'OTHER_TERM_KEPT', 6] |
| b/case 欠けた seed: n = 7 → 全ラベル INSUFFICIENT、予測は採点しない | 通過 | ['INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT', 'INSUFFICIENT'] |
| b/case 欠けた seed: 欠けの理由を記録 | 通過 | {'9': '無い: per_task.csv,online.npz,fresh_online_t100.npz,fresh.csv,unit_stats.npz,layer_stats.csv'} |
| b/case 再生なし: J4 INSUFFICIENT、P3 採点なし | 通過 |  |
| b/case 生きた unit なし: その seed は 29 課題が欠測で J4 に使えず（n = 9）、ラベルは SIMILAR | 通過 | (9, {'0': 29, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}, 'MOTION_SIMILAR') |
| b/縮退（≤ 100 で縮退、列 distinct_ctx_500upd）と S-entropy の外れの一覧（ok_entropy = False の連鎖、z = (CE − H)/SE） | 通過 | ({96, 95}, 'distinct_ctx_500upd', 1, 5.0) |
| c/W trial_A4/seed0: within_motion.npz を読める（課題 [np.int64(1), np.int64(2), np.int64(3)]、量 5 つ有限）、alive[..., u_end] が unit_stats.npz の chain_dead の否定とビットで同じ（向きの確かめ） | 通過 | {'tasks': [np.int64(1), np.int64(2), np.int64(3)], 'alive_both': {np.int64(1): 1024, np.int64(2): 1024, np.int64(3): 1023}, 'net5k_L0_med': {np.int64(1): 0.025488868355751038, np.int64(2): 0.021762000396847725, np.int64(3): 0.010422906838357449}} |
| c/W trial_A4/seed0: chain_stats.csv の縮退の列と S-entropy の列を読める | 通過 | {'degenerate': [], 'ctx': {1: '411', 2: '175', 3: '126'}, 'entropy_fail': []} |
| c/W trial_S2/seed0: within_motion.npz を読める（課題 [np.int64(1), np.int64(2), np.int64(3)]、量 5 つ有限）、alive[..., u_end] が unit_stats.npz の chain_dead の否定とビットで同じ（向きの確かめ） | 通過 | {'tasks': [np.int64(1), np.int64(2), np.int64(3)], 'alive_both': {np.int64(1): 1024, np.int64(2): 1016, np.int64(3): 1024}, 'net5k_L0_med': {np.int64(1): 0.032106367871165276, np.int64(2): 0.026270633563399315, np.int64(3): 0.0493636392056942}} |
| c/W trial_S2/seed0: chain_stats.csv の縮退の列と S-entropy の列を読める | 通過 | {'degenerate': [], 'ctx': {1: '4154', 2: '4151', 3: '4170'}, 'entropy_fail': []} |
| c/W srec_with: within_motion.npz を読める（課題 [np.int64(1), np.int64(2)]、量 5 つ有限）、alive[..., u_end] が unit_stats.npz の chain_dead の否定とビットで同じ（向きの確かめ） | 通過 | {'tasks': [np.int64(1), np.int64(2)], 'alive_both': {np.int64(1): 1024, np.int64(2): 1024}, 'net5k_L0_med': {np.int64(1): 0.30886058509349823, np.int64(2): 0.3491814285516739}} |
| c/W srec_with: chain_stats.csv の縮退の列と S-entropy の列を読める | 通過 | {'degenerate': [], 'ctx': {1: '4154', 2: '4151'}, 'entropy_fail': []} |
| c/W trial_S2: loss ≠ aux、per_task.csv の online_all = mean(loss) − H（J1 は loss を使う） | 通過 | [(0.748019019, np.float64(0.7480190187509498), np.float64(0.747126479368969)), (0.257373713, np.float64(0.2573737133880615), np.float64(0.25847551824054715)), (0.269508878, np.float64(0.26950887784856414), np.float64(0.2687921518434868))] |
| c/W replay_try/seed0: replay_task<t>.npz・replay_compare.csv・replay_provenance.json を読める | 通過 | {'info': {'complete': True, 'n_tasks': 2, 'tasks_missing': [3, 4, 5, 6, 7, 8, 9, 10, 41, 43, 44, 45, 46, 47, 48, 49, 50, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100], 'online_bitwise': 0, 'state_bitwise': 0, 'online_absdiff_max': 0.02357947826385498, 'online_absdiff_mean_med': 0.0032200990349054337, 'state_reldiff_max': 0.5523800823539263, 'n_compare': 2}, 'net5k_L0': {2: 0.3438579589128494, 42: 0.613 |
| c/net100 の登録前の参照を a2_j4_fallback.txt から読む（L0 1.663 [1.370, 1.779]・全行 1.165 [1.085, 1.779]） | 通過 | {'source': 'results/chargpt_detlabel_1023/prereg/a2_j4_fallback.txt', 'sha256': '5eb9d37e420737ab7ef8570228d4684ab00565a06ba8780935d83fb0cc001eb2', 'script': 'analysis/chargpt_detlabel_1023/prereg/a2_j4_fallback.py', 'origin': 'white-san の A2 の保存状態（u100_task<t>.pt と task<t>.pt）から登録前に出した値。生きた unit は課題末の鎖 probe', 'L0': {'alive': {'median': 1.663, 'min': 1.37, 'max': 1.779}, 'allrows': {'median': 1.1 |
| c/J compute_J の A2_full（arm A2）を読める。後期の J6・J6-LN の数（loadsign は 0/10・10/10）と、loadsign との窓の中央値の差（記録） | 通過 | {'counts': (0, 10), 'late_window_absdiff_max': {'s_original_self': 0.0, 's_original_other': 0.0, 's_original_residual': 0.0, 's_original_direct_K': 0.0, 's_radial': 0.0, 's_original_exact_K': 0.0, 'original_other_abs_share': 8.326672684688674e-16, 'n_alive': 0.0}} |
| c/通し: W の形の within_motion（課題 1〜3）と replay（課題 2・42）で J4 が出る、J6 は A2_full、S-entropy と net100 の 2 つの値の行が judge.md に出る | 通過 | {'J4': ('MOTION_SMALLER', 0.03362459695763524, 27), 'J6': ('OTHER_TERM_FLIPS', 0), 'entropy': (30, 0), 'S2_J1': 'LOSS_PERSISTS'} |
