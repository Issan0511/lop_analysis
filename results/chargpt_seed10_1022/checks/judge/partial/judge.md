# chargpt_seed10_1022 の判定（judge.py）

- 作成 2026-10-22 22:57:47 +0900・script sha256 193f8eb67209ad73…・spec sha256 72b9e065df166c42…
- 規則・閾値・窓は spec のまま。予測は spec の表の Opus 5.5 の欄を採点しただけ。「未完」= 出力が揃っていない単位がある（その部分のラベルは出さない）。

## ラベル

| 部分 | ラベル | 数 | 符号検定の p（両側 exact） | spec |
| --- | --- | --- | --- | --- |
| A1 検算 | PASS（700/700 で κ > 0・破れ 0） | κ > 0 700/700・破れ 0 | — | :48 |
| A2（主） | LOAD_ALIGNS | s_early > s_init 10/10（+10 −0 0:0） | 0.0020 | :49-50 |
| A3 | INIT_PUSH_COMPARABLE | r_mag ≥ 0.1 10/10 | — | :52 |
| B（C3-2 L1 計量 D 早期） | 未完 | ③a 0・①+② 0・③b 0 / 0 | — | :81-84 |
| C1（lnex1） | 未完 | 生き返らない 0/0（対象外 0・未完 9） | 窓 S+1〜E: —（+0 −0） | :127-131・:241-242 |
| C2（gamma03f） | 未完 | 26 末 ≤ 0.5 0/0（資格外 0・未完 10） | — | :138・:243 |
| D（SiLU） | D1 未実施・D2 未実施・D3 未実施 | 出力なし | — | :170-174 |
| D（GELU） | D1 未実施・D2 未実施・D3 未実施 | 出力なし | — | :170-174 |
| D（SiLUcap） | D1 未実施・D2 未実施・D3 未実施 | 出力なし | — | :170-174 |

## 記録

### A（C3-2 の L1、seed ごと）

| seed | n_live init | s_init | 帯 ±（内） | s_early | s 39〜50 | s 89〜100 | r_mag | r_mag（生きた unit） | full s early | fixed-next s early |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3/seed0 | 512 | 0.469 | 0.066（True） | 0.825 | 0.522 | 0.561 | 1.811 | 1.811 | 1.000 | 0.665 |
| C3/seed1 | 512 | 0.514 | 0.066（True） | 0.835 | 0.522 | 0.518 | 1.902 | 1.902 | 1.000 | 0.655 |
| C3/seed2 | 512 | 0.490 | 0.066（True） | 0.867 | 0.547 | 0.599 | 1.708 | 1.708 | 1.000 | 0.689 |
| C3/seed3 | 512 | 0.504 | 0.066（True） | 0.850 | 0.503 | 0.665 | 1.871 | 1.871 | 1.000 | 0.688 |
| C3/seed4 | 512 | 0.482 | 0.066（True） | 0.840 | 0.502 | 0.576 | 1.996 | 1.996 | 1.000 | 0.664 |
| C3/seed5 | 512 | 0.465 | 0.066（True） | 0.888 | 0.477 | 0.494 | 2.063 | 2.063 | 0.992 | 0.675 |
| C3/seed6 | 512 | 0.494 | 0.066（True） | 0.863 | 0.502 | 0.510 | 1.243 | 1.243 | 1.000 | 0.699 |
| C3/seed7 | 512 | 0.477 | 0.066（True） | 0.850 | 0.486 | 0.460 | 1.890 | 1.890 | 1.000 | 0.682 |
| C3/seed8 | 512 | 0.498 | 0.066（True） | 0.837 | 0.555 | 0.501 | 1.823 | 1.823 | 1.000 | 0.654 |
| C3/seed9 | 512 | 0.482 | 0.066（True） | 0.875 | 0.531 | 0.459 | 1.533 | 1.533 | 1.000 | 0.677 |

- 帯に入る seed 10/10・課題 89〜100 の s の中央値 < 0.6 の seed 9/10。未完の単位: なし。
- ReLUcap-2（L1）と L0: ReLUcap/seed0 s_init 0.469 s_early 0.795 / L0 s_init 0.514 s_early 0.375; ReLUcap/seed1 s_init 0.514 s_early 0.783 / L0 s_init 0.508 s_early 0.348; ReLUcap/seed2 s_init 0.490 s_early 0.855 / L0 s_init 0.510 s_early 0.371; ReLUcap/seed3 s_init 0.504 s_early 0.837 / L0 s_init 0.496 s_early 0.377; ReLUcap/seed4 s_init 0.482 s_early 0.811 / L0 s_init 0.494 s_early 0.334; ReLUcap/seed5 s_init 0.465 s_early 0.775 / L0 s_init 0.525 s_early 0.268; ReLUcap/seed6 s_init 0.494 s_early 0.801 / L0 s_init 0.484 s_early 0.348; ReLUcap/seed7 s_init 0.477 s_early 0.766 / L0 s_init 0.504 s_early 0.375; ReLUcap/seed8 s_init 0.498 s_early 0.811 / L0 s_init 0.475 s_early 0.377; ReLUcap/seed9 s_init 0.482 s_early 0.783 / L0 s_init 0.477 s_early 0.359
- cos²・Q（C3-2、課題 1・10・25・50・100）: C3/seed0 cos² 0.800/0.435/0.460/0.323/0.217; C3/seed1 cos² 0.800/0.591/0.449/0.392/0.002; C3/seed2 cos² 0.784/0.607/0.471/0.346/0.044; C3/seed3 cos² 0.773/0.604/0.463/0.340/0.162; C3/seed4 cos² 0.812/0.593/0.481/0.380/0.287; C3/seed5 cos² 0.786/0.589/0.494/0.424/0.245; C3/seed6 cos² 0.784/0.608/0.483/0.334/0.180; C3/seed7 cos² 0.794/0.599/0.467/0.297/0.241; C3/seed8 cos² 0.818/0.567/0.467/0.360/0.334; C3/seed9 cos² 0.812/0.591/0.457/0.310/0.241

### B（seed ごとの沈む unit の割合の中央値 ③a / ①+② / ③b と最大、ρ の中央値、①②の符号の一致）

- C3: 揃った seed 0・欠け 98・検査落ち 0
- ReLUcap: 揃った seed 0・欠け 99・検査落ち 0（記録のラベル 未完）

### C

- C1: 未完の単位 27
  - lnex1: 生き返らない 0/0・ラベル（同じ規則）未完・符号検定 窓 S+1〜E p —・判定窓 p —
  - nocommon: 生き返らない 0/0・ラベル（同じ規則）未完・符号検定 窓 S+1〜E p —・判定窓 p —
  - seed0 S 50 E 64；lnex1: 未完；nocommon: 未完
  - seed1 S 43 E 59；lnex1: 未完；nocommon: 未完
  - seed2 S 50 E 93；lnex1: 未完；nocommon: 未完
  - seed4 S 50 E 65；lnex1: 未完；nocommon: 未完
  - seed5 S 25 E 46；lnex1: 未完；nocommon: 未完
  - seed6 S 50 E 77；lnex1: 未完；nocommon: 未完
  - seed7 S 41 E 53；lnex1: 未完；nocommon: 未完
  - seed8 S 50 E 74；lnex1: 未完；nocommon: 未完
  - seed9 S 50 E 66；lnex1: 未完；nocommon: 未完
- C3（記録だけ・ReLUcap-2）: 未完の単位 20
  - lnex1: 生き返らない 0/0・ラベル（同じ規則）未完・符号検定 窓 S+1〜E p —・判定窓 p —
  - seed0 S 25 E 50；lnex1: 未完
  - seed1 S 25 E 68；lnex1: 未完
  - seed2 S 45 E 59；lnex1: 未完
  - seed3 S 40 E 58；lnex1: 未完
  - seed4 S 25 E 60；lnex1: 未完
  - seed5 S 41 E 59；lnex1: 未完
  - seed6 S 40 E 60；lnex1: 未完
  - seed7 S 50 E 69；lnex1: 未完
  - seed8 S 42 E 59；lnex1: 未完
  - seed9 S 25 E 58；lnex1: 未完
- C2: 未完の単位 30・gamma03（副）未完（0/0）・gamma03f で再び死ぬ seed 0（10 seed のうち）/ 資格の中 0

### D

- SiLU: 未実施（出力なし）
- GELU: 未実施（出力なし）
- SiLUcap: 未実施（出力なし）

## 予測の採点（Opus 5.5 の欄。数は書き換えていない）

| 問い | spec | 結果 | 結果に付けた確率 | Brier |
| --- | --- | --- | --- | --- |
| A1 | :67 | True | 0.95 | 0.003 |
| A2 | :68 | LOAD_ALIGNS | 0.60 | 0.285 |
| A2rec | :69 | True | 0.40 | 0.360 |
| A3 | :70 | True | 0.70 | 0.090 |
| Arec | :71 | True | 0.50 | 0.250 |
| B | :93 | 未完 | 採点しない | — |
| C1 | :157 | 未完 | 採点しない | — |
| C1rec | :158 | 未完・未実施・対象なし | 採点しない | — |
| C2 | :159 | 未完 | 採点しない | — |
| C2rec | :160 | 未完・未実施・対象なし | 採点しない | — |
| D1 | :185 | 未実施 | 採点しない | — |
| D2 | :186 | 未完・未実施・対象なし | 採点しない | — |
| D3 | :187 | 未完・未実施・対象なし | 採点しない | — |

## 読みの選択（spec の文だけで決まらない所）

- **(1) B の「沈む unit」**: 生きた unit に限らない（4 項の和 < 0 の全 unit）。spec §4（spec:79）の「沈む unit は 4 項の和が負の unit」の数を出した写し元の集計 `$CG/analysis/chargpt_cl_1019_quicklook/rho_kappa_answers.py`（a6517244、blob 015e3c5f）の carriers()（37〜46 行）が、alive で絞らずに tot < 0 の unit で share_dn = Σ項/Σ和 を取る（spec §2 の 0.02/0.05/0.84/0.09 はこの値）。rho_split の出力（rho10.py）は unit ごとの 4 項と alive を持ち、集計の定義は持たないので、この集計に合わせた。③b は others_path（others_gamma は ③b の内訳で 4 項の和に入れない。rho_split.py:184 の閉合と同じ）。記録として、生きた unit に限った割合も並べる。
- **(2) D1 の「課題 50 までに 5 課題以上続く」**: 5 課題の続きが課題 50 までに終わる（始まり t で t+4 ≤ 50）と読む。判定の量は固定の一様 probe（layer_stats.csv の probe = uniform、u = 10000、layer = 3）の r_med > +1 かつ phi1_mean ≥ 0.75（spec_chargpt_cl_1019.md:581 の追補 6 §5 の定義。前例 judge_m2.py:138 の「t から t+4 まで全部」の形を一様 probe に移した）。記録として、続きの始まりだけが課題 50 以内の読み（t ≤ 50）の数も出す。
- **(3) D2 の基準**: 基準 = 逃げ始めの課題 t_esc（D1 の続きの始まり）の前の 5 課題（t_esc−5〜t_esc−1。課題 1 より前は無いので、t_esc ≤ 5 なら有る課題だけ）の L2 の r_med（一様 probe）の平均。課題 t_esc〜100 の課題末に r_med ≤ 基準 − 1 があれば「移る」。t_esc = 1 なら基準が無く判定できない（数に入れず、その数を書く）。前例 judge_m2.py:140 は「直前 5 課題の移動平均」で別の形なので使わない（spec:173 は固定の基準）。
- **(4) A の生きた unit の probe**: load10.py の n_alive・own_sink は写し元 measure.py（:121）の定義のまま、課題 t の鎖の probe（init は課題 1 の鎖）で生きた unit を決める。spec:42 は「一様 probe で z > 0 のトークンがある unit」と書くが、spec:39 の「定義は chargpt_load_fold_1021 §2 のまま」と照合（seed 1000 で一致）に合わせた load10.py の出力を、そのまま使う。A2 の記録の帯 0.5 ± 3√(0.25/n_live) の n_live も、この n_alive（init の L1）。一様 probe で数え直すかは主の判断（compute_A の報告の 1）。
- **(5) A の対象の腕と層**: A2・A3 は C3-2 の L1（spec:53 で ReLUcap-2 と L0 は記録だけ）。A1 は 2 腕 × 10 seed × 課題末 35 = 700 状態（spec:48）。r_mag は写し元 report.py:36 と同じく own_abs_median（全 512 unit の |G_own| の中央値）。生きた unit だけの own_abs_median_alive の比は記録。
- **(6) A2 の符号検定**: s_early > s_init を +、< を −、等しいものは除いて両側 exact 符号検定（二項 1/2）。spec:50 の「9/10 で 0.021」と一致する。
- **(7) B の「割合の中央値」と最大**: seed ごとに、課題 k+1 ∈ {2,3,5,10} の各状態で share(③a)、share(①)+share(②)、share(③b) を出し、4 課題の中央値を取って最大の項を決める。rho10 の検査（check_eval.all_pass）が落ちた状態は使わず、その seed は「検査落ち」として未完と同じく数えない。
- **(8) C1 の符号検定の窓**: spec:131 の「窓の L1 の死の平均」の窓は spec:101 の定義（S_k+1〜E_k）とした。記録として判定窓 [R'_k, E_k] の平均の差の検定も出す。対の差は lnex1 − control（どちらも byakko の再生、追補 1）。0 の差は除く。
- **(9) C の「死」の量**: 各課題末の layer_stats.csv の probe = uniform・layer = 1・frac_p_lt005（u = 課題の更新数）。境は spec のまま「≤ 0.5 で生き返り」「> 0.5 で生き返らない」「≥ 0.9 で死」。
- **(10) C2 の記録（再び死ぬ）**: gamma03f が課題 26 末に生き返り（≤ 0.5）、課題 27〜31 末のどこかで ≥ 0.9 になった seed を数え、予測（spec:160）は 10 seed のうちの数で採点する。資格（control ≥ 0.9）の seed の中の数も並べる。
- **(11) C1 の記録の予測（nocommon 8/9）**: 対象の数 n が 9 のときは文どおり 8 以上。n ≠ 9 のときは追補 1 の規則（n − 1 以上）で読み、その旨を書く。
- **(12) D3 の δ**: δ = 3 × 0.0084 = 0.0252（前例 judge_m2.py:16 と同じ。spec:174 の 0.025 はその丸め）。0.025 と 0.0252 の間に入る値があれば記録に書く。量は judge_m2.py:23-33 のまま（online_all の課題 2〜10 と 91〜100 の平均、fresh gap = online_all − fresh の online_all、頭 100 更新 = online.npz の loss[:, :100] − H の平均）。条件 1 の「課題 1 末 ≤ 0.1」は 3 つに入れず記録にする。
- **(13) 予測の採点**: 結果のラベルに付けた確率と Brier（多クラスは Σ(p − 1[結果])²）を書く。結果が表に無いラベル（INSUFFICIENT・未完など）は採点しない。

## 入力

- load: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/load
- rho: /tmp/claude-1000/-home-kubo-project-Nakatsuka/fff216c2-f017-401b-869a-e3d0a73aac33/scratchpad/rho
- revival: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/revival
- silu4: /home/kubo/project/Nakatsuka/lop_seed10/results/chargpt_seed10_1022/silu4
- revival_plan_sha256: 77f4e1bef1b902e6320dace2349b4fa7ffc058ee57dcdc1defd0749d2d22180c
