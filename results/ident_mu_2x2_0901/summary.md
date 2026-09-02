# ident_mu_2x2_0901 — 可識別性 × µ の 2×2（純化版）

事前登録: `specs/spec_ident_mu_2x2_0901.md`＋`specs/spec_ident_mu_2x2_0901_addendum1.md`（追補1）。主判定は **I_CELLS_INVALID_S_OP**。S-op が FAIL したため I+ セル（IM・Im）は無効で、走ったのは `iM`, `im`, `im_nowd`, `iM_nowd`, `std_anchor` の 5 腕。対比に使った共通完走 seed（iM / im）= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (n=10)。

要因 4 セルはすべて b 限定 WD λ=0.001 の下にある（spec D8・§3.5）。**主判定の読みは「b 拘束下」限定であり、拘束なしの世界の I/M 主効果は測っていない**（橋は `im_nowd` と committed `Aexact` まで）。

窓は B02 = task 51–100、B10 = task 451–500。共主 endpoint は E-drift（`mean(log10 unfit)` の B10−B02）と E-level（同 B10）。**要因計画の主判定は出さない**（追補1 §3）。等価限界は Δ=0.15 dex・Δ_int=0.5 dex、床は 1e-16。

## 判定

| pred | scope | verdict | evidence | ci_basis | ci_degenerate |
| --- | --- | --- | --- | --- | --- |
| P-main | 2x2 main verdict | I_CELLS_INVALID_S_OP | S-op FAILED → I+ セル（IM・Im）を無効と宣言し、要因計画の主判定は出さない（spec §7 の S-op 行）。以下は I− 側の登録済み対比と R-ext のみ。共通完走 seed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; n=10 |  |  |
| E-drift | M_i: iM - im [dex] | OUT_POS | +0.2806 CI [+0.2139, +0.3312]; ci_degenerate=False | paired percentile | 0 |
| E-drift | M main effect band (I+ cells invalid) | OUT_POS | bands={'M_i': 'OUT_POS'}; interaction=not computed (I+ cells invalid); margin=0.15 dex; interaction_margin=not applicable dex |  |  |
| E-level | M_i: iM - im [dex] | OUT_POS | +5.6879 CI [+5.6619, +5.7158]; ci_degenerate=False | paired percentile | 0 |
| E-level | M main effect band (I+ cells invalid) | OUT_POS | bands={'M_i': 'OUT_POS'}; interaction=not computed (I+ cells invalid); margin=0.15 dex; interaction_margin=not applicable dex |  |  |
| S-floor | B02/B10 floor_frac, all running arms | PASS | iM/B02=0; iM/B10=0; iM_nowd/B02=0; iM_nowd/B10=0; im/B02=0; im/B10=0; im_nowd/B02=0; im_nowd/B10=0; std_anchor/B02=0; std_anchor/B10=0 |  |  |
| S-ceiling | B02 level range over 2 cell(s) | CEILING_CONTAMINATED | range=5.407291 dex; threshold=3.0; levels={'iM': -0.31815396009326213, 'im': -5.72544482694612} |  |  |
| L | E-drift vs E-level ladder | CONSISTENT | E-drift=OUT_POS; E-level=OUT_POS |  |  |
| R-ext | extinction by 5M: im vs im_nowd | BWD_PREVENTS_EXTINCTION | n=10; im_nowd extinct=10/10 CI=0.692-1.000 tasks=[112, 249]; im extinct=0/10 CI=0.000-0.308 tasks=None; reference Aexact=10/10 tasks=[154, 454]; E-level im=-5.6879 / im_nowd=0.0000 (diff=-5.6879 dex, REPORT); strict_dead B10 im=0.0002 / im_nowd=1.0000; median tau im=501.0 / im_nowd=99.5; alive==0 rule disagreements=0 | Clopper-Pearson |  |
| R-ext-M+ | extinction task: iM vs iM_nowd | BWD_DELAYS_EXTINCTION_UNDER_MU | n=10; iM - iM_nowd on the extinction task = +37.5000 CI [+15.6975, +60.6000]; ci_degenerate=False; median task iM=105.5 / iM_nowd=50.5; extinct iM=10/10 / iM_nowd=10/10 (REPORT); E-level B10 iM=0.0000 / iM_nowd=0.0000; b(all units) B10 iM=-0.0000 / iM_nowd=-0.1373; censored={'iM': 0, 'iM_nowd': 0} | paired percentile |  |
| S-ceiling | im_nowd seeds at the ceiling (B10 >= -0.05 dex) | REPORT_ONLY | 10 seeds |  |  |
| D | iM L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.984900->1.000000 |  |  |
| R | iM L1 eff_rank B10 | REPORT_ONLY | 1.000000 |  |  |
| U | iM u_norm / bypass_share B10 | REPORT_ONLY | u_norm=0; bypass_share=0; \|bypass\|=0 |  |  |
| ledger | iM B / M / b(all units) B02->B10 | REPORT_ONLY | B=+0.022629->+nan; M=-0.058697->+nan; b_all=-0.000486->-0.000000 |  |  |
| A | iM E-level B10 / E-drift / E-onset | REPORT_ONLY | E-level=0.000000; E-drift=+0.318154; median tau=22.0; censored=0 |  |  |
| D | im L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.000080->0.000180 |  |  |
| R | im L1 eff_rank B10 | REPORT_ONLY | 19.344016 |  |  |
| U | im u_norm / bypass_share B10 | REPORT_ONLY | u_norm=0; bypass_share=0; \|bypass\|=0 |  |  |
| ledger | im B / M / b(all units) B02->B10 | REPORT_ONLY | B=-0.226009->-0.230802; M=+0.000000->+0.000000; b_all=-0.078871->-0.077615 |  |  |
| A | im E-level B10 / E-drift / E-onset | REPORT_ONLY | E-level=-5.687868; E-drift=+0.037577; median tau=501.0; censored=10 |  |  |
| D | im_nowd L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.829580->1.000000 |  |  |
| R | im_nowd L1 eff_rank B10 | REPORT_ONLY | 1.000000 |  |  |
| U | im_nowd u_norm / bypass_share B10 | REPORT_ONLY | u_norm=0; bypass_share=0; \|bypass\|=0 |  |  |
| ledger | im_nowd B / M / b(all units) B02->B10 | REPORT_ONLY | B=-0.702676->+nan; M=+0.000000->+nan; b_all=-0.856006->-0.968836 |  |  |
| A | im_nowd E-level B10 / E-drift / E-onset | REPORT_ONLY | E-level=0.000000; E-drift=+3.744021; median tau=99.5; censored=0 |  |  |
| D | iM_nowd L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.996720->1.000000 |  |  |
| R | iM_nowd L1 eff_rank B10 | REPORT_ONLY | 1.000000 |  |  |
| U | iM_nowd u_norm / bypass_share B10 | REPORT_ONLY | u_norm=0; bypass_share=0; \|bypass\|=0 |  |  |
| ledger | iM_nowd B / M / b(all units) B02->B10 | REPORT_ONLY | B=-0.004604->+nan; M=+0.570866->+nan; b_all=-0.136605->-0.137305 |  |  |
| A | iM_nowd E-level B10 / E-drift / E-onset | REPORT_ONLY | E-level=0.000000; E-drift=+0.106213; median tau=15.5; censored=0 |  |  |
| D | std_anchor L1 strict_dead_frac B02->B10 | REPORT_ONLY | 0.875420->0.980440 |  |  |
| R | std_anchor L1 eff_rank B10 | REPORT_ONLY | 1.555217 |  |  |
| U | std_anchor u_norm / bypass_share B10 | REPORT_ONLY | u_norm=0; bypass_share=0; \|bypass\|=0 |  |  |
| ledger | std_anchor B / M / b(all units) B02->B10 | REPORT_ONLY | B=-0.257486->-0.067678; M=-0.773462->-0.313076; b_all=-0.514983->-0.948050 |  |  |
| A | std_anchor E-level B10 / E-drift / E-onset | REPORT_ONLY | E-level=-0.314743; E-drift=+1.323076; median tau=64.0; censored=0 |  |  |
| exclusion | iM | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | im | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | im_nowd | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | iM_nowd | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |
| exclusion | std_anchor | ARM_VALID | status=COMPLETE; excluded=[]; included=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |  |  |

## 水準・死・バイパス

| arm | wd_b | window | mean_log10_unfit | L1_dead | L1_eff_rank | b_median_all | u_norm | bypass_share | median_tau | n_censored | floor_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iM | 0.001 | B02 | -0.318154 | 0.9849 | 1.35953 | -0.000486218 | 0 | 0 | 22 | 0 | 0 |
| iM | 0.001 | B10 | 6.07526e-18 | 1 | 1 | -2.21181e-21 | 0 | 0 | 22 | 0 | 0 |
| im | 0.001 | B02 | -5.72544 | 8e-05 | 19.4739 | -0.0788715 | 0 | 0 | 501 | 10 | 0 |
| im | 0.001 | B10 | -5.68787 | 0.00018 | 19.344 | -0.0776153 | 0 | 0 | 501 | 10 | 0 |
| im_nowd | 0 | B02 | -3.74402 | 0.82958 | 9.991 | -0.856006 | 0 | 0 | 99.5 | 0 | 0 |
| im_nowd | 0 | B10 | 6.07526e-18 | 1 | 1 | -0.968836 | 0 | 0 | 99.5 | 0 | 0 |
| iM_nowd | 0 | B02 | -0.106213 | 0.99672 | 1 | -0.136605 | 0 | 0 | 15.5 | 0 | 0 |
| iM_nowd | 0 | B10 | 6.07526e-18 | 1 | 1 | -0.137305 | 0 | 0 | 15.5 | 0 | 0 |
| std_anchor | 0 | B02 | -1.63782 | 0.87542 | 6.61907 | -0.514983 | 0 | 0 | 64 | 0 | 0 |
| std_anchor | 0 | B10 | -0.314743 | 0.98044 | 1.55522 | -0.94805 | 0 | 0 | 64 | 0 | 0 |

## 登録副判定 R-ext（死の主張）

- 判定: **BWD_PREVENTS_EXTINCTION**
- n=10; im_nowd extinct=10/10 CI=0.692-1.000 tasks=[112, 249]; im extinct=0/10 CI=0.000-0.308 tasks=None; reference Aexact=10/10 tasks=[154, 454]; E-level im=-5.6879 / im_nowd=0.0000 (diff=-5.6879 dex, REPORT); strict_dead B10 im=0.0002 / im_nowd=1.0000; median tau im=501.0 / im_nowd=99.5; alive==0 rule disagreements=0
- **`LOP_CURED` と読み替えない。** dead と機能が逆向きに動いた実例が 2 件ある。

## フラグ

- `CEILING_CONTAMINATED`: 要因 4 セルの B02 水準差が 3 dex を超えたため **E-drift 単独では読まない**。

## 引いてはいけない線（spec §9）

- **主判定の読みは「b 拘束下（λ=1e-3）」限定。** λ は移送値であり、結果を見て選び直さない。
- `I+` は「可識別性あり」ではなく**ゲートに触れない経路での可識別性あり**である。A2 のバイパスが運ぶのは出力側の大域的な加法成分 1 自由度であり、std がタスク内で使っている per-unit の実効バイアスの分け前ではない。
- per-unit の可識別性は前活性のタスク依存 DC そのものなので µ と分離できない（補題）。本走はそこを閉じない。
- 要因 4 セルはいずれも「境界ごとの µ の引き直し」を持たない。本走の µ は**静的な壁**の効果である。
- `im_nowd` が全滅しても発見として引かない（committed `Aexact` の再現）。`S-mu` の PASS も構成上ほぼ恒真。
- `std_anchor` は 2 因子が縮退した外部アンカーであって要因計画のセルではない。
- `strict_dead` は主判定に使っていない（R-ext の全滅到達だけが明示的な例外で、死についてのみ語る）。

## 集計上の注意

- `unfit` は 32 パターン上の**分散比**で DC を見ないので、task 内で定数のバイパスは `unfit` を動かさない。**I 主効果が E-drift に出たら、それは表現力のアーティファクトではありえず力学経由である**（spec §6.1）。DC を見る量は `eval_loss_exact`（バイパス込みで記録）である。
- 同じ理由で `bypass_share` は分散の分け前ではなく**出力パワーの分け前**。
- E-onset（REPORT_ONLY）は「一度閾値を下回ったあとの最初の上抜け」と定義した（素直に読むと初期過渡で全 seed が tau=1 になるため）。**spec の字義から離れて残っている唯一の点。**
- R-ext の全滅は `strict_dead_frac == 1.0` で判定し、`alive == 0` の別定義との食い違い件数を併記する（spec §6.4 の裁定どおり）。
