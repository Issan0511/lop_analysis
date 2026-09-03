# valley_off_0903 — 谷の逃走・走 A（オラクルなしの自然な condA）

spec: `specs/spec_valley_off_0903.md` / vault: 可塑性喪失/spec/谷の逃走_走A_spec_0903.md

**登録判定（`G_off`）: PARTIAL**
（満たした行: PARTIAL）

窓はタスク終端 10 点のみ（親走 `gate_dose_0830` の U_k と同じ作り方）。沈下は `layer1_zmax <= 0`、凍結は `|layer1_mob| < 1e-6`（**本走のロガー**。`gate_dial_0902` の `u_fr` 経由の凍結率とは別定義）。対照は別走の committed 値で、ユニット別量は `p_hat` 代用（ReLU では厳密・ELU/leaky では空欄）。

## 主 endpoint

| 腕 | 5M 発症 | 5M median log10 U | 1M 発症 | 1M median log10 U |
| --- | --- | --- | --- | --- |
| G_off | 10/10 | -0.0055 | 1/10 | -1.9413 |
| S_off | 10/10 | -0.4985 | 1/10 | -1.8979 |
| R_off（対照・転記） | 10/10 | -0.0930 | 9/10 | -0.6748 |
| E_off（対照・転記） | 0/10 | -1.9704 | 0/10 | -2.1641 |
| LR_off（対照・転記） | 0/10 | -2.2755 | 0/10 | -3.2451 |

## paired 差（対 `R_off`・別走の committed 値）

| 対比 | 窓 | 点 | percentile CI | 符号 |
| --- | --- | --- | --- | --- |
| G_off-R_off | 5M | +0.0806 | [+0.0035, +0.2724] | 9:1 |
| G_off-R_off | 1M | -1.1631 | [-1.8459, -0.8037] | 0:10 |
| S_off-R_off | 5M | -0.4129 | [-0.4556, -0.3058] | 0:10 |
| S_off-R_off | 1M | -1.2103 | [-1.3623, -0.8878] | 0:10 |

**5M では ReLU が天井にいるので「ReLU より悪い」は 5M では検定できない。検定は 1M 窓の行だけ。**

## 末尾窓のユニット別量（5M・tasks 491-500）

| 腕 | 沈下率 | 深さ中央値 | 凍結率 | 谷の向こう率 | 出所 |
| --- | --- | --- | --- | --- | --- |
| G_off | 1.0000 | 13.47 | 0.9825 | 1.0000 | layer1_mob |
| S_off | 0.9995 | 18.36 | 0.3690 | 0.9865 | layer1_mob |
| R_off | 0.9915 | 5.77 | 0.9915 | nan | layer1_p_hat (exact on ReLU: E_x phi' = p_hat) |
| E_off | 0.4545 | 9.34 | nan | nan | unavailable (no mob column, proxy invalid off ReLU) |
| LR_off | 0.6300 | 3.99 | nan | nan | unavailable (no mob column, proxy invalid off ReLU) |

## 事前予測の照合（spec §5・走の前に固定）

| # | 的中 | 中身 |
| --- | --- | --- |
| A1 | ✗ | n_onset_5m=10, n_onset_1m=1 |
| A2 | ✗ | window=1M, point=-1.163128617098634, ci=[-1.8458977826410474, -0.8036578561972048], sign=0:10 |
| A3 | ✓ | submerged_frac=1.0, depth_median=13.468639612197876, frozen_frac=0.9824999999999999, frozen_source=layer1_mob, n_seeds_deeper=10 |
| A4（REPORT_ONLY） | ✓ | n_onset_5m=10, depth_median=18.359731674194336, submerged_frac=0.9995 |
| A5（REPORT_ONLY） | ✓ | median_rate_per_unit_record=0.0020375924815036993, total_escapes_within_task=10147, total_escapes_across_boundary=10568 |

的中 3/5。

## 前段チェック

- S1_omp: PASS
- S_dial: PASS
- S_fd: PASS
- S_num: PASS
- S_limit: PASS
- S_mob: PASS
- S_log_b: PASS
- S_pair: PASS
- S_ref: PASS
- S_floor: PASS
- S_ci: PASS

## 引用上の注意

- 対照は**別走の committed 値**。ペアリングは init・教師・入力実現まで（step 1 以降は軌道が分岐する）。
- 0/10 は「5M までに観測しなかった」（片側 95% 上限 0.2589）。
- 凍結率・沈下率は出所と窓を添えて引用する。
- `layer1_dose` は**測っただけ**の量（本走はオラクルを掛けていない）。
