# clamp0 summary

**verdict: PARTIAL**

| arm | orig | clamp | n_onset | log10_u | log10_u_orig | log10_u_clamp | log10_u_relu | d_orig | d_orig_lo | d_orig_hi | d_orig_sign | d_clamp | d_clamp_lo | d_clamp_hi | d_clamp_sign | d_relu | d_relu_lo | d_relu_hi | d_relu_sign | relu_equivalent | depth_q50 | frozen | revive_across | family_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gz_b1_1216 | G_b1_1216 | Gc_b1_1216 | 8 | -0.7663 | -0.003016 | -0.5276 | -0.2747 | -0.7643 | -1.149 | -0.5484 | 10:0 | -0.06519 | -0.5483 | 0.05172 | 7:3 | -0.3621 | -0.8643 | -0.2221 | 10:0 | 0 | 6.118 | 0.9055 | 1925 | MIXED |
| Sz_b3_1216 | S_b3_1216 | Sc_b3_1216 | 10 | -0.4674 | -9.022e-06 | -0.4617 | -0.2747 | -0.4674 | -0.6151 | -0.2876 | 10:0 | 0.04696 | -0.01654 | 0.1404 | 4:6 | -0.213 | -0.3408 | -0.1185 | 9:1 | 0 | 6.052 | 0.9545 | 1517 | FLOOR_IRRELEVANT |

引用上の注意: 用量 12.16・1 層・幅 100・seed 0–9。0/10 は片側 95% 上限 0.2589 の強さ。
対照は別走の committed ログ（init・教師・入力列・flip は同一、軌道は step 1 以降で分岐）。対照ログの sha256 は provenance.json の reference_logs。
