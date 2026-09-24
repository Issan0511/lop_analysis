# Input-scope effective-displacement validation

Run state: **INTERIM / INCOMPLETE**; scope: **LR-only interim, 60/130 registered series selected**. 60/60 selected series complete. Full registration has 130 series.

## Interpretation and definitions

Reference P1–P4 use the first fixed input bank and the previous registered rules. Actual A_BALANCE/A_PLATEAU use the current bank for Q and X, with G_pre measured before the weight update. In actual rows V=Vprev_old for the ledger and trends, while c and rho use Vprev_current so their geometry matches Q/X; the source is explicit in `c_rho_covariance_source`. They are separate questions. An exact variance or paired-bank identity is a bookkeeping check, not scientific support.

When this run is INCOMPLETE, displayed group verdicts are provisional diagnostics, not final claims.

X0Y1 removes the persistent task cue while changing targets; X1−X0 is a paired context contrast, not a pure causal effect of input mean. Fixed input does not guarantee stationary updates or a plateau. STOPPED means low relative update supply, not necessarily zero gradient or inability to learn. LOW_RESPONSE is an operational three-task marker, not proven loss of plasticity.

CondA uses changing mean/support with constant centered covariance; MNIST X1 also changes centered covariance. The mean channel and paired-ID function change are reported separately from centered V/Q/X. A_BALANCE may pass through input-switch cancellation and is not by itself evidence of update-driven pruning. No observed G_pre is used as held-out prediction.

## Fixed-input sufficiency on observed horizons

| environment | arm | window | question | verdict | n_pass | seed_median | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conda | LR_k1_X0Y1 | primary400 | P2 | FAIL | 0 | 0.2895 | 0.204 | 0.7399 |
| conda | LR_k1_X0Y1 | primary400 | P4 | NOT_ESTABLISHED | 0 | -0.03608 | -0.04816 | -0.01529 |
| conda | LR_k7_X0Y1 | primary400 | P2 | FAIL | 1 | 0.2558 | -0.08505 | 0.8677 |
| conda | LR_k7_X0Y1 | primary400 | P4 | NOT_ESTABLISHED | 0 | -0.013 | -0.1323 | 0.01449 |
| mnist | LR_X0Y1 | long150 | P2 | FAIL | 0 | 1.192 | 0.8226 | 1.216 |
| mnist | LR_X0Y1 | long150 | P4 | NOT_ESTABLISHED | 0 | -0.09553 | -0.1661 | 0.3345 |
| mnist | LR_X0Y1 | primary50 | P2 | FAIL | 0 | 0.7477 | 0.5564 | 0.9749 |
| mnist | LR_X0Y1 | primary50 | P4 | NOT_ESTABLISHED | 0 | 0.4875 | 0.215 | 0.6392 |

A finite-horizon failure supports only insufficiency under the observed optimizer, activation, and horizon.

## All group verdicts

| environment | arm | window | metric | question | verdict | n_pass | n_seed | seed_median | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conda | LR_k1_X0Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1338 | -0.7031 | 0.7825 |
| conda | LR_k1_X0Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4091 | 0.1344 | 0.5605 |
| conda | LR_k1_X0Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.1338 | -0.7031 | 0.7825 |
| conda | LR_k1_X0Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4091 | 0.1344 | 0.5605 |
| conda | LR_k1_X0Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.2795 | 0.1923 | 0.3403 |
| conda | LR_k1_X0Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1011 | -0.1406 | -0.04128 |
| conda | LR_k1_X0Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -1.476 | -2.541 | -1.059 |
| conda | LR_k1_X0Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.5899 | 0.4674 | 0.6404 |
| conda | LR_k1_X0Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -1.476 | -2.541 | -1.059 |
| conda | LR_k1_X0Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.5899 | 0.4674 | 0.6404 |
| conda | LR_k1_X0Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.05275 | 0.01521 | 0.1679 |
| conda | LR_k1_X0Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.2153 | -0.2449 | -0.2013 |
| conda | LR_k1_X0Y1 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.6995 | 0.5422 | 0.8031 |
| conda | LR_k1_X0Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.2895 | 0.204 | 0.7399 |
| conda | LR_k1_X0Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.6995 | 0.5422 | 0.8031 |
| conda | LR_k1_X0Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.2895 | 0.204 | 0.7399 |
| conda | LR_k1_X0Y1 | primary400 | reference | P3 | FAIL | 3 | 5 | 0.1198 | 0.02702 | 0.4874 |
| conda | LR_k1_X0Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.03608 | -0.04816 | -0.01529 |
| conda | LR_k1_X0Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1202 | -0.3168 | 0.2148 |
| conda | LR_k1_X0Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.5433 | 0.4258 | 0.7657 |
| conda | LR_k1_X0Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | 0.1202 | -0.3168 | 0.2148 |
| conda | LR_k1_X0Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.5433 | 0.4258 | 0.7657 |
| conda | LR_k1_X0Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.07748 | 0.03077 | 0.189 |
| conda | LR_k1_X0Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1054 | -0.1411 | -0.07916 |
| conda | LR_k1_X1Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.2178 | -0.1994 | 0.2635 |
| conda | LR_k1_X1Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3983 | 0.1683 | 0.4949 |
| conda | LR_k1_X1Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.2178 | -0.1994 | 0.2635 |
| conda | LR_k1_X1Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.3983 | 0.1683 | 0.4949 |
| conda | LR_k1_X1Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.2685 | 0.2418 | 0.4483 |
| conda | LR_k1_X1Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.05776 | -0.08606 | -0.02635 |
| conda | LR_k1_X1Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -1.329 | -2.749 | -0.5112 |
| conda | LR_k1_X1Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4881 | 0.3966 | 0.8933 |
| conda | LR_k1_X1Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -1.329 | -2.749 | -0.5112 |
| conda | LR_k1_X1Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.4881 | 0.3966 | 0.8933 |
| conda | LR_k1_X1Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1808 | 0.01913 | 0.2263 |
| conda | LR_k1_X1Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.3255 | -0.5121 | -0.29 |
| conda | LR_k1_X1Y1 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.7925 | 0.6274 | 0.8422 |
| conda | LR_k1_X1Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3827 | 0.1521 | 0.5387 |
| conda | LR_k1_X1Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.7925 | 0.6274 | 0.8422 |
| conda | LR_k1_X1Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.3827 | 0.1521 | 0.5387 |
| conda | LR_k1_X1Y1 | primary400 | reference | P3 | FAIL | 1 | 5 | 0.2191 | 0.04312 | 0.2645 |
| conda | LR_k1_X1Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.001902 | -0.01245 | 0.08122 |
| conda | LR_k1_X1Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | 0.2298 | -0.2275 | 0.2591 |
| conda | LR_k1_X1Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4818 | 0.4163 | 1.108 |
| conda | LR_k1_X1Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | 0.2298 | -0.2275 | 0.2591 |
| conda | LR_k1_X1Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.4818 | 0.4163 | 1.108 |
| conda | LR_k1_X1Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.09957 | 0.06123 | 0.1673 |
| conda | LR_k1_X1Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.08772 | -0.1751 | -0.045 |
| conda | LR_k7_X0Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1338 | -0.7031 | 0.7825 |
| conda | LR_k7_X0Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4091 | 0.1344 | 0.5605 |
| conda | LR_k7_X0Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.1338 | -0.7031 | 0.7825 |
| conda | LR_k7_X0Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4091 | 0.1344 | 0.5605 |
| conda | LR_k7_X0Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.2795 | 0.1923 | 0.3403 |
| conda | LR_k7_X0Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1011 | -0.1406 | -0.04128 |
| conda | LR_k7_X0Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -1.476 | -2.541 | -1.059 |
| conda | LR_k7_X0Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.5899 | 0.4674 | 0.6404 |
| conda | LR_k7_X0Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -1.476 | -2.541 | -1.059 |
| conda | LR_k7_X0Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.5899 | 0.4674 | 0.6404 |
| conda | LR_k7_X0Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.05275 | 0.01521 | 0.1679 |
| conda | LR_k7_X0Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.2153 | -0.2449 | -0.2013 |
| conda | LR_k7_X0Y1 | primary400 | actual | A_BALANCE | FAIL | 2 | 5 | 0.8449 | 0.5236 | 1.027 |
| conda | LR_k7_X0Y1 | primary400 | actual | A_PLATEAU | FAIL | 2 | 5 | 0.2558 | -0.08505 | 0.8677 |
| conda | LR_k7_X0Y1 | primary400 | reference | P1 | FAIL | 1 | 5 | 0.8449 | 0.5236 | 1.027 |
| conda | LR_k7_X0Y1 | primary400 | reference | P2 | FAIL | 1 | 5 | 0.2558 | -0.08505 | 0.8677 |
| conda | LR_k7_X0Y1 | primary400 | reference | P3 | FAIL | 2 | 5 | 0.1472 | 0.04138 | 0.248 |
| conda | LR_k7_X0Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.013 | -0.1323 | 0.01449 |
| conda | LR_k7_X0Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -0.08235 | -0.1626 | 0.3479 |
| conda | LR_k7_X0Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.6707 | 0.4408 | 0.7662 |
| conda | LR_k7_X0Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | -0.08235 | -0.1626 | 0.3479 |
| conda | LR_k7_X0Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.6707 | 0.4408 | 0.7662 |
| conda | LR_k7_X0Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1364 | 0.04285 | 0.219 |
| conda | LR_k7_X0Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.08641 | -0.1558 | -0.08094 |
| conda | LR_k7_X1Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1294 | 0.0981 | 0.2987 |
| conda | LR_k7_X1Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4873 | 0.4315 | 0.524 |
| conda | LR_k7_X1Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.1294 | 0.0981 | 0.2987 |
| conda | LR_k7_X1Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4873 | 0.4315 | 0.524 |
| conda | LR_k7_X1Y0 | primary400 | reference | P3 | PASS | 4 | 5 | 0.06714 | 0.03248 | 0.1037 |
| conda | LR_k7_X1Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.07062 | 0.002383 | 0.2281 |
| conda | LR_k7_X1Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1936 | -0.6529 | 0.5132 |
| conda | LR_k7_X1Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.229 | 0.1396 | 0.616 |
| conda | LR_k7_X1Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | 0.1936 | -0.6529 | 0.5132 |
| conda | LR_k7_X1Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.229 | 0.1396 | 0.616 |
| conda | LR_k7_X1Y0 | secondary100 | reference | P3 | PASS | 5 | 5 | 0.04304 | 0.03061 | 0.09417 |
| conda | LR_k7_X1Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.185 | -0.2439 | -0.1244 |
| conda | LR_k7_X1Y1 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.835 | 0.7159 | 0.8714 |
| conda | LR_k7_X1Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3462 | 0.0797 | 0.4602 |
| conda | LR_k7_X1Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.835 | 0.7159 | 0.8714 |
| conda | LR_k7_X1Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.3462 | 0.0797 | 0.4602 |
| conda | LR_k7_X1Y1 | primary400 | reference | P3 | FAIL | 2 | 5 | 0.1225 | 0.09655 | 0.1406 |
| conda | LR_k7_X1Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.06075 | -0.005777 | 0.07891 |
| conda | LR_k7_X1Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | 0.1217 | 0.0423 | 0.3994 |
| conda | LR_k7_X1Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.5312 | 0.3416 | 0.8179 |
| conda | LR_k7_X1Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | 0.1217 | 0.0423 | 0.3994 |
| conda | LR_k7_X1Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.5312 | 0.3416 | 0.8179 |
| conda | LR_k7_X1Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.06014 | 0.02114 | 0.1446 |
| conda | LR_k7_X1Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1036 | -0.1396 | -0.05672 |
| mnist | LR_X0Y0 | primary50 | actual | A_BALANCE | FAIL | 0 | 5 | -0.3293 | -0.788 | 0.194 |
| mnist | LR_X0Y0 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 1.107 | 0.8419 | 1.414 |
| mnist | LR_X0Y0 | primary50 | reference | P1 | FAIL | 0 | 5 | -0.3293 | -0.788 | 0.194 |
| mnist | LR_X0Y0 | primary50 | reference | P2 | FAIL | 0 | 5 | 1.107 | 0.8419 | 1.414 |
| mnist | LR_X0Y0 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.03209 | 0.01989 | 0.09271 |
| mnist | LR_X0Y0 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.2651 | -0.4338 | -0.005621 |
| mnist | LR_X0Y1 | long150 | actual | A_BALANCE | PASS | 5 | 5 | 0.95 | 0.9398 | 0.9619 |
| mnist | LR_X0Y1 | long150 | actual | A_PLATEAU | FAIL | 0 | 5 | 1.192 | 0.8226 | 1.216 |
| mnist | LR_X0Y1 | long150 | reference | P1 | PASS | 5 | 5 | 0.95 | 0.9398 | 0.9619 |
| mnist | LR_X0Y1 | long150 | reference | P2 | FAIL | 0 | 5 | 1.192 | 0.8226 | 1.216 |
| mnist | LR_X0Y1 | long150 | reference | P3 | FAIL | 0 | 5 | 0.5469 | 0.5215 | 0.554 |
| mnist | LR_X0Y1 | long150 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.09553 | -0.1661 | 0.3345 |
| mnist | LR_X0Y1 | primary50 | actual | A_BALANCE | PASS | 5 | 5 | 0.9245 | 0.91 | 0.942 |
| mnist | LR_X0Y1 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.7477 | 0.5564 | 0.9749 |
| mnist | LR_X0Y1 | primary50 | reference | P1 | PASS | 5 | 5 | 0.9245 | 0.91 | 0.942 |
| mnist | LR_X0Y1 | primary50 | reference | P2 | FAIL | 0 | 5 | 0.7477 | 0.5564 | 0.9749 |
| mnist | LR_X0Y1 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.3649 | 0.3588 | 0.3884 |
| mnist | LR_X0Y1 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.4875 | 0.215 | 0.6392 |
| mnist | LR_X1Y0 | primary50 | actual | A_BALANCE | FAIL | 0 | 5 | 0.6517 | 0.6278 | 0.7095 |
| mnist | LR_X1Y0 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 1.663 | 1.622 | 1.884 |
| mnist | LR_X1Y0 | primary50 | reference | P1 | FAIL | 0 | 5 | 0.3127 | 0.2483 | 0.349 |
| mnist | LR_X1Y0 | primary50 | reference | P2 | FAIL | 0 | 5 | 1.54 | 1.402 | 1.75 |
| mnist | LR_X1Y0 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.3764 | 0.2795 | 0.4062 |
| mnist | LR_X1Y0 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.04302 | -0.07288 | 0.1262 |
| mnist | LR_X1Y1 | primary50 | actual | A_BALANCE | FAIL | 0 | 5 | 0.6764 | 0.6477 | 0.6977 |
| mnist | LR_X1Y1 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 1.65 | 1.541 | 1.69 |
| mnist | LR_X1Y1 | primary50 | reference | P1 | FAIL | 0 | 5 | 0.3252 | 0.253 | 0.408 |
| mnist | LR_X1Y1 | primary50 | reference | P2 | FAIL | 0 | 5 | 1.613 | 1.426 | 1.654 |
| mnist | LR_X1Y1 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.3531 | 0.3265 | 0.4505 |
| mnist | LR_X1Y1 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.01284 | -0.1227 | 0.0954 |

## Paired comparisons

All registered X/Y, k, activation, Snake, and horizon contrasts appear seed by seed in `paired_seed.csv`, with 5-seed median and bootstrap95% intervals in `paired_group.csv`. `group_stats.csv` gives the same intervals for every numeric seed summary field. Every listed field is reported, including null contrasts; no outcome field was selected after viewing results.

## Traceability

`source_paths.csv` records snapshot and input paths, `source_windows.csv` records fit/held-out/late windows, and `spec_hashes.csv` records the exact preregistration hashes. `transitions.csv` contains per-task raw/reference/actual values and the paired decomposition. `unit_late.csv` holds late per-unit aggregates. `closure.csv` contains fixed-reference held-out trajectories only.

## Response and learning context

| environment | arm | window | seed | response_censoring | low_response_onset_task | active_window_status | dynamics_label | late_task_start_mse | late_mse | late_task_start_acc | late_train_acc | late_grad_w1_start_norm | late_grad_w1_end_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conda | LR_k1_X0Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003436 | 0.0003578 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 0.0003287 | 0.0003275 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0004082 | 0.0004921 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003256 | 0.0003305 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001296 | 0.001397 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0008945 | 0.0008958 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0002388 | 0.0002083 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003733 | 0.0003752 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0005677 | 0.0005676 | nan | nan | nan | nan |
| conda | LR_k1_X0Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.0003679 | 0.0003661 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.471 | 0.007418 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.622 | 0.006066 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.514 | 0.01318 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.207 | 0.006144 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.36 | 0.006016 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.325 | 0.005464 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.695 | 0.0121 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.252 | 0.008204 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 1.764 | 0.00443 | nan | nan | nan | nan |
| conda | LR_k1_X0Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.87 | 0.003315 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.04834 | 0.0005241 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.01321 | 0.000462 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1732 | 0.0005444 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 0.04976 | 0.0004889 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.08576 | 0.0005922 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.01469 | 0.0006786 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.052 | 0.0002643 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.02702 | 0.0002087 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.07076 | 0.0006897 | nan | nan | nan | nan |
| conda | LR_k1_X1Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.01834 | 0.0004353 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.597 | 0.01328 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.589 | 0.009303 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.778 | 0.003958 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.509 | 0.005051 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.211 | 0.00829 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.149 | 0.006798 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.837 | 0.01112 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.266 | 0.008715 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 1.763 | 0.008873 | nan | nan | nan | nan |
| conda | LR_k1_X1Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.847 | 0.006814 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003436 | 0.0003578 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 0.0003287 | 0.0003275 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0004082 | 0.0004921 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003256 | 0.0003305 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001296 | 0.001397 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0008945 | 0.0008958 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0002388 | 0.0002083 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0003733 | 0.0003752 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.0005677 | 0.0005676 | nan | nan | nan | nan |
| conda | LR_k7_X0Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.0003679 | 0.0003661 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.256 | 0.007665 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.571 | 0.005 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.242 | 0.009899 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.678 | 0.006451 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 7.266 | 0.01205 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 4.801 | 0.003601 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 7.365 | 0.0182 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | 5.435 | 0.00573 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.283 | 0.004595 | nan | nan | nan | nan |
| conda | LR_k7_X0Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.827 | 0.005662 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.06629 | 0.0003861 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.01335 | 0.0003313 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.3906 | 0.00207 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.02654 | 0.0003843 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.08208 | 0.0007827 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.004504 | 0.0008334 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1145 | 0.0001775 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.02291 | 0.0002665 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1615 | 0.000965 | nan | nan | nan | nan |
| conda | LR_k7_X1Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 0.0241 | 0.0003985 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.05 | 0.005281 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 3.987 | 0.006362 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.12 | 0.004843 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.685 | 0.006829 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.666 | 0.007948 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 3.231 | 0.004224 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 6.065 | 0.003973 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.746 | 0.006702 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 6.743 | 0.01426 | nan | nan | nan | nan |
| conda | LR_k7_X1Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.979 | 0.005892 | nan | nan | nan | nan |
| mnist | LR_X0Y0 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 0.002297 | 0.002297 |
| mnist | LR_X0Y0 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 0.0003395 | 0.0003411 |
| mnist | LR_X0Y0 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 4.164e-05 | 3.672e-05 |
| mnist | LR_X0Y0 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 0.0001399 | 0.000139 |
| mnist | LR_X0Y0 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 0.0004251 | 0.000439 |
| mnist | LR_X0Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1013 | 1 | 19.02 | 0.06461 |
| mnist | LR_X0Y1 | long150 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09831 | 0.9965 | 10.98 | 0.132 |
| mnist | LR_X0Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.0985 | 0.9992 | 12.85 | 0.08239 |
| mnist | LR_X0Y1 | long150 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1006 | 0.9949 | 7.954 | 0.1775 |
| mnist | LR_X0Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09842 | 1 | 14.52 | 0.05493 |
| mnist | LR_X0Y1 | long150 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09628 | 0.9999 | 9.417 | 0.06704 |
| mnist | LR_X0Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09467 | 1 | 13.15 | 0.05348 |
| mnist | LR_X0Y1 | long150 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09928 | 0.9999 | 8.225 | 0.07826 |
| mnist | LR_X0Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1078 | 1 | 17.59 | 0.04678 |
| mnist | LR_X0Y1 | long150 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09947 | 1 | 9.954 | 0.0479 |
| mnist | LR_X1Y0 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09492 | 0.9991 | 14.79 | 0.07354 |
| mnist | LR_X1Y0 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09625 | 0.9962 | 21.45 | 0.0586 |
| mnist | LR_X1Y0 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09975 | 1 | 16.03 | 0.006776 |
| mnist | LR_X1Y0 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09992 | 0.9979 | 18.67 | 0.03826 |
| mnist | LR_X1Y0 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1004 | 1 | 18.6 | 0.01394 |
| mnist | LR_X1Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1054 | 0.9972 | 16.44 | 0.01005 |
| mnist | LR_X1Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09908 | 0.9874 | 17.61 | 0.08543 |
| mnist | LR_X1Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1003 | 1 | 16.29 | 0.007649 |
| mnist | LR_X1Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1001 | 1 | 18.46 | 0.01213 |
| mnist | LR_X1Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09592 | 0.9956 | 18.88 | 0.02852 |


## Post hoc: user's question about D/(2|c|), 2026-09-24

This diagnostic was added after the interim outcomes, in response to whether the measured effective radius is above or far below D/(2|c|). It does not replace registered verdicts. Sources: threshold_relation_posthoc.csv (seed rows), threshold_relation_posthoc_groups.csv (seed medians). Windows and numerator timing are explicit in those files.

Let R=sqrt(Vprev), D=sqrt(Q), and Rstar=D/(-2c), defined here only when c<0 and Q,Vprev>0. The taskwise ratio is R/Rstar=-2X/Q. Therefore closeness computed from the same observed update is algebraic balance, not an independent prediction. The registered window B=-2sum(X)/sum(Q) is a Q-weighted average of signed per-task balance; it need not equal a median of ratios, especially when c changes sign.

| arm | window | median within seed, then across seeds: R/Rstar | interpretation |
| --- | --- | ---: | --- |
| MNIST LR X0Y1 | task41–50 | 0.932961 | c<0 in all late transitions; close below, sometimes above |
| MNIST LR X0Y1 | task121–150 | 0.957075 | c<0 in all late transitions; close below, sometimes above |
| CondA LR k1 X0Y1 | task321–400 | 1.550220 | only c<0 transitions; c<0 count median55/80 |
| CondA LR k7 X0Y1 | task321–400 | 1.864043 | only c<0 transitions; c<0 count median58/80 |

For MNIST task121–150, the within-window second-half/first-half mean Rstar ratio has seed median1.058700, while that of R is1.069143. The observed zero-drift threshold is itself rising. This is descriptive, not proof of future threshold tracking.

For CondA, above-threshold negative-c tasks shrink centered V by the exact identity. Other tasks increase V; a positive net drift across the window is compatible with a median above-threshold ratio among only the negative-c tasks. c>=0 tasks have no positive Rstar under this erosion formula. A stochastic equilibrium could still involve both signs; their coexistence alone does not disprove equilibrium.


### Post hoc: which component moves the threshold?

Source: threshold_components_posthoc.csv, comparison=late50_to_late150, LR_X0Y1/reference, paired seed ratios of mean values in task41–50 versus121–150. Seed-median ratios: D=1.424182, |c|=.872882, Rstar=1.646722, R=1.681922. Ratios are independently aggregated across seeds; dividing the displayed medians need not exactly reproduce the Rstar ratio. Fixed input does not make the update statistics fixed. These same-update diagnostics motivate a conditional moving-threshold model but are not an independent predictive validation.


### Post hoc: original RL-CIFAR source does not establish a plateau

Responding to Issa asking why RL converges, inspected the exact CSV used by the original claim (sgd_postfit_cifar_0923/posthoc_c_eff.csv, provenance in original_rlcifar_posthoc_provenance.json), rather than the0918 reference used in the first validation. This is a fresh posthoc audit of existing data, not new training. Sources: original_rlcifar_threshold_posthoc.csv and original_rlcifar_plateau_posthoc.csv.

Original Adam arm A, 10seeds: paired ratios of task41–50 means over task31–40 means have medians R=1.083107, D=1.052662, Rstar=1.057580. Applying the same P1/P2 definitions to the available t2..50 transitions gives P1 10/10, P2 0/10; late V log-log slope seed median=.519777, late B=.949297. This does not prove unbounded growth, but a completed convergence of width is not demonstrated. The original note itself reports positive per-task growth of squared width; the claim that effective growth has stopped is stronger than these observations.

Other original arms: F P2 0/10, SA_iid2/10, S_hi1/10, all P1 10/10. These are reported counts under a posthoc common gate; original outcomes and source files were not changed.


### Post hoc clarification: CondA SGD

Issa clarified the condition of interest was CondA SGD. This uses the prior completed effdisp_validation_0924 k1_sgd arm (leaky .1, SGD .01, 100tasks, 1persistentbit flip per task), not the current Adam inputscope factorial. Registered source: prior seed_summary.csv/verdict.csv, conda/k1_sgd/effective, late81–100: P1/P2 0/5. Median task-start MSE2.5297185, task-end MSE.0211025; all NO_LOW_RESPONSE.

Additional posthoc source prior_conda_sgd_threshold_posthoc.csv: median negative-c task fraction=.6; paired mean R ratio task81–100 over61–80=1.073788. No fixed-input SGD factorial has been run. Partial cancellation and a small effective-update activity flag must not be called a failure to learn individual tasks.
