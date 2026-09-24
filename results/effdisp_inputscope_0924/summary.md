# Input-scope effective-displacement validation

Run state: **COMPLETE**; scope: **all 130 registered series**. 130/130 selected series complete. Full registration has 130 series.

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
| conda | SNA06_k1_X0Y1 | primary400 | P2 | FAIL | 0 | -0.1072 | -1.36 | 2.194 |
| conda | SNA06_k1_X0Y1 | primary400 | P4 | NOT_ESTABLISHED | 0 | 0.0006465 | -0.1124 | 0.09655 |
| conda | SNA06_k7_X0Y1 | primary400 | P2 | FAIL | 0 | 0.3392 | -0.2131 | 0.8898 |
| conda | SNA06_k7_X0Y1 | primary400 | P4 | NOT_ESTABLISHED | 0 | 0.0009955 | -0.07483 | 0.02129 |
| mnist | LR_X0Y1 | long150 | P2 | FAIL | 0 | 1.192 | 0.8226 | 1.216 |
| mnist | LR_X0Y1 | long150 | P4 | NOT_ESTABLISHED | 0 | -0.09553 | -0.1661 | 0.3345 |
| mnist | LR_X0Y1 | primary50 | P2 | FAIL | 0 | 0.7477 | 0.5564 | 0.9749 |
| mnist | LR_X0Y1 | primary50 | P4 | NOT_ESTABLISHED | 0 | 0.4875 | 0.215 | 0.6392 |
| mnist | SNA06_X0Y1 | long150 | P2 | FAIL | 0 | 0.447 | 0.2594 | 0.7335 |
| mnist | SNA06_X0Y1 | long150 | P4 | NOT_ESTABLISHED | 0 | 0.2977 | 0.0481 | 0.3884 |
| mnist | SNA06_X0Y1 | primary50 | P2 | FAIL | 0 | 0.4391 | 0.3076 | 0.4409 |
| mnist | SNA06_X0Y1 | primary50 | P4 | NOT_ESTABLISHED | 0 | 0.251 | 0.2064 | 0.3979 |

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
| conda | SNA06_k1_X0Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | -46.59 | -263 | -19.4 |
| conda | SNA06_k1_X0Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4774 | 0.4581 | 0.7624 |
| conda | SNA06_k1_X0Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | -46.59 | -263 | -19.4 |
| conda | SNA06_k1_X0Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4774 | 0.4581 | 0.7624 |
| conda | SNA06_k1_X0Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.3672 | 0.2209 | 0.5111 |
| conda | SNA06_k1_X0Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.04685 | -0.07958 | -0.01597 |
| conda | SNA06_k1_X0Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -4.543 | -90.11 | -2.15 |
| conda | SNA06_k1_X0Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.447 | 0.1844 | 0.6268 |
| conda | SNA06_k1_X0Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -4.543 | -90.11 | -2.15 |
| conda | SNA06_k1_X0Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.447 | 0.1844 | 0.6268 |
| conda | SNA06_k1_X0Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1912 | 0.1631 | 0.2841 |
| conda | SNA06_k1_X0Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.0634 | -0.12 | 0.01811 |
| conda | SNA06_k1_X0Y1 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.4029 | -0.5532 | 1.946 |
| conda | SNA06_k1_X0Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | -0.1072 | -1.36 | 2.194 |
| conda | SNA06_k1_X0Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.4029 | -0.5532 | 1.946 |
| conda | SNA06_k1_X0Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | -0.1072 | -1.36 | 2.194 |
| conda | SNA06_k1_X0Y1 | primary400 | reference | P3 | FAIL | 2 | 5 | 0.2274 | 0.1443 | 1.331 |
| conda | SNA06_k1_X0Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.0006465 | -0.1124 | 0.09655 |
| conda | SNA06_k1_X0Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -0.2987 | -0.8144 | 2.463 |
| conda | SNA06_k1_X0Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.2009 | -0.6008 | 0.6234 |
| conda | SNA06_k1_X0Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | -0.2987 | -0.8144 | 2.463 |
| conda | SNA06_k1_X0Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.2009 | -0.6008 | 0.6234 |
| conda | SNA06_k1_X0Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.2614 | 0.097 | 0.3956 |
| conda | SNA06_k1_X0Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.02917 | 0.006097 | 0.09094 |
| conda | SNA06_k1_X1Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | -1.446 | -11.4 | 0.264 |
| conda | SNA06_k1_X1Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.31 | 0.07419 | 0.6906 |
| conda | SNA06_k1_X1Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | -1.446 | -11.4 | 0.264 |
| conda | SNA06_k1_X1Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.31 | 0.07419 | 0.6906 |
| conda | SNA06_k1_X1Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.4117 | 0.251 | 0.6054 |
| conda | SNA06_k1_X1Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.05679 | -0.08373 | -0.02902 |
| conda | SNA06_k1_X1Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -4.606 | -5.82 | -2.108 |
| conda | SNA06_k1_X1Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3833 | 0.2646 | 0.5369 |
| conda | SNA06_k1_X1Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -4.606 | -5.82 | -2.108 |
| conda | SNA06_k1_X1Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.3833 | 0.2646 | 0.5369 |
| conda | SNA06_k1_X1Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1603 | 0.09399 | 0.2167 |
| conda | SNA06_k1_X1Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1628 | -0.2119 | -0.08674 |
| conda | SNA06_k1_X1Y1 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | 0.4774 | -0.8187 | 1.646 |
| conda | SNA06_k1_X1Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.1713 | -1.295 | 2.335 |
| conda | SNA06_k1_X1Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.4774 | -0.8187 | 1.646 |
| conda | SNA06_k1_X1Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.1713 | -1.295 | 2.335 |
| conda | SNA06_k1_X1Y1 | primary400 | reference | P3 | FAIL | 2 | 5 | 0.2401 | 0.1508 | 1.202 |
| conda | SNA06_k1_X1Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.0132 | -0.1809 | 0.1087 |
| conda | SNA06_k1_X1Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -0.6423 | -0.9382 | 2.127 |
| conda | SNA06_k1_X1Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3632 | -0.6006 | 0.616 |
| conda | SNA06_k1_X1Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | -0.6423 | -0.9382 | 2.127 |
| conda | SNA06_k1_X1Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.3632 | -0.6006 | 0.616 |
| conda | SNA06_k1_X1Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.2562 | 0.07828 | 0.683 |
| conda | SNA06_k1_X1Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.0223 | -0.03532 | 0.04846 |
| conda | SNA06_k7_X0Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | -46.59 | -263 | -19.4 |
| conda | SNA06_k7_X0Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4774 | 0.4581 | 0.7624 |
| conda | SNA06_k7_X0Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | -46.59 | -263 | -19.4 |
| conda | SNA06_k7_X0Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4774 | 0.4581 | 0.7624 |
| conda | SNA06_k7_X0Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.3672 | 0.2209 | 0.5111 |
| conda | SNA06_k7_X0Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.04685 | -0.07958 | -0.01597 |
| conda | SNA06_k7_X0Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -4.543 | -90.11 | -2.15 |
| conda | SNA06_k7_X0Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.447 | 0.1844 | 0.6268 |
| conda | SNA06_k7_X0Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -4.543 | -90.11 | -2.15 |
| conda | SNA06_k7_X0Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.447 | 0.1844 | 0.6268 |
| conda | SNA06_k7_X0Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1912 | 0.1631 | 0.2841 |
| conda | SNA06_k7_X0Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.0634 | -0.12 | 0.01811 |
| conda | SNA06_k7_X0Y1 | primary400 | actual | A_BALANCE | FAIL | 1 | 5 | 0.6652 | 0.3159 | 1.052 |
| conda | SNA06_k7_X0Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3392 | -0.2131 | 0.8898 |
| conda | SNA06_k7_X0Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.6652 | 0.3159 | 1.052 |
| conda | SNA06_k7_X0Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.3392 | -0.2131 | 0.8898 |
| conda | SNA06_k7_X0Y1 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.5495 | 0.2364 | 0.9228 |
| conda | SNA06_k7_X0Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.0009955 | -0.07483 | 0.02129 |
| conda | SNA06_k7_X0Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | 0.3216 | -0.5634 | 1.472 |
| conda | SNA06_k7_X0Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.3879 | -0.1114 | 0.6251 |
| conda | SNA06_k7_X0Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | 0.3216 | -0.5634 | 1.472 |
| conda | SNA06_k7_X0Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.3879 | -0.1114 | 0.6251 |
| conda | SNA06_k7_X0Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1959 | 0.09181 | 0.4637 |
| conda | SNA06_k7_X0Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.007323 | -0.06868 | 0.008749 |
| conda | SNA06_k7_X1Y0 | primary400 | actual | A_BALANCE | FAIL | 0 | 5 | -3.089 | -6.044 | -1.576 |
| conda | SNA06_k7_X1Y0 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4328 | 0.3582 | 0.4839 |
| conda | SNA06_k7_X1Y0 | primary400 | reference | P1 | FAIL | 0 | 5 | -3.089 | -6.044 | -1.576 |
| conda | SNA06_k7_X1Y0 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.4328 | 0.3582 | 0.4839 |
| conda | SNA06_k7_X1Y0 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.4036 | 0.342 | 0.5561 |
| conda | SNA06_k7_X1Y0 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.001231 | -0.01008 | 0.02509 |
| conda | SNA06_k7_X1Y0 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -3.955 | -9.735 | -3.289 |
| conda | SNA06_k7_X1Y0 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4307 | 0.3451 | 0.688 |
| conda | SNA06_k7_X1Y0 | secondary100 | reference | P1 | FAIL | 0 | 5 | -3.955 | -9.735 | -3.289 |
| conda | SNA06_k7_X1Y0 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.4307 | 0.3451 | 0.688 |
| conda | SNA06_k7_X1Y0 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.1448 | 0.07786 | 0.2581 |
| conda | SNA06_k7_X1Y0 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.08542 | -0.1403 | -0.03014 |
| conda | SNA06_k7_X1Y1 | primary400 | actual | A_BALANCE | FAIL | 1 | 5 | 0.8041 | 0.06959 | 1.189 |
| conda | SNA06_k7_X1Y1 | primary400 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.267 | 0.1629 | 0.745 |
| conda | SNA06_k7_X1Y1 | primary400 | reference | P1 | FAIL | 0 | 5 | 0.8041 | 0.06959 | 1.189 |
| conda | SNA06_k7_X1Y1 | primary400 | reference | P2 | FAIL | 0 | 5 | 0.267 | 0.1629 | 0.745 |
| conda | SNA06_k7_X1Y1 | primary400 | reference | P3 | FAIL | 0 | 5 | 0.6759 | 0.2648 | 1.129 |
| conda | SNA06_k7_X1Y1 | primary400 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.04055 | -0.09548 | -0.02488 |
| conda | SNA06_k7_X1Y1 | secondary100 | actual | A_BALANCE | FAIL | 0 | 5 | -0.3993 | -1.025 | 1.093 |
| conda | SNA06_k7_X1Y1 | secondary100 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.6148 | -0.07583 | 0.8492 |
| conda | SNA06_k7_X1Y1 | secondary100 | reference | P1 | FAIL | 0 | 5 | -0.3993 | -1.025 | 1.093 |
| conda | SNA06_k7_X1Y1 | secondary100 | reference | P2 | FAIL | 0 | 5 | 0.6148 | -0.07583 | 0.8492 |
| conda | SNA06_k7_X1Y1 | secondary100 | reference | P3 | FAIL | 0 | 5 | 0.2575 | 0.1176 | 0.677 |
| conda | SNA06_k7_X1Y1 | secondary100 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.01317 | -0.04824 | 0.0301 |
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
| mnist | SN05_X0Y1 | primary50 | actual | A_BALANCE | PASS | 5 | 5 | 0.9589 | 0.9305 | 1.023 |
| mnist | SN05_X0Y1 | primary50 | actual | A_PLATEAU | PASS | 4 | 5 | 0.1134 | 0.02893 | 0.2499 |
| mnist | SN05_X0Y1 | primary50 | reference | P1 | PASS | 5 | 5 | 0.9589 | 0.9305 | 1.023 |
| mnist | SN05_X0Y1 | primary50 | reference | P2 | PASS | 4 | 5 | 0.1134 | 0.02893 | 0.2499 |
| mnist | SN05_X0Y1 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.1008 | 0.08561 | 0.135 |
| mnist | SN05_X0Y1 | primary50 | reference | P4 | SEPARATED | 4 | 5 | 0.4288 | 0.3876 | 0.4975 |
| mnist | SNA03_X0Y1 | primary50 | actual | A_BALANCE | PASS | 5 | 5 | 0.9475 | 0.9349 | 0.9837 |
| mnist | SNA03_X0Y1 | primary50 | actual | A_PLATEAU | FAIL | 3 | 5 | 0.1593 | 0.05119 | 0.2709 |
| mnist | SNA03_X0Y1 | primary50 | reference | P1 | PASS | 5 | 5 | 0.9475 | 0.9349 | 0.9837 |
| mnist | SNA03_X0Y1 | primary50 | reference | P2 | FAIL | 3 | 5 | 0.1593 | 0.05119 | 0.2709 |
| mnist | SNA03_X0Y1 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.096 | 0.06294 | 0.1327 |
| mnist | SNA03_X0Y1 | primary50 | reference | P4 | NOT_ESTABLISHED | 3 | 5 | 0.459 | 0.2784 | 0.5655 |
| mnist | SNA06_X0Y0 | primary50 | actual | A_BALANCE | FAIL | 0 | 5 | 0.5997 | 0.5047 | 0.6623 |
| mnist | SNA06_X0Y0 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.9627 | 0.6729 | 1.035 |
| mnist | SNA06_X0Y0 | primary50 | reference | P1 | FAIL | 0 | 5 | 0.5997 | 0.5047 | 0.6623 |
| mnist | SNA06_X0Y0 | primary50 | reference | P2 | FAIL | 0 | 5 | 0.9627 | 0.6729 | 1.035 |
| mnist | SNA06_X0Y0 | primary50 | reference | P3 | PASS | 5 | 5 | 0.1627 | 0.07813 | 0.199 |
| mnist | SNA06_X0Y0 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | -0.1672 | -0.2724 | 0.07879 |
| mnist | SNA06_X0Y1 | long150 | actual | A_BALANCE | PASS | 5 | 5 | 0.9429 | 0.9077 | 0.9849 |
| mnist | SNA06_X0Y1 | long150 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.447 | 0.2594 | 0.7335 |
| mnist | SNA06_X0Y1 | long150 | reference | P1 | PASS | 5 | 5 | 0.9429 | 0.9077 | 0.9849 |
| mnist | SNA06_X0Y1 | long150 | reference | P2 | FAIL | 0 | 5 | 0.447 | 0.2594 | 0.7335 |
| mnist | SNA06_X0Y1 | long150 | reference | P3 | FAIL | 0 | 5 | 0.4233 | 0.4144 | 0.4558 |
| mnist | SNA06_X0Y1 | long150 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.2977 | 0.0481 | 0.3884 |
| mnist | SNA06_X0Y1 | primary50 | actual | A_BALANCE | PASS | 5 | 5 | 0.954 | 0.9422 | 0.9579 |
| mnist | SNA06_X0Y1 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.4391 | 0.3076 | 0.4409 |
| mnist | SNA06_X0Y1 | primary50 | reference | P1 | PASS | 5 | 5 | 0.954 | 0.9422 | 0.9579 |
| mnist | SNA06_X0Y1 | primary50 | reference | P2 | FAIL | 0 | 5 | 0.4391 | 0.3076 | 0.4409 |
| mnist | SNA06_X0Y1 | primary50 | reference | P3 | FAIL | 0 | 5 | 0.1954 | 0.1524 | 0.2074 |
| mnist | SNA06_X0Y1 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.251 | 0.2064 | 0.3979 |
| mnist | SNA06_X1Y0 | primary50 | actual | A_BALANCE | FAIL | 1 | 5 | 0.8678 | 0.8563 | 0.9124 |
| mnist | SNA06_X1Y0 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 1.271 | 1.004 | 1.302 |
| mnist | SNA06_X1Y0 | primary50 | reference | P1 | FAIL | 0 | 5 | 0.4925 | 0.3143 | 0.5337 |
| mnist | SNA06_X1Y0 | primary50 | reference | P2 | FAIL | 0 | 5 | 0.8478 | 0.6692 | 1.07 |
| mnist | SNA06_X1Y0 | primary50 | reference | P3 | PASS | 5 | 5 | 0.05189 | 0.01982 | 0.08261 |
| mnist | SNA06_X1Y0 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.03816 | -0.175 | 0.2426 |
| mnist | SNA06_X1Y1 | primary50 | actual | A_BALANCE | FAIL | 2 | 5 | 0.8968 | 0.8815 | 0.9299 |
| mnist | SNA06_X1Y1 | primary50 | actual | A_PLATEAU | FAIL | 0 | 5 | 0.9504 | 0.7318 | 1.207 |
| mnist | SNA06_X1Y1 | primary50 | reference | P1 | FAIL | 0 | 5 | 0.491 | 0.4013 | 0.5822 |
| mnist | SNA06_X1Y1 | primary50 | reference | P2 | FAIL | 0 | 5 | 0.8752 | 0.6461 | 0.9962 |
| mnist | SNA06_X1Y1 | primary50 | reference | P3 | PASS | 5 | 5 | 0.07358 | 0.0463 | 0.1084 |
| mnist | SNA06_X1Y1 | primary50 | reference | P4 | NOT_ESTABLISHED | 0 | 5 | 0.0162 | -0.1337 | 0.186 |

## Paired comparisons

All registered X/Y, k, activation, Snake, and horizon contrasts appear seed by seed in `paired_seed.csv`, with 5-seed median and bootstrap95% intervals in `paired_group.csv`. `group_stats.csv` gives the same intervals for every numeric seed summary field. Every listed field is reported, including null contrasts; no outcome field was selected after viewing results.

## Traceability

`source_paths.csv` records snapshot and input paths, `source_windows.csv` records fit/held-out/late windows, and `spec_hashes.csv` records the exact preregistration hashes. `transitions.csv.gz` contains per-task raw/reference/actual values and the paired decomposition; `transitions_compression.json` records its round-trip SHA256 check. `unit_late.csv` holds late per-unit aggregates. `closure.csv` contains fixed-reference held-out trajectories only.

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
| conda | SNA06_k1_X0Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.006039 | 0.005828 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002039 | 0.002047 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002226 | 0.002337 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001345 | 0.001352 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003364 | 0.002818 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003015 | 0.002825 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001784 | 0.002529 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002153 | 0.002154 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003287 | 0.003973 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002973 | 0.002946 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.471 | 0.006483 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.643 | 0.01304 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 2.498 | 0.01376 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.205 | 0.01462 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.308 | 0.01042 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.336 | 0.01042 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 2.806 | 0.0179 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.248 | 0.01417 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 1.747 | 0.01047 | nan | nan | nan | nan |
| conda | SNA06_k1_X0Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.893 | 0.01039 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1662 | 0.003513 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.06684 | 0.00377 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.4273 | 0.00148 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1232 | 0.002127 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.4162 | 0.005489 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1384 | 0.002686 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.09007 | 0.001051 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.02448 | 0.001219 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1246 | 0.001608 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.01273 | 0.002314 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.605 | 0.01174 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.692 | 0.01883 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 2.717 | 0.01054 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.401 | 0.01306 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.498 | 0.01605 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.32 | 0.02348 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 2.742 | 0.01357 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 2.184 | 0.01336 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 1.831 | 0.01119 | nan | nan | nan | nan |
| conda | SNA06_k1_X1Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 2.996 | 0.01176 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.006039 | 0.005828 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002039 | 0.002047 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002226 | 0.002337 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001345 | 0.001352 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003364 | 0.002818 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003015 | 0.002825 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.001784 | 0.002529 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002153 | 0.002154 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.003287 | 0.003973 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.002973 | 0.002946 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.231 | 0.01214 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.512 | 0.01322 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 4.258 | 0.01281 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 5.718 | 0.01196 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 7.135 | 0.007475 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.779 | 0.01337 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 7.085 | 0.02593 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.445 | 0.01859 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.284 | 0.01616 | nan | nan | nan | nan |
| conda | SNA06_k7_X0Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.801 | 0.01247 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.4477 | 0.003005 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1013 | 0.002061 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 1.13 | 0.002004 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1653 | 0.002031 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.4133 | 0.008297 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.05953 | 0.00317 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.07674 | 0.001312 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.008269 | 0.001144 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.1614 | 0.001243 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y0 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 0.02009 | 0.002196 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | secondary100 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 4.127 | 0.007661 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | primary400 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.654 | 0.01407 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | secondary100 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 3.69 | 0.01022 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | primary400 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | OTHER_ACTIVE | 5.252 | 0.01566 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | secondary100 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 4.959 | 0.0264 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | primary400 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 4.929 | 0.01107 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | secondary100 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 7.193 | 0.01325 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | primary400 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.427 | 0.01319 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | secondary100 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | STOPPED | 6.868 | 0.01521 | nan | nan | nan | nan |
| conda | SNA06_k7_X1Y1 | primary400 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | 5.329 | 0.009386 | nan | nan | nan | nan |
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
| mnist | SN05_X0Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.1013 | 1 | 14 | 0.0004646 |
| mnist | SN05_X0Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.0985 | 1 | 12.91 | 0.0006269 |
| mnist | SN05_X0Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.09842 | 1 | 12.33 | 0.0006632 |
| mnist | SN05_X0Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.09467 | 1 | 13.09 | 0.0005103 |
| mnist | SN05_X0Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.1078 | 1 | 12.97 | 0.000431 |
| mnist | SNA03_X0Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.1013 | 1 | 12.71 | 0.0004689 |
| mnist | SNA03_X0Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.0985 | 1 | 12.75 | 0.0007021 |
| mnist | SNA03_X0Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | PLATEAU_WITH_ACTIVITY | nan | nan | 0.09842 | 1 | 11.63 | 0.0005716 |
| mnist | SNA03_X0Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09467 | 1 | 12.7 | 0.0005358 |
| mnist | SNA03_X0Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1078 | 1 | 13.26 | 0.0006084 |
| mnist | SNA06_X0Y0 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 0.0001854 | 0.0001847 |
| mnist | SNA06_X0Y0 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 2.454e-05 | 2.475e-05 |
| mnist | SNA06_X0Y0 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.9942 | 1 | 0.01321 | 3.89e-05 |
| mnist | SNA06_X0Y0 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 3.009e-05 | 3.102e-05 |
| mnist | SNA06_X0Y0 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 1 | 1 | 4.283e-05 | 4.253e-05 |
| mnist | SNA06_X0Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1013 | 1 | 21.46 | 0.0007381 |
| mnist | SNA06_X0Y1 | long150 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09822 | 1 | 19.2 | 0.0004163 |
| mnist | SNA06_X0Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.0985 | 1 | 19.72 | 0.0007153 |
| mnist | SNA06_X0Y1 | long150 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1007 | 1 | 18.29 | 0.0004914 |
| mnist | SNA06_X0Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09842 | 1 | 18.1 | 0.0007844 |
| mnist | SNA06_X0Y1 | long150 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09631 | 1 | 17.86 | 0.0005594 |
| mnist | SNA06_X0Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09467 | 1 | 19.32 | 0.0005541 |
| mnist | SNA06_X0Y1 | long150 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09925 | 1 | 18.34 | 0.0004416 |
| mnist | SNA06_X0Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.1078 | 1 | 21.45 | 0.0006184 |
| mnist | SNA06_X0Y1 | long150 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | NEAR_BALANCE_WITH_ACTIVITY | nan | nan | 0.09947 | 1 | 19.79 | 0.0005078 |
| mnist | SNA06_X1Y0 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1012 | 1 | 30.3 | 0.0007085 |
| mnist | SNA06_X1Y0 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1037 | 1 | 32.1 | 0.001385 |
| mnist | SNA06_X1Y0 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.096 | 1 | 28.22 | 0.001637 |
| mnist | SNA06_X1Y0 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09533 | 1 | 33.51 | 0.00137 |
| mnist | SNA06_X1Y0 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09958 | 1 | 38.12 | 0.001098 |
| mnist | SNA06_X1Y1 | primary50 | 200 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09883 | 1 | 30.94 | 0.001312 |
| mnist | SNA06_X1Y1 | primary50 | 201 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1021 | 1 | 31.15 | 0.003719 |
| mnist | SNA06_X1Y1 | primary50 | 202 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.09667 | 1 | 27.14 | 0.00125 |
| mnist | SNA06_X1Y1 | primary50 | 203 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.097 | 1 | 29.09 | 0.0007483 |
| mnist | SNA06_X1Y1 | primary50 | 204 | NO_LOW_RESPONSE | nan | NO_LOW_RESPONSE | TRANSIENT_GROWTH | nan | nan | 0.1003 | 1 | 37.98 | 0.001241 |
