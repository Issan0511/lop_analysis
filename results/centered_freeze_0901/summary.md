# centered_freeze_0901 — P-1 result

## Verdict

| pred | scope | verdict | evidence |
| --- | --- | --- | --- |
| P1 | late tasks 451-500 strict_dead_frac | BIAS_ROUTE_DECISIVE | free 0.463940 -> frozen 0.000000; reduction 1.000000; frozen-free -0.463940 CI [-0.514620, -0.410039] |
| P1-final | step 5M strict_dead_frac (supportive) | REPORT_ONLY | free 0.469000 -> frozen 0.000000; frozen-free -0.469000 CI [-0.515000, -0.420000] |
| P2 | late tasks 451-500 exact-support unfit | FROZEN_WORSE | free 0.0039701 -> frozen 0.228112; frozen-free +0.224142 CI [+0.205488, +0.2455] |

## Paired seed endpoints

| seed | dead_free_late | dead_frozen_late | dead_free_final | dead_frozen_final | unfit_free_late | unfit_frozen_late | unfit_free_final | unfit_frozen_final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.528 | 0 | 0.5 | 0 | 0.0042487711 | 0.22880107 | 0.0019782709 | 0.13891178 |
| 1 | 0.5826 | 0 | 0.55 | 0 | 0.0046935906 | 0.22218051 | 0.0045520333 | 0.092080291 |
| 2 | 0.3384 | 0 | 0.36 | 0 | 0.0037562029 | 0.27444706 | 0.0035576889 | 0.23883293 |
| 3 | 0.3084 | 0 | 0.31 | 0 | 0.0037517142 | 0.24595217 | 0.0068698925 | 0.45350281 |
| 4 | 0.548 | 0 | 0.56 | 0 | 0.0046120661 | 0.21842148 | 0.0018509283 | 0.40068198 |
| 5 | 0.4096 | 0 | 0.45 | 0 | 0.0038357073 | 0.19188459 | 0.0018002071 | 0.23092496 |
| 6 | 0.5266 | 0 | 0.54 | 0 | 0.0034968223 | 0.19643385 | 0.012555706 | 0.29349857 |
| 7 | 0.4742 | 0 | 0.5 | 0 | 0.0040841473 | 0.2947733 | 0.006344319 | 0.23358193 |
| 8 | 0.4732 | 0 | 0.5 | 0 | 0.0037411616 | 0.21027413 | 0.0052940118 | 0.13092694 |
| 9 | 0.4504 | 0 | 0.42 | 0 | 0.0034808504 | 0.19794934 | 0.0016276675 | 0.16116833 |

## Interpretation

- condA・centered の終盤死は、b を 0 に凍結すると事前登録した「ほぼ消失」域まで減った。**この設定では b 経路が決定的**である。
- 機能の副次判定は **FROZEN_WORSE**。ただし freeze_bias は表現力も変えるため、dead の機能コストを単独では同定しない。
- スコープは condA・1層幅100・center_alpha=0.01・T=10,000・batch=1・plain SGD・5M step に限定する。

## Sanity

- S0: 30k free replay は既存 L1w100_A1 と一致。
- S2: frozen の全記録点で b は厳密に 0。
- S3: 全支持点記録の恒等式・1/32 量子化・有限性を通過。
