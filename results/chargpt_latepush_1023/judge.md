# 判定（登録 spec §4・§8）

| arm | layer | J1 | n J1 | J2 | n residual | n share↓ | n share evaluable | J3 | n J3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | L0 | NOT_LOADING | 0 | PARTIAL | 9 | 2 | 2 | NOT_TRACKING_COS2 | 0 |
| C3 | L1 | NOT_LOADING | 0 | PARTIAL | 8 | 0 | 0 | NOT_TRACKING_COS2 | 5 |
| ReLUcap | L0 | NOT_LOADING | 0 | RESIDUAL_DILUTES | 10 | 10 | 10 | NOT_TRACKING_COS2 | 0 |
| ReLUcap | L1 | NOT_LOADING | 0 | NOT_RESIDUAL_DILUTES | 5 | 0 | 0 | TRACKS_COS2 | 8 |

主判定は C3 L1。各条件は10 seed 中8以上。L0・ReLUcap は記録。J3 は課題末35点、init 込み36点は補助。
