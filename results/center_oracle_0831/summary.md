# center_oracle_0831

- P1: **BOTH_CONTRIBUTE**
- P2: **ORACLE_INCREASES_DEATH**

追補1の真の切替499点版を主解析とし、初期過渡を含む500点版はREPORT_ONLY。
R_499 = 0.5167 [0.4858, 0.5805]
Aexact Δβ_boundary_499 = -2.062 [-2.158, -2.012]
A1 Δβ_boundary_499 = -4.199 [-4.271, -3.447]
R_500 (REPORT_ONLY) = 0.4897 [0.4572, 0.5518] → BOTH_CONTRIBUTE
Aexact Δβ_internal_4500 = -0.01198 [-0.05731, 0.02731]
A1 Δβ_internal_4500 = 1.933 [1.79, 2.19]
startup 0→1000: Aexact -0.02814 / A1 -0.2787
strict_dead_frac: Aexact 1 / A1 0.5; gap 0.5 [0.46, 0.595]
continuous-dead fraction among final dead: Aexact 1 / A1 0.6248

## 必須の交絡

**オラクル中心化は µ を消すと同時に、タスク可識別性を完全に消す。** Aexact−A1 の差には「EMA遅れの差」と「可識別性の差」が同居する。書いてよいのは境界降下がEMA遅れ窓に起因する／しないという範囲であり、「µの効果を測った」「centeringを改善すればLoPを防げる」とは書かない。

## S0 amendment

元specの『step 0全記録量bit一致』はS-tautの `M≡0` と両立しないため、実行前amendmentに従い、介入前state・raw stream・final envを一致対象とした。介入後のstep 0相違列はprovenanceに列挙する。

実行時間: 378.6 sec。alpha sweepは未実施。
