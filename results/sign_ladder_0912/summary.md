=== 進捗ゲート ===
  G0 参照の損失 >= 2.0 pt : 7.74 7.47 7.83  -> PASS
  G1 committed 参照と bit 一致 : maxabs=0.0 比較2400/2400 件 -> PASS   (results/long_horizon_acts_0910/GELU_none_s0_units.npz)
  G2 新腕のゲートが一度も負にならない : GELUF=0 GELUA=0  -> PASS
  G3 谷を越えたユニット >= 20 : GELU [83, 84, 84]  GELUF [72, 76, 78]  GELUA [72, 71, 76]  -> PASS
  G4 GELUA の可動性が GELU+0.10 以上 : GELU 0.406(n45k)  GELUF 0.675(n33k)  GELUA 0.843(n27k)  -> PASS

=== 主判定 ===
arm         L(s0)     L(s1)     L(s2)       平均                   ΔL 対 GELU
GELU         7.74      7.47      7.83     7.68
GELUF        4.93      4.53      4.64     4.70   +2.81 +2.94 +3.19  (平均 +2.98)
GELUA        4.81      4.52      4.54     4.62   +2.93 +2.95 +3.29  (平均 +3.06)

  GELUF=reduces  GELUA=reduces  ->  **VALLEY_CAUSAL**
  記名: Claude VALLEY_IRRELEVANT / Issa VALLEY_CAUSAL  -> Issa 的中

=== 副次（ラベルにしない）===
arm       z̄(t400)  sd(t400)    cnorm  past_zc  acc late
GELU         -6.89      3.31     9.13     83.7     84.72
GELUF        -4.47      4.15    13.24     75.3     87.95
GELUA        -3.67      4.25    14.25     73.0     88.12

  参考（0910 の committed L_ref 平均）: ReLU 4.78 / leaky 4.36 / ELU1 4.13 / SiLU 6.93

=== 水準か劣化か（acc %・3 seed 中央値）===
arm    t20     t50     t100    t200    t300    t400        base    late      L
GELU      92.63   90.86   89.31   86.21   85.13   85.51   92.37   84.76   7.61
GELUF     92.68   91.52   90.20   88.74   87.52   87.93   92.68   88.01   4.67
GELUA     92.74   91.91   90.73   89.68   88.09   87.83   92.74   88.09   4.65

base(t16-20) は 3 腕で 92.37/92.68/92.74 と同じ。差は水準ではなく劣化。
