# bias_margin_0814 Phase 1: レジーム探索 (spec §3)

µ=0 (c=0, κ=1) を厳密に保った条件B。width [5, 20], target_hidden 100, lr [0.01], seeds [0, 1], 200000 step。

経路1 = target_noise_sd スイープ (K=1e4)、経路2 = K 短縮 (noise=0)。

## セル別 (seed 平均、dead_end 降順)

 route         tag  noise_sd     K  width  dead_end  dead_max  eval_ratio  eval_end  b_mean_end  b_std_end  b_min_end  beta_mean_end  p_min_end  finite  n  c1_finite  c2_dead  c3_lop
route1   n2_K10000       2.0 10000     20     0.850     0.925      0.3388    0.6327     -2.7706     0.5499    -4.1996        -1.0216     0.0000    True  2       True     True   False
route2     n0_K100       0.0   100     20     0.725     0.775      0.3477    0.5745     -1.0775     0.1179    -1.3313        -1.3362     0.0008    True  2       True     True   False
route1   n2_K10000       2.0 10000      5     0.700     1.000      0.2228    0.5112     -2.2572     0.8427    -4.0692        -0.8525     0.0000    True  2       True     True   False
route2     n0_K100       0.0   100      5     0.500     0.600      0.2486    0.5595     -0.2849     0.7391    -1.1156        -0.4387     0.0145    True  2       True     True   False
route1   n1_K10000       1.0 10000     20     0.300     0.375      0.1283    0.2165     -1.2754     0.3374    -1.9807        -0.9605     0.0055    True  2       True     True   False
route1   n0_K10000       0.0 10000      5     0.000     0.000      0.0678    0.1393      0.0471     0.7354    -0.4359         0.0220     0.3672    True  2       True    False   False
route1 n0.5_K10000       0.5 10000     20     0.000     0.000      0.0706    0.1173     -0.7051     0.2265    -1.0523        -0.6787     0.0928    True  2       True    False   False
route1 n0.5_K10000       0.5 10000      5     0.000     0.000      0.0764    0.1729     -0.3450     0.3842    -0.8355        -0.2557     0.2602    True  2       True    False   False
route1   n0_K10000       0.0 10000     20     0.000     0.000      0.0362    0.0599     -0.2442     0.1400    -0.4586        -0.2465     0.3052    True  2       True    False   False
route1   n1_K10000       1.0 10000      5     0.000     0.100      0.0912    0.2020     -0.8306     0.4273    -1.2857        -0.5476     0.1550    True  2       True    False   False
route2    n0_K1000       0.0  1000      5     0.000     0.000      0.0881    0.1689      0.6137     0.3201     0.1505         0.6411     0.5630    True  2       True    False   False
route2    n0_K1000       0.0  1000     20     0.000     0.000      0.1081    0.1721     -0.1035     0.2494    -0.6367        -0.1332     0.1852    True  2       True    False   False


- 基準1 (発散なし): 12/12
- 基準2 (dead_frac ≥ 0.1): 5/12
- 基準3 (eval_loss ≥ 1.5倍): 0/12

## 採用セル

 route       tag noise_sd     K width dead_end dead_max eval_ratio  eval_end b_mean_end b_std_end b_min_end beta_mean_end p_min_end finite n c1_finite c2_dead c3_lop
route1 n2_K10000      2.0 10000    20     0.85    0.925   0.338825  0.632653  -2.770605   0.54992  -4.19964     -1.021577       0.0   True 2      True    True  False

注: 基準3 (LoP 症状) を満たすセルが無いため、基準2 までを満たすセルのうち dead_frac 最大を採用した (仕様に規定の無いケース。summary に明記)。

- status: OK_NO_LOP
