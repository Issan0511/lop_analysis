# rank_int_0814 Phase 0 (spec §2): t_int 妥当性・回復/予防ラベル・srank_target

入力: results/coupling_fbw_0813 の A_w10_bfull / A_w20_bfull (seed 0–4)。

**仕様逸脱 (要先生確認)**: 離陸時刻の字義定義 (初期値と 1M 値の半値) は、full-batch アームでは v0 >> v_1M (初期損失が高くほぼ 0 まで降下後に再上昇) のため閾値が初期値を下回り、全 seed が step=0 で自明に「離陸」して退化する (evalloss_takeoff 列)。ラベルには意図 (低平坦部からの離陸) に沿うロバスト定義 「argmin 以降で min + 0.5*(v_1M − min) を最初に上抜く step」(evalloss_takeoff_robust 列) を用いた。label_spec は字義定義によるラベル。

 seed  width  srank_t50  srank_t50_passed_150k  srank_t50_passed_300k  dead_at_tint  dead_at_tint_ok  dead_at_300k  dead_at_300k_ok  evalloss_takeoff  evalloss_takeoff_robust label_spec label  srank_target  evalloss_1M
    0     10     9000.0                   True                   True          0.10             True          0.10             True               0.0                 161000.0         回復    予防       3.28474     0.559547
    1     10    12700.0                   True                   True          0.10             True          0.10             True               0.0                 131000.0         回復    回復       3.55285     0.341673
    2     10   110700.0                   True                   True          0.30            False          0.40            False               0.0                 121000.0         回復    回復       4.48014     0.315224
    3     10    19000.0                   True                   True          0.20            False          0.20            False               0.0                 471000.0         回復    予防       3.41733     0.444917
    4     10    31800.0                   True                   True          0.40            False          0.50            False               0.0                 141000.0         回復    回復       4.20347     0.443268
    0     20    17000.0                   True                   True          0.35            False          0.10             True               0.0                 411000.0         回復    予防       5.47947     0.705332
    1     20    14000.0                   True                   True          0.55            False          0.35            False               0.0                 821000.0         回復    予防       5.47238     0.094605
    2     20    96000.0                   True                   True          0.35            False          0.10             True               0.0                 401000.0         回復    予防       4.69754     0.258451
    3     20     8000.0                   True                   True          0.45            False          0.45            False               0.0                 171000.0         回復    予防       5.59142     0.089874
    4     20    12400.0                   True                   True          0.25            False          0.05             True               0.0                 231000.0         回復    予防       6.00119     0.008219


- t_int=150k で (a) srank t50 通過済み ∧ (b) dead ≤ 0.15: 2/10 seed が適格 (不適格 seed も除外せず層別報告)
- 不適格 seed: w10/s2(srank_t50=110700, dead=0.300), w10/s3(srank_t50=19000, dead=0.200), w10/s4(srank_t50=31800, dead=0.400), w20/s0(srank_t50=17000, dead=0.350), w20/s1(srank_t50=14000, dead=0.550), w20/s2(srank_t50=96000, dead=0.350), w20/s3(srank_t50=8000, dead=0.450), w20/s4(srank_t50=12400, dead=0.250)
- ラベル分布: {(10, '予防'): 2, (10, '回復'): 3, (20, '予防'): 5}
- srank_target (step0 stable_rank_W_alive): {10: 3.788, 20: 5.448} (幅別平均)
