**完了した readout_separation_1111 の独立解釈**

対象は全280 continuation（ELU 180、ViT 100）、全15 sourceのfrozen assay。主な違いは、**ELUでは試したS/P/Oのどのreadout方向でも縮小の大きな長期改善が残り、ViTではS/Oに残る縮小効果がP後には弱まる**こと。ただしこれは方向・尺度・optimizerを指定した介入応答の違いであり、「ELUは純粋な大きさ、ViTは純粋な整列」という微視的機構の二分ではない。

出典は [contrasts.csv](contrasts.csv) の一意キー `(model, metric, kind, contrast)`、[seed_metrics.csv](seed_metrics.csv) の `(model, seed, arm_id)`、[summary.md](summary.md)。以下の表は既存CSVを転記したもので新しい検定は追加していない。特記しない表の `metric=U`。数値は **U差×100、pp、対応付きseed bootstrapの記述的95% CI**。ELUは10 seeds・4000更新、ViTは5 seeds・1500更新で、両者をプールした比較はしない。Uは固定訓練集合の期間平均accuracy欠損であり、test精度や総所要時間ではない。CIは多重比較補正なし。0を跨ぐ差を同等性と呼ばず、登録practical thresholdの1 ppとも区別する。全continuationでhead Adamをclearしており、保存head optimizerを持つ通常継続へ無条件に一般化しない。

**1. P/O × scale：ELUの方向に頑健な尺度依存と、ViTのPに依存する尺度効果。** scaleは `U(c1)−U(c0.1)`、正なら縮小が改善。directionは `U(S)−U(P/O)`、正ならP/Oが改善。interaction `scale_by_P/O` はSの縮小効果−P/Oの縮小効果。class-centered Wを0.1倍しfull-set平均logitsをbias補償する操作なので、raw W全体の単純倍率やconfidenceだけの操作ではない。

| kind / contrast | ELU (`model=mlp`) | ViT (`model=vit`) |
| --- | ---: | ---: |
| `scale / S_keep` | +19.461 [+18.629, +20.646] | +6.797 [+2.149, +11.466] |
| `scale / P_keep` | +19.479 [+18.354, +20.786] | -1.424 [-3.753, +1.021] |
| `scale / O_keep` | +19.799 [+18.991, +20.590] | +7.283 [+0.461, +14.140] |
| `scale / S_clear` | +29.973 [+29.215, +31.027] | +15.344 [+12.075, +19.343] |
| `scale / P_clear` | +29.361 [+28.253, +30.647] | +0.884 [-0.156, +2.047] |
| `direction / S-P_c1_keep` | -0.764 [-1.778, +0.195] | +9.178 [+5.292, +14.442] |
| `direction / S-P_c0.1_keep` | -0.746 [-1.376, -0.110] | +0.957 [-1.885, +4.169] |
| `direction / S-P_c1_clear` | -0.014 [-1.087, +0.985] | +15.954 [+12.854, +20.151] |
| `direction / S-P_c0.1_clear` | -0.627 [-1.239, -0.075] | +1.494 [+0.162, +2.831] |
| `direction / S-O_c1_keep` | -0.116 [-1.091, +1.064] | -0.028 [-6.562, +3.750] |
| `direction / S-O_c0.1_keep` | +0.222 [-0.373, +0.847] | +0.457 [-4.894, +5.147] |
| `interaction / scale_by_P_keep` | -0.018 [-1.033, +0.962] | +8.222 [+2.707, +14.460] |
| `interaction / scale_by_P_clear` | +0.612 [-0.352, +1.618] | +14.460 [+10.063, +19.090] |
| `interaction / scale_by_O_keep` | -0.338 [-1.211, +0.613] | -0.486 [-11.311, +7.155] |

ELUではP/O後にも約19–20 ppの縮小効果が残り、P/Oによる効果変更のCIは0を跨ぐ。これは「試した方向で尺度介入の利益が頑健」という結果であって、方向効果が厳密にゼロ、整列が無関係、あるいはnormだけが原因という結論ではない。ViTではPのscale1改善とscale-by-P interactionがbody keep/clearの両方で支持されるため、保存body AdamだけでP効果を説明することもできない。一方Oでは縮小利益が残り、S−Oの差は未解決。Pは列別class置換で列間Gramも変え、Oはclass直交回転でWᵀWを保つため、「旧classとの対応を乱す操作」一般ではP効果を説明できない。Pのどの幾何成分が必要十分かは未同定。continuationのPはbackward mapだけでなくforward確率・bias・以後のhead適応も変えるため、局所のPによるtrunk norm低下が長期改善を媒介したという経路も、このfactorialだけでは確定しない。ViT P_keepの縮小効果−1.424 ppはCIが0を跨ぎ、縮小が有害／利益が厳密に消失したと断定しない。

**2. acuteと全期間：初期の改善と長期改善を同一視できない。** acuteは200pre、lateは200postを使う別の時間窓であり、因果媒介成分ではない。primary Uも200postを使うため、releaseで予測が変わる腕ではacute/lateの期間加重和と厳密には一致しない。

| model / kind / contrast | U（全期間） | U_acute（0–200） | U_late（200–終了） |
| --- | ---: | ---: | ---: |
| `mlp / scale / S_keep` | +19.461 [+18.629, +20.646] | -0.564 [-0.977, -0.164] | +20.515 [+19.659, +21.745] |
| `mlp / scale / P_keep` | +19.479 [+18.354, +20.786] | -0.354 [-0.636, -0.115] | +20.523 [+19.340, +21.896] |
| `mlp / scale / O_keep` | +19.799 [+18.991, +20.590] | -0.636 [-1.021, -0.213] | +20.875 [+20.027, +21.705] |
| `mlp / scale / S_clear` | +29.973 [+29.215, +31.027] | +2.632 [+2.252, +2.975] | +31.412 [+30.622, +32.512] |
| `vit / scale / S_keep` | +6.797 [+2.149, +11.466] | +0.474 [-1.762, +2.711] | +7.770 [+2.553, +12.917] |
| `vit / scale / P_keep` | -1.424 [-3.753, +1.021] | -2.352 [-4.449, -0.743] | -1.282 [-3.709, +1.457] |
| `vit / scale / O_keep` | +7.283 [+0.461, +14.140] | -1.018 [-3.377, +1.309] | +8.560 [+0.936, +16.294] |
| `vit / direction / S-P_c1_keep` | +9.178 [+5.292, +14.442] | +0.714 [-2.743, +4.171] | +10.481 [+6.528, +16.061] |

ELUのbody keepでは縮小が最初の200更新の平均精度を小さく悪化させながら、以後に大きく改善する。body clearならS縮小はacuteから改善する。したがって「縮小は最初から学習を速める」「最初の局所gradientが小さければ即座に高精度になる」という説明は不適切。ViTのP改善も主にlateで明瞭であり、acuteのS−P CIは0を跨ぐ。ViT P後の縮小はacuteで悪化が支持されるが、全期間では未解決となる。

**3. T/G/TG：前向きと後向きは分離して操作できるが、足し算の原因割合にはならない。** Tは最初の200更新だけ `zbar0+0.1(z−zbar0)`、head勾配は倍率1のstraight-through。Gはbodyへの勾配倍率0.1、headへの倍率1。TGは両方。基準Nはphysical c1/body keep。forwardのgain1はN−T、gain0.1はG−TG。body_gainのcf1はN−G、cf0.1はT−TG。interactionはN−T−G+TG。正はその条件でcf/gを下げる利益。

| kind / contrast | ELU (`model=mlp`) | ViT (`model=vit`) |
| --- | ---: | ---: |
| `forward / S_gain1` | +10.120 [+9.455, +10.789] | +1.535 [-5.558, +7.315] |
| `body_gain / S_cf1` | +2.372 [+2.076, +2.698] | +7.977 [+6.274, +9.491] |
| `forward / S_gain0.1` | +0.920 [+0.725, +1.127] | -9.275 [-12.647, -5.032] |
| `body_gain / S_cf0.1` | -6.828 [-7.230, -6.380] | -2.834 [-6.264, +1.574] |
| `interaction / forward_by_gain_S` | +9.200 [+8.610, +9.808] | +10.810 [+6.113, +14.282] |
| `forward / P_gain1` | +8.850 [+8.479, +9.231] | -21.001 [-28.412, -12.109] |
| `body_gain / P_cf1` | +2.349 [+2.035, +2.599] | -2.440 [-4.714, -0.217] |
| `forward / P_gain0.1` | +0.411 [+0.229, +0.597] | -22.464 [-33.156, -13.488] |
| `body_gain / P_cf0.1` | -6.089 [-6.462, -5.725] | -3.903 [-10.642, +2.837] |
| `interaction / forward_by_gain_P` | +8.439 [+8.060, +8.838] | +1.462 [-3.991, +8.522] |

ELUではS/PのT単独とG単独がともに全期間改善するが、TにGを追加すると悪化する。ViTではSのGは改善する一方、SのT単独は未解決、PのTは大きく悪化しGも小さく悪化する。したがって「confidenceを下げれば両モデルで害が減る」「body勾配を弱めれば常に改善する」とはいえない。Tはclass-specific zbar0へ寄せるのでargmaxも変わりうる。通常の温度操作やconfidenceだけの介入と呼ばない。Gで確率/head勾配が一致するのは共有状態での演算子比較であり、更新後の軌道で同じpが続く保証はない。200以後は全てnativeへ戻るので、full差はこの200更新方策とその後の適応の総効果である。

時間窓も異なる。

| model / kind / contrast | U（全期間） | U_acute（0–200） | U_late（200–終了） |
| --- | ---: | ---: | ---: |
| `mlp / forward / S_gain1` | +10.120 [+9.455, +10.789] | +2.066 [+1.647, +2.548] | +10.544 [+9.848, +11.257] |
| `mlp / body_gain / S_cf1` | +2.372 [+2.076, +2.698] | -2.763 [-2.974, -2.546] | +2.642 [+2.331, +2.982] |
| `vit / forward / S_gain1` | +1.535 [-5.558, +7.315] | -3.777 [-6.544, -0.829] | +2.352 [-5.933, +8.969] |
| `vit / body_gain / S_cf1` | +7.977 [+6.274, +9.491] | -0.353 [-1.183, +0.333] | +9.258 [+7.246, +11.038] |
| `vit / forward / P_gain1` | -21.001 [-28.412, -12.109] | -7.695 [-10.679, -4.711] | -23.049 [-32.006, -12.335] |
| `vit / body_gain / P_cf1` | -2.440 [-4.714, -0.217] | -4.103 [-5.354, -2.851] | -2.185 [-4.635, +0.218] |

ELU Gはacuteで悪化してもfullで改善し、ViT SのGはacuteで未解決でもlateで改善する。これらから初期の勾配低下を、そのまま初期精度改善や保護されたbodyの量へ換算することはできない。ViT PのT悪化はrelease前からあり、全てを解除時の不連続だけへ帰属する説明も足りない。

**4. physical0.1とTG：同じ初期p/勾配でも、学習する関数は同じ軌道を進まない。** 以下はphysical0.1−TGで、負ならphysicalが優位。

| model / kind / contrast | U（全期間） | U_acute（0–200） | U_late（200–終了） |
| --- | ---: | ---: | ---: |
| `mlp / parameterization / physical01_minus_TG_S` | -16.169 [-17.144, -15.340] | -1.965 [-2.440, -1.509] | -16.917 [-17.925, -16.053] |
| `mlp / parameterization / physical01_minus_TG_P` | -16.718 [-17.902, -15.684] | -2.193 [-2.623, -1.740] | -17.483 [-18.718, -16.394] |
| `vit / parameterization / physical01_minus_TG_S` | -8.096 [-10.983, -6.034] | -4.550 [-5.720, -3.258] | -8.642 [-11.901, -6.414] |
| `vit / parameterization / physical01_minus_TG_P` | -23.479 [-32.413, -16.103] | -5.807 [-9.258, -2.825] | -26.199 [-36.543, -17.225] |

全てでphysicalがrelease前から優位。初期body/head勾配の一致だけでは、その後の学習性能の一致を導かない。head Adamがclearされ最初の同じhead座標更新ΔW,Δbが得られても、そのhead更新によるlogit応答はphysicalで1倍、TGで0.1倍となる。exact arithmeticでは共通body更新後h′でも、class-centering Cを用いて `C(z_phys,new−z_TG,new)=0.9 C(ΔW h′+Δb)`。これは初回関数更新の具体的な差であり、全200更新のhead学習速度が一定10倍という主張ではない。その後はpも勾配も分岐する。さらにTGは200でforward/gainを解除し、physicalは同じ解除を持たないため、全期間の差には200以後の方策差・適応も含む。head応答倍率への全差の媒介帰属は未同定。

**5. native Jacobian：両方とも、今回の変更に対しては保護的な働きがある。** transmissionはaltered−nativeなので、正が悪化。forwardは共有状態でnativeと同じ。ViT final gamma/beta勾配も共有状態では保存する。

| model / kind / contrast | U（全期間） | U_acute（0–200） | U_late（200–終了） |
| --- | ---: | ---: | ---: |
| `mlp / transmission / elu_unity_c1` | +11.302 [+10.827, +11.831] | +1.449 [+1.230, +1.717] | +11.820 [+11.317, +12.378] |
| `mlp / transmission / elu_unity_c0.1` | +2.148 [+1.946, +2.367] | -0.180 [-0.345, +0.027] | +2.271 [+2.059, +2.500] |
| `vit / transmission / ln_remove_radial_c1` | +10.853 [+8.826, +12.880] | +1.623 [+0.804, +2.546] | +12.273 [+10.003, +14.543] |
| `vit / transmission / ln_remove_radial_c0.1` | +4.355 [-2.035, +12.355] | +3.294 [+1.400, +4.683] | +4.518 [-2.597, +13.667] |
| `vit / transmission / ln_remove_radial_mean_c1` | +10.680 [+8.668, +12.737] | +1.590 [+0.787, +2.708] | +12.079 [+9.818, +14.340] |
| `vit / transmission / ln_remove_radial_mean_c0.1` | +4.209 [-2.009, +11.519] | +3.303 [+1.393, +4.716] | +4.348 [-2.567, +12.699] |

ELU derivativeを1へ置換すると、局所S1/S1 actualの平均Adam preactivation変位は負から正へ反転する（−0.006520→+0.002441；[frozen_interpretation.md](frozen_interpretation.md) のキー/CI参照）が、full Uは11.302 pp悪化する。よって「平均位置の下向きpushを消せば改善する」という単一媒介説明は支持されない。ただしunityは全gradientの方向・normを同時に変えるため、負方向shiftそのものが無害と結論することもできない。

ViTではradial subtractionを除くとscale1で約10.85 pp悪化し、mean subtractionも除いた結果は約10.68 pp悪化。これはnative final-LN VJPがこの方策比較で成績を守る証拠であり、LNが読み出し成長の害を作ったという証拠ではない。radial-onlyとradial+meanの点推定が近くてもmean成分の因果効果ゼロを証明しない。scale0.1ではacute悪化は支持される一方、fullのCIは広く0を跨ぐ。これらは最終LNの入力VJPを変更したsurrogateであってLN層除去ではなく、dynamic gradientはnormmatchしていないため、方向と倍率のどちらがfull差を媒介したかは未同定。

**6. body Adam memory：単なる「蓄積した悪いmoment」ではない。** memoryの符号はkeep−clearで、正ならclearが改善。bodyにはViT final gamma/betaのmomentsと全非head step countersを含む。

| kind / contrast | ELU (`model=mlp`) | ViT (`model=vit`) |
| --- | ---: | ---: |
| `memory / S_c1` | -4.356 [-4.686, -4.016] | -5.990 [-6.601, -5.222] |
| `memory / S_c0.1` | +6.156 [+5.755, +6.595] | +2.557 [+0.367, +4.768] |
| `memory / P_c1` | -3.607 [-3.945, -3.259] | +0.786 [-2.334, +4.297] |
| `memory / P_c0.1` | +6.275 [+5.920, +6.666] | +3.094 [+1.675, +4.865] |
| `interaction / scale_by_memory_S` | -10.512 [-10.671, -10.331] | -8.547 [-10.784, -6.569] |
| `interaction / scale_by_memory_P` | -9.882 [-10.256, -9.600] | -2.308 [-4.238, +0.372] |
| `interaction / direction_by_memory_c1` | -0.750 [-1.124, -0.415] | -6.775 [-9.728, -3.935] |
| `interaction / direction_by_memory_c0.1` | -0.120 [-0.481, +0.228] | -0.537 [-3.243, +2.451] |

Sでは両モデルともunscaledのbody memoryを保持した方が良く、縮小後はclearした方が良い。ELUではP後も同じ反転がある。ViTのP c1は未解決だが、P c0.1ではclearが改善する。optimizerはreadoutの尺度・方向変更への応答を変えており、勾配normだけからupdateを推定できない。局所の同一p・backward scale0.1でも保存Adamの方向がnativeと異なる結果と整合するが、それが長期interactionの全媒介であるとは示していない。clearはfirst/second momentsとstepをまとめて変えるので、どの成分が原因かも未分離。Oのbody-clear continuationはなく、Oについてmemory interactionは判定できない。

**7. 到達・終点と、残る微視的なgap。** `seed_metrics.csv` のphysical S/P/O/body_keepでは、ELU c1は全てfirst T90が0/10、c0.1は全て10/10で到達しsustainedも10/10。未到達を4000更新で到達した扱いにせず、T90差や総step短縮を有限値で断定しない。ViTではP c1/c0.1とも到達5/5であり、P後に大きな遅れが必ず残るという結果ではない。U改善は終点精度改善と別で、例えばViT S scaleのaccuracy_end差(c1−c0.1)は +1.464 [-0.456, +4.744] ppで未解決、Pでは +0.128 [+0.040, +0.288] pp。尺度利益のUだけから最終精度の優位へ言い換えない。

今回のfrozen actual-labelでは、ELUのS固定backward P/Oで平均位置pushが弱まり、physical P/P・O/Oでは負のAdam位置変位が残る。ViTのS固定backward Pではfeature/pre-LN normが減らずtrunk normが減り、Oではtrunk normが増える。登録uniform-label secondaryもこの定性的構造を支持するが、uniformはラベルに関する期待gradientでありAdam(E[g])≠E[Adam(g)]、norm(E[g])≠E[norm(g)]。

したがってELUの残差・readout・activation derivativeの組合せと、ViTの方向・行列幾何・LNを含む上流伝達の違いは具体化できた。しかし前者の位置shift、後者のtrunk norm/angleのどれが遅れをどれだけ媒介したかは未同定。特にViTのP効果を最終LNのnorm減衰だけで説明するには足りず、特定upstream層やGram成分へ原因を局在化してもいない。

ViTは `W→0.1W, gamma,beta→10倍` でlogits・pre-LN/trunk VJPを保存するgaugeを持つため、raw W norm単独は関数不変な原因量ではない。W diag(gamma)・W beta+bを考慮する必要があるが、それらのnorm一つもP/Oの違いを説明しない。このgaugeはAdamのhead/affine座標更新・以後の軌道を保存しない。ELUとViTではsource task・深さ・活性化・データ・更新予算が異なるので、違いをLNの有無だけに因果帰属できない。

回答として最も強く言えるのは、**ELUの大きいreadoutに伴う適応不利益は今回のP/Oでも残る尺度依存として現れ、ViTではその不利益がPという方向・幾何の介入に敏感だった。どちらもforward、backward、optimizerの相互作用であり、native activation/LN Jacobianを外せば解決する害ではない。** なぜ元の訓練でreadoutが成長したか、未知の方向変換・別状態でも同じか、局所機構の媒介割合は、この設計からは決まらない。

読込確認：`contrasts.csv` 539行、`seed_metrics.csv` 280行、model別source数10/5。実モデル再構築・学習・GPU・raw曲線の再集計・実行コード変更は行っていない。入力SHA256：

- `contrasts.csv`: `2e85548c3e55d3d83e9a705e105cca3726256471f7fa23473e1db45706860bfa`
- `seed_metrics.csv`: `16c2a5d0391dcee338ce9b202f5028260a5e7c284cc11fa7e17af2a9a65e5bc4`
