この文書は完了した **全15 source・2520 frozen probe** の局所結果の解釈である。この局所解釈はcontinuationの完走前に作成した。完了した学習比較との統合はsynthesis.mdに示す。ここでは後の学習遅れ・回復・害についての結論は出さない。ELU task-end10 の10 seedsと ViT SL task-end100 の5 seedsは別々に集計しており、アーキテクチャ自体の因果比較ではない。

全表で `label_channel=actual`。`F/B/J` は `forward_bank/backward_bank/jacobian`、数値は seed平均と対応付きseed bootstrap 10000回の記述的95% CI。`S1/P1/O1` は元・列別置換・class orthogonalのscale1、`S01` はclass-centered scale0.1。特に指定しない限り [probe_estimates.csv](probe_estimates.csv) の `(model, actual, F, B, J, metric)` が一意の出典となる。cosineはノルムではなく方向であり、CIが0を跨ぐことを同等性の証明としない。

**ELU：大きな負方向の平均位置変化は、固定したreadout normだけでは説明できない。** `gd_push_mean>0` は unit SGD が全1200例の平均preactivationを下げる符号で、`adam_keep_mean_preact_shift<0` は保存Adamを使ったvirtual stepが実際にその平均を下げる符号。勾配源は共通の128例であり、次の実訓練batchそのものではない。

| model | F/B/J | gd_push_mean | adam_keep_mean_preact_shift |
| --- | --- | --- | --- |
| mlp | S1/S1/native | 0.295854 [0.272158, 0.31878] | -0.00652003 [-0.00725878, -0.00579927] |
| mlp | S1/P1/native | -0.0094727 [-0.0301339, 0.0105735] | 0.00130727 [0.000482433, 0.00213822] |
| mlp | S1/O1/native | 0.00597045 [-0.0356084, 0.0522299] | 0.000778142 [-0.000259822, 0.00171778] |
| mlp | P1/P1/native | 0.388773 [0.356979, 0.418684] | -0.00832365 [-0.00941828, -0.00716834] |
| mlp | O1/O1/native | 0.320548 [0.299802, 0.341942] | -0.00655759 [-0.00743897, -0.00562953] |
| mlp | S1/S1/elu_unity | -0.0229409 [-0.0363237, -0.00967056] | 0.0024407 [0.00112665, 0.00407668] |
| mlp | P1/P1/elu_unity | -0.54161 [-0.779409, -0.305978] | 0.0130665 [0.00604022, 0.0199184] |

固定した `p=S1` で backwardだけPへ変えると、trunk勾配ノルムは 1.46474 [1.37895, 1.55258] から 1.31854 [1.26942, 1.36986] へ比較的小さく変わる一方、平均GD pushは0付近となり、Adamの平均変位は正になる。paired `P_backward_geometry` はS−Pで、push差 0.305327 [0.26793, 0.343427]、Adam変位差 -0.0078273 [-0.00885479, -0.00681031]。出典：[probe_contrasts.csv](probe_contrasts.csv)。

しかし forwardとbackwardの両方をPにした物理bank `P1/P1/native` では、下向きpushと負のAdam変位が残る。Oも同様の対角パターンを示す。したがって「P後も残るpush＝整列を除いた純粋な大きさの害」とは読めない。ここで直接示されるのは、forwardが作る残差とbackward mapの組合せ、およびELU derivativeによる重み付けが平均位置変化を左右すること。forward・残差・Wを固定してderivativeを1にすると平均pushとAdam変位の符号が反転するが、これだけで長期の害や活性化を交換した効果は決まらない。

**ViT：PとOの差は、最終LNだけのノルム減衰では説明が足りない。** 以下は全て `model=vit, actual, F=S1, J=native` で、同じ完全な確率ベクトルを保つ。

| backward_bank | g_feature_unreduced_mean_example_norm | g_pre_norm | g_trunk_norm |
| --- | --- | --- | --- |
| S1 | 1.73494 [1.63893, 1.82734] | 0.0018177 [0.00144671, 0.00237835] | 59.8258 [48.3431, 72.6348] |
| P1 | 1.75383 [1.66445, 1.83551] | 0.00211603 [0.00169548, 0.00276218] | 13.7968 [10.2086, 18.6266] |
| O1 | 1.70332 [1.62548, 1.77367] | 0.00196542 [0.00154826, 0.00259355] | 71.4093 [58.6945, 81.4397] |

paired `P_backward_geometry` (S−P) は、head-input norm差 -0.0188925 [-0.0329493, -0.00031513]、pre-LN norm差 -0.000298332 [-0.000389168, -0.000234146]、trunk norm差 46.029 [37.8651, 53.6509]。Pでは前二者が減っていないのにtrunkで減る。OではtrunkのS−O差が -11.5834 [-19.8924, -3.27448] と逆方向である。Pは列間Gramを変え、Oはそれを保つため、この対照は両者を一括して「整列除去」とする説明に反する。ただし、特定のupstream層・方向・Gram成分が原因だとまでは識別していない。

**同じforward/WでJacobianを変えると、両モデルとも方向が変わる。勾配の大きさを揃えてもAdamの変位は同じにならない。** 下表は全て `actual, F=B=S1`。`nm_` はtrunk勾配L2を各seedのnative値へ揃えた操作で、ViTのfinal gamma/beta勾配は別扱いのまま。`cos_native` の参照は同じF/B/labelのnative gradientまたはnative Adam updateである。

| model | J | g_trunk_norm | nm_g_trunk_cos_native | nm_adam_keep_trunk_norm | nm_adam_keep_trunk_cos_native |
| --- | --- | --- | --- | --- | --- |
| mlp | elu_unity | 5.73391 [5.17698, 6.24806] | 0.294089 [0.264066, 0.323056] | 0.274451 [0.266704, 0.281952] | 0.260253 [0.241056, 0.284462] |
| vit | ln_remove_radial | 102.914 [78.9991, 128.296] | 0.64512 [0.572439, 0.716669] | 0.479693 [0.447745, 0.511641] | 0.555132 [0.520226, 0.594313] |
| vit | ln_remove_radial_mean | 102.793 [78.9832, 128.248] | 0.645136 [0.57225, 0.716663] | 0.479803 [0.447799, 0.511807] | 0.554798 [0.519808, 0.594064] |

ELUのnative Adam normは 0.0975687 [0.0909583, 0.105049] で、normmatched unityの方が大きい。normmatched unityの平均位置変位も 0.00118207 [0.000235423, 0.00218033] と正で、nativeの負変位から反転する。ViT native Adam normは 0.481801 [0.448712, 0.514569]。radial-removeのnormmatched値は近い点推定でも、更新方向cosineは上表の約0.555である。点推定やCIの重なりを更新normの同等性判定には使わない。

ViTのradial-removeは、native trunk norm 59.8258 [48.3431, 72.6348] に対し上表の値となり、paired差は 43.088 [29.7872, 56.3887]。これはfinal LNのradial subtractionが局所のtrunk normと方向に作用する証拠であり、LNを除去したモデルの結果でも、その作用が有害/有益という判定でもない。radialとmean subtractionの両方を戻しても伝達は `(gamma*g)/sigma` であり、identityにはならない。

**同一pのbackward scaleは勾配ではほぼ純粋な倍率だが、保存Adamにはそのまま移らない。** `actual, F=S1, B=S01, J=native` は、`S1/S1/native` に対してtrunk gradient normが0.1倍、gradient cosineが1。対応するAdam方向は次の通り。

| model | adam_keep_trunk_cos_baseline | adam_clear_trunk_cos_baseline |
| --- | --- | --- |
| mlp | 0.679814 [0.665453, 0.698092] | 0.99302 [0.992395, 0.993634] |
| vit | 0.827936 [0.817269, 0.837865] | 0.99968 [0.999635, 0.999716] |

この比較の参照は同じlabelの `S1/S1/native` である。保存momentsを使うと、勾配方向が同じでも更新方向が変わる。clearでもepsilonや丸めのため厳密なscale invarianceとは称さない。

**ViTのaffineとgauge。** `actual, F=B=S1` でnative/radial-remove/radial+mean-removeのfinal `g_gamma_norm` は全て 1.17947 [1.03013, 1.32955]、`g_beta_norm` は 0.181742 [0.125224, 0.236808]。これは各shared stateでaffine勾配を保存する設計とexact checksに一致する。介入後の軌道全体で同じ勾配が続くことは意味しない。

S/P/Oのscale1はraw `head_norm` が全て 4.41591 [4.28284, 4.58391]、`effective_Wgamma_centered_norm` も全て 5.09794 [4.82293, 5.36764]。それでも上記trunk応答は異なるため、raw/effective normのどちらか一つで局所伝達を説明できない。出典：[probe_fullset_estimates.csv](probe_fullset_estimates.csv)、`model=vit, bank=S1/P1/O1`、各metric。

5つのViT source全てでgauge検査PASS。`W→0.1W, gamma,beta→10倍` によるlogits relative L2誤差の最大は5.815e-8、trunk全体では2.265e-6（per-parameter最大5.719e-6）。これはactive loss行の同関数性を確認する数値検査であり、raw W shrinkの無害性やAdam軌道の同一性を示さない。検査の出典は `probes/probe_vit_SL_s{10252010..10252014}_t100.json` の `checks.gauge_logits`, `checks.gauge_trunk_vjp_global`, `gauge_check`、完了数は [probe_validation.json](probe_validation.json)。

今回の局所差を言い換えるなら、**ELUでは残差/readout/activation derivativeの組合せが平均位置の下向きpushを作ることが直接見え、ViTではhead入力勾配normが減らなくてもtrunk応答が減り、方向・行列の幾何とLNを含むupstream伝達を考慮する必要がある。両者ともAdamが勾配と実変位の関係を変える。** これは選んだ状態と128例での局所的な違いであり、「ELUは大きさだけ、ViTは整列だけ」という分離や、後の遅れの原因割合を支持する結果ではない。

解釈上の境界：uniform-label channelは期待gradientの診断であり、Adam(E[g])とE[Adam(g)]を混同しない。物理scale0.1とTGは初期p/body/head gradientを合わせるが、head Adamをclearした最初の同じ座標更新 `deltaW,deltab` がlogitsに反映される倍率はphysicalでは1、TGでは0.1。exact arithmeticで共通body更新後のh'にも `C(z_phys,new−z_TG,new)=(1−0.1)C(deltaW h'+deltab)`（Cはclass-centering）が成り立つ。その後はpが分岐するので、全期間の学習速度が一定10倍という意味ではない。さらにTGのforward/gain方策は200更新で解除される一方、physicalは初期重み介入であり同じ解除を持たない。全期間のphysical−TG差には、この200以後の方策差と適応も含まれるため、初期head再学習の反映倍率だけへ帰属できない。この代数関係も、学習比較の成績とは独立に成り立つ代数関係である。

再現用の読込範囲は上記3 aggregate CSVと [probe_seed_records.csv](probe_seed_records.csv)、完了済み15 probe JSONのみ。continuationのraw曲線はこの解釈に使用していない。

登録済みのsecondary channelでも定性的な構造は同じだった。`label_channel=uniform, jacobian=native` のMLP `gd_push_mean` はS1/S1で0.31107 [0.29896,0.32214]、S1/P1で0.00292 [−0.01054,0.01816]、P1/P1で0.40117 [0.38008,0.41981]。同じ順の `adam_keep_mean_preact_shift` は−0.006926 [−0.007653,−0.006239]、+0.000990 [0.000408,0.001623]、−0.008740 [−0.009585,−0.007854]。ViTの同じS1確率での `g_trunk_norm` の対応付きS−P差は18.793 [14.823,23.015]、S−O差は−33.695 [−46.205,−21.186]。出典は `probe_estimates.csv` の上記キー、`probe_contrasts.csv` の `(vit,uniform,P_backward_geometry/O_backward_geometry,g_trunk_norm)`。actualを主解析に保ち、このchannelはラベルに関する期待勾配の補助診断とする。ここでのAdamはAdam(E[g])、normは‖E[g]‖であり、E[Adam(g)]やE[‖g‖]ではない。
