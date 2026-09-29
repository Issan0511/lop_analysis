# 共有入力ロジスティック模型：固定読み出しと上端の検証

2026-09-29。spec section Bの主行列135条件、線形再パラメータ化対照45条件。既存RL-MNISTの再現実験ではない。事前にデータ・初期値・設定を保存したcold-startの有限時間比較。主行列のdtypeはfloat64、全データAdam 4000歩、学習率0.001、beta=(0.9,0.999)、epsilon=1e-8。初期値から読み出しgainだけを変えた。後から学習済みvを変更する介入ではない。

## 結果

**生の上端はsmall gainで高くなるが、標準化した上端の突出は一貫して増えない。** 9 model×dataset、各5 paired seed、計45組の全てでgain0.1の生の上端がgain10より高かった。一方、(max−median)/sが高かったのは18/45組。ここでELU8はunit内平均を先に取り、seedを単位に比較している。45組は異なる条件を含む記述集計であり、45独立replicateの統計検定ではない。

特にELU1 balancedでは、小さいgainほど生の上端は高いが、標準化tailはむしろ低い。ELU1 rareではmax/sの小gain−大gain差は1.331、そのうち(max−median)/sの差は0.142なので、約89%はmedian/sの違いである。これはこの条件の記述的な差の分解であり、媒介因果の割合ではない。ELU8 rareでは標準化tail差は平均0.0047でほぼ変わらない。

下表は全て5seed平均（ELU8はunit平均を先に取る）。sは入力128点にわたる母標準偏差。max/sは比を各unitで計算してから平均しており、平均max÷平均sではない。

| Model | Dataset | gain | loss | raw max | s | max/s | (max−median)/s | p+ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| linear | balanced_separable | 0.1 | 0.5120 | 16.9292 | 7.0281 | 2.3972 | 2.3972 | 0.5000 |
| linear | balanced_separable | 1.0 | 0.1756 | 7.0306 | 2.9359 | 2.4010 | 2.4010 | 0.5000 |
| linear | balanced_separable | 10.0 | 0.0301 | 2.9981 | 1.1758 | 2.5501 | 2.5501 | 0.5000 |
| linear | rare_separable | 0.1 | 0.1810 | 16.3239 | 6.3421 | 2.5685 | 2.9294 | 0.3438 |
| linear | rare_separable | 1.0 | 0.0794 | 4.4155 | 2.2339 | 1.9742 | 2.8539 | 0.1766 |
| linear | rare_separable | 10.0 | 0.0217 | 0.8683 | 0.6300 | 1.3831 | 2.8025 | 0.0641 |
| linear | random_labels | 0.1 | 0.6632 | 11.8247 | 4.4346 | 2.6805 | 2.6805 | 0.5000 |
| linear | random_labels | 1.0 | 0.6619 | 1.4259 | 0.5189 | 2.7450 | 2.7450 | 0.5000 |
| linear | random_labels | 10.0 | 0.6619 | 0.1426 | 0.0519 | 2.7450 | 2.7450 | 0.5000 |
| elu1 | balanced_separable | 0.1 | 0.5363 | 17.4116 | 5.6559 | 3.0695 | 2.2737 | 0.7391 |
| elu1 | balanced_separable | 1.0 | 0.1767 | 9.8673 | 2.9444 | 3.3619 | 2.4078 | 0.8094 |
| elu1 | balanced_separable | 10.0 | 0.0300 | 3.2395 | 1.1916 | 2.7191 | 2.5506 | 0.5375 |
| elu1 | rare_separable | 0.1 | 0.1923 | 16.4652 | 5.9809 | 2.7429 | 2.9431 | 0.3891 |
| elu1 | rare_separable | 1.0 | 0.0931 | 5.0568 | 2.3713 | 2.1330 | 2.8531 | 0.2219 |
| elu1 | rare_separable | 10.0 | 0.0221 | 0.9258 | 0.6568 | 1.4120 | 2.8009 | 0.0672 |
| elu1 | random_labels | 0.1 | 0.6582 | 17.6375 | 5.7838 | 3.0827 | 2.6616 | 0.6734 |
| elu1 | random_labels | 1.0 | 0.6348 | 4.1611 | 2.3823 | 2.6873 | 2.5531 | 0.4734 |
| elu1 | random_labels | 10.0 | 0.6612 | 0.0769 | 0.0587 | 1.2920 | 2.7382 | 0.2031 |
| elu8 | balanced_separable | 0.1 | 0.3744 | 13.4806 | 4.3364 | 3.1191 | 2.2063 | 0.7840 |
| elu8 | balanced_separable | 1.0 | 0.0825 | 6.5952 | 2.0488 | 3.2267 | 2.5108 | 0.7383 |
| elu8 | balanced_separable | 10.0 | 0.0070 | 2.6377 | 0.8707 | 3.0271 | 2.5326 | 0.6719 |
| elu8 | rare_separable | 0.1 | 0.1495 | 17.3151 | 6.3586 | 2.7326 | 2.7098 | 0.4922 |
| elu8 | rare_separable | 1.0 | 0.0286 | 9.4245 | 3.6852 | 2.4226 | 2.6028 | 0.4264 |
| elu8 | rare_separable | 10.0 | 0.0023 | 3.0770 | 1.1315 | 2.5690 | 2.7051 | 0.4463 |
| elu8 | random_labels | 0.1 | 0.6188 | 17.1434 | 6.3623 | 2.7140 | 2.5209 | 0.5670 |
| elu8 | random_labels | 1.0 | 0.4171 | 15.5412 | 5.6766 | 2.7458 | 2.5899 | 0.5482 |
| elu8 | random_labels | 10.0 | 0.1866 | 9.0209 | 3.1627 | 2.8470 | 2.6225 | 0.5773 |

## Paired seedでの方向

下表の数値はgain0.1がgain10を上回ったseed数/5。各seed内でデータと初期hidden parameterを共有する。

| Model | Dataset | raw max | max/s | (max−median)/s |
|---|---|---:|---:|---:|
| linear | balanced_separable | 5/5 | 1/5 | 1/5 |
| linear | rare_separable | 5/5 | 5/5 | 3/5 |
| linear | random_labels | 5/5 | 2/5 | 2/5 |
| elu1 | balanced_separable | 5/5 | 5/5 | 1/5 |
| elu1 | rare_separable | 5/5 | 5/5 | 3/5 |
| elu1 | random_labels | 5/5 | 5/5 | 3/5 |
| elu8 | balanced_separable | 5/5 | 3/5 | 0/5 |
| elu8 | rare_separable | 5/5 | 3/5 | 3/5 |
| elu8 | random_labels | 5/5 | 0/5 | 2/5 |

## 分布形についての厳密な制約

固定入力集合について z=wᵀx+b とすると、

\[
\frac{z_{\max}}s=\frac{z_{\max}-\operatorname{median}z}s+\frac{\operatorname{median}z}s.
\]

さらに入力共分散をΣとするとs²=wᵀΣwなので、

\[
\frac{z_{\max}-\operatorname{median}z}s
=\frac{\max_x w^Tx-\operatorname{median}_x w^Tx}{\sqrt{w^T\Sigma w}}.
\]

右辺はbにもwの正の一様な伸縮にも依存しない。同じ入力集合でこの量が変わるには、wの方向が変わる必要がある。これはscalarの振幅だけの説明では導けない。今回の±入力ではmedian z=bなので、max/sの変化をbiasの移動と方向の変化に明確に分けられる。この特殊な対称入力がRL-MNISTを代表するとは主張しない。

## 線形模型の厳密な再パラメータ化

logit=g(wx+b)+cでg>0を固定し、a=gw、d=gbと置く。元のhidden gradientは有効係数(a,d)に対するgradientのg倍。Adam momentはそれぞれg倍とg²倍になるため、(a,d)はgain1、学習率ηg、epsilon/gで学習する模型と厳密に対応する。初期値もa₀=gw₀,d₀=gb₀に合わせる必要がある。出力bias cは学習率η、epsilonを変えない。

従って線形模型のgain依存は「別の原因の力」ではなく、初期logitと有効学習率・epsilonが変わった問題として完全に記述できる。元のhidden zが同じ、という対応ではなく、logitと損失が同じになる対応である。primary gain比較は初期hiddenを同じにした介入なので、initial logitは異なる。ELUは非同次であり、この等価性をそのままELUに使えない。

## 独立実装と数値精度の監査

- 3モデルそれぞれseed2/rare/gain10でnumpy勾配をtorch autogradと照合。最大絶対誤差1.12e-15以下。
- 中央差分4座標（wの最初と最後、b、c）との最大誤差4.76e-10以下。
- torch.optim.Adamとの32歩照合：parameter最大誤差4.86e-17以下、momentも照合。
- 線形の再パラメータ化は45条件の全4000歩を比較。同じ物理状態・対応するmomentからの1歩を7時点で独立に比較し、gradient誤差2.22e-16、parameter誤差1.78e-15。
- separable全条件の全軌道logit差は1.50e-13以下。random labelのgain0.1/1も3.78e-15以下。
- random label gain10では、1000歩まで全seedが1e-15程度で対応するが、その後一部seedで丸め差が増幅する。全軌道最大logit差0.01459、損失差1.63e-6。最終checkpointでは最大8.65e-5。この腕について「浮動小数点で全軌道が一致した」とは報告しない。
- 当初p−yを直接計算した実装では、対称データで厳密に0であるbias勾配に丸めが入り、Adamで増幅された。安定なsigned-label residual、対になった和、線形模型の厳密なゼロbias勾配の保存に変更した。データ・モデル・hyperparameterは不変。初回コードとconfig、変更理由も保存した。
- 上記random/gain10のみ、事後の数値監査としてlongdouble（表示名float128、mantissa63 bits）でも繰り返した。3seedでは全軌道差約1e-18、2seedでは最大0.01848の差が生じ、損失差は1.57e-6。高精度化だけで全軌道一致を保証できないことも記録する。これは新しい科学的条件の探索ではなく、等価な漸化式の数値感度の確認である。

## 遅い数値分岐の追加診断（事後、5条件のみ）

主行列を変更せずrandom-label/linear/gain10の5seedだけを再計算した。seed0の1410歩、seed2の3580歩で二つの等価表現のlogit差が初めて1e−12を超えた。その時のgradient L2はそれぞれ1.26e−14、2.41e−13。直近100歩のloss範囲は1.11e−16、2.22e−16で、数値上は停留していた。hidden RMSの最小座標は7.05e−4、8.10e−4でepsilonより十分大きい。従ってこの遅い分岐はepsilon支配ではない。

停留点近傍でRMSを一時的に固定した線形化を補助診断に使った。P=diag(η/(RMS+epsilon))、Hをhidden Hessianとすると、preconditioned Hessianの固有値λごとのmomentum写像は

\[
r^2-\{1+\beta_1-(1-\beta_1)\lambda\}r+\beta_1=0.
\]

固定Pの安定域は0<λ<2(1+β₁)/(1−β₁)=38。最初の分岐点でλmax(P^1/2 H P^1/2)は39.00と38.79だった。勾配がほぼゼロになってRMSが減衰し、この局所安定域を越えたところで丸めが増幅するという説明と整合する。これはPを固定した近似的な局所診断であり、時変のAdam全系についての大域的収束定理ではない。

最終4000歩で等価表現の違いが分布指標に与える影響は、最大でもraw maxが9.03e−7、max/sおよびcentered tailが1.59e−5、p+は0だった。主表のgainによる差を変える規模ではない。shared_sensitivity_crossings.csvとshared_sensitivity_endpoint_effects.csvに全値を保存し、再実行用の短いsourceもshared_sensitivity_diagnostic.py.txtに保存した。

## この検証から言えないこと

- 実験はcold startであり、課題51で学習済みvを10倍にすると上端が下がるという応答を再現していない。
- equal-time比較なのでlossも異なる。同じ精度・同じlossでの平衡を比較したものではない。
- full batchであるため、ミニバッチ雑音によるvの抑制はここには存在しない。固定v後のhidden dynamicsを切り出しただけ。
- 1-unit/8-unit・binary・paired synthetic inputs。RL-MNISTの100-unit/10-class/課題切替/過去のAdam履歴を再現していない。
- 大きい生の上端を得ることと、本体から離れた細い上端を得ることは別。特にp+が低いRL-MNISTの分布全体を今回の模型で説明できたとは言えない。

## 再現とファイル

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/logistic-v-height-mpl /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/logistic_v_height_0929/shared_models.py --out results/logistic_v_height_0929
```

- shared_config.json, shared_design.npz：fit前に保存した定義・実データ・初期値・SHA256。
- shared_units.csv：per unit、7 checkpointの全分布指標。shared_runs.csv：unit内平均したrun指標。
- shared_summary.csv：5seed平均と範囲。shared_paired.csv：seed対応差。
- shared_states_*.npz：7 checkpointのhidden/output parameters、最終gradients・Adam moments。線形matched control parametersも保存。
- shared_checks.json, shared_equivalence_checkpoints.csv：独立検算と等価性監査。
- shared_shapes.png：生上端、max/s、(max−median)/sを別列に表示した図。
- shared_provenance.json：実行時のcode SHA・git hash・環境。解析codeは未commit状態で実行したため、git hashに加えてcode SHAが実行ファイルの識別子。
