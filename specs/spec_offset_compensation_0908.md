# c=-2：折れ目群の出力補償後にleak下降勾配が残るか
親: [[本人自由と周囲固定_比較結果_0908]] / 状態: 未実施
依頼: Issa「検証して」2026-09-08 / 設計: Codex
run_id: offset_compensation_0908

## 問いと範囲
「自己項の集団の帳尻を合わせるためにleak群が遠くへ行く」仮説を検証する機構介入。元のonline長期軌道そのものの再現ではない。既存40M checkpointから同じタスク・32入力・中心化平均を固定する。新しいkickやタスク切替はしない。
元ログは全時点の全パラメータ・群出力を持たないので、保存状態からの短い継続と、同状態への最小二乗対照を用いる。
元のz座標で h(z)=leaky(z,a=.1)-2。折れ目z=0、活性化の零点z=+2。出力biasはc_outと呼ぶ。

## データ・群分け
/home/issan/Projects/obsidian-research-data/act_offset_review_0908/tail/ckpts/LRoffm2_lr0p00125_1216_step40000000.pt
対応logs_tail全10seedでbaselineのzmean,zmin,zmax,v,eval_loss_exactを照合。
初期32入力で L:zmax<0（全点leak）、K:zmin<=0かつzmax>=0（折れ目を跨ぐ）、U:zmin>0（全点正枝）。全100unitの排他的分割、閾値ちょうど0はK。群は途中で変更しない。
「0付近」はKという支持による操作的定義。平均zが0に近い群やhが0に近い井戸群とは同一でない。各群の個数・初期位置分布を必ず報告し、K/Lが空なら該当seedは操作不成立。
KとUの役割混同を防ぐため別条件を置く。

## GD条件
全条件同じ初期状態。lr=.00125、float64、full-batch、10000更新（元の1タスク分の更新数）。入力支持・教師・中心化平均固定。CPU1thread/nice10/max900秒。
- all: 全W,b,v,c_outを学習（自然な固定タスク継続の対照）。
- K: KのW,b,vだけ学習。L,U,c_out固定。主条件。
- Kc: KのW,b,vおよびc_outを学習。
- KU: K,UのW,b,vだけ学習。L,c_out固定。
- KUc: K,UのW,b,vおよびc_outを学習。
0,1,2,5,10,20,50および100から10000まで100ごとに記録。
Lを固定した条件でもマスク前の勾配を記録し、「今固定を解除したらどちらへ動くか」を測る。終点で全更新1stepを実際に適用し、Lの平均前活性変化が予測と一致することを検算。
全記録点で固定パラメータの不変性を検査。発散(loss>1e8,param>1e6,非有限)は停止記録し欠測を有限値と誤集計しない。

## 量・主判定
r(x)=yhat-y。R=RMS(r)/RMS(r0)が「全入力の出力差残存比」。平均だけの差はabs(mean(r))/RMS(r0)と別記録。初期RMS<=1e-8なら操作不成立。
g_i=-grad_bi - mean(x) dot grad_Wi は固定入力分布の平均前活性の全更新方向（lrなし）。実際の1step予測はlr*g_i。bias方向も別測定。
主評価窓は最後の500更新の記録点9500..10000。
「補償済み」: 窓の全記録点でR<=.1（初期の予測誤差RMSの90%以上を削減）。平均出力だけの一致を補償済みとしない。初期誤差自体のMSEと教師分散も併記。
主seed指標: 窓×Lのmean(g_i)/RMS_L(g_i at0)。負なら平均として下降。初期勾配RMS<=1e-10ならこの比は不成立。符号相殺を隠さないため、負方向成分RMSの初期全勾配RMS比、下降unit割合、各unit gを併記。
主条件Kで操作成立・補償済みseedが6以上あれば、seed単位bootstrap5000回・rng202609081020のmedianの95%CIを計算。
CI上限<-.1: DOWNWARD_DRIVE_REMAINS_AFTER_COMPENSATION。
CI全体が[-.1,.1]内: NO_MATERIAL_NET_DRIFT_AFTER_COMPENSATION（平均の正味のみ・全unit停止とは言わない）。
CI下限>.1: UPWARD_DRIVE_AFTER_COMPENSATION。
その他INCONCLUSIVE。補償済みが6未満ならNOT_ENOUGH_COMPENSATED_SEEDS。Kc/KU/KUc/allは副測定で主条件を差し替えない。
この条件付き判定は全seedへの一般化ではない。母数・非到達seedを全て報告。

## 力の分解と帳簿
全入力でrho(x)=1+mean(x) dot x。self_i=-2*v_i² E[h_i h'_i rho]、rest_i=g_i-self_i。bias版はrho=1。恒等式と全点leakのg_b=-2*a*v*mean(r)を検算。
K/U/Lの出力およびc_outの変化を別保存。K条件はKだけが出力を変えることを検算。
全点leakの固定タスクでは完全なr=0なら全勾配0は恒真。これを自己項仮説の支持とは呼ばない。自己項が非零でもrestが打ち消す場合を明示する。

## 同じ初期状態への静的副対照（全10seed・GDの代用ではない）
1. mean_only: c_outだけを解析的に調整しmean(r)=0にする。平均一致後も入力別残差が残る状態。
2. oracle_K: Kのvだけの最小ノルム増分で32入力のrを最小二乗で補償（c_out固定）。
3. oracle_Kc: Kのvとc_outの最小ノルム増分。
4. oracle_KUc: K/Uのvとc_outの最小ノルム増分。
各特徴行列rank/condition、係数変化量、残差、Lのg/self/restを報告。oracleは「そこで補うことが可能か」を問うもので、GDが選ぶ経路や元の運動の証拠ではない。
affine投影でrをspan(1,x)に直交化すると全点leakの全隠れ勾配が消えることを配管検査に用いる（恒真・独立証拠に数えない）。

## 実装検査・記録
checkpoint/log一致、独立autogradによる全パラメータ勾配、自己/rest恒等式、全点leakの平均残差恒等式、固定不変性、解除1step照合。offsetは元specの式から実装し、古いnetsの無対応名へのfallthroughを避ける。
spec単独commit→実装commit→実行・結果commit/push。rawとコードhash保存。数値出力はresults/offset_compensation_0908/verdict.csvとsummary.mdに集約。
Codexの事前予測: 平均出力だけの一致はbの下降を消すが、W経由の平均前活性方向は残り得る。全入力の補償ならLの全勾配は小さくなるはず。ただしKがそれを達成できるかとGDでの速さは未確定。Issaの予測を代筆しない。
