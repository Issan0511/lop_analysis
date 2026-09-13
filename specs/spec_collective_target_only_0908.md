# 本人自由・周囲固定の比較 spec 0908
状態: 未実施 / run_id: collective_target_only_0908
親: 少数ユニットキックと集団応答_結果_0908 §11
依頼: Issa「比較を実施して」 / 設計: Codex / 2026-09-08
前実験のpulse/hold結果は既知。以下は新しい周囲固定条件に対する実行前登録であり、既知のpulseの独立追試とは扱わない。

## 問い
本人の自己項の回復に、周囲のユニットの学習が必要か。対象固定の逆向き介入。自己項の回復と前活性の位置への回復は別に測る。

## 継承
collective_kick_0908 と同じ4モデル（leaky100、elu100、Snake .75/1幅5）、5Mcheckpoint、全seed、対象unit集合、sigma、±4sigma biasキック。
同じ32入力・教師・中心化平均を固定。float64/full-batch GD/保存lr/2000更新。記録0,1,2,5,10,20,50,100,200,500,1000,2000。
新たなタスク切替・online・初期化変更・対象の選び直しはしない。

## 条件
9腕。いずれも対象TのW,b,vは学習可能。
- free_sham / pulse_plus / pulse_minus: 全員のW,b,v,cを学習。前実験の同名腕と再現照合する。
- alonec_sham / alonec_plus / alonec_minus: 周囲（非T）のW,b,vを固定。対象W,b,vと出力bias cは学習。
- alone_sham / alone_plus / alone_minus: 周囲W,b,vに加えcも固定。対象W,b,vだけ学習。
各介入は同じ固定マスクのshamと比較。主比較はcを両方で学習可能にしたpulse vs alonec。aloneはcによる代替も止めた副対照。固定するのは周囲であり本人ではない。

## 主endpointと判定
前の自由対照で回復が観測されたleaky100・上キックを主比較とする。選択理由は既知であり、全4モデル・両符号も記述する。
S_T=sum_T v^2 E(phi phi')、D_T=S_T(kick,0)-S_T(sham,0)。
各seedの2000更新時点で e_T=abs(S_T(kick,t)-S_T(sham,t))/abs(D_T)。
paired d_s=e_T(alonec)-e_T(pulse)。正なら周囲が学習する方が本人の自己項の回復が良い。
主統計量median_s d_s。全10seedの両条件が有限・操作成立した場合だけ判定。
seed bootstrap 5000回、percentile95%、rng=202609081940。
実質差の目安は初期自己項差の10%=.1:
CI下限>.1: PEER_LEARNING_BENEFITS_TARGET_RECOVERY
CI上限<-.1: PEER_LEARNING_HINDERS_TARGET_RECOVERY
CI全体が[-.1,.1]内: NO_MATERIAL_PEER_DEPENDENCE_IN_WINDOW
その他INCONCLUSIVE。失敗・操作不成立はINCOMPLETE。
これは2000更新窓の指標上の効果で、数学的必要性や長期の不在の証明ではない。
操作成立 abs(D_T)>max(1e-10,1e-6*A_sham0) は前specを継承。

## 副測定
- e_Tおよび総Sの絶対残存比、全unitのM,m+,m-,v,zmean,zstdと実際の全更新方向。
- 対象の平均前活性の差のRMS/初期キックRMS。位置の回復とvによる代替を区別。
- 全32入力の予測差RMS/初期RMS、損失、c変化。
- 対象自己項の変化をM変更/v変更に帳簿分解。成分は因果寄与と呼ばず、中央値を足さない。
- 非Tとcが固定条件では実際に不変か全記録点検査。t=0の対象勾配は全条件で同じ（同じkickのpulse/alonec/alone）を検査し、その後の違いが更新可能な自由度によるものか確認。
- 各seedで絶対値を取ってから中央値。符号付き中央値のゼロを復元と誤読しない。seedを混ぜて一つの集団を作らない。

## 検算・失敗・予測
前runnerのbaseline/実装勾配/autograd/noop検査を継承。pulseとfree_shamの既存全記録点の再現を照合。非有限/損失>1e8/パラメータabs>1e6で停止記録。前のSnake .75 pulse+ seed1の発散も隠さず報告。
最大900秒、CPU1thread、nice10、同一プロセス内batch。元データは読取りのみ。rawを別パスへ保存。
Codex予測（Issaの予測ではない）: 周囲が止まっても本人に共有残差の勾配は残るため、一定の回復は起こり得る。周囲自由が回復を速めるか、残差を先に消して本人の回復を弱めるかは未確定。cの学習可能性が代替経路になる可能性をalone条件で分ける。
spec単独commit→実装commit→結果commit。登録主判定は結果後に変更しない。
