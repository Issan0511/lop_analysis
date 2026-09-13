# zero_attraction_learning_0913 — 新しい学習軌道で零点と重み収縮を測る

親: design_zero_attraction_0913.md / 状態: 検証中 / 実装・本走前に固定
目的: 原点傾き2のSnakeで着座と入力重み収縮が同居するかを新たに再現し、Leakyの傾き/定数の格子で適用範囲を調べる。
これは自然学習での形状効果を検定する第一段階。自己項の因果やCE/Adamへの一般化は本走だけで確定しない。

## 構成
edge_law_0905 runner、condA、幅100・1層、二乗損失、SGD、10seedを1プロセス。
seeds0–9, generator_offset=0、用量12.16、target_mu_norm=3.041、centered_layers=[1]。
各条件500task、1task=10000update、5Mupdate。
新しい共通lr=.005。全条件30kpreflight後、非有限パラメータ/非有限lossが1条件でも出た場合のみ、
全19条件の共通lrを.0025、さらに.00125の順に落とす。最初に全条件が通ったlrを登録実行値とする。
全て失敗なら本走しない。性能を見たlr選択はしない。ステップ数は変更しない。
CPU float32、torch版・マシン・コードhashをprovenanceに固定。OMP/MKL/OPENBLAS各1。
チェックポイント:0,200000,1000000,5000000。seed別にプロセスを分割しない。

## 19条件
Snake alpha=1、位相normal/peak/valley × q=-.5,0,+.5 の9条件。
normal: z+sin(z)^2+q、gate=1+sin2z、curv=2cos2z。
peak: z+sin2z/2+q、gate=1+cos2z、curv=-2sin2z。
valley: z-sin2z/2+q、gate=1-cos2z、curv=2sin2z。
normal q0は既存snakeと同じコード経路。
Leaky a=.1,.3,.7 × q=-.5,0,+.5 の9条件（既存leaky_off実装）。
linear: leaky_relu a=1,q=0の1条件。
位相移動は研究室PCにあるsnake_phase_0904の再現ではなく、この明示式の新実験。
全条件初期W,b,v、入力/教師RNGは対応。同じ初期予測ではないことを明示する。
固定の出力補正で初期予測を揃える介入は本走に混ぜない。

## 観測と集約
全unitを含む。ALIVEで除外しない。early task2–20、middle21–100、late451–500。
初期task1の学習は別記し、task間収支から除外。
主量 F = sqrt(mean_{unit,t in late}||w_free||^2 / mean_unit||w_free_init||^2)、seedごと。
主比較: peak,q0 対 normal,q0 のlog(F)のseed対応差。
95% paired percentile bootstrap5000回、rng2026091301。seedが独立単位。
- CI上端<log(.8): STRONG_PHASE_SUPPRESSION（20%以上の抑制）
- 上記以外でCI上端<0: DIRECTIONAL_PHASE_SUPPRESSION
- CI下端>0: PHASE_INCREASE
- その他: INCONCLUSIVE
別判定: peakのlog(F)のseed平均CI上端<0なら POPULATION_SHRINKAGE。
log(F)比の確証はこの1比較に限定、他の格子は探索的効果量/CIとして表示する。
初期からは増大しているが対照より小さい場合を「縮んだ」と書かない。

副量:
- 全W、free、fixed15、b、v、Var_x(z)、zbar、零点z0との差、loss/unfitのearly/late。
- 各条件の単調性から数値で一意な零点を解き、phi(z0)=0を検算する。
- near-root=|zbar-z0|<=.1とall-input-near=max|z-z0|<=.1の割合を別記。
- finalで選ぶnear-root群は記述のみ。±.05,±.2感度。unit内の初期比中央値も示す。
- vが小さい群も落とさず、出力寄与Var(v phi)、v^2 E[phi'^2]を表示。
- 幅の縮小と機能的可塑性を同一視しない。early/late lossと到達水準を併記。
- task端点のG=Q+R、D,N,cをfree射影で計算。全WやMNIST中心化Wのcとは呼ばない。
- 0norm/0updateのcは未定義を記録。集約後の平均cからGを再構成しない。

## 同じcheckpointでの機序測定（新しい長期学習ではない）
全19条件×step0/200k/1M/5M、10seed・100unit、全32入力・固定教師/入力平均。
1unitずつ他unitとvを固定して摂動する。全unit同時摂動とは別。
平均摂動b±epsilon、epsilon=.01,.05,.1。
幅摂動w_free*(1±epsilon)、bで平均を厳密に補償、epsilon=.01,.05,.1。
総勾配、MSEのselfとrestを独立に計算し、autogradで照合する。
mean方向は元W,b更新を通したDeltazbarも記録。
幅方向は実w_free勾配の動径と、平均保持接線の方向微分を別に記録。
対称差から局所復元係数を測る。平均回帰の条件付き散布図だけで復元と呼ばない。
この局所介入は、他unitが動く自然軌道で同じ復元が残る証明ではない。
同じモデル・同じ誤差RMSで残差の入力整列を反転する代数対照、
q変更を出力biasで初期補償したW勾配不変対照、v=0零対照も実施する。

## preflight / 検査
- new Snake8名のfn/grad/curvをfloat64 autogradと差分で照合。root/slope/offsetの既知点も照合。
- normal q0が既存snakeと同経路。既存netsの関数・既存runnerを変更しない。
- 全19条件30k、有限W/b/v/loss、19×10seed、ログ列・checkpoint・lr一致。
- 初期W,b,vとRNG状態が全条件で一致。教師・入力列は同じ、activationで乱数を消費しない。
- preflightでsupport再構成、free variance、self+rest、実更新恒等式、記録stepの対応を検査。
- 有限性以外のperformanceに基づいて条件を落とさない。発散は理由つきで報告。
- peak/normal/valleyを誤って同じ関数にした変異で既知傾き検査が落ちること。
- 同じconfigの短縮走2本のstate/log一致で再現性確認。
- 1プロセスRSSを実測し、full記録配列の増分を加味。並列数は
  min(12, floor((MemAvailableGiB-4)/(1.5*projected_peakGiB)))、下限1。
- 実行前にspec/config/実装のcommit hashを保存。コード変更中に本走しない。
- 本走のinputファイル/ckpt/loghash、torch/python/platform、seed、判定の集約順を保存。

## 実施範囲
この登録が覆うのは19条件のSGD学習と同じcheckpointでの機序測定。
MSEでのAdam比較、MNIST/CE、射影の長期介入は包括設計の後続段階で、この結果の一般化先として保留する。
