# 追補: 自己項からtask侵食角までの自然更新追跡 0913

親: spec_zero_attraction_learning_0913.md / 格: 本走中に起案した追加解析。追加解析の実装・実行前に固定。
起案時点でtask100のcheckpoint記述を見ている。未知結果への独立な事前予測とはしない。
本走の主判定・条件は変更しない。現在進むSGD実験の保存状態を再生し、新しい比較学習軌道を増やさない。

## 問いと対象
終端のself/restだけでは、どちらがtask区間の侵食cを作ったか決まらない。
対象: 通常/peak/valley Snakeのq0、Leaky a=.1のq0,±.5、linearの7条件。
結果によって代表を選ばず、この7条件を固定する。
checkpoint task20と100から、task21と101の各10000更新を追跡。
各10seed・100unit、CPU float32の元runnerと同じ演算経路。
checkpointにはRNGがないため、同じ初期configでenv.stepだけを200000/1000000回進めてinput RNGを復元する。
保存flip_state/tと教師/初期条件を照合する。再生したtask末尾のw_free/v/Wnorm/zbarを元の完全ログと照合。
一致しなければ自然軌道の再生と呼ばず結果を保留する。

## 保存と集計
milestone: update1,20,200,1000,10000。
各stepの更新前u=w_freeと実duを使い、
du_self=-eta * 2 v^2 phi phi' x_free、
du_rest=-eta * 2 v(delta-v phi)phi' x_free、
du_round=du_actual-du_self-du_rest。
self/restはfloat64で帳簿を作り、元更新のfloat32丸めと生勾配丸めの差はdu_roundとして記録する。
丸めをselfかrestに押し付けない。weight decayなし、plainSGD。

各stepのR_self=2<u,du_self>,R_rest,R_round、Q=||du_actual||^2を積む。
区間のG=||u_end||^2-||u_start||^2=sum(R_self+R_rest+R_round+Q)を検算。
cND=-<u_start,du_task>。
I=-sum<u_step,du_step>、K=sum_{r<s}<du_r,du_s>とするとcND=I+K。
Iをself/rest/roundに分け、D^2=sum Q+2Kも照合する。
**task末尾cの違いを自己項だけに帰属せず、残りの項と経路Kを同じ単位で比較する。**
c未定義のunitを0で補完しない。unitやstepを独立標本にしない。
seed内unit平均→seed間平均、seed別値・窓を全て保存する。仮説の新しい支持/棄却ラベルは追加しない。

入力置換による前活性の跳びと、パラメータ更新による移動は分ける。
task21/101の始点と終点に全32点支持の平均/分散を保存。位相ごとのroot位置も記録。
MSE自己項は分解であり、除去介入ではない。AdamやCEへの一般化は未測定。

## 検査
- 元のW,b,v、teacher、input stream、env時刻、固定入力平均を復元。
- 新task末尾のuは元ログの正しいstep列と照合（原則byte一致、max_absも記録）。
- v/Wnorm/zbarはfloat32記録の丸めを踏まえatol5e-5/rtol1e-5。
- 全10seed・milestone/transition数を確認。
- G、D^2、cNDの恒等式はfloat64の帳簿でatol1e-7/rtol1e-8。
- 相殺で大項が残差になるため、相対誤差だけでなく絶対誤差とroundの大きさを表示。
