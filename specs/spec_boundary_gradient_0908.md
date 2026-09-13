# 切替直後の勾配と実更新 spec 0908
親: [[タスク切替後の前活性追跡_0908]] / 状態: 検証中 / 更新: 2026-09-08
実行前固定。既存軌道を見て選んだ追跡窓のため研究全体としては事後追跡。新規の勾配分解と集計仕様は実行前に固定する。主張の採否は変更しない。

## 問いと範囲
切替後の急降下は、現在の誤差勾配、L2、Adamの持ち越しのどれが担うか。Wとbのどちらで動くか。正側の適合と負側の移動がいつ起こるかを群別の出力修正と照合する。自己項による集団釣り合いの証明を目的としない。
CPU boundary_groups_0908の保存状態からSNA/LR × none/l2、seed0,1,2、task101..119を各100更新再開。元CUDA軌道と異なることは既知。CPU保存軌道への一致を毎更新検査し、CUDAの勾配を測ったと書かない。
batch16、lr.001、Adam beta1=.9,beta2=.999、eps1e-8、L2勾配2*.001*parameter（Wとb両方）、SnakeのEMAを含め元実装を保持。固定の512 test probeと元のdata/batch RNGを使う。
主窓は1..20、補助窓1..50,1..100。最初50更新は毎更新測定、100は累積endpoint。20更新で全運動が決まるとは仮定しない。

## 分解
固定probe平均muで第一層のDelta zbar = Delta W mu + Delta b。
設定点はDelta star = mu_global sum Delta W + Delta b。task alignment = Delta zbar - Delta starも保存。probeとglobal平均の定義は旧走と同一。
各更新のAdam denominatorを実更新値に固定し、numeratorを
beta1*m_previous + (1-beta1)*g_CE + (1-beta1)*g_L2
に分解、各項を同じ係数でDelta W,Delta bへ写す。加算一致を検査する。過去mには過去CEとL2両方が含まれる。これは実更新の加算的帳簿であり「L2を外した場合」の因果効果ではない。denominator由来の効果を別の加算項にしない。
現在CE勾配は実ミニバッチのz>0/z<=0入力寄与に分割し、W,b勾配の和との一致を検査。Snakeでは正負は傾きの大小と同義ではない。上流dL/da、局所phi'、dL/dzの個別値を保存し、勾配が大きい理由を区別できるようにする。
学習率.001のSGDなら生じるDelta zbarも参照方向として保存。Adamの実速度と同じ量として比較しない。SGD方向と実更新の反対符号割合（両方絶対値>1e-8）を報告。
CE勾配の平均項と相関項: gW = (sum_batch dL/dz)*mu + sum_batch (dL/dz)*(x-mu)。このmuは固定probe平均。b勾配と合わせたSGD方向を分解する。ミニバッチsumのdL/dzはCEmeanに含まれる1/Bを保持。

## 群と出力
旧定義のD_only,N_only,both,restを切替時に固定。Dは瞬間跳びの負側25ユニットまで、Nは切替後abs(zbar)最小25。Nは相対順位であり絶対ゼロ近傍ではない。Snakeにはゼロの折れ目はない。
各stepの全100ユニット生値を残す。群集計はユニット平均→19境界平均→3seed中央値と範囲。空群はNA、seedを独立単位とする。CIや仮説検定は行わない。
step1,5,10,20,30,50,100に旧実装の5ブロック32 coalitionのCE Shapleyを使う。4群の第一層活性と下流をbefore/afterで切り替え、CE改善との加算一致を検査する。切替直後step0からの累積修正であり、微小更新速度や因果介入とは呼ばない。

## 主報告と判定
1. 1..20累積のDelta zbarをW/bとAdam3項（各W/b）の両方で列挙。1..50/100も併記。
2. 各群の1..20負方向寄与massは各ユニットの累積成分をmin(value,0)にしてから絶対値を足し100で割る。bとWの下降massの比をseedごとに示す。正味比の爆発を避け、上向き成分も併記する。「biasが原因」と因果判定はしない。
3. 現CE方向と実更新の反対符号率、持ち越し/CE/L2の累積寄与、正負入力別CE寄与、CE修正の時間経過を報告。
4. 主機構判定は記述に留める。自己項帰属、正側が先に動くから負側が動く因果、L2長期効果は未検証と明記する。

## 検算・停止・保存
先に全4腕seed0 task101の100更新で元CPU記録と照合。計装の追加autogradが学習勾配を変えないことを別forwardのgradで比較（first step）。全走でもprobe zのmaxabs誤差2e-5以下、加算分解誤差2e-6以下、CE Shapley誤差1e-9以下を必須。失敗なら先に原因を直し、結果に有効な検算を残す。予測に合わないseedは除外しない。
CPU1thread、同時実行は高々2プロセス、元GPU実験を触らない。1腕seedの上限15分。全体30分で途中なら未完を明記する。
生ログはresults/boundary_gradient_0908/rawに保存しgitignore。ソース状態/コード/spec/dataのSHA256、出力SHA256と検算をprovenanceへ。生ログをobsidian-research-dataへチェックサム照合付きで退避し、CSV・図・summaryと結果ノートをcommit/push。spec本文は実行後変更しない。
