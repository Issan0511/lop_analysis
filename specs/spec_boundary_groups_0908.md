# 境界の急降下群・0近傍群・設定点更新の検証 0908
親: [[出力補償後のleak下降勾配_結果_0908]] / 状態: 未実施
run_id: boundary_groups_0908 / Codex / 2026-09-08
Issaの依頼: 1 急降下した本人の位置回復、2 出力修正の分担、3 群別の負の設定点更新を検証。

## 元データと限界
既存 results/pmnist_boundary_0908 の SNA/LR × none/l2-1e-3、seed0,1,2。第1層を主対象。二層目のstar2=row_sum(W2)+b2は置換平均の設定点ではないので主張に使わない。
既存npzはtask101..120のend/jump/starと300更新までの平均前活性のみ。出力・重み・alpha状態はなく、Q2の正確な事後復元は不能。Q1/Q3を既存ログから読み、全3問を同じ記録で答えるため追加記録の再実行を行う。
元provenanceは4腕で上書きされLR/noneのみ残る。ファイル名・PREDICTION.mdとソースを照合し、腕固有の完全なprovenanceが欠ける旨を残す。

## 再実行条件
元pmnist_0905.pyの凍結コピーを使用し、データパスだけ既存data/mnistを参照。SNAはAdaptiveSnake(c=.6,beta=.01)、LRはa=.1。Adam lr=.001、L2=.001を全W,bに加える元のcoupled方式。120task、各625step、batch16、同じinit/permutation/data/batch/probe乱数stream。probeは元と同じtest512枚。
既存GPU走を優先し、追加走はCPU1thread×最大4プロセス、nice10。CPU算術はGPUと非同一になり得るため、同じseedでも旧軌道との一致を仮定しない。比較誤差を保存し、完全一致しなければ「同条件CPU再実行」として既存結果と別に集計。
開始前に元boundary runnerのCPU短縮走と観測を追加したrunnerの一致を検査。記録関数が乱数・訓練・alpha状態を変えないことを確認。計算上限1プロセス900秒。source/data/spec/code hashと各腕provenance、初期・task100終了・各境界始終状態とAdam/Snake状態を保存する。
新しい学習介入や新しいseed選択はせず、元の4腕3seedに記録を追加する。

## 群の定義（境界ごと・更新前に決定して固定）
d_i=jump1_i-end1_i。
D（急降下）: d_i< -1e-6のunitのうちdが小さい順に最大25unit。上昇unitを無理に入れない。
N（0近傍）: |jump1_i|の小さい順に25unit。タイはunit indexで解決。
排他的4群 D_only=D\\N、N_only=N\\D、both=D∩N、rest=残り。
Nは平均前活性が0に近い相対順位による定義で、支持が折れ目を跨ぐ群とも、ゲートが大きい群とも同義ではない。特にSnakeの0は折れ目ではない。Nの実際のzmean範囲と各群のprobe上平均|phi'|を併記する。群を結果に合わせて選び直さない。

## Q1: 位置回復
300step、および625step（タスク末尾）でr_i=z_i(t,s)-jump_iを測る。
Dについて recovery=Σ_D r_i*(-d_i)/Σ_D d_i²（1なら初期落下を同方向に打ち消した）、leftover=sqrt(Σ_D(z_i(t,s)-end_i)²/Σ_D d_i²)（0なら元位置）、戻った割合 |z_i(t,s)-end_i|<=.1|d_i| を報告。
分母<=1e-10やD空は操作不成立。各境界値→seed内平均→3seed値と中央値を報告。境界・unitを独立seedと見なさない。
既存625stepはtask101..119についてend1[t+1]を使用（これは新置換で学習し終えた同じtaskの末尾）。既存300stepは101..120。

## Q3: 設定点
star_i=mean(train pixels)*Σ_j W1_ij+b1_i。境界瞬間はパラメータ不変でstar不変。
Δstar=star(after)-star(before)を各群でmeanとsum、負成分sum、負unit割合に分ける。行和項とbias項も別保存（CPU再実行）。
既存star[t+1]-star[t]はtask101..119の625step分。star[119]-star[100]は19タスクの積算で「20更新分」としない。群ラベルを各taskで固定し、群別sumの全task積算が総driftに一致することを検算。
WDはAdamに混ぜたL2で、単純な直接収縮更新として因果分解しない。

## Q2: 出力修正の分担（二層非線形を無理にユニット出力の和にしない）
同じ新置換・同じprobe512枚・ラベルで0→300/625stepのCE低下を測る。
第1層活性A1を上の4群に分け、各群を境界直後/再適応後の値に切り替える4ブロック＋下流(W2,b2,W3,b3,Snake第2層alpha)の前/後を切り替える1ブロック、計5ブロックの32組を全評価。
第1層alphaの変化はA1側に含む。各ブロックのShapley CE改善寄与を全32組から正確に計算し、5寄与の総和=CE_before−CE_afterを検算。負寄与も隠さず報告。これは観測終点を使う帳簿分配であり、群を固定した因果介入ではない。
CE改善<=1e-6なら分担率は定義せず絶対寄与を報告。D_only+bothをD側寄与、N_onlyは「下がった群とは別の0近傍群」の寄与として比較。bothを二重加算しない。下流も独立列に残す。精度とlogit差RMSも併記。
新境界の適応後CEが改善しているか、Dの位置が残るか、N_onlyがD側より大きい寄与を持つかを同じ境界で対応させる。

## 読み
n=3なので有意差・普遍的機構を宣言せず、全3seed値と中央値を主報告する。
「置き去り＋他群補償」の記述的署名: 同じ境界でD leftover>.5、CE低下>1e-6、N_only寄与>D_only+both寄与かつN_only寄与>0。件数を全境界に対して報告するが判定ラベルにしない。
この署名だけでは負のstar driftの原因は確定しない。Q3でどの群に負更新が残るかを併記。
元の|refit|/|strip|は本人の符号付き回復の証拠として扱わない。旧集計、CPU新集計、再現誤差を分ける。
spec単独commit→実装commit→検算・実行→結果commit/push。生状態はgit外にsha付き保存。登録後の変更は理由を追記し、主定義を差し替えない。
