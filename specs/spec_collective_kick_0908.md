# 少数ユニットへの大きなキックと集団応答 spec 0908

親: 活性化アニメ_自己項釣り合い事後検証_0908 / 状態: 未実施
run_id: collective_kick_0908 / 起案: Codex / 2026-09-08
Issa の「集団が戻るか、一部を大きく動かすと他のユニットも補うか」を検証する短時間の機構パイロット。既存5M状態は既知であり、新しい介入結果に対して実行前に固定する。LoP改善の確証走ではない。

## 1. 問いと競合仮説

主問い: 少数のユニットを大きく動かして固定したとき、他のユニットの自己モーメントが反対向きに変わり、集団のずれを補うか。
競合: 出力誤差は回復するが自己モーメントは補わない／読み出し・出力biasだけが補う／動かした本人だけが戻る／他のユニットが補う。
全seedを混ぜた相殺を証拠にしない。自己項の勾配が戻すことはleakyでは代数的に既知であり、主判定にはしない。

## 2. データと固定する条件

5M checkpoint:
- p3_extend_0902 LR_1216、E_1216: 各10seed×幅100、元lr=.01、中心化あり。
- snake_out_0904/results/_diag_w5_snake_0905/asweep_a075 と asweep_a10 内の SN5_a1: 各20seed×幅5、元lr=.0036/.0037、中心化なし。

checkpointの全32入力を復元。教師、flip_state、入力支持、中心化平均を固定し、同一入力支持で2000回のfull-batch二乗損失GDを行う。lrは保存値、計算float64、weight decayなし。元のonline学習・タスク切替をそのまま再現するものではない。各条件は独立コピーから開始、結果を混ぜずモデル別・seed別に評価する。

sigma_s=max(median_i std_x(z_i),.001) を介入前状態から計算。
対象集合 T はnumpy default_rng(202609081754+seed).choice(H,k,replace=False)で選ぶ。k=5(幅100、5%)／k=1(幅5、20%)。活性・読み出し・結果を見て選び直さない。
全体キック: 全bへ±.25 sigma_s。
少数キック: Tのbだけへ±4 sigma_s。W,v,cは介入直後には変更しない。

## 3. 条件（11腕）

- free_sham: 無介入、全パラメータ更新。
- global_plus/minus: 全体キック、その後全更新。
- pulse_plus/minus: 少数キック、その後全更新。
- hold_sham: 無介入、TのW,b,vを固定。他ユニットのW,b,vと出力cは更新。
- hold_plus/minus: 少数キック、TのW,b,vを介入後の値に固定。他はhold_shamと同じ。
- readout_sham: 無介入、全W,bとTのvを固定。他ユニットのvと出力cだけ更新。
- readout_plus/minus: 少数キック、readout_shamと同じ更新マスク。

holdはT自身の回復やTのvによる回避を止める。必ず同じ固定マスクのshamと比較する。readout腕は、出力回復が他ユニットの前活性の移動を必要とするかを見る対照。固定したこと自体を元の力学と同一視しない。

## 4. 記録と主endpoint

記録update=[0,1,2,5,10,20,50,100,200,500,1000,2000]。t=0はキック後・更新前。
各seed・各unitでzmean,zmin,zmax,std,b,v,m+=E(max(phi phi',0)),m-=E(max(-phi phi',0)),M=m+-m-,E|phi'|,実際の全勾配によるdelta b/lr,delta zmean/lrを記録。全32入力のphi・予測、W,b,v,cも保存する。
自己項S=sum_i v_i^2 M_i、絶対量A=sum_i v_i^2(m+ + m-)、rho=|S|/Aをseed内で計算。Tと非T別も保存。

hold ± とhold_shamの対応差に対し
D_T=S_T(kick,0)-S_T(sham,0)
C_S(t)=-[S_other(kick,t)-S_other(sham,t)]/D_T。
C_S=1は他ユニットがキックによるSのずれを全補償、0は補償なし、負は増幅。
|D_T|>max(1e-10,1e-6*A(sham,0)) を操作成立条件とする。不成立例は明記し選び直さない。
主endpointはLR_1216・2000更新で、seedごとにhold+/-のC_Sを平均し、その10seed中央値。全10seedの両符号で操作成立・有限完走が必要。
5000回のseed bootstrap(percentile95%,rng=202609081755)で中央値CIを計算。CIが[.8,1.2]内ならNEAR_COMPLETE_COMPENSATION、CI下限>0ならPARTIAL_COMPENSATION、CI上限<=0ならNO_POSITIVE_COMPENSATION、その他INCONCLUSIVE。操作不成立／発散／時間切れがあれば主判定INCOMPLETE。これは固定タスク・2000更新内の応答であり、長期一般命題に拡張しない。p値は作らない。

副endpoint:
- 各符号、各モデルのC_S・seed分布。
- 幾何由来の差 sum_other v_sham(t)^2 [M_kick(t)-M_sham(t)] と、読み出し由来の差 sum_other [v_kick(t)^2-v_sham(t)^2] M_kick(t)。和がS_otherの対応差と一致することを検算。
- 非Tユニットのzmean変化のRMS、および平均変化。隠れ状態の変化とv,cによる代替を区別する。
- 予測の初期キック差d0(x)、その後の非Tとcの予測差d_other(x,t)。出力補償 C_out=-E[d_other*d0]/E[d0^2]、全予測差のRMS／初期RMS、損失のsham対応差。初期RMS<=1e-10は出力比未定義として保存。
- global/pulseは同じfree_shamに対するS・平均z・予測の差を追跡。絶対rhoも出す。「元の軌道との差が縮む」と「S=0へ戻る」は別に読む。
- globalのt=0で、kickの符号と全勾配による集団中心の応答が反対向きか。自己項だけの戻りを主判定にしない。

## 5. 検算

- baseline zmean・損失は既存ログ照合を継承。checkpoint/source SHAを保存する。
- 全batch解析勾配をsrc/nets.pyのgrads_batch平均と照合。小さな独立例でautogradとも照合する。
- no-op条件と更新マスクが同じなら軌道一致。固定対象W,b,vは全時点で不変、最初の非Tパラメータは完全一致。
- t=0の他ユニットの勾配差は -2 v_j E[d0 phi'_j]（bias）と入力xを掛けたW版に一致する。この恒等式は配管検算で、集団平衡仮説の支持とは数えない。
- SのT/非T和、幾何/読み出し差の和を検算。rho分母0は未定義。
- 非有限／損失>1e8／パラメータ絶対値>1e6で該当seed-armを停止し発散として記録。落ちたseedを隠して平均しない。

## 6. 事前予測・限界・資源

Codex予測（Issaの予測ではない）: 触っていないユニットへの即時応答は共有残差を通じて生じる。出力回復は自己モーメント回復より一般的で、C_outが大きくてもC_Sが小さい分岐があり得る。leakyの全負対象ではbiasキックが出力の定数成分を動かすためcでの代替が有力。Snakeで必ず自己釣り合いが強いとは予測しない。

局所の読み出しが大きいと大きなキックは非線形・不安定になり得る。発散時にlr・対象・振幅を結果に合わせて変更しない。凍結支持full-batchとonlineの差、幅100/5の条件差が残る。
CPU1thread・nice10、seedプロセス並列なし。実行の壁時計上限900秒、時間切れは部分記録を保存しINCOMPLETE。元ログ・checkpoint・既存実験には書き込まない。
spec単独commit→実装commit→実行・結果commitを守る。長期学習を追加する場合はこの短い結果を確証扱いせず別の設計を固定する。
