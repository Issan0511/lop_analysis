# 走1改訂: RC/raw/ReLUでの継続ReDo × CHB

run id: `redo_chb_rlcifar_1018`。予測者・改訂指示者Issa、起草Codex。改訂指示2026-10-18 02:36:16 JST。旧走を差し替えて上書きせず、新しいrunとして登録する。共通の完全性・数値・保存規約は `spec_six_runs_common_1018.md` を継承し、走1の箱・操作・予測については本書を優先する。新規評価の前に本書をcommitする。

## 0. 既知結果、改訂理由、時系列

- 旧走 `redo_position_2x2_1018` は5+1/ReLU/stdでt28末の一回ReDo/物理bias shiftを比較した事前登録試行であり、ホストCodexが既に2026-10-18 02:10 JSTに完走した。保存commit `abd5220b`、主判定MIXED、採用予測POSITION_CARRIESは支持されなかった。旧spec・全結果・予測不成立を保存し、未実施や成功へ変更しない。主走から外した理由と旧結果を報告する。今後の旧走再実行は行わない。
- 改訂時点で、RC/rawのCHBが200課題まで高いonlineを保った旧結果と、走4RMの交換/lr対照結果は既知である。CHB seed10–19の既存結果も参照して設計するため、CHBそのものの未知予測の成功とは数えない。新しいReDo×CHBの比較を、新しい評価結果が出る前に登録する。
- 旧5+1の一回介入ではReDo/物理bias shiftはいずれも改善したが、POSITION_CARRIESは成立せずMIXEDだった。周期ReDo/扉Hの旧結果とこの試行は別プロトコルであり、同一の操作確認とも扱わない。その性能差をすべて階数喪失の因果寄与へ帰属しない。ReDo43%とH約49%はseed集合・集約法が異なる。旧F51を残差の因果分解へ読み替えない。

## 1. 問いと比較の射程

乱数ラベルCIFAR/ReLU/rawで、最初から継続するReDo、既存CHB、両方の方策が200課題の後期再学習をどこまで保つかを比較する。劣化後に位置だけを急性に戻す実験ではない。

CHBは入力中心化C、隠れ特徴中心化H、b1ゼロ化、b2のdecoupled WDを含む既存パッケージ。none/ReDoではnative refを用いる。したがって主ラベルは **CHB_POLICY_CARRIES** とし、純粋な位置変数・bias単独・WD単独の因果効果とは解釈しない。raw refは早期から失敗し得るので、この比較単独でゆっくり蓄積するdriftだけを同定したとはしない。

## 2. 固定した箱と4腕

継承元 `src/relu_doors_0919.py`、`spec_relu_doors_0919.md`、`ch_chb_performance_1011`。継承ソース・データはhashで固定する。新worktreeの起点はmain `094489c30bcdf196946179ba07ed0e8cadd3a53f`。

- CIFAR-10、seedごとの固定1200枚、画像/255のraw、std割りなし。課題ごとに独立一様乱数ラベル。3072–100–100–10、biasあり、native ReLUはclamp(min=0)。
- seed10–19をR=10で束ね、200課題×400epoch×75更新=30,000更新/課題、batch16。全腕をtask1から新規実行する。主窓t181–200、早期参照t31–50。
- float32、CUDA、TF32 off、native決定論設定。Adam lr1e−3、betas(.9,.999)、eps1e−8、課題境界でもmoment・時刻を継続。実装検査seed100–109。本走のCPU移植・seed/課題削減はしない。
- 各seedの初期P、画像id、ラベルstream、バッチ順を対応させる。腕の実行順はnone→redo→CHB→both、GPUを逐次使用する。CHBによる入力/特徴変換は意図した差であり、腕間の初期logits同一を要求しない。

| 腕 | 方策 |
|---|---|
| none | native ref。C/H/Bなし、ReDoなし |
| redo | native ref＋下記の継続ReDo |
| CHB | native CHB、ReDoなし |
| both | native CHB＋同じ定義の継続ReDo |

4腕を全て再走し、旧対照の平均だけを混ぜない。none/CHBがnative宿主と一致することをGPU release前に確認する。

## 3. CHBの厳密な継承

Cは固定1200枚のseed別平均画像を引く。Hは両隠れ層のReLU出力からunit別EMAを引き、beta=.01、初期0、課題をまたいで保持する。native forward/backward→Adam→B→EMA更新の順を維持し、EMAには更新前forwardのzを用いる。

BはAdam後にb1を0、b2を(1−lr×.1713)倍にする。lambda=.1713は既存probe値をそのまま継承し、再探索しない。初期b1はnativeのランダム値であり、最初のAdam更新後からゼロにする。重みWDなし、b3は自由。biasのAdam momentと大域時刻もnative通り保持する。

## 4. 継続ReDoと合成の字義

τ=.1、周期F=1560更新を既存 `baselines_cifar5p1_1008` の採用値から移す。これは**5+1からの固定方策の移植**であり、RCで最適に較正されたReDoとは称さない。F51の780更新/課題と本走30,000更新/課題では相対頻度が違うことを明示する。この設定の陰性結果を全ReDo設定の不能性へ拡張しない。

1. task1からの絶対更新数kが1560の倍数になったとき、当該更新のnative Adam/B/EMAをすべて終えてから、CUDA graph外で一回ReDoを行う。warmup中は行わず、task境界で周期を戻さない。1腕につき6,000,000更新、予定3846イベント。
2. 直近4学習バッチの画像idを保存し、計64枚を同じ現在P/EMAで一度forwardする。ラベルは使わず、画像idはstreamを進めない。両隠れ層のmaskを、再初期化前の同じforwardから同時に固定する。
3. スコアは次のaffine層へ実際に渡る隠れ特徴hについて、score_i=mean|h_i|/(mean_j mean|h_j|+1e−9)、D={score_i≤.1}。CHBではh=ReLU(z)−EMAである。これはReLU前/直後の値だけで採点する別方策ではない。centered score、pre-H ReLU score、strict/native deathを別列にする。
4. seed別 `b5_redo_reinit` stream、fan_inに対応するnative一様初期化を用い、隠れ1→隠れ2の順でdrawする。まず全隠れ層の対象入力行を初期化しbiasを0、その後全対象出力列を0にする。入力W行/出力W列のAdam m/vは0。**bias m/v、global時刻、HのEMAは保持**する。W3の対象列以外とb3は不変。
5. ReDo後にB/Hを再適用しない。両腕で同じreinit stream名・seedを使うが、maskと個数は軌道により異なるので、同じdraw配列になるとは主張しない。generator state、更新時計、直近4batch id、event数とunitのreset generationをcheckpointに含める。

τ=.1は完全死亡だけでなく低活動の生存unitも選ぶ。またCHBのcentered feature活動とnative ReLUの死亡は同じ指標ではない。「ReDoが死亡だけを操作した」とは呼ばない。EMA保持を含む合成方策が本走の登録条件である。

## 5. 指標と字義通りの判定

E[a,s]=mean online_acc[a,s,t] over t181–200。onlineは30,000学習バッチの更新前argmax精度、memoとは別。先にseed内平均を作る。

R=E[redo]−E[none]、C=E[CHB]−E[none]、B=E[both]−E[none]。7区間はR,C,B,B−C,B−R,C−R,I=B−R−C。n10、df9、float64、各mean±t(9,1−.05/(2×7))×sampleSD/sqrt10。通常95%区間、全10差、中央値、符号数も併記。SD=0は一点区間/DEGENERATE_SD。

POSは下端>0、NI(A,B)はA−Bの下端>−.005、EQは区間全体が[−.005,+.005]内。2ptなど新しい成功幅を足さない。

| 主ラベル | 登録条件 |
|---|---|
| CHB_POLICY_CARRIES | POS(C) AND NI(CHB,both) AND POS(C−R) |
| REDO_POLICY_CARRIES | POS(R) AND NI(redo,both) AND POS(R−C) |
| MIXED | どちらも成立しない、または複数成立 |

別軸のADDITIVEフラグはPOS(R) AND POS(C) AND EQ(I)。主ラベルと競合させず、両方報告する。

「ReDoではCHBへ届かない」の確認は **REDO_SHORTFALL ⇔ lower(C−R)>.005**。単にReDo−noneが非有意なだけでは成立しない。全腕の連続精度と差を常に出し、CHB約99%という旧値を本走の新しい合否閾値にしない。完全性/発散/検査違反があれば科学ラベルを出さない。

## 6. 診断、機構の読み、反証

各task終端とReDoイベント直前/直後に、固定1200枚で両層のunit別m/s/r・b・幅、pre-H/actual h活動、strict z>0 gate/native微分の平均・分散、死亡率、reset maskを記録する。s=0と非有限rは別個に保持し、epsilonで隠さない。task終端には宿主定義のeff_rankも記録し、代数的rankとは呼ばない。

reset generationを追跡し、再初期化直後にも死んでいたunit（復活不成立）と、一度復活してから後に死んだ世代を分ける。ReDoが実際に死亡/休眠を低下させたか、なお位置が沈むかは操作確認と機構診断として出す。性能ラベルの成立から死亡除去やdrift停止を逆算しない。

世代診断の主定義は固定1200枚上のstrict alive=max_x z_i(x)>0、dead=max_x z_i(x)≤0。native微分が全て0かは別列とする。各reset後は両層の入力行/出力列操作を全て終えた後に測り、post-reset aliveだけを復活成立とする。後の登録snapshotでalive→deadを観測したときに観測上の再死亡を記録し、連続時間の寿命とは称さない。生存したまま次のresetで置換された世代は置換による打切りとする。自己resetされていないunitがReDo前alive→直後deadとなった場合はEVENT_ASSOCIATED_DEATHとして別記し、driftによる再死亡へ数えない。イベント間のsnapshotで観測された死亡も原因をdriftだけへ断定しない。

CHBがnoneを改善しなければ、この箱での予防効果は支持されない。ReDoが改善してCHB水準へ非劣性で届けば、今回の「継続ReDoでは不足しCHBが運ぶ」という予測を限定/撤回する。bothだけが高ければ相補性として報告する。CHBが優越してもC/H・b1固定・b2 WDの必要性はこの2×2だけでは分離できない。bias経路の分解は走3などと合わせて議論する。正しく実装された予測外れをS不合格にして捨てない。

「Drift, Not Death」は、死亡/低活動unitを除くこの方策だけでCHBの長期性能を再現できなかった場合の、箱と方策を限定した説明として扱う。普遍的な死の無関係、純粋なdrift mediation、5+1にも共通する単一機構を確立したとはしない。

## 7. S検査と指定変異

| S | 本物で要求 | 落とす変異 |
|---|---|---|
| S-native | ReDo offのnone/CHBがnative宿主のP/m/v/t/EMAと一致。CUDA R10 graph/eagerとwarmup巻戻し | task時刻reset、raw/std取り違え、初期b1をゼロ化 |
| S-cadence | 絶対1560周期、64画像/4batch、warmup非介入、両mask同時採点 | task-local周期、旧batch32の2batch、更新前ReDo |
| S-score | actual delivered feature採点、τ境界<=、centered/nativeの別名 | pre-Hだけ採点、閾値<へ変更、EMAを二度更新 |
| S-redo | 元recycler独立参照と入力行・出力列・moments・draw順が一致 | bias moment/tcもゼロ、出力列を残す、層順を逆転 |
| S-composition | 非零EMA/bias/momentでCHB→ReDo、EMA保持、B/H再適用なし | EMA reset、順逆転、b1/b2以外WD、ReDo後B/H再実行 |
| S-observer | 記録有無でP/EMA/RNG/stream軌道一致、resumeとgeneration一致 | 診断でbatch/reinit RNG消費、buffer/generator復元漏れ |
| S-generation | 全recycler後のstrict revival、置換打切り、event前後とイベント間の死亡を区別 | nativeゼロ点をstrict aliveへ算入、置換を死亡に数える、他層reset直後の死亡をdrift扱い |
| S-readout | 4×10×200完全行、seed結合・7区間・境界・独立ADDITIVE・非有意を未決 | seed並べ替え誤対応、欠測除外、nonsigをfailure/EQ、有限CSV書換え |

CPU合成検査と実データCUDA releaseを分ける。独立担当のsource-bound承認、CUDA検査、実データ/stream/出力hashの証跡が揃うまで本走へ進まない。原ソースは編集しない。GPU不在・競合は実行状態として保存し、偽のPASSを作らない。

## 8. Issaの採用予測と保存

採用方向予測は **CHB_POLICY_CARRIES かつ REDO_SHORTFALL**。CHBは長期性能を保ち、この移植ReDoではその水準へ届かないと予測する。ReDoにより一度復活したunitもdriftに伴い再び失われる、という機構予測は診断で検証する。確率・精度・効果量の数値予測は指定なし。記名は「Issaの2026-10-18 02:36:16 JSTの改訂指示をCodexが定式化」。

spec/実装/全import/データとsubset/stream/初期状態/各checkpoint/診断/解析をhashで保存。per_task、単位配列、イベント/世代、checks、GPU release、独立review、provenance、verdict、summary、停止/再開履歴を残す。4腕の同一seedの画像/ラベル/順序対応を検証してから集計する。旧CHB/none値の未検証再利用はしない。

既存のホストcontrollerは別セッションで走3/6等を実行中で、新runを自動認識しない。既存queue/state/HEADを変更せず、本走は新run専用の独立queueと共有GPU lockで調整する。旧走1は既存5走queueから既に外れている。raw退避/main統合/cleanupはCLAUDE.mdに従い、他セッションのworktreeを変更しない。
