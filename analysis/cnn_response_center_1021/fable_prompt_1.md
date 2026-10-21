# CNNの残る介入2本の設計レビュー

あなたはClaude API Fableへの新規ワーカー依頼です。既存チャットへの通信ではありません。Issaは2026-10-21に『CNN埋めましょう』『今回はちゃんと回してほしい。あとFableapiつかって相談していいよ』と指示しました。目的はV13機構論文のCNNの部品0〜5で不足する因果接続を必要最小限の登録走で埋めることです。読み取りだけ担当してください。ファイル編集、実験、他チャットへの送信はしないでください。他の作業者がいます。

作業ディレクトリ: /home/issan/Projects/claude。先に /home/issan/Projects/claude/CLAUDE.md を読んでください。repo /home/issan/Projects/claude/proj_004_drift はmainが古いので、今回の数値根拠は下記Vaultの結果ノートを読んでください。Vaultへは書かないでください。必要なノート:
- /home/issan/Projects/obsidian-research/可塑性喪失/論文作成/V13段落地図_機構を軸に_1019.md
- /home/issan/Projects/obsidian-research/可塑性喪失/測定/入力を中心化したleakyCNN_結果_1010.md
- /home/issan/Projects/obsidian-research/可塑性喪失/測定/CNN自己成分と実効入力方向_上昇収縮と重み蓄積_事後_1011.md
- /home/issan/Projects/obsidian-research/可塑性喪失/測定/CNNの最終隠れ層で装填を測る_κの再整列・切替の押し・課題の中の戻り_結果_1020.md
- /home/issan/Projects/obsidian-research/可塑性喪失/測定/実ラベル5p1CNN_課題あたり更新数の延長_結果_1019.md

現状: 0装填、1平均/bias、2向きと戻りは既に測定/介入あり。4はf2の上流一次絶対寄与97–100%とc1の75更新cotangent分解あり、f2の上流self/otherは未。大きな残りは3CNN capと5CNNの初期特徴/予測一致を保つ応答場操作。新しい箱を探索し直さない。

案A runid cnn_response_center_1021:
- RL-CIFAR leaky.1、固定1200画像、16/16/100/100 CNN、batch16、Adam1e-3、30000更新/課題、seed10–19。
- 同時点t30のLR（生入力）とLRc（CIFAR trainチャネル平均を引くだけ）の完全状態が実在。LRcとLRは初期化/画像/ラベル/batchstream同じ。LRのt30online .889,c1 p=.01,rank5.9、LRc .943,p=.10,rank13。c2/fcゲートはほぼ同じ。
- c1のみ、LR self / LR←LRc / LRc self / LRc←LRの4腕。全腕共通task31ラベル/順序1課題、受け手のAdam全状態保持、上流/後段/入力変換は受け手固有。場donorの入力変換はdonor固有。
- 凍結R0,D0から、a_tilde=a_R0+[phi(z_D0+(z_Rtheta-z_R0))-phi(z_D0)]。分岐時deltaを先に計算して0、全特徴/logit/損失/MaxPool winner一致。介入は局所phi'場であってdonorの全Jacobianではない。各受け手にself-shamとnaturalの一致検査。参照状態へ勾配は流さない。
- MaxPool直前に適用。初期backpropはreceiver winner上のdonor phi'を見るので、全空間とreceiver-winner上の平均phi'/分散/pを両方記録。全チャネル固定、結果を見て層やunitを選ばない。
- 主判定: task31の最初3000更新online精度でLR←LRc改善、LRc←LR悪化を10seed対応差で判定。全30000と最後3000も副指標、成功窓の選び直しなし。主2方向Holm補正両側sign test .05、平均差/seed差/CIも出す。両方向支持のみBIDIRECTIONAL。
- 倍率対照を足す価値は？救済方向だけ、receiver winner上でdonorの平均phi'と一致するchannel別cを固定し、a=a_R0+c_j[phi(z_Rtheta)-phi(z_R0)]。平均傾きだけ同じでgate配置は自然のまま。4腕主にこの1腕を加える案。これでも片側化媒介の完全同定ではない点は明示。

案B runid cnn_width_cap_1021:
- 実ラベルCIFAR1005+1、同じ16/16/100/100 CNN、ReLU/ELU(alpha3.6)、lr3e-4、seed2000–2009、各課題7800更新、30課題。課題1末の完全checkpointあり。task2–30をcap/naturalで再生（2×2）、予防を測る。
- 全4siteのunit/channel別前活性幅を制御（行ノルムcapと区別）。参照s_refはt1固定probe2000枚の幅、convは画像×空間をpoolしたSD（within/betweenも診断）。Adam直後に上流→下流順で現batchのbias-free前活性分散vを測り、Vtmp=.99 Vprev+.01 v、q=min(1,s_ref/sqrt(Vtmp))、W行をq倍、Vnew=q²Vtmp。bias/head/Adamは直接変更しない。q>1なし、zero-width安全分岐。これはEMA幅の制御、現在の全probe幅の厳密上限保証ではない。介入後の実幅超過を常時計測。
- Wのみ縮めるのでm'=q(m-b)+b、幅だけの純粋な媒介ではない。全4層かけるので下流入力も動く。naturalも同じprobe診断を行いRNG/EMA状態を変えない。
- 主: latehard21/23/25/27/29の最初780更新online差。副:最初100/all7800/taskendtest/freshとの差、m,s,r,phi',dead,rank、cap頻度と超過量。ReLUは強LoP（18.88% first780,dead.94）、ELUは応答低下（.65→.27）してもfreshより良いので崩壊救済と呼ばない。
- ReLUにcapが無効と決めつけない（文字GPT2blockcap10seedは救済）。Bは幅制御が成績/応答をどう動かすかを両活性化で試す。

回答してほしいこと: (1) この2本の優先順と主張上の十分さ (2) 最大の設計欠陥/省ける過剰設計 (3) 平均倍率対照を最初から1腕入れるか (4) cap実装をどこまで簡素にすれば問いを外さないか (5) 各方向の予測と反証時の読み。原稿用の中心文1つも。日本語で結論先、2000字程度。網羅的な追加実験リストは要らない。完了条件は具体的な修正勧告を返すことです。
