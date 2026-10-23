# 実ラベル5+1 CNNの幅の上限：cnn_width_cap_1021

状態：**REGISTERED仕様（2026-10-21）— 凍結commit/pushと親のRELEASE後に実データのforward・再生・学習を行う。登録commitは各出力のprovenanceに記録する。**

## 1. 問いと位置づけ

CNNにおいて、前活性幅の成長を上限で制御する介入が後期の学習速度を変えるかを検証する。既存の課題予算延長走で幅が増大した実ラベル5+1 CIFAR-100、予算7,800更新/課題を使う。ReLUとELU（α=3.6）をそれぞれnatural/capで比較する。幅の増大が大きな失敗に必須であるとの一般命題、あるいはReLUではcapが必ず無効との予測は置かない。

ELUは既存走で新品より成績が低くない。ELUに改善が出ても「崩壊を救済した」とは呼ばず、後期の学習速度・応答への介入効果とする。全4層への介入なので、特定の層だけが効果を担ったとは断定しない。

capで改善しない、または差が非有意でも、それだけで「幅は原因でない」とは結論しない。EMAの制御が実際にどの層・時期で幅を抑えたか、応答・性能の変化、上流入力の変化と介入の範囲を併せて読む。

## 2. 固定条件と出発点

- 既存 `cifar5p1_cnn_budget_1019` のb7800、seed 2000–2009、R_lr0.0003 / ELU_a3.6_lr0.0003。
- conv5×5 3→16→MaxPool2、conv5×5 16→16→MaxPool2、FC1024→100→100→100出力。4活性化site c1/c2/f1/f2、全affineにbias、BN/LNなし。
- 30課題、hard/easy交互、hardは5クラス2,500枚、easyは1クラス500枚。batch32、各課題7,800更新。固定正規化、augmentationなし。
- Adam lr3e-4、β1=.9、β2=.999、ε=1e-8、float32、AMP/TF32なし、deterministic、cuDNN benchmark off。
- 元の課題1末完全checkpoint（weights・Adam・activation状態・更新数）から課題2–30を再生する。natural/capは同じ状態と同じbatch列から出発し、task/head/optimizerのresetを加えない。
- 元checkpointは40slot束。各腕10seedを抽出し、新走は同じ20slot構成のnaturalとcapを別々に計算する。束サイズに依存する数値差は再現検査で評価する。過去のnatural成績を新capの対照に流用しない。
- 保存元の絶対パス、ファイルSHA256、元meta、selected indices、データtar SHA256、batch/probe SHA256、コードcommit、spec SHA256、precision/runtime/GPUをprovenanceに記録する。

## 3. 介入の厳密な定義

元40slotのb00から課題1の7,800更新を無介入で再生し、各更新のAdam直後、同じbatch上の前活性分散のEMAを測る。初回のpost-Adam分散でEMAを初期化し、以後 `.99 V+.01 v`。課題1末の `V_ref` と `s_ref=sqrt(V_ref)` を参照とする。b01のP/m/v/Adam step/activation状態/更新数が元保存状態とbit一致することを必須とし、不一致なら参照を採用せず停止する。元40slotから参照を20slotへ抽出する。

参照は課題1のhardデータ上の量であり、hard/easy間の入力分布の切替もcapの作動・幅超過を変え得る。発火・超過量にはtask_typeを明示し、長期の幅成長と課題分布の変化を混同しない。

convは画像×空間のpooled分散、FCは画像方向の分散。母分散（correction=0）を使う。biasはunit/channel内で一定なので、前活性とbiasを除いた `q=W h` の分散は数学的には等しい。固定probe（既存の各クラス20枚、合計2,000枚）の課題1末幅もpooled/within/betweenを区別して診断として保存するが、capの参照には使わない。

EMAは保存された `V_ref` から開始する（sqrt後の再二乗による丸めで置き換えない）。各Adam更新の直後、同じ学習batchを用いてc1→c2→f1→f2の順に、各siteで以下を行う。

1. すでに上流siteにcapを適用した入力から、そのsiteのaffineを計算する。
2. `v=Var(q)`、`V_tmp=.99 V + .01 v`。
3. `a=min(1, s_ref/sqrt(V_tmp))`。`V_tmp=0`ではa=1。`s_ref=0,V_tmp>0`ではa=0。正の幅の任意floor/epsilonは加えない。
4. cap腕だけそのunit/channelのW行とbiasをともにa倍し、`V=a² V_tmp`。natural腕はa=1としてW/bを変更せず、`V=V_tmp`。
5. 同じ入力でaffineを再計算し、実際の介入後のbatch幅を測定し、活性化・poolを通して次siteへ渡す。

隠れ層W行と対応biasの同率縮小以外は行わない。head（W/b）、Adamの1次/2次モーメント、Adam stepを変更しない。自然腕も同じ診断を実行する。介入用EMAはactivation内の既存診断EMAと別に持つ。

局所的に入力を固定したその投影の瞬間は `m'=a m, s'=a s`、a>0かつs>0ならrは不変である。上流も縮小する全4層介入なので、次の層の入力分布やその後の学習軌道は変わる。capの性能差だけから全網の純粋な幅の媒介効果とは結論しない。また上限制御はEMAへのもので、毎時刻の固定probe幅や現在batch幅を厳密に上限以下にするものではない。

## 4. 記録と評価

主評価はlate hard tasks 21,23,25,27,29の最初780更新のonline accuracy。各taskでpre-updateの同じforwardのbatch accuracyを平均し、その5task平均をseedの1標本とする。

- ReLU、ELUそれぞれの対応差 `cap-natural` 10seedを報告する。
- 0差を除いた両側exact sign testを行い、2検定にHolm補正、family α=.05。全差・平均差・中央値差・正/負/0の個数も報告する。
- 改善/悪化は補正p<.05かつ多数seedの符号と平均差の符号が同じ場合に限る。符号検定と平均差が逆向きなら異質性ありとして両方を示し、単一の改善/悪化判定を出さない。その他は未確定。非有意を同等としない。
- 有限でない走は除外しない。発生位置と全seedの生結果を示し、その腕の主検定は不成立とする。

副評価：first100、全7,800更新online、task末test、CE。各腕自身のearlyhard（3,5,7,9）→latehard（21,23,25,27,29）のfirst780変化もseed別に報告し、cap/naturalの後期水準差と時間低下を分ける。既存同seed新品との差は精度設定・コード・再現を確認できた場合だけ参考値として別記し、新controlと混ぜない。

課題1末と各課題末の固定probe、各課題の学習画像の開始/末で、全4siteのunit別m,s,r,φ′平均/標準偏差、p(z>0)、dmax、dead(dmax<1e-6)、low-response(dmean<.1)、zero-output、effective rankを保存する。convはwithin/between幅も残す。rの分母ゼロは未定義として数を報告する。

介入チェック：全4siteの参照幅をunit別に保存し、初期幅ゼロunit数を各seed/siteで報告する。convのpooled幅・within幅・between幅の量名を維持する。unit別cap適用回数・比の平均/最小、EMA幅、当該batchの介入前/後幅、参照幅超過の回数と最大比を記録。固定probe幅とEMA幅を分ける。task30の固定probe幅のseed内unit中央値をnatural/capで対比し、幅が下がったかを性能差と独立に報告する。

## 5. 実行前の検査と停止規則

実データ検査・再生はspec凍結後に行う。

1. synthetic：pooled/channel/FC分散を独立のNumPy計算と照合。W/b同率縮小で幅・平均がa倍、定義されたrが不変となることを検査。
2. synthetic：順次prefixが上流cap後の入力を下流に渡すこと。head/Adam状態不変、naturalが元engineの更新と一致。EMA初回batch初期化、capture rewind、restart、途中chunk再開、RNG/activation状態を検査。
3. mutation：軸誤り、biasを縮め忘れること、Adamモーメント縮小、下流を古い入力で計算、EMA補正なしを検出する。
4. 読み出したcheckpointのslot順・shape/dtype・batch/probe hashと元メタを検査。実データ診断が訓練状態を変更しないこと。
5. 元40slotで保存boundaryから短い再生を行い、保存された更新損失とのbit一致/差を記録。新20slot naturalでも差を記録する。元hostと同条件の再現ができない場合は精度・engine・束の差を調べ、失敗を明記して親に報告する。古い成績を同じ対照扱いしない。
6. natural/capの同20slotで、cap無効shamは自然と一致し、cap開始前のP/m/v/act等は一致する。

バグ、非有限値、メタ不一致は停止して原因を記録。数値不調を理由にseedを替えたり黙って再走しない。仕様変更は結果を見る前後を区別した追補をcommitし、元記録を残す。

## 6. 保存・再開・終了

- 出力 `results/cnn_width_cap_1021/{natural,cap}/`。per_task.csv、probe_stats.csv、train_stats.csv、cap_stats.csv、steps/*.npz、units/*.npz、state.pt、checkpoints、provenance.jsonを保存する。
- 途中chunkおよびtask境界から再開できる完全stateをatomicに保存。P/m/v/step/activation/cap EMA/参照幅/途中記録/idx/j/全CPU・CUDA・NumPy・Python RNGを含める。
- 全GPUコマンドで `/tmp/lop_analysis_gpu.lock`、各出力で排他lockを取る。競合なら実行せず終了する。resumeのmetaにコードhash・runtime・precisionを含め、不一致を拒否する。初回環境記録を保持し、仕様変更を通常resumeとして扱わない。
- すべての10seedを完了してから主判定を出す。途中成績を見て腕・閾値・seed・予算を選び直さない。
- 集計は両腕COMPLETED、task2–30全件×2活性化×10seedの重複なし、bad_stepが全-1、全評価が有限であることを必須とする。STOPPEDや後半の非有限値があれば、late taskのCSVが揃っていても主判定未成立と記録する。
- 生データを `obsidian-research-data/cnn_width_cap_1021/` へSHA256 manifest付きで退避し、spec/code/summary/manifestをcommit。作業は指定worktreeに限り、main checkoutや他セッションのworktreeを変更しない。
