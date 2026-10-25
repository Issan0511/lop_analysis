# ViT: 相1の整列介入と相2の適用条件を検証（1025）

状態: 前回の層別診断を知った上での探索的追試。Issa「両方検証して」。この方法と判定を新しい結果を見る前にcommitする。対象はSL全5 seed、task10/60末の10保存状態、6 FFN。別々に学習したSL/R0Lの比較を同一状態の介入で補う。

## 共通

元の重み・AdamをtarのSHAで照合し、元sourceのhashを照合する。旧画像250枚、元手順で生成する次課題11/61の250画像、固定validation64枚のCLSを使う。61は新しい仮想次課題で未学習。r=mu/(不偏SD+1e-12)、有効SD>1e-8。全パラメータ経路でS=J_r g、正が沈降。SGDと保存Adam分母固定（momentumなし）を分ける。各seedでunit平均・中央値・沈む割合を作り、seed中央値と範囲を報告する。

## A: 整列を同一状態で操作

前回と同じclass-centered head Vc、F=(Qold-u)^T Hold、L=kappa F、R=Vc-L。biasとclass平均は保持。headを V(alpha)=R+alpha L、alpha=0,.5,1 に実際に置き換え、logits/probabilityも再計算する。さらに (i) alpha=0を元Vcと同じFrobenius normへ拡大、(ii) 旧ラベルの画像間permutationから作ったFをRとLの両方に直交化し、Lと同じnormにした4本でLを置換する。後者はRを保持し、head normを変えず、旧ラベルとの整列を除く対照。乱数は1025025+seed*1000+task*10+対照番号。

各headの次課題画像に対する独立一様ラベル期待CEを逆伝播する。head zeroで勾配が消えるという自明な対照だけを成功基準にしない。alpha=1の固定probability下のL/R分解は前回との再現・検算として別に残す。これとprobabilityを再計算する介入は区別する。

保存Adam状態から、250画像の期待勾配を一度入れた有限の1 stepも各介入で計算する。保存momentum・二次moment・step数・lrを使う。これは期待勾配に対する全batch stepであり、元のminibatch Adam stepの期待値ではない。bodyのrの変化を直接測り、JVP方向と比較する。元checkpointを上書きしない。

事前の記述判定A（task別・層別）: 元headで全unit平均S_Adam>0、かつ4整列除去対照の平均よりS_Adamが大きい、の両方が4/5 seed以上なら LOCAL_ALIGNMENT_EFFECT。有限stepでも元headのunit平均Δrが対照より負のseed数を併記する。alpha1,.5,0の単調性・norm調整の有無・単なるhead縮小で説明できる可能性を別々に報告する。必要性・長期の媒介率へ拡張しない。

## B: 相2の共同fitと実CEの違い

前回のP2=J^T[H(H^T H)^+]は切替期待の条件付き模型。通常学習にそのまま適用して外れたことは相2の反証にはしない。今回はbiasを含むHbar=[H,1]とclass-centered W=[V^T;b^T]、Y=C(Q-u)、C=10を用いる。旧HbarのSVDでW*=Hbar^+Y、Gplus=(Hbar^T Hbar)^+を作る。閾値s²>1e-8*smax²、1e-6/1e-10も照合する。

旧入力について、body特徴への信号を次の通り厳密に分解する（W_bodyはbias行を除く）:

- actual_fit=(p-Q) W_body^T/N
- actual_switch=(p-u) W_body^T/N
- sq_fit=(Hbar W-Y) W_body^T/(CN)
- sq_switch=(Hbar W) W_body^T/(CN)
- ls_fit=(Hbar W*-Y) W*_body^T/(CN)
- ls_switch=(Hbar W*) W*_body^T/(CN)
- fit_gap=sq_fit-ls_fit、switch_gap=sq_switch-ls_switch
- ce=actual_fit-sq_fit=actual_switch-sq_switch

従ってactual_fit=ls_fit+fit_gap+ce、actual_switch=ls_switch+switch_gap+ce。帳簿の恒等性は機構の証拠に数えない。旧課題のiidラベル平均の理論切替信号はbeta Hbar Gplus_body/(CN)、beta=C(C-1)=90。次課題画像の予測でも**旧課題のGplusを使う**。次画像からGramを作り直す前回の形は、旧readoutの共同fitを運ぶ予測とは異なる。新旧入力で検査し、等方head模型HnewとGram（逆でない）模型Hbar_new Goldを同じ経路の対照にする。

通常学習では、固定Hbar上のiid旧ラベル平均は E[(Hbar W*-Y)W*_body^T]=0 となる。これは相2が実学習に無いという命題ではなく、この条件付き模型から通常学習の一方向の平均を導けないことを意味する。個別課題のls_fit、fit_gap、CE補正を測って、どの仮定が実状態と離れているか示す。特徴とラベルが共同に学習された依存性は、この凍結模型では証明できない。

正の対照: 凍結Hbar上で独立一様ラベル256組を生成し共同LSでfitする。旧/次入力の二乗損失・切替信号の平均を理論と比べる。前半/後半128組も保存する。実網との一致の代用にしない。理論の厳密なcovariance恒等式を別に検算する。W*へ実headを置換した実CEの通常学習勾配も測り、二乗損失模型の結果と区別する。

事前の記述判定B（task別・層別）: 次課題切替期待と相2のBA>=.65、Spearman>=.5、等方head対照よりBA>=.05高い、を4/5 seed以上なら P2_SWITCH_DIRECTION_MATCH。押しの大きさ、課題ごとのLS偏差、head fit誤差、CE補正を併記し、方向一致だけで共同fit成立としない。比較母集団は実測非ゼロunit。予測ゼロは符号不一致に数える。片方の符号が10unit未満ならBA未定義。

## C: 切替なしの通常学習を実更新で確認

同じ10保存状態から元Adamを再開し、(i) 旧画像・旧ラベルのまま50更新、(ii) 同じ旧画像を新しい独立ラベルへ替えて50更新、を対にする。head/bodyとも学習する。batch128、lr1e-4、augmentation/dropoutなし、両腕で同じbatch順。batch順は1025125+seed*100+task、relabelは1025225+seed*100+taskの独立stream。元走の完全再生ではなく、保存末状態からの継続介入。0/1/5/20/50更新で固定probeのrとlossを測る。切替なしでも前段が沈むか、最終段は戻るかを直接測る。全unitのpairedΔrを集約し、4/5 seedの符号一致を記述する。これだけで既存の相2機構を確定しない。

## gateと保存

source/checkpoint SHA、元CE/精度の再現、head norm/直交制約、LS正規方程式、gradient/位置の分解恒等式を検算。float64のhead代数は1e-8、float32逆伝播/JVPは合計スケールに対して2e-3。有限Adam式は実torch.optim.Adamとseed0/task10で照合。前回のbaseline次課題方向を再現。失敗したら結果を解釈せず修正と再実行の記録を残す。

コード/spec/resultsをgit、生配列・ログを `/home/issan/Projects/obsidian-research-data/vit_mechanisms_1025/` へ保存。元データと他の走を変更しない。終了後mainへmerge/pushし、このworktree/branchだけ片付ける。

## 追補D: Issaの「入力データのばらつき」仮説（新結果閲覧前）

Issaの追加発言を受けた検査。A/Bの10状態計算は完了ログだけ確認済みで、数値はまだ未閲覧。元の登録判定は変更しない。

画像ごとに違う入力依存Jacobian J_nと、headから戻る誤差信号e_n=(p_n-q_n)Vcの対応を測る。g_fit=E[J_n^T e_n]、g_mean=E[J_n^T]E[e_n]、g_cov=g_fit-g_mean。同じbody・logits・元の各画像の信号を保ち、backwardの信号だけ全画像平均へ替えることで共分散を除く。4本の画像間permutationも使い、信号の周辺分布を変えず対応を崩す対照にする。乱数は1025425+seed*1000+task*10+番号。これは「入力分散そのもの」の介入ではなく、入力依存の微分経路と誤差の相関への介入である。

SL全5seed、task10/60のupdate100/500、計20状態。SGDは全状態、保存Adam分母と有限Adam stepは末の10状態だけ。押しは全body経路を固定probeのrへ運ぶ。pred_cov=Cov(J,pVc)、target_cov=Cov(J,qVc)も測り、cov=pred_cov-target_covを検算する。大きな二項の差が実勾配になる可能性を残す。

各層・窓でmean/covの平均S、RMS、実fitとの方向一致を記録。記述条件D: fitとcovのunitベクトルの相対L2差<=.2かつBA>=.8のseedが4/5以上なら COVARIANCE_CARRIES_FIT。これは画像依存相関が押しを運ぶという判定で、ばらつきだけが沈降の符号を決める証明ではない。mean信号・4permutation信号を与えた有限1 stepも対にする。通常学習のfitが沈むunitのうち、mean信号へ替えると沈む方向が消える割合を併記する。

さらにCと同じ10起点から、旧ラベルを保った全250画像のfull-batch Adamを50更新する。50minibatch更新と更新回数を揃え、入力を平均画像に潰さずbatch抽選ノイズだけを除く。異なるbatch sizeなので画像提示回数は一致しない。これで沈降が残ればbatch抽選ノイズはその継続区間の必要条件でないが、データ間の違いや過去の学習履歴を否定しない。seedごとの方向と100→500の既存区間を分ける。
