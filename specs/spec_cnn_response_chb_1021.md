# CNN CH′対CH′＋B：長期予防で保った全4層の応答場を交換する

実験ID: `cnn_response_chb_1021` / 2026-10-21

登録状態: REGISTERED

## 目的と前の走との関係

Issaの1021の指示「CNNもCHとCHBでCH落ちたやん／明らかにこれは設定が悪いので回し直ししてください」を受けた再設計である。

前の`cnn_response_center_1021`は、LeakyReLUの生入力対入力中心化だけの履歴を使い、c1だけを交換した。一方向の初期改善と終盤の不利益という結果は保持する。しかし、既に救済が確認されているCNNのCH′対CH′＋Bと、その応答の因果効果を接続する設計にはなっていなかった。本走はReLUの同条件CH′対CH′＋Bを使い、後段に未交換の低応答を残さないため全4隠れ層を交換する。活性化・履歴・対象層が違うので前走への追加seedとして合算しない。

## 既知の出発点と全seed

親走`cnn_chprime_bias_1018`の同一精度の再走CHfとCHf_B、課題30末、seed10–19の完全状態を使う。親の後期21–30オンライン平均49.518%対94.155%、全10seedでB追加が改善したことは既知である。CHfの全seedが崩壊したわけではない。元のP183はTF32が異なるので使わない。

保存先は`/home/issan/Projects/obsidian-research-data/cnn_chprime_bias_1018/host_gpu_recovery_1018/results/cnn_chprime_bias_1018/`。`state.pt`のSHA256はCHf=`936b3e285a979f97097234dd76c816fa24417e773f5ce2efbf19c6f6c8d0c4c3`、CHf_B=`6dc78598f3f7407eebb80b35077eada4a543dc66380d60e2c69e25335a4428a1`。元の退避manifestの対応するstate・10個の課題30末checkpoint・CSV・provenanceのhashと照合する。完全状態のtask=30、tc=900000、パラメータ・Adam・EMA・保存generatorの対応も検査する。

既存状態の監査は既知の親走の確認であり、新しい介入結果ではない。新しい介入を走らせるのはspec/codeのcommitとpush後に限る。seed除外・条件探索・窓変更で双方向の成立を追わない。

## 固定する箱

1200枚の固定CIFAR-10訓練画像、毎課題独立の一様乱数ラベル。conv5×5(3→16)→ReLU→pool2を2段、FC100→ReLUを2段、head10、全層bias。入力は訓練50000枚の固定チャネル平均を引く。隠れ出力の各scalar特徴をEMAで中心化するCH′（convはpool後のチャネル×位置、FCはunit）。beta=.01、初期化・ラベル・batch列・Adamは親走から継承する。

新課題31を1課題、400epoch×75=30000更新、batch16、Adam lr=.001、beta=(.9,.999)、eps=1e-8、WDなし。seed10–19、固定順10slot bundle。CUDA matmul/cuDNN TF32はともにfalse、deterministic=true、cuDNN benchmark=false。親と同じtorch2.13.0+cu130を使う。実行環境と全依存コードのhashを保存する。

## 4腕と続行する処方

| 腕 | 受け手 | 応答場の参照 | Bの続行 |
|---|---|---|---|
| CH_self | CHf t30 | CHf t30 | なし |
| CH_swap | CHf t30 | CHf_B t30 | なし |
| CHB_self | CHf_B t30 | CHf_B t30 | あり |
| CHB_swap | CHf_B t30 | CHf t30 | あり |

全腕で受け手の入力、全重み、Adamのm/v/tc、EMA中心化状態、同じ次課題のラベルとbatch順を維持する。CHBのbias射影は元のκ=.2で継続する。同じ受け手のself/swap間で処方は同一である。B継続と停止の比較は今回追加しない。

## 全層の人工関数と中心化

各自の自然forwardで固定参照`z_R0,l(x)`と`z_D0,l(x)`を作る。画像global ID、同indexチャネル、空間位置を対応させ、同じ意味の特徴に整列したとは主張しない。学習中の受け手の当該層前活性を`z_theta,l`とし、全4隠れ層で

`delta=z_theta,l-z_R0,l`

`z_eff=z_D0,l+delta`

`a_l=phi(z_R0,l)+[phi(z_eff)-phi(z_D0,l)]`

とする。phiは親の`torch.clamp(z,min=0)`（ゼロのnative微分1）。参照に勾配を流さない。deltaが0なら固定活動をそのまま返すAnchorAddを使う。供与者が大尺度で正側にある場合の差分相殺を避けるため、角括弧は次の等価な安定演算で計算する：`z_D0>=0`かつ`z_eff>=0`ならdelta、`z_D0>=0`かつ`z_eff<0`なら`-z_D0`、`z_D0<0`なら`clamp(z_eff,min=0)`。ゼロの微分はnativeと照合する。

参照raw zと通常のネット計算は親のfloat32を保ち、上のdelta・実効座標・安定差分・AnchorAddの和だけは**全4腕共通でfloat64**として、人工活動をfloat32へ戻す。自己場だけを自然ReLUへ迂回させない。登録前の合成75更新検査では全float32の差分再構成がCHBで丸めを累積させたため、この演算を採った。float64化後は同じ固定許容差を満たした。実効座標の診断はfloat64、微分ゲートは元の学習dtypeへ戻す。これは実データの介入結果を見る前の数値実装の確定である。

開始時には各層の全活動、pool winner、中心化後特徴、logits、損失を受け手自然と一致させる。応答は供与者参照に受け手のdeltaを足した位置で動く。全Jacobianの交換ではない。受け手の入力と重みのJacobians、EMAとBの処方は受け手に残る。

EMAは**学習forwardで実際に使った人工活動**から更新する。convは人工活動のpool後を画像平均、FCは人工活動の画像平均。毎Adam更新後、親の順序で`.99*m+.01*batch_mean`を行う。raw zへnative ReLUを掛け直した活動で代用しない。幅Vはraw zの記録であり学習率の制御に使わない。selfも同じ演算を通す。

CHBの投影はAdam→EMAの後、同一batch上のbiasなしq=W hの画像軸母分散から幅を計算し、biasを±.2sへclipする。conv幅は各位置の画像軸分散を位置平均して平方根を取る。c1→c2→f1→f2の順に修正biasと人工活動、更新後の受け手EMAから次層入力を作る。head・W・Adamは直接変更しない。Bの幅を実効zから作らない。

## 主判定と副指標

主指標は更新1–3000の更新前online accuracy。救済差=`CH_swap−CH_self`、逆差（悪化量）=`CHB_self−CHB_swap`。全seed10–19を含め、各差の両側exact sign test（厳密0のみtie）、2検定Holm、family alpha=.05。平均差と多数派がともに正かつ補正有意で支持。両方向支持=BIDIRECTIONAL、一方向=ONE_DIRECTION、なし=NONE。有意な逆方向はOPPOSITE、平均と多数派不一致はHETEROGENEOUSを優先する。前走と同じ判定規則を使う。seed bootstrap percentile CIを20000回、seed1021で併記する。

副：全30000更新・最後3000更新・更新1001–3000の精度/CE、固定集合のmemo、各層のraw/effectiveの位置・幅・r・開口率・native平均微分/分散・厳密ゼロ・p<.005率、pool winner上の実効応答、活動effective rank、bias/Wノルム/Adam分母、B発火率を保存する。更新0/100/750/3000/15000/30000で全unitを記録する。更新の相対量として層別`||ΔW||/||W||`、`RMS(Δb)`、Adam分母の代表値、勾配をAdam分母で割った量を保存する。現在の実効ゲートと初期供与者/受け手ゲートの一致率、deltaと固定オフセットの尺度も記録する。1001–3000はAdamの初期過渡を除いた感度記述で、独立した主検定ではない。これらは診断で主判定を救済しない。

非有限値・seed失敗は除外せず、以後accuracy0/CE+∞を適用し全seedの主結果に残す。中断は保存した完全epoch状態から同じidentityで再開し、成功seedのみの再走はしない。既に成立している自然履歴の差で介入差を割って媒介率としない。全4層のどこが効いたかは同定しない。

## 実行前検査

合成検査：全4層の開始時forward一致、供与者微分・finite difference、zero微分1、自己場、画像/チャネル/層を誤らせる変異検出、安定差分の大尺度例、実活動EMA、Bの独立参照と層順、保存/復元、診断不変を確認する。

実GPU検査：全source/checkpoint/hash/stream、全画像の開始一致とpool winner、自然/selfの開始勾配・1step（EMA/B含む）と75stepの丸め差、graph/eagerの全状態、captureによる巻戻し、同状態からの反復、診断がP/Adam/EMA/RNGを変更しないことを確認する。元の自然網は親CF.forwardと元のB演算で照合する。元評価表の再現は元chunk20で行い、memo差≤1/1200、他の数値はabs差/max(1,abs旧値)<1e-4を要求し、bit一致と最大差も保存する。基準不合格を成績に合わせて緩めず原因を修正して再検査する。

実測前に固定する許容差：開始活動・pool winner・logits・損失、capture巻戻し、診断不変、同状態反復、保存復元は完全一致。自然/selfの開始勾配は`atol=1e-7, rtol=1e-6`、1step/75stepおよびgraph/eager全状態は各要素`abs差/max(1,abs参照)<1e-4`。最大差と完全一致の有無も併記する。ReLUの未使用fixAと旧診断のalpha/seat等に保存済みのNaNは対応するNaN同士を一致として扱い、P/Adam/EMAや新しい学習の非有限値と混同しない。評価のtask30ラベルは保存label generatorと同一の30回の生成を復元して検証し、task31の新ラベルとは分ける。

CUDA graphはPyTorch2.13の既存走で必要だった`set_override_stale_capture_stream(True)`をプロセス内だけ有効にし、その設定を記録する。共有GPU lockと出力lockを使う。他のGPU実験と並走せず、空くまで待つ。

## 予測と結論の範囲

Codexの事前予測はBIDIRECTIONAL。ただし全4層の応答を下げても、残る特徴と出力head等により逆方向が小さい可能性がある。健康場を与えても尺度・Adam・平均経路の履歴は戻らず、救済が不十分または不利益となり得る。1課題の操作から長期維持や自然な媒介を断定しない。登録前から既知のCH′＋Bの利益を否定/再証明する走ではなく、その保存状態の応答への介入を検証する走である。

操作するのはReLUの局所微分ゲートであり、pool選択・重み・全Jacobianの交換ではない。ゲートの密度と配置の寄与も分離しない。Adamを保持するため、効果には新しい勾配と受け手の適応状態の相互作用が含まれる。更新量の診断を加えても、それだけで実効ステップ幅の因果効果を分離したとはしない。人工活動は負になり得る。後半の結果には参照からの変位、固定オフセット、中心化とBの適応を含む人工網の学習が反映される。

設計相談は登録前に1回実施したFable (`claude-fable-5-1`) の抽象設計レビューを参照する。全層化、B継続、実活動EMA、Adam過渡・ゲート移動の記録を採用し、片側検定への変更案は採用しない。相談結果と採否は`analysis/cnn_response_chb_1021/design_review.md`に保存する。

## 保存と完了条件

`results/cnn_response_chb_1021/`にsource manifest、preflight、各腕の完全状態・全seed曲線・診断・provenanceと主集計を保存する。主集計は全4腕10seed、予定30000更新の完了または登録された非有限失敗規則を満たしてから行い、曲線から独立再計算する。中断を完了として扱わない。終了コード・条件/コードhash・欠損・失敗を報告する。

結果はVaultへ旧結果を保って記録し、V13の部品5/1への接続と限界を更新する。生データをSHA256/bytes manifest付きで退避し、結果/spec/codeをmainへmerge/pushしてから、この実験のworktree/branchだけを片付ける。既存のcap実験については監視と完了報告の担当範囲を維持し、そのworktreeを変更・片付けしない。
