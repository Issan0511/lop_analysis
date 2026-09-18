# Joudaki ViT activation battle — 2026-09-19

状態: 事前登録。ユーザーは RL-CIFAR MLP 13 活性化比較と同様の比較を Joudaki ViT で依頼。
本走の値は未観測。原典 GELU・合成入力 seed 100 の速度だけ測定済み。
問い: MLP で観測した Snake/kunekune の優位は LN・残差を持つ ViT と実画像のタスク列にも移るか（V10 の1箱限定）。機構の因果同定はしない。

## 原典と選択

原典: https://arxiv.org/html/2510.00304v3 Appendix B.
公式実装 https://github.com/ajoudaki/loss-of-plasticity commit 161217078ba52107c94a16602af958a321d62ce3。
`src/models/vit.py` を無改変で保存し、各 TransformerMLP の act だけ差し替える。
原論文のタスクは Tiny ImageNet 200クラスを重複なし5クラス×40タスク・500更新/タスク。
公開 continual.yaml の既定（5タスク×2クラス、20 epoch）と異なるので論文の40×500更新を採用。
入力64×64、patch8、embed384、depth6、heads6、FFN1536、CLS、pre-LN affineあり、dropout/attention dropout=.1。
Adam lr=1e-4（論文のViT用）、betas=.9/.999、eps=1e-8、WD=0、float32。
Batch128。各epoch最後の少数バッチも使用（2500枚/タスク、20 batch/epoch、25 epoch=500更新）。
元コード同様に各タスクの開始時（t1含む）全200クラスの出力層W,bを0へ。Adam momentは持ち越す。
損失とonline accuracyは現タスク5クラスにmask。偶然水準20%。他タスクの忘却は主指標にしない。
標準化は ImageNet mean(.485,.456,.406),std(.229,.224,.225)。学習は64 crop/pad8、flip .5、brightness/contrast/saturation jitter .1（ランダム順）。評価は拡張なし。
画像クラスはImageFolder同様synset名のソート順。taskクラス順は Python Random(seed).shuffle(range(200))。
原論文の完全な数値再現とは呼ばない（乱数stream、実行環境、実験ごとの未公開overrideが一致しない）。

## 13 腕と適応則

`spec_rlcifar_mlp_battle_0918.md` §3 と追補1の13腕をそのまま使用:
SNA,KKA,KKA23,KKT1,R,LK001,LR,LK03,SL,RSL,ELU,SILU,GELU。
学習する6層のFFN非線形だけ交換。Attention softmax/LN/headはそのまま。
SNA系: c=.6,beta=.01,alpha clip[.005,3],V初期1。各層各FFN channelに1つのVを持つ。
ViTへの拡張: バッチ×全token（CLS含む）を標本軸にし、unbiased=False分散のEMAを学習forward終了後に更新。
forwardは更新前Vからalphaを計算。Vを勾配で更新しない。evalではV不変。
RSLは学習時要素ごとU(.125,.333)、evalは.229。専用生成器を使いdropout/data streamを消費しない。
ELUはF.eluの非inplaceで、出力+1による人工微分を使わない。
同seedの全腕でモデル初期値・class順・batch順・augmentation seed・dropout RNGを対応させる。
seed0–9で本走。検査seed100以上。腕固有tuningなし。

## 判定と出力（結果を見る前に固定）

主指標: 各タスク500更新の更新前online正解数/提示画像数。窓=t21–40の平均。
早期=t1–10、後期=t31–40、低下=早期−後期。val accuracyと最終train accuracyも記述。
崩壊ラベル=窓<.5（元実験の操作的基準を継承、chance=.2）。
seed対応の両側正確符号検定、差0は除外。勝ち=逆方向seed<=1かつp<.05。
主比較はSNA対12腕、KKA−SNA、KKA23−KKA、KKT1−KKA。
元実験のA,K1,K2,K3のラベルを継承。ただし非有意は同等性の証明ではない。
全比較の未補正pとHolm補正pを併記し、未補正順位は探索的と明記。中央値の差も併記。
SNA系が上位に来るかは未知。仮予測: LNによりMLPほど大差はつかない（70%）；KKAとSNAに未補正有意差なし（60%）。
発散: loss/gradient/parameterの非有限で当該走を停止・記録。勝手なlr変更・再初期化で救わない。
不完全な本走はランキング集計を拒否。欠測・発散は数を明示。seedを黙って減らさない。

## 保存・検査・実行

全タスクで全重みと適応Vのsnapshot（float32）、固定評価入力（タスク別16画像）のFFN前活性（float16、overflow時float32）、z統計、実勾配によるdphi統計、位相帯外率を保存。
Snapshotはinitと各task終端。optimizerと全RNGは毎task checkpointに保存し原子的置換で再開可能。
タスク表はcheckpointから復元可能。固定評価画像の識別子を保存。正確なzはsnapshotと入力で再計算。
データと生出力は初めから ~/Projects/obsidian-research-data/joudaki_vit_battle_0919/ 下。
共有datasetは ~/Projects/obsidian-research-data/datasets/tiny-imagenet-200/。コピー・削除時にsymlinkを辿らない。
検査: GELU無改変モデルとのlogits/grad一致、13腕finite/gradient、適応EMA軸とeval凍結、RSL乱数分離、task mask/head reset、checkpoint再開一致、集計の不完全拒否。
本走前に検査と実装をcommit。provenanceにcommit/source sha256/config/env/class順を保存。
合成入力GELU benchmark: 73.55ms/step, peak1.83GB, parameters10,824,008。13×10×20,000stepは学習のみ53.12 GPU時間。実データ・保存で延びる。
原典よりseedを5→10へ増やす理由は元バトルの対応符号検定（5seedでは全勝でも両側p=.0625）を保つため。
本走は1GPU逐次、STOPでtask境界で停止。終了後に集計・図・Obsidian結果ノート・manifestを保存しmain統合。

## 追補1 — ユーザーからの設定比較の依頼（本走開始前）

上のTiny ImageNet案を登録した直後、ユーザーはデータ/タスクの選択に対して「比較してほしいかも 難易度の違いとか」と回答。
したがって上記はTiny ImageNetを採用した場合の候補仕様として保持し、本走開始は保留。
まずRL-CIFARとの課題、学習量、計算量、解釈の違いを比較する。2条件を実測比較した結果はまだ存在しない。
Joudaki ViT単体の合成入力benchmarkだけは両入力形状で測り、性能・可塑性の結果と混同しない。

## 追補2 — Tiny ImageNet採用（実装・本走前）

ユーザー「流石にTinyImageNetかな」によりTiny ImageNet案を採用。追補1の本走保留を解除する。
13腕×seed0–9、40タスク×500更新を採用。CIFAR本走は行わない。
実装は原典モデルを直接使用。画像拡張はPillowで同分布のcrop/flip/ColorJitterを実装し、torchvisionには依存しない。
乱数の実現は原典と異なるが、seedごと画像・batch・拡張は全腕で一致。

## 追補3 — 原典の正解率と保存量の照合（本走前）

原典train_continual.pyは損失を現タスク5クラスへmaskするが、正解率は全200出力のargmaxを使用していた。
当方の登録主指標は5クラス内の正解率を保持し、原典と同じ全200出力の正解率を`online_global_acc` / `train_global_acc` / `val_global_acc`として併記する。
この読み出しの差を隠さず、原論文の図との直接数値一致は主張しない。
全snapshot保存量はモデル約230GB、前活性（圧縮前）約100GB、optimizer付き最終checkpoint約17GB。
空き約550GBのローカルディスクで実行し、25GiB未満ならtask境界で停止する。元データへのsymlinkは作らない。
ユーザーから高速化の依頼を受け、float32/TF32無効のままfused Adamとtorch.compileを速度測定する。
採用するエンジンと検査結果は本走前に追加記録する。

## 追補4 — 高速化エンジン（本走前）

合成入力seed100・batch128の100更新実測（初回warmupを除く）:
GELU eager79.60ms、fused Adam77.98ms、compile+fused61.80ms。
KKA eager139.42ms、compile+fused62.84ms。KKAは2.22倍。
`torch.compile(fullgraph=True)` と fused Adam を全腕同じ設定で採用する方針。
float32・TF32無効を維持。Inductorのfallback_random=TrueでdropoutのATen乱数列を保ち、RSLの私有乱数はcompile外で事前生成する。
数値融合により丸め順は変わるので、長期軌道のbit一致は主張しない。元実装との1更新の出力・勾配・EMAを比較し、同一エンジンでcheckpoint再開がbit一致することを本走前の条件とする。
13腕×10seedの学習時間はこの2腕の速度から約45時間、実データと保存の overhead は別途実測。

## 追補5 — 高速化前検査で見つかった差と修正（本走前）

適応Vをforward内部でin-place更新した版では、KKAのcompile勾配がeagerから最大relative L2 .002589ずれた（GELU/RSLは約2e-6）。
V更新を元MLPと同様optimizer更新の後へ移し、forwardではdetachした分散を保持するだけにしたところ、KKAも最大1.942e-6へ縮小した。
この修正版を使う。forward時のalphaは旧Vから計算し、学習後のVだけを次stepへ持ち越す。

検査の補足: 最初のAdam更新後のqkv biasに相対誤差だけを当てると失敗した。attentionのkey biasの勾配は解析的に0であり、丸めによる微小差をAdamのepsが増幅するため、ゼロ近傍パラメータへの相対誤差は検査として不適切だった。
forward/gradient/EMAは256×float32 epsilonのrelative L2を基準とし、Adam単独は同じ勾配を渡してforeachとfusedを比較する。
異なる丸め勾配による最初のAdam更新差は、g→lr*g/(|g|+eps)のLipschitz定数lr/epsから導いた成分別の上界で検査する。
長期軌道の一致は主張しない。失敗した初期検査もraw/preflightに保持した。

## 追補6 — 本走前の検査と実画像コスト

修正版のGELU/KKA/RSLでforward・gradient・EMAと同一勾配でのfused Adamの数値検査が全件PASS。
KKA/RSLは実画像・本番サイズでtask境界再開を試し、モデル・Adam・RNG・診断・保存前活性が連続実行とbit一致。
13腕の同一初期化/finite backward、class/batch/augmentation対応、task maskとhead reset、集計の不完全拒否もPASS。
seed101・実画像500更新・本番の評価/保存込みの初回タスクはGELU44.52秒、KKA46.84秒（初回shapeコンパイルを含む）。
試走はseed100/101のみで、本走seed0–9の判定値はまだ見ていない。約45時間はsteady-state学習の概算で、保存等を含む実時間は長くなる。
