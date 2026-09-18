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
