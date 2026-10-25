# ViT: 層差を作る更新経路と平均・幅の検証（1025）

状態: 既存の層別結果を知った探索的追試。新しい測定の前にspec・評価コードをcommitする。
親: vit_phases_1025、vit_mechanisms_1025（863b5058まで）。目的はIssaの「前段は通常学習の沈降、後段は切替の沈降」という層差を、実際の更新経路と位置の式で説明すること。既存のjoint-LS模型の成功へ読み替えない。

前のgoal turnはprogress: head介入、共同LSの反証的照合、50更新、入力依存共分散とfull-batch対照を完了・mainへ保存した。ただし層差の機構は未同定。今回の結果が揃っても、未検証の普遍的な深さ則を完成扱いしない。

## 理論から区別するもの

切替の独立一様新ラベル期待はL_switch=log(C)+KL(u||p)、通常学習はL_fit=L_switch−(q−u)^T f。前者は旧出力の確信を弱め、後者は正解マージンを増やす。これは厳密な損失分解だが、各FFNのμ/σの符号を保証しない。

各FFNの前活性z=w^T u+b、固定probe上のm=E[u]、Σ=Cov(u)、μ=w^T m+b、s²=w^TΣw。微分は

dr=(m^Tdw+db+w^Tdm)/s−μ(2w^TΣdw+w^TdΣw)/(2s³)。

自unitのfc1更新、上流の表現変化、平均、幅を分ける。通常学習の共分散が大きいというだけでは、これらの符号までは決まらない。学習済み特徴の条件付きの説明を目指す。

## A. 一次の押しを作るparameter群

SL、5 seed、task10/60末の10状態。旧画像のfit、次課題画像11/61の独立ラベル期待switchと整列成分load。loadは前回と同じ最終分類headの整列であり、各FFNのfc2を分解したものではない。SGDと保存Adam分母固定（momentum除外）。

排他的群: embedding（patch、CLS、位置）、各blockのattention＋norm1、norm2、fc1、fc2、tail（最終LN＋head）。全parameterを過不足なく分類し、各群の独立JVPで全6 FFNのμ・s・rへ運ぶ。全群和を独立な全parameter JVPと照合。対象FFNより後の群の直接寄与0も確認。fc1解析式も照合。局所fc1と上流の和を既存生配列と照合する。

探索的条件付き仮説: 最終FFNの次入力loadについて、上流のr押しのL2比 ||S_up||/(||S_up||+||S_own||)≥.75を4/5 seed以上で満たす。満たさなくても全seed全層を残す。符号付き平均の比は相殺で発散するため主比にしない。群の符号・大きさ・平均項・幅項は全層報告。

## B. 実Adamの50更新を、各stepで分離

前回と同じ10状態、同じ旧画像、旧ラベル継続／独立再ラベルの2腕、同じ乱数とbatch順、50 stepを再現。元Adamを再開、head/bodyを更新。各step直前直後のprobe入力u、fc1重みw,bを取り、z00=w0u0+b0、z10=w1u0+b1、z01=w0u1+b0、z11=w1u1+b1を計算する。

rのlocal/upstream二要因のShapley差を取り、各辺をさらにμと1/(s+ε)の積の対称分解にする。4項（local mean、local width、upstream mean、upstream width）が実Δrへunitごとに厳密に足し合う。各stepで足し、50更新の実変位へ照合。有限差分なのでReLU境界越えを残す。これは実現した経路上の反実仮想置換の帳簿であり、長期の唯一の因果寄与率ではない。

前回のstep1/5/20/50のrを再現する。全unit平均と中央値は別に報告し、加法帳簿はunit平均で示す。原走100→500の区間とは区別。

## C. 上流を固定した学習を実行する

task60末・5 seed、旧ラベル継続／再ラベル、target FFN1と6。各targetにつき(1)そのfc1を固定、(2)その前活性へ影響する上流parameterを全て固定し、そのfc1と後段は学習する。合計40 trajectory、各50更新。baselineはBの同じ状態・ラベル・batch順。

固定parameterはrequires_grad=False、grad=NoneにしてAdam momentumでも動かないようにする。全腕で元Adamから開始。固定対象の重み差ゼロを確認。50更新後のΔrとCEを報告。baselineの平均変位と同符号で、upstream固定時の絶対変位がbaselineの半分以下となるseed数、own固定時にbaseline絶対変位の半分以上が同符号で残るseed数を各条件で報告する。両条件が4/5以上ならTRANSPORT_SUPPORTED、それ以外はNOT_ESTABLISHED。変位が数値ゼロ（|unit平均Δr|≤1e−6）なら判定分母から除かず不成立とする。

## 検算・出力・制約

元モデルsource/checkpoint SHAとCEを照合。全5seed、失敗を除外しない。Aのledgerの最大差は対応する成分の最大絶対値で割って2e−3以下、Bの加法恒等式はfloat64で1e−8以下（絶対許容1e−10）、再現rは最大絶対差2e−5以下。有限の再合成zは元float32推論との最大差2e−5以下。gate seed0/task10を先行し、不一致時は直してから進む。

GPU既存走を停止せずメモリ45%上限。sourceはvit_ffn_ablation_1025/source、モデルは元vit_long_1024 tarからSHA照合して直接読む。数値はresults/vit_transport_1025、rawとログはobsidian-research-data/vit_transport_1025へ直接保存。結果を見た後にこのspecを変更せず、変更が必要なら理由つき追補を別commitにする。
