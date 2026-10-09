# 任意の学習済み CNN 状態での CE 駆動方向：全 hidden 勾配と自己項の区別

2026-10-09。独立導出・監査メモ。親稿は編集していない。

## 1. 結論と射程

任意の学習済み deep CNN 状態について、**新ラベルが現在状態と独立な一様ラベル**なら、全 hidden パラメータの新 CE 勾配が平均前活性を下げる十分条件を、logit とその平均前活性方向感度のクラス順位・confidence から与えられる。旧 head の初期値、旧 head 学習ステップ数、完全 fit、旧ラベルと hidden 表現の独立性は必要ない。

条件を満たす状態の非空性は、実 RLCIFAR の全構造、全 bias、共有 conv、全 conv channel の重なり、MaxPool の strict winner を保った開集合で示せる。一つ目の構成は 4 hidden 層の任意の channel/unit に同時に使える。二つ目は異なる画像が異なる top class を持ち、同じ構成の literal CE self network と full network が同じ下降方向を持つ。

これは **条件付きの任意時点定理** である。実学習軌道がこの集合に入る・留まること、切替後の長期的な沈降は別の主張であり、ここでは証明しない。元の V12 の logdet/孤立 unit capacity という「自己項」と、以下の CE channel-ablation self は同一ではない。

## 2. 学習済み全状態での正確な恒等式

固定した画像 x_1,...,x_N と、現在の全パラメータ θ（全 conv/FC/bias/head）を考える。θ は任意の旧学習履歴に依存してよい。C≥2 とし、f_n(θ)∈R^C は logits、p_n=softmax(f_n)、u_C=1/C は一様 class ベクトル。新ラベル y_n は、現在の θ と画像を条件付けて E[y_n|θ,x]=u_C を満たす。一様 iid は十分だが、以下の一次期待値だけなら新ラベル間の独立性は不要。

m(θ) を対象 hidden channel/unit のデータ・空間平均前活性とする。Euclidean parameter metric に対して

  u=∇_θ m,   R_n=D_θ f_n[u]∈R^C

を定義する。hidden の m なので head 座標の u は 0 だが、他の全 upstream hidden 座標は省かない。現在点で ReLU の閾値と MaxPool の同率 winner を避けると通常微分が使える。

新 CE の平均 L_new=N^{-1}Σ_n[-y_n·f_n+logΣ_c exp(f_nc)] に対して

  G := E_new[〈∇m,∇L_new〉 | θ,x]
     = (1/N)Σ_n R_n·(p_n-u_C).                         (1)

証明：chain rule ∇L_new=N^{-1}Σ_n Df_n^T(p_n-y_n) を ∇m と内積し、新ラベルだけを条件付き平均する。旧ラベル・表現依存の問題は現れない。

**ここで R は全 θ を通った JVP である。** target kernel block だけの微分でも、f の head-fit map を含む再学習微分でもない。全 hidden を更新する実際の新 CE 勾配と、m の内積である。MaxPool は現在選択された経路の chain rule に既に含まれる。

## 3. クラス順位による非循環な十分条件

任意の確率 p と実ベクトル R には

  R·(p-u_C)
    = (1/C)Σ_{c<d}(p_c-p_d)(R_c-R_d)                  (2)

が成り立つ。右辺を展開すれば CΣ_c p_cR_c-(Σ_cp_c)(Σ_cR_c) となる。softmax は class 順序を保つので、各画像 n と全 class pair に対して

  (f_nc-f_nd)(R_nc-R_nd) ≥ 0                         (3)

なら G≥0。一つでも両差が非零で同符号の pair があれば G>0。

(3) は、求めたい総和 G の符号をそのまま仮定したものではなく、各 class 間の相対増分の構造を指定する十分条件である。多数 class では必要条件でなく、反順序 pair がある場合にも G>0 は起こる。C=2 では唯一の pair の符号条件なので、必然的に全体の符号条件と同値になる。

class-common logit や R 成分は (1)-(3) に影響しない。したがって head の絶対符号の正負ではなく、class contrast が本質になる。

## 4. top class と confidence だけによる開いた条件

全 class pair の順位整合は不要である。画像 n の一意な top class を t とし、d_c=R_t-R_c (c≠t) と置くと

  g_n := R·(p-u_C)
       = Σ_{c≠t}(1/C-p_c)d_c.                        (4)

### 4.1 強い confidence の簡単な条件

  Δ=min_{c≠t}(f_t-f_c)>log(C-1),
  γ=min_{c≠t}(R_t-R_c)>0                             (5)

なら、各 loser は p_c≤(1+exp Δ)^{-1}<1/C なので

  g_n ≥ γ(p_t-1/C)
      ≥ γ{[1+(C-1)exp(-Δ)]^{-1}-1/C} > 0.           (6)

loser 同士の f と R の順序は任意でよい。(5) は厳密不等式だから、微分可能なパラメータ領域内で開条件である。複数画像で top class が違っても構わない。

### 4.2 confidence が低いときの contrast-spread 条件

  a=min_{c≠t}d_c>0,   b=max_{c≠t}d_c,
  B=Σ_{c≠t:p_c>1/C}(p_c-1/C).

すると (4) の正係数には a、負係数には b を使い

  g_n ≥ a(p_t-1/C)-(b-a)B.                            (7)

したがって

  a(p_t-1/C)>(b-a)B                                  (8)

でも厳密に正。特に全 loser に対する R contrast が同じ正値（a=b）なら、top logit が一意であれば十分で、(5) の大きい confidence は不要。(8) も strict top/gate/winner の領域で開条件。

この条件は「最大 class の増え方は他 class より大きいが、loser contrast のばらつきは confidence に比べて小さい」という比較である。最小・最大 contrast と確率質量だけを用い、目的の加重和をそのまま仮定していない。

### 4.3 数量的な摂動余裕

class-common 成分を除いて ||f'-f||₂≤ε_f、||R'-R||₂≤ε_R とする。softmax の Euclidean Lipschitz 定数は 1/2、||p'-u_C||₂≤sqrt(1-1/C) なので

  |g'_n-g_n|≤sqrt(1-1/C) ε_R+(1/2)ε_f||R-centered(R)||₂.

従って上記の正 lower bound が右辺を上回れば、摂動後にも g'_n>0。strict winner/gate を維持する近傍では f,R が連続であり、パラメータ開集合の議論を直接裏付ける。

## 5. 全 hidden GD による平均の減少

現在点の新ラベルから計算した全パラメータ GD を θ^+=θ-η∇L_new とする。

第一 conv の平均前活性 m はパラメータの affine 関数なので

  E[m(θ^+)-m(θ)|θ,x] = -η G                          (9)

が任意の η>0 で正確に成り立つ。これは更新後に MaxPool winner が替わっても m 自体には関係しない。ただし次時点でも G>0 とは言っていない。

深い層の平均前活性では upstream の変化を含むため m は非線形。全新ラベル実現の更新線分上で m が C²、||∇²m||₂≤M、E||∇L_new||²≤Q なら

  E[m(θ^+)-m(θ)|θ,x] ≤ -ηG+(M/2)η²Q.                (10)

G>0 のとき、M Q>0 なら 0<η<2G/(M Q) で負。有限データ・有限 class・strict ReLU/MaxPool margins なら、十分小さい η で共通の微分可能領域に留まるためこの条件は非空である。M Q=0 なら二次項は不要。

ミニバッチの unbiased な Euclidean SGD でも追加平均で一次項は (1)。Adam、momentum、label-dependent preconditioning には (1) をそのまま使わない。正定値で新ラベルと独立な固定 preconditioner P なら R_n=Df_n[P∇m] と置き換える。momentum 項には別の内積項が加わる。

## 6. 全 hidden 層に通用する RLCIFAR の開いた非空 family

構造は

  Conv5(3→16,pad2) → ReLU → MaxPool2
  → Conv5(16→16,pad2) → ReLU → MaxPool2
  → FC1024→100 → ReLU → FC100→100 → ReLU → FC100→10

で、全層の bias を学習可能とする。hidden の重みをすべて strictly positive、入力を非負、hidden preactivation を strictly positive とする。全 MaxPool winner は一意とする。最終 head の class 順序が各 feature 座標で共通で、ある順列に並べると

  V_{c+1,a}-V_{c,a}>0,   b_{c+1}-b_c>0               (11)

とする。rank-one head は要求しない。

各 hidden 層の任意の channel/unit の平均前活性 m に対して ∇θm≥0。理由は、upstream の chain rule に現れる入力、hidden weight、ReLU derivative、選択された MaxPool derivative が非負であり、m の平均係数も正だからである。head と downstream パラメータの ∇m 成分は 0。

同じ理由で最終 hidden H の S_n=DH_n[∇m]≥0。各画像において S_n≠0：対象 bias の ∂m/∂b は 1 で、対象 channel の pool winner もその bias を受け、全 positive downstream weights を通る少なくとも一つの active 経路がある。対象が FC hidden unit でも同様。

よって (11) から f_{n,c+1}>f_{n,c} と

  R_{n,c+1}-R_{n,c}
    = Σ_a(V_{c+1,a}-V_{c,a}) S_{na} >0              (12)

を得る。(3) が全画像・全 pair で strict に成立し G>0。この論法では全 channel が各画像で active、全 hidden convolution channel の入力・下流が重なる。他 unit を除去する操作はない。

strict input/weight/activation/order/pool margins は有限個の連続不等式なので、条件を満たす一状態の周囲に full parameter と input の開集合がある。同一 class を全画像で最上位とする family ではあるが、一つの rank-one point に限定されない。

### 親の full-architecture 数値 witness の独立監査

/tmp/cnn_full_ce_check_1009.py と .json を読み、パラメータ生成、全パラメータ JVP、直接の gradient projection、pairwise 展開を確認した。こちらで torch 実行の再現はしていない。この節の証明は数値の丸めに依存しない。

2 枚の異なる正画像、全 hidden weights positive、非 rank-one 摂動を入れた head に対し：

|確認量|記録値|
|---|---:|
|最小 hidden preactivation|0.1757128031|
|最小 MaxPool winner gap|3.7325226143e-7|
|最小 head 隣接 class 重み差|0.0049551089|
|最小 output bias 隣接 class 差|0.0099225712|
|最小 logit 隣接 class 差|0.3074400073|

各層の channel/unit 0 の m に対する G は順に 1.1150887737, 6.3281273898, 8.1357218087, 6.6689276112。全パラメータの直接 projection、(1)、(2) が丸め誤差内で一致。これは arbitrary specified state の certificate であり、実学習軌道測定ではない。

## 7. 異なる画像が異なる class を予測する厳密な全構造 family

§6 の共通順位に限定されない、(5) を満たす別の構成を与える。以下は full RLCIFAR の全 channel/unit を用い、全 bias を trainable とする。最初に簡単な exact base point を作り、その strict margin から全 support を持つ開集合へ拡張する。

### 7.1 画像と共有 convolution

r,c=0,...,31 として P_{rc}=1+(32r+c)/1024。2 画像は 3 RGB channel とも

  x^{(a)}_{krc}=a P_{rc}/12,   a∈{1,6}.              (13)

各入力 pixel は 0<x<1。Conv1 の全 16 filter は center の RGB 各重みを 4、他の係数を 0、bias を 0 とする。各 channel の z1=aP>0。各 2×2 pool winner は右下、最小 gap は 1/1024。

Conv2 の全 16 filter は center に各 input channel から重み 1/16、他を 0、bias を 0 とする。16 channel 全部を混合する。二度目の pool も右下が一意な winner で、gap は少なくとも 2/1024。最終 8×8 map は元画像の row/col 3,7,...,31 の aP。全 16 channel で同じである。

その P の空間平均は

  c0=1+(32·17+17)/1024=1585/1024.

最終 conv output 1024 座標の平均を c0 で割った scalar h は正確に h=a。

### 7.2 二つの hidden FC と 10-class head

FC1 の全 100 unit は各 conv feature から重み 1/(1024 c0)。前半 50 unit（A group）の bias は 0、後半 50 unit（B group）は -2。この出力は前半 h、後半 t=(h-2)_+。

FC2 の前半 50 unit は全 A input から 1/50、全 B input から 0.1/50、bias 0。後半 50 unit は全 A input から 0.1/50、全 B input から 1/50、bias -0.5。その ReLU 出力は

  A=h+0.1t,    B=(0.1h+t-0.5)_+.                    (14)

従って FC2 の重みは全て正。FC1 の重みも全て正。B group は低画像では inactive、高画像では active であり、両画像を通して全 hidden unit が使用される。

head の全 class に全 FC2 unit から共通重み 1/100 を入れる。追加で class 1 に全 B unit から 6/50、class 2 に全 A unit から 3/50 を入れる。class 1,2 の bias は 0、class 3,...,10 の bias は -3。共通項 (A+B)/2 を除いた logits は

  f=(6B,3A,-3,...,-3).                               (15)

全 head weight は正である。class-common 項を含む実 logits の順位・softmax・CE projection は (15) と同じ。

### 7.3 target mean と full-parameter 感度

m を第一 conv の channel 0 の画像・全空間平均前活性とする。m は affine なので u=∇θm はこの filter と bias にだけ非零。入力の augmented patch ξ_{ns}=(patch_{ns},1) は非負で

  Dz1_{n,0,s}[u]=ξ_{ns}·mean_{n',s'} ξ_{n's'}≥1.     (16)

他の第一 conv filter は u によって直接変わらないが、第二 conv 以降は全 channel が chain rule により変わる。Conv2 の target input 重みは 1/16 だから

  r_h:=Dh[u]≥1/(16c0)=64/1585>0                    (17)

が各画像で成立する。MaxPool は現在の strict winner を通り、選ばれた target z1 も (16) を満たす。第一 conv 以外を freeze して学習する仮定はない。m の upstream が第一 conv だけなので u のその他の成分が数学的に 0 なのである。

共通 class 成分を除いた f,R は次の通り：

|h|A|B|f1,f2,others|R1,R2,others|
|---|---|---|---|---|
|1|1|0|0,3,-3|0,3r_h,0|
|6|6.4|4.1|24.6,19.2,-3|6.6r_h,3.3r_h,0|

低画像は class 2 が top、最小 logit gap 3。高画像は class 1 が top、最小 gap 5.4。いずれも log 9 より大きい。R top-loser 最小 gap はそれぞれ 3r_h,3.3r_h>0。従って (5) により両画像の g_n>0、全新ラベル期待 CE 勾配によって m は正確に下がる。

低画像 g_n/r_h=2.5047420333、高画像 5.5951622985。これらは閉形式 softmax を数値評価した参考値で、厳密符号の証明は gap の不等式だけで足りる。

独立 autodiff 確認 /tmp/cnn_ce_global_sign_verify_1009.py は clone の .venv の torch で実行し、全 assert が通過した。全パラメータの直接 CE gradient projection と (1)/(2) が一致し、full G=2.6976480913、self G=0.8773017131。全パラメータに正の 1e-8 スケール乱数摂動を入れた full-support/non-rank-one 状態でも full G=2.6976772954、self G=0.8773137200 で、両ネットワークの strict gate/winner margins と (8) が保持された。この確認は state witness であり実訓練軌道ではない。

### 7.4 full-support/open family

base point の off-center conv weights が 0、同層 channel が同一であることは必要条件ではない。有限個の gate/winner/top-logit/top-R の strict gap があるので、全パラメータと画像に十分小さい摂動を加えても (5) を維持できる。特に全 off-center conv weight を小さい正値にし、各 channel/filter/FC/head を互いに異なる小さい非 rank-one 摂動で動かせる。その点の周りには全 conv weight が正で、同じ strict 条件を満たす開球がある。

従って 2 枚の異なる画像、全 channel の共有入力・下流での重なり、異なる predicted top class、strict MaxPool winner、全 trainable bias を持つ **非空 open family** である。全画像・全 class の順位が共通という仮定はない。

## 8. 「自己項と同じ向き」をこの CE 構成で明示する

一般の (3)/(5)/(8) だけから、孤立ネットワークの符号は導けない。孤立する操作や refit の有無を定義する必要がある。

§7 の例では CE の literal channel-ablation self を次のように定義する：

- 第一 conv の channel 0 だけを残し、他の 15 channel はネットワークから除去する。
- 残る第一 conv の重み、第二 conv の target input 重み 1/16、全 FC/head と bias は full と同じ値にする。
- この孤立ネットワークの forward と全 gate/pool winner を再計算する。head refit はしない。
- m は残る第一 conv channel の元と同じ平均前活性で、同じ新 uniform CE を使う。

これは V12 の「target の gate/readout/output-bias を残して kernel capacity を計算する self」と異なる、明確に定義した CE self である。

この self では h=a/16。低画像 h=1/16、高画像 h=6/16=3/8 で、いずれも FC1/FC2 の B group が inactive。A=h で

  f_self=(0,3h,-3,...,-3),
  R_self=(0,3r_self,0,...,0),   r_self≥64/1585>0,

となる（class-common 項を除く）。class 2 が一意な top なので

  g_self=3r_self(p_self,2-1/10)>0.                   (18)

これは (8) の a=b=3r_self に対応する。参考数値 g_self/r_self は h=1/16 で 1.0893851095、h=3/8 で 1.7633298887。

従って、この非空 family では full も literal CE self も平均前活性を下げる。self/full 両方で strict gate/winner margins と strict (8)/(5) margins があるため、前節の十分小さい摂動後にも両方の符号が保持され、単一の退化点に限定されない。

全状態で full/self の符号一致を主張するのではない。一般定理に加えて、この明確な self を含む非空 family に対して方向一致まで示したものである。既存の multiple-head finite CE counterexample は無条件の長期一致が成立しないことを示しており、これと矛盾しない。

## 9. 何が解決し、何が残るか

解決した点：

1. old-head 1 step / frozen features / weak-unit / square loss / random-feature independence を仮定せず、任意の既学習 deep CNN 状態の exact CE 新ラベル平均に使える。
2. mean の full upstream dependence、全 downstream paths、全 channel overlap、共有 filter、MaxPool winner を全 θ の JVP で保持する。
3. 非循環な class-order または confidence/contrast 条件から符号を証明し、実 RLCIFAR 全構造において strict な非空 open family を与える。
4. 一つの明示的 family では、異なる predicted class を持つ画像を含め、定義した literal CE self と full の下降方向が一致する。

残る点：

1. 実 RLCIFAR 旧学習の各切替時点で条件が成立するかは、checkpoint の f,R,gate/winner margins を測る必要がある。
2. 条件が学習過程で形成・維持される法則、長期累積沈降、既存の「並び平均」からの導出は未証明。
3. arbitrary Adam/momentum、BN、label-switch 後も過去 momentum を継続する効果は別途扱う必要がある。
4. CE self と元の logdet self の同一視はできない。元の自己項と同じだと言うには、その定義を保った追加比較が必要。
