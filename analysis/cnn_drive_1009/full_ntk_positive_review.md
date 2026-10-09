# Native full-NTK / FC suffix / 32×32 reference：独立解析監査

2026-10-09。対象 /tmp/cnn_full_ntk_positive_1009.md、特に §9 を読了。数値計算は行っていない。

**判定：PASS。all-raw-bias を含む suffix formulas は正しい。実 32×32 RGB、Conv5×5/pad2 を二段、MaxPool2 を二段、16/16 channels、FC100/100 の full-row-rank reference は、以下の解析的構成で非空性を示せる。12×12 の数値 witness に依存しない。**

## 1. 全 raw Jacobian blocks の監査

第一 augmented filter を a₁u₁rᵀ、第二 weight を u₁[j]C[i,δ] とし、第一層以降の bias を値として 0 に置く。これら bias は全て独立な trainable coordinates として Jacobian に残す。

mean 方向 q に対して v(q)=a₁u₁+qκe_c、t(q)=a₁+qk、k=κu₁[c]>0。第二 preactivation は t(q)U。fixed strict branch 上の suffix は入力に線形なので：

- 第一 Conv weight/bias の Jacobian は q に依存しない。
- 第二 Conv weight の各 input-channel block は v(q)[j] 倍。
- 第二 Conv より後の全 weight Jacobian は t(q) 倍。
- 第二 Conv とそれ以降の全 bias Jacobian は q に依存しない。注入した basis vector に作用する downstream linear Jacobian が scale に依存しないため。

従って

  K'_full(0)=2a₁k(G₂+G_suf)

は正しい。literal self では第二 preactivation が u₁[c]v_c(q)U になるだけで同じ gate/winner を通り、

  K'_self(0)=2a₁k(G₂+u₁[c]²G_suf)

も正しい。self の G₂ が full と同じなのは、suffix の input Jacobian が正の入力 rescaling に依存せず、削除後の input-channel 和だけが変わるため。

最後の hidden features を scale t=1 で F とすると、自由な output-head weight block が (FFᵀ)⊗I_C を G_suf に含む。F の full row rank が両微分の正定値下界を与える。これには全 output bias / hidden FC bias の Jacobian を取り除く操作はない。

bias の値が 0 という条件は厳密 reference identity の条件。actual open neighborhood で非零 bias を許すときは、既存の Jacobian perturbation certificate を使うという本文の区別も正しい。

## 2. 実 32×32 input での二つの独立 spatial features

0-based indexing とする。Conv5/pad2→Pool2→Conv5/pad2→Pool2 の final pool coordinate q の一次元 receptive support は

  [4q-6,4q+9]

で長さ 16。q=2 と q=5 ならそれぞれ [2,17] と [14,29] で、全て original image 内にある。

三 RGB planes とも共通の定数 c>0 の画像を I₁ とし、I₂ は同じ画像に pixel (8,8) の正 bump を加えたものとする。例えば c=1/4、bump height=1/4 なら全画素は (0,1) 内に置ける。bump は一 colour plane でも全 planes でもよい。

この **実際の二画像**から augmented patch mean μ と r=μ/||μ|| を作る。全 spatial/color coefficient は正で、bias coefficient も正。padding のため offset ごとに係数が違っても、内側の定数画像への convolution response は位置によらない。

final pool cells を left=(row 2,col 2)、right=(row 2,col 5) とする。両者の receptive fields は image 内にあり、I₁ の対応する全中間 fields は定数。正の全-offset 第二 kernels C[i,δ] を選ぶと、各第二 channel i のこの二出力は同じ正値 A_i。

I₂ の bump は left field に入り、right field には入らない。全経路係数が正で、定数 baseline の pool 候補は同値なので、少なくとも一つの left pool candidate が strict に増加し、final left output も A_i+Δ_i、Δ_i>0 となる。right は A_i のまま。

同じ r を両画像に用いて比較しているため、r 自体が二画像から定義されることはこの議論を壊さない。列を (right,left) と並べた二画像の minor は

  det [[A_i,A_i],[A_i,A_i+Δ_i]]=A_iΔ_i>0。

列を (left,right) と並べる場合は -A_iΔ_i であり、必要なのは非零性。これで flatten した 2×1024 Conv feature matrix は row rank 2 を持つ。

## 3. μ を変えずに全 pool ties を除去できる

上の簡単な定数 baseline は ties を持つので、そのまま strict-winner reference と呼んではいけない。次の摂動で除去できる。

  I₁→I₁+εS,   I₂→I₂-εS。

二画像の pointwise sum が変わらないため、padding を含む dataset mean augmented patch μ、従って r は **厳密に変わらない**。十分小さい ε||S||∞ により両画像の正値・(0,1) 範囲と非零 minor は保たれる。S 自体は符号付きでよい。「画像の正値を保つ微小摂動」と表現するのが正確。

r を固定した network は入力の piecewise-affine function。第一 pool の異なる候補は、正の finite kernel と異なる spatial support を持つため異なる affine forms であり、tie は proper affine hyperplane である。

第一 pool の routing を固定した各 cell 内でも、隣接する第二 Conv 候補は異なる extreme input support を持つ。5×5 の全係数が正なので、その片方だけに入る extreme pixel の係数は非零であり、二候補が同一 affine form になることはない。境界では padding で片側が切られても、反対側の extreme support が異なる。

第一 routing の有限個の候補ごとに第二 pool ties も proper affine hyperplanes に含まれる。従って有限個の tie 集合を避ける任意に小さい generic S が存在し、両画像・全 channels の全 pool winners を一意にできる。ReLU の正値は positive input/kernel と第一 positive bias により維持される。

この手順は単なる「random perturbation なら多分 ties が消える」ではなく、μ を固定した有限 piecewise-affine map の proper tie sets を避ける解析的構成である。

## 4. 16/16 channels と FC100/100

第一層は任意の正 unit vector u₁∈R¹⁶ により全 16 filters を配置できる。第二層は任意の strictly positive finite kernels C[i,δ] を 16 output channels に置く。全 input channels と全 25 offsets が使用される。前節の minor は一つの第二 output channel の二 spatial positions だけで既に非零なので、他の channels を含めても row rank 2 は保たれる。

得られた A∈R^{2×1024} は strictly positive / full row rank。独立な二列 I を選び、FC1 の最初の二 output columns を selector+ε·1 とする。ε>0 を十分小さくすれば全 weight が正で、

  A[:,I]+ε(A1)1ᵀ

の determinant は非零のまま。残り 98 outputs も正 weights で作る。全 preactivation は strict positive。FC2(100→100) でも同じ操作を繰り返せば、最後の F∈R^{2×100} は strictly positive / full row rank。

最後の自由な 10-class head には class-centered weights と非零 class contrast を選べる。全 ordinary biases は trainable coordinates のままで、reference では第一 Conv 以降の値だけを 0 にする。

以上により actual dimensions の厳密 reference が存在する。strict margins、full row rank、K' の正定値下界、正の reference CE drive から、既存の perturbation certificate は全 raw parameters（独立 bias を含む）の非空 open neighborhood を与える。

## 5. 射程

これは native 5×5 + 2FC architecture の fixed-state / open-neighborhood capacity-self と fresh-label CE direction の非空性である。actual training trajectory、label reuse、Adam、長期不変性、ReLU death の定理ではない。この区別を保てば、§9 と追加 existence paragraph を統合してよい。
