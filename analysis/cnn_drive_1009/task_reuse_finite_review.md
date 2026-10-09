# 一般 CNN の有限 reused-label task：独立最終監査

2026-10-09。対象 task_reuse_finite.md、対応 verify script、保存済み JSON を読了。指示どおり検算プログラムは再実行していない。

**判定：PASS。一般定理、具体的な負 mean / mixed-gate 例、全 NTK / literal-self の厳密符号と半径 .005 の安定性に、blocking defect は見つからない。**

## 1. 新ラベルの平均と actual changing-state trajectory

課題開始時点の θ₀ と label-independent な schedule / learning rates を条件付け、その後一度だけ抽出した Y を課題中に再利用する設定は正確。f,p,R を θ₀ で固定した参照では gradient が Y に線形なので、

  E D_ref=Σ_nq_nR_n·(p_n-u_C)

は厳密。課題途中の θ_t と Y の依存を捨てる操作はない。

全 raw parameters の actual SGD と参照を、同じ Y に対して pathwise に比較する誤差評価も正しい。τG*≤r による球内不変性は帰納で閉じ、gradient Lipschitz bound から

  E_H=(||u||L*G*/2)[τ²-Ση_t²]

が得られる。第一 Conv mean が全パラメータの affine 関数なので、mean 自体について Taylor の追加誤差は不要。

## 2. label reuse の分散

各画像の reused label は一つの確率変数として重みをまとめてから二乗するため、Cov(Y_n)=P/C と画像間独立性より

  Var D_ref=Σ_nq_n²||PR_n||²/C

となる。fresh-per-occurrence の場合は各出現重みの二乗和となり、等 learning rate / batch size、E 回の equal coverage では比が E。本文はこれを実 Adam の分散ではなく、凍結参照の分散として限定している。

一般の sampling-with-replacement batch を許す際は q_n に画像の出現 multiplicity を数える必要があるが、本文の permutation / full-batch 設定では現記法で問題ない。

## 3. 具体例の幾何と CE margin

filter と画像から次の値を独立に追跡できる：

- target channel mean は [0.7·1.3+0.7·1.03]/2-1=-0.1845。
- u=(0.35,0.35,0,1)、||u||²=1.245<(9/8)²。
- pool features は [[0.3,0.03],[0.03,0.3]]。det=0.0891>0 で rank 2、両 channel が両画像に正反応。
- 1.3 の画像では preactivation が (0.3,0.04,-0.22,-0.48)、1.03 の画像では (0.03,-0.176,-0.382,-0.588)。正負 gate が混在し、最小 absolute margin は 0.03。
- ReLU 後の最小 winner gap も 0.03。winner は両画像とも rho=1 の位置。
- winner の mean-direction derivative は 1.35、head の target column は 0.24a なので R=0.324a。
- logits の scalar coefficients は 0.0945、0.0702。

0≤t≤0.1 で p_c≥e^(-0.2)/3≥4/15。centered a=(-1,0,1) について Var_p(a)≥(4/15)||a||²=8/15。これを 0 から t まで積分した Γ=44469/3125000 は正しい有理下界であり、数値 CE gradient の符号を仮定していない。

## 4. 半径 .005、Jacobian / Hessian、期待 net sink

各 augmented patch norm は √2<3/2。radius .005 の全 parameter ball で site 変化は高々 .0075、winner gap の変化は高々 .015<.03。従って全 gate / winner は固定される。

cell 内では h=A_nθ_conv、||A_n||≤3/2、f=Vh+b_out。Jacobian block の二乗和 bound

  ||Df||²≤||V||²(3/2)²+||h||²+1

と提示された初期 norm / perturbation bounds から J*=5/4 が正当化される。

混合 Hessian の二項は

  δV₁Aδθ₂+δV₂Aδθ₁。

二つの parameter block の norm を Cauchy–Schwarz でまとめると、bilinear operator bound は Q*≤3/2。余分な係数 2 を入れる必要はない。

√2≤3/2 を使えば

  G*=(3/2)(5/4)=15/8,
  L*=(5/4)²/2+(3/2)²=97/32。

これらは全 batch / labels に一様。H=8、η=1/10000 の τG*=.0015<.005、参照誤差 1.79033203125e-6 と Γ による期待 sink 下界 306999423/32000000000000>0 は整合している。

保存済み JSON は全 9 assignments を列挙し、mean が上がる実現も含めた期待を報告している。したがって per-realization sign を期待 sign に取り違えていない。

## 5. full NTK と literal capacity-self

winner augmented inputs Xi を使う Conv block、head block、output-bias block をそれぞれ含めた

  K=(XiXiᵀ)⊗(VVᵀ)+(HHᵀ+11ᵀ)⊗I₃

は正しい。Conv bias は Xi の末尾 1 に含まれる。対象 mean 方向では Conv Jacobian block は変わらず、head-feature Gram の微分だけが残るため

  K'=(r_hhᵀ+hr_hᵀ)⊗I₃。

literal self は他 Conv channel とその head column を除き、残る Conv bias と output bias を保持する。verify script は残る全 Jacobian を再計算しており、削除した parameter に対応する zero columns を持つ実装も Gram として同値。

full の容量微分は class a 方向とその直交二方向に分けると

  (0.4455/3.5895)+(0.891/3.1089)
   =10185021/24798659>0。

self も同じ class-space 分解、または本文の有限有理行列 inversion で 25201935/61683554>0 を得る。script は SymPy rational matrix で厳密に評価し、別に all-raw autograd NTK/JVP との一致を確認する構造である。

K' の sample-space 固有値は r_h·h±||r_h||||h|| で、負の固有値もある。本文は PSD を仮定せず full/self 各 resolvent を評価しており、ここに仮定の飛躍はない。

## 6. Eq. (7)：ball 全体の capacity sign

この fixed bilinear cell では stacked J が affine、J'=DJ[u] が constant。N=2 と各 sample の上界から

  ||J||≤√N J*,
  ||J-J₀||≤√Nρ*r,
  ||J'||≤√Nρ*||u||

が得られる。従って

  ||K-K₀||≤2N J*ρ*r=.0375,
  ||K'-K₀'||≤2N(ρ*)²||u||r=.050625

は正しい。full/self とも同じ保守的 bounds を使える。

K≥0、λ=1 より resolvent norm≤1、resolvent difference≤||K-K₀||。NC=6 次元の trace bound と ||K₀'||<1.1 を使うと

  |Δ(∂qΦ)|≤(6/2)[.050625+.0375·1.1]=.275625。

これは両初期 derivative（約 .41071、.40857）より小さい。従って radius .005 の全 ball で full/self capacity derivative は strict positive。この robustness は数値軌道の点検だけに依存していない。

## 7. 最終的な射程

本成果は、negative mean、mixed ReLU gates、nonproportional rank-2 overlapping channels、trainable Conv/head/output biases を持つ非空例で、有限 reused-label task の期待 net update が元の capacity-self の沈降側へ進むことを保証する。

同じ球内で全 raw parameters が実際に動くことも扱っている。一方、無限 task 反復、gate/winner crossing、Adam への外挿は行っていない。これらの限定は本文と結果に明記されている。
