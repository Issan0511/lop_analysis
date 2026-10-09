# 実際に全 Conv と head が共同更新される CNN の長期沈降 family

2026-10-09。全パラメータ共同更新の条件付き長期定理。

## 0. 得られたことと前提の所在

共有 convolution、ReLU、MaxPool、複数 channel の重なり、自由な multiclass/spatial head を持つ CNN に対し、**全 raw Conv/head parameters を同時に SGD 更新する無限時間過程**を構成する。pool 後の feature matrix は任意に高い有限 rank にできる。固定 head や frozen prediction の定理ではない。

適切な初期条件と、現在パラメータだけから決める scalar learning rate の下で、

  B_t → B_0(I-P_U),
  a_t → a_∞>0,
  a_∞² ≤ a_0²-||B_0P_U||_F²                          (0.1)

が almost surely 成り立つ。U は pooled image vectors の張る空間、a_t は各 Conv の正の共通振幅、B_t は全 spatial/class head。初期 head が画像上で非自明なら、全 hidden 層・全 channel の平均前活性は最終的に初期値より厳密に低い。

さらに L=2 の場合、対象 first-Conv channel の mean 方向で **full NTK と channel-isolated NTK の双方の logdet capacity derivative が正**であることを閉形式で示す。CE の期待 drive はこの capacity-self の沈降方向を逆転しない。CE の strict sign は class contrast が非零のときに限る。

明示的な制限：

1. 毎 optimizer step で、新 label を fresh iid uniform に引き直す。task 内で同じ labels を固定して多 epoch 学習するモデルではない。
2. optimizer は以下で指定する adaptive scalar learning rate の SGD。標準 Adam への無条件な移行はしない。
3. Conv は 1×1、hidden/output bias は全て無し。全 channel が同じ空間 field の正の倍数になる対称性を初期化に置く。全 raw weight は trainable だが、この対称 manifold が不変になる。
4. これは full parameter 空間の開集合ではなく、不変な対称 family。feature matrix 自体を rank 1 に制限してはいない。

## 1. ネットワーク：全 channel が重なり、spatial head は自由

有限画像集合 n=1,...,N、画像分布 π_n>0、C≥2 classes。まず d_0=1 の正の grayscale input X_n(s)>0 を用いる。RGB にする場合は x_n(s)=X_n(s)u_0、u_0∈R^3 を正の単位ベクトルとしてよい。

L 個の hidden block を

  1×1 Conv(W_l) → ReLU → MaxPool

とし、各 W_l∈R^{d_l×d_{l-1}} は全要素を独立に学習する。最後の feature map を flatten し、自由な linear C-class head V に接続する。bias は使わない。

各 u_l∈R^{d_l} は全成分正、||u_l||₂=1。初期化を

  W_l=a_l u_l u_{l-1}^T,
  V_{c,j,s}=B_{c,s}u_{L,j}                            (1.1)

とする。全 W_l の要素は正で、全 input/output channel が互いに接続する。head B∈R^{C×S} は class と最終 spatial position ごとに自由で、rank 1 を要求しない。class-common 成分を除くため 1_C^T B=0 を初期条件とする。

全 a_l>0 なら、hidden preactivation は正で、ReLU は active。各 MaxPool は X_n の正の scalar 倍に作用するので、winner は a_l や B に依存しない。元画像で各 relevant pool window の最大値が一意となるように選ぶ。この strict winner は a_l>0 の間、無限時間にわたり維持される。

X_n を L 段 pool して得た最終 spatial vector を x_n∈R^S_{>0} とする。α=Π_l a_l と置けば

  H_{n,j,s}=α u_{L,j} x_{n,s},
  f_n=α Bx_n.                                        (1.2)

画像行列 X∈R^{N×S} の row が x_n^T なら rank(H)=rank(X)。異なる pooled spatial patterns を選べば rank は 2 以上、任意の min(N,S) まで増やせる。channel 方向は対称だが、data/spatial feature matrix は rank 1 でない。

例えば S=N の x_n=1_S+e_n は正かつ full rank。この pooled vector は、各最終 pool cell 内の値を小さい strictly increasing pattern にし、その唯一最大値を指定した x_{n,s} とすれば、正入力・全段 strict winner を持つ元画像から厳密に実現できる。

## 2. 全 raw SGD が family を保つこと

各 step t に画像 minibatch を iid π から取り、各画像の label を新たに iid uniform に取る。full data batch を毎回用いて labels だけ fresh にしても同じ結論になる。p_n=softmax(α Bx_n) とし、minibatch gradient matrix を

  A_t=(1/M)Σ_{i=1}^M (p_{n_i}-e_{y_i})x_{n_i}^T
       ∈R^{C×S}                                      (2.1)

と定義する。K=√2 max_n||x_n||₂ とすれば

  ||A_t||_F≤K                                        (2.2)

が全 minibatch/label 実現で成り立つ。

(1.1) 上で raw parameter の chain rule を計算すると

  ∇_{W_l} L = (α/a_l)〈B,A_t〉_F u_l u_{l-1}^T,
  ∇_V L_{c,j,s} = α (A_t)_{c,s}u_{L,j}.              (2.3)

理由：各位置の forward channel vector は u_{l-1} の倍数、backward channel vector は u_l の倍数。全 channel が同じ strict pool routing を通るため、空間和が pooled x_n を復元する。従って channel outer product の外に出る gradient 成分がない。

head の class-row sum gradient は 0 なので 1_C^T B=0 も不変。

特に全 hidden coefficient を同じ a>0 に初期化すると、共通 scalar learning rate η_t の **全 raw-matrix SGD** は

  a_{t+1}=a_t-η_t a_t^{L-1}〈B_t,A_t〉,
  B_{t+1}=B_t-η_t a_t^L A_t                           (2.4)

を満たし、全 a_l の一致も保つ。

注意：L 層は独立した trainable matrices であって、一つの tied scalar parameter を微分しているのではない。一致した値 a を持つ L 個の独立座標の gradient が等しいため一致が維持される。∂(a^L)/∂a として余分な L 倍を入れてはいけない。

## 3. 全 label 実現で保たれる正の不変領域

初期条件を

  D_0=a_0²-||B_0||_F²>0                              (3.1)

とする。任意の deterministic sequence

  0<γ_t≤1/2,  Σ_t γ_t=∞,  Σ_t γ_t²<∞                (3.2)

を用いる。例えば γ_t=0.1/(t+1)^0.6。

現在のパラメータだけで learning rate を

  D_t=a_t²-||B_t||_F²,
  η_t=γ_t√D_t/(K a_t^L),                            (3.3)
  λ_t:=η_t a_t^{L-1}=γ_t√D_t/(K a_t)

と決める。これは現在の new label/minibatch を見る前に定まり、label-dependent learning rate ではない。

(2.4) から正確に

  D_{t+1}
   =D_t-λ_t²[a_t²||A_t||_F²-〈B_t,A_t〉²].           (3.4)

Cauchy–Schwarz で bracket≥D_t||A_t||²≥0。一方 (2.2) で bracket≤a_t²K²。従って

  (1-γ_t²)D_t≤D_{t+1}≤D_t.                          (3.5)

また

  |a_{t+1}-a_t|
   ≤γ_t√D_t ||B_t||/a_t
   ≤(γ_t/2)a_t,                                     (3.6)

最後の不等式は 2√D_t||B_t||≤D_t+||B_t||²=a_t² による。よって a_{t+1}>0。

帰納的に全時刻で a_t>0、D_t>0。さらに

  d_*:=D_0Π_{t≥0}(1-γ_t²)>0,
  d_*≤D_t≤D_0,  a_t≥√d_*.                           (3.7)

Σγ²<∞ が無限積を正に保つ。全 layer の正 weight、active ReLU、strict MaxPool routing は、未来の全 label 実現に対して維持される。

## 4. 学習中の conditional CE drive は常に沈降側

F_t を新 minibatch/label を引く前までの履歴とする。fresh uniform label により

  E[A_t|F_t]
   =Σ_n π_n(p_n-u_C)x_n^T.

そこで

  c(a,B):=〈B,E[A_t|F_t]〉
    =Σ_n π_n (Bx_n)·[softmax(a^L Bx_n)-u_C].         (4.1)

各画像 z=Bx_n について

  z·(softmax(a^Lz)-u_C)
    =(1/C)Σ_{c<d}(z_c-z_d)(p_c-p_d)≥0               (4.2)

である。a>0 なので p の class 順序は z と同じ。class-centered B の場合、厳密に正でないのは Bx_n=0 のときだけ。

従って全学習時点で

  E[a_{t+1}|F_t]=a_t-λ_t c(a_t,B_t)≤a_t.            (4.3)

a_t は非負 supermartingale であり、有限な a_∞ へ almost surely 収束する。現在の B_t が旧 labels に依存していても、(4.1) は fresh new label の条件付き平均なので独立性の誤用はない。

ここで「毎 step fresh」が本質的。同じ labels を task 内で保持すれば、F_t にその labels が含まれ、E[e_y|F_t]=u_C とはできない。

## 5. head と振幅の長期極限を閉じる

以下は固定 state certificate でなく、(2.4) で実際に変わり続けるパラメータの極限の証明である。

### 5.1 累積 drive が有限

head norm の conditional 更新は

  E[||B_{t+1}||²|F_t]
    =||B_t||²-2λ_t a_t c(a_t,B_t)
       +λ_t²a_t² E||A_t||²
    ≤||B_t||²-2λ_t a_t c(a_t,B_t)+γ_t²D_0.          (5.1)

従って

  R_t=||B_t||²+D_0Σ_{k=t}∞γ_k²

は非負 supermartingale で、期待の telescoping により

  Σ_t λ_t a_t c(a_t,B_t)<∞  almost surely.

λ_t a_t=γ_t√D_t/K≥γ_t√d_*/K なので

  Σ_t γ_t c(a_t,B_t)<∞  almost surely.               (5.2)

a_t は収束し、||B_t||<a_t だから各実現上で (a_t,B_t) は compact set に入る。

### 5.2 drive 自体が 0 へ行く

(3.3)-(3.7) から

  |a_{t+1}-a_t|≤γ_t√D_0,
  ||B_{t+1}-B_t||≤γ_t√D_0.                           (5.3)

c(a,B) は a≥√d_*>0 の compact set 上で Lipschitz。従って各実現で有限な L_* が存在し、

  |c(a_{t+1},B_{t+1})-c(a_t,B_t)|≤L_*γ_t.

非負 sequence c_t に対し、Σγ=∞、γ→0、Σγ c<∞、|Δc|≤L_*γ なら c_t→0。これは離散 Barbalat の初等形である：もし δ 以上の excursion が無限回あれば、δ/2 から δ への移動と戻りには合計 γ-time が少なくとも δ/(2L_*) 必要で、その間の Σγ c は正の定数以上を費やすため (5.2) に反する。δ/2 以上から戻らない場合も Σγ=∞ に反する。

よって

  c(a_t,B_t)→0.                                     (5.4)

### 5.3 学習された class contrast が消える

各実現の compact set 上では logits が有界なので、softmax の各成分はある p_*>0 以上。class-centered z に対して softmax Jacobian の quadratic form は

  z^T[diag(p)-pp^T]z
   =min_b Σ_c p_c(z_c-b)²
   ≥p_*||z||².

uniform logits から a^L z まで積分すると

  z·[softmax(a^Lz)-u_C]≥a^L p_*||z||².

従って (5.4) は、全 π_n>0 と a≥√d_* により

  B_t x_n→0  for every n                            (5.5)

を与える。

U=span{x_1,...,x_N}、P_U をその Euclidean orthogonal projector とする。有限次元の正の Gram eigenvalue を使えば (5.5) は B_tP_U→0 と同値。一方 A_t(I-P_U)=0 なので (2.4) から

  B_t(I-P_U)=B_0(I-P_U)  for every t.

以上より

  B_t→B_∞=B_0(I-P_U)  almost surely.                 (5.6)

全学習画像上の logits は 0、確率は uniform に収束する。画像に見えない head 成分だけが初期値のまま残る。

### 5.4 最終 mean の厳密な沈降

D_t は (3.5) により減少し、D_∞∈[d_*,D_0] へ収束する。a_t²=D_t+||B_t||² と (5.6) から

  a_∞²=D_∞+||B_0(I-P_U)||²
      ≤D_0+||B_0(I-P_U)||²
      =a_0²-||B_0P_U||².                             (5.7)

従って ||B_0P_U||>0 なら

  0<√d_*≤a_∞≤√(a_0²-||B_0P_U||²)<a_0             (5.8)

が **almost surely** 成り立つ。これは最終期待値だけの比較でなく、全確率 1 の実現で初期より下がるという結論である。

layer l/channel j の平均前活性は、ある固定 μ_l>0 に対して

  m_{l,j}(t)=u_{l,j} μ_l a_t^l.

よって全 hidden 層・全 channel で

  m_{l,j}(∞)<m_{l,j}(0)                              (5.9)

となる。第一層では各 step の conditional expected drift も非正。深い層では a^l の非線形性による離散二次項があるため毎 step の supermartingale は主張しないが、(5.9) の長期比較は正確である。

個々の step では a_t が増えることもある。長期的な限界値の下降と、単発 update の符号は区別する。

## 6. 非空性と「rank 1 だけ」の限界

任意の正の full-rank pooled image matrix X、任意の class-centered B_0 で 0<||B_0||<a_0 を選べばよい。B_0 は rank min(C-1,S) まで取れ、||B_0P_U||>0 も通常の非零条件。feature matrix H の rank は X と等しい。

一例：N=S=4、x_n=1_4+e_n、C=3、

  B_0=0.05 [[1,-1,0,0],
            [0, 1,-1,0],
            [-1,0, 1,0]],   a_0=1.

||B_0||²=0.015、X は full rank なので B_∞=0、

  a_∞≤sqrt(0.985)<0.992472.

d_1,d_2,... は任意の有限 channel 数。全 hidden connection は正で全 channel が重なり、全画像に全 channel が活性化する。異なる空間 feature と複数 class contrast があり、support-separated でも head-only fit でもない。

残る対称性は channel 方向であり、W_l が rank 1 outer product、head の各 spatial position の channel vector が u_L に比例する。full parameter 空間でこの manifold から離れた摂動が無限時間小さいままかは未証明。有限時間の continuity を無限時間へ延長してはいけない。一般 CNN へ移すには transverse stability または summable off-manifold error の評価が別に必要。

## 7. L=2：元の capacity-self の向きへの厳密な接続

unscaled NTK K=JJ^T、J=∂vec(f)/∂(全 raw Conv/head parameters) とする。λ>0 に対して

  Φ_λ=(1/2)logdet(I+K/λ)

を capacity とする。loss normalization に伴う正 scalar factor は符号を変えない。

### 7.1 target first-Conv mean 方向

L=2、W_1=a u_1u_0^T、W_2=a u_2u_1^T。target channel c の first-Conv 平均前活性 m_c は

  m_c=a u_{1,c} μ_X,

ここで μ_X>0 は input scalar field の全画像・空間平均。全 raw parameter における mean 方向 ∇m_c は W_1 の row c に μ_X u_0 を持ち、他は 0。

この方向へ q だけ動かして

  v_1(q)=a u_1+q μ_X e_c,
  k=μ_X u_{1,c}>0,  t(q)=a+qk.

q=0 の近傍では全 active gate/winner が保たれ、

  f(q)=a t(q) b,  b=vec(BX^T),
  R=∂_q f(0)=ak b=(k/a)f(0).                        (7.1)

従って新 uniform CE の期待 mean projection は

  G_CE=Σ_n π_n R_n·(p_n-u_C)
       =(k/a)Σ_n π_n f_n·(p_n-u_C)≥0.                 (7.2)

class-centered B なら strict は少なくとも一つの Bx_n≠0 の場合に限る。Bx_n=0 が全画像で成立する瞬間は CE drive は 0。

### 7.2 full network の全 NTK blocks

Q=(XX^T)⊗I_C とする（vectorization の並びにより同値な permutation が入るだけ）。raw Conv1 の Jacobian は q に依存せず、その Gram は a²bb^T。

raw Conv2 の column (output i,input j) の Jacobian は u_{2,i}v_{1,j}(q)b なので、Gram は ||v_1(q)||²bb^T。

head column の Jacobian は a t(q)u_{2,i}x_{n,s} と class indicator なので、head Gram は a²t(q)²Q。

従って **全学習 block を含めて**

  K_full(q)=[a²+||v_1(q)||²]bb^T+a²t(q)²Q,
  K'_full(0)=2ak[bb^T+a²Q] ⪰0.                     (7.3)

X≠0 なので K'_full(0)≠0。λ>0 の resolvent は正定値であり

  ∂_q Φ_full(0)
   =(1/2)tr[(λI+K_full)^(-1)K'_full]>0.              (7.4)

target block だけを微分する近似は用いていない。Conv2 と head の q 依存が明示的に含まれる。

### 7.3 literal channel-isolated capacity self

first-Conv の target channel c だけを残し、他の first-Conv channels を除く。残す W_2 の input column は a u_2u_{1,c}、head B は同じ値。残った network の全 Conv/head Jacobian を再計算する。head refit はしない。

v_{1,c}(q)=a u_{1,c}+qμ_X とすると

  f_self(q)=a u_{1,c}v_{1,c}(q)b,
  K_self(q)=[a²u_{1,c}²+v_{1,c}(q)²]bb^T
           +[a u_{1,c}v_{1,c}(q)]²Q,
  K'_self(0)=2ak[bb^T+a²u_{1,c}²Q]⪰0, ≠0.          (7.5)

従って

  ∂_q Φ_self(0)>0.                                  (7.6)

これは CE の符号を「self」と新しく命名したものではなく、target channel を孤立させた network の **logdet capacity derivative** を実際に比較している。

既存の単 hidden-unit 定義にある output-bias kernel block を付け加えても、それは q に依存しない PSD block なので、(7.3)/(7.5) の derivative と strict capacity sign は変わらない。ただし long-time 共同学習定理自体は bias 無しの architecture に対するもの。学習中に独立 output bias を自由に更新しても本 family が保たれる、と主張してはいけない。

### 7.4 方向一致の正確な言い方

full/self capacity はどちらも、mean を上げる方向 q に strict に増える。従って capacity の負勾配が指定する方向は mean を下げる側。

実際の新 uniform CE の期待 drive も (7.2) で mean を下げる側で、head が画像上で非自明なら strict に一致する。Bx_n=0 の停止状態では capacity-self derivative はなお正だが CE drive は 0。従って無条件の言い方は「capacity-self の沈降方向を反転しない」であり、strict sign equality は active class contrast がある時点に限る。

(5.8) の長期 strict net sink は ||B_0P_U||>0 から得ており、この停止点の注意によって弱まらない。

## 8. Adam に移す際に残る正確な項

この節は Adam の長期結論ではなく、どの構造が移り、どこから追加証明が必要かを特定する。

### 8.1 当該 Adam 状態にも成り立つ conditional signal

balanced a_1=...=a_L は使わず、任意の a_l>0 と B について (1.1) 上では

  α=Π_l a_l,
  c(α,B)=Σ_nπ_n(Bx_n)·[softmax(αBx_n)-u_C]≥0,
  E[∂_{a_l}L|state]=(α/a_l)c(α,B)≥0.               (8.1)

従ってこの式を、その Adam 過程が実際に訪れる現在状態で評価するのは正当。SGD の B_t,a_t を Adam 状態として代入してはいけない。

入力側を含む全 u_0,...,u_L を各 channel で uniform にし、Adam moments を 0 から開始すると、同じ layer 内の Conv entries と、同じ spatial/class の head channel entries は同じ gradient history を持つ。通常の coordinate Adam も channel-symmetry manifold 自体は保つ。ただし各 layer の有効 scale、epsilon、moments が違うため a_l の balanced equality は一般には保たれない。

β1²<β2 の場合、Adam normalized coordinate update には有限な deterministic bound がある。これを使い現在の正 a_l に応じて predictable scalar learning rate を cap すれば a_l>0 は維持できる。これは累積残差の恒等式と組み合わせる際の非空状態 family を提供するが、SGD の balance inequality (3.5) は回復しない。

### 8.2 Lorentz balance を壊す項

仮にある対称設定で共通 a が保たれ、実際の optimizer の方向を d_a,D_B と書けても、

  a^+=a-η d_a,  B^+=B-η D_B

なら

  D^+-D
   =-2η[a d_a-〈B,D_B〉]
      +η²[d_a²-||D_B||²].                           (8.2)

SGD では d_a=a^{L-1}〈B,A〉、D_B=a^L A なので一次項が厳密に相殺する。Adam では過去 moment と coordinate 別分母により、一般に a d_a≠〈B,D_B〉。この一次項には自動的な符号がなく、Ση²<∞ だけで吸収できない。

従って、SGD の不変領域・supermartingale・最終 a_∞ bound をそのまま Adam に移せない。必要なのは (8.2) の累積誤差を制御する仮定、または actual Adam trajectory の予測可能な signal と denominator/momentum residual の比較定理である。

fixed iid driver で Adam の定常平均が逆転する既存 certificate は、単に時間平均を取るだけではこの問題が消えないことを示す。一方、本稿の moving SGD family は、追加条件の下では本当に変化する CNN の長期沈降まで証明できることを示す。二つの過程を混同しない。

## 9. 一般 CNN への拡張に必要な追加仮定

- bias を学習する場合：logit bias と hidden shift が radial signal (4.1) を壊す項を制御する。
- 一般 3×3/5×5 filters、channel 非対称 perturbation：channel outer-product manifold からの逸脱が累積しない transverse stability を示す。
- 同一 random-label task を多 step 学習する場合：fresh uniform conditional expectation の代わりに、task 内の labels/state 依存を保った平均を導く。
- Adam：actual optimizer residual の累積制御、または別の不変量を見つける。(8.2) の一次項を落とさない。

本稿の条件はこれらを既に解いたという主張ではない。しかし、支えを分離した toy kernel や frozen-state certificate に留まらず、全 channel overlap・full-rank spatial data・全 Conv/head の実共同更新を保ったモデルで、capacity-self の向きと長期 net sink までを一本の証明でつないでいる。


## 10. 実際の自動微分との検算

[verify_moving_ce_kernel.py](verify_moving_ce_kernel.py) は二段の共有 Conv→ReLU→MaxPool、3 input/hidden channels、10 classes、4 images、rank 4 の pooled image matrix を構成する。全 raw parameters に CE の自動微分を掛け、fresh labels による300回の同時 SGD と (2.4) を照合した。最大 parameter 誤差は 1.34e-15 未満、balance identity の誤差は 6.67e-16 未満だった。

初期状態と300更新後の両方で、全 raw Conv/head parameters から40×40の NTK を直接求め、(7.3)、(7.5) の full/self kernel と方向微分を照合した。最大誤差は3.56e-15未満。両者の容量微分と CE の期待 mean projection は正だった。結果は [moving_ce_kernel.json](../../results/cnn_drive_1009/moving_ce_kernel.json)。

初期 a=1.2、||B||²=0.38235615296 なので、本定理による上界は a∞²≤1.05764384704。300更新の a=1.18665100989 は極限値を測ったものではない。無限時間の主張は §3–5 の証明から従い、有限の数値軌道による外挿ではない。


## 11. 「動作し続けるのは ReLU だからか」への位置づけ

この証明は ReLU の正側 φ(z)=z のみを使う。正側の線形性は raw gradient の outer-product 構造、次数の揃った更新式、balance の相殺を成立させる。ReLU の0での折れ目や負側をまたぐ挙動は扱っていない。

一方、ReLU が最後まで active な理由は、ReLU 一般の性質ではなく、本稿が選んだ正入力・bias 無し・初期 D_0>0・適応学習率から a_t≥√d_*>0 を証明したことにある。平均前活性は正の定数×a_t^l であり、低下しても正の下限を持つ。この結果を負側への沈降やユニットの死の証明として使わない。他の活性化関数への拡張は本稿では行っていない。
