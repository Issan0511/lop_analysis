# 同じ ReLU CNN で、task 内 label reuse を許した長期沈降定理

2026-10-09。全パラメータ共同更新と課題内ラベル再利用の長期定理。

## 0. 結論

既存の moving CNN family を変えず、**一つの task につき各画像に uniform random label を一度だけ割り当て、その label を固定したまま H≥2 回の共同 SGD 更新に再利用する**場合へ長期定理を延長する。H は任意の固定有限整数。各 task 内の画像 schedule は full batch、または各画像を同じ重みで覆う minibatch schedule でよい。

学習率に明示的な damping を入れると、全 raw Conv/head parameters が実際に共同更新される過程で

  B_t→B_0(I-P_U),  a_t→a_∞>0,
  a_∞²≤a_0²-||B_0P_U||_F²                         (0.1)

が almost surely 成り立つ。初期 head が画像上で非自明なら、全 hidden 層・全 channel の平均前活性の最終値は初期値より厳密に低い。

元の per-step fresh-label martingale は使用しない。task 開始点で凍結した H 個の gradient と、実際に task 内で変化する state の gradient の差を、全状態で一様な O(H²δ_k²) に抑える。task 境界の almost-supermartingale から軌道有界性を導くので、軌道が有界であるという仮定も不要。

制限は残る：1×1 Conv/ReLU/MaxPool、bias 無し、channel-symmetric initial family、指定された decaying/damped scalar SGD。Adam、標準一定 learning rate、一般の非対称 5×5 CNN の長期結論ではない。ReLU 以外の活性化へは拡張しない。

## 1. 変えないネットワークと raw parameter 更新

L 個の 1×1 Conv→ReLU→MaxPool block と、その最終 spatial features 全体を flatten した自由な multiclass head を用いる。全 Conv と head の raw weights を独立に学習する。hidden/output bias は無し。

正の unit vectors u_0,...,u_L と、正の scalar input fields X_n(s)>0 に対し

  input_n(s)=X_n(s)u_0,
  W_l=a_l u_lu_{l-1}^T,
  V_{c,j,s}=B_{c,s}u_{L,j}.                           (1.1)

各 relevant pool window の最大値は一意とする。a_l>0 なら全 channel の ReLU は active、pool winner は X_n の正の scalar 倍に対する選択なので、パラメータ振幅には依存しない。

L 段 pool 後の正の spatial vector を x_n∈R^S とし、R=max_n||x_n||₂>0、K=√2 R と置く。各 a_l が同じ a>0 なら

  f_n=a^L Bx_n.                                     (1.2)

image/spatial matrix X の row を x_n^T とすれば、実 feature matrix の rank は rank(X)。rank 1 を要求せず、B も自由な C×S matrix。初期条件として 1_C^T B_0=0 を置く。

現在の label assignment Y=(Y_1,...,Y_N)、画像 minibatch weights q_n≥0, Σ_nq_n=1 に対して

  A(z;Y,q)=Σ_n q_n[softmax(a^L Bx_n)-Y_n]x_n^T,
  z=(a,B).                                          (1.3)

常に ||A||_F≤K。全 raw parameter の chain rule は

  ∇_{W_l}L=a^{L-1}〈B,A〉u_lu_{l-1}^T,
  (∇_V L)_{c,j,s}=a^L A_{c,s}u_{L,j}.                (1.4)

従って全 hidden 振幅を等しく初期化し、全 raw parameters に同じ scalar learning rate η を適用すると、channel symmetry と a_1=...=a_L が維持され、実更新は正確に

  a^+=a-ηa^{L-1}〈B,A〉,
  B^+=B-ηa^L A.                                     (1.5)

ここで L 個の独立 matrices の値が一致しているのであって、一つの tied scalar a を微分するのではない。gradient に余分な L 倍は入らない。

## 2. task と schedule

task k=0,1,... の開始時に、各画像 n に Y_{k,n} を independently uniform one-hot class label として割り当てる。新 task の Y_k は過去の全 parameters、labels、optimizer history と独立。

この label を固定したまま H 回更新する。step j=0,...,H-1 の minibatch weights を q_{k,j,n} とし、

  (1/H)Σ_{j=0}^{H-1}q_{k,j,n}=π_n>0,
  Σ_nπ_n=1                                          (2.1)

という equal weighted coverage を要求する。各 task で同じ π を使う。

例：

- full batch：q_{k,j,n}=π_n を全 step で使う。
- N 枚を一定 batch size で一巡する epoch を E 回行う：各 epoch の順序は任意の permutation、π_n=1/N、H=E·N/batch_size。

schedule の random permutation は labels と独立に選んでよい。実際、以下の凍結和の等式は (2.1) が pathwise に成り立てば順序によらない。

task k の同じ deterministic scalar δ_k を H 回使う。条件は

  0<δ_k≤1/2,
  Σ_kδ_k=∞,  Σ_kδ_k²<∞.                            (2.2)

具体例は

  δ_k=κ/[H(k+1)^ρ],  0<κ≤1/2,  1/2<ρ≤1.           (2.3)

H=30000 のような大きな固定有限値も定理に含まれる。ただし (2.3) と以下の damping は明示的な optimizer 条件であり、標準一定 learning rate の主張ではない。

## 3. learning rate と正値不変量

p=L+1 とし、各 inner step の現在 state で

  D=a²-||B||_F²,
  η=δ_k√D/[K a^L(1+a^p)]                           (3.1)

を用いる。η は現在の update の前にある state から計算する。同じ task 内ではその state は既に Y_k に依存してよい。ここで新たな label-independent per-step conditioning を仮定していない。

初期条件を

  a_0>0,  D_0=a_0²-||B_0||²>0                     (3.2)

とする。β(a)=1/(1+a^p)、λ=ηa^{L-1}=δ_kβ(a)√D/(Ka) と置けば、(1.5) から

  D^+=D-λ²[a²||A||²-〈B,A〉²].                      (3.3)

Cauchy–Schwarz により bracket≥D||A||²≥0、上から a²K² 以下なので

  (1-δ_k²)D≤D^+≤D.                                 (3.4)

また

  |a^+-a|≤δ_kβ(a)√D||B||/a
           ≤(δ_kβ(a)/2)a<a.                        (3.5)

最後は 2√D||B||≤D+||B||²=a²。従って全 labels/schedules の実現で a は正のまま、D は正のまま減少する。

全時刻で

  d_*:=D_0Π_{k≥0}(1-δ_k²)^H>0,
  d_*≤D≤D_0,  a≥√d_*.                              (3.6)

Σδ²<∞ が無限積を正に保つ。必要なら完全に陽な下界として

  d:=D_0 exp[-(4/3)HΣ_kδ_k²]≤d_*                  (3.7)

を使える。これは δ_k²≤1/4 と log(1-x)≥-(4/3)x による。

以下では d=d_* または (3.7) のいずれかを固定する。この d は初期条件・H・deterministic step schedule だけで定まる。

## 4. damping による global boundedness / Lipschitz bound

### 4.1 convex domain を使う

state norm は ||(a,B)||²=a²+||B||_F²。次の閉 convex domain を考える：

  C_d={ (a,B): a≥sqrt(d+||B||_F²) }.                 (4.1)

右辺は (sqrt(d),vec(B)) の Euclidean norm なので、その epigraph は convex。従って二つの actual states を結ぶ線分も C_d 内にある。

**D≤D_0 という上側条件は convex ではないので、global Lipschitz 証明の領域には含めない。** 線分内部で D>D_0 になっても以下の bound は成立する。

### 4.2 一 step を δ_k F と書く

raw update (1.5),(3.1) は正確に

  z^+=z+δ_k F(z;Y,q),

  F(z;Y,q)
   =-s(z) T(z;Y,q),
  s(z)=√D/[K(1+a^p)],
  T(z;Y,q)=(〈B/a,A(z;Y,q)〉, A(z;Y,q)).             (4.2)

||B/a||≤1、||A||≤K より ||T||≤√2K。また √D≤a であり、p≥2 なら a/(1+a^p)≤1。従って C_d 全体で

  ||F||≤√2.                                         (4.3)

actual trajectory 上では D≤D_0 も使えるので

  ||F||≤C_F:=√2 min{1,√D_0}.                        (4.4)

label 値と minibatch schedule によらない bound である。

### 4.3 明示的な global Lipschitz constant

softmax Jacobian の Euclidean operator norm は 1/2 以下。例えば symmetric covariance matrix の絶対行和は 2p_c(1-p_c)≤1/2 なので得られる。

z∈C_d では ||B||≤a。任意の unit state direction に対して

  ||D(a^L Bx_n)||≤R√(L²+1)a^L.

よって

  ||D A||≤C_A a^L,
  C_A=(R²/2)√(L²+1).                                (4.5)

一方、

  ||∇s||
   ≤(1/K)[√2 a/{√d(1+a^p)}
              +p a^p/(1+a^p)²]
   ≤(1/K)[√2/√d+p/4].                              (4.6)

また ||D(B/a)||≤√2/a から

  ||DT||≤√2K/a+√2C_A a^L.                           (4.7)

product rule、√D≤a、p=L+1 を用いて

  ||DF||
   ≤||∇s||·||T||+s||DT||
   ≤ 2/√d+p/(2√2)+√2+(R/2)√(L²+1)
   =:L_F<∞.                                         (4.8)

これは C_d 全体で成立する。convexity により任意の二点について

  ||F(z;Y,q)-F(z';Y,q)||≤L_F||z-z'||.               (4.9)

が得られる。trajectory の上界を仮定した計算ではない。d は既に pathwise 不変量から得た deterministic positive constant。

## 5. task 内 correlation を O(H²δ_k²) にまとめる

task k の開始 state を Z_k=z_{k,0}、終了 state を Z_{k+1}=z_{k,H} とする。まず pathwise に

  Z_{k+1}-Z_k
   =δ_kΣ_{j=0}^{H-1}F(Z_k;Y_k,q_{k,j})+e_k,        (5.1)

  e_k=δ_kΣ_j[F(z_{k,j};Y_k,q_{k,j})
                       -F(Z_k;Y_k,q_{k,j})].

(4.4) より ||z_{k,j}-Z_k||≤jδ_kC_F。従って

  ||e_k||≤(L_F C_F/2)H(H-1)δ_k²
          =:C_Hδ_k².                                (5.2)

この e_k が、同じ labels を使ううちに B と Conv が変わり、その state と labels が相関する効果を丸ごと含む。e_k を martingale とみなしていない。

F は A に線形で、task 開始点では s,B/a は固定。equal coverage (2.1) により pathwise に

  Σ_j F(Z_k;Y_k,q_{k,j})=H F(Z_k;Y_k,π).           (5.3)

従って actual task dynamics は

  Z_{k+1}
   =Z_k+Hδ_k F(Z_k;Y_k,π)+e_k,
  ||e_k||≤C_Hδ_k².                                 (5.4)

これは full-batch frozen reference の gradient を実行したという意味ではない。実際には H 回 raw gradients を更新しており、その exact algebraic decomposition である。

例 (2.3) では task drift の係数は Hδ_k=κ/(k+1)^ρ で、その総和は発散する。一方、

  C_Hδ_k²
   =(L_F C_F/2)[(H-1)/H] κ²/(k+1)^(2ρ)

の総和は有限。label reuse による相関を「無視する」のではなく、無限時間で足しても有限な高次誤差へ抑えることが決め手である。

## 6. task 境界で初めて uniform-label 平均する

G_k を task k の新 label assignment を引く前までの履歴とする。Z_k は G_k-measurable、Y_k は G_k と独立な iid uniform labels。

z=(a,B) に対し

  c(z)=Σ_nπ_n(Bx_n)·[softmax(a^L Bx_n)-u_C]≥0.      (6.1)

非負性は softmax の class 順序、または logsumexp の radial convexity による。class-centered B なら c=0 は全 Bx_n=0 と同値。

さらに

  h(z)=√D/[K a(1+a^p)] c(z)≥0.                      (6.2)

と置くと、task の凍結 reference に限って

  E[F_a(Z_k;Y_k,π)|G_k]=-h(Z_k).

実際の H-step dynamics (5.4) から

  E[a_{k+1}|G_k]
    ≤a_k-Hδ_k h(Z_k)+C_Hδ_k².                       (6.3)

ここで a_k は task 開始点の振幅。同じ task の途中で E[Y_{k,n}|current history]=u_C とする不正な操作はしていない。

## 7. boundedness を仮定せずに導く

ε_k=C_Hδ_k² とし、

  Q_k=a_k+Σ_{r=k}∞ ε_r.

Σδ²<∞ なので tail は有限。(6.3) から

  E[Q_{k+1}|G_k]≤Q_k-Hδ_k h(Z_k).                   (7.1)

Q_k は非負 supermartingale。したがって Q_k は有限な極限に almost surely 収束し、tail→0 により

  a_k→a_∞<∞  almost surely.                         (7.2)

また期待を telescoping すると

  Σ_k Hδ_k h(Z_k)<∞  almost surely.                 (7.3)

が得られる。各有限時刻では (4.4) により state の増分が deterministic に有界なので、ここで使う可積分性にも循環はない。

||B_k||<a_k、a_k≥√d より、各実現の boundary trajectory は結果として compact set に入る。さらに

  max_{j<H}||z_{k,j}-Z_k||≤Hδ_kC_F→0,              (7.4)

なので全 inner steps も有界で、a は全更新列に沿って同じ a_∞≥√d>0 へ収束する。

D も単調に正の D_∞ へ収束するので、raw learning rate は

  η_{k,j}/δ_k→√D_∞/[K a_∞^L(1+a_∞^p)]>0

を満たす。従って全 updates にわたり Ση=∞、Ση²<∞。有限の総 learning-rate budget で学習を止めたために得た極限ではない。

## 8. visible head は消える

h(z) は -E[F_a(z;Y,π)] なので C_d 上で L_F-Lipschitz。(4.4) から

  |h(Z_{k+1})-h(Z_k)|≤L_F Hδ_k C_F.                (8.1)

h≥0、Σδ=∞、δ→0、Σδ h<∞ と (8.1) は、離散 Barbalat の初等形から

  h(Z_k)→0                                         (8.2)

を与える。証明：固定 ε>0 以上の excursion が無限回あれば、ε/2 と ε の間を移動するには正の δ-time が必要で、そのたび Σδ h が正の定数以上増える。戻らず ε/2 以上に留まる場合も Σδ=∞ に矛盾する。

(3.4) から D_k→D_∞≥d、(7.2) から a_k→a_∞≥√d。よって h/c の係数は正の有限値へ収束し、

  c(Z_k)→0.                                         (8.3)

各実現で logits は compact set 内に有界。softmax の最小成分の正の下界と covariance quadratic form を積分すれば、ある実現依存の m_*>0 が存在して

  c(a_k,B_k)≥m_*Σ_nπ_n||B_kx_n||².                 (8.4)

class centering 1_C^T B_k=0 は各 raw SGD update で保たれるため、class-common 成分を別に処理する必要はない。全 π_n>0 より

  B_kx_n→0  for every n.                            (8.5)

U=span{x_n}、P_U をその orthogonal projector とする。有限次元の正の Gram eigenvalue により B_kP_U→0。一方、全 inner step で A(I-P_U)=0 なので

  B_t(I-P_U)=B_0(I-P_U).

従って boundary だけでなく (7.4) を使って全 updates に対し

  B_t→B_∞=B_0(I-P_U)  almost surely.                (8.6)

全学習画像上の logits は 0、出力確率は uniform に収束する。

## 9. strict net mean sink

全 inner steps で D_t は減少し、

  d≤D_∞≤D_0.

balanced family の恒等式 a_t²=D_t+||B_t||² と (8.6) により

  a_∞²
   =D_∞+||B_0(I-P_U)||²
   ≤D_0+||B_0(I-P_U)||²
   =a_0²-||B_0P_U||².                              (9.1)

初期 active head ||B_0P_U||>0 なら

  0<√d≤a_∞≤sqrt(a_0²-||B_0P_U||²)<a_0            (9.2)

が almost surely 成り立つ。

各 hidden layer l/channel j の平均前活性は、固定 μ_l>0 に対して

  m_{l,j}(t)=μ_l u_{l,j} a_t^l.

従って全 hidden 層・全 channel で

  m_{l,j}(∞)<m_{l,j}(0).                            (9.3)

個々の inner step は上向きにもなり、task 境界の条件付き drift にも O(H²δ²) の誤差がある。それでも最終比較 (9.3) は pathwise energy identity と head の消失から厳密に得られる。

このモデルでは a_∞>0、ReLU gates は全有限時点で active で極限でも正のまま。従ってこれは mean-preactivation の strict decrease の定理であり、gate death や可塑性喪失の証明ではない。

## 10. capacity-self の向きとの関係

L=2 の場合、各 actual state は以前と同じ positive channel-symmetric family 上にあるため、target first-Conv channel の mean 方向 q に対する full/self NTK formulas は変わらない。

k=μ_X u_{1,c}>0、b=vec(BX^T)、Q=(XX^T)⊗I_C とすれば

  K'_full(0)=2ak[bb^T+a²Q] ⪰0, ≠0,
  K'_self(0)=2ak[bb^T+a²u_{1,c}²Q] ⪰0, ≠0.

λ>0 の full と literal channel-isolated logdet capacity は、mean を上げる方向に strict に増加する。その負勾配は mean を下げる側であり、(9.3) の長期 net sink と一致する。

task 内で既知になった reused labels に対する実 CE gradient の符号を、毎 step uniform 平均に置き換えてはいけない。ここで保証するのは task 開始点の新 assignment に対する凍結平均と summable within-task error、そして長期極限の方向である。task 内の各実 update が自己項側を向くという主張ではない。

## 11. 非空性・実験 schedule への届く範囲

任意の有限 H、正で full-rank の pooled spatial images、0<||B_0||<a_0 の class-centered head、(2.3) の step sequence を選べば条件を満たす。N=S=4、x_n=1+e_n、C=3、既存の rank-2 head example などをそのまま使える。全 channel は各画像で重なり、全 Conv と head が共同更新される。

task あたり 400 epochs、1200 images、batch 16 なら H=30000 であり、各 epoch の random permutation は (2.1) の equal coverage を満たす。本定理はその **label reuse / image coverage という schedule** を扱える。

ただし optimizer は指定された decaying/damped SGD、architecture は bias 無しの対称 1×1 CNN。実 RL-CIFAR の standard Adam、momentum 継続、独立 bias、一般 5×5 channel geometry、標準 learning-rate schedule まで証明したものではない。


## 12. 実 raw SGD の検算

[verify_task_reuse_longtime.py](verify_task_reuse_longtime.py) は3チャネル・二段Conv・10クラス・4画像・batch2を用い、各課題で4 epochs、8更新、同じラベルを再利用する。80課題・640回の全 raw SGD と閉形式の再帰が1.56e-15未満で一致した。balance identity の誤差は6.67e-16未満。課題開始点で凍結した更新との誤差は全課題で明示上界以下だった。

最初の課題では実際の振幅は上がった。それでも条件下の長期結論は §6–9 の確率論的証明から成立する。有限軌道の終値を無限時間の極限として使っていない。結果は [task_reuse_longtime.json](../../results/cnn_drive_1009/task_reuse_longtime.json)、独立監査は [task_reuse_longtime_review.md](task_reuse_longtime_review.md)。
