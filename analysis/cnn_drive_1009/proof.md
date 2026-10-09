# CNN の駆動源：全チャネルを残す有限の符号保証（1009）

状態: 検証中。ユーザーの依頼全体は未完了。以下は新しい条件付き定理と、無条件化を阻む反例である。実際の RL-CIFAR の二つの Conv と二つの FC を Adam・CE で同時訓練する全期間の定理とはしない。

## 0. 何の向きを示すか

既存の「自己項」は少なくとも三種類を区別する必要がある。

1. **自己形**: 出力とその位置微分の内積。以下では $S_c=\langle R_c,H_c\rangle_F$。
2. **自己モデルの容量微分**: 他チャネルを除いた NTK の $\frac12\log\det$ の微分。
3. **実際の切替の駆動**: 旧ラベルを学習した重みで、新しい独立ラベルに対して行う更新の平均。

既存 V12 の順列補題は、標的の gate/readout 両経路を除いた逆カーネルを固定した比較量に対するもの。有限強度の全ネットワークや実現可能な CNN の重み分布に対する証明ではない。本稿はその補題を流用せず、全行列に対する決定論的な条件を使う。

## 1. CNN、Pool、共通位置

固定した $N$ 枚の画像（または固定した上流特徴）から取り出すパッチを $x_{ns}\in\mathbb R^d$ とする。$n$ は画像、$s$ は畳み込み位置であり、$s$ ごとに別のラベルを与えない。

チャネル $c$ は一つの共有パラメータ $\theta_c=(w_c,b_c)$ を持ち、

$$z_{ncs}=\tilde x_{ns}^{\mathsf T}\theta_c,\qquad \tilde x_{ns}=(x_{ns},1).$$

Pool 領域 $B$ ごとに

$$H_{n,cB}=\max_{s\in B}\phi(z_{ncs})=\phi(\max_{s\in B}z_{ncs})$$

とする。後の等式は $\phi$ が単調非減少のときのみ使う。ReLU、leaky、softplus には使え、一般の SiLU/GELU には使わない。

全畳み込み位置に対する平均パッチを $\bar{\tilde x}$ とし、

$$m_c=\bar{\tilde x}^{\mathsf T}\theta_c,\qquad u=\bar{\tilde x},\qquad \theta_c(q)=\theta_c+qu.$$

この $q$ は bias だけを動かす変数ではない。入力を固定した SGD の一歩では

$$\Delta m_c=-\eta\,\partial_q L.$$

したがって $\partial_qL>0$ が沈める側である。$dm_c/dq=\|\bar{\tilde x}\|^2$ も正。

最大前活性の位置 $s^*_{ncB}$ が一意で、活性化の折れ目から離れている点では、

$$R_{n,cB}:=\partial_qH_{n,cB}=\phi'(z_{ncs^*})k_{ns^*},\qquad k_{ns}=\tilde x_{ns}^{\mathsf T}\bar{\tilde x}.$$

これは共有フィルタの方向微分をそのまま計算したもの。$x_{ns}\ge0$ が成分ごとに成立すれば $k_{ns}\ge1$。ReLU の場合は $H_c,R_c\ge0$ であり、$S_c=\langle R_c,H_c\rangle_F>0$ は、生きた出力が一つあれば成立する。

MaxPool の winner が保たれるのは局所的である。winner と他位置の前活性差を $\delta_s>0$ とすると、$|q|\,|k_{s^*}-k_s|<\delta_s$ を全位置で満たす区間では保たれる。bias だけなら全 $k_s=1$ なので winner は任意の共通シフトで保たれるが、平均パッチ方向ではこの追加条件が要る。

## 2. 学習後の実更新についての厳密式

### 仮定 F（readout を当てはめる模型）

- hidden CNN と画像を固定し、全チャネル・全 Pool 領域を flatten した $H\in\mathbb R^{N\times P}$ を作る。出力 bias を学習する場合は $1$ の列を足し、その列の位置微分は $0$。
- 旧ラベル行列 $Y\in\mathbb R^{N\times C}$ は、固定した hidden CNN と独立に新たに抽出し、平均 $0$、$\mathbb E[YY^{\mathsf T}]=\nu I_N$、$\nu>0$。独立な中心化一様 one-hot なら $\nu=(C-1)/C$。同じ旧ラベルで既に hidden を学習していた場合に、この独立性を後付けしてはならない。
- 線形 readout を二乗誤差と $\lambda\|V\|_F^2/2$、$\lambda>0$ で厳密に当てはめる。
- 新ラベル $Y'$ は $Y$ と独立、平均 $0$。切替直後の hidden 勾配を取るとき、学習済み $V$ は固定する（fit の写像を再微分しない）。

全 readout の解、全画像カーネル、計量は

$$V^*=H^{\mathsf T}(K+\lambda I)^{-1}Y,\quad K=HH^{\mathsf T},\quad Q=K(K+\lambda I)^{-2}.$$

**定理 F1（実際の切替勾配）**。新損失を $L'=\|HV^*-Y'\|_F^2/(2N)$ とすると、

$$\boxed{\mathbb E_{Y,Y'}[\partial_qL']=\frac\nu N\operatorname{tr}(R_c^{\mathsf T}QH_c).}$$

証明。$R$ を標的チャネルの列だけ非零の $\partial_qH$ とおく。

$$\partial_q L'=N^{-1}\operatorname{tr}[(HV^*-Y')^{\mathsf T}RV^*].$$

まず $Y'$ の平均を取り、$V^*$ を代入し、$\mathbb E[YY^{\mathsf T}]=\nu I$ を使う。$K$ と $(K+\lambda I)^{-1}$ は可換なので上式となる。□

これは近似でも弱チャネル極限でもない。他チャネルはすべて $K$ に残る。ただし「hidden 固定で readout だけを fit」の模型であり、元の全層同時訓練とは違う。

### 非循環な符号条件

正の対角行列 $D$ を選び、

$$\epsilon=\|D^{-1/2}QD^{-1/2}-I\|_2,$$
$$S_D=\langle D^{1/2}R_c,D^{1/2}H_c\rangle_F,\qquad
N_D=\|D^{1/2}R_c\|_F\|D^{1/2}H_c\|_F.$$

**定理 F2（全強度の保証）**。$S_D>0$ かつ $\epsilon<S_D/N_D$ なら

$$\mathbb E[\partial_qL']\ge\frac\nu N(S_D-\epsilon N_D)>0.$$

証明。$Q=D^{1/2}(I+E)D^{1/2}$、$\|E\|_2=\epsilon$ と書き、Frobenius の Cauchy–Schwarz 不等式で残りを抑える。□

ReLU と非負パッチなら $S_D>0$ は入力・活性化から従う。符号条件として残るのは、**全チャネルを含む当てはめ計量 $Q$ の画像間の混ざりの強さ**と、標的の出力・位置微分の角度の比較である。駆動の符号自体や、rest 項の符号は仮定していない。$D=\operatorname{diag}Q$ が一つの選択。

条件は強い。ランク不足で $Q$ が特異なら、この全空間の形では必ず通らない（$\epsilon\ge1\ge S_D/N_D$）。満たさない状態で逆向きだと断定することもできない。一方、他チャネルの強度はゼロでなくてよく、後述の実現例では ridge がカーネル全体を支配していない。

### ランク不足でも虚無にならない版

実 CNN では画像数より最終特徴数が少ないことがある。上の全空間条件だけを最終結果にすると、その箱では使えない。$\Pi$ を $\operatorname{range}K=\operatorname{range}H$ への直交射影とし、$Q$ の最大・最小の**正の**固有値の比を $\kappa_+(Q)$ とする。

**定理 F3（特徴が張る空間での保証）**。$S=\langle R_c,H_c\rangle_F>0$、

$$c_\Pi=\frac{S}{\|\Pi R_c\|_F\|H_c\|_F},\qquad
\frac{\kappa_+(Q)-1}{\kappa_+(Q)+1}<c_\Pi$$

なら、F1 の実切替勾配は正である。$H_c=\Pi H_c$ だから、自己内積は射影で変わらない。

証明。$Q$ を $\operatorname{range}K$ 上だけに制限すると正定値である。正の固有値の中点を $\alpha$、半幅を $\delta$ とすれば

$$\operatorname{tr}(R_c^{\mathsf T}QH_c)
\ge\alpha S-\delta\|\Pi R_c\|_F\|H_c\|_F>0.$$

これは rank が $N$ より小さくても成立する。また $\kappa_+(Q)\le\kappa_+(K)$。実際 $0<a\le b$ に対して $[b/(b+\lambda)^2]/[a/(a+\lambda)^2]$ は $a/b$ と $b/a$ の間なので、正の固有値のどの二つの比にもこの上限がある。□

ReLU の生きた標的要素で $t_{nB}=H_{n,cB}/R_{n,cB}\in[a,b]$、$0<a\le b<\infty$ とする。死んだ要素では両方ゼロ。$\chi=b/a$ とすると、

$$c_\Pi\ge\frac{\langle R_c,H_c\rangle_F}{\|R_c\|_F\|H_c\|_F}
\ge\frac{2\sqrt\chi}{1+\chi}.$$

最後の不等式は $t^2\le(a+b)t-ab$ を、$R_{n,cB}^2$ に比例する重みで平均し、$\mathbb E[t]/\sqrt{\mathbb E[t^2]}$ を最小化すると得られる。同じ証明は F2 の対角重みでも成立する。したがって、全カーネルの正の固有値の幅と、標的の生きた出力の相対的なばらつきだけから、符号を見る前に十分条件を判定できる。

**文字通り自己モデルとの比較**。他チャネルを除き、同じ出力 bias の扱いで head を当てはめ直した自己モデルにも F2/F3 を適用できる。全体と自己モデルの両方が条件を満たせば、実際の切替更新の向きが一致する。自己形が正というだけで、任意の自己モデルの fitted-head 勾配まで自動で正とはしない。

### 有限リッジの取り違えを避ける

F1 の計量は $Q=K(K+\lambda I)^{-2}$ であり、$P=(K+\lambda I)^{-1}$ ではない。$K$ が可逆で $\lambda\to0$ の補間極限では $Q\to K^{-1}$ となる。有限 $\lambda$ で容量微分をそのまま切替更新と呼ばない。

## 3. 容量について、文字通り自己モデルと同符号になる条件

これは F1 と別の命題。局所的にカーネルが

$$A(q)=A_0(q)+K_{-c},\qquad A_0(q)=\lambda I+K_{\rm fixed}+J_c(q)J_c(q)^{\mathsf T}$$

と分かれ、$K_{-c}\succeq0$、$K_{\rm fixed}\succeq0$ が $q$ に依存しないとする。$A_0$ は他チャネルを除いた自己モデル、$A$ は全チャネルを残したモデル。

自己と全体の容量の傾きを

$$a_0=\partial_q\tfrac12\log\det A_0,\qquad a=\partial_q\tfrac12\log\det A$$

と定義する。評価する一点で

$$X=A_0^{-1/2}J_c,\quad X'=A_0^{-1/2}J'_c,\quad B=A_0^{-1/2}K_{-c}A_0^{-1/2}$$

とおく。この白色化自体を微分するわけではない。$B$ の最小・最大固有値を $\beta_-,\beta_+$、

$$\varepsilon_B=\frac{\beta_+-\beta_-}{2+\beta_++\beta_-},\quad c_0=\frac{|a_0|}{\|X\|_F\|X'\|_F}$$

とする。

**定理 C（有限の他チャネルの下で自己符号を保持）**。$a_0\ne0$、$\varepsilon_B<c_0$ なら $\operatorname{sign}a=\operatorname{sign}a_0$。

証明。$C=(I+B)^{-1}$ とすれば $a=\langle X',CX\rangle_F$、$a_0=\langle X',X\rangle_F$。$C$ の最大・最小固有値の中点を $\alpha>0$、半幅を $\delta$ とする。すると

$$|a-\alpha a_0|\le\delta\|X\|_F\|X'\|_F,\qquad \delta/\alpha=\varepsilon_B.$$

仮定より誤差が $\alpha|a_0|$ より小さいので符号は同じ。□

他チャネルは任意に弱い必要がない。例えば $B=\beta I$ なら $\beta$ の大きさに関係なく $a=a_0/(1+\beta)$。その近傍でも厳密な余裕がある。ただし CNN でこの条件が成立するかは実現例・実状態で検査する必要があり、抽象 PSD 行列が作れるだけでは実網の証拠にしない。

### 複数 Pool 領域と共有重み

一つの Conv→ReLU/leaky→MaxPool→Flatten→線形 readout の局所領域では、winner と活性側が固定される。readout を固定すると、標的の共有フィルタに対する出力 Jacobian は

$$\nabla_{\theta_c}f_n=\sum_B v_{cB}\,\phi'(z_{ncs^*})\tilde x_{ns^*}$$

であり、$q$ に対する微分はゼロ。したがって動くカーネル部分は標的 readout の $H_cH_c^{\mathsf T}$ で、$J_c=H_c$、$J'_c=R_c$ とできる。標的自身の gate カーネルは $K_{\rm fixed}$ に残す。他チャネルの gate/readout は $K_{-c}$ に残す。複数領域間の head の符号が任意でもこの分解は成立する。

滑らかな活性化では共有フィルタの Jacobian 自体も動くため、その列も $J_c,J'_c$ に入れる。定理 C は依然使えるが、$a_0>0$ は別に示す必要がある。ReLU の局所 gate 微分がゼロであることを、閾値横断による gate 変化がないという意味に拡張しない。

## 4. CE でも厳密に示せる初期の命題

hidden を固定し、$H\ge0$、$R\ge0$ とする。クラス数 $C\ge2$、$u=\mathbf1/C$、旧ラベル $Y_n=e_{c_n}-u$ は独立一様。ゼロ readout から旧ラベルの full-batch CE を一歩だけ学習すると

$$V=\kappa H^{\mathsf T}Y,\quad F=HV=\kappa K Y,\quad \kappa=\eta/N>0.$$

独立な新一様ラベルへの切替直後について、

$$\mathbb E[\partial_qL_{\rm CE,new}]\ge0.$$

証明。新ラベルを平均すると $N^{-1}\sum_{na}R_{na}V_a^{\mathsf T}(\operatorname{softmax}F_n-u)$。$\mathbb E V_a=0$ なので $u$ は消える。旧ラベル $Y_m$ 以外を固定すると $F_n=Z+\kappa K_{nm}Y_m$ で、$Z$ は $Y_m$ と独立。$K_{nm}\ge0$。関数

$$g_Z(t)=\mathbb E_Y[Y^{\mathsf T}\operatorname{softmax}(Z+tY)]$$

は $g_Z(0)=0$、

$$g_Z'(t)=\mathbb E_Y[Y^{\mathsf T}(\operatorname{diag}p-pp^{\mathsf T})Y]>0$$

（有限 logits、$C\ge2$）なので、$t\ge0$ で非負。係数 $\kappa R_{na}H_{ma}$ がすべて非負だから全和も非負。$R_{na}>0,H_{ma}>0,K_{nm}>0$ の組があれば厳密に正。□

これは有限幅、有限歩幅、真の CE での定理で、MaxPool の選ばれた位置を含む。ただし head の一歩学習という制限は本質的。多歩学習で生まれる label-dependent な係数に、同じ独立性の証明を流用しない。

## 5. 非空例と反例

検算結果の正確な値は `results/cnn_drive_1009/verification.json` を正本とする。以下は検算する構成で、数値の全列挙を一般定理の証明の代わりにはしない。

### 非空例（学習済み output bias あり）

2 枚の $6\times6$ RGB 画像。画像 1 は左上 $5\times5$ の赤成分が 1、画像 2 は同領域の緑成分が 1、他は 0。valid $5\times5$ Conv の各対応チャネルのフィルタ係数を一様に

$$w_1:(R,G)=(1/25,1/250),\qquad w_2:(R,G)=(1/250,1/25),\quad b_1=b_2=0$$

とする。各 $2\times2$ MaxPool の左上が一意に最大となり、

$$H=\begin{pmatrix}1&0.1&1\\0.1&1&1\end{pmatrix},\qquad R_c=(89/8,89/8)^{\mathsf T}.$$

最後の列は学習する出力 bias。$\lambda=0.1$ で $Q$ の対角を $D$ にすると、$\epsilon=310812000/576632100<11/20$、$S_D/N_D=11/\sqrt{202}$。より保守的な相対幅 $\chi=10$ の下界でも $11/20<2\sqrt{10}/11$ なので、F2 の不等式は丸め誤差に頼らず厳密な余裕を持つ。他チャネルも output bias も学習し、他チャネルの出力は標的と同オーダー、$\lambda$ は $K$ の固有値 0.81 と 3.21 より小さい。

同じ画像・フィルタで現在の head を $v_1=v_2=1$ とした容量の模型も定理 C を満たす。標的自身の gate と出力 bias を残すと

$$A_0=\begin{pmatrix}28.1&2.1\\2.1&27.11\end{pmatrix},\qquad
K_{-c}=\begin{pmatrix}26.01&1.1\\1.1&27\end{pmatrix}.$$

Gershgorin の区間から $25I\prec A_0\prec31I$、$24I\prec K_{-c}\prec29I$。白色化した $B$ は $(24/31)I\prec B\prec(29/25)I$ を満たすので $\varepsilon_B<299/3049$。一方、白色化前の角度 $11/\sqrt{202}>3/4$ と $\kappa(A_0)<31/25$ から $c_0>18/31$。よって $\varepsilon_B<c_0$ が厳密に従う。数値で符号を見ることを仮定にした例ではない。自己容量微分は約 $0.40556$、全容量微分は約 $0.21353$。

画像を二組に複製した $N=4$ では $\operatorname{rank}K=2<N$ となり F2 は不成立だが、F3 は通る。この追加例は、全画像を独立なラベルで扱ったまま、ランク不足版の条件が非空であることを確認するためのもの。

### 反例（実際の切替勾配が逆になる）

同じ形状で、赤い左上 $5\times5$ の値を画像 1 で $0.1$、画像 2 で $1$ とする。標的フィルタの赤成分をすべて $0.4$、bias を 0 とすれば標的出力 $h=(1,10)^{\mathsf T}$。他チャネルの赤フィルタをすべて $86/75$、bias を $4/3$ とすれば $g=(4.2,30)^{\mathsf T}$。両者の MaxPool は左上が一意に最大。全位置の平均パッチ方向に沿う標的位置微分は

$$r=(1691/800,971/80)^{\mathsf T}>0.$$

$H=[h,g,\mathbf1]$、$\lambda=0.1$ の exact head fit では

$$r^{\mathsf T}Qh=-960510059/32310855272<0.$$

一方 $r^{\mathsf T}h>0$。旧・新 Rademacher ラベル全 16 通りの直接計算も同じ値になる（損失を画像数で割る場合はさらに $1/N$）。これは正の入力・共有 Conv・一意の MaxPool・他チャネルも output bias も学習した実現可能な反例である。「他チャネルは PSD カーネルを足すだけだから符号は変えられない」は誤り。

この反例から他チャネルだけを除き output bias を残した自己モデルでは、同じ量が $377364911/333500644>0$。したがってこの例は、自己形だけでなく、実際に当てはめた自己モデルと全モデルの切替勾配の符号反転でもある。

## 6. 全 hidden 層の共同学習を許す定理

完全な証明・非空な学習過程の構成は `joint_training.md`。全 hidden パラメータを $\theta$、最終特徴を $H(\theta)$、対象の平均前活性を $m(\theta)$ と置く。深い層の $m$ では上流も含めて $u=\nabla_\theta m$、$R=DH[u]$ と定義する。旧ラベルに依存する学習後状態について、新ラベルだけを平均した全 hidden 勾配の射影は

$$G_Y=N^{-1}\operatorname{tr}(V_Y^{\mathsf T}R_Y^{\mathsf T}H_YV_Y).$$

旧ラベルを引く前の参照 $\theta_0$ と、同じ $Y$ に対する ridge head $V_0(Y)$ を置く。参照の正の期待勾配の下界 $M_{\rm ref}>0$ は F2/F3 で示す。全旧ラベルに対して一様に

$$\|H_Y-H_0\|_2\le a,\quad\|R_Y-R_0\|_2\le b,\quad
\|V_Y-V_0(Y)\|_F\le e,\quad\|V_0(Y)\|_F\le B$$

とし、$h_0=\|H_0\|_2,r_0=\|R_0\|_2$ と置く。

**定理 J（有限の全層共同学習）**。

$$E_{\rm grad}=\frac{(B+e)^2(bh_0+r_0a+ab)+e(2B+e)r_0h_0}{N}<M_{\rm ref}$$

なら、$\mathbb E_YG_Y\ge M_{\rm ref}-E_{\rm grad}>0$。

証明。各 $Y$ について $\|R_Y^{\mathsf T}H_Y-R_0^{\mathsf T}H_0\|_2\le bh_0+r_0a+ab$。二次形式の差を、行列の差と左右の head の差の三項に分け、ノルムで抑えてから平均する。学習後の $H_Y$ に旧ラベルの等方性を再適用しない。□

head の正規方程式残差を $s$、参照の当てはめ残差上界を $E_0$ とすれば、$e\le s/\lambda+aE_0/\lambda+aB/(2\sqrt\lambda)$。出力 bias を含む全 head に $\lambda>0$ の ridge をかけた模型である。これらは表現・感度・当てはめ残差の条件であり、実際の駆動の符号は仮定していない。

厳密な正の参照の余裕があれば、有限期間の head 学習と正の有限 hidden 学習率によって、全 hidden 層を毎ステップ更新しながら J を満たす非空な領域を構成できる（`joint_training.md` §6）。二段 Conv の直接検算では、両 Conv を100ステップ更新し、参照の構造的下界 $1.197735$ に対して誤差上界 $0.613031$、従って実際の期待勾配は少なくとも $0.584704>0$。全旧ラベル列挙の実値は $1.815869$。この具体値は float64 の検算で、丸めを囲った数値証明ではない。正の学習率を許す非空性は別に厳密に証明している。

第1層の $m$ はアフィンなので、次の共通学習率 SGD 一歩では期待変化が厳密に $-\eta\mathbb E G_Y$。深い層では、滑らかなステップ区間上の Hessian 上界による $O(\eta^2)$ 項を加え、それを一次の余裕が上回る歩幅を選ぶ。ReLU の折れ目をまたいでこの Taylor 評価を無条件に使わない。これは二乗損失の共同学習についての定理であり、CE の停留条件にそのまま置換しない。

## 7. CE の多歩学習

完全な証明は `ce_training.md`。$H,R\ge0$、$h=\|H\|_2$、$d=\|R\|_2$、$G=HH^{\mathsf T}$、$Q_*=\langle R,GH\rangle_F>0$、$r_* =\max_n\sum_mG_{nm}$ とする。head 学習率の総和 $S$ が

$$0<S\le\min\left\{\frac N{r_*},\frac{2N}{h^2},\frac{8Q_*}{5\mathrm e\,Cdh^5}\right\}$$

なら、任意の有限更新回数で切替の期待勾配が正。線形なラベル応答の参照の余裕が $O(S^2)$、CE の多歩更新による誤差が $O(S^3)$ であることを明示的な定数つきで示す。旧ラベル依存の小さな hidden の変化と感度の負の成分も、追加のノルム上界で許す。

旧課題の十分な学習後まで自動で延長はできない。正の特徴 $H=\left(\begin{smallmatrix}1&2\\2&1\end{smallmatrix}\right)$、共有 Conv の平均パッチ方向 $R_{:,1}=(9,1.5)^{\mathsf T}$ で、ゼロ head から CE-GD を長く回すと、自己モデルの方向は非負なのに全モデルの期待方向が負になる。`verification.json` の `ce_head_training` は1・2・10・100・1000更新を全旧ラベルで直接計算する。「一歩の定理を繰り返せば長期も同じ」の反例である。

## 8. 実 RL-CIFAR への適用に残るもの

元の RL-CIFAR CNN は Conv→φ→MaxPool を二段、その後 FC→φ→FC→φ→出力、batch 16、CE、Adam。ここまでの定理だけでこの全体の更新を説明済みにしない。

- 深い網では標的 filter を動かすと下流の特徴と全層 Jacobian が変わる。全 NTK の容量微分は $\sum_\ell\operatorname{tr}(A^{-1}J_\ell J_\ell'{}^{\mathsf T})$ であり、標的のカーネルだけを微分した式ではない。
- 上流が更新されると $\Delta m=\bar{\tilde x}^{\mathsf T}\Delta\theta+\theta^{\mathsf T}\Delta\bar{\tilde x}+\Delta\theta^{\mathsf T}\Delta\bar{\tilde x}$。自分の更新の向きだけで全体の沈降を証明したことにはならない。
- 独立に一様抽出したミニバッチの最初の SGD 勾配は full-batch 勾配に対して不偏。batch 16 という数だけでは期待の向きは変わらない。過去の勾配に依存する Adam の分母と momentum にはこの結論をそのまま使えない。
- 一歩の条件付き期待から長期の時間平均へ移すには、条件の軌道上での維持と、雑音の制御が必要。並びの平均やラベルの平均を、そのまま時間平均と呼ばない。
- F2 や C は十分条件であり、既存の RL-CIFAR 状態が条件内にあることは未測定。既存の `per_task.csv` と hist だけでは全行列を復元できない。

## 9. 既存導出との照合

- V12 通し稿 1007: 216・320・324・417・436・446・730・745・749・1151・1171・1175 行（2026-10-09 時点）。CNN と全層 CE/Adam は既存の証明範囲外。
- `押しの向き_文の型を替えた定理_未導出4項目_1005`: 70–110、213–215、495–503、555–589 行。自己形と自己項、$P/P^0$、形式的置換と実現可能な重みの違い。
- `押しの向き_証明の進み_門の道と段ごとの格_1002`: §16.23、1555–1575 行。小ラベル展開のラベルとの揃いが logdet に一致することと、有限 ridge の実切替勾配の非勾配残差を区別。

本稿の F1 は独立に exact head fit の模型を定義し直したものなので、上記の非勾配残差を落とした式ではない。異なる模型の厳密式である。

## 10. 反復学習で自己方向が長期に残る非空な CNN 構成

完全な証明は [長期定理と反復 CNN の構成](longtime.md)。一般には、F2/F3 のスペクトル・特徴角度・振幅の条件が崩れる時刻で過程を停止し、条件付きラベル平均と実現した時間平均の差をマルチンゲールとして制御する。条件の軌道上での維持は別に必要であり、並びの平均を時間平均へ読み替えたものではない。

条件の維持まで証明できる例として、RGB の三群・各三枚、計九画像に共有 $1\times1$ Conv→ReLU→$2\times2$ MaxPool→10クラス線形 head を適用する。各チャネルが別の画像群でだけ応答する**支えの分離**を置く。出力 bias はなし。中心化 one-hot ラベルに対する二乗損失で head を毎回 exact ridge fit し、次のラベルへの full-batch SGD で全 Conv 重みと bias を更新する。新ラベルを次の旧ラベルとして再利用する通常の切替でも、旧ラベルの群内和 $S$ が $\|S\|^2\ge3-9/10=2.1$ を満たすため、新ラベルだけの条件付き平均で自己方向の駆動が厳密に正となる。

具体値 $A=2.1,B=2,\delta=0.1,\lambda=0.1,\eta=0.005$ では、各チャネルの初期応答 $h_0=0.1$、支えが崩れる上界 $H_{\rm exit}=3.9$。停止した応答の和に対する非負 supermartingale の評価から、**模型内の将来のランダムラベルに関して、少なくとも $1-3h_0/H_{\rm exit}=12/13$ の確率で全期間その支えが保たれる**。その事象では全チャネルで $h_t\to0$、平均前活性の累積変化は約 $-0.0616667$ に収束する。他チャネルを除いた自己モデルとも方向が一致する。共有 Conv の実際の自動微分による200更新と導出した再帰式の最大差は $4.44\times10^{-16}$ だった。

これは条件付きの正の駆動と長期間の累積沈降を示す例であり、勾配の時間平均が正の定数に収束するという主張ではない。この例では駆動の振幅も消え、時間平均はゼロへ向かう。支えの分離、出力 bias なし、二乗損失、exact head fit の制限を外した、実 RL-CIFAR の二段 Conv・非線形 FC・CE・Adam における長期の自己方向は未解決である。
