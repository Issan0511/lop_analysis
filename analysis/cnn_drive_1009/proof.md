# CNN の駆動源：全チャネルを残す有限の符号保証（1009）

状態: 検証中。ユーザーの依頼全体は未完了。以下は新しい条件付き定理と、無条件化を阻む反例である。実際の RL-CIFAR の二つの Conv と二つの FC を Adam・CE で同時訓練する全期間の定理とはしない。

1009 追記: §11 以降に、任意の学習済み full CNN 状態で使える CE 条件、真の Adam 更新の条件、微小 overlap の障害、Adam の定常偏りを追加した。CE の符号保証は現在、head の初期学習だけに限定されない。ただし、その新しい条件の実学習軌道での維持は未証明である。

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

## 11. 任意の学習済み CNN 状態での CE 符号保証

詳細は [ce_trained_state.md](ce_trained_state.md)。現在の全パラメータを固定し、新しい一様ラベルだけを平均する。対象の平均前活性を $m$、全パラメータ方向を $u=\nabla m$、logits と感度を $f_n,R_n=Df_n[u]$ とすると、

$$G=\mathbb E\langle\nabla m,\nabla L_{\rm new}\rangle
=\frac1N\sum_n R_n\cdot(p_n-\mathbf1/C)
=\frac1{NC}\sum_n\sum_{c<d}(p_{nc}-p_{nd})(R_{nc}-R_{nd}).$$

従って各画像で logits と感度のクラス順位が同じなら $G\ge0$、strict な対があれば $G>0$。旧ラベルと学習後特徴の独立性、head の最適当てはめ、旧学習の短さは必要ない。多クラスでは目的の総和の正をそのまま仮定する条件ではなく、各クラス対の構造を指定する十分条件である。

より弱く、各画像の top class $t$ に対して、全 loser との logit 差が $\log(C-1)$ より大きく、感度差 $R_t-R_c\ge\gamma>0$ なら、loser 同士の順位によらず $G_n\ge\gamma(p_t-1/C)>0$。固定した実状態から検査できる。

非空性は実 RL-CIFAR と同じ二段 Conv・二段 hidden FC・全 bias の構造で証明した。異なる二画像が異なる top class を持ち、全 Conv チャネルが両画像に反応し、MaxPool winner が strict な開集合を構成した。別に全 hidden 重みが正で head のクラス順位が各座標で共通な開集合では、全四層の mean に対して成立する。

第一 Conv の mean については、全層を同時 SGD 更新しても $\mathbb E\Delta m=-\eta G$ が厳密。深層 mean では全 upstream 変化を含めた $O(\eta^2)$ の余りを抑える。CE channel-ablation self と full がともに下がる非空例も示したが、その self は既存 logdet の自己項と同一視しない。

## 12. CE の期待勾配から真の Adam 更新へ

詳細は [adam_state.md](adam_state.md)。現在の CNN と過去の Adam 状態を固定し、新ラベルによる座標勾配を $g_j=\mu_j+\xi_j$ とする。過去の first moment は任意の実数、second moment は非負。正の共通係数 $\kappa$ と

$$a_j=\beta_1m_j^-,\quad b=1-\beta_1,\quad
h_j(g)=\frac1{\sqrt{\beta_2v_j^-+(1-\beta_2)g^2}+\widetilde\epsilon}$$

を使うと、実際の Adam の降下方向 $A$ は

$$\mathbb E[u^\top A]=\kappa\sum_j u_j\{(a_j+b\mu_j)\mathbb E h_j(g_j)+b\operatorname{Cov}(g_j,h_j(g_j))\}.$$

現在の新ラベルが分母にも入るため、分母を期待値で置き換えない。勾配の全ラベル範囲と分散から、分母変化の上界を求め、CE の余裕・座標間倍率差・過去 momentum・分母との共分散を分けた十分条件を示した。

さらに、クラス係数が正負対称で logits が $f_{nc}=a_ct_n+k_n$、$t_n>0$、hidden 座標感度が非負の場合は、勾配雑音が厳密に対称になる。過去 first moment が関連座標で非負なら、任意の過去 second moment、有限 epsilon、任意の有限 batch で期待 Adam 方向が正になる。odd かつ strictly increasing な関数 $g/(\sqrt{c+dg^2}+\epsilon)$ の性質から証明する。strict な余裕は小摂動後にも残る。

この正例と反例を、実 RL-CIFAR と同じ全構造に埋め込んだ。10 クラス・batch 16 の $10^{16}$ 通りの独立ラベルを有限の和へ集約し、真の共有重みと全層同時 Adam の第一 Conv mean 変化を評価した。正例では沈降、3 対 7 クラスの弱い予測偏りでは期待 SGD 勾配が正でも最初の Adam 更新平均が逆向きになる。この一歩の反例だけから長期の反転は結論しない。

## 13. 非比例な微小 overlap に対する長期証明の障害

詳細は [overlap_obstruction.md](overlap_obstruction.md)。二画像に対して、共有 depthwise Conv→ReLU→MaxPool の特徴を

$$H=\begin{pmatrix}a&\varepsilon b\\\varepsilon a&b\end{pmatrix},\qquad a,b,\varepsilon>0$$

とする。両チャネルが両画像に活性で、特徴は非比例。10 クラスの異なる旧ラベルを固定して新ラベルだけを平均すると、任意に小さい $\varepsilon$ でも $a$ を十分小さくすれば full の駆動が孤立自己モデルと逆向きになる。平均前活性方向と真の共有 Conv 勾配で検算した。

同じ状態で旧ラベルも独立に平均し直すと正になり、F1 と矛盾しない。通常のラベル再利用では旧ラベルと表現が依存するため、その再平均を黙って適用できない。この反例は「全正振幅領域で各旧ラベルに対する条件付き方向を保つ」という、前の支え分離証明の性質の一様な拡張を阻む。長期沈降そのものを否定する反例ではない。

## 14. Adam の初期反転と、長期に残る定常偏りを分ける

ユーザーの指摘どおり、§12 の最初の一歩の反例だけでは、長期の期待方向を反証できない。別に Adam の履歴を無限に伸ばした定常状態を解析した。

二点分布の一般定理は [adam_stationary_twopoint.md](adam_stationary_twopoint.md)。$g_t=q+\mu-\operatorname{Bernoulli}(q)$、$q>1/2$ では、$\mu=0$ の定常 Adam 平均が厳密に負になる。各過去勾配を一つだけ残して条件付けると、正の勾配の方が大きい分母を作ることから従う。定量的な負の余裕と正の $\mu$ の許容範囲も証明した。batch 16 のうち一画像だけが対象に反応する CNN に埋め込め、他画像のラベルも独立な10クラスのままである。

16 枚全てが寄与する場合も、[adam_stationary_b16.md](adam_stationary_b16.md) で別に証明した。

$$g_t=0.900001-K_t/16,\quad K_t\overset{\rm iid}{\sim}\mathrm{Binomial}(16,0.9),\quad
\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}.$$

この固定分布で $\mathbb E g=10^{-6}>0$ なのに、定常 Adam 方向には

$$-0.000358609<\mathbb E\frac m{\sqrt v+\epsilon}<-0.000271557<0$$

という厳密な区間が付く。定常の joint moments を有理数で求め、二次展開の残差を六次・十二次 moment で抑え、平方根は整数演算で有理区間に囲った。数値シミュレーションの符号を証明の代わりにしたものではない。独立な joint-moment 再帰でも照合した。

定常列のエルゴード性と、通常の初期値・bias correction からの差の幾何減衰により、実現した長期時間平均もこの負の定常期待へ収束する。従って Adam の勾配と分母の相関は、長期平均だけでは必ずしも消えない。

**重要な限定**: これらは固定した CNN 状態が作る iid 勾配分布への Adam の定常応答である。実際に CNN の重みを更新すれば logits と勾配分布が変わる。実 RL-CIFAR の長期軌道が逆転したとは証明していない。また、正の平均勾配が小さい構成である。実際の自己方向の余裕が定常補正を上回ることを示せれば、Adam を含む方向保証は依然可能である。


## 15. 重なる複数チャネルの CE 共同学習で、条件維持と長期沈降まで証明

詳細は [moving_ce_longtime.md](moving_ce_longtime.md)。L 段の共有 1×1 Conv→ReLU→MaxPool と、自由な多クラス・全空間位置の線形 head を用いる。bias は無し。正の単位 channel vectors u_l に対して W_l=a_l u_l u_{l-1}^T、head の channel 方向は u_L と揃え、初期 a_l=a_0>0 とする。これは channel 対称性の仮定だが、全 raw parameters を同時に学習し、pool 後の画像行列 X は full rank にできる。他チャネルは同じ画像に全て反応する。

毎 optimizer step でラベルを独立一様に引き直す。全 raw SGD が上の family を厳密に保ち、A_t を通常の CE head gradient matrix とすると、

$$a_{t+1}=a_t-\eta_t a_t^{L-1}\langle B_t,A_t\rangle,\qquad B_{t+1}=B_t-\eta_t a_t^L A_t.$$

D_t=a_t²−||B_t||²、D_0>0 とし、K=√2 max_n||x_n||、η_t=γ_t√D_t/(K a_t^L)、0<γ_t≤1/2、Σγ_t=∞、Σγ_t²<∞ と選ぶ。現在の新ラベルを見る前に決まる学習率である。このとき全ラベル実現に対して

$$(1-\gamma_t^2)D_t\le D_{t+1}\le D_t,\qquad a_t>0$$

が保たれる。さらに現在状態を固定した新ラベル平均は

$$\mathbb E[\langle B_t,A_t\rangle\mid\mathcal F_t]
=\sum_n\pi_n(B_tx_n)\cdot[\operatorname{softmax}(a_t^LB_tx_n)-\mathbf1/C]\ge0.$$

この条件付き符号、非負超マルチンゲール、head の累積駆動と連続性から、U=span{x_n} として

$$B_t\longrightarrow B_0(I-P_U),\qquad
0<a_\infty^2\le a_0^2-\|B_0P_U\|_F^2$$

がほぼ確実に成立する。従って初期 head が画像上で非自明なら、**全 hidden 層・全チャネルの最終平均前活性が初期より厳密に下がる**。一歩ごとの実現が全て下がるとは主張しない。また極限の a は正であり、ReLU の死まで示したものではない。

L=2 の第一 Conv mean 方向では、全 raw parameter の NTK K と、他の第一 Conv チャネルを除いた自己モデル K_self の双方について、方向微分 K'、K'_self が非零の半正定値となる。従って元の logdet 容量の微分は両方 strict に正。実 CE の期待勾配はその沈降方向を逆転せず、logits の class contrast が非零なら strict に同符号となる。出力が uniform になった点では CE の駆動はゼロだが容量微分は正なので、全時点の strict 一致とはしない。

検算では rank 4 の画像特徴、3 hidden channels、10 classes、二段 Conv の全 raw SGD 300 更新を導出式と照合した。parameter 誤差は 1.34e-15 未満、初期・更新後の full/self 全 NTK と微分の誤差は 3.56e-15 未満。

**範囲**: 毎更新 fresh labels、bias 無し、1×1 Conv、channel 対称 family、指定の適応 SGD。実 RL-CIFAR の task 内ラベル再利用・bias・5×5 Conv・非線形 FC・Adam を同時に扱った定理ではない。特に Adam は D の更新に一次の非相殺項を作り、この証明をそのまま移せない。

## 16. 沈降信号が小さくなる極限でも、Adam の補正は相対的に小さいとは限らない

詳細は [adam_smallscale.md](adam_smallscale.md)。固定した状態から出る iid 勾配の family g_s=sξ+s^kμ を考える。ξ は中心化有界、μ>0、epsilon>0 を固定する。定常 Adam は有限残差つきで

$$\mathbb E[A_s]=\frac{\mu s^k}{\epsilon}
-\frac{s^2}{\epsilon^2}\mathbb E[M\sqrt V]
+\frac{s^3}{\epsilon^3}\omega\mathbb E[\xi^3]+R_k(s).$$

M,V はノイズの定常一次・二次移動平均。自己方向の正の CE 平均が O(s³) なら、非対称性による O(s²) 補正が優勢になる場合がある。従って sqrt(v)≪epsilon という絶対誤差の小ささだけでは期待値の符号を保証できない。

一方、ξ の分布が厳密に対称なら、各過去勾配一つについて条件付けた奇関数の単調性により、任意の有限 s>0 で定常 Adam の期待方向が正となる。さらに

$$\left|\frac{\mathbb E[A_s]}{\mu s^k/\epsilon}-1\right|
\le\frac{s(D+\mu s^{k-1})}{\epsilon}\to0.$$

非空性として、負の平均前活性、MaxPool、両画像群で重なる二チャネル、rank 2 の特徴、10 classes、batch 16 を持つ CNN の family を構成した。正負対称なクラス係数によって、実 CE 勾配が信号 O(s³)・対称ノイズ O(s) となる。これも実際の学習軌道で対称性が維持されるという主張ではなく、固定状態での定常応答とその小振幅極限の定理である。


## 17. 長期平均で消える momentum 境界と、消えるとは限らない分母の偏り

ユーザーの長期平均に関する直観がそのまま正しい部分もある。一次移動平均 M_t=β₁M_{t−1}+(1−β₁)g_t では、各実現について厳密に

$$\frac1T\sum_{t=1}^T(M_t-g_t)
=\frac{\beta_1}{(1-\beta_1)T}(M_0-M_T).$$

従って M_T=o(T) なら、非加重の時間平均で momentum のずれは消える。一歩の momentum の向きだけでは長期反転を主張できない。

Adam で残る違いは、実更新が M_t そのものではなく、r_t M_t、r_t=(√v_t+epsilon)^{-1} だという点にある。固定 iid 勾配の定常状態では

$$\mathbb E[r_tM_t]=\mathbb E[g_t]\,\mathbb E[r_t]+\operatorname{Cov}(M_t,r_t).$$

beta を固定した二次指数移動平均は、t を増やしても有効な履歴幅が無限に伸びない。有限四次 moment を持つ iid 勾配について

$$\operatorname{Var}(v_\infty)=\frac{1-\beta_2}{1+\beta_2}\operatorname{Var}(g^2),$$

なので一般には定常状態でも分母の揺れが残る。その相関は長期平均を取るだけでは除けない。§14 が厳密な負の定常平均の例、§16 が相関を消す十分条件の一つを与える。どちらも固定分布の結論と実 CNN の変化する軌道を分ける必要がある。


## 18. 実際に変わる Adam 軌道での累積符号を判定する恒等式

詳細は [adam_cumulative_positive.md](adam_cumulative_positive.md)。固定入力の第一 Conv mean m=uᵀθ を対象とし、現在の実 CNN の勾配 g_t と条件付き平均 μ_t=E[g_t|F_{t−1}] を使う。Adam の全履歴、bias correction、現在の勾配に依存する分母を残したまま、累積下降量を

$$m_0-m_T=S_T+M_T+R_{\mathrm{den},T}+B_{\mathrm{mom},T}$$

と厳密に分ける。S_T は予測可能な正の scalar 倍で重み付けした CE 平均駆動、M_T は martingale、R_den は実分母と scalar 参照の差、B_mom は momentum の端点と参照重みの時間変化である。後二者を norm で抑え、自己方向の累積余裕と比較すれば、momentum の符号を仮定せず実 Adam の累積沈降を保証できる。正の信号が発散する場合と、有限な最終下降量の場合の確率評価は区別する。

非空な moving CNN の幾何として、入力も含めて一様な channel vectors を用いた §15 の family は通常の座標 Adam 自体にも保存される。層ごとの振幅は不揃いになり、head のクラス中心化も保たれるとは限らない。現在振幅から決める非零の学習率上限により、全将来ラベル実現で振幅の正を保てる。各時点の raw CE 平均の沈降符号と、L=2 の元の容量自己項への接続は残るが、iid ラベルで Adam の残差がその正の信号より小さいことまでは未証明。

別に、各画像を全クラスについて一回ずつ batch に入れる場合は、実勾配が新ラベル平均と厳密に一致する。この **balanced label enumeration** と上の対称 CNN・学習率上限では、Conv gradient は各実現で非負。通常の座標 Adam が全 Conv と head を共同更新しても平均前活性は単調に下がり、初期 class contrast があれば長期の net decrease は strict。この構成は iid ラベルの batch と異なり、ラベルノイズと分母の相関を除く特殊な正例である。


## 19. 課題内で同じラベルを再利用する長期 CE-SGD

詳細は [task_reuse_longtime.md](task_reuse_longtime.md)。§15 と同じ全 Conv/head が動く CNN で、新ラベルは課題の最初にだけ割り当て、課題内では任意の固定有限 H 回使い回す。画像順は毎 epoch シャッフルしてよく、各画像の総 batch 重みを揃える。

現在の振幅 a、head B、D=a²−||B||² と p=L+1 に対し、学習率を

$$\eta_{k,j}=\frac{\delta_k\sqrt{D_{k,j}}}{K a_{k,j}^L(1+a_{k,j}^{L+1})},\qquad
\delta_k=\frac{\kappa}{H(k+1)^\rho},\quad 0<\kappa\le\tfrac12,\quad\tfrac12<\rho\le1$$

とする。全ラベル実現で D は正の下限を保つ。d をその下限とすると、更新 vector field F は凸領域 a≥√(d+||B||²) 全体で有界かつ明示的な大域 Lipschitz 定数を持つ。したがって、実際の H 更新と、課題開始点で凍結した参照更新との差は一様に O(H²δ_k²)。

新ラベルの平均は課題開始時点でだけ取り、この誤差を残す。主駆動 Hδ_k の総和は発散し、相関誤差 H²δ_k² の総和は有限となる。課題境界の非負超マルチンゲールから軌道有界性を導き、head の画像上の成分が消えることを示した。結果は §15 と同じ

$$B_t\to B_0(I-P_U),\qquad 0<a_\infty^2\le a_0^2-\|B_0P_U\|_F^2.$$

従って、全 hidden mean の長期の strict net decrease と、二段モデルで元の full/self 容量の方向との一致を、毎更新ラベルを再抽選せずに証明できる。task 内の個々の勾配の符号は保証しない。H=30000 は実 RL-CIFAR のラベル・coverage スケジュールに対応できるが、指定の減衰・damping 付き SGD、対称1×1 Conv、bias無しという制限は残る。極限でも正側の ReLU であり、死の定理ではない。

## 20. 負の平均前活性・混在 gate・課題内ラベル再利用の有限期間保証

詳細は [task_reuse_finite.md](task_reuse_finite.md)。一般 CNN の全パラメータを同時に SGD 更新する有限課題を扱い、第一 Conv mean m=uᵀθ を対象にする。課題開始点のクラス順位等による平均駆動を、課題中の状態変化と比較する。半径 r の球で全ラベルの gradient norm≤G、gradient Lipschitz≤L、τ=Ση_t、τG≤r なら、

$$\left|m_0-m_H-D_{\rm frozen}(Y)\right|
\le\frac{\|u\|LG}{2}\left(\tau^2-\sum_t\eta_t^2\right).$$

同じ Y を全更新に再利用したまま比較する。期待参照の正の余裕が右辺を上回れば、実課題の期待下降を保証する。

非空例は二画像・重なる二チャネル・rank 2・全 bias trainable の Conv/ReLU/MaxPool/3-class head。mean は −0.1845、正負の ReLU sites が混在。全9ラベル割当を各8更新使い回す。学習率10⁻⁴、全パラメータ半径0.005で、期待下降には厳密な有理数下界9.59373196875e−6が付く。実 raw SGD の全列挙は約1.41899461157e−5で、mean が上がる個別実現も含む。

さらに full と literal self の全 raw NTK 容量微分を有理数で正と示し、その符号が半径0.005の球全体で保たれることも証明した。ここでは K' は不定符号であり、PSD を仮定していない。従って「全ReLUが正側」という制限を、有限課題の正例では外せた。無限反復や gate crossing は別である。

同じ画像のラベルを E epochs 使い回すと、凍結参照のラベル雑音分散は、各出現でラベルを引き直す場合の E 倍になる。独立な更新が E 倍増えたようには扱わない。この分散式も全列挙で確認した。

## 21. 過去 moments を引き継ぐ通常 Adam と、課題内ラベル再利用

詳細は [adam_task_reuse.md](adam_task_reuse.md)。課題境界で実 CNN・全 moments・global step を固定する。新しい全画像のラベル割当 Y は一度だけ引き、各 Y について「parameters は開始点に固定し、moments だけは課題の全 H 更新進める」参照を作る。この参照でも勾配と分母の相関、ラベル再利用、global bias correction、旧 moments を省かない。

全 parameter の移動量と gradient の Lipschitz 上界から、実 CNN との勾配差・一次 moment 差・二次 moment の平方根の差を抑える。平方根の差には norm の逆三角不等式を使うため、過去 second moment が0でも適用できる。参照の厳密なラベル期待 Ψ と有限誤差 E により

$$\mathbb E[m_0-m_H\mid\text{課題開始時点}]\ge\Psi-E$$

を得る。これは Adam(Eg) への置換ではない。

非空例は共有 Conv/ReLU/MaxPool、rank 2 の二画像、二チャネル、自由な二クラス空間 head、全10 raw parameters の通常 Adam。旧課題の iid ラベル (0,1) を2更新した実到達状態から、次の4通りの iid ラベル割当を各3更新再利用する。旧 moments は保持。外向き有理区間演算で期待下降下界>1.976e−5を証明した。一つのラベル割当では mean が上がり、momentum の符号も変わるため、各実現の符号を仮定した正例ではない。元の full/self 容量微分も同じ下降側にある。

この条件が無限の課題反復で保たれることは未証明。実 RL-CIFAR の1200画像では厳密全ラベル和や保守的な移動量上界が重く、その走に適用済みとは言わない。


## 22. 5×5 Conv・hidden FC・全 bias を含む、元の full/self 容量の開いた符号保証

詳細は [full_ntk_positive.md](full_ntk_positive.md)。二段の本来の空間畳み込み、ReLU、MaxPool、任意有限個の hidden FC/ReLU、自由な多クラス出力、全 raw bias を含める。1×1 カーネルを5×5の中央へ埋めただけの構成ではない。

参照状態では第一 Conv の各 augmented filter を mean augmented patch の方向に揃え、第二 Conv の input-channel 方向を対応させる。第一 Conv 以後の bias は値を0に置くが、その全 Jacobian 列を含める。正の scalar t に対する下流の正斉次性から、元の全 raw NTK について

$$K'_{\rm full}=2a_1 k(G_2+G_{\rm suffix}),\qquad
K'_{\rm self}=2a_1 k(G_2+u_{1c}^2G_{\rm suffix})$$

を得る。G_2 は第二 Conv の全空間offsetの weight block、G_suffix は以後の全 Conv/FC/head weight block。bias blocks の方向微分は0だが、容量の逆行列には全て含む。

最終 hidden feature の画像行列 F が full row rank なら、自由な head block が (FFᵀ)⊗I を含み、両 K' に正の固有値下限が付く。さらに J と J'=D_uJ の参照からの norm 距離を使う有限誤差評価で、**全 raw parameter 空間の開集合**へ拡張する。実状態のチャネルは非比例、各 bias は非零でもよく、参照の対称性を実状態にそのまま要求しない。新ラベル CE の方向も別の logit/JVP 誤差評価で同じ沈降側と示す。

二段5×5 Conv・全 bias・10クラス、444 raw parameters の検算では、全座標に独立な有限摂動を加えた非比例チャネルの状態で、full/self K' の正の下限と CE 方向の余裕を確認した。hidden FC 追加は、full row rank を保つ正の重みの明示構成と全 Jacobian block の導出で解析的に扱う。32×32 RGB・16→16 Conv・FC100→100 についても、二画像の局所 bump と receptive field の違いから非零の特徴minorを作り、mean patchを保つ微小な画像摂動でpool tiesを外す解析構成を独立に確認した。

これで実 RL-CIFAR 型の Conv/FC 構造に対する元の自己項の条件付き符号保証へ進めた。ただし固定状態・開近傍の定理であり、通常 Adam の長期軌道がその条件を維持するとはまだ証明していない。
