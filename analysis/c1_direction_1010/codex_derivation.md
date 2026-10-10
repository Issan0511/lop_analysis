# 隠れ層に読まれる層の押しの向き — 独立導出と批評

作成: 2026-10-10（Codex、学習・GPU 実行なし）

格は `[厳密]`、`[恒等式]`、`[仮定つき]`、`[候補]`、`[書けない]` のいずれかを付ける。以下では

\[
L_u={1\over N}\sum_n\left(\operatorname{logsumexp}f_n-{1\over K}\sum_c f_{nc}\right)
\]

とし、すべての \(\delta_l=\partial L_u/\partial z_l\) はすでに \(1/N\) を含む。この規約なら以下の画像・位置和に追加の \(1/N\) はない。

## 1. 式 (1)〜(4) の検算

### 1.1 式 (1)

[恒等式] zero-pad 込みの入力 patch を \(x_{np}\)、その全画像・全位置平均を

\[
M={1\over N\,32^2}\sum_{n,p}x_{np}
\]

とすれば、\(m_j=\langle W_{1j},M\rangle+b_{1j}\)、したがって \(u_j=\nabla m_j=(M,1)\) である。\(u_j\) 方向に動かしたときの局所 preactivation の接線は

\[
K_1(n,p)=\langle M,x_{np}\rangle+1
\]

なので、固定状態で

\[
G_j=\langle\nabla m_j,\nabla L_u\rangle
=\sum_{n,p}\delta_{1j}(n,p)K_1(n,p).
\]

[恒等式] 第 1 pool の各窓 \(q\) で autograd が選んだ winner を \(p^*(n,j,q)\) とすると、

\[
G_j=\sum_{n,q}\gamma_j(n,p^*)K_1(n,p^*)H_{1j}(n,q),
\qquad \gamma_j=\phi'(z_{1j}). \tag{1}
\]

よって (1) の正規化と符号は正しい。\(p^*\) は \(n,j,q\) のすべてに依存し、\(K_1\) は必ず winner の位置で評価する。

[厳密] max-pool tie または leaky ReLU の \(z=0\) では古典的な微分は一意でない。PyTorch が選ぶ winner と片側微分を固定すれば (1) は実装上の恒等式であり、分枝を固定しない古典微分の恒等式とは書けない。

[恒等式] \(K_1\) の位置平均は \(\overline K_1=\|M\|^2+1\) だが、各 \(K_1(n,p)\) がこの値なのではない。raw CIFAR は画素が \([0,1]\) なので \(K_1\ge1\) だが、画像・位置による変動は残る。

### 1.2 式 (2)

[恒等式] full state で得た \(\delta_2\)、gate、pool route を固定すると、\(H_1=C_{W_2}^{\!*}\delta_2\) は \(W_2\) に線形である。従って

\[
W_2=W_2^0+\Delta W_2^{\rm drift}+\Delta W_2^{\rm rest}
\quad\Longrightarrow\quad
G_j=G_j^0+G_j^{\rm drift}+G_j^{\rm rest} \tag{2}
\]

は厳密である。

[厳密] (2) は現在の cotangent \(\delta_2\) を三つの重み経路へ流す代数分解であり、三つの counterfactual network の loss gradient の和ではない。各 \(W_2\) 部分で forward/backward を再計算して \(\delta_2\) まで変えれば、(2) は成立しない。

[恒等式] \(\mu_2=\operatorname{mean}_{n,q'}\operatorname{unfold}(P_1)\ne0\) とすれば、行ごとの射影

\[
\Delta W_{2k}^{\rm drift}
=\langle\Delta W_{2k},\widehat\mu_2\rangle\widehat\mu_2
\]

とその直交残差の分解は厳密であり、\(\langle\Delta W_{2k}^{\rm drift},\mu_2\rangle=\langle\Delta W_{2k},\mu_2\rangle\) である。\(\mu_2=0\) ならこの射影は定義できない。

[恒等式] 現在の \(\mu_2\) を固定した重み寄与は厳密だが、初期から現在までの c2 mean の実変化全体ではない。正確には

\[
\bar z_{2k}(t)-\bar z_{2k}(0)
=\langle\Delta W_{2k},\mu_2(t)\rangle
+\langle W_{2k}^0,\mu_2(t)-\mu_2(0)\rangle
+\Delta b_{2k}.
\]

従って `drift` は「現在の平均入力に沿う重み変化」であり、上流変化と bias を含む c2 沈降全体ではない。

[候補] \(W_2^0\) が対称乱数でも、現在の \(\delta_2\)、gate、winner は同じ \(W_2^0\) を含む学習軌道の関数である。従って \(G^0\) が半々になることは恒等式からは出ず、P1 の「init は偶然帯」は独立性を仮定しない限り実測予測に留まる。

### 1.3 式 (3) の境界項

[恒等式] \(\Lambda=\{0,\ldots,15\}^2\)、offset 集合 \({\cal O}=\{-2,\ldots,2\}^2\) とし、forward conv を

\[
z_{2k}(n,t)=\sum_{j,o}W_{2,kjo}P_{1j}(n,t+o)\mathbf1\{t+o\in\Lambda\}+b_{2k}
\]

と書く。\(V_o=\{t\in\Lambda:t+o\in\Lambda\}\)、

\[
D_k=\sum_{n,t\in\Lambda}\delta_{2k}(n,t),\qquad
D_{k,o}=\sum_{n,t\in V_o}\delta_{2k}(n,t)
\]

とおけば、正確な和は

\[
\sum_{n,q}H_{1j}(n,q)
=\sum_{k,o}W_{2,kjo}D_{k,o}
=\sum_k s_{kj}D_k+B_j, \tag{3-exact}
\]

\[
s_{kj}=\sum_oW_{2,kjo},\qquad
B_j=-\sum_{k,o}W_{2,kjo}
       \sum_{n,t\notin V_o}\delta_{2k}(n,t).
\]

[恒等式] \(D_k=\partial L_u/\partial b_{2k}\) は c2 bias 座標の押しである。ただし c2 の full mean 押し \(G_k>0\) から \(D_k>0\) は導けない。前者には \(W_2\) と upstream \(W_1,b_1\) の座標も入るため、`D̄_k > 0 は既知` ではなく P3 で直接測る新しい仮定である。

[仮定つき] derivation.md の (3) は \(B_j\) を捨てた近似である。periodic padding、境界 tap が 0、または各 missing strip 上の \(\delta_2\) の和が 0 なら等式になる。

[厳密] 5×5、pad 2、16×16 では \(|V_o|=(16-|o_1|)(16-|o_2|)\) である。全 site-offset 対の幾何学的な平均欠落率は 14.4375%、角 offset の最大欠落率は 23.4375% である。これは誤差の上界ではなく、\(\delta_2\) や \(W_2\) が境界に集中したり \(s_{kj}\) が cancellation したりすれば、\(B_j\) が主項より大きく符号を反転できる。

### 1.4 式 (4) の正確な出発点

[恒等式] c1 の pooled map 上で

\[
A_j(n,q)=\gamma_j(n,p^*)K_1(n,p^*)
\]

とし、\(\mu_{j,o}\) を \(\mu_2\) の channel \(j\)、offset \(o\) 成分とする。\(\langle\Delta W_{2k},\widehat\mu_2\rangle=-c_k\) なら、zero padding を残した exact master identity は

\[
G_j^{\rm drift}
=-{1\over\|\mu_2\|}
  \sum_k c_k\sum_o\mu_{j,o}
  \sum_{n,t\in V_o} A_j(n,t+o)\delta_{2k}(n,t). \tag{4-exact}
\]

[仮定つき] 境界を無視し、\(A_j\equiv\bar A_j>0\)、\(\mu_{j,o}\approx\bar P_{1j}\) とすれば

\[
G_j^{\rm drift}
\approx-\bar P_{1j}{25\bar A_j\over\|\mu_2\|}
       \sum_k c_kD_k. \tag{4}
\]

従って derivation.md の (4) の符号は、この近似の下では正しい。定数にすべき量は \(\overline{\gamma K_1}\) であり、これを \(\bar\gamma\,\bar K_1\) と書くにはさらに \(\operatorname{Cov}(\gamma,K_1)=0\) が要る。

[恒等式] zero padding 下の実際の block は

\[
\mu_{j,o}={1\over N|\Lambda|}\sum_{n,t\in V_o}P_{1j}(n,t+o).
\]

空間定常を仮定しても \(\mu_{j,o}\approx v_o\bar P_{1j}\)、\(v_o=|V_o|/256\) であり、\(\bar P_{1j}\mathbf1_{25}\) ではない。\(\sum_ov_o=21.390625\) であり、\(\delta_2\) も空間一様なら padding が両側に入り \(\sum_ov_o^2\approx18.3961\) となるので、理想係数 25 から約 26% ずれ得る。

[仮定つき] (4) が壊れる主な条件は、(i) \(A_j=\gamma K_1\) と shifted \(\delta_2\) の相関、(ii) leaky の \(\gamma\in\{0.1,1\}\) または Snake の \(\gamma\in[0,2]\) の空間変動、(iii) winner と \(K_1\) の相関、(iv) c2 pool 由来の \(\delta_2\) の空間疎性、(v) boundary concentration、(vi) \(\mu_2\) block の非定常・符号混在、(vii) \(\sum_kc_kD_k\) の cancellation である。std 入力では \(K_1\) 自身の符号変化も加わる。

[書けない] (a) \(c_k>0\)、(b) \(D_k>0\)、(c) \(\bar P_{1j}\) の符号だけから、実 grid 上の \(G_j^{\rm drift}\) の符号は書けない。少なくとも shifted \(\gamma K_1\)-\(\delta_2\) 相関、boundary、block shape の余りが主項より小さいことが要る。

## 2. 「2 層では透過項が無い」の正確な文

[恒等式] 直接 softmax readout を持つ 2 層 MLP を \(f_n=Vh_n+b\) とし、

\[
e_n={p_n-\mathbf1/K\over N},\qquad
D=\sum_ne_n=\bar p-\mathbf1/K=\nabla_bL_u
\]

とする。hidden unit \(i\) への cotangent は \(H_i(n)=V_{:i}^{\!T}e_n\) である。\(a_{in}=K_n\phi'(z_{in})\)、\(\bar a_i=N^{-1}\sum_na_{in}\) とすれば

\[
G_i=\bar a_iV_{:i}^{\!T}D
 +\sum_n(a_{in}-\bar a_i)V_{:i}^{\!T}e_n. \tag{5}
\]

第 1 項が画像定数モードの透過項、第 2 項が画像中心化された項である。

[厳密] softmax が与える厳密な恒等式は \(\mathbf1^TD=0\) だけであり、各 \(D_c=0\) ではない。従って readout 列の class-common 成分は消えるが、class-centered 成分との内積 \(V_{:i}^TD\) は一般には残る。

[厳密] 定数モードが消える必要十分条件は \(\bar a_iV_{:i}^TD=0\) である。十分条件は、(i) aggregate prediction \(\bar p\) が一様、(ii) \(L_u\) が output bias について停留、(iii) \(V_{:i}\perp D\)、または (iv) \(\bar a_i=0\) である。

[仮定つき] 旧課題 CE が output bias について停留し、かつ旧ラベルの度数が厳密に各 \(N/K\) なら \(D=0\) になる。有限個の iid 乱数ラベルでは、旧 CE の bias 勾配がほぼ 0 でも \(D\approx\widehat\pi_{\rm old}-\mathbf1/K\) であり、固定された一課題では通常 0 でない。

[仮定つき] 旧ラベルをほぼ fit し、度数揺らぎを iid multinomial とみなすと

\[
\mathbb E\|D\|_2^2={1-1/K\over N}.
\]

\(N=1200,K=10\) では各成分の標準偏差は 0.00866、\(\|D\|_2\) の rms は 0.0274 である。従って

\[
|T_i|\le |\bar a_i|\,
\|V_{:i}-\overline V_i\mathbf1\|_2\,\|D\|_2
=O_p(N^{-1/2})
\]

は、readout norm と \(\bar a_i\) が \(O(1)\) という追加仮定の下での見積もりである。未正規化 residual \(p-1/K\) を使う規約なら \(D,T_i\) はともに \(N\) 倍になる。

[書けない] \(V\) は同じ旧ラベルで育っており度数揺らぎと独立ではないため、透過項の符号が半々とは書けない。未 fit、bias 非停留、または中心化項が小さい場合には透過項が無視できるとも書けない。

[仮定つき] 推奨する文は次である。「2 層では、uniform-target の output-bias gradient が 0、または各 readout 列がそれと直交するなら、入力定数モードの透過項は厳密に消える。fit 済み iid 有限標本では通常 \(O_p(N^{-1/2})\) の残差であり、構造的な 0 ではない。」

## 3. 透過項の非循環な符号定理と反例

### 3.1 CNN 版の十分条件

[恒等式] \(C_v\) を kernel \(v\) の zero-padded forward convolution とする。\(v_j=\mu_{2,j}/\|\mu_2\|\) を \(\widehat\mu_2\) の channel \(j\) block とすれば、(4-exact) は簡潔に

\[
G_j^{\rm drift}=-\sum_kc_k\langle C_{v_j}A_j,\delta_{2k}\rangle. \tag{6}
\]

[仮定つき] \(\bar P_{1j}\ne0\) とし、実 block を

\[
v_j={\bar P_{1j}\over\|\mu_2\|}\mathbf1+r_j
\]

と分ける。\(B_j=C_{\mathbf1}A_j\)、\(\bar B_j\) をその画像・位置平均とし、次を仮定する。

1. \(c_k\ge0\) で少なくとも一つは正である。これは過去の学習で得た state 量を直接測る仮定であり、現在の \(G_j\) の符号を仮定しない。
2. \(A_j\ge0\)、\(\bar B_j>0\) である。
3. \(C=\sum_kc_kD_k>0\) で、gate・位置相関の余り
   \[
   E_{\rm gate}=\sum_kc_k\langle B_j-\bar B_j,\delta_{2k}\rangle
   \]
   が \(|E_{\rm gate}|<\bar B_jC\) を満たす。
4. block の AC・boundary 余り
   \[
   E_{\rm ac}=\sum_kc_k\langle C_{r_j}A_j,\delta_{2k}\rangle
   \]
   が
   \[
   |E_{\rm ac}|<{|\bar P_{1j}|\over\|\mu_2\|}
   \left(\bar B_jC+E_{\rm gate}\right)
   \]
   を満たす。

[仮定つき] 上の仮定の下では

\[
\operatorname{sign}G_j^{\rm drift}=-\operatorname{sign}\bar P_{1j}. \tag{7}
\]

[厳密] 証明は (6) に block 分解を代入する一行である。括弧

\[
Z=\sum_kc_k\langle B_j,\delta_{2k}\rangle
=\bar B_jC+E_{\rm gate}>0
\]

に対し、\(G_j^{\rm drift}=-(\bar P_{1j}/\|\mu_2\|)Z-E_{\rm ac}\) であり、仮定 4 が主項の符号を守る。

[厳密] この定理は循環していない。\(c_k\) は過去の weight displacement、\(D_k\) は現在の downstream bias gradient、\(A_j,\bar P_{1j}\) は forward/route 量、二つの余りは測定可能な相関であり、結論の \(G_j^{\rm drift}\) の符号を前提にしていない。

[仮定つき] より強いが簡単な十分条件は、\(r_j=0\)、\(c_k\ge0\)、\(A_j\ge0\)、\(\delta_{2k}(n,t)\ge0\) が点ごとに成り立ち、どこかで積が正となることである。この場合は covariance 仮定なしで (7) が出るが、実 CNN の \(\delta_2\) に点ごとの同符号は期待できない。

[厳密] 「\(c_k>0\) の率」「\(D_k>0\) の率」が各 0.85 という周辺率だけでは \(C>0\) すら保証しない。負の少数チャネルの大きさが正の多数を上回り得るため、P3 では率に加えて \(\sum_kc_kD_k\) と余裕を直接出す必要がある。

### 3.2 明示反例

[厳密] gate 相関だけで反転する 2 点 MLP 反例がある。\(\mu_i=1,c=1\)、\(\delta_2=(2,-1)\) とすれば \(D=1>0\) である。\(K=1\)、leaky gate により \(A=(0.1,1)\)、\(\Delta W_i^{\rm drift}=-1\) とすると

\[
G_i^{\rm drift}=-\sum_nA_n\delta_{2n}=-(0.2-1)=+0.8.
\]

\(-\operatorname{sign}\mu_i\) は負を予言するが実際は正である。\(A>0,c>0,D>0\) だけでは無相関条件の代わりにならない。

[厳密] AC kernel と gate の空間相関でも反転する。2-site circular conv で \(\delta=(2,0)\)、drift kernel を正の scale を除いて \(w=(1/2,-3/2)\) とすると、DC 和は \(-1\)、\(H=(1,-3)\)、\(\sum H=-2=sD\) である。ところが \(A=(1,0.1)\) なら

\[
G=A\cdot H=0.7>0,
\]

一方、定数化は \(\bar A\sum H=0.55(-2)=-1.1<0\) を与える。すなわち DC 和の符号が正しくても AC×gate 相関だけで (1) の符号は逆転する。\(-w\) は DC が正で AC が大きい \(\mu_2\) block として読める。

[厳密] boundary だけでも、\(D_k>0\) だが特定 offset の valid 領域和 \(D_{k,o}<0\) となるよう、missing boundary に大きな正の \(\delta_2\) を置ける。その場合 (3-exact) の符号は DC 近似から反転する。

[書けない] これらの代数反例が通常の RL-CIFAR 学習軌道で高頻度に実現するとは書けない。示しているのは、(a)〜(c) だけでは定理にならず、相関・boundary の余裕が論理的に必要だということである。

## 4. c2 が自己形に従う理由の候補

[候補] 「c2 を読む \(W_3\) は位置別なので centered/self-consistent 項を作りやすく、c1 を読む共有 \(W_2\) ではそれが弱い」はもっともらしい。標的 1 channel 当たりの outgoing parameter は、\(W_3[:,k,:,:]\) が \(100\times64=6400\)、\(W_2[:,j,:,:]\) が \(16\times25=400\) で 16 倍違う。

[厳密] ただし \(W_3\) も画像ごとに別の parameter を持つわけではなく、全画像で共有される。画像方向の有効 rank は pooled-feature 行列と 100-unit bottleneck に制限される。一方、共有 \(W_2\) に入る \(\delta_2(n,q)\) 自体は dense head 由来の画像依存性を運ぶため、「\(W_2\) は画像記憶を持てない」は強すぎる。

[厳密] \(\Delta W_3^{\rm rest}\perp\mu_3\) は「現在の平均入力方向でない」というだけで、自己整合項と同義ではない。そこには centered memory、AC、旧 \(\mu_3\) 方向、識別方向が混ざる。従って `rest > 0` が測れても、それだけで位置別記憶説は確定しない。

[候補] 別の説明は、c1 では \(D_2\) が揃って大きな DC 透過を作るが、c2 では \(D_3=\sum_n\delta_3\) が unit 間で相殺して透過自体が小さい、というものである。この場合、c2 で self/rest が特別に強い必要はない。

[候補] ほかに、c1 の transpose-conv では channel・offset・boundary 間の cancellation が強いこと、c2 は head に近く位置別 readout が centered-feature covariance を作りやすいこと、c2 mean 方向の接線が upstream \(W_1,b_1\) 成分も含むことが候補になる。

[厳密] 既存 LR の c2 は中心化自己形の符号一致が高い一方、\(G\) と自己形の順位相関は弱い。従って既知なのは主に符号則であり、\(H_2\propto P_2\) という大きさの比例までは観測済みでない。

[候補] 次の測定で説明を区別できる。

1. c2 の \(G_k\) を own block \((W_{2k},b_{2k})\) と upstream block \((W_1,b_1)\) に厳密分解する。
2. \(D_{3u}\)、\(c_u=-\langle\Delta W_{3u},\widehat\mu_3\rangle\)、\(G_k^{\rm drift}\) を直接測る。透過が小さいなら \(D_3\) 相殺説、drift が負で大きいのに rest が勝つなら競合説を支持する。
3. \(G_k=\langle T_k,H_{2k}\rangle\) を DC と centered covariance に厳密分けし、\(H_{2k}^{\rm rest}\) を centered \(P_{2k}\) に回帰して、傾き、寄与率、残差を出す。
4. \(\Delta W_3^{\rm rest}\) を centered feature 行列 \(\{\operatorname{vec}P_{2n}-\mu_3\}\) の row-space と null-space に分ける。画像記憶説なら row-space の寄与が正を担うはずである。
5. \(\delta_3\) を固定した post-hoc 計算で、\(W_3\) の 64 位置を channel 内で permute または tie して \(G_2\) を再計算する。元配置だけで正寄与が強ければ位置対応説を支持する。因果判定には、将来、parameter 数を合わせた shared/local readout の再学習が要る。
6. parameter 数だけでなく、c1/c2 の画像 Gram の diagonal mass、effective rank、\(H_l\) を \(P_l\) で説明する \(R^2\) を比較する。

[書けない] 現在の保存済み符号率だけから、c2 の正符号を「位置別 \(W_3\) の画像記憶が原因」と一意に決めることは書けない。

## 5. Adam は透過項の符号を変え得るか

[厳密] 現在状態と過去 Adam state を固定し、座標勾配を \(g_r\)、平均を \(\mu_r\)、平均方向を \(u_r\) とすると、一歩の期待降下方向は正の共通係数を除いて

\[
\sum_r u_r\left[
 (\beta_1m_r^-+(1-\beta_1)\mu_r)\,\mathbb E h_r(g_r)
 +(1-\beta_1)\operatorname{Cov}(g_r,h_r(g_r))
\right],
\]

\[
h_r(g)={1\over\sqrt{\beta_2v_r^-+(1-\beta_2)g^2}+\widetilde\epsilon}.
\]

従って過去 momentum、座標ごとの倍率差、現在勾配と分母の共分散の三つが SGD の \(u\cdot\mu\) の符号を変え得る。

[厳密] 正の対角倍率だけでも一般の内積符号は保存しない。例えば \(u=(1,1)\)、\(\mu=(2,-1)\) なら \(u\cdot\mu=1>0\) だが、\(D=\operatorname{diag}(0.1,1)\) では \(u\cdot D\mu=-0.8\) である。

[厳密] ただし純粋な rank-one 平均経路 \(g_{kr}=D_k\mu_r\)、\(D_k>0\) に正の対角前処理だけを掛けるなら、\(\mu\) 射影は \(D_k\sum_rd_{kr}\mu_r^2>0\) であり、反平行性そのものは反転しない。反転には rest との混合、過去 momentum、または確率的な分母相関が要る。

[厳密] \(G^0+G^{\rm drift}+G^{\rm rest}\) は SGD 期待勾配の線形帳簿である。Adam は三成分の和である stochastic gradient から一つの \(v\) を作るため、各成分を別々に Adam 化した量の加法分解は一般にない。

[候補] 既知の c1 の最初の一歩の有意な反転が高々 1.7% であることは、Adam が c1 の自己形不一致の主因ではないことを強く支持する。少なくとも観測された多数の \(G_j<0\) を Adam で説明する必要はない。

[書けない] この一歩の低反転率から、各 component の dominance が Adam 後にも保存されること、75/750 step の累積符号、または長期の符号保存までは書けない。

## 6. 3 層 MLP の raw/std と「W3 の成長 × 初期 W2 列」

[恒等式] 3 層 MLP で

\[
a_{in}=K_n\phi_1'(z_{1in}),\qquad
e_n={p_n-\mathbf1/K\over N},\qquad
\Gamma_n=\operatorname{diag}\phi_2'(z_{2n})
\]

とすれば

\[
G_i=\sum_n a_{in}\,(W_2[:,i])^T\Gamma_nW_3^Te_n. \tag{8}
\]

固定状態で \(W_2=W_2^0+W_2^{\rm drift}+W_2^{\rm rest}\)、\(W_3=W_3^0+\Delta W_3\) と分ければ

\[
G_i=\sum_{a\in\{0,{\rm drift},{\rm rest}\}}
    \sum_{b\in\{0,\Delta\}}G_i^{a,b} \tag{9}
\]

が厳密である。ここで

\[
G_i^{a,b}=\sum_n a_{in}(W_2^a[:,i])^T\Gamma_n(W_3^b)^Te_n.
\]

[厳密] 旧ノートの「\(W_3\) の成長 × 初期の \(W_2\) 列」は主に \(G_i^{0,\Delta}\) である。今回の透過候補は \(G_i^{{\rm drift},0}+G_i^{{\rm drift},\Delta}\) であり、別の項である。従って後者を前者の「正体」とは書けない。

[厳密] 現 spec の \(W_2\) だけの分解では

\[
G_i^0=G_i^{0,0}+G_i^{0,\Delta}
\]

なので、旧ノートが名指した項は `init` 側に入る。\(W_2^0\) が乱数でも、gate、\(e_n\)、\(\Delta W_3\) は同じ \(W_2^0\) を通って育つため、\(G_i^0\) の符号が半々とは限らない。

[恒等式] MLP では convolution boundary がなく、\(\mu_{2,i}=\bar h_{1i}\) がそのまま 1 coordinate である。従って

\[
G_i^{\rm drift}
=-{\bar h_{1i}\over\|\mu_2\|}
  \sum_kc_k\sum_na_{in}\delta_{2k}(n). \tag{10}
\]

[仮定つき] \(c_k>0\)、\(D_k=\sum_n\delta_{2k}(n)>0\)、\(a_{in}\) と \(\delta_{2k}(n)\) の相関余りが主項より小さいなら、(10) は \(\operatorname{sign}G_i^{\rm drift}=-\operatorname{sign}\bar h_{1i}\) を与える。従って raw の \(\bar h_{1i}<0\) では沈める側、std の \(\bar h_{1i}>0\) では浮かせる側となり、観測された raw/std の差とは整合する。

[候補] raw では drift と自己整合が同符号、std では競合する、という読みは有力な追加機構である。しかし std では \(K_n<0\) が約 36% あり、\(a_{in}\)-\(\delta_2\) 相関を捨てにくいので、raw より符号予測が弱い。

[候補] 決定的な測定は (9) の 3×2 表を raw/std 別に出し、特に \(G^{0,\Delta}\) と \(G^{{\rm drift},*}\) の符号・絶対寄与を比較することである。現 P8 の三分解だけでは旧候補と新候補を区別できない。

[書けない] 既知の全体符号率 84〜97% / 37〜67% だけから、どちらの交差項が支配したか、また P8 の 0.85 / 0.70 の率になるかは書けない。

## 7. P1〜P9 の事前批評と私の対抗予測

[候補] 総評は Q1=`DRIFT_SIGN_ONLY` 寄り、Q3=`PARTIAL` 寄りである。Q2 は現 P9(a) では必要条件も十分条件も測っていないため、その判定規則を変えるべきだと考える。

| 予測 | 格 | 批評と対抗予測 |
|---|---|---|
| P1 | `[候補]` | \(G^{\rm drift}<0\) は最も残りやすいが、0.85 と dominance 0.70 は強い。対抗は符号一致 0.65〜0.85、dominance 0.5〜0.7 の `DRIFT_SIGN_ONLY`。また \(G^0\) は current \(\delta_2\) と依存するため偶然帯とは限らない。dominance は \(|G^{\rm drift}|>|G^{\rm rest}|\) でなく、少なくとも \(|G^{\rm drift}|>|G^0+G^{\rm rest}|\) でも判定すべきである。 |
| P2 | `[候補]` | c2 で \(G^0+G^{\rm rest}>0\) が支配することはあり得るが、rest 単独を自己項とは同定できない。\(G^{\rm drift}<0\) は \(D_3\) 未知なので、対抗は drift の符号が偶然帯、正の `non-drift` が支配である。 |
| P3 | `[候補]` | c2 の \(c_k>0\) が多数という予測はもっともらしい。一方、fc1 の \(c_u>0\) と \(D_k>0\ge0.85\) は平均 preactivation の沈降や full \(G_k>0\) から出ない。率だけでなく \(\sum_kc_kD_k\)、負側の magnitude、相関余りを本判定量にするべきである。`[書けない]` 具体的な率は計算前には書けない。 |
| P4 | `[候補]` | \(t20>t1\) の endpoint はあり得るが、移動する \(\mu_2\)、gate、\(D_k\)、component cancellation のため全時点の単調増加は外れやすい。対抗は「上昇傾向だが非単調、t5〜t10 で peak/plateau」である。 |
| P5 | `[候補]` | \(\bar P_{1j}\le0\) は zero-pad 込みの \(\mu_2\) block 全体の符号を保証せず、0 近傍ほど AC・boundary 余りが勝つ。無閾値の 0.70 は外れ得る。対抗は該当不足または 0.5〜0.65。\(\sum_o\mu_{j,o}< -\epsilon\) か、主項/余り margin で条件付ければ正への反転は残る。 |
| P6 | `[候補]` | Snake 4 腕を束ねた drift 負 0.70 は強い。\(\gamma\) の空間相関と既知の腕間差から、対抗は腕別 0.5〜0.7 である。full が半々という事実だけから drift/rest の拮抗は出ない。 |
| P7 | `[候補]` | 入力中心化で \(K_1\approx1\) なら定数化は改善し得るので、drift 負は P1 より保ちやすい可能性がある。`[書けない]` ただし 0.85、full c1 \(\le0.40\)、c2 \(\ge0.85\) はこの導出だけでは書けない。対抗は drift 負 0.70〜0.85、full の率は未指定である。 |
| P8 | `[候補]` | raw の drift 正は 0.7〜0.85 程度あり得る。std の drift 負 0.70 と rest 正 0.70 は \(K_n<0\) と相関のため外れやすい。対抗は std drift が偶然帯〜0.65、\(G^{0,\Delta}\) が大きく、Q3 は raw のみ成立する `PARTIAL`。 |
| P9 | `[厳密]` | (a) は Q2 の必要条件でも十分条件でもない。結合が一定でも \(D_k(t)\) が短い正から長い負へ変われば、最初に浮いて正味で沈める。\(\sum_k\langle\Delta W_{2k},\widehat\mu_2\rangle\) は channel \(j\)、\(D_k\) 重み、Adam 更新を落とし、signed cancellation のため「50%」も不安定である。`[候補]` 対抗は変化が 75〜750 step に分散し、(b) は step 75 のみ強く 750 では弱化または反転。Q2 には task-start の \(W_2\) を固定する反実仮想と evolving \(W_2\) の累積 \(\Delta m_1\) を比較する。固定側でも沈めば「結合強化が必要」は棄却する。 |

[厳密] P9(b) の snapshot SGD 勾配は累積 Adam 変位ではない。Q2 の帳簿には、各 step の実 Adam preconditioner を共通にした full/component の射影を時間積分するか、少なくとも固定-state counterfactual と実軌道の差を取る必要がある。

[書けない] 学習を実行していないため、上の対抗帯を超える正確な率、seed 数、最終 verdict は書けない。

## 8. 結論

[恒等式] (1) は固定 route の下で正しく、(2) は current \(\delta_2\) を固定した線形帳簿として正しい。

[恒等式] zero padding 下の (3) には (3-exact) の境界残差があり、\(\mu_2\) block にも offset ごとの coverage が入る。

[仮定つき] (4) の符号則は、反平行 weight 成分、正の weighted downstream bias push、\(\bar P_{1j}\) の符号に加え、gate・boundary・AC の余りが主項を下回るときの十分条件として書ける。

[候補] この機構は LR c1 と 3 層 MLP raw/std を同じ符号則で読む有力候補だが、`drift = 原因、rest = 自己整合、init = 偶然` という三つの因果ラベルは現分解からは出ない。

[書けない] 現時点では c1 の外れの大きさ、c2 の自己形優勢の原因、課題内の正味沈降をこの一機構だけで確定することは書けない。
