# DC 和の透過と、その履歴 — 独立導出（第 2 ラウンド）

作成: 2026-10-10（Codex。学習・GPU 実行なし）

[厳密] 格は `[厳密]`（恒等式ではないが条件なしで成り立つ結論）、`[恒等式]`、`[仮定つき]`、`[候補]`、`[書けない]` のいずれかで付ける。以下の δ は loss の標本平均の係数をすでに含む。

[厳密] 保存済み MLP 診断の実装は表示値として \(N\delta\) を使う。このノートの規約へ移すと \(H,\bar D,G,C_D,Cpl,\alpha,\beta\) は一様に \(1/N\) 倍されるが、以下の恒等式、符号、share は変わらない。

## 1. DC バケットの厳密な透過恒等式

### 1.1 固定状態での master identity

[恒等式] c1 pool 後の格子を \(\Lambda\)、5×5 offset 集合を \({\cal O}\)（\(|{\cal O}|=25\)）とし、forward convolution を

\[
z_{2k}(n,t)=\sum_{j,o}W_{2,kjo}P_{1j}(n,t+o)
             {\bf1}\{t+o\in\Lambda\}+b_{2k}
\]

と書く。pool winner、gate、現在の \(\delta_2\) を固定して

\[
A_j(n,q)=\gamma_j(n,p^*(n,j,q))K_{1j}(n,p^*(n,j,q)),
\]

\[
F_{kjo}=\sum_{n,t:\,t+o\in\Lambda}
 A_j(n,t+o)\delta_{2k}(n,t)
\]

とおけば、接続テンソルの任意の成分 \(U_{kjo}\) が運ぶ c1 の押しは

\[
G_j[U]=\sum_{k,o}U_{kjo}F_{kjo}. \tag{1}
\]

[厳密] 式 (1) は full state の cotangent と分枝を固定した線形帳簿である。\(U\) ごとに forward/backward を再計算する counterfactual loss の恒等式ではない。max-pool tie では autograd が選んだ分枝を固定する必要がある。

### 1.2 DC/AC 分解

[恒等式] 各 \((k,j)\) について

\[
\Delta s_{kj}=\sum_o\Delta W_{2,kjo},\qquad
R_{kjo}=\Delta W_{2,kjo}-{\Delta s_{kj}\over25},\qquad
\sum_oR_{kjo}=0
\]

とおくと、

\[
G_j[\Delta W_2]=G_j^{\rm DC}+G_j^{\rm AC},
\]

\[
G_j^{\rm DC}=\sum_k\Delta s_{kj}\widetilde A_{kj},
\qquad
\widetilde A_{kj}={1\over25}\sum_oF_{kjo}, \tag{2}
\]

\[
G_j^{\rm AC}=\sum_{k,o}R_{kjo}F_{kjo}. \tag{3}
\]

[恒等式] 転置畳み込みを zero-pad の valid relation ごと作用素に含めれば、(2) は brief2 の式そのものである。

\[
\widetilde A_{kj}
={1\over25}\sum_{n,q}A_j(n,q)
       \bigl({\bf1}_{5\times5}\star^\top\delta_{2k}\bigr)(n,q). \tag{4}
\]

[厳密] 従って「\(G_j^{\rm DC}=\sum_k\Delta s_{kj}\widetilde A_{kj}\) は厳密か」への答えは **はい** である。ただし、現在の \(\delta_2\)、gate、pool route を固定した \(\Delta W_2\) 経路の寄与、という意味に限る。

[恒等式] 同じ固定状態の帳簿では

\[
G_j^{\rm DC}+G_j^{\rm AC}
=G_j^{\rm drift}+G_j^{\rm rest}
=G_j[\Delta W_2]. \tag{4a}
\]

[厳密] これは一般には full \(G_j\) ではない。full には \(G_j[W_2^0]\) が別に加わる。

### 1.3 \(\widetilde A_{kj}\) と \(\bar D_k\) の関係

[恒等式] \(M=N|\Lambda|\)、\(\bar D_k=\sum_{n,t}\delta_{2k}(n,t)\) とし、

\[
Q_j(n,t)={1\over25}\sum_{o:\,t+o\in\Lambda}A_j(n,t+o)
\]

と定義する。すると

\[
\widetilde A_{kj}=\sum_{n,t}\delta_{2k}(n,t)Q_j(n,t)
=\bar Q_j\bar D_k+M\operatorname{Cov}_{n,t}(Q_j,\delta_{2k}), \tag{5}
\]

ここで \(\bar Q_j=M^{-1}\sum_{n,t}Q_j(n,t)\) である。

[恒等式] \(r(q)\) を位置 \(q\) を覆う valid offset の個数とすれば、

\[
\bar Q_j={1\over25M}\sum_{n,q}r(q)A_j(n,q). \tag{6}
\]

従って gate と \(K_1\) は \(A_j\) を通って係数と共分散の両方に入り、zero-pad 境界は \(r(q)\) と \(Q_j\) の空間変動を通って入る。

[厳密] 16×16、pad 2 では位置平均の coverage は \(\bar r=21.390625\) である。\(A_j\equiv a_j\) でも \(\bar Q_j=(21.390625/25)a_j=0.855625a_j\) であり、\(Q_j\) は境界で \(9a_j/25\)、内部で \(a_j\) となる。

[仮定つき] periodic padding、または boundary と gate の空間変動が \(\delta_2\) と無相関で、\(A_j\) がほぼ定数なら、\(\widetilde A_{kj}\approx \bar A_j\bar D_k\)（zero-pad なら coverage 係数つき）となる。

[書けない] \(A_j>0\) と \(\bar D_k>0\) だけから \(\widetilde A_{kj}>0\) とは書けない。\(\delta_2\) が混符号なら、式 (5) の共分散が平均項を上回れる。

### 1.4 AC が平均の押しへ入る経路

[恒等式] \(\bar F_{kj}=25^{-1}\sum_oF_{kjo}\) とすれば、\(\sum_oR_{kjo}=0\) より

\[
G_j^{\rm AC}
=\sum_{k,o}R_{kjo}(F_{kjo}-\bar F_{kj})
=25\sum_k\operatorname{Cov}_{o}(R_{kj\cdot},F_{kj\cdot}). \tag{7}
\]

これは AC kernel が、shift ごとに異なる \(\gamma K_1\)-\(\delta_2\) の空間的重なりを読む厳密な経路である。

[恒等式] 同じことを c1 格子上で書く。\(U_j=\sum_kR_{kj}\star^\top\delta_{2k}\)、\(\bar A_j=M^{-1}\sum A_j\)、\(\bar U_j=M^{-1}\sum U_j\) とすれば

\[
G_j^{\rm AC}=M\bar A_j\bar U_j
             +M\operatorname{Cov}_{n,q}(A_j,U_j). \tag{8}
\]

[厳密] \(V_o=\{t\in\Lambda:t+o\in\Lambda\}\) とする。periodic padding なら \(\sum U_j=0\) なので (8) は純粋な空間共分散になる。zero-pad では

\[
M\bar U_j=-\sum_{k,o}R_{kjo}
 \sum_{n,t\notin V_o}\delta_{2k}(n,t), \tag{9}
\]

という boundary 平均も残る。従って「AC の寄与 = \(\gamma K_1\) との共分散」だけでは zero-pad 下では一項不足する。

### 1.5 現在の \(\widehat\mu_2\) 射影と DC バケットは別の基底

[恒等式] \(\mu_2\ne0\) とし、行 \(k\) の current-mean 射影を

\[
\lambda_k=\langle\Delta W_{2k},\widehat\mu_2\rangle=-c_k,
\qquad
\Delta W_{2k}^{\rm drift}=\lambda_k\widehat\mu_2
\]

と書けば、channel \(j\) への寄与は

\[
G_j^{\rm drift}=\sum_{k,o}\lambda_k\widehat\mu_{2,j,o}F_{kjo}. \tag{10}
\]

[恒等式] DC との差は

\[
G_j^{\rm DC}-G_j^{\rm drift}
=\sum_{k,o}\left({\Delta s_{kj}\over25}
       -\lambda_k\widehat\mu_{2,j,o}\right)F_{kjo}. \tag{11}
\]

[恒等式] \(r_j=\sum_o\widehat\mu_{2,j,o}\)、\(\widehat\mu^{\rm AC}_{2,j,o}=\widehat\mu_{2,j,o}-r_j/25\) とすれば、drift 自身も

\[
\Delta W^{\rm drift}_{2,kj,o}
={\lambda_kr_j\over25}+\lambda_k\widehat\mu^{\rm AC}_{2,j,o},
\qquad
s^{\rm drift}_{kj}=\lambda_kr_j,
\qquad
s^{\rm rest}_{kj}=\Delta s_{kj}-\lambda_kr_j. \tag{11a}
\]

[厳密] 従って current `rest` には一般に履歴的な DC 成分も入り、current `drift` にも \(\widehat\mu_2\) block の AC 成分が入る。`drift = DC`、`rest = AC` ではない。

[恒等式] drift の DC 係数行列は \(S^{\rm drift}=\lambda r^\top\) なので rank 1 以下だが、\(\Delta S=(\Delta s_{kj})\) は一般に rank 16 まで持てる。

[恒等式] unfolding mean の定義から、後で (13) に定義する箱和 \(B_j\) を使えば

\[
r_j=\sum_o\widehat\mu_{2,j,o}
={1\over\|\mu_2\|M}\sum_{n,t}B_j(n,t)
={\bar B_j\over\|\mu_2\|}. \tag{11b}
\]

[厳密] 従って drift の DC 輪郭が直接読むのは zero-pad coverage つき箱平均であり、通常の \(25\bar P_{1j}\) とは一般に一致せず、符号も保証されない。

[厳密] 一つの出力 channel \(k\) ごとに、current-\(\mu_2\) 射影は \(\operatorname{span}\{\mu_2(t)\}\) という 1 次元基底である。DC バケットは input channel ごとに独立な定数 kernel を持つので 16 次元である。さらに zero-pad 下の \(\mu_{2,j,o}\) は offset ごとの coverage を持ち、一般には定数 kernel ですらない。両分解は包含関係にない。

[恒等式] 実際の更新を \(U_{2k}(r)=W_{2k}(r+1)-W_{2k}(r)\) とすれば、現在の射影係数と DC 履歴は

\[
\lambda_k(t)=\sum_{r<t}\langle U_{2k}(r),\widehat\mu_2(t)\rangle,
\qquad
\Delta s_{kj}(t)=\sum_{r<t}\sum_oU_{2,kjo}(r). \tag{12}
\]

[候補] 過去の更新が各時点の \(-\widehat\mu_2(r)\) に成分を持つとき、\(\bar P_{1j}\) の符号変更は \(\widehat\mu_2(r)\) と \(\widehat\mu_2(t)\) の重なりを弱め、式 (12) 左の current projection では過去成分を相殺し得る。一方、右の固定 DC 座標は各 \((k,j)\) の和をそのまま蓄積する。これが、現在の \(\widehat\mu_2\) が履歴の基底として狭すぎる、という読みの正確な内容である。

[書けない] \(\bar P_{1j}\) の符号変更だけから、実際の \(\lambda_k(t)\) の cancellation の大きさや、DC が AC より大きいことまでは書けない。

## 2. \(\Delta s_{kj}\) の履歴

### 2.1 勾配と residual の分解

[恒等式] c2 site \(t\) の周りの箱和を

\[
B_j(n,t)=\sum_{o:\,t+o\in\Lambda}P_{1j}(n,t+o)
\]

とすれば、offset 座標の勾配和は

\[
g^\Sigma_{kj}
:=\sum_o{\partial L\over\partial W_{2,kjo}}
=\sum_{n,t}\delta_{2k}(n,t)B_j(n,t). \tag{13}
\]

[恒等式] \(W_{kjo}=s_{kj}/25+R_{kjo}\) と再パラメータ化した scalar \(s_{kj}\) の偏微分は \(g^\Sigma_{kj}/25\) である。一方、元の 25 座標を同じ learning rate の SGD で更新したときの **kernel 和の変化** は

\[
s_{kj}(r+1)-s_{kj}(r)=-\eta_r g^\Sigma_{kj}(r). \tag{14}
\]

[恒等式] full-data mean loss の出力 residual を \(e_n^y=(p_n-y_n)/N\)、\(e_n^u=(p_n-\mathbf1/K)/N\) とし、現在の network の backprop 作用素を \({\cal J}_{2}^{\top}\) とすれば

\[
\delta_2^y={\cal J}_{2}^{\top}((p-y)/N),
\qquad
\delta_2^u={\cal J}_{2}^{\top}((p-\mathbf1/K)/N),
\]

\[
\delta_2^y=\delta_2^u
+{\cal J}_{2}^{\top}((\mathbf1/K-y)/N),
\qquad
g^{\Sigma,y}_{kj}=g^{\Sigma,u}_{kj}+g^{\Sigma,{\rm label}}_{kj}. \tag{15}
\]

[厳密] minibatch step では上の \(N\) をその step の batch size に置き換える。

[仮定つき] 新しい iid uniform label をまだ network が見ておらず、その label が現在状態と独立なら、固定状態で \(\mathbb E_y g^{\Sigma,{\rm label}}_{kj}=0\) である。従って新課題の最初の期待勾配は uniform-target の式になる。

[厳密] 課題内では状態がその実ラベルに依存するので、式 (15) の label 項の時間積分の期待が 0 とは限らない。

### 2.2 切替の負の DC 更新と、課題内の戻り

[恒等式] \(\bar B_j=M^{-1}\sum B_j\) とすれば

\[
g^\Sigma_{kj}=\bar B_j\bar D_k+M\operatorname{Cov}_{n,t}(B_j,\delta_{2k}). \tag{16}
\]

[恒等式] 通常の unweighted pooled mean \(\bar P_{1j}\) を基準にすると、同じ式は

\[
g^\Sigma_{kj}
=25\bar P_{1j}\bar D_k
+(\bar B_j-25\bar P_{1j})\bar D_k
+M\operatorname{Cov}_{n,t}(B_j,\delta_{2k}). \tag{16a}
\]

[厳密] (16a) の第二項が zero-pad coverage と空間分布による箱平均のずれ、第三項が \(\delta_2\) による正の裾などの選択を含む。

[仮定つき] 切替時に \(\bar B_j>0\)、\(\bar D_k>0\) で、

\[
M\operatorname{Cov}(B_j,\delta_{2k})
>-\bar B_j\bar D_k
\]

なら \(g^\Sigma_{kj}>0\) なので、SGD の \(\Delta s_{kj}\) は負である。より強い十分条件は \(B_j\ge0\)、\(\delta_{2k}\ge0\) が点ごとに成り立ち、積がどこかで正になることである。

[厳密] 「\(\bar D_k>0\) かつ箱和が各点で正」だけではまだ足りない。例えば \(\delta=(2,-1)\)、\(B=(1,3)\) なら \(\sum\delta=1>0\)、\(B>0\) だが \(\sum\delta B=-1<0\) である。

[恒等式] weight decay のない vanilla SGD なら、区間 \({\cal I}\) の累積は

\[
\Delta s_{kj}({\cal I})
=-\sum_{r\in{\cal I}}\eta_r
 \left[g^{\Sigma,u}_{kj}(r)+g^{\Sigma,{\rm label}}_{kj}(r)\right]. \tag{17}
\]

[書けない] \(p-y\) だけから課題内の label 項が必ず \(s\) を「戻す」符号になるとは書けない。fit により \(p-y\to0\) なら後半の更新は止まるが、それは前半の変位を逆向きに消すことを意味しない。

[候補] 実測された負の \(\Delta s\) は、切替ごとの uniform-target 成分が課題内の反対符号成分より累積で勝った、または Adam の履歴がその向きを保った結果、と読むことはできる。ただし endpoint だけでは両者を区別できない。

### 2.3 \(\bar P_{1j}<0\) の後にも \(\Delta s_{kj}<0\) が残る三つの理由

[恒等式] 箱和の空間平均は

\[
\sum_{n,t}B_j(n,t)=\sum_{n,q}r(q)P_{1j}(n,q). \tag{18}
\]

従って unweighted な \(\bar P_{1j}\le0\) は、coverage-weighted な \(\bar B_j\le0\) も、各箱和が負であることも含意しない。

[候補] max-pool 後の \(P_1\) は平均が少し負でも正の裾を持ち得る。\(\delta_2\) がその裾の箱を強く読むと式 (16) の covariance 項が正になり、現在も SGD comparator は \(\Delta s<0\) を作れる。

[恒等式] Adam では、bias correction 込みの moment を \(\widehat m_{kjo,r},\widehat v_{kjo,r}\) とすれば、weight decay がない場合の実変位は

\[
\Delta s_{kj}
=-\sum_r\eta_r\sum_o
{\widehat m_{kjo,r}\over\sqrt{\widehat v_{kjo,r}}+\epsilon}. \tag{19}
\]

[厳密] 式 (19) は一般に \(-\sum_r\eta_rg^\Sigma_{kj,r}\) ではない。offset 間の符号が混じると、momentum と座標別の正規化が和の符号を変え得る。ただし全 offset の moment が同符号なら正の分母だけで反転はしない。

[候補] 61〜73% が残る第一候補は履歴蓄積、第二候補は式 (18) と covariance による「現在の平均と実際に読まれる箱」のずれ、第三候補は式 (19) の optimizer 効果である。既知の endpoint は三候補すべてと両立する。

### 2.4 三候補を一度に分ける一つの測定

[恒等式] channel \(j\) が最後に \(\bar P_{1j}>0\) から \(\le0\) へ入った step を \(\tau_j\) とする。各 optimizer step \(r\) の実 minibatch \({\cal B}_r\) 上で \(M_r=|{\cal B}_r||\Lambda|\) とし、\(\bar B_{j,r},\bar D_{k,r},\operatorname{Cov}_r\) も同じ minibatch と同じ loss 正規化で計算して、次の **符号反転後 DC ledger** を保存する。

\[
\Delta s_{kj}(t)
=\underbrace{\Delta s_{kj}(\tau_j)}_{H_{kj}:\,\text{pre-crossing history}}
+\underbrace{-\sum_{r=\tau_j}^{t-1}\eta_r\bar B_{j,r}\bar D_{k,r}}_{M_{kj}:\,\text{box mean}}
+\underbrace{-\sum_{r=\tau_j}^{t-1}\eta_rM_r
 \operatorname{Cov}_{r}(B_j,\delta_{2k})}_{C_{kj}:\,\text{spatial/tail}}
+\underbrace{R^{\rm opt}_{kj}}_{\text{actual update}-(M+C)}. \tag{20}
\]

[恒等式] \(R^{\rm opt}\) を実 optimizer 変位と \(M+C\) の差として定義すれば (20) は帳簿上の恒等式であり、同じ minibatch の weight-decay なし vanilla SGD なら \(R^{\rm opt}=0\) である。

[候補] \(H\) 優勢なら履歴、\(M\) または \(C\) が反転後も負なら箱和・正の裾、\(R^{\rm opt}\) が負を担えば Adam/momentum を支持する。複数項が同時に大きい場合は単一原因には同定しない。

## 3. MLP の「切片」と透過

### 3.1 回帰切片は \(H\) の平均そのものではない

[恒等式] MLP で

\[
a_{in}=K_n\phi_1'(z_{1in}),\quad
\kappa_i=\sum_na_{in},\quad
H_{in}=\sum_kW_{2,ki}\delta_{2,kn},\quad
\mu_i={1\over N}\sum_nh_{1in}
\]

とする。\(\operatorname{Var}_n(h_{1in})>0\) の unit について、切片つき OLS を

\[
H_{in}=\alpha_i+\beta_i h_{1in}+r_{in}
\]

と書けば

\[
\bar H_i={1\over N}\sum_nH_{in}
={1\over N}\sum_kW_{2,ki}\bar D_k,
\qquad
\alpha_i=\bar H_i-\beta_i\mu_i. \tag{21}
\]

[厳密] \(\alpha_i\) は \(H\) の画像平均ではなく、回帰直線を \(h_{1i}=0\) へ外挿した値である。平均水準は \(\bar H_i=\alpha_i+\beta_i\mu_i\) である。

[恒等式] 回帰表の分解

\[
G_i=\beta_i\sum_na_{in}h_{1in}
    +\alpha_i\kappa_i+\sum_na_{in}r_{in} \tag{22}
\]

自体は正しい。しかし同じ式は

\[
G_i=\underbrace{\kappa_i\bar H_i}_{T_i^{\rm const}}
+\beta_i\sum_na_{in}(h_{1in}-\mu_i)
+\sum_na_{in}r_{in} \tag{23}
\]

とも書ける。

[厳密] 原点の取り方に依らない透過項は

\[
T_i^{\rm const}={\kappa_i\over N}\sum_kW_{2,ki}\bar D_k, \tag{24}
\]

である。\(\alpha_i\kappa_i=T_i^{\rm const}-\beta_i\mu_i\kappa_i\) は、定数モードと回帰基底の原点補正を混ぜるので、それ単独を物理的な透過項とは呼べない。

[恒等式] 回帰を使わない不変な二分解は

\[
G_i=\kappa_i\bar H_i
+\sum_n\left(a_{in}-{\kappa_i\over N}\right)
       (H_{in}-\bar H_i). \tag{25}
\]

[候補] (25) の第二項には学習で育った自己整合成分が含まれ得る。ただし観測された centered \(R^2=0.01\)〜0.07 から、第二項全体を \(H\propto h_1\) の自己整合項と同一視するのは強すぎる。

### 3.2 \(\alpha_i\)、\(\mu_i\)、育った \(W_2\) の関係

[恒等式] \(T_i^0=\sum_kW^0_{2,ki}\bar D_k\)、\(T_i^\Delta=\sum_k\Delta W_{2,ki}\bar D_k\) とおけば

\[
N\alpha_i=T_i^0+T_i^\Delta-N\beta_i\mu_i. \tag{26}
\]

[恒等式] \(\mu\ne0\) とし、\(c_k=-\langle\Delta W_{2k},\widehat\mu\rangle\)、\(C_D=\sum_kc_k\bar D_k\) とおいて、\(\Delta W_2=\Delta W_2^{\rm drift}+\Delta W_2^{\rm rest}\) と分ければ

\[
T_i^\Delta=-\widehat\mu_i C_D+T_i^{\rm rest},
\qquad
T_i^{\rm rest}=\sum_k\Delta W^{\rm rest}_{2,ki}\bar D_k, \tag{27}
\]

\[
N\alpha_i=T_i^0+T_i^{\rm rest}
-\mu_i\left({C_D\over\|\mu\|}+N\beta_i\right). \tag{28}
\]

[恒等式] unit ごとの積と、全 unit を足した関係は

\[
\mu_iT_i^\Delta
=-{\mu_i^2\over\|\mu\|}C_D+\mu_iT_i^{\rm rest}, \tag{29}
\]

\[
\sum_i\mu_iT_i^\Delta
=\sum_k\bar D_k\langle\Delta W_{2k},\mu\rangle
=-\|\mu\|C_D. \tag{30}
\]

[厳密] (30) は global な反平均方向の結合を厳密に表すが、各 \(i\) の \(T_i^{\rm rest}\)、\(T_i^0\)、\(-N\beta_i\mu_i\) は残る。従って \(\operatorname{sign}\alpha_i\) は \(\mu_i\) または \(T_i^\Delta\) だけでは決まらない。

[恒等式] full drift と、そのうち画像定数モードだけはそれぞれ

\[
G_i^{\rm drift}=-\widehat\mu_i Cpl_i,
\quad
Cpl_i=\sum_kc_k\sum_na_{in}\delta_{2,kn}, \tag{31}
\]

\[
T_i^{{\rm drift,const}}
=-{\kappa_i\widehat\mu_i\over N}C_D. \tag{32}
\]

[恒等式] 両者の差は

\[
Cpl_i-{\kappa_i\over N}C_D
=\sum_kc_k\sum_n\left(a_{in}-{\kappa_i\over N}\right)
 \delta_{2,kn}, \tag{33}
\]

すなわち gate つき接線と \(\delta_2\) の画像共分散である。

### 3.3 raw 90% 対 std 18〜28% を \(\mu\) の符号で説明できるか

[仮定つき] \(C_D>0\)、\(\kappa_i>0\) で、(32) が他の定数モード成分を上回れば、その透過は \(-\operatorname{sign}\mu_i\) である。従って raw の \(\mu_i<0\) では沈める側、std の \(\mu_i>0\) では浮かせる側という定性的差を説明できる。

[厳密] しかし観測表は \(\alpha_i\kappa_i\) の符号であり、(28) の \(-\beta_i\mu_i\) だけでも、育った DC 透過が 0 の場合に同じ raw/std の切片差を作れる。また std では \(\kappa_i\) の符号も仮定なしには固定できない。

[書けない] よって 90% 対 18〜28% の **正確な率**、またはその差が learned throughput による割合は、\(\mu_i\) の符号だけからは書けない。\(\kappa_i\bar H_i\) を \(W_2^0\)、drift、rest に直接分けて測る必要がある。

### 3.4 2 層との関係

[恒等式] 直接 readout の 2 層でも \(H_{in}=V_{:i}^{\top}e_n\)、\(D=\sum_ne_n\) なので

\[
\bar H_i={1\over N}V_{:i}^{\top}D,
\qquad
T_i^{\rm const}={\kappa_i\over N}V_{:i}^{\top}D. \tag{34}
\]

[厳密] 従って「2 層に無い透過項の MLP 版」は言い過ぎである。同じ定数モードは 2 層にもあり、3 層で違うのは hidden cotangent の和 \(\bar D_k\) が softmax の class-simplex 制約だけでは消えず、育った中間接続を通れる点である。

[仮定つき] fit 済み、iid uniform label、output-bias がほぼ停留、normalized loss、readout norm と平均接線 \(\kappa_i/N\) が \(O(1)\) なら、2 層の \(D=\bar p-\mathbf1/K=O_p(N^{-1/2})\) であり、(34) も \(O_p(N^{-1/2})\) と期待される。

[書けない] 2 層の透過が「現れない」とは書けない。厳密に消えるのは \(D=0\)、\(V_{:i}\perp D\)、または \(\kappa_i=0\) のときである。

## 4. V12 に書く文

### 4.1 元の文の節ごとの格

| 節 | 格 | 判定 |
|---|---|---|
| 「押しは自己整合項＋透過項の和」 | `[候補]` | 定数モード＋中心化 remainder なら恒等式だが、中心化 remainder 全体を自己整合と呼ぶことはできない。CNN ではさらに AC と boundary を保持する必要がある。 |
| 「下流の押しが接続重みを通って上流へ透過」 | `[恒等式]` | MLP は (24)、CNN の DC bucket は (2)〜(5)。ただし CNN の offset-DC と MLP の current-\(\mu\) 射影は同じ基底ではない。 |
| 「透過項の符号は上流平均出力の逆」 | `[仮定つき]` | current drift に限定し、\(Cpl_i>0\)（定数モードなら \(\kappa_iC_D>0\)）と remainder 非支配が必要。履歴を持つ DC bucket には現在の平均出力との逆符号則はない。 |
| 「下流が沈める側ならよい」 | `[書けない]` | その条件だけでは符号を書けない。`[厳密]` full layer の \(G>0\) や \(\bar D_k>0\) の多数率では足りず、\(C_D=\sum_kc_k\bar D_k>0\) または gate-weighted \(Cpl_i>0\) が必要。 |
| 「2 層では \(\delta\) の画像和が \(O(N^{-1/2})\)」 | `[仮定つき]` | fit、iid balanced、bias 停留、有界 norm などの条件つきなら \(O_p(N^{-1/2})\)。softmax だけが保証するのは class 和 0。 |
| 「なので 2 層では現れない」 | `[書けない]` | 構造的な 0 ではなく、小さいという条件つき評価まで。 |

### 4.2 言い過ぎを削った候補

> [恒等式] MLP の押しには画像定数／中心化分解 (25) が、CNN の \(\Delta W_2\) 経路には kernel-DC／AC 分解 (2)〜(3) がそれぞれ成り立ち、zero-pad boundary は各係数の中に保持される；[候補] 中心化／AC 側には 2 層で見た自己整合成分が含まれるが、それと同一ではない；[仮定つき] 学習された接続の **現在の平均入力方向への射影**が反平行で、対応する gate-weighted 下流結合が正かつ remainder が小さいとき、その drift 成分は上流平均出力と逆符号になるが、蓄積 DC には履歴がある；[仮定つき] 直接 readout の 2 層にも同じ定数モードは存在するものの、fit 済み iid 有限標本、output-bias 停留、有界 readout と \(\kappa_i/N=O(1)\) の下では \(O_p(N^{-1/2})\) と期待される。

## 5. DC バケット測定の盲検予測

[候補] 根拠は、既知の proxy \(T_j^\Delta=\sum_k\Delta s_{kj}\bar D_k\) と full \(G_j\) の符号一致が t5〜t20 で 0.86〜0.96 と高い一方、(5) の gate/boundary covariance と (7) の AC が未測定であること。この情報だけから dominance までは予測しない。

[候補] 第1・第2列は各時点の seed×\(j\)（160 channel）を micro-pool、第3列は seed×\(k\)×\(j\)（2560 対）を micro-pool する予測である。Spearman は各 seed の 16 個の \(j\) 上で計算し、10 seed の中央値を取るものとして固定する。16 点の順位なので帯を広く置く。

[厳密] 符号一致は積が厳密に正のものとし、0 は一致に数えない。実測で numerical tolerance を使うなら結果を見る前に固定すべきであり、特に小さい \(\bar D_k\) では率が動き得る。

| t | \(\operatorname{sign}G^{DC}=\operatorname{sign}G\) | \(|G^{DC}|>|G^{AC}|\) | \(\operatorname{sign}\widetilde A_{kj}=\operatorname{sign}\bar D_k\) | \(\rho_S(T^\Delta,G^{DC})\) |
|---|---:|---:|---:|---:|
| 1 | 0.68〜0.81 | 0.50〜0.70 | 0.68〜0.84 | 0.45〜0.72 |
| 2 | 0.75〜0.88 | 0.52〜0.73 | 0.65〜0.82 | 0.42〜0.70 |
| 5 | 0.83〜0.93 | 0.58〜0.80 | 0.70〜0.86 | 0.48〜0.76 |
| 10 | 0.92〜0.98 | 0.65〜0.86 | 0.78〜0.92 | 0.55〜0.82 |
| 20 | 0.90〜0.97 | 0.62〜0.84 | 0.80〜0.94 | 0.45〜0.76 |
| 30 | 0.80〜0.91 | 0.58〜0.80 | 0.80〜0.94 | 0.45〜0.78 |

[候補] 元の Q1 で用いた採点域 t=5/10/20 の平均なら \(G^{DC}\) と full \(G\) の符号一致は 0.88〜0.95、従って **0.85 以上は通る寄り**と予測する。全時点で各々 0.85 以上、という強い版は t1/t2 で外れる寄りである。DC bucket 自体は事後追加であり、この帯を事前登録済みとは扱わない。

[候補] \(|G^{DC}|>|G^{AC}|\) は採点域平均 0.62〜0.82 と予測し、0.85 級の dominance は予測しない。proxy の符号一致は、AC が同符号で DC より大きい場合にも高くなれる。

[候補] \(\widetilde A_{kj}\) と \(\bar D_k\) の符号一致の中心予測は 0.75〜0.88、表の広い envelope は 0.65〜0.94 とする。t10 以降の中心は 0.8 以上を本命とする。raw では \(A_j>0\) が追い風だが、\(\bar D_k\) が cancellation で小さい対と gate/boundary covariance が外れを作る。

[候補] \(T^\Delta\) と \(G^{DC}\) の Spearman は符号一致ほど高くなく、中心予測 0.55〜0.75、表の広い envelope 0.42〜0.82 とする。\(G_j^{DC}\approx\bar Q_jT_j^\Delta\) でも、現在 gate による正の scale \(\bar Q_j\) が channel 間で違えば順位は入れ替わる。履歴と現在 gate のずれが大きい t20 は低下候補である。

[候補] spec §6.2 (iii) で外れやすい登録外の読みは、「proxy の高い符号一致」から「DC が **大きさまで** c1 の浮く向きを運ぶ」へ進む部分である。弱い読みである「t10〜t30 の符号を頑健に指す」は残ると予測するが、AC が同方向でより大きいなら、結論は「DC 減少が向きの指標で、AC も同方向に寄与する」まで弱めるべきである。

[厳密] 現在 \(\bar P_{1j}\le0\) でも \(\Delta s_{kj}<0\) が残ることだけから、当該対の \(G^{DC}<0\)、DC dominance、または履歴・正の裾・Adam のどれが原因かは導けない。

[書けない] 学習を実行せず、\(Q_j\)-\(\delta_2\) covariance、offset covariance、boundary residual を見ない限り、上の帯より狭い割合や Spearman は書けない。

## 6. 最短の結論

[恒等式] DC bucket の式 (2)〜(5) は zero-pad を作用素に残せば厳密であり、proxy \(\sum_k\Delta s_{kj}\bar D_k\) との差は gate と boundary の weighted covariance である。

[厳密] current-\(\mu_2\) drift は 1 次元、per-channel DC は 16 次元であり、前者は符号を変える現在の平均を基底に使うため、蓄積履歴を表すには狭い。

[仮定つき] 負の \(\Delta s\) は正の downstream sum と正の箱平均から生じるが、混符号 \(\delta_2\) では covariance 余裕が必要であり、実 Adam の履歴は勾配和の単純積分ではない。

[厳密] MLP の不変な透過量は \(\alpha_i\kappa_i\) ではなく \(\kappa_i\bar H_i\) であり、2 層にも同じ項は存在する。

[書けない] 現データだけから DC の magnitude dominance、\(\alpha\) の raw/std 差に占める learned-throughput の割合、\(\Delta s<0\) の三候補の内訳は書けない。
