# Batch 16 の固定 iid 勾配で、Adam の定常・長期平均も逆転する

2026-10-09。これは**固定した予測値から生成する iid 勾配列を Adam に入力する模型**の定理であり、パラメータ更新に応じて予測値が変わる CNN/CE の学習軌道の定理ではない。

## 1. 結論

$$g_t=0.900001-K_t/16,\qquad K_t\overset{\rm iid}{\sim}{\rm Binomial}(16,0.9)$$

に、$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$ の Adam を適用する。SGD の平均勾配は厳密に

$$\mathbb E g_t=10^{-6}>0.$$

一方、定常状態の Adam の正規化方向 $A=m/(\sqrt v+\epsilon)$ について、**有理数演算と平方根の有理区間だけで**

$$\boxed{-0.000358608266201<\mathbb E A<-0.000271557965466<0}$$

を証明できる。初期値ゼロ・通常の bias correction から始めた Adam も、同じ iid driver の下では

$$\frac1T\sum_{t=1}^T\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
\longrightarrow\mathbb E A<0\quad\text{a.s.}$$

したがって、この模型では逆転は first step の過渡現象ではなく、**定常期待値と実現した無限時間平均の両方に残る**。SGD の時間平均は $+10^{-6}$ に収束するので、符号が反対である。

これは actual CNN の長期反転を証明しないが、「first-step の逆転は長期平均を取れば必ず消える」という一般命題への反例になる。

## 2. 定常状態を無限履歴で定義する

二側無限 iid 列 $\{g_t\}_{t\in\mathbb Z}$ を使って

$$m_t=(1-\beta_1)\sum_{j=0}^{\infty}\beta_1^j g_{t-j},\qquad
v_t=(1-\beta_2)\sum_{j=0}^{\infty}\beta_2^j g_{t-j}^2$$

と定める。勾配は17値で有界なので両級数は絶対収束する。以下を置く。

$$\mu=\mathbb E g=10^{-6},\quad r=\mathbb E g^2=0.005625000001,\quad
\Delta=v-r.$$

この分布では

$$M=\max|g|=0.900001,\qquad z_* =\min g^2=(0.025001)^2=0.000625050001.$$

定常状態で常に $|m|\le M$、$z_*\le v\le M^2$。特に分母は確率的にも確定的にもゼロへ近づかない。

## 3. 定常 moment を厳密な有理数で求める

$g_k=900001/10^6-k/16$、

$$p_k={16\choose k}(9/10)^k(1/10)^{16-k},\qquad k=0,\ldots,16$$

なので、以下は全て有理数で計算できる。$Z=g^2$ の中心 moment を

$$c_n=\mathbb E[(Z-r)^n],\quad c_0=1$$

とし、中心化した $Z-r$ の cumulant を $\kappa_n$ とする。$\kappa_1=0$、一般に

$$\kappa_n=c_n-\sum_{j=1}^{n-1}{n-1\choose j-1}\kappa_j c_{n-j}.$$

独立な履歴の重みつき和なので、$\Delta$ の cumulant は

$$\kappa_n(\Delta)=\kappa_n\frac{(1-\beta_2)^n}{1-\beta_2^n}.$$

その中心 moment $d_n=\mathbb E\Delta^n$ は $d_0=1$ として

$$d_n=\sum_{j=1}^n{n-1\choose j-1}\kappa_j(\Delta)d_{n-j}.$$

有限履歴での独立和の公式から得て、無限履歴へは有界収束で移せる。必要な値は

$$d_2\simeq3.671325507628815\times10^{-8},$$
$$d_6\simeq7.783593266219985\times10^{-22},\qquad
d_{12}\simeq3.727305052310703\times10^{-41}.$$

joint moments も独立性から厳密に

$$\mathbb E m^2=\mu^2+(r-\mu^2)\frac{1-\beta_1}{1+\beta_1},$$

$$\mathbb E[m\Delta]=\frac{(1-\beta_1)(1-\beta_2)}{1-\beta_1\beta_2}
\mathbb E[(g-\mu)(g^2-r)],$$

$$\mathbb E[m\Delta^2]=\mu d_2+
\frac{(1-\beta_1)(1-\beta_2)^2}{1-\beta_1\beta_2^2}
\mathbb E[(g-\mu)(g^2-r)^2]$$

となる。最後の式では、中心化した三因子の履歴インデックスが全部一致する項だけが残る。数値表示は

$$\mathbb E m^2\simeq0.0002960526325789474,$$
$$\mathbb E[m\Delta]\simeq2.787524777006938\times10^{-7},\qquad
\mathbb E[m\Delta^2]\simeq1.238323982634509\times10^{-11}.$$

## 4. 2次展開と、その残差を落とさない符号証明

$f(v)=v^{-1/2}$ を $r$ の周りで展開する。

$$T_2(v)=r^{-1/2}-\frac12r^{-3/2}\Delta+\frac38r^{-5/2}\Delta^2.$$

その期待値の中心項は

$$T:=\mathbb E[mT_2(v)]
=\frac\mu{\sqrt r}-\frac{\mathbb E[m\Delta]}{2r^{3/2}}
+\frac{3\mathbb E[m\Delta^2]}{8r^{5/2}}
\simeq-0.0003150831158335774.$$

この値の負だけを証明とすることはできない。以下で残差を定量化する。

### 4.1 $v\ge0.7r$ の場合

$f'''(v)=-15/(8v^{7/2})$。$r$ と $v$ の間も $0.7r$ 以上なので、Taylor の剰余から

$$|f(v)-T_2(v)|\le\frac5{16}(0.7r)^{-7/2}|\Delta|^3.$$

Cauchy–Schwarz により、この領域からの期待誤差は

$$B_{\rm good}\le\frac5{16}
\sqrt{\frac{\mathbb E[m^2]d_6}{(0.7r)^7}}
<0.000039160891331.$$

### 4.2 $v<0.7r$ の場合

この事象では $|\Delta|>0.3r$ なので、12次 moment による Markov の不等式は

$$P_{\rm bad}\le\frac{d_{12}}{(0.3r)^{12}}
<6.989877410\times10^{-8}.$$

また $v\ge z_*$、$-r\le\Delta<0$ より

$$f(v)\le z_*^{-1/2},\qquad |T_2(v)|\le\frac{15}{8\sqrt r}.$$

従って悪い事象からの誤差は

$$B_{\rm bad}\le M\left[z_*^{-1/2}+\frac{15}{8\sqrt r}\right]P_{\rm bad}
<0.000004088982178.$$

### 4.3 Adam の $\epsilon$ を落とさない

$$\left|\frac m{\sqrt v+\epsilon}-\frac m{\sqrt v}\right|
\le\epsilon\frac{|m|}{v}\le\frac\epsilon{z_*}|m|.$$

したがって

$$B_\epsilon\le\frac\epsilon{z_*}\sqrt{\mathbb E m^2}
<0.000000275276859.$$

三つを足して

$$|\mathbb E A-T|\le B_{\rm good}+B_{\rm bad}+B_\epsilon.$$

上界を使っても $\mathbb E A<-0.000271557965466<0$ であり、残差込みで符号が確定する。

## 5. 浮動小数点の符号判定に頼らない証明書

`/tmp/cnn_adam_stationary_b16_1009.py` の証明部分は Python 標準ライブラリの `Fraction` と整数 `isqrt` だけを使う。正の有理数 $x$ の平方根を、$s=10^{80}$ として

$$n=\left\lfloor\sqrt{\left\lfloor x s^2\right\rfloor}\right\rfloor,
\qquad n/s\le\sqrt x\le(n+1)/s$$

で囲み、両端を二乗して包含を再確認する。

中心項は $T=C/\sqrt r$、

$$C=\mu-\frac{\mathbb E[m\Delta]}{2r}+\frac{3\mathbb E[m\Delta^2]}{8r^2}<0$$

と書ける。従って $\sqrt r$ の**上端**で割ったものが $T$ の上界になる。good error は正の平方根の上端、bad error の $1/\sqrt r$ は平方根の下端、epsilon error は $\sqrt{\mathbb E m^2}$ の上端を使う。$\sqrt{z_*}=0.025001$ は厳密な有理数である。

最後の総上界がゼロ未満かは、浮動小数に変換する前の有理数で比較・assert している。JSON に全ての moment/cumulant、誤差上界、最終区間を有理数として保存した。小数値は表示専用である。

## 6. 定常期待値から、初期値ゼロの長期時間平均へ

定常 $A_t$ は iid 列の可測な shift 関数であり、stationary ergodic である。$|A_t|\le M/\sqrt{z_*}$ なので可積分で、ergodic theorem により

$$T^{-1}\sum_{t=1}^T A_t\to\mathbb E A\quad\text{a.s.}$$

次に通常の初期値 $m_0=v_0=0$ から始め、bias correction を行った $\widehat m_t,\widehat v_t$ を、同じ未来の勾配列を使う定常過程 $m_t^*,v_t^*$ に結び付ける。有限履歴の平均と過去 tail の平均を比較すると

$$|\widehat m_t-m_t^*|\le2M\beta_1^t,\qquad
|\widehat v_t-v_t^*|\le M^2\beta_2^t.$$

どちらの $v$ も $z_*$ 以上なので、$m/(\sqrt v+\epsilon)$ の偏微分上界から

$$\left|\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
-\frac{m_t^*}{\sqrt{v_t^*}+\epsilon}\right|
\le\frac{2M}{\sqrt{z_*}}\beta_1^t
+\frac{M^3}{2z_*^{3/2}}\beta_2^t.$$

右辺は和が有限な幾何級数である。従って時間平均への初期化・bias correction の影響はゼロになり、通常初期化の Adam も同じ負の極限を持つ。first step だけの議論ではない。

## 7. 補助 Monte Carlo の設計

証明は Monte Carlo を必要としないが、同じ script の `--mc` で追加検算できる。独立seedを8本、各seedで131,072回を burn-in として捨て、16,777,216回を測定する。測定は合計134,217,728回。

誤差は65,536連続値ごとの batch mean から評価し、batch 長を2倍・4倍にした再集計も出す。連続した Adam 出力を iid とみなした標準誤差は使わない。これらの区間は漸近的な batch-means 診断であり、厳密な確率区間ではない。数学的な符号保証は上の有理数証明書による。

MC では burn-in 後の bias correction を省略するが、その差は幾何減衰し、この十分長い burn-in では表示精度よりはるかに小さい。厳密証明は省略した実装に依存せず、通常の bias correction も含めている。

実行結果は標本平均 $-0.0001798995840$、batch-means 標準誤差 $0.0000868229431$。batch 長を2倍・4倍にすると標準誤差はそれぞれ $0.0000863527928$、$0.0000838550227$ だった。最初の集計による漸近的95%区間は約 $[-0.000350073,-0.000009727]$。標本平均自体は厳密な定常期待値区間より浅い負値で、Taylor 中心値との差は約1.6標準誤差、厳密区間までの最短距離は約1.1標準誤差である。有限サンプルの平均と真の定常期待値は区別する。厳密な符号の根拠は Monte Carlo の中心値ではなく §4–5 の証明書である。

## 8. 解釈の限界

- $p_{\rm pred}=0.900001$ は全時刻で固定される。実際に logit を更新すれば $p_{\rm pred}$ と勾配分布が変わるため、この stationary 証明をそのまま実学習の結論にはできない。
- この定理が否定するのは「Adam の first-step の符号反転なら、長期平均には関係しない」という一般的な除外の仕方である。固定 driver の時点で、定常の負の平均が厳密に存在する。
- Adam の分母と numerator の依存は定常状態でも残る。ここでは $\mathbb E[m\Delta]>0$ の補正が、小さい正の $\mathbb E g$ を上回ることを、残差込みで示した。
- 同じことが RL-CIFAR のどのチャネルで起きるか、あるいは駆動源理論の自己方向を実学習の長期に反転させるかは、別に検証が必要である。
