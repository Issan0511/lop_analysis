# CNN の装填と押し：独立の導出ノート

作成日：2026-10-20。
対象：`specs/spec_kappa_reload_cnn_1020.md` の登録時の定義。
このノートは固定状態での数学的導出である。
新しい測定や学習は実施していない。
参照した実装は `analysis/kappa_reload_1020/measure.py` と `analysis/reload_dial_1020/analyze.py` の `kappa_parts` である。
参照した既知の結果は `results/cnn_drive_verify_1009/summary.md` と `results/c1_direction_1010/summary.md` である。
Obsidian の検索ツールはこのセッションでは呼び出せなかったため、Vault の最新内容は根拠に含めない。

## 0. 記号と結論の範囲

$e=\mathbf 1/K$ と置く。
$P=I-\mathbf 1\mathbf 1^\top/K$ をクラス中心化の射影とする。
$\phi_n\in\mathbb R^H$ は最終隠れ層の出力である。
$\ell_n=V\phi_n+b$、$V^c=PV$、$b^c=Pb$ とする。
$y_n$ は旧課題の one-hot ラベルである。
$\tilde y_n$ は切替後のラベルである。
$L$ は旧ラベルの平均 CE である。
$L_u$ は新ラベルの条件付き期待 CE である。
両者を区別する。

$\kappa$ の正値は、旧ラベルと最終読み出しの**集計した内積**の正値を意味する。
各 unit の正値、各画像の正マージン、切替後の各 unit の沈降は別の命題である。

依頼文の「最終隠れ層では $R_n$ が入力によらない」は修正が必要である。
固定なのは head の列 $v_i$ である。
$g_{in}$ と $K_n$ は一般に画像に依存する。
$u_i=\nabla_\theta m_i$ を全パラメータで定義した場合には、さらに上流経路が加わる。

## 1. 定理：構造に依存しない $\kappa$ の下界

### 主張

$F=\sum_n(y_n-e)\phi_n^\top$、$A=\langle V^c,F\rangle_F$、$B=\sum_c(n_c-N/K)b^c_c$ とする。

恒等式は $A=\sum_n[(\ell_{n,y_n}-\bar\ell_n)-b^c_{y_n}]$ である。

下界は $A\ge N(\log K-L)-B$ である。

$\|F\|_F>0$ ならば $\kappa\ge [N(\log K-L)-B]/\|F\|_F^2$ である。

### 仮定

1. $N\ge1$、$K\ge2$ である。
2. 同じ $N$ 枚と同じ旧ラベルを $F$、$L$、$n_c$ の計算に使う。
3. logit は全画像で共通の有限な $V,b$ による affine head $V\phi_n+b$ である。
4. $\pi_n=\operatorname{softmax}(\ell_n)$ である。
5. $L=-(1/N)\sum_n\log\pi_{n,y_n}$ は自然対数による重みなし平均 CE である。
6. $y_n$ は $K$ クラスの one-hot である。
7. $\kappa$ を論じるときだけ $F\ne0$ を仮定する。

$\phi_n$ を生成する網の構造は仮定しない。
$\phi_n$ の非負性も微分可能性も必要ない。
CNN、Transformer、特徴学習済みの網、固定特徴のいずれでもよい。
ラベルの独立性や一様性も必要ない。
最適化法、学習率、収束、停留性、補間、重み減衰の有無も必要ない。
重み減衰を使って学習していても、この式の $L$ に入れるのは CE の部分である。
通常の温度以外を使う場合は、softmax に実際に入力する logit に合わせて $V,b$ を定義する。
乱数を含む特徴抽出器では、同じ実現の特徴と logit を使う条件付き恒等式として読む。
one-hot は指定された表記の仮定であり、確率ラベルへは $\ell_{n,y_n}$ を $y_n^\top\ell_n$ に置き換えて拡張できる。

### 証明

$\mathbf1^\top V^c=0$ より $\langle V^c,(y_n-e)\phi_n^\top\rangle_F=y_n^\top V^c\phi_n$ である。

$P\ell_n=V^c\phi_n+b^c$ より $y_n^\top V^c\phi_n=\ell_{n,y_n}-\bar\ell_n-b^c_{y_n}$ である。

画像について和を取ると最初の恒等式を得る。

$\sum_c b^c_c=0$ より $\sum_n b^c_{y_n}=\sum_c n_c b^c_c=B$ である。

画像ごとの CE を $L_n=\log\sum_c e^{\ell_{nc}}-\ell_{n,y_n}$ と置く。

Jensen の不等式から $\log\sum_c e^{\ell_{nc}}\ge\bar\ell_n+\log K$ を得る。

したがって $\ell_{n,y_n}-\bar\ell_n\ge\log K-L_n$ である。

和を取ると $A\ge N(\log K-L)-B$ を得る。

正数 $\|F\|_F^2$ で割ると $\kappa$ の下界を得る。

隙間を $J_n=\log\sum_c e^{\ell_{nc}}-\bar\ell_n-\log K$ と置くと、より詳しく $A=N(\log K-L)-B+\sum_nJ_n$ が成立する。

$J_n=D_{\rm KL}(e\|\pi_n)\ge0$ である。

### 等号条件

有限な logit では、等号は各画像について全クラスの logit が等しいとき、かつそのときに限る。
これは全画像で $\pi_n=e$ と同値である。
この場合 $L=\log K$ かつ $A=-B$ である。
$F\ne0$ なら $\kappa$ の下界の等号条件も同じである。
$F=0$ の場合には $A$ の式だけが成立し、$\kappa$ は未定義である。
$K=1$ では常に $F=0$ となる。

### この箱で満たされているか

spec は f2 の出力 $a4$ と head $W5,b5$ を明示している。
その構造は定理の線形 head の仮定を満たす。
$N=1200$、$K=10$、$H=100$ である。
MLP の実装も同じ中心化と同じ和の定義を使っている。
float64 の計算は恒等式の数値照合であり、数学的仮定ではない。
$F\ne0$ は状態ごとに確認する必要がある。
特に死んだ ReLU 状態では $\kappa$ が未定義になりうる。
$\kappa/\kappa_{lb}$ を正の倍率として読むには $\kappa_{lb}>0$ が必要である。
この定理は spec の予測 P2 の「比が 2 以上」を導かない。

## 2. 補題：H1 に使える正値条件と bias の補正

### 主張

$L<\log K-B/N$ ならば $A>0$ である。
この条件のもとでは $F\ne0$ も自動的に従う。
したがって $\kappa>0$ である。

これは H1 を「現課題の旧ラベル構造 $F$ に対する読み出しの集計した射影係数が正」と読む場合の補題になる。

### 仮定

第1節の仮定と、表示した厳密な損失条件を使う。

### 証明

第1節の右辺が正なので $A>0$ である。
$F=0$ なら $A=0$ となるため矛盾する。

$V^c=\kappa F+E$、$\langle E,F\rangle_F=0$ と置けば、$\kappa$ は原点を通る最小二乗射影の係数である。

この分解は恒等式である。
$E$ が小さいことはこの恒等式からは従わない。
$E$ を初期読み出し $v^0$ と同一視することもできない。

$A=\sum_i\|F_i\|^2\kappa_i$ なので、$A>0$ は全 unit の $\kappa_i>0$ を意味しない。

bias の補正を $\hat p_c=n_c/N$ と書くと $B/N=\langle\hat p-e,b^c\rangle$ である。

Cauchy–Schwarz より $|B|/N\le\|\hat p-e\|_2\|b^c\|_2$ である。

別の確定的な上界は $|B|/N\le\|\hat p-e\|_1\|b^c\|_\infty$ である。

旧ラベルが iid 一様なら $\operatorname{Var}(n_c)=N(1/K)(1-1/K)$ である。

固定 $K$ で $n_c-N/K=O_p(\sqrt N)$ である。

さらに $\mathbb E\|\hat p-e\|_2^2=(1-1/K)/N$ である。

したがって $\|b^c\|_2=O_p(1)$ なら $B/N=O_p(N^{-1/2})$ である。

この結論に $b^c$ と旧ラベルの独立性は必要ない。
ただし bias の大きさが制御されるという仮定は必要である。

$N=1200,K=10$ では $\sqrt{\mathbb E\|\hat p-e\|_2^2}\simeq0.0274$ である。

各クラスの比率の標準偏差は $\sqrt{0.1\times0.9/1200}\simeq0.00866$ である。

この $0.0274\|b^c\|_2$ は規模の目安であり、実現ごとの確定上界ではない。

$\sigma_b=\|b^c\|_2/\sqrt K$ と置くと、同じ目安は $0.0866\sigma_b$ になる。

bias をラベルと独立に固定した場合に限り、$\mathbb E[B/N]=0$ かつ $\operatorname{sd}(B/N)=\|b^c\|_2/\sqrt{KN}=\sigma_b/\sqrt N$ である。

学習済み bias にこの独立性の式を適用してはならない。

### 等号条件

厳密な損失条件には等号を含めない。
境界 $L=\log K-B/N$ では $A=\sum_nJ_n\ge0$ である。
境界上でも非一様な予測が一枚でもあれば $A>0$ である。
境界上で $A=0$ となるのは全予測が一様な場合に限る。
Cauchy–Schwarz の上界の等号は、非零の $\hat p-e$ と $b^c$ が比例するときに成立する。

### この箱で満たされているか

spec の旧ラベル生成は画像ごと独立の一様乱数である。
クラス数の偏りの確率的な見積もりは適用できる。
有限標本は完全均衡ではないため、$B=0$ と置くことはできない。
spec だけから bias のノルムや損失条件の成立は確認できない。
実測の $B/N$ と $\log K-L$ を直接比較する必要がある。
この補題は、再装填が読み出し更新と特徴更新のどちらで生じたかを同定しない。
この補題は、H1 から沈降を得るための自己項優勢まで証明しない。

## 3. 恒等式：一般層の押しと最終層への還元

### 主張

$u_j=\nabla_\theta m_j$、$R_{nj}=Df_n[u_j]$ と置く。
ここで $f_n$ は画像 $n$ の logit ベクトルを返す写像である。

$G_j=\langle\nabla_\theta m_j,\nabla_\theta L_u\rangle=(1/N)\sum_n\langle\pi_n-e,R_{nj}\rangle$ が成立する。

全バッチ Euclid GD の一次変位は $\mathbb E[\Delta m_j]=-\eta G_j+o(\eta)$ である。

### 仮定

固定状態を条件とした新ラベルの周辺期待が $\mathbb E[\tilde y_n\mid\theta,X]=e$ である。
期待損失の式には新ラベル同士の独立性までは必要ない。
パラメータの計量は spec の10テンソルに対する Euclid 計量である。
微分可能な点を扱う。
ReLU の折れ点や maxpool の同点では、固定した微分規約による autograd の恒等式と実際の方向微分を区別する。
適応 Snake の状態は checkpoint の値に固定する。

### 証明

$L_u=(1/N)\sum_n[\log\sum_c e^{\ell_{nc}}-e^\top\ell_n]$ である。

$\nabla_\theta L_u=(1/N)\sum_n Df_n^\top(\pi_n-e)$ である。

$u_j$ との内積を取り、随伴の定義を使えば表示式を得る。

$\pi_n-e$ はクラスで中心化されているため $R_{nj}$ を $R_{nj}^c=PR_{nj}$ に置き換えても値は変わらない。

### 等号条件

これは不等式ではなく恒等式である。
一次変位の式は有限の学習率での完全な等式ではない。
上流を固定した affine な $m_i$ の own 更新では、更新前勾配を使う一歩の変位は有限の $\eta$ でも厳密である。

### この箱で満たされているか

新課題のラベル生成は一様期待を満たす。
既知の `cnn_drive_verify_1009` はこの恒等式を照合している。
Adam の更新や適応状態の更新を含む実再生は、この Euclid GD の式そのものではない。

### 3a. f2 の own 経路と $K_n$ の出所

**主張。** f2 の own 感度は $R_{ni}^{own}=K_ng_{in}v_i$ である。

**仮定。** $x_n=a3_n$ とその平均 $\bar x$ を固定する。
動かす座標は $W4[i],b4[i]$ だけとする。
$z_{in}=W4[i]\cdot x_n+b4[i]$、$m_i=W4[i]\cdot\bar x+b4[i]$ とする。

**証明。** own 座標に制限した $\nabla m_i$ は $(\bar x,1)$ である。

この方向に $z_{in}$ を微分すると $Dz_{in}[u_i^{own}]=x_n\cdot\bar x+1=K_n$ となる。

$+1$ は bias 座標から生じる。

続いて活性化と head を微分すると $R_{ni}^{own}=K_ng_{in}v_i$ を得る。

したがって $NG_{own,i}=\sum_nK_ng_{in}\langle v_i^c,\pi_n\rangle=S_{exact,i}$ である。

単に $b4[i]$ を単位量動かす方向なら、感度は $g_{in}v_i$ である。

この方向は $\nabla m_i$ の own 部分とは異なる。

全パラメータ方向では $R_{ni}^{full}=K_ng_{in}v_i+Df_n[u_i^{up}]$ となる。

したがって full 感度まで一本の $v_i$ に平行とは限らない。

**等号条件。** 表示式は固定した微分規約での恒等式である。

**この箱で満たされているか。** spec の $G_{own}$ と $S_{exact}$ はこの式に一致する。
CNN の $x_n$ は隠れ特徴なので、$K_n\ge0$ は定義からは保証されない。
全成分非負の $x_n$ なら $K_n\ge1$ だが、LR と Snake にこの条件を自動適用できない。

### 3a 補足：$\kappa$ と自己形までの追加段階

**主張。** 最終 head の固定列が存在することで、$S_{exact}$ を $\kappa F$ と残差に分けられる。
$S_{self}$ との一致には追加の近似が必要である。

**仮定。** $F\ne0$ とし、$V^c=\kappa F+E$ と書く。
$w_{in}=K_ng_{in}$ と置く。

**証明。** $S_{exact,i}=\kappa\sum_{n,m}w_{in}\phi_{im}\langle y_m-e,\pi_n\rangle+\sum_nw_{in}\langle E_i,\pi_n\rangle$ である。

予測誤差を $\epsilon_n=\pi_n-y_n$ と置くと、厳密に $S_{exact,i}=S_{self,i}+C_i+E_i^{push}+P_i^{push}$ と分かれる。

ここで $S_{self,i}=\kappa(1-1/K)\sum_nw_{in}\phi_{in}$ である。

非対角項は $C_i=\kappa\sum_{n\ne m}w_{in}\phi_{im}(y_m^\top y_n-1/K)$ である。

読み出し残差の項は $E_i^{push}=\sum_nw_{in}\langle E_i,y_n\rangle$ である。

予測誤差の項は $P_i^{push}=\sum_nw_{in}\langle v_i^c,\epsilon_n\rangle$ である。

$\pi_n$ を $y_n$ で置き換えると $S_{oh,i}$ になる。

そのうえで対角 $m=n$ だけを残すと $S_{self,i}$ になる。

学習済みの $w_{in},\phi_{im}$ は旧ラベルに依存する。
したがって旧ラベルの iid 性だけから非対角項の期待を零とはできない。

**等号条件。** $S_{exact,i}=S_{self,i}$ の必要十分条件は三つの残差の和が零となることである。
各残差が零であることは十分条件である。

**この箱で満たされているか。** spec は $S_{oh}$ と $S_{self}$ を別々に測る設計である。
その区別は上の二段階に対応する。
$\kappa>0$ だけでは $S_{self,i}>0$ さえ保証されない。
$\sum_nK_ng_{in}\phi_{in}>0$ が追加で必要である。
この条件と $|C_i+E_i^{push}+P_i^{push}|<S_{self,i}$ があれば $S_{exact,i}>0$ が従う。

## 4. 中間層で何を測るか

### 4a. 補題：感度の旧ラベル整列と押しの誤差境界

**主張。** $\kappa$ と同じ意味の固定 head 射影は、一般の conv 層と f1 には存在しない。
代替として感度ごとの旧ラベル整列 $\kappa_{R,j}$ を定義できる。
その正値だけでは $G_j>0$ は従わない。

**仮定。** 第3節の $R_{nj}$ を使う。
own と full のどちらを使うかを指定する。
$D=\sum_n\|y_n-e\|^2=N(1-1/K)$ と置く。

**証明。** $\kappa_{R,j}=D^{-1}\sum_n\langle R_{nj}^c,y_n-e\rangle$ と定義する。

$\pi_n-e=(y_n-e)+(\pi_n-y_n)$ より $G_j=(1-1/K)\kappa_{R,j}+\varepsilon_j$ である。

ここで $\varepsilon_j=(1/N)\sum_n\langle R_{nj}^c,\pi_n-y_n\rangle$ である。

$\rho_j=[(1/N)\sum_n\|R_{nj}^c\|^2]^{1/2}$ と置く。

$e_\pi=[(1/N)\sum_n\|\pi_n-y_n\|^2]^{1/2}$ と置く。

Cauchy–Schwarz より $|\varepsilon_j|\le\rho_je_\pi$ である。

したがって $(1-1/K)\kappa_{R,j}>\rho_je_\pi$ は $G_j>0$ の十分条件である。

$(1-1/K)\kappa_{R,j}<-\rho_je_\pi$ は $G_j<0$ の十分条件である。

$\|\pi_n-y_n\|^2\le2(1-\pi_{n,y_n})^2\le2[-\log\pi_{n,y_n}]$ より $e_\pi\le\sqrt{2L}$ でも評価できる。

この損失による境界は、実測の $e_\pi$ を使う境界より緩い。

例えば $K=N=2$、$y_1=(1,0)$、$y_2=(0,1)$ とする。

$R_1=(2,0)$、$R_2=(1,0)$、$\pi_1=(0.5,0.5)$、$\pi_2=(0.01,0.99)$ とする。

このとき $\kappa_R=0.5>0$ だが $G=-0.245<0$ である。

平均 CE も約 $0.352<\log2$ である。

この例は予測確信度と感度の重みの違いによる反転を示す。

追加の測定候補は、$\kappa_R$、$\rho_j$、$\varepsilon_j$、符号余裕 $(1-1/K)\kappa_R-\rho_je_\pi$ である。

正規化した整列は $c_{R,j}=\sum_n\langle R_{nj}^c,y_n-e\rangle/[\sqrt{\sum_n\|R_{nj}^c\|^2}\sqrt D]$ と定義できる。

この量は感度の規模と方向を分離する。

**等号条件。** 分解自体は恒等式である。
誤差境界の等号は、画像を連結した $R^c$ と $\pi-y$ が比例するときに成立する。
$G_j=(1-1/K)\kappa_{R,j}$ の必要十分条件は $\varepsilon_j=0$ である。

**この箱で満たされているか。** spec は $G_{own}$ と $G_{full}$ を測るが、$\kappa_R$ は登録量ではない。
ここでの追加量は提案である。
新しい測定を実施したという意味ではない。

### 4b. 命題：中間層に CE だけから正の下界は移せない

**主張。** 旧 CE が小さいことだけから、中間層の $\kappa_R$ や $G_j$ の正値は導けない。

**仮定。** 下流写像に一般の非線形性を許す。
第1節と同じ線形 head が網の末尾にあることは維持する。

**証明。** 障害を順に分ける。

1. 第1節は logit の値 $\ell_{n,y_n}-\bar\ell_n$ の下界である。
   $R_n$ は指定方向の微分であり、値そのものではない。
   関数値の下界を微分して導関数の下界にはできない。
2. 中間層から logit までの Jacobian は、下流の gate と pool の選択を通じて画像に依存する。
   したがって通常は全画像に共通の $K\times H_j$ 行列 $V_j$ を取り出せない。
   固定した有限データで線形回帰を当てることは可能だが、その回帰残差と切片の寄与を管理する必要がある。
3. 仮に各画像で局所的な affine 表現を作れても、係数と切片が画像ごとに異なる。
   第1節の $F$ との単一の Frobenius 内積にはならない。
4. full の $\nabla m_j$ は上流座標も含む。
   その方向は他 unit の特徴も同時に変える。
   そのため対象 unit の下流経路だけでは full 感度を表せない。

具体的な反例として、二画像の固定入力を $x_1=1,x_2=-1$ とする。

対象の中間前活性を $z_n=t+x_n$ とし、$m=(z_1+z_2)/2=t$ とする。

下流の一つの特徴を $h(z)=Cz-a(z^2-1)$ とする。

最終 head を $\ell_n=(h(z_n),0)$ とする。

$t=0$ で旧ラベルを $y_1=(1,0)$、$y_2=(0,1)$ とする。

logit は $(C,0)$ と $(-C,0)$ なので $L=\log(1+e^{-C})$ である。

$C>0$ を大きくすると、$a$ によらず $L$ は任意に小さくなる。

一方で $u=\partial_t$、$R_1=(C-2a,0)$、$R_2=(C+2a,0)$ である。

したがって $\kappa_R=-2a$ である。

$q=\operatorname{sigmoid}(C)-1/2>0$ と置くと $G=-2aq$ である。

$a$ の符号と大きさで、損失を固定したまま感度の整列と押しを両方向に動かせる。

この反例には上流項がない。

したがって上流項は追加の障害であり、不可能性に必須の条件ではない。

追加仮定のある境界は作れる。

例えば $R_{nj}^c=\lambda\ell_n^c+d_n$ が全画像で共通の $\lambda>0$ により成立すれば、$\sum_n\langle R_{nj}^c,y_n-e\rangle\ge\lambda N(\log K-L)-\sqrt D\sqrt{\sum_n\|d_n\|^2}$ である。

これは第1節の logit マージンの下界と Cauchy–Schwarz による。

この形には、logit の値と感度を結びつける追加の構造仮定が入っている。

**等号条件。** 不可能性の主張に等号条件はない。
最後の条件付き境界の等号には、logit の Jensen 等号と残差の反平行条件がともに必要である。

**この箱で満たされているか。** f1 と conv の下流には非線形活性化がある。
conv の下流には pool もある。
共通の affine readout や $R^c=\lambda\ell^c+d$ の小残差は spec から保証されない。
反例は一般命題を否定する構成であり、登録された CNN の実測結果ではない。
「中間層では $\kappa$ が定義できない」は、最終 head と同じ意味の $\kappa$ が自然には定義できない、という範囲で使う。
補助 probe を新設した場合には、その probe 自身に対して第1節の定理を適用できる。
その probe の CE は元の網の CE とは別物である。

## 5. 補題：上流項の符号は一般には定まらない

### 主張

$G_{up,i}$ と $G_{own,i}$ の同符号は先験的には保証されない。
共有した上流座標による unit 間の交差項は反転を起こしうる。

### 仮定

$x_n=x_n(\theta_{up})$、$J_n=\partial x_n/\partial\theta_{up}$ とする。
$\bar J=(1/N)\sum_nJ_n$ とする。
$w_i=W4[i]^\top$ とする。
$\delta_{in}=g_{in}\langle v_i^c,\pi_n\rangle$ とする。

### 証明

$a_i=\nabla_{up}m_i=\bar J^\top w_i$ である。

$\nabla_{up}L_u=(1/N)\sum_n\sum_h\delta_{hn}J_n^\top w_h$ である。

したがって $G_{up,i}=(1/N)\sum_{n,h}\delta_{hn}\,w_i^\top\bar J J_n^\top w_h$ である。

$\bar J J_n^\top$ は各画像について一般に半正定値ではない。
$h\ne i$ の交差項の符号も制限されない。

十分条件の一例として、全画像で $J_n=J$ と仮定する。

$w_i^\top JJ^\top w_h=0$ が全 $h\ne i$ で成立すると仮定する。

この場合 $G_{up,i}=\|J^\top w_i\|^2\bar\delta_i$ である。

$K_n=K_0>0$ も全画像で共通なら $G_{own,i}=K_0\bar\delta_i$ である。

両者が非零なら同符号になる。

より直接には、$a_i\ne0$ のとき $\nabla_{up}L_u=\lambda_i a_i+r_i$、$r_i\perp a_i$ と分解できる。

その場合 $G_{up,i}=\lambda_i\|a_i\|^2$ なので、同符号の必要十分条件は $\lambda_iG_{own,i}>0$ である。

この条件は幾何学的な書き換えであり、学習から $\lambda_i$ の符号を予言する定理ではない。

反例として、共有した一つの上流出力を $x=t$ とする。

二つの最終隠れ unit を $z_1=x+1$、$z_2=-3x+1$ とする。

両 unit に ReLU を使い、head を $\ell=(\operatorname{ReLU}(z_1)+\operatorname{ReLU}(z_2),0)$ とする。

$t=0$ では両 gate が開いている。

$q=\operatorname{sigmoid}(2)-1/2>0$ と置く。

unit 1 の平均前活性は $m_1=t+1$ である。

$x=0$ なので $G_{own,1}=q>0$ である。

一方で $\partial_tL_u=q(1-3)=-2q$ なので $G_{up,1}=-2q<0$ である。

したがって $G_{full,1}=-q<0$ となる。

### 等号条件

$G_{up,i}=0$ は $\nabla_{up}m_i$ と $\nabla_{up}L_u$ が直交するときに成立する。
どちらかが零の場合も含む。
$G_{full,i}$ と $G_{own,i}$ が同符号になるために $G_{up,i}$ 自体が同符号である必要はない。
逆符号でも $|G_{up,i}|<|G_{own,i}|$ なら full の符号は維持される。

### この箱で満たされているか

spec の f2 は上流特徴が学習されるため、この交差項を持つ。
同符号の十分条件は spec から確認できない。
`c1_direction_1010` の定数モードも、下流の余接が接続を通って平均方向に透過する構造を持つ。
ただし $G_{up}$ はパラメータのブロックによる分解である。
定数モードは画像・位置の平均と中心化残差による分解である。
両者は同じ項ではない。
例えば c1 は $G_{up}=0$ でも定数モードを持つ。
上の反例は全画像を同じに複製しても成立するため、平均モードだけでも反転できることを示す。
既知の c1 の実測から f2 の $G_{up}$ の符号を転用することはできない。

## 6. 命題と条件付き予想：$\kappa\|F\|$ と Keep

### 主張

$\kappa$ 単独の比から Keep の比は決まらない。
$Q=|\kappa|\|F\|_F$ は射影された読み出しのノルムである。
$Q$ だけでも $\operatorname{med}_i|S_{exact,i}|$ は決まらない。

### 仮定

$F\ne0$ とし、$V^c=\kappa F+E$ とする。
以下では正値補題が成立する状態を考えて $Q=\kappa\|F\|_F>0$ とする。
$T_i=\sum_nK_ng_{in}(\pi_n-e)$ と置く。
中央値は spec の「生きた unit」の集合で取る。

### 証明

$S_{exact,i}=\langle v_i^c,T_i\rangle$ である。

$\widehat F=F/\|F\|_F$、$d_i=\langle\widehat F_i,T_i\rangle$、$r_i=\langle E_i,T_i\rangle$ と置く。

厳密に $S_{exact,i}=Qd_i+r_i$ である。

したがって $|S_{exact,i}|\le Q\|\widehat F_i\|\|T_i\|+\|E_i\|\|T_i\|$ である。

正の一般下界は存在しない。
$T_i$ が $v_i^c$ に直交すれば、読み出しが非零でも押しは零になる。

$M=\operatorname{med}_i|S_{exact,i}|$、$D_T=\operatorname{med}_i|d_i|$、$r_{max}=\max_i|r_i|$ と置く。

各 unit で $\big||Qd_i+r_i|-Q|d_i|\big|\le|r_i|$ なので $|M-QD_T|\le r_{max}$ である。

同じ集合についての中央値は、各要素を一様に最大 $r_{max}$ 動かしても最大 $r_{max}$ しか動かないためである。

残差が小さいときに限り、$M\simeq QD_T$ と読める。

二つの境界で残差が小さく基準の中央値が正なら、$\mathrm{Keep}_{own}\simeq(Q_k/Q_1)(D_{T,k}/D_{T,1})$ である。

この箱は $N$ が共通なので、$S_{exact}$ と $G_{own}=S_{exact}/N$ の中央値比は同じである。

$Q_k/Q_1=(\kappa_k/\kappa_1)(\|F_k\|/\|F_1\|)$ である。

したがって $\kappa$ が縮んでも $\|F\|$ の増大で $Q$ が保たれうる。

$Q$ が縮んでも $D_T$ の増大で押しが保たれうる。

$Q$ と $D_T$ がともに縮み、残差が小さければ押しも縮む。

$Q$ が保たれていても、gate、$K_n$、予測の確信度、クラス方向の相殺が変われば $D_T$ は縮みうる。

$E$ の押しが支配する場合には、$Q$ と押しの対応は弱くなりうる。

unit 間で読み出しが少数に集中すれば、全体ノルムが保たれても中央値は縮みうる。

自己項優勢を別途仮定すると、$M\simeq\kappa(1-1/K)\operatorname{med}_i|\sum_nK_ng_{in}\phi_{in}|$ とも書ける。

この式にも特徴、gate、平均入力方向の変化が残る。

### 等号条件

$S_{exact,i}=Qd_i+r_i$ は恒等式である。
全 unit で $r_i=0$ なら $M=QD_T$ が厳密に成立する。
$r_i=0$ はこの中央値等式の必要条件ではない。
残差と中央付近の順位に偶然の相殺があっても等式は成立しうる。

### この箱で満たされているか

$Q/\sqrt H$ は spec の「学んだ読み出しの大きさ」に一致する。
MLP の実装は符号を外して $|\kappa|\|F\|/\sqrt H$ を記録している。
$\kappa>0$ の状態では両表記は同じである。

spec は $\kappa$、$\|\kappa F\|/\sqrt H$、射影の説明割合、$S_{exact}$、$S_{self}$ を並べる設計である。
この組合せは $\kappa$ 単独より情報が多い。
ただし Frobenius ノルムでの小残差だけでは、各 unit の $r_i$ が小さいことを保証しない。
大きな $T_i$ が残差を増幅しうるためである。

$\mathrm{Keep}_{full}$ には $G_{up}$ の大きさと own との相殺も入る。
その比を最終 head の $Q$ だけから予言できない。
生きた unit の集合が変わると中央値の母集団も変わる。
基準中央値が零なら Keep は未定義である。
死んだ ReLU の空集合を中央値零と見なして比を作ってはならない。

**条件付き予想。** 課題間で $Q$ と感度・整列の分布が保たれ、残差と上流の相殺が増えなければ、押しの中央値も同じ桁に保たれると予想する。
これは spec の P9 と整合するが、下界定理から P9 の数値 $0.3$ は導けない。
$\kappa_k/\kappa_1\ge0.8$ も低 CE だけからは導けない。
低 CE は各状態の正値を助けるが、$\|F_k\|^2$ や Jensen の隙間の課題間変化を制限しない。
