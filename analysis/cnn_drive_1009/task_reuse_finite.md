# 同じラベルを課題内で繰り返す、一般 CNN の有限期間 SGD 証明

2026-10-09。任意の深さ・共有 Conv・MaxPool・trainable bias を許す条件付き定理。第一 Conv の平均前活性を対象にする。初期状態は旧学習と任意に依存してよいが、新しい課題ラベルは開始時点から独立に一様抽出する。ラベルは課題中に再抽選せず、同じ画像には同じラベルを使う。

## 1. 課題開始時点だけで平均する

全パラメータを θ、開始状態 θ₀、固定入力の第一 Conv mean を m=uᵀθ とする。u は対象 filter の mean augmented patch であり、他のパラメータ成分は0。課題の画像ラベル Y₁,...,Y_N は iid uniform C-class。H 回の minibatch 列 B_t と正の学習率 η_t は新ラベルから独立で、開始時点で条件付けて固定する。画像順のランダムな reshuffle もラベルから独立なら条件付けてよい。

実際の全パラメータ SGD は θ_{t+1}=θ_t−η_t g_t(θ_t;Y)。そのままの同じ Y を全 t で再利用する。開始点で凍結した参照下降を

  D_ref(Y)=Σ_t η_t uᵀg_t(θ₀;Y)

とする。参照だけを凍結し、実過程は全パラメータを更新する。

開始点の logits f_n、probability p_n、R_n=Df_n(θ₀)[u] と、各画像の総重み

  q_n=Σ_{t:n∈B_t} η_t/|B_t|

を用いると、厳密に

  E_Y D_ref=Σ_n q_n R_n·(p_n−1/C).                 (1)

課題中の状態を条件付けて古い Y を独立扱いしていない。等学習率・equal coverage の場合 q_n=τ/N、τ=Ση_t なので、右辺は τ G₀、G₀=N⁻¹Σ R_n·(p_n−1/C)。G₀ の正は既存の class-order/contrast 条件や以下の実現例で導く。

## 2. 全パラメータが動く誤差を有限に抑える

θ₀ の半径 r の Euclidean ball 内で全 batch/label に対して

  ||g_t(θ;Y)||≤G*,
  ||g_t(θ;Y)−g_t(θ';Y)||≤L*||θ−θ'||

を仮定する。これらは目的の符号ではなく、forward Jacobian/Hessian の上界から得る幾何的条件。τG*≤r なら全 label 実現の実軌道が ball 内に留まる。帰納的に ||θ_t−θ₀||≤G*Σ_{s<t}η_s だから、

  |[m(θ₀)−m(θ_H)]−D_ref(Y)|
   ≤E_H:=||u|| L*G* Σ_t η_t Σ_{s<t}η_s
   = (||u||L*G*/2)[τ²−Σ_tη_t²].                  (2)

従って (1) が E_H を上回れば、実際の全共同更新後の期待 mean は strict に下がる。各実現が下がることや、毎 step の条件付き期待が正であることは要求しない。

固定 winner/gate の ball で ||Df_n||op≤J*、||D²f_n[v,w]||₂≤Q*||v||||w|| なら、CE について

  G*=√2 J*,  L*=(J*)²/2+√2 Q*

が使える。softmax Jacobian の operator norm≤1/2 と ||p−e_y||≤√2 から従う。strict な有限個の ReLU/MaxPool margins は正の ball を与える。負の前活性や閉じた ReLU があってもよく、境界をまたがないことだけをこの局所評価で要求する。

これは一般 CNN の有限期間定理であり、ここから無限時間の条件維持や Adam を結論しない。

## 3. ラベル再利用で雑音の大きさはどう変わるか

P=I−11ᵀ/C とすると、課題開始時点の参照について

  Var_Y D_ref = Σ_n q_n² ||P R_n||²/C.             (3)

同じ画像のラベルは何度現れても同じ変数なので、回数を数えてから二乗する。各出現でラベルを独立に引き直すなら、代わりに Σ_tΣ_{n∈B_t}(η_t/|B_t|)²||P R_n||²/C。

等学習率・同じ batch size で E epochs の equal coverage なら、再利用の分散は fresh-per-occurrence の E 倍になる。期待の一次項は同じでも、noise の累積が異なる。実 RL-CIFAR の400 epochsというラベルスケジュールでは、この凍結参照の分散比は400。これは実 Adam の分散や学習後状態の分散を計算した値ではない。

## 4. 負の mean、混在する gate、非比例チャネルを持つ非空例

二画像は赤と緑の RGB 画像で、唯一の非零 colour plane は

  rho=((1,.8),(.6,.4)).

共有 Conv1×1 は二チャネル。filter rows を

  (1.3,1.03,0), (1.03,1.3,0)、両 bias=−1

とし、ReLU→MaxPool2→3-class linear head を接続する。全 Conv/head の重みと全 bias を学習する。pool 後の特徴は

  H=((.3,.03),(.03,.3))

で rank 2。両チャネルが両画像に反応する。target channel mean は −.1845。前活性には正と負が混在する。最小 absolute ReLU margin は .03、最小 pooled winner gap も .03。

a=(−1,0,1)、V_{c,:}=a_c(.24,.15)、output bias=.018a とする。logits は画像ごとに t_n a、t=(.0945,.0702)。target mean gradient は u=(.35,.35,0,1)、winner response の方向微分は両画像で1.35。従って R_n=.324 a。

A(t)=Σ a_c softmax(ta)_c とすると G₀=.324[A(.0945)+A(.0702)]/2。0≤t≤.1 では softmax の各確率≥e^(−.2)/3≥4/15。中心化 a の squared norm は2なので A'(t)=Var(a)≥8/15。よって厳密に

  G₀≥Γ=(81/250)(8/15)(1647/20000)
       =44469/3125000>0.                            (4)

## 5. 具体的な正の半径・学習率・課題長

全パラメータ球の半径 r=1/200=.005 とする。winner augmented patch norm は √2<3/2。各 preactivation/activation の変化は高々(3/2)r、winner gap の変化は高々3r=.015<.03。従って ball 全体で元の gate と winner が保たれる。

この cell では h_n=A_n θ_conv、||A_n||≤ρ*=3/2、f_n=Vh_n+b_out は bilinear。||V₀||F<.401、max||h_n(θ₀)||<.302 より

  ||Df_n||²≤(.401+r)²(3/2)²+(.302+(3/2)r)²+1
            <(5/4)².

混合二次微分は δV₁A_nδθ₂+δV₂A_nδθ₁ なので Q*≤3/2（二つの parameter block の Cauchy–Schwarz により余分な2は不要）。有理数上界として

  ||u||≤9/8、G*=15/8、L*=97/32

を使える。

H=8、η=1/10000、τ=1/1250 とすると τG*=.0015<r。式(2)の誤差上界は1.79033203125e−6、式(4)から

  E[m₀−m_H]≥τΓ−E_H
    =306999423/32000000000000
    =9.59373196875e−6>0.                            (5)

これは finite-step 数値の符号だけではなく、ball 全体の解析上界と有理数比較による保証である。全9 label assignments を実 raw SGD で8回ずつ更新すると、期待下降は約1.41899461157e−5、最大参照誤差8.56e−8未満。負の下降（meanが上がる）ラベル実現も含む。分散式(3)は全列挙と一致し、fresh-step labels に対する比は8。

## 6. 元の全 NTK と literal capacity-self も接続する

λ=1、Φ=(1/2)logdet(I+K)、K=JJᵀ は全 raw Conv/head/bias の Jacobian Gram。Xi の rows は winner augmented input (1,0,0,1)、(0,1,0,1)。sample-major vectorization で

  K=(XiXiᵀ)⊗(VVᵀ)+(HHᵀ+11ᵀ)⊗I_3.

target column h=(.3,.03)ᵀ、r_h=(1.35,1.35)ᵀ なら、mean 方向 q に

  K'=(r_h hᵀ+h r_hᵀ)⊗I_3.

他の Conv channel を除き、その head column も除く literal self では H→h、V→V[:,target] とする。Conv bias と output bias は残し、全 Jacobian を再計算する。K' は同じ形。有限有理数で厳密に

  ∂q Φ_full=10185021/24798659>0,
  ∂q Φ_self=25201935/61683554>0.                    (6)

K' は一般に不定符号であり、PSD を仮定していない。他チャネルがある全体の resolvent と、孤立自己の resolvent の双方を実際に評価している。

さらにこの符号は上の半径 .005 の ball 全体で保たれる。J はこの bilinear cell で affine、J'=D J[u] は constant。N=2、ρ*=3/2、J*=5/4、||u||≤9/8 により

  ||K−K₀||≤2N J*ρ*r=.0375,
  ||K'−K₀'||≤2N(ρ*)²||u||r=.050625.

full/self とも同じ上界が使える。開始点の ||K₀'||op<1.1 は、小行列 r_hhᵀ+hr_hᵀ の固有値 r_h·h±||r_h||||h|| から従う。K≥0 と λ=1 なので resolvent norm≤1、resolvent identity から

  |∂q Φ(θ)−∂q Φ(θ₀)|
    ≤(NC/2)[.050625+.0375×1.1]
    =.275625 < min(∂q Φ_full(θ₀),∂q Φ_self(θ₀)).    (7)

従って実際の全 label 軌道が通る ball 全体で、元の容量自己項と full 容量の下降方向が mean を下げる側。式(5)は実際の reused-label task の期待 net update がその向きに進むことを保証する。CE を self と呼び直したものではない。

## 7. 限界

- 有限の課題期間。無限反復で球内に留まる保証は別。
- optimizer は同じ正 scalar learning rate の全 raw SGD。Adam の分母は別の問題。
- 負の mean と inactive sites を許すが、指定期間の gate/winner changes は抑える。
- 全ラベル平均での方向であり、各 label realization の方向ではない。
- 具体例は1×1一段Conv。一般定理は深いCNNにも使えるが、実RL-CIFARの各状態で上界を測ったものではない。

検算: verify_task_reuse_finite.py、結果 ../../results/cnn_drive_1009/task_reuse_finite.json。コードのNTK/JVPは上の有理数式と丸め誤差内で一致。無限時間やAdamへの外挿には使わない。

独立監査: [task_reuse_finite_review.md](task_reuse_finite_review.md)。
