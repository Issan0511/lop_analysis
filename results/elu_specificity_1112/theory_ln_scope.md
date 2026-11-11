# 最終LayerNormの局所作用と、学習履歴の解釈範囲

elu_specificity_1112 の補助的な代数整理。新規測定・実験・媒介効果推定は行っていない。以下は実数演算での式であり、特記しない限り同じ入力状態・同じパラメータでの局所比較である。

**有限εとaffineを含む正確なVJP。** 一例の最終LN入力を $z\in\mathbb R^d$ とし、

\[
C=I-\frac{\mathbf1\mathbf1^\top}{d},\qquad
q=Cz,\qquad v=\frac{q^\top q}{d},\qquad s=\sqrt{v+\epsilon},\qquad
h=\Gamma\frac q s+\beta,\quad \Gamma=\operatorname{diag}(\gamma).
\]

ここでCは中心化射影であり、実験のhead permutationを表すPとは別の記号である。

任意の上流勾配 $g=\partial L/\partial h$ に対し、$a=\Gamma g$、$a_c=Ca$ と置くと、

\[
\boxed{\frac{\partial L}{\partial z}
=\frac{a_c}{s}-\frac{q(q^\top a)}{d\,s^3}.}
\]

$q\ne0$ のとき、$a_r=q(q^\top a)/(q^\top q)$、$a_t=a_c-a_r$ と分ければ、

\[
\boxed{\frac{\partial L}{\partial z}
=\frac{a_t}{s}+\frac{\epsilon}{s^3}a_r.}
\]

平均方向は厳密に除去される。中心化入力 $q$ に沿うradial成分の係数は $\epsilon/s^3$、それに直交する中心化成分の係数は $1/s$。したがって両係数の比は **$\epsilon/(v+\epsilon)$** である。これは「γを上流勾配へ掛けた後」の分解であり、任意のaffine γを持つLN全体の特異値をそのまま列挙したものではない。

$v\gg\epsilon$ ならradial伝達は強く抑制されるが、有限εでは一般にゼロではない。$v\ll\epsilon$ では相対抑制は弱まり、全体の係数も大きくなり得る。$q=0$ ではradial方向自体を定義せず、VJPは $C\Gamma g/\sqrt\epsilon$ となる。よって「LNは常にradial勾配を完全に消す」「LNなら勾配は常に小さい」とは言えない。

**classifierのW縮小はLNで相殺されない。** 分類器を $\ell=Wh+b$、CEのlogit勾配を $r=p-y$ とする。最終LNの上流は $g=W^\top r$。$\mathbf1^\top r=0$ なので、クラス共通の行成分を除いた $W_c$ で $g=W_c^\top r$ と書ける。同じ $z,\gamma,\beta,r$ のもとで $W_c\mapsto cW_c$ とすれば、

\[
\nabla_z L\mapsto c\,\nabla_z L.
\]

LNの分母 $s$ はclassifierより手前の $z$ から計算され、Wを縮めても変わらないためである。実際のphysical shrinkは確率も変えるので、一般形は $cJ_{\mathrm{LN}}^\top W_c^\top r(c)$ となり、「勾配が常にちょうどc倍」とは限らない。平均logit補償も各例の確率一致を保証しない。Pの有無とscaleの比較は別に保つ。

長期にはγやheadの再成長が起こり得る。例えばfull Wをc倍にし、同時にγ・βを1/c倍すれば、bを固定してlogitを保存できる。γだけを1/c倍にする場合にも、$b'=b+(1-c)W\beta$ と補償すれば同じ関数になる。しかしこれは追加のパラメータ変更であり、**LNがW縮小を自動的に打ち消す性質ではない**。測るべき有効な写像には $W_c\operatorname{diag}(\gamma)$ も含まれる。同じ関数を表す再パラメータ化でも、Adamの状態とその後の学習経路が同じとは限らない。

**zero headの回復をradial抑制だけに帰属できない。** 本実験のZはhead weight・biasをゼロにし、head Adamも消す。fork直後は $W=0\Rightarrow g=0$ なので、LNの有無にかかわらずraw body勾配がゼロになる。これはradial成分だけの抑制ではなく、上流の全方向を遮断する初期条件である。それでもbody Adamをkeepした腕では過去のmomentumにより実変位が生じ得る。headは最初の更新から再学習し、その後のbody勾配・確率・LN affine・optimizer状態も変わる。Z_keep、Z_clear、F0の長期差はこれらを含む方策の差であり、局所LN VJPの一項の因果量ではない。Z_clear−F0も、この条件での学習済みbodyの効用比較であり不可逆な能力損失の測定ではない。

**全負領域のELU共通シフトと、直後のLN。** $x=\operatorname{ELU}(z)=e^z-\mathbf1$ とし、全座標が負で、共通シフトδの後も全座標が負とする。すると、

\[
x'=\operatorname{ELU}(z+\delta\mathbf1)
=e^\delta x+(e^\delta-1)\mathbf1.
\]

ここで $q_x=Cx$、$v_x=\|q_x\|^2/d$ とすれば、LN直前の中心化成分は $q_x'=e^\delta q_x$ なので、

\[
u' = \frac{q_x}{\sqrt{v_x+\epsilon e^{-2\delta}}},\qquad
\left.\frac{\partial u'}{\partial\delta}\right|_{0}
=\frac{\epsilon q_x}{(v_x+\epsilon)^{3/2}}.
\]

ε=0かつ $v_x>0$ なら、ELU出力の共通位置変化と正のscale変化がLNで厳密に消え、固定γ・βのaffine出力も不変になる。有限εでは近似にとどまり、負方向への大きなシフトで $e^{2\delta}v_x$ がε以下になると相殺は崩れ、出力はβへ近づく。γを含む上流勾配aに対する共通シフト方向の微分も $\partial L/\partial\delta=\epsilon q_x^\top a/(v_x+\epsilon)^{3/2}$ となる。

この式には「同一LN group内の全座標が負」「同じδが全座標へ加わる」「ELUの直後にそのgroupを正規化する」という条件がある。混在する正負領域、unitごとに異なる変位、線形写像と残差加算を挟むViTのFFN出力には一般にそのまま適用できない。MLPのfloat32 expm1実装では負のtailが丸めで−1へ一致し、実際の微分もゼロになる場合がある。正規化は既に失われた有限精度の差を復元しない。ViTのF.eluとはこの数値的tailも同一ではない。

**今回の問いへの範囲。** 最終LNがその瞬間の伝達を変えることは上式で明確である。一方、学習済みγ・head scale、LNより手前の表現、過去のoptimizer状態は別の要因である。Study Aはactivation×final affine LNを最初から割り当てた総効果、Study BのLN比較はReLUで学習したall-five-LN/no-LN履歴の比較であり、最終LNだけの瞬間的な除去ではない。この代数だけから、以前のMLP/ViT差を「LNだった」と確定したり、今回Aの利益をELU共通シフトの相殺で説明済みとしたり、その媒介割合を計算したりはできない。
