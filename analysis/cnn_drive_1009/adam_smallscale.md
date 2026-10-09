# Small-gradient stationary Adam: when epsilon domination does and does not recover the self direction

2026-10-09. Here “self signal” means the explicitly positive mean CE gradient in the specified family; this note does not separately identify it with the original isolated-network logdet derivative. Consider a family of iid gradient streams at fixed CNN states, indexed by a scale s>0. The conclusions concern each stream's stationary Adam response: first the long-time stationary limit at fixed s, then s→0. They do not, by themselves, justify exchanging these limits or substituting stationary averages along a changing random CNN trajectory s_t→0.

## 1. Setup

Let ξ_t be iid, Eξ=0, |ξ|≤K, Var(ξ)=σ². For k=2 or k=3 let

  g_t(s)=s ξ_t+s^k μ, μ>0,
  δ=μ s^(k−1).

For 0≤β₁,β₂<1 define normalized stationary noise moments

  M=(1−β₁)Σ_i β₁^i ξ_{−i},
  R=(1−β₂)Σ_i β₂^i ξ_{−i},
  V=(1−β₂)Σ_i β₂^i ξ_{−i}²,
  W_δ=sqrt(V+2δR+δ²).

The actual stationary moments are m=s(M+δ), v=s²W_δ². With fixed epsilon ε>0, the expected Adam descent direction is

  F(s)=E[s(M+δ)/(ε+sW_δ)].

All formulas retain numerator/denominator dependence. Write

  ω=(1−β₁)(1−β₂)/(1−β₁β₂),
  C₁=E[M sqrt(V)],
  C₂=E[MV]=ω E[ξ³],
  D=E|M|+E sqrt(V)
    ≤σ[1+sqrt((1−β₁)/(1+β₁))]≤2K.

The equality for C₂ follows by independence and centering: only equal lag indices contribute. The first coefficient C₁ is a different nonlinear weighted-noise moment; it cannot generally be replaced by a multiple of Eξ³.

## 2. Expansion with a finite explicit remainder

The exact algebraic identity

  1/(ε+sW)=1/ε−sW/ε²+s²W²/ε³
            −s³W³/[ε³(ε+sW)]

gives

  F(s)=μs^k/ε − C₁s²/ε² + C₂s³/ε³ + R_k(s),

where, with Q=K+|δ|,

  |R_k(s)|
   ≤ s² |δ|(D+|δ|)/ε²
      +s³[|δ|σ²(1+2ω)+|δ|³]/ε³
      +s⁴Q⁴/ε⁴.                         (R)

This is a finite bound, not an unqualified asymptotic notation. It holds without first requiring sQ<ε; that inequality makes the bound useful for the intended epsilon-dominated regime.

Proof of the replacement terms. W_δ is the weighted ℓ² norm of the sequence ξ_i+δ, so |W_δ−W_0|≤|δ|. Hence

  |E[(M+δ)W_δ]−C₁|≤|δ|(D+|δ|).

Also, directly expanding the quadratic gives

  E[(M+δ)W_δ²]
   =C₂+δσ²(1+2ω)+δ³,

using E[MR]=ωσ² and E[M]=E[R]=0. The final algebraic remainder is bounded by s⁴Q⁴/ε⁴. □

## 3. Relative sign conclusions for k=2 and k=3

For k=2,

  F(s)=s²(μ/ε−C₁/ε²)+O(s³).

Therefore a strict inequality **εμ>C₁** gives the positive sign for all sufficiently small s, with an explicit finite-scale certificate obtained by comparing the positive leading margin with (R) and the displayed s³ term. If εμ<C₁, the sign is negative for sufficiently small s.

For k=3,

  F(s)=−C₁s²/ε²+s³(μ/ε+C₂/ε³)+O(s⁴).

Consequences:

- C₁>0: negative Adam bias dominates the positive self signal μs³/ε as s→0.
- C₁<0: its leading contribution helps the positive direction.
- C₁=0: the cubic coefficient μ/ε+C₂/ε³ determines the leading sign, provided it is nonzero.
- C₁=C₂=0: the self coefficient is positive and the remainder is O(s⁴), so the sign is eventually positive.

Thus “sqrt(v)≪ε, so Adam is approximately m/ε” is an **absolute** approximation. It does not imply a relative approximation to an expected signal that vanishes faster than the noise correction. An O(s²) denominator-correlation error can overwhelm an O(s³) expectation even though each individual normalized gradient is close to m/ε.

## 4. Symmetric noise gives both exact positivity and relative recovery

Assume the law of ξ is symmetric about zero. Then the whole iid history is invariant under ξ_i↦−ξ_i, implying C₁=C₂=0 and cancellation of every analogous zero-shift odd numerator/even denominator expectation.

### 4.1 Exact positive sign at every finite scale

In fact the sign needs no small-s approximation. Write stationary weights α_i=(1−β₁)β₁^i and b_i=(1−β₂)β₂^i. Condition on all ξ_j except ξ_i. Its contribution to the numerator is α_i(ξ_i+δ). The remaining squared denominator is a fixed nonnegative constant C. The scalar function

  x ↦ x/[ε+s sqrt(C+b_i x²)]

is odd and strictly increasing for ε>0, including b_i=0. Symmetry gives

  E[(ξ_i+δ)/(ε+s sqrt(C+b_i(ξ_i+δ)²)) | other lags]>0

for every δ>0. Summing the positively weighted contributions proves

  **F(s)>0 for every s>0, μ>0, k≥1 and 0≤β₁,β₂<1.**

This is an exact stationary Adam theorem. It does not assume a nonnegative preexisting first moment: that moment is averaged with its actual stationary dependence on the history. Conditioning on individual iid lags is essential; merely assuming joint central symmetry for an arbitrarily dependent history would not justify this proof.

### 4.2 Finite relative error in the epsilon-dominated regime

Let f_s(w)=w/(ε+sw). It is 1/ε-Lipschitz and at most w/ε. The identity

  F(s)=sδ/ε−(s²/ε)E[(M+δ)f_s(W_δ)]

and symmetry E[M f_s(W_0)]=0 give

  |F(s)−μs^k/ε|
   ≤ μs^(k+1)(D+μs^(k−1))/ε².

Consequently,

  |F(s)/(μs^k/ε)−1|
   ≤ s[D+μs^(k−1)]/ε →0.                (S)

This is the relative statement missing from the generic epsilon-domination argument. It holds even for k=3 or larger because symmetry cancels the noise-only bias before comparison with the small self signal.

## 5. A nonempty shared-CNN family with drift of order h³ and noise of order h

The symmetry requirement can follow from an actual finite CNN structure, rather than being imposed directly on the optimizer response.

Use B=16 images, 8 red and 8 green, with 2×2 spatial intensities ρ=(1,.8,.6,.4). A shared Conv1×1 has two channels, with red/green weights

  channel 1: (1+s,1+.1s),
  channel 2: (1+.1s,1+s),
  both biases: −1.

Follow with ReLU→MaxPool2. For 0<s<.1, the unique top-left winners have pooled rows

  H_red=s(1,.1), H_green=s(.1,1).

Nonwinning preactivations are negative. Both channels respond on every image, the pooled feature matrix has rank two, and neither channel is a vanishing-strength perturbation of the other: their relative strengths remain fixed. The actual first-channel mean preactivation is

  m_1=−.3+.385s,

so the family has a negative bulk mean and small surviving pooled responses. Its MaxPool winner gaps stay bounded away from zero; only the winning ReLU activation approaches its threshold.

Take ten class coefficients

  a=(−4.5,−3.5,...,4.5),
  q_0=(.2,.1),
  head V_ic=s q_0,i a_c,
  output bias b_c=s²*.05 a_c.

The logits are exactly f_nc=s² T_n a_c, with T_red=.26 and T_green=.17. Define

  A(t)=Σ_c a_c exp(ta_c)/Σ_c exp(ta_c).

Then A(0)=0, A′(t)=Var_{softmax(ta)}(a)>0 and A′(0)=8.25. For each target Conv coordinate j,

  g_j(s)=s ξ_j+s³ μ_j(s),
  ξ_j=−(1/B)Σ_n d^0_nj a_{Y_n},
  μ_j(s)=(1/B)Σ_n d^0_nj A(s²T_n)/s²>0,

where d^0_nj≥0 is its fixed sensitivity after extracting the head's factor s. Independent uniform ten-class labels make ξ_j **exactly symmetric** at finite B=16. Cross-coordinate dependence is allowed because the directional expectation is a sum of the coordinate expectations.

For target red weight, green weight, and bias, respectively,

  μ_red(s)→.2145,
  μ_green(s)→.14025,
  μ_bias(s)→.35475.

The actual mean-patch direction weights are (.35,.35,1), so its CE self-direction coefficient approaches

  .35(.2145+.14025)+.35475=.4789125>0.

Every coordinate therefore has positive stationary Adam mean for all finite s>0 by section 4.1. As s→0, section 4.2 gives

  E[Adam mean direction]
     =(s³/ε)[.4789125+o(1)]>0.

The coordinate noises have bounds K_red=K_green=.45, K_bias=.9; (S) supplies concrete finite-s error bounds. Output biases are included. Other channels overlap on the target images and are finite for every s>0.

This proves a nonempty structural positive family matching the **h³ signal versus h noise** scaling. It is a family of fixed states and stationary optimizer responses. An unconstrained training update can destroy its class-head symmetry or move it off the prescribed scale family, so its temporal invariance is a separate problem.

## 6. A skew-noise family where epsilon domination still fails relatively

Take ξ=q−Bernoulli(q), q>1/2. As in the stationary two-point proof, condition on one history lag. Since sqrt(v) is increasing,

  ω(1−q)(2q−1)/2 ≤ C₁ ≤ ωq(2q−1)/2,
  C₂=ωq(1−q)(2q−1).

In particular C₁>0. The k=3 self signal loses asymptotically, no matter how fixed and positive ε is. This remains true inside an arbitrarily strong epsilon-dominated regime.

For β₁=.9, β₂=.999, q=.6, ε=10^−8, s=10^−12, μ=1,

  sK/ε=6×10^−5,
  E[g]=s³=10^−36>0,
  F(s)≤−3.96371579148×10^−13<0.

The upper bound includes every term of the finite remainder (R), with no Monte Carlo estimate of C₁. The variance contribution to the denominator is already much smaller than epsilon, yet the expected direction is opposite because the self signal is smaller still.

This noise shape can also arise from ten-class CE with six favored classes: pooled feature h=s, class-head vector s times the centered favored/other coefficient, and logits of size s². The favored probability satisfies P(s)−.6=.24s²+O(s⁴), so the gradient is s[.6−I_favored]+s³μ(s), μ(s)→.24>0. It has precisely the signal/noise powers in question and the skew term above. This observation concerns fixed-state scaling, not a proved live training trajectory.

## 7. Zero third moment alone is not the structural cancellation needed

For example, let ξ take values −2,−1,3/2 with probabilities 1/7,2/5,16/35. Then

  Eξ=0, Eξ³=0, but E[ξ|ξ|]=2/35>0.

At β₂=0, C₁=(1−β₁)E[ξ|ξ|]>0 even though C₂=0. Thus eliminating the third moment does not generally eliminate the leading small-scale Adam bias. Exact symmetry is one clear sufficient structural mechanism; directly controlling C₁ and C₂ is another.

For approximate symmetry, let ξ_s couple to a symmetric bounded variable ζ_s with both bounded by K and E|ξ_s−ζ_s|²≤d_s². Coupling iid histories coordinatewise gives

  |C₁(s)|≤2K d_s,
  |C₂(s)|≤3ωK²d_s.

For k=3, the sufficient rate d_s=o(s) makes the leading asymmetric correction negligible relative to s³/ε (with fixed ε and μ bounded below). A fixed small asymmetry is not enough for a uniform s→0 guarantee: its O(s²) correction can eventually dominate. Thus positivity in an open neighborhood at each fixed s must not be confused with one fixed-width neighborhood that works all the way to zero scale.

## 8. Verification and interpretation

`verify_adam_smallscale.py` uses exact Fraction arithmetic to verify finite upper/lower bounds for standard β₁,β₂ and epsilon. It checks:

- k=3 positive self signal with strictly negative Adam mean under skew noise;
- k=2 sign reversal when its leading positive margin is too small;
- k=2 positive Adam sign when εμ exceeds the entire C₁ interval with remainder accounted for;
- the symmetric relative bound and positive sign;
- the zero-third-moment counterexample above.

Outputs are in `../../results/cnn_drive_1009/adam_smallscale.json`.

The useful positive conclusion is conditional but substantive: **symmetric bounded iid label noise, arising here from paired class-head coefficients, makes stationary Adam preserve the self direction and recover its relative magnitude in the epsilon-dominated limit.** Without an odd-noise cancellation or an explicit margin bound, epsilon domination alone only provides an absolute approximation and does not settle the sign of a cubic dying-channel signal.

### Literal CNN/autograd check of the positive scale family

`verify_adam_smallscale_cnn.py` constructs the batch-16 shared Conv/ReLU/MaxPool/ten-class head above and differentiates actual CE while holding the constructed parameter values fixed. At s=.1,.01,.001, the mean preactivations are −.2615, −.29615, −.299615. The expected CE mean gradients divided by s³ are .4789082838, .4789124996, .4789125000, approaching the proved coefficient .4789125. The exact-coordinate decomposition g=sξ+s³μ(s) agrees with autograd to at most 3.47e−18 in these checks. The minimum MaxPool winner gap stays above .2. Outputs are saved in `../../results/cnn_drive_1009/adam_smallscale_cnn.json`. This verifies the CNN realization and scale algebra; stationary Adam positivity follows from the analytic symmetry theorem, not from a finite simulation that could miss such small expectations.
