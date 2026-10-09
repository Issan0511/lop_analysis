# Stationary Adam bias does not necessarily disappear under long averaging

2026-10-09. Independent proof audit and quantitative extension of the parent's stationary two-point construction.

**Scope.** This is a theorem about Adam's stationary response to an iid gradient stream generated at a fixed CNN state. Its almost-sure time-average conclusion is about that optimizer response. It is not a proof that a changing CNN trained over many tasks follows this stationary process, nor a counterexample to every proposed long-term CNN self-direction theorem. It does refute the shortcut “averaging Adam for a long time automatically removes its denominator/gradient correlation.”

## 1. Stationary moments and the centered two-point stream

Let q∈(1/2,1), X_t iid Bernoulli(q), and

  g_t^0=q−X_t.

Thus g=q>0 with probability 1−q and g=−(1−q)<0 with probability q, and E[g]=0. For arbitrary 0≤β₁,β₂<1, define stationary Adam moments on a two-sided iid sequence:

  m_t=(1−β₁)Σ_{i≥0}β₁^i g_{t−i},
  v_t=(1−β₂)Σ_{i≥0}β₂^i g_{t−i}²,
  A_t=m_t/(sqrt(v_t)+ε), ε>0.

At β=0 the weight at i=0 is one and all later weights are zero. Both series converge absolutely. The identity

  g²=(2q−1)g+q(1−q)

holds for the two possible gradient values. In particular, the larger positive gradient always contributes more to Adam's denominator than the more frequent negative gradient.

## 2. Exact strict stationary sign

**Theorem S1.** For every q∈(1/2,1), every 0≤β₁,β₂<1, and every ε>0,

  E[A_t]<0,

although E[g_t]=E[m_t]=0.

Proof. Write α_i=(1−β₁)β₁^i and b_i=(1−β₂)β₂^i. Condition on every gradient except g_{t−i}, and let V be their contribution to v_t. For h(v)=1/(sqrt(v)+ε),

  E[g_{t−i}h(v_t) | other gradients]
  =q(1−q)[h(V+b_i q²)−h(V+b_i(1−q)²)]≤0.

It is strictly negative whenever b_i>0, because q²>(1−q)² and h is strictly decreasing. Now multiply by α_i and sum. Exchanging the sum and expectation is justified by absolute domination: |g|≤q, h≤1/ε and Σα_i=1. At lag i=0 both α_0>0 and b_0>0, hence the total is strictly negative. When β₂=0 the later-lag terms are zero rather than strictly negative; lag zero still proves strictness. □

This proof retains the same current gradient in the numerator and denominator. It does not replace v by E[v], and it requires no independence between m_t and v_t.

## 3. Quantitative negative margin

Let

  ω=(1−β₁)(1−β₂)/(1−β₁β₂).

All possible v_t lie in [(1−q)²,q²]. On that interval,

  −h′(v)=1/[2sqrt(v)(sqrt(v)+ε)²]
         ≥1/[2q(q+ε)²].

The conditional difference in section 2 is therefore at most

  −b_i(1−q)(2q−1)/[2(q+ε)²].

Since Σ_i α_i b_i=ω,

  F(0):=E[A_t]
  ≤−Γ₀,
  Γ₀=ω(1−q)(2q−1)/[2(q+ε)²]>0.

This is an explicit finite-β bound. It is not an asymptotic expansion in β₂ or an empirical estimate.

## 4. A positive SGD mean with a negative stationary Adam mean

Couple shifted streams through the same Bernoulli sequence:

  g_t^μ=q+μ−X_t.

Their mean is μ. Define their stationary Adam responses F(μ). Let 0<δ<1−q and |μ|≤δ. For the zero-shift stationary moments, put

  r_t=(1−β₂)Σ_iβ₂^i g_{t−i}^0.

Then exactly

  m_t^μ=m_t^0+μ,
  v_t^μ=v_t^0+2μr_t+μ².

Set r_min=1−q−δ>0 and M=q+δ. Every |g_t^μ| is at least r_min and at most M, so v_t^μ≥r_min² and |m_t^μ|,|r_t+μ|≤M. Differentiating the samplewise response gives the uniform bound

  |d/dμ [m_t^μ/(sqrt(v_t^μ)+ε)]|
  ≤L_δ,
  L_δ=1/(r_min+ε)+M²/[r_min(r_min+ε)²].

Consequently,

  F(μ)≤−Γ₀+L_δ μ.

**Theorem S2.** Every finite positive shift

  0<μ<min(δ,Γ₀/L_δ)

has E[g_t^μ]=μ>0 but stationary E[A_t^μ]<0. This supplies a quantitative nonempty interval rather than relying solely on an unspecified continuity neighborhood.

## 5. Long time averages, ordinary initialization, and bias corrections

A_t^μ is a bounded measurable shift-equivariant function of a two-sided iid sequence, hence stationary and ergodic. Birkhoff's theorem therefore gives

  (1/T)Σ_{t<T}A_t^μ → F(μ)<0 almost surely,
  (1/T)Σ_{t<T}g_t^μ → μ>0 almost surely.

The same limits hold when Adam starts from any finite m_0 and v_0≥0. Couple the forward stream to a stationary version using the same future gradients. Then

  Δm_t=β₁^t Δm_0,
  Δv_t=β₂^t Δv_0.

Using sqrt(v) continuity and the denominator lower bound ε,

  |A_t−A_t^stationary|
  ≤|Δm_t|/ε + G sqrt(|Δv_t|)/ε²,

where G bounds the stationary numerator. The right side is summable and thus its Cesàro mean tends to zero. Usual Adam bias corrections 1−β₁^t and 1−β₂^t approach one geometrically and likewise do not change the limiting average.

Thus the reversal here is genuinely present in the stationary expectation and almost-sure long average. It is not merely an initial zero-moment effect. This conclusion still conditions on supplying the fixed-state iid gradient stream; it does not assert an unchanged stream while the CNN parameters themselves move.

## 6. Independent ten-class labels in a batch of sixteen

A batch size of 16 does not force every target parameter to average 16 active examples. Suppose a target channel is active on exactly one image in the batch and closed on the other 15. For six favored classes out of ten, the target derivative can be

  g_actual=(1/16)(P−I_favored),
  I_favored~Bernoulli(q), q=6/10,
  P=q+μ.

All 16 image labels can remain mutually independent and uniform across ten classes. The other 15 labels simply have zero derivative for this particular channel. This is neither a batch-one assumption nor perfectly correlated labels; it is finite effective support of a CNN gate.

Scaling every supplied gradient by s>0 gives m_actual=s m_base and v_actual=s²v_base, so

  m_actual/(sqrt(v_actual)+ε)
  =m_base/(sqrt(v_base)+ε/s).

Apply S2 with s=1/16 and effective epsilon ε_eff=16ε.

For the explicit finite parameters

  q=.6, P=.60001, μ=10^−5,
  β₁=.9, β₂=.999, ε=10^−8,
  s=1/16, ε_eff=1.6×10^−7, δ=.001,

section 3 gives Γ₀≈.000110119972103 and section 4 gives L_δ≈8.192566492254. Therefore

  E[g_actual]=6.25×10^−7>0,
  stationary E[A_actual]
    ≤−.000110119972103 + 8.192566492254×10^−5
    =−2.81943071804×10^−5<0.

Every number in this inequality is rational once the stated lower-bound formula is used. `/tmp/cnn_adam_stationary_bounds_1009.py` verifies it in exact Fraction arithmetic. The allowed positive-shift interval from this certificate extends to approximately 1.34414499×10^−5.

### A shared-Conv embedding with another nonzero overlapping channel

Take 16 2×2 RGB images with intensities ρ=(1,.8,.6,.4): one red image and 15 green images. A target shared Conv1×1 channel has red weight 2, green weight .1, bias −1, followed by ReLU→MaxPool2. It has pooled feature 1 on the red image and zero on all green images. Its positive winner is unique. A second channel has red weight 2, green weight 1.5 and bias −1, so its pooled features are 1 on the red image and .5 on the green images.

Let a_c=.4 on the six favored classes and −.6 on the other four. Define

  z=log[(P/(1−P))*(.4/.6)].

The target readout is a; the second readout is (z−1)a. Both are finite and nonzero, and both channels contribute on the red image. Its total centered logits are z a, giving favored probability P=.60001. On each green image the target derivative is zero. The target red-weight and bias gradients are both exactly (P−I_favored)/16.

The actual target first-layer mean-patch direction has components u_red=.7/16 and u_bias=1. Its proposed mean descent direction is therefore (1+.7/16) times the scalar Adam response, and has the reversed stationary sign. The other channel is not removed; its prediction cancels most of the target's current logit contribution, producing the small positive CE mean signal.

The CNN parameters are held fixed when defining this stationary-response experiment. Applying its proposed Adam updates indefinitely would change the logits and gates and would require a separate trajectory analysis.

## 7. What this does and does not settle

Established:

- Adam's gradient/denominator correlation can survive both stationary averaging and almost-sure time averaging.
- A positive but small ordinary mean gradient can coexist with a negative stationary Adam direction at finite standard β₁,β₂ and finite epsilon.
- This can occur for a target in a batch of 16 independent ten-class labels, with actual shared Conv/ReLU/MaxPool geometry and another active overlapping channel.

Not established:

- Reversal over the actual changing RL-CIFAR training trajectory.
- A stationary two-point identity for the different case in which the target receives equal sensitivity from all 16 independently labeled images. That case has a 17-point binomial gradient distribution and requires a different proof or separate numerical evidence.
- Transfer of the earlier fresh-Adam 3-favored-class counterexample to stationarity. Its stationary bias can have the opposite sign; the fresh-step example must not be reused as a long-term proof.

The correct implication is that long averaging alone does not justify dropping Adam from the theory. Proving that it becomes irrelevant for the actual CNN requires a further property of that learned trajectory or its gradient distribution.
