# Independent audit: constant-rate boundary and ten-class rational verifier

2026-10-09. Reviewed `analysis/cnn_drive_1009/adam_constant_rate_boundary.md`, especially sections 1-6, and read `verify_adam_tenclass.py` against my ten-class equilibrium and bias-response derivations. No repository files were changed and no training was run.

**Verdict: PASS for the stated impossibility results and verifier formulas.** Sections 7-8 are consistent with the existing finite-horizon averaging estimates. One minor precision improvement is noted below; it does not affect the conclusions.

## 1. Each output bias cannot converge to a finite value

At the first batch of each task, the network and old optimizer state are measurable with respect to the old history. Conditioning additionally on the new batch indices does not reveal its newly independent labels. The class-c count therefore has the stated Binomial(B,1/C) law. Two adjacent attainable counts give a prediction-independent lower probability for a gradient of magnitude at least 1/(2B).

The conditional recurrence argument is sound: iterating the conditional failure bound gives `(1-p_*)^K` for avoiding the event at K successive task starts. A countable union over possible final successes proves infinitely many occurrences almost surely. Within-task label reuse has not been ignored.

For this output-bias coordinate, |g_t|<=1 globally, and zero-initialized second moments give `vhat_t<=1`. A finite bias limit would imply vanishing increments, hence q_t->0, mhat_t->0, and m_t->0. The first-moment recurrence with beta1<1 would then imply g_t->0, contradicting the recurring lower bound. The argument holds separately for each of the finitely many class biases, so the simultaneous almost-sure conclusion is valid. It needs no frozen-network assumption or hidden-layer smoothness.

The adjacent-moment estimate in section 2 is correct. A large g_t forces at least one of |m_t| and |m_(t-1)| above `(1-beta1)a/(1+beta1)`, giving the asserted actual-update lower bound. Infinitely many such gradient times yield infinitely many update times even if adjacent pairs overlap. The standard coefficient is exactly `eta/[608(1+epsilon)]`.

## 2. Fixed compact containment and path unboundedness

For any fixed compact finite-parameter set K and the finite fixed image dataset, continuity and finiteness of softmax logits give a uniform `p_c<=1-kappa` with kappa>0. This remains true despite changes of ReLU or MaxPool branch; only forward continuity is needed.

On a block where every task label is c, every batch gradient satisfies g_t<=-kappa as long as the path stays in K. The bound

    m_l<=-kappa+(1+kappa)beta1^l

allows arbitrary previous retained moments because |m|<=1 globally. After the finite burn period, bias correction increases the magnitude of the negative moment and the denominator remains at most 1+epsilon. The subsequent positive b_c increments must traverse more than its width in K if the forced block is long enough. Hence either another parameter has already left K or b_c itself must leave.

Each predetermined M-task block has positive all-class-c probability `C^(-NM)`. Disjoint blocks use independent task label assignments; the within-task shuffles cannot invalidate the forcing event. This proves both the survival bound and zero probability of permanent containment in any fixed compact K.

Taking the countable union over integer boxes `[-R,R]^P` correctly yields almost-sure unbounded supremum of the complete raw trajectory. It does not prove that every coordinate diverges, that a coordinate tends to infinity, or that late-time distributions cannot remain tight. The document correctly distinguishes all these claims and emphasizes the potentially astronomical waiting times.

## 3. Hidden observables and stationary velocity

The output-bias conclusions do not prohibit convergence or a downward net offset of a selected hidden mean. Section 4 correctly allows zero hidden gradients, recurrent behavior, distributional convergence, and finite-time sinking.

For an integrable stationary hidden observable m, `E[m_(t+1)-m_t]=0` is valid. The stronger nonergodic almost-sure claim also holds: for every a>0,

    sum_T P(|m_T|>aT)=sum_T P(|m_0|>aT)<infinity.

The first Borel-Cantelli lemma requires no temporal independence, so m_T/T->0 almost surely. Telescoping then gives zero asymptotic average velocity. The explicit integrability and autonomous stationary-process qualifications are essential and are present. Continuing global bias correction is correctly separated from an invariant law of asymptotically uncorrected augmented dynamics.

Minor wording: section 5 item 2 defines a negative velocity using a limsup. Its exact equivalent is `liminf_T (1/T)sum_(t<T)u^Tq_t>0`; existence of a time-average limit is not required. Replacing “a strictly positive long-run average” by this liminf statement would remove that small ambiguity.

## 4. Finite constant-rate estimates

The finite-horizon tracking term is the existing pathwise estimate with `sum delta_k^2=K eta^2`. Constant weights remove step-size variation from the Poisson telescoping term, leaving a boundary term at most 2U_i eta and parameter changes at most `K eta omega_i(B eta)`. The Taylor and martingale terms have the correct `K eta^2` and `eta sqrt(K/alpha_i)` factors. The within-task margin is retained.

At fixed clock horizon S=K eta, these bounds vanish as stated when eta tends to zero. This is a finite-horizon result and does not contradict eventual compact exit at any fixed positive eta.

The energy estimate follows from iterating

    E_(k+1)<=(1-a eta)E_k+[R_(k+1)-R_k]

when 0<=1-a eta<=1. Summation by parts bounds the weighted remainder increments by 2 sup|R|, giving the displayed `E_K<=(1-a eta)^K E_0+2e_E`. Combined with an endpoint error and the local normal-distance bound, the stated finite mean-drop criterion is sound. Dividing the finite cumulative bound by K eta produces the section 8 orders, with a nonvanishing deterministic error at fixed eta.

## 5. `verify_adam_tenclass.py` matches the analytical notes

The verifier correctly checks:

- The ten-class bias RMS floor 1/40, variance `(3/40)^2`, retained-label covariance bounds, cross bound 171/10^6, and gamma lower bound with epsilon rather than 2epsilon.
- The predictive coefficient `H/(NC)` and its rank-nine positive subspace, without claiming full positive definiteness or hidden equilibrium.
- The six-versus-four hidden negative stationary margin and the reflected q=.9 output-bias gauge-positive margin for the one-image H=1 case.
- The B16 intermittent-sensitivity overlap `(1-beta1)(1-beta2)(beta1 beta2)^(2H-1)`, effective epsilon `16epsilon`, and positive-RGB perturbation bound `9*(51/80)*delta/epsilon` at delta=1e-16.
- The first-task-batch conditional probability min(P(L=1),P(L=2))=P(L=2), and the exact nonvanishing-jump coefficient.

All sign comparisons use Fraction arithmetic; floats are display-only. The limitations in the saved output explicitly separate frozen stationary fields from moving-CNN experiments and a target hidden coordinate from the full filter mean. I read this verifier but did not invoke its file-writing main routine.

## 6. Final scope

Under the listed ordinary-Adam, fixed-positive-rate, no-decay/no-projection, freely trained output-bias, finite-dataset, endlessly renewed random-task assumptions, finite convergence of every raw parameter and permanent compact containment cannot hold. The document does not overextend this to hidden sinking itself. Its finite constant-rate alternative and its stationary-level alternatives are logically compatible with the impossibility results.
