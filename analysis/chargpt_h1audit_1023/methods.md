# What the fixed-state audit can identify

All formulas use mean losses; F retains its original sum convention.
The final FFN is the target. Its input features x, output a, and all upstream
parameters stay fixed. K=x dot mean(x), with no bias term in this architecture.

## Stopping the old-task update is not decorrelating the readout

For an ordinary untied linear classifier, the readout gradient is

    grad(v_i) = mean_n [(p_n-y_n) phi_in].

Even if this gradient vanishes, it does not require

    mean_n [(y_n-u) v_i^0] = 0

or the gated, K-weighted version of that condition. A component that already
helps predict the label can remain in place when p approaches y. In the GPT
the embedding and head are tied, giving additional training gradients; that
does not turn the stopping condition into a decorrelation requirement either.

The analysis is of a switch to independent uniform new labels, whose error
expectation is p-u. Hence, even in the one-hot correct limit, the switch error
is y-u rather than zero. The exact diagnostic separates

    S(V,p-u) = S(V,y-u) + S(V,p-y).

This identity is an algebraic statement at the same saved state. Replacing the
error by y-u does not mean that a network with better training was constructed.

The training implementation uses a fixed number of Adam updates per task
(`run_task`, lines 530 onward in the vendored source), with no correctness-based
early stop. The task-M loss excludes the first position because it lacks two
context characters. The inherited analysis probe includes all 128 positions;
the true-conditional control explicitly uses the stationary conditional at
that first position. Its probe KL is therefore not the exact training-loss KL.

## Which residual corresponds most closely to the question

Write T=Gamma P U and G=T^T T. Before the final LN variance derivative,
the effective unit readout is V^c T_i / sigma_n. Its class vector decomposes as

    V^c T_i = kappa G_ii B_i + R_i,
    R_i = kappa sum_{j!=i} B_j G_ji + kappa F_r T_i + V^perp T_i,
    B_i = sum_m (y_m-u) a_im/sigma_m.

Thus `oracle_unitrest_direct` is precisely the empirical old-label inner
product of this R_i, weighted by K_n gate_ni/sigma_n. It measures what
remains after the own-unit loading is removed, including other units and
the head's projection residual. It is not just the matrix V^perp.

R_i is the closest direct-path counterpart here to H1's v_i^0. The identification
is still not exact: the effective feature is a_i/sigma and the exact derivative
includes the LN radial term. Report these qualifications alongside the result.

## Input-pair diagonal and unit diagonal are distinct

For oracle errors xi_n=y_n-u,

    S_load,i = kappa/N sum_{m,n} K_n gate_ni (xi_m dot xi_n) h_m^T J_n U_i.

The input diagonal is m=n. It has xi_n dot xi_n=1-1/C, independently of the
realized labels. Splitting F into own FFN unit / other FFN units / residual
splits h_m in this equation; it does not remove the m!=n terms.

With fixed h,J,K,gate,kappa and independently sampled uniform labels,
the ensemble expectation is the input diagonal D. Actual learned features
are not independent of training labels, and this stochastic Markov task
does not supply IID per-input labels. The calculation of D therefore tests
the stated frozen-feature model, not the true ensemble of trained networks.

In particular, the direct own-unit input diagonal is

    kappa (1-1/C) G_ii mean_n[K_n gate_ni a_ni/sigma_n^2].

It is nonnegative for ReLU and K>=0. If the observed total differs, the
identity phi*phi'=z_+ has not failed. The input off-diagonal, other-unit,
residual, fit-error, or LN terms cannot all be discarded.

## Interpretation constraints

- The true conditional label distribution is nondegenerate, even in early tasks.
  Accuracy less than one is not itself evidence of failed learning.
- Half of unit pushes being positive is not a test of independence or randomness.
- A large non-diagonal term can reinforce the positive direction early and oppose
  it late; an approximation can predict the sign without approximating the magnitude.
- Changing input dimension is an architectural intervention. Saved-state ranks and
  column overlaps describe geometry but do not measure that intervention's effect.
- The decomposition does not identify which training update produced the geometry.
