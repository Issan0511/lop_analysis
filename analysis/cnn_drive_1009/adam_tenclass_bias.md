# Ten-class B16 output-bias response: a positive predictive common-image block

2026-10-09. Independent derivation and exact rational certificate for C=10 uniform iid image labels, N=1200, B=16, independently reshuffled epochs, retained task labels, ordinary Adam beta1=.9, beta2=.999, epsilon=1e-8. The response bounds hold at every stationary phase for any finite number of epochs per task. The requested E=400 gives H=30000 updates and the final coefficient stated below.

**Result.** At uniform predictions, the output-bias common-probability derivative satisfies `gamma_r>119/50=2.38` at every phase. In centered full-logit coordinates, the task-summed bias contribution to the predictive normal matrix is

    M_bias=beta_bias 1_N1_N^T tensor P_C,
    P_C=I_C-(1/C)1_C1_C^T,
    beta_bias=(1/(NC))sum_(r=1)^H gamma_r.

For N1200/C10/H30000, `beta_bias>119/20=5.95`. This matrix is positive semidefinite and strictly positive on the common-image class-contrast subspace; it has rank C-1, not N(C-1). Hidden/head equilibrium and the remaining normal directions are separate obligations.

## 1. Exact multiclass bias gradient and a deterministic noise floor

Fix a class c and let p0=1/C=1/10. A batch has count K_(t,c) in that class. Define its centered class-frequency noise

    X_t=K_(t,c)/16-p0.

When this class has the same probability p_c=p0+mu on every image, its raw output-bias gradient is exactly

    g_t=mu-X_t.

Unlike the normalized binary half-contrast formula, this raw multiclass bias formula has no factor 1/2, so the Adam denominator below uses epsilon, not 2epsilon.

Let a_l=(1-beta1)beta1^l and b_l=(1-beta2)beta2^l, and define the stationary phase quantities

    A=sum_l a_l X_(t-l),
    B0=sum_l b_l X_(t-l),
    V=sum_l b_l X_(t-l)^2, sigma=sqrt(V).

Then the raw stationary bias quotient is

    q_c(mu)=(mu-A)/[sqrt(V-2mu B0+mu^2)+epsilon].

At mu=0, every possible batch satisfies

    |X_t|>=min_(k=0,...,16)|k/16-1/10|=1/40.       (1.1)

The closest count is k=2, giving 1/8-1/10=1/40. Because the EMA weights sum to one,

    sigma>=1/40                                   (1.2)

for every history. This deterministic floor is specific to the noninteger expected count B/C=1.6. It is not a claim about general multiclass hidden coordinates.

For |mu|<=1/80, every |mu-X_t| remains at least 1/80. More generally, the same floor holds when each image's class probability differs from p0 by at most 1/80. Thus the bias response is locally smooth in the independent image probabilities, and differentiation under expectation is justified by deterministic derivative bounds on the weighted histories.

The exact derivative in the common-probability direction is therefore

    gamma_r=E[1/(sigma+epsilon)
              -A B0/{sigma(sigma+epsilon)^2}].     (1.3)

The subtraction and normalization agree with direct differentiation. A and B0 retain their correlation with each other and with V.

## 2. Reused-label epoch covariances

For the fixed class, the per-image indicators are iid Bernoulli(p0), even though the full vector of class indicators for one image is multinomial. Put

    v=p0(1-p0)=9/100,
    r=E X_t^2=v/B=9/1600=(3/40)^2.

For distinct optimizer positions, the same epoch covariance calculation as in the long-reuse binary proof gives:

- Same epoch: disjoint batches have covariance zero.
- Different epochs of one task: independent permutations conditional on that task's label vector give covariance v/N.
- Different tasks: covariance zero.

Thus any normalized nonnegative scalar history weights w obey

    E[(sum_l w_l X_l)^2]
       <=v[(1/B-1/N)sum_l w_l^2+1/N].              (2.1)

This bound keeps the within-task population-count correlation. It does not assume independent batch noises and is uniform at every epoch/task phase, irrespective of the number of reused-label epochs.

For the two geometric EMA weights, define

    U_i=(1/B-1/N)(1-beta_i)/(1+beta_i)+1/N.

Then

    E A^2<=v U_1,
    E B0^2<=v U_2,
    E|A B0|<=v sqrt(U_1 U_2).                      (2.2)

At N=1200 and the standard betas, exact rational arithmetic gives

    U_1 U_2<(19/10000)^2,
    v U_1=279/760000,
    v U_2=6219/79960000,
    E|A B0|<171/1000000.                          (2.3)

Cross-class dependence need not be discarded: a coordinatewise Adam quotient uses only this class's marginal history. The cross-class softmax coupling is included separately in section 4.

## 3. Finite positive response certificate

By Jensen and the RMS moment bound,

    E[1/(sigma+epsilon)]
       >=1/(E sigma+epsilon)
       >=1/(sqrt(r)+epsilon)
       =1/(3/40+epsilon).

By the deterministic floor (1.2),

    E[|A B0|/{sigma(sigma+epsilon)^2}]
       <=40^3 E|A B0|
       <64000*(171/1000000)=1368/125=10.944.

No lower-tail event or conditional-independence argument for a ratio is needed. Substituting into (1.3) gives, at every stationary phase,

    gamma_r > 1/(3/40+epsilon)-1368/125.           (3.1)

For epsilon=1e-8, the exact rational expression is

    2239998632/937500125
       =2.389331555555793... >119/50.              (3.2)

The decimal is only explanatory. The strict certificate is a Fraction comparison.

## 4. Predictive coordinates, exchangeability, and every factor of C

Let f_n in R^C denote the raw logits and z_n=P_C f_n their centered full-logit vector. This coordinate convention differs from the scalar binary half-contrast; no binary factor is imported.

First regard all image probabilities p_(n,c) as independent formal variables. Each bias-coordinate quotient depends only on the histories of the probabilities for its own class. At uniform probabilities, class and image exchangeability imply, phase by phase,

    partial E[q_(b_c)] / partial p_(n,d)
       =delta_(c,d) gamma_r/N.                    (4.1)

The sum over n is precisely the common-probability derivative gamma_r. Uniform softmax has Jacobian

    D_(f_n) p_n=(1/C)P_C.

For the task-summed field, the bias-response derivative from the stacked centered logits to the C raw biases is consequently

    L_bias=[sum_r gamma_r/(NC)]
                 (1_N^T tensor P_C).             (4.2)

The centered-logit Jacobian of the raw output biases is

    J_bias=1_N tensor P_C.

Using P_C^2=P_C yields

    J_bias L_bias
       =[sum_r gamma_r/(NC)]1_N1_N^T tensor P_C.   (4.3)

Thus the factor 1/N comes from image exchangeability, 1/C comes from softmax, and there is no extra half-gradient factor. If the field is phase-averaged, divide L_bias and the displayed coefficient by H.

For E=400, H=30000, N=1200, C=10,

    beta_bias > (H/(NC))*(119/50)=119/20=5.95.

The sharper rational lower bound for this coefficient obtained from (3.2) is

    1119999316/187500025=5.9733288888894815....

The matrix is `[(sum_r gamma_r)/C]` times the orthogonal projection onto vectors constant across images and centered across classes. Its only nonzero eigenvalue is `(sum_r gamma_r)/C`, with multiplicity C-1. The other image-contrast directions need the remaining parameter blocks.

## 5. Baseline gauge motion versus predictive response

At uniform predictions, class permutation symmetry makes the expected bias field a common-class vector

    Fbar_bias=kappa 1_C.

The present proof does not need kappa=0 and does not claim a strict nonzero value for this particular N1200/B16/H30000 configuration. Nonzero common-class Adam drift is possible, as the separate ten-class stationary counterexample shows. In all cases,

    (1_N tensor P_C)Fbar_bias=0.

Thus the bias contribution itself is at a predictive equilibrium, and its normal derivative (4.3) is restoring on the common-image class-contrast subspace. Its raw gauge component may still move.

This does not establish equilibrium of the full CNN. Free head-weight and hidden expected fields must be included. In particular, the ten-class hidden counterexample in the companion note shows that uniform predictions alone do not imply `Fbar_hidden=0`, and the hidden contribution can generate a nonzero predictive drift that no gauge quotient removes. A positive bias block cannot repair a missing full-field equilibrium identity by itself.

## 6. Reproducible exact Fraction certificate

The following standard-library computation was executed successfully. It does not sample labels, simulate Adam, or train a network.

```python
from fractions import Fraction as F

b1, b2, eps = F(9, 10), F(999, 1000), F(1, 10**8)
N, B, C, H = 1200, 16, 10, 30000
p0 = F(1, C)
v = p0 * (1 - p0)
U1 = (F(1, B) - F(1, N)) * (1 - b1) / (1 + b1) + F(1, N)
U2 = (F(1, B) - F(1, N)) * (1 - b2) / (1 + b2) + F(1, N)
assert U1 * U2 < F(19, 10000)**2
assert v * U1 == F(279, 760000)
assert v * U2 == F(6219, 79960000)
assert v / B == F(3, 40)**2
floor = min(abs(F(k, B) - p0) for k in range(B + 1))
assert floor == F(1, 40)
cross_upper = v * F(19, 10000)
assert cross_upper == F(171, 1000000)
correction_upper = cross_upper / floor**3
assert correction_upper == F(1368, 125)
gamma_lower = 1 / (F(3, 40) + eps) - correction_upper
assert gamma_lower == F(2239998632, 937500125)
assert gamma_lower > F(119, 50)
bias_coefficient_lower = F(H, N * C) * gamma_lower
assert bias_coefficient_lower == F(1119999316, 187500025)
assert bias_coefficient_lower > F(119, 20)
```

## 7. Scope

The proof is uniform over stationary phases and any finite E at N1200/B16 with independent fresh epoch permutations, iid uniform ten-class image labels, and taskwise reuse. The target E=400 is included. Ordinary finite-initialized Adam and actual moving parameters still require the separate initialization/tracking/averaging argument.

Established here: a deterministic bias RMS floor, a strictly positive multiclass common-probability response, and the exact positive semidefinite predictive bias block with its normalization. Not established here: full hidden/head equilibrium, full normal stability, a complete multiclass invariant tube, or convergence of all raw parameters including gauge modes.

Repository reproduction: [verify_adam_tenclass.py](verify_adam_tenclass.py), [exact certificate](../../results/cnn_drive_1009/adam_tenclass.json).
