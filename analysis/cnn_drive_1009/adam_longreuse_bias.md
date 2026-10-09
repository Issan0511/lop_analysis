# Binary batch-16 bias response with N=1200 and 400 reused-label epochs

2026-10-09. This extends the stationary output-bias sign certificate to the requested long task: \(N=1200\), batch size \(B=16\), \(M=N/B=75\) batches per epoch, \(E=400\) independently reshuffled epochs, and \(H=30000\) updates per task. Labels are drawn once per task and reused through every reshuffle. Adam's first and second moments retain the entire past.

**Result.** For the standard \(\beta_1=.9,\beta_2=.999,\epsilon=10^{-8}\), every stationary phase at uniform binary logits has common-output response derivative

\[
 \Gamma_{\rm phase}>\frac{209}{100}=2.09.
\]

The task-summed output-bias contribution to the normal matrix is consequently

\[
 J_bL_b=\beta_b\,11^T,\qquad
 \beta_b=\frac1N\sum_{r=1}^{H}\Gamma_r
 >\frac{209}{4}=52.25.
\]

The principal proof uses an exhaustive integer certificate over the **1201 possible total positive-label counts**, followed by rational inequalities. It does not depend on an external ultra-logconcavity theorem, approximate probability computation, independent-batch surrogate, training run, or Monte Carlo estimate.

Artifacts:

- verify_adam_longreuse_bias.py: standalone standard-library verifier.
- ../../results/cnn_drive_1009/adam_longreuse_bias.json: successful certificate report, including hashes/bitlengths rather than the large coefficient arrays.

This proves a rank-one stationary bias block at uniform binary outputs. It does not by itself establish the other normal directions, actual moving-CNN convergence, ten-class behavior, or the complete RL-CIFAR trajectory.

## 1. Exact scalar response and the previous large-H obstruction

At each task, the \(1200\) labels are independent uniform signs. Each epoch independently draws a uniform permutation and partitions it into \(75\) consecutive batches of \(16\) distinct images. Write the signed batch mean as

\[
 X_t=\frac1{16}\sum_{n\in I_t}Y_{k(t),n}.
\]

For a uniform binary half-contrast \(z_n=z\), put \(\mu=\tanh z\). The raw output-bias gradients are \(\pm(\mu-X_t)/2\). With normalized stationary weights

\[
 a_\ell=(1-\beta_1)\beta_1^\ell,\qquad
 b_\ell=(1-\beta_2)\beta_2^\ell,
\]
\[
 A=\sum_{\ell\ge0}a_\ell X_{t-\ell},\quad
 B_0=\sum_{\ell\ge0}b_\ell X_{t-\ell},\quad
 V=\sum_{\ell\ge0}b_\ell X_{t-\ell}^2,\quad
 \sigma=\sqrt V,\quad e=2\epsilon,
\]

the exact plus-bias response is

\[
 q_+(\mu)=
 \frac{\mu-A}{\sqrt{\sum_\ell b_\ell(\mu-X_{t-\ell})^2}+e},
 \qquad q_-(\mu)=-q_+(\mu).
\]

At zero contrast, \(\mu'(0)=1\), and

\[
 \Gamma_r=
 \mathbb E\left[
 \frac1{\sigma+e}
 -\frac{AB_0}{\sigma(\sigma+e)^2}
 \right].
 \tag{1.1}
\]

The zero-RMS convention and differentiation justification are the same as in the short-task bias note: stationary weighted Cauchy--Schwarz gives

\[
 |A|\le K_0\sigma,\qquad |B_0|\le\sigma,\qquad
 K_0^2=\frac{(1-\beta_1)^2}
 {(1-\beta_2)(1-\beta_1^2/\beta_2)}
 =\frac{370}{7}<64.
 \tag{1.2}
\]

The scalar derivative is bounded by \(1/e+K_0/(4e)\), including the derivative \(1/e\) at an all-zero history. Hence the expectation derivative is justified without a deterministic RMS floor.

The earlier task-weight bound treated all \(30000\) within-task gradients as potentially maximally correlated. Its squared-group-weight factor approaches one at this H, losing both the useful variance estimate and the lower-RMS concentration. That is a limitation of that bound, not evidence of a sign reversal. The following proof uses the actual epoch structure.

## 2. Exact pair covariance under fresh epoch reshuffling

The unconditional batch variance is \(r=1/B=1/16\). For two distinct optimizer positions:

1. In the same epoch, the batches are disjoint, so their signed means have covariance zero.
2. In different epochs of the same task, the two permutations are independent conditional on the task labels. Both conditional means equal the population label mean \(\bar Y\); therefore their covariance is \(\mathbb E\bar Y^2=1/N\).
3. In different tasks, the labels and schedules are independent, so the covariance is zero.

In particular every off-diagonal covariance is at most \(1/N\). For any normalized nonnegative weights \(w_\ell\),

\[
 \mathbb E\left(\sum_\ell w_\ell X_{t-\ell}\right)^2
 \le
 \left(\frac1B-\frac1N\right)\sum_\ell w_\ell^2+\frac1N.
 \tag{2.1}
\]

Applying the exact squared-geometric-weight sums gives phase-uniform bounds

\[
 V_A:=
 \left(\frac1B-\frac1N\right)\frac{1-\beta_1}{1+\beta_1}
 +\frac1N
 =0.004078947368\ldots,
\]
\[
 V_{B_0}:=
 \left(\frac1B-\frac1N\right)\frac{1-\beta_2}{1+\beta_2}
 +\frac1N
 =0.000864182091\ldots.
 \tag{2.2}
\]

These bounds do not increase with E. A rational square comparison in the verifier proves

\[
 V_AV_{B_0}<\left(\frac{19}{10000}\right)^2,
 \qquad
 \mathbb E|AB_0|<\frac{19}{10000}.
 \tag{2.3}
\]

The \(1/N\) term expressly retains the dependence created by reused task labels. Replacing it by zero would be an incorrect iid-gradient argument.

## 3. The exact conditional epoch generating function

Fix all labels of a task, and let \(K\in\{0,\ldots,1200\}\) be their positive count. In one new uniform epoch permutation, let \(k_i\) be the positive count in batch i. Then

\[
 \Pr(k_1,\ldots,k_{75}\mid K)
 =\frac{\prod_{i=1}^{75}\binom{16}{k_i}}{\binom{1200}{K}},
 \qquad \sum_i k_i=K.
 \tag{3.1}
\]

The squared means have the integer representation

\[
 X_i^2=\frac{(8-k_i)^2}{64}.
\]

Use \(q=7/8\) and define

\[
 Q_{\rm epoch}=\sum_{i=1}^{75}(8-k_i)^2,\qquad
 P(x)=\sum_{k=0}^{16}\binom{16}{k}q^{(8-k)^2}x^k.
\]

For coefficients \(c_K=[x^K]P(x)^{75}\), the exact conditional generating function is

\[
 f_K:=\mathbb E[q^{Q_{\rm epoch}}\mid\text{fixed task labels}]
 =\frac{c_K}{\binom{1200}{K}}.
 \tag{3.2}
\]

It depends only on K, not on which images carry the positive labels.

### 3.1 A finite integer certificate for the worst total label count

Remove the common denominator \(8^{64}\) from P. Its integer coefficients are

\[
 a_k=\binom{16}{k}7^{(8-k)^2}8^{64-(8-k)^2}.
 \tag{3.3}
\]

Let \(C_K=[x^K](\sum_{k=0}^{16}a_kx^k)^{75}\). The standalone verifier computes them by repeated integer convolution and checks **all 1201 inequalities**

\[
 C_K\binom{1200}{600}
 \le C_{600}\binom{1200}{K},
 \qquad 0\le K\le1200.
 \tag{3.4}
\]

Every comparison passes, with equality only at \(K=600\). This proves directly that \(f_K\le f_{600}\), without invoking a general sequence theorem. The largest coefficient has 15556 bits. The JSON report records a SHA-256 digest of the ordered length-prefixed unsigned big-endian coefficients, their bitlength, and the comparison count; the verifier reconstructs the actual integers.

Put

\[
 \mu_q=2^{-16}\sum_{k=0}^{16}\binom{16}{k}q^{(8-k)^2},
 \qquad
 p_c=2^{-1200}\binom{1200}{600}.
\]

Since \(c_{600}\le P(1)^{75}\), equations (3.2)–(3.4) imply the useful uniform bound

\[
 \boxed{\quad
 f_K\le f_{600}\le\frac{\mu_q^{75}}{p_c}
 \quad\text{for every fixed task label assignment.}\quad}
 \tag{3.5}
\]

Display values are \(\mu_q=0.692388321068\ldots\) and
\(p_c=0.0230281452686\ldots\); the certificate uses their exact rational values.

## 4. Thirteen completed epochs control the RMS at every phase

At any current optimizer phase, take the thirteen most recent **completed** epochs. Ignore the current partial epoch. A completed epoch has 75 updates. Every included batch has lag less than \(75(13+1)=1050\), even at the last phase of the current epoch. Thus, uniformly at every task/epoch phase,

\[
 V\ge w\sum_{\text{these 13 epochs}}\sum_{\text{batches}}X_t^2
 =\frac{w}{64}Q,
 \qquad
 w=(1-\beta_2)\beta_2^{1050}.
 \tag{4.1}
\]

The selected epochs can cross a task boundary. Conditional on every task label vector involved, their fresh permutations are still independent. Their fixed conditional label counts need not be equal. The uniform bound (3.5) therefore gives

\[
 \mathbb E[q^Q\mid\text{all relevant task labels}]
 \le\left(\frac{\mu_q^{75}}{p_c}\right)^{13}.
 \tag{4.2}
\]

This is the precise conditional-independence step. No labels are redrawn for this argument, and no independence of the batches within an epoch is asserted.

Take the variance threshold \(\ell=1/100\). The exact rational arithmetic gives

\[
 \left\lceil\frac{64\ell}{w}\right\rceil=1830.
 \tag{4.3}
\]

If \(V<\ell\), then \(Q\le1830\); this rounded bound is slightly conservative. Since \(0<q<1\), Markov's inequality applied to \(q^Q\), followed by (4.2), proves

\[
 \Pr(V<1/100)
 \le
 \frac{\mu_q^{975}}{p_c^{13}q^{1830}}
 <2^{-80}.
 \tag{4.4}
\]

The last inequality is checked by exact Fraction comparison. Its display logarithm is approximately \(-65.03217\), but neither that logarithm nor any floating-point value is used to prove the inequality.

Only 975 past full-epoch batches were used; all other nonnegative terms in the stationary second moment were dropped. The argument is uniform over the 30000 phases, including task boundaries.

This is an infinite-history stationary bound. It is not a lower-tail assertion for Adam's first few updates after zero initialization, before thirteen epochs exist. Any transfer to that finite-initialized moving process must keep the separate initialization/tracking error from the averaging theorem.

## 5. Strict positive common-mode derivative

The unconditional marginal variance remains \(1/16\), so

\[
 \mathbb E\frac1{\sigma+2\epsilon}
 \ge\frac1{\mathbb E\sigma+2\epsilon}
 \ge\frac1{1/4+2\epsilon}.
 \tag{5.1}
\]

On \(V\ge1/100\), the denominator in the correction satisfies
\(\sigma(\sigma+2\epsilon)^2\ge(1/10)^3=1/1000\).
Equation (2.3) consequently bounds that part of its absolute expectation by \(19/10\).

On the remaining histories, (1.2) gives

\[
 \frac{|AB_0|}{\sigma(\sigma+2\epsilon)^2}
 \le
 \frac{K_0\sigma}{(\sigma+2\epsilon)^2}
 \le\frac{K_0}{8\epsilon}<\frac1\epsilon.
 \tag{5.2}
\]

Using (4.4) in the exact response (1.1),

\[
 \boxed{
 \Gamma_r>
 \frac1{1/4+2\epsilon}
 -\frac{19}{10}
 -\frac{2^{-80}}{\epsilon}
 >\frac{209}{100}.
 }\tag{5.3}
\]

All comparisons in the last step are rational at \(\epsilon=10^{-8}\). The intermediate lower bound is approximately \(2.09999968\). This proves a positive expectation at each stationary phase, not merely a favorable phase sum.

## 6. Output-bias normal block and conventions

The shuffled schedule law is invariant under permutations of the 1200 image identities. At zero logits, the output-bias frozen field depends on the images only through their independent formal contrasts. Its response rows therefore have equal entries across image indices. With a field summed over the H task phases,

\[
 L_{b_+,n}=\frac1N\sum_{r=1}^{H}\Gamma_r,\qquad
 L_{b_-,n}=-\frac1N\sum_{r=1}^{H}\Gamma_r.
\]

The half-contrast Jacobian has constant bias columns \(+1/2\) and \(-1/2\). Hence

\[
 J_bL_b
 =\left(\frac1N\sum_{r=1}^{H}\Gamma_r\right)11^T
 =\beta_b11^T,\qquad
 \beta_b>\frac{30000}{1200}\frac{209}{100}
 =\frac{209}{4}.
 \tag{6.1}
\]

There is no additional factor of two: the normalization \(2\epsilon\) already scaled out the raw bias-gradient factor \(1/2\), and the two Jacobian-column products each contribute half of the displayed coefficient. If a phase-averaged field is used instead, divide L and \(\beta_b\) by H.

## 7. Reproduction and exact-arithmetic boundary

Run, using any recent Python with the standard library:

    python verify_adam_longreuse_bias.py

An optional output path can be supplied with --output. The default report is `results/cnn_drive_1009/adam_longreuse_bias.json` in the repository.

The executed certificate checks:

1. all 1201 conditional-label-count inequalities (3.4);
2. polynomial symmetry and the sum-of-coefficients identity;
3. the exact integer threshold 1830;
4. the rational tail bound (4.4);
5. the rational squared variance comparison (2.3);
6. \(K_0^2=370/7<64\);
7. \(\Gamma_r>209/100\) and \(\beta_b>209/4\).

The report contains exact small rational bounds, source/certificate hashes and large-rational bitlengths. Decimal values and elapsed time are explanatory output only. The successful run took about one second and did not access data, draw random numbers, construct an optimizer, or simulate a training history.

## 8. Optional structural explanation and remaining limits

The finite certificate (3.4) has a general structural explanation: the sequence
\(\binom{16}{k}q^{(8-k)^2}\) is ultra-logconcave of order 16, because division by the binomial coefficients leaves a logconcave sequence. The convolution-preservation theorem makes the normalized 75-fold coefficient sequence logconcave; symmetry then places its maximum at 600. One primary-source statement is Theorem 1.1 of [Gurvits, a proof of Liggett's convolution theorem](https://arxiv.org/pdf/0804.1181). This theorem is not a premise of the verified finite proof: (3.4) was checked directly for every required index.

The variance and completed-epoch arguments actually do not depend on E after independent fresh epoch permutations are stipulated. They therefore also work for any other fixed finite E at the same N and B. The stated target remains E=400; no improvement to constant-rate moving-CNN theory follows from this observation.

This result closes the large-reuse obstruction for the **binary uniform-output bias block**. It leaves separate obligations for the complete CNN normal matrix, full/self capacity signs, a realizable high-dimensional reference, local invariant neighborhoods, stochastic averaging with sufficiently small rates, and the original ten-class experiment. A stationary expectation over the retained history is not a fresh-label conditional expectation at every realized optimizer step.

## 9. Whole-task inverse-RMS certificate for the 1200-image geometry

The repository verifier additionally checks, with exact integers and fractions, `r=binom(16,8)^75/binom(1200,600)<(.999)^300`. Since `p_N<1`, this gives `p_N[r beta2^(-75*8/2)]^E<1` for every finite E, including E400. Thus the whole-task inverse-eighth-moment criterion in [the regularity proof](adam_b16_regularity.md) applies. This is a separate regularity certificate from the bias lower-tail certificate above.
