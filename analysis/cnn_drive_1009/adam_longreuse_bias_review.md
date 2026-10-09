# Independent audit: N1200/B16/E400 stationary binary bias certificate

2026-10-09. Reviewed `adam_longreuse_bias.md`, `verify_adam_longreuse_bias.py`, and its saved JSON certificate. No repository files or report artifacts were changed. No training experiment was performed.

**Verdict: PASS.** The proof establishes the stated strict stationary phase-uniform bias response `Gamma>209/100` and task-summed common-mode coefficient `beta_b>209/4`. The conditioning across task boundaries, completed-epoch lag bound, exact integer certificate, and epsilon/bias factors are valid.

I additionally replayed the standalone integer/Fraction verifier in memory without calling its file-writing entry point. All assertions passed in under one second. Its source hash and the reconstructed polynomial certificate, integer threshold, probability bound, Gamma lower bound, and normal-coefficient lower bound agree exactly with the saved report. This is an arithmetic proof replay, not a stochastic simulation.

## 1. Exact response and epoch covariance

The normalization of raw output-bias gradients +/-1/2 correctly changes the quotient epsilon to `2 epsilon`. Thus

    Gamma=E[1/(sigma+2epsilon)
        -A B0/{sigma(sigma+2epsilon)^2}]

has the stated sign and factors. The stationary bound `K0^2=370/7<64` and the derivative convention at an all-zero history justify differentiating the common-contrast response without a deterministic RMS floor.

For different optimizer positions, the covariance calculation preserves the reused-label dependence:

- Same-epoch batches use disjoint independent labels, giving unconditional covariance zero.
- Different epochs of one task have independent permutations conditional on that task's fixed label vector. Both conditional means equal its population label mean, so the unconditional covariance is exactly 1/N.
- Different tasks have independent labels and schedules, giving covariance zero.

Therefore all off-diagonal covariances are at most 1/N. Nonnegative normalized EMA weights give exactly the upper bound

    (1/B-1/N) sum_l w_l^2+1/N.

The 1/N term has not been omitted. The squared geometric-weight sums and the rational comparison `V_A V_B0<(19/10000)^2` are correct and do not deteriorate with the number of epochs per task.

## 2. Conditional epoch generating function and integer comparison

Conditional on any particular task-label assignment with K positives, the positive positions of a new uniformly permuted epoch form a uniform K-subset. Consequently the probability of ordered batch counts k_i is `prod_i binom(16,k_i)/binom(1200,K)` with sum k_i=K. This proves the stated coefficient formula without assuming independent batches within the epoch.

The signed batch mean has square `(8-k_i)^2/64`, so the integer polynomial coefficients

    a_k=binom(16,k)7^((8-k)^2)8^(64-(8-k)^2)

are precisely the coefficients of the desired q=7/8 generating polynomial after clearing one common denominator 8^64. Repeated integer convolution computes its 75th power. Comparing

    C_K binom(1200,600)<=C_600 binom(1200,K)

correctly cancels the common polynomial denominator and verifies the maximal conditional generating function over all 1201 possible label counts. The code checks every index; the report records equality only at K=600. Polynomial symmetry and the sum-of-coefficients identity provide additional internal checks.

The subsequent upper bound is also correctly normalized:

    f_K<=f_600<=P(1)^75/binom(1200,600)
        =mu_q^75/p_center.

No external ultra-logconcavity result is necessary. I did not rely on the optional external theorem in the explanatory section; the finite integer certificate directly covers every required count.

## 3. Thirteen completed epochs and task-boundary conditioning

For full phase uniformity, interpret the selected epochs as the thirteen preceding complete epochs, excluding the epoch containing the current optimizer position even at its last step. If the current within-epoch phase is r in {1,...,75}, the oldest selected batch has lag at most

    13*75+r-1<=1049.

Using `(1-beta2) beta2^1050` for every retained weight is therefore conservative. No task-boundary exception is present: task lengths contain an integer number of epochs, and selection is by deterministic time position, not by observed labels, gradients, or favorable epochs.

Condition on all label vectors of the tasks intersected by the selected epochs. Every selected epoch still has an independent fresh uniform permutation under this conditioning, including epochs in the same task and epochs on opposite sides of a task boundary. The conditional counts K may differ across tasks, but the generating-function upper bound is uniform in K. Therefore the product of thirteen bounds is justified exactly as written.

With Q the integer squared-count sum, `V>=w Q/64`. The exact ceiling is 1830. The implication `V<.01 => Q<=1830` is safely rounded; using the ceiling instead of the stricter integer threshold only weakens the tail bound. Because q is below one, this event implies `q^Q>=q^1830`. Markov's inequality therefore has the stated direction and produces

    P(V<.01)<=mu_q^975/[p_center^13 q^1830]<2^(-80).

All powers and comparisons used for this conclusion are integer/Fraction operations. Display logarithms do not enter the proof. The result is stationary and does not assert that thirteen prior epochs exist during the first updates after finite initialization.

## 4. Positive phase response and normal coefficient

The positive term is at least `1/(1/4+2epsilon)` because the unconditional squared batch mean remains 1/16. On V>=.01 the correction denominator is at least .001, so the unconditional absolute numerator bound yields a contribution below 1.9. The event need not be independent of A or B0.

On the bad event, weighted Cauchy--Schwarz gives the deterministic bound

    |A B0|/[sigma(sigma+2epsilon)^2]
       <=K0/(8epsilon)<1/epsilon.

Combining these bounds with the exact tail certificate gives the claimed rational lower bound, exceeding 209/100, at every stationary phase. It is not merely a positive phase average.

Exchangeability across the 1200 image identities makes both output-bias response rows constant across image coordinates. Multiplying by the raw half-contrast Jacobian columns +/-1/2 gives

    beta_b=(1/N)sum_(r=1)^H Gamma_r
           >(30000/1200)(209/100)=209/4.

The two raw columns contribute two halves. There is no missing factor of two, batch mean factor, or phase normalization.

## 5. Code and exact-arithmetic boundary

The verifier uses exact Python integers for all polynomial coefficients and cross-products, exact Fraction arithmetic for the PGF/RMS/variance/final sign comparisons, and integer ceiling arithmetic for the threshold. Conversion to floats, logarithms, hashes, and elapsed time occurs only in descriptive output. The saved source hash agrees with the reviewed source. The successful in-memory replay reconstructed the same decisive certificate fields.

The script is explicitly for the fixed N1200/B16 constants; hardcoded 8 and 64 in the count polynomial correctly implement this scope. It does not claim to be a generic verifier for arbitrary batch sizes.

## 6. Remaining scope

This closes the large-reuse difficulty for the binary uniform-output stationary output-bias block under fresh independent epoch reshuffling. It does not establish full normal rank/stability, a native N1200 reference, capacity-self signs, ten-class equilibrium, an actual invariant training region, or constant-rate convergence. The note explicitly retains these obligations and the finite-initialization tracking requirement. No such result is inferred from the arithmetic certificate.
