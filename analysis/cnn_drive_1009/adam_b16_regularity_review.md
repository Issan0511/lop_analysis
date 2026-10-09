# Independent review: batch-16 stationary Adam regularity

2026-10-09. Reviewed adam_b16_regularity.md, with particular attention to the compact-uniform event, all-raw simultaneous RMS bounds, expectation derivatives, and the distinction from finite realized optimizer updates.

**Result: PASS. No blocking issue or required correction found.**

## 1. Uniform all-raw gradient event

The relative sensitivity condition is sufficient, including negative reference sensitivities. Expanding the batch gradient around \(c_j\) gives exactly

\[
 |g_j+c_j\bar Y_I|
 \le |c_j|\{e_j+(1+e_j)\mu\}.
\]

For an unbalanced batch of 16 signs, \(|\bar Y_I|\ge2/16\). The permitted error is at most \(|c_j|/16\), so every raw coordinate obeys \(|g_j|\ge|c_j|/16\) on the same event. This bound holds for all parameters in the compact neighborhood at once, because the hypotheses were imposed uniformly on that neighborhood. No coordinate union bound is necessary.

The example \(e_j,\mu\le1/(4B)\) satisfies the required inequality. Nonzero but very small \(c_j\) only enlarges the later constants; it does not invalidate the event. The proof explicitly requires a slightly larger neighborhood for differentiating at boundary points of the chosen compact set.

## 2. Exact whole-task probability with true reused labels

If every batch of an epoch is balanced, the task's complete label set is globally balanced. Conditional on any fixed globally balanced assignment, the positive-label positions in a uniform permutation are a uniform \(N/2\)-subset. Exactly \(\binom{16}{8}^{N/16}\) such subsets balance all batches. Thus the single-epoch conditional probability is

\[
 r=\frac{\binom{16}{8}^{N/16}}{\binom{N}{N/2}}.
\]

Independent epoch permutations remain independent conditional on that fixed label assignment. Consequently the exact probability that all E epochs fail to provide an unbalanced batch is \(p_Nr^E\). This does not require new labels within the task, independent minibatches within an epoch, or independent label assignments across epochs.

The event is only a sufficient nonzero-gradient event, as the note correctly states: unequal sensitivities can make some count-balanced batches nonzero as well.

I independently checked the stated exact fractions for \(N=32\) and \(N=48\) using integer binomial coefficients and Fraction arithmetic. Both match. For \(E=2\), the displayed approximate failure probabilities also match.

## 3. Lag, geometric waiting time, and inverse moments

For the T-th completed past task and its phase a, the lag from current phase r is \(TH+r-a\). The maximal possible lag is \(TH+r-1\); since \(0<\beta_2<1\), this gives the lower RMS bound in (3.1) with the correct direction.

The same first-good-task variable T works for every coordinate and every parameter in the compact neighborhood. Tasks are iid, hence \(T\) is geometric with failure probability p. The inverse-moment calculation is precisely

\[
 \mathbb E\,\beta_2^{-THq/2}
 =\frac{(1-p)\beta_2^{-Hq/2}}
 {1-p\beta_2^{-Hq/2}},
\]

when \(p\beta_2^{-Hq/2}<1\). Multiplication by the deterministic lag and coordinate-floor factors yields the note's (3.3). This establishes an integrable **supremum over the compact neighborhood**, which is stronger than merely proving a pointwise inverse moment at each parameter separately.

For the first-batch event and \(E=2\), \(H=4\) or \(6\), I checked exactly that \(p_B\beta_2^{-4H}<1\), so q=8 is valid. For the stronger whole-task event, the comparisons

\[
 r_{32}< (999/1000)^8,\qquad
 r_{48}< (999/1000)^{12}
\]

were also checked as rational inequalities. They indeed imply q=8 for every fixed finite E, because \(H=(N/16)E\) and
\(p_N[r\beta_2^{-(N/16)q/2}]^E<1\).

This arbitrary-finite-E statement requires the independently drawn uniform epoch permutations used by the proof. The constants are not uniform as E grows; the note does not claim otherwise.

## 4. The quotient derivatives and the actual required moment order

The Hilbert-space history representation is valid. Its derivative bounds follow because the squared Hilbert coefficients sum to one. The finite set of possible smooth batch-gradient maps and uniform derivative bounds justify termwise differentiation into the Hilbert space through order three. The first-moment history uses summable positive weights and is treated similarly.

The norm-derivative bounds in (4.2) are safe multilinear operator bounds. In particular, the second derivative of the Hilbert norm costs only \(1/\sigma\), and its third derivative costs at most a constant times \(1/\sigma^2\). Applying these bounds before expanding the scalar square root avoids unnecessarily high inverse powers.

I expanded \(Q=m(\sigma+\epsilon)^{-1}\) independently:

\[
 D^3Q
 =D^3m\,f+3D^2m\,Df+3Dm\,D^2f+mD^3f,
 \qquad f=(\sigma+\epsilon)^{-1}.
\]

Substitution of the note's norm bounds gives exactly its C0, C1, and C2:

\[
 \|D^3Q\|\le C_0+C_1/\sigma+C_2/\sigma^2.
\]

Thus the inverse-second moment suffices for the C3 stationary expectation. The established inverse-eighth moment is a valid conservative reserve, not a necessary derivative order or an omitted extra assumption. Inverse-first and zeroth moments follow automatically from the second or eighth moment.

On almost every infinite history, the common good-task bound makes \(\sigma_j(\theta)>0\) throughout the entire compact neighborhood. Hence the history-wise derivatives exist and are continuous there. The uniform integrable bound from the previous section permits both differentiation under expectation and dominated continuity. This proves C3 expectation and locally bounded third derivatives as claimed. The independent \((z,S)\) product-neighborhood version obeys the same argument and gives the C2 Hadamard coefficient required by the endpoint construction.

No condition such as \(\beta_1^2<\beta_2\) is needed for this particular regularity estimate, because m and its derivatives are bounded directly. That condition remains relevant to the separate universal normalized-update bound; the note does not confuse the two uses.

## 5. Correct scope of the smoothness conclusion

The result concerns the infinite-history frozen stationary expected field. A finite realized optimizer history can have zero RMS, and its quotient need not be C3 there. The proof expressly acknowledges this and does not use finite-update smoothness in the tracking argument.

The stationary-law expectation is also not a fresh-label conditional expectation given an actual trained parameter state. The reused labels and schedule history remain within the whole-task innovations. The moving-state transfer must still use stochastic averaging and the separate geometry theorem.

The local structural assumptions are sufficient and noncircular: same-sign, quantitatively comparable nonzero sensitivities across nearby images, small contrasts, and a strict fixed ReLU/Pool routing cell. They are not asserted for arbitrary CIFAR images or arbitrary trained states. Rank, capacity signs, and normal stability remain separate obligations.

## 6. Audit actions

Only the exact combinatorial fractions and inverse-moment inequalities were evaluated, using a short in-memory Fraction check. No training, Monte Carlo, or finite-history experiment was rerun. No repository files or another agent's note were edited.
