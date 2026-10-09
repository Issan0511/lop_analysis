# Independent review: batch-16 stationary output-bias response

2026-10-09. Reviewed `adam_b16_bias.md` and the general relative-response estimate in `adam_b16_reference.md`. The principal equations were independently derived before reading the completed bias note. No repository files were changed and no CNN experiments were rerun.

**Verdict: PASS.** I found no missing within-task covariance, incorrect phase normalization, tail constant, or output-bias factor. The positivity certificate is valid for H=4 or H=6 at the specified standard Adam constants. It does not by itself prove the remaining normal directions or the moving-network theorem.

## 1. Common-mode derivative and its interchange

Scaling the raw bias gradient by 1/2 changes the ordinary epsilon to `e=2 epsilon`, giving the exact normalized quotient

    q_+(mu)=(mu-A)/[sqrt(V-2mu B0+mu^2)+e].

At mu=0 its derivative is

    Gamma=E[1/(sigma+e)-A B0/{sigma(sigma+e)^2}].

The correction has a minus sign. Both A and B0 use the full, correlated batch history. This is not a quotient of expectations or an independent-label-per-step substitution.

The weighted Cauchy--Schwarz constant is exactly

    K0^2=(1-beta1)^2/[(1-beta2)(1-beta1^2/beta2)]=370/7.

It gives `|mu-A|<=K0 sigma_mu`. Since the RMS of a common shift is 1-Lipschitz, every existing derivative obeys `|q_+'|<=1/e+K0/(4e)`. At a zero-RMS history the local scalar quotient is `h/(|h|+e)`, with derivative 1/e at zero. Consequently bounded difference quotients justify differentiation under expectation. No positive deterministic RMS floor has been inserted.

## 2. Task correlations and phase weights

For a batch of 16 distinct iid image labels,

    E X^2=1/16,
    E X^4=(3*16-2)/16^3=46/4096.

These are one-batch marginal facts, which remain true conditional on a label-independent schedule. The note does not assume independence of batches within a reused-label task.

For EMA beta and current phase r, the squared task-weight sum is

    (1-beta^r)^2+beta^(2r) T,
    T=(1-beta^H)/(1+beta^H).

It is at most T: viewed as a convex quadratic in x=beta^r, it equals T at both endpoints x=beta^H and x=1. This includes the current partial task; no omitted boundary term is present.

Within each task, the L2 triangle inequality bounds a weighted batch-noise sum by `sqrt(E X^2)` times the total task weight, regardless of its internal correlations. Independent centered task contributions therefore give

    E A^2<=r2 T1,
    E B0^2<=r2 T2,
    E|A B0|<=r2 sqrt(T1 T2).

The same argument applied to the nonnegative squared noises gives `E V_k^2<=r4 W_k^2`. Independence is used only when combining different tasks.

## 3. Exact lower-tail constant and positive response

The elementary nonnegative-variable inequality `exp(-x)<=1-x+x^2/2`, multiplied over independent tasks, yields

    E exp(-t V)<=exp[-t r2+(t^2/2) r4 T2].

The infinite-history limit is legitimate by bounded convergence of the exponentials. Optimizing at the threshold ell=r2/4 gives the exponent

    (r2-ell)^2/(2 r4 T2)
       =9 r2^2/(32 r4 T2)=9/(92 T2).

Thus the displayed `P(V<1/64)` bound has the correct coefficient.

Jensen supplies the positive term `E[1/(sigma+e)]>=1/(1/4+e)`. On the good event, dropping e in the denominator and using the preceding L2 bound gives exactly `32 sqrt(T1 T2)`. On the bad event, `|A|<=K0 sigma` and `|B0|<=sigma` give

    |A B0|/[sigma(sigma+e)^2]<=K0/(4e)=K0/(8 epsilon).

The event does not need to be independent of A or B0; the good contribution uses an unconditional upper bound, and the bad contribution uses a deterministic pointwise bound.

The rational bounds in section 6 safely imply Gamma>2.9 at H=4 and H=6. In particular the intentionally loose exponential replacement by powers of 2 still leaves positive margins. The lower bound is a sufficient finite-H certificate; the note correctly does not extrapolate its positivity to arbitrary large H.

## 4. Output-bias matrix factor

Image exchangeability makes each bias-response row constant across the N image coordinates. The common-output derivative is the sum of that row, hence

    L_(b+,n)=sum_r Gamma_r/N,
    L_(b-,n)=-sum_r Gamma_r/N.

Multiplication by the independent raw Jacobian columns +1/2 and -1/2 adds the two half-contributions, giving

    J_bias L_bias=[sum_r Gamma_r/N] 11^T.

There is no missing factor 1/2 and no additional batch-size factor. The scalar normalization `e=2 epsilon` already accounts for the raw gradient scale. The phase-summed versus phase-averaged convention is also stated correctly.

## 5. Bias regularity and the general nonbias relative bound

The bias note's selected first-batch event has failure probability `binom(16,8)/2^16`. Its completed-task geometric waiting time gives the stated uniform inverse-RMS moments. The q=6 condition holds for H=4 and H=6 and dominates its conservative C3 bound using inverse RMS to power 5. This agrees with the stronger all-coordinate argument in `adam_b16_regularity.md`; the latter improves the required power using the Hilbert norm chain rule. The bias note never claims that each finite-history update is C3 at a zero RMS.

The reference note's general first-derivative estimate is also correct. At zero contrast, `|m|,sigma<=gamma` and

    |Dq-Dm/epsilon|
      <=gamma[|Dm|+||Dg||_(l2(beta2))]/epsilon^2.

For a single logit coordinate, `Dm=(s_n/B)A_n` and the Hilbert derivative norm is `|s_n|sqrt(B_n)/B`. Uniform shuffled inclusion gives `E A_n=E B_n=B/N` and `E sqrt(B_n)<=sqrt(B/N)`. After summing phases and dividing by the leading `c0 s_n`, the relative error constant is precisely

    c0=H/(N epsilon),
    C_B=1+sqrt(N/B).

Both the mean-batch factor 1/B and the inclusion probability B/N are present. A zero-RMS history has derivative Dm/epsilon because its quotient residual is quadratic; this establishes first differentiability only. For exchanging the first derivative and expectation, a uniform local Lipschitz bound follows from the same quotient identity and the RMS norm inequality. Higher derivatives properly rely on the separate inverse-moment proof.

## 6. Scope retained

This audit verifies a stationary frozen response and its smoothness, not statewise fresh-label expectations along actual training. Whole-task reuse and moment carry are preserved. The result is compatible with the remaining local stability, endpoint, and observable-averaging arguments, but those components must still be supplied. The original RL-CIFAR task duration, ten classes, constant learning rates, and arbitrary image geometries are not covered by this certificate.
