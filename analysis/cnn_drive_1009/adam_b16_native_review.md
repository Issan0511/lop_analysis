# Code and proof review: native batch-16 finite verifier

2026-10-09. **Verdict: PASS / approve in the stated finite-verifier scope. No blocking issue or required code change found.**

Reviewed:

- /home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/verify_adam_b16_native.py
- /home/issan/Projects/claude/wt/cnn_drive_1009/analysis/cnn_drive_1009/adam_b16_reference.md

The full verifier was not rerun. The code and its mathematical identities were inspected directly; the only executed check parsed the Python source to confirm the result file receives an actual trailing newline. No repository file was edited.

## 1. Scope and architecture accounting

The executable finite witness has N=32, B=16, E=2, H=4 and 3712 independent raw coordinates. It uses RGB24-by-24 images, Conv5/pad2 widths 2/2, two pools, FC32/FC32, and a binary affine head with all ordinary biases. Flattening after the pools gives \(2\cdot6\cdot6=72\), matching the first FC shape. The literal-self model keeps the same second-Conv output width and therefore also has 72 flattened features.

The companion's analytic construction uses the larger native RGB32-by-32, Conv16/16, FC100/100 architecture and also covers N=48. The numerical script is a smaller finite witness of the same raw identities; it is not an execution on that larger network. Its output architecture and scope fields accurately record the smaller dimensions. Reporting should preserve this distinction.

The script neither executes Adam training nor estimates the stationary field, a probability of success, or an admissible infinite-time learning rate. Its documentation explicitly says so. The independently proved bias coefficient is a named external premise of the spectral bound, not a quantity silently inferred from these finite checks.

## 2. Rank-32 input construction and zero mean perturbations

Lines 88–131 implement the companion's level-simplex construction correctly. The 32 selected red-plane pixels at spatial indices \((4r+3,4c+3)\) correspond to selected pooled positions of the center-dominated two-Conv network. The code does not merely presume that correspondence: it computes the actual input-to-last-feature derivative minor and asserts a positive smallest singular value.

With that minor \(B_A\), the vector beta is \(d^TB_A\), and dividing the columns of the centering matrix by beta gives

\[
 T=(I-11^T/N)\operatorname{diag}(1/\beta_i).
\]

The subsequent scalar normalization does not change its rank, level nullspace, or zero column sum. Thus \(T\beta=0\) keeps all perturbed images at the same head-contrast level, while \(1^TT=0\) keeps their average input equal to the base image. The code explicitly checks both identities.

The selected Jacobian minor has full column rank. The companion's proof of full feature-row rank is valid: a row relation first has coefficient sum zero after multiplication by positive d, then belongs to the one-dimensional left nullspace of T, hence vanishes. The code also independently checks the final 32-by-32 hidden feature singular values.

The image-separation loop preserves the actual full and literal-self pool winners, the positive input range, and coordinatewise relative sensitivity closeness. Hidden positivity follows from positive hidden weights/biases and positive images and is also checked after scaling for both models. The strict final pool-gap check prevents a zero-margin winner selection from being mistaken for a fixed routing neighborhood.

## 3. Actual first-Conv mean direction and literal self

Lines 50–54 use the mean of the actual padded input patches across the data and all convolution positions. The unfold ordering matches the Conv kernel layout. The selected first-channel bias coefficient is exactly one; every other raw coordinate has zero mean coefficient. Therefore this is the gradient of the actual spatial/data first-Conv preactivation mean, not a pure bias direction or a selected-patch surrogate.

The symmetric image construction makes this vector unchanged from the base image. The code checks both the image mean and the packed raw direction directly.

Lines 57–60 implement literal channel isolation: they retain first-Conv output channel 0, its bias, and the corresponding input slice of the second-Conv kernel. Other second-Conv outputs, later layers, the head, and their original biases are retained. The self model's head is not refit. Its logits need not be zero merely because the full model's logits are zero; the code makes no such assertion.

The self capacity direction is recomputed with the self parameter shapes and describes the same first-channel affine mean. This is consistent with a separate literal-self capacity calculation.

## 4. Raw scaling and spectral inequality

Lines 134–139 scale the last hidden FC weights and bias, together with the two head rows, by t, and the canceling output bias by \(t^2\). Because the reference head rows are antisymmetric, this is exactly the required head-contrast scaling.

The mask at lines 224–228 selects last-FC weights/biases and output-head weights, namely the order-t columns of the **raw** contrast Jacobian. Earlier coordinates have order \(t^2\); the final two output-bias columns remain order one and are excluded from the small-sensitivity approximation. No derivative with respect to a tied amplitude is substituted for the independent raw Jacobian.

The code obtains its initial scale proposal from the leading full-row-rank block, then recomputes the complete nonbias Jacobian at the actual scaled point. The decisive ratio is

\[
 \rho_{\rm spec}
 =C_B(\gamma/\epsilon)
 \frac{\|R/t\|_{\rm op}\|R/t\|_F}
 {\sigma_{\min}(R/t)^2},
 \qquad C_B=1+\sqrt{N/B}.
\]

This is algebraically the companion's finite relative normal-error ratio; the t normalization cancels in the ratio. The lower bound at line 249 restores \(t^2\) exactly. Using the complete normalized matrix retains the small normal modes that distinguish the 32 nearby images.

Conditional on the separately proved nonnegative bias common-mode contribution, the resulting lower bound is valid for the symmetric part of the actual normal linearization. The output's conditional field name and explanatory string communicate that dependency explicitly.

For the target direction, lines 251–257 verify positive sensitivities and an entrywise relative reserve. The resulting N-vector is a componentwise lower bound on \(L^Tu\). Because its entries are positive, its squared norm is a valid lower bound on \(\|L^Tu\|^2\), the exact coefficient for the endpoint ray \(LL^Tu\). The script does not claim that its variable named ray is the full P-dimensional ray itself; the exported name correctly calls it a coefficient lower bound.

## 5. Normalized full and literal-self capacity

Lines 142–160 compute the complete raw logit Jacobian J and its directional derivative \(J'\) with an actual JVP in the fixed mean direction. The output-bias columns have zero directional derivative, as checked explicitly.

With \(R=J_{\rm nonbias}/t\), \(R'=J'_{\rm nonbias}/t\), the script uses

\[
 K=K_b+t^2RR^T,\qquad
 K'/t^2=R'R^T+RR'^T.
\]

Thus its trace solve is exactly the capacity slope divided by \(t^2\), at ridge one. Flattening image-major/class-minor gives the stated bias kernel \(11^T\otimes I_2\). The same computation applies to the independently packed self architecture.

The denominator-error bound is valid: for \(A=I+K_b\succeq I\) and \(C=RR^T\succeq0\), the resolvent identity gives

\[
 \|(A+t^2C)^{-1}-A^{-1}\|_{\rm op}
 \le t^2\|C\|_{\rm op}.
\]

Pairing with the nuclear norm of the symmetric kernel derivative yields exactly the factor \(t^2/2\) used in the code. The internal variable named limiting removes the denominator correction at the chosen t; its derivative numerator can still contain the finite order-\(t^2\) contribution from earlier layers. This does not invalidate the bound, and no result field mislabels it as an independently evaluated exact t=0 derivative.

The companion's separate joint image/scale continuity argument is also consistent. At coincident images, the leading head-weight derivative gives a strict positive scalar-image identity block, while the last-FC contribution is positive semidefinite. With fixed positive ridge, the normalized capacity expression is continuous at the two-parameter corner. Image separation can therefore be fixed first and scale chosen second. No uniform unnormalized capacity margin as \(t\to0\) is assumed.

## 6. Batch CE sign, normalization, and label reuse

The binary convention is coherent throughout: class index 0 has \(Y=+1\), index 1 has \(Y=-1\), and \(z=(f_0-f_1)/2\). Therefore

\[
 \nabla_\theta \ell
 =J_z^T(\tanh z-Y).
\]

PyTorch cross_entropy uses the mean batch reduction, so the expected raw batch gradient in lines 181–182 correctly includes exactly one division by B=16. There is no extra factor two from the half-contrast convention and no omitted inclusion factor.

The sensitivity matrix is computed from the same fixed network state as the batch forward. There is no batch-dependent normalization layer that would invalidate selecting its image rows. The checker compares every raw coordinate after normalizing by its sensitivity envelope, which is materially stronger than a single absolute tolerance when the early raw gradients are extremely small.

Within each recorded task, one label vector is reused for both independently generated permutations. The four displayed label vectors are finite identity checks, including one random assignment; they are not presented as exhaustive enumeration or a measurement of the probability law. This is appropriate for validating the algebraic CE identity.

## 7. Companion derivative estimate and regularity boundary

The proof of the relative response estimate in section 2 of the companion is sound. At positive RMS, the numerator-denominator difference is bounded with \(|m|,\sigma\le\gamma\) and the Hilbert norm inequality for the RMS derivative. At zero RMS, every weighted past gradient is zero and \(q-Dm/\epsilon\) has quadratic remainder, giving the same first-derivative formula.

For a single output coordinate, \(Dm=s_nA_n/B\) and the second-moment derivative norm is \(|s_n|\sqrt{B_n}/B\). Exchangeable sample inclusion gives \(EA_n=EB_n=B/N\), while Jensen gives \(E\sqrt{B_n}\le\sqrt{B/N}\). These yield exactly \(C_B=1+\sqrt{N/B}\), including the mean-loss normalization.

This does not assert pointwise C2/C3 at a zero-RMS finite history. The companion expressly leaves higher expectation regularity to the separately audited inverse-moment theorem. Its near-identical sensitivity margin supplies a strict unbalanced-count event and can be shrunk to satisfy that theorem on a compact neighborhood.

## 8. Limitations and review conclusion

All non-rational numerical margins here are float64 finite checks, not outward-rounded formal interval certificates. The analytic construction supplies the nonempty-family argument; the script is a reproducible finite witness and implementation check. A successful run does not by itself prove the stationary bias sign, probability budget, noexit event, or long-time endpoint.

The code and companion maintain these distinctions. No security, data-destructive, or unintended external-action issue is introduced by this bounded local verifier. No requested correction was identified.
