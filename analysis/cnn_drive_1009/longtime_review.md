# Independent audit of cnn_longtime_1009.md

2026-10-09. Reviewed the centered-one-hot L=10, M=3, N=9 revision and verified the two requested clarification fixes after their application. Verdict: the displayed conditional-drift, stopped-supermartingale, nonexit-probability, and limiting-sink claims are valid within the explicitly stated toy model. No blocking mathematical defect found.

## Checks passed

1. **Shared Conv SGD normalization.** An active group's selected augmented patch is a=(e_c,1), ||a||²=2. Thus full weight-and-bias SGD gives the factor 2η/N in the h recurrence. The actual mean-patch direction has inner product k with a, giving Δm=(k/2)Δh. No bias-only update or independent per-position weights have been substituted.
2. **Old-label dependence.** Conditioning includes the current representation and old labels. Only the fresh new labels are averaged. The surviving coefficient ||S_old||² is nonnegative without independence from the representation. For three centered-one-hot labels among ten classes it is at least 3−9/10=2.1. Ordinary new-to-old label reuse therefore causes no missing independence step in this family.
3. **Finite-step positivity.** The lower bound on the multiplicative factor uses |<S,S′>|≤M² and ||S||²≤M². The stated 2ηM²/(Nλ)<1 guarantees every currently positive h remains positive, including an update that exits the upper support threshold.
4. **Support stopping.** τ is the first state at which some h≥H_exit. Every update before τ is computed inside the proved pooled-support geometry; stopping freezes the resulting state at τ. All h values at the stopping state remain nonnegative. Overshoot is permitted and does not invalidate the stopping argument.
5. **Doob inequality.** Each stopped h, and therefore their stopped sum, is a nonnegative supermartingale. Exit implies stopped sum≥H_exit. Thus P(τ<∞)≤Σh_0/H_exit. For the stated parameters this is 1/13, so permanent pooled-support survival has probability at least 12/13.
6. **Limit h→0.** Nonnegative supermartingale convergence gives finite limits. Expected total predictable decrement is at most h_0, so its nonnegative sum is finite almost surely. On permanent survival, any positive limit plus Ση=∞ and ||S_old||²≥2.1 would force an infinite decrement sum. Therefore every limit is zero.
7. **Actual mean and cumulative direction.** The exact affine relation yields m_T−m_0→−kh_0/2 and ΣηZ→kh_0/2 on permanent survival. This allows finite upward updates and entails zero, not positive, limiting ordinary gradient average under constant step size.
8. **General stopped martingale section.** Predictability of 1{t<τ}, the variance summability requirement, martingale-series convergence plus Kronecker's lemma, and the finite-time Chebyshev bound are consistent. A positive absolute drift lower bound forever would conflict with a lower-bounded mean; the text explicitly avoids claiming this for the toy.
9. **Numerical scope.** The text now reports the actual M=3,L=10 shared-Conv/autograd update check, not only the earlier scalar M=1 verification. These finite computations are correctly separated from the infinite-time proof.

## Clarifications requested and confirmed applied

- “Same activation supports” now means pooled feature supports across image groups. Nonwinning locations can cross their own ReLU thresholds; they do not affect the output derivative while the winner remains unique and positive.
- “No activation reaches zero” now refers to the currently positive pooled h_c.
- The signal-relative martingale summation starts after L_{t+1} first becomes positive.

## Scope restrictions that must survive integration

- This is a one-convolution, support-separated, exact-ridge-head, squared-loss, full-batch-SGD family. Other channels are finite and train, but they have disjoint pooled supports and therefore no competing head contribution on the target's active group.
- No trainable output intercept is included. Adding one destroys the displayed block decomposition and is not covered.
- The proof is not for the actual two-Conv/two-FC RL-CIFAR network, CE training, Adam, or joint hidden training during each old-head fitting stage.
- The probability statement is at least 12/13 under the toy's future label randomness, not a statement that 92.3% of realistic CNN initializations or real training runs sink.
- The angle certificate remains strictly positive at every finite surviving state, but an absolute signal margin need not remain positive uniformly; the drift amplitude vanishes.
- Because support separation removes competition on the target group, this family should be presented as a rigorous nonempty long-time construction and a bridge for the general program, not as a completed solution of finite overlapping-channel CNN competition.
