# Independent review: native all-bias CNN reference neighborhood

2026-10-09. Reviewed `adam_native_bias_reference.md`. No numerical experiment was rerun; the review concerns the analytic construction and its stated scope.

**Verdict: PASS. No blocking mathematical issue found.** The reference construction is nonempty for the stated native 32x32, Conv5/pad2, width-16, two-FC100 architecture. It includes every ordinary bias and a full-dimensional neighborhood of independently variable raw parameters. The result itself remains binary and one-image, and its local moving-trajectory result is an averaged-ODE theorem pending the separate stochastic averaging/tube argument.

## 1. Strict native routing and nonzero raw sensitivities

The center-offset construction works for both full and literal-self networks. With identical strictly increasing RGB planes, positive center-offset channel connections, and zero hidden biases, each convolution produces a positive channel-dependent multiple of its input scalar field. Both MaxPool2 operations select the unique bottom-right entry of each window. Removing other first-layer channels still leaves a positive center route, so the self network has the same strict baseline ordering.

There are finitely many preactivations and pool comparisons. Hence sufficiently small positive off-center coefficients and sufficiently small positive hidden biases preserve the strict margins simultaneously in both architectures. Once all weights/biases are strictly positive, sufficiently small independent perturbations give an open raw-parameter neighborhood; no equality or tying constraint is needed for the final neighborhood.

The spatial dimensions are consistent: 32x32 -> 16x16 -> 8x8, with 16 output channels, gives flatten dimension 1024 before FC100 and FC100.

Padding does not invalidate the nonzero-gradient assertion. A concrete common interior route is the second Conv winner at spatial location (7,7) in its 16x16 map, which is the bottom-right site of an interior final-pool window. Its center-offset incoming first-pool site is (7,7), whose first Conv winner is (15,15). Every first Conv 5x5 offset at that site lies strictly inside the 32x32 image. Thus every Conv1 offset/input-channel coefficient has a strictly positive selected route through a positive Conv2 center weight and the dense positive FC suffix. Every Conv2 offset at (7,7) also receives a strictly positive first-pool activation, since its incoming indices lie in {5,...,9}^2. This covers all raw spatial offsets and input/output channel combinations. Perturbations preserve the selected route.

The positive input, active ReLUs, dense positive FC weights and positive output-contrast vector then give positive hidden sensitivity for every hidden raw weight and bias. Raw binary head coordinates have sensitivity +/-h_i/2, and output biases have +/-1/2. For every alpha>0 all these sensitivities are nonzero. The alpha=0 limiting capacity statement must remain separate from the nonzero-sensitivity neighborhood, as the manuscript correctly does.

## 2. Mean direction and nonnegative-coefficient polynomials

The 75 first-filter patch means are positive even after averaging padded patches, since each offset sees positive image entries at some spatial positions. The bias direction is exactly one. Consequently the directional derivative of every first-layer preactivation in the target channel is at least one, and its positive downstream paths imply every component of D_u h is strictly positive.

Within a fixed active/winner branch, each hidden output is a polynomial with nonnegative coefficients in the hidden weights and biases. Convolution sharing adds sums of monomials; padding contributes only zeros; a selected MaxPool entry is a coordinate selection. None introduces a subtraction. Evaluating at positive weights and nonnegative biases therefore gives nonnegative first derivatives and nonnegative mixed derivatives in a nonnegative mean direction. This justifies both `grad psi>=0` and `D_u grad psi>=0`, including hidden-bias Jacobian columns.

The same reasoning applies after deleting the other first-layer channels. Positive downstream biases mean the self network need not be a scalar rescaling of the full network, but no rescaling is used in the proof.

## 3. Full and literal-self NTK formulas include every block

For independently trainable head rows initialized at +/-alpha d, the free head-weight block contributes `||h||^2 I_2`, the two output biases contribute `I_2`, and every hidden raw coordinate contributes through the row pair `(+alpha partial_j psi, -alpha partial_j psi)`. Thus

    K=(||h||^2+1)I_2+alpha^2||grad psi||^2 vv^T

is exact. There is no omitted hidden-bias or output-bias block and no tied-head Jacobian substitution.

Differentiating along the actual first-mean direction gives the stated K'. Its first term is a strictly positive multiple of I_2 because h and D_u h are componentwise positive. Its rank-one term is PSD by the polynomial derivative argument. Hence both full and self K' are positive definite, with the stated lower bounds, for every alpha including alpha=0. This proves positive logdet-capacity slope for every positive ridge.

The explicit E_K perturbation bound is valid: expand `J'J^T+JJ'^T` around its reference and bound the three error products and their transposes using Frobenius norms. A strict margin below the reference minimum eigenvalue preserves positive definiteness of K', not merely one selected ridge's trace sign. This supplies an open neighborhood allowing independent changes to all raw coordinates.

## 4. Output-bias cancellation and self-network interpretation

Choosing b=-alpha psi sets the full network's contrast to zero without changing any hidden activation, derivative, or raw NTK block. Moving the two output biases by +/-z0 produces half-contrast z0 and Euclidean parameter displacement sqrt(2)z0, as used later.

The retained output bias generally does not cancel the literal-self contrast. The manuscript explicitly confines the self conclusion to its capacity derivative and does not claim that the self network's CE/Adam drive has the same sign. This distinction is necessary and is handled correctly.

The actual full-network comparison is valid: on the positive-contrast side, positive target sensitivities and positive mean-patch coefficients make every relevant term of the diagonal stationary Adam mean point toward first-mean decrease. The independent-task grouping extends this marginal sign to full-batch H-fold reuse. It does not assert a conditional sign at an arbitrary realized optimizer state.

## 5. Averaged ODE neighborhood and endpoint margin

The output biases give the state-independent lower bound `||grad z||^2>=1/2`. Together with positive diagonal coefficients, this implies the stated normal contraction and the finite path-length ratio. The local Lipschitz claim follows from the coupled EMA/RMS bounds and epsilon>0; in the nonzero-sensitivity neighborhood the stronger smoothness used by the companion endpoint-map proof is also available.

The bounds on z0 in equation (15) place the initial point inside r/4 of the equilibrium and bound all subsequent path length by r/4. Thus the first-exit argument closes without assuming future boundedness. Normal contraction then gives z->0, and finite path length gives a parameter endpoint. Integrating the positive mean-to-contrast rate ratio gives the strict lower bound (16).

Multiplying the phase-averaged field by H to use a block-time convention only rescales time; the normalized path and endpoint bounds are unchanged. The actual stochastic theorem must use one convention consistently, which the companion block formulation does.

## 6. Small-bias limit and remaining scope

The simultaneous choice tau=alpha and z0=c alpha^2 is compatible with the strict local construction: hidden-bias and target-contrast sign radii can shrink as O(alpha), whereas gates, capacity margins and the norm M have uniform local bounds. The required z0 radius inequalities therefore hold for sufficiently small positive alpha. No uniform lower bound on hidden drift or sink survives alpha->0, and none is claimed.

The restrictions are correctly stated: one image, binary labels, strict active routing, and a local averaged trajectory in this reference note. It does not claim a multiclass many-image RL-CIFAR trajectory, neuron death, or applicability beyond ReLU.

No correction is required for mathematical correctness.
