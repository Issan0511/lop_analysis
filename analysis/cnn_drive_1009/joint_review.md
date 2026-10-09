# Independent final audit: finite joint hidden/head extension

2026-10-09. Reviewer: existing_drive_proof.

Reviewed /tmp/cnn_joint_extension_1009.md (204 lines), its implementation /tmp/cnn_joint_extension_check_1009.py, its recorded output /tmp/cnn_joint_extension_check_1009.out, and F1/F2/F3 in the integrated analysis/cnn_drive_1009/proof.md.

**Verdict: the principal formulas and the constructive nonempty finite-joint-GD argument are correct under the stated square-loss/ridge and differentiability assumptions. No algebraic blocker found.** The scope qualifications below must remain explicit. This audit did not measure the actual RL-CIFAR trajectory.

## 1. Old-label dependence and the reference: PASS

Lines 16–36 correctly distinguish the actual endpoint (H_Y,R_Y,V_Y), which can depend on old labels, from H_0,R_0 chosen before those labels.

For each fixed old label Y,

    E_new[grad m dot grad_hidden L_new]
       =(1/N)tr(V_Y^T R_Y^T H_Y V_Y).

This is the ordinary fixed-trained-head gradient, not a derivative through its fitting map. Taking R_Y=DH(theta_Y)[grad m(theta_Y)] correctly includes every hidden pathway and upstream contribution of a deep mean.

For V_0=H_0^T P_0 Y, substitution and cyclic trace give

    E_old[(1/N)tr(V_0^T R_0^T H_0 V_0)]
       =(nu/N)tr(R_0^T K_0 P_0^2 H_0).

No isotropy conditional on theta_Y is invoked. Coupling actual and reference endpoints with the same Y and bounding errors before averaging is valid. This repairs the independence problem in applying a frozen-network random-label identity to a label-trained representation.

## 2. Uniform endpoint error: PASS

Lines 40–66 are algebraically exact before the inequalities. With A_Y=R_Y^T H_Y,

    ||A_Y-A_0||_2 <= b h_0+r_0 a+ab.

The decomposition into one A-error term and two head-error terms yields

    |G_Y-G_0(Y)|
      <=[(B+e)^2 D_A+e(2B+e)r_0h_0]/N.

The fact that A is generally nonsymmetric causes no problem: |tr(X^T A Z)|<=||X||_F ||A||_2 ||Z||_F applies directly. The bound is pathwise for every old label, so averaging preserves it.

Important qualification: a,b,e,B must bound all old labels in their support (or be replaced by an explicit expectation/moment inequality). A maximum over a few sampled labels cannot certify the unsampled expectation. The document correctly states this at line 68. In the Rademacher N=2 witness the four old labels exhaust the support.

## 3. Ridge stationarity residual: PASS

Lines 74–93 correctly use the sum-loss residual

    S=(H^T H+lambda I)V-H^T Y.

The key normal-equation identity was independently expanded:

    (H^T H+lambda I)(V*(H,Y)-V_0)
       =Delta H^T (Y-H_0V_0)-H^T Delta H V_0.

The first term costs a E/lambda; the second costs a B/(2 sqrt(lambda)) because the maximum of sigma/(sigma^2+lambda) is 1/(2 sqrt(lambda)). Adding s/lambda for head nonstationarity is correct.

The universal bounds E<=Y_max and B<=Y_max/(2 sqrt(lambda)) are also correct. E_0=lambda(K_0+lambda I)^-1Y is a contraction; the head map has the singular-value bound above.

**Assumption to spell out in the final theorem:** lambda>0 regularizes every head coefficient in V, including the column corresponding to output bias. An unregularized output bias does not have the same lambda-strong-convexity or inverse bounds. The current formulas imply this but a reader might assume the common convention of excluding bias from weight decay. Mean-loss normalization also changes lambda and the residual unless scaled consistently (already noted at line 77).

## 4. Representation and full-mean sensitivity bounds: PASS

Lines 99–113 correctly obtain

    a<=L delta,
    b<=(J||u_0||+LM)delta.

The decomposition (DH_Y-DH_0)u_0+DH_Y(u_Y-u_0) includes the changing statistic direction; it does not silently freeze upstream derivatives of a deeper mean.

The operator norm 2-to-F bounds Frobenius changes in H and R and therefore also bounds their matrix spectral norms used in section 3. This norm conversion is sound.

Strict activation and unique-pool-winner margins provide a nonzero region where finite ReLU CNN outputs are polynomial in the hidden parameters. All derivative constants exist on a compact ball inside that region. This is a genuine local assumption, not permission to ignore gates or winners that switch outside the ball.

## 5. Positive hidden-learning-rate family: PASS

Lines 117–144 give a valid actual joint optimization process, with every hidden layer assigned the same strictly positive rate alpha and the head rate beta>0.

For beta<=1/(h_bar^2+lambda), the head update matrix I-beta(H_t^T H_t+lambda I) has norm <=rho=1-beta lambda<1. Starting at V_init=0,

    ||V_t||_F <= h_bar Y_max/lambda = V_bar.

The hidden-gradient estimate G=L(h_bar V_bar+Y_max)V_bar follows from the derivative of H and the Frobenius bound on (HV-Y)V^T. The bootstrap T alpha G<=delta keeps each iterate in the chosen ball.

Using the section-4 identity between any two representations yields the global head-map Lipschitz constant

    L_*=Y_max/lambda +Y_max/(4lambda)
       =5Y_max/(4lambda).

The tracking recurrence

    e*_(t+1)<=rho e*_t +L_* L alpha G

correctly compares the updated head first to V*(H_t,Y), then to V*(H_(t+1),Y). Summing it and adding V*(H_T,Y)-V*(H_0,Y) gives exactly the line-142 bound.

The existence argument is nonempty: choose finite T to suppress rho^T B, then choose a strictly positive alpha small enough for the remaining errors and the ball condition. For a fixed T this gives a nonzero interval of hidden learning rates, not merely a theorem evaluated at alpha=0. It does not establish that a particular practical rate is inside the interval.

Terminology: all hidden layers participate in the optimization; symmetry can still make some individual gradients zero. The witness correctly checks that both Conv weight tensors actually change, while allowing bias gradients to cancel for opposite labels.

## 6. Deeper-mean finite new step: PASS, with a smooth-segment condition

Lines 150–158 are Taylor's theorem:

    Delta m <= -eta grad m dot g +(M eta^2/2)||g||^2.

The expectation bound and eta<2gamma/(M G_new^2) follow. M must be a common Hessian bound for m over every relevant old/new-label step segment, and E||g||^2 must exist and obey the stated bound. For M=0 the affine case is treated separately and no division by zero is used.

**Keep explicit:** m must be C^2 on the segment, or have an appropriate Lipschitz-gradient remainder bound. An a.e. Hessian bound is insufficient if a ReLU kink is crossed. A positive gate/winner margin plus a step-length bound is one way to ensure this. For a first-layer affine mean, the exact mean-update identity does not require the network's gates to stay fixed after the update.

The update metric is scalar-rate Euclidean SGD/GD. A layer-dependent or Adam preconditioner changes the projection to grad m^T P grad L and requires a separate argument. Simultaneous head updates do not directly enter m, which is a hidden-parameter statistic.

## 7. Witness and exact nonemptiness: PASS at the stated evidence levels

I inspected the actual computation graph and output:

- both Conv layers and biases are hidden parameters;
- m is the actual first-Conv channel mean across the dataset and spatial positions;
- u=grad_hidden m uses all hidden coordinates;
- R is differentiated from the actual graph;
- the output-bias feature is included and ridge-regularized;
- all four old Rademacher label configurations are enumerated;
- the reported stationarity bound, endpoint error and expected-gradient quantities match the formulas.

I did not independently rerun the PyTorch trajectory in this reviewer environment. I did independently verify the exact rational reference inequalities using Python Fraction arithmetic:

    kappa_plus =3234321/1000000,
    kappa_minus=793881/1000000,
    epsilon=346768354897219580/633912538409375989
           =0.5470287048862239... <11/20,
    c^2=10201/16762=0.6085789285288151... >(39/50)^2.

The exact positive gaps are

    11/20-epsilon
      =37670824558744279/12678250768187519780 >0,
    c^2-(39/50)^2=3749/20952500 >0.

The input/filter construction also has strict ReLU and pooling margins at the reference: first-layer minimum activation 0.064, smallest winning-MaxPool gap 0.02. The second-layer activations are positive. Thus the exact F2 reference and the general positive-alpha existence theorem establish nonemptiness without relying on float64 endpoint arithmetic.

The numerical lower bound 0.584703647829285 is a floating-point evaluation of a rigorous inequality over the four-label support. The document correctly avoids calling that a machine-checked real-arithmetic certificate. Keep that distinction. The implementation checks pool margins at the old-step iterates; if making a separate claim about all continuous segments or a deeper-mean new step, add corresponding segment/margin bounds rather than infer them from those samples.

## 8. Scope qualifications that affect the final claim

1. This joint result uses square loss plus a fully ridge-regularized linear head. It is not a joint-CE convergence theorem. The separate CE short-training-time theorem should remain separately identified.
2. The positive-reference perturbation theorem proves negative expected mean drift under its certificate. It does not by itself prove equality with the literal isolated-unit logdet self term. Section 9 already distinguishes that limitation. If using the phrase “self direction,” identify the positive structural self shape/reference and distinguish the separate literal-self capacity theorem.
3. For arbitrary deep m the full statistic direction is grad_hidden m; using only the target filter direction would drop upstream mean motion. The reviewed R definition avoids this.
4. The theorem is conditional on persistence of its spectral/geometric and displacement bounds. It does not establish their invariance over indefinitely many tasks.
5. The two-Conv square-loss witness is not the exact RL-CIFAR architecture, dataset, CE/Adam optimizer or 16-image training trajectory. That distinction is explicit and should remain in the final answer.

Subject to these conditions, the extension is a valid conditional CNN theorem with genuinely positive finite hidden learning rates, label-dependent learned representations, all downstream pathways, and a nonempty family of realizable examples.
