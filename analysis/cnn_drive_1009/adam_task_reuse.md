# Standard Adam with taskwise reused random labels: a finite task-boundary theorem

2026-10-09. Scope: ordinary Adam, shared ReLU CNNs, all network parameters updated, and optimizer moments retained across tasks. The theorem conditions only at a task boundary. It never treats a label already revealed inside that task as freshly independent. A concrete nonempty CNN certificate below uses ordinary iid labels, not balanced label enumeration.

## 1. The task-boundary probability space

At a task boundary, let G contain the entire past: the current full network theta_0, Adam's uncorrected moments b_0,v_0, the global optimizer step n, past labels, and past batches. All these quantities may depend arbitrarily on the old tasks.

Draw one new assignment Y=(Y_1,...,Y_N), independently of G, with independent uniform C-class labels. Reuse the same Y throughout the next H updates. Let B_1,...,B_H be a sequence of minibatches and eta_1,...,eta_H positive learning rates fixed at the boundary. A random batch schedule may be conditioned on as well, provided it is independent of the new labels conditional on G. Thus repeated permutations, repeated images and unequal batch contents are allowed.

The actual updates are

    g_r(Y)=grad loss_{B_r}(theta_{r-1};Y),
    b_r=beta_1 b_{r-1}+(1-beta_1)g_r,
    v_r=beta_2 v_{r-1}+(1-beta_2)g_r^{odot 2},
    bhat_r=b_r/(1-beta_1^{n+r}),
    sigma_r=sqrt(v_r/(1-beta_2^{n+r})),
    q_r=bhat_r/(sigma_r+epsilon),
    theta_r=theta_{r-1}-eta_r q_r.

All vector operations in the last three lines are coordinatewise. Bias correction uses the global step n+r, not the task-local step. No momentum is reset. This is Adam without weight decay or AMSGrad.

Let u be the fixed mean input-patch vector embedded in the chosen first-Conv parameter coordinates, with zeros in all other coordinates. Include its augmented constant coordinate if a first-Conv bias is trained. The actual mean-preactivation decrease over the task is

    D(Y)=u^T(theta_0-theta_H)=sum_r eta_r u^T q_r.

After the assignment Y has been observed, E[g_r | within-task past] is not generally a uniform-new-label gradient. None of the following proof uses that false identity.

## 2. A frozen-parameter reference retaining all label and optimizer dependence

For each complete assignment Y, evaluate the whole schedule at the boundary state:

    g_r^0(Y)=grad loss_{B_r}(theta_0;Y).

Starting from the actual b_0,v_0, evolve reference moments using these gradients:

    b_r^0=beta_1 b_{r-1}^0+(1-beta_1)g_r^0(Y),
    v_r^0=beta_2 v_{r-1}^0+(1-beta_2)(g_r^0(Y))^{odot 2}.

Use the same global bias corrections and define bhat_r^0, sigma_r^0, q_r^0 and

    D^0(Y)=sum_r eta_r u^T q_r^0,
    Psi=E_Y[D^0(Y) | G,B_1,...,B_H].

The reference is a calculation at theta_0, not a replacement training trajectory. It still uses the same assignment in every occurrence of every image and retains its nonlinear effect on both moments. In general it is not Adam applied to an expected gradient. For finite N,C the expectation is an explicit finite sum over C^N assignments; computing that sum may be impractical in a real dataset, but the mathematical quantity is unambiguous.

## 3. Geometry bounds implying a uniform moving-trajectory error

Assume beta_1^2<beta_2 and that the optimizer was originally initialized with zero moments. Its current moments may be nonzero and arbitrary consequences of its actual past. The bias-corrected universal bound

    |q_{r,j}| <= K,
    K=[(1-beta_2)(1-beta_1^2/beta_2)]^{-1/2}

holds for every coordinate and every gradient history; adding epsilon only decreases the ratio. Weighted Cauchy-Schwarz proves this bound. It also applies to the reference, because appending the reference gradients to the actual old history gives another valid Adam gradient history. A sharper valid bound may replace K.

Consequently every actual trajectory obeys

    ||theta_r-theta_0||_infinity <= rho_r,
    rho_r=K sum_{k<=r} eta_k.

Take a radius R>=rho_H. Suppose the following condition is verified on the entire box ||theta-theta_0||_infinity<=R, for every batch and label assignment:

    |g_{r,j}(theta;Y)-g_{r,j}(theta_0;Y)|
        <= L_j ||theta-theta_0||_infinity.                (G)

Then

    |g_{r,j}(Y)-g_{r,j}^0(Y)| <= e_{r,j}:=L_j rho_{r-1}.

Condition (G) is a gradient-geometry condition, not an update-sign condition. For a ReLU/MaxPool CNN it can be checked by positive activation margins, pool-winner gaps, interval bounds on the intermediate maps, and Hessian row-sum bounds inside that fixed branch. It is enough that the entire radius-R box stays in one such branch. More general Lipschitz-gradient regions are also admissible.

For cross entropy with logits f_c, the Hessian is

    Hess loss = J^T [diag(p)-pp^T] J
                +sum_c (p_c-1_{Y=c}) Hess f_c.

Thus bounds on the current-state logit Jacobians and logit Hessians throughout the box give L_j, uniformly over labels. The second term must be included: a multilayer network is not linear in all its jointly trained parameters.

### 3.1 Moment errors without differentiating a square root at zero

Put t=n+r and define

    A_{r,j} = [(1-beta_1)/(1-beta_1^t)]
              sum_{k<=r} beta_1^{r-k} e_{k,j},

    C_{r,j} = sqrt{[(1-beta_2)/(1-beta_2^t)]
              sum_{k<=r} beta_2^{r-k} e_{k,j}^2}.

Then, pathwise for every assignment,

    |bhat_{r,j}-bhat_{r,j}^0| <= A_{r,j},
    |sigma_{r,j}-sigma_{r,j}^0| <= C_{r,j}.           (M)

The first inequality follows by linearity of the moment recursion. For the second, write sigma as the Euclidean norm of the vector containing the common old-history component sqrt(beta_2^r v_0/(1-beta_2^t)) and the weighted new gradients. The reverse triangle inequality gives (M). No positive lower bound on v_0 and no derivative of sqrt at zero is needed.

Define the explicitly computable per-coordinate error

    E_{r,j}(Y) = [A_{r,j}+|q_{r,j}^0(Y)| C_{r,j}]
                 /[max(0,sigma_{r,j}^0(Y)-C_{r,j})+epsilon].

Then

    |q_{r,j}(Y)-q_{r,j}^0(Y)| <= E_{r,j}(Y).         (Q)

Indeed the exact quotient difference can be written

    q-q^0 = [(bhat-bhat^0)+q^0(sigma^0-sigma)]/(sigma+epsilon),

and sigma>=max(0,sigma^0-C). This formula keeps the current-label dependence of the denominator rather than declaring it mean zero.

## 4. The task-boundary sign certificate

Define

    E_task(Y)=sum_r eta_r sum_j |u_j| E_{r,j}(Y),
    E_bar=E_Y[E_task(Y) | G,B_1,...,B_H].

The following bounds hold:

    |D(Y)-D^0(Y)| <= E_task(Y) for every assignment Y,

    boxed: E_Y[D(Y) | G,B_1,...,B_H] >= Psi-E_bar.   (T)

Therefore the finite, current-state condition Psi>E_bar proves strictly positive conditional expected mean decrease over this actual H-step task. It permits individual assignments and intermediate Adam updates with the opposite sign. It depends only on the boundary parameters, boundary moments, current-state Jacobian/Hessian geometry, the label law, the batch schedule, and step sizes. The unknown final displacement is not an assumption.

### 4.1 A more explicit connection to the raw self signal

If desired, define mu_r^0=E_Y[g_r^0(Y)|G,B_1,...,B_H]. Reuse of labels does not prevent the boundary identity

    E_Y[bhat_r^0]=beta_1^r b_0/(1-beta_1^{n+r})
                 +sum_{k<=r} [(1-beta_1)beta_1^{r-k}/(1-beta_1^{n+r})] mu_k^0.

Only linearity of expectation is used; the different g_k^0 are not assumed independent. For a boundary-measurable scalar reciprocal-denominator proxy d_r>0, put

    P=sum_r eta_r d_r u^T E_Y[bhat_r^0],
    E_ref_den=sum_r eta_r E_Y[sum_j |u_j bhat_{r,j}^0|
                     |1/(sigma_{r,j}^0+epsilon)-d_r|].

Then E_Y[D|boundary]>=P-E_ref_den-E_bar. A geometric lower bound on u^T mu_k^0 supplies the fresh-label self signal; the signed old moment contribution remains explicit. Neither this sufficient condition nor (T) assumes that momentum always has a desired sign. The exact-reference version (T) is often less conservative.

## 5. A nonempty ordinary-random-label shared CNN certificate

This is a conditional second-task result on an actual reachable history, not an invariant long-time result.

Use two single-input-channel images of shape 1x4:

    image 1=(1,1/2,2,1),
    image 2=(2,1,1,1/2).

Use shared Conv1x1 with two output channels, ReLU, MaxPool(1x2), flatten, and a free 2-class linear spatial head. No hidden or output bias is trained. All ten raw Conv/head parameters are updated by ordinary Adam; no reduced-parameter optimizer is substituted. The two pooled spatial vectors are (1,2) and (2,1), so the image-feature matrix has rank two. Every channel responds positively on every image and every spatial position.

Initialize both Conv weights to 7/10. Set the head rows, in channel-major spatial order, to

    V_0=(1/10,1/20,1/10,1/20),
    V_1=(-1/10,-1/20,-1/10,-1/20).

Use beta_1=9/10, beta_2=999/1000, epsilon=10^{-8}, eta=10^{-5}. Start Adam's moments at zero once, before the previous task.

In the previous task, the two independent uniform labels happen to be (0,1), an event of probability 1/4. Use this same assignment for two full-batch updates. Retain every parameter and optimizer moment. At the new task boundary draw two fresh independent uniform labels and reuse them for H=3 full-batch updates. There are four equiprobable new assignments. Each image has one label throughout the task; no image is copied once per class.

The target is channel 0's true first-Conv mean preactivation. Its mean input is mu_X=9/8, so zbar=(9/8)w_0.

### 5.1 Uniform geometry bounds are explicit

K<73 follows by squaring the rational formula in section 3. Across all five updates, regardless of labels, each raw parameter moves less than

    5 eta 73 = .00365 < .01.

Thus the Conv weights stay strictly positive, and the chosen spatial maxima stay fixed. On the slightly larger box |w_j|<=W=.73, |V_{c,j,s}|<=V=.16, let Q=sum_s x_s=3 and xmax=2. Logits are bilinear in w,V, with

    max_i sum_c |partial_i f_c| <= max(2VQ,W xmax)=1.46,
    max_c sum_i |partial_i f_c| <= 2(V+W)Q=5.34.

The softmax covariance has absolute row sums at most 1/2. Hence the first term in the CE Hessian has row sums bounded by 1.46*5.34/2=3.8982. The bilinear-logit Hessian contribution has row sums at most 2Q=6. Therefore every gradient coordinate is 10-Lipschitz in infinity norm throughout the required region, uniformly over all four assignments.

For the three new-task updates, use rho_r=73r eta and e_r=10*73(r-1)eta. The reference moments start from the rigorously enclosed actual two-step old-task state.

### 5.2 The sign is certified with exact rational intervals

The companion script performs outward rational interval arithmetic, including directed square-root bounds from integer isqrt and exp bounds from a Taylor series with a geometric tail. No floating point is used to establish the inequalities. It yields approximately

    Psi       = 2.159696621e-5,
    E_bar    <= 1.832169368e-6,
    Psi-E_bar > 1.976e-5 >0.

The directly enclosed conditional mean of the actual three-step CNN/Adam trajectory is about 2.159664048e-5. This direct enclosure is an additional check; theorem (T) uses the frozen-reference margin and the uniform movement bound above.

For the new assignment (0,0), the actual task mean decrease is negative, about -9.573817e-6. Its target first moment changes from about +.000564 at the first new-task update to -.015059 and -.029121. Thus the positive conditional task result does not follow from a pointwise sign assumption. Averaging over the four ordinary iid assignments is essential.

### 5.3 Connection to the original capacity-self direction

At this channel-symmetric boundary, f_n=2w V_{:,0,:}x_n and partial_{w_0}f_n=f_n/(2w). Therefore the fresh-label raw mean projection is

    E_Y[partial_{w_0}loss] mu_X
       =mu_X/(2w) average_n f_n^T(softmax(f_n)-uniform)>0,

because the current logits have nonzero class contrast. This uses the actual state resulting from the reused old labels.

The same mean-increasing direction is the literal capacity-self direction. Write X for the two-row pooled image matrix, Q=(XX^T) tensor I_2, and perturb w_0 by q mu_X. The raw-Conv Jacobian is independent of w, while the head Gram is (sum_j w_j^2)Q. Thus

    partial_q K_full=2w_0 mu_X Q,
    partial_q K_self=2w_0 mu_X Q,

where the self network retains only channel 0 and its existing head columns. Both are nonzero PSD, so both logdet-capacity derivatives are strictly positive for every positive ridge. Their negative-gradient direction lowers the mean. The finite-task Adam certificate therefore establishes actual expected motion in that same direction, with old-label dependence and full optimizer history retained.

## 6. What this does and does not bridge toward long time

At consecutive task boundaries, write nu_k=E[D_k | task-k boundary]. If a certificate of the form (T) provides nu_k>=a_k along the actual states, then

    sum_{k<=K}D_k >= sum_{k<=K}a_k + sum_{k<=K}(D_k-nu_k),

and the second sum is a task-boundary martingale. Standard stopped martingale variance bounds can control it. This is the correct level at which to condition when labels are reused inside tasks.

The theorem here does not prove that the certificate persists, that the a_k have a positive cumulative margin for infinitely many tasks, or that the random-label example's favorable old history repeats indefinitely. Those require an invariant region or other trajectory control. The example proves nonemptiness of a genuine current-state task certificate on a positive-probability reachable Adam history.

For RL-CIFAR's 1,200-image tasks, 400 epochs of 75 minibatches give H=30,000; labels are assigned once per task and Adam moments carry across tasks. The schedule fits the conditioning structure above. The C^N reference sum and the conservative rho_H/Lipschitz error can be prohibitive, and no claim is made that this certificate is positive for that training run. The actual biases and general channel geometry also need to be included in any such check.

Artifacts:

- `verify_adam_task_reuse.py`: exact interval certificate, optional `--torch` real Conv/ReLU/MaxPool/autograd comparison.
- `../../results/cnn_drive_1009/adam_task_reuse.json`: certificate output including all four assignments, carried moments, exact rational bounds, and optional Torch comparison.

The proof adds finite taskwise label reuse to actual Adam under explicit geometric error control. It does not import a per-step iid-label assumption, replace E[Adam(g)] by Adam(E[g]), or extend the activations beyond ReLU.

Independent audit: [adam_task_reuse_review.md](adam_task_reuse_review.md).
