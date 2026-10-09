# Ten-class native CNN: a nonempty local self-direction theorem at constant rate

2026-10-09. This is a finite-horizon theorem for ordinary Adam, with all raw
weights and biases free, retained moments, and true taskwise label reuse.
It uses ReLU only. It does not prove the standard CIFAR run or fixed-rate
infinite-time sinking. In particular, it does not extend the binary equilibrium
theorem by assuming that ten-class uniform predictions are an Adam equilibrium.

The architecture is RGB32, Conv5/pad2(16), ReLU, MaxPool2,
Conv5/pad2(16), ReLU, MaxPool2, FC100, ReLU, FC100, ReLU, and a ten-class head.
There are N=1200 distinct images, batch size B=16, E=400 fresh independently
shuffled epochs per task, and H=30000 updates per task. Each task assigns every
image an independent uniform ten-class label; it reuses that assignment for
all E epochs. Tasks are independent. Standard Adam has beta1=.9, beta2=.999,
epsilon=1e-8, zero moments at initialization, global bias correction, and no
weight decay, moment resets, parameter constraints, or projection.

## 1. What is proved, and the restrictions that remain

There are a dataset, a finite ridge lambda>0, a reference theta*, and an open
ball of independently variable raw parameters about theta* such that:

1. The stationary task-summed Adam field has a strictly positive projection
   onto the true first-filter mean direction u. This sign is derived below
   from the class-head geometry, not assumed as a property of a future path.
2. The original full and literal-self all-raw NTK capacity derivatives in
   direction u are both strictly positive throughout that ball.
3. For each prescribed failure probability alpha>0 and a sufficiently small
   fixed clock horizon S>0, there is a positive constant learning rate eta
   such that the actual jointly trained CNN, through K=ceil(S/eta) tasks,
   stays in this ball for every label history and has

       P[m(theta_K) <= m(theta_0)-d S/2] >= 1-alpha,

   where d>0 is a uniform local task-summed mean-field bound. Initial points
   range over a smaller full-dimensional ball, with common constants.

The target filter's effective gradient support is one image and one spatial
site. The other 1199 images are distinct and are actually trained, with their
own independent labels and ordinary batches, but this filter is ReLU-inactive
on them. This is an open gating condition, rather than frozen coordinates or
correlated labels. The initial head is specially chosen; its exact equalities
are needed only to construct theta*, and arbitrary sufficiently small raw
perturbations are allowed in the theorem. The capacity ridge is selected by
a finite sufficient bound. No practical learning-rate threshold is certified.
The first numerical witness starts with negative mean. Section 2.1 supplies
a separate analytical reference with positive mean, which can decrease while
remaining positive. Neither onset of negative mean, gate death, full-parameter
convergence, nor permanently signed velocity is proved. The number K can be
arbitrarily large by taking eta smaller, while
S stays small. That is not fixed-eta infinite time or the actual eta=.001 run.

## 2. A native, shared-convolution reference with one active target site

All input pixels are nonnegative. Choose one target image with a single
5-by-5 RGB patch of ones, centered at an interior site such as (3,3). Every
other pixel of this image lies strictly between 0 and .1. The other 1199
images are mutually distinct and have all pixels strictly between 0 and .1.
Set the target first-Conv kernel's 75 entries to one and its bias to -74.5.

At the matching site its preactivation is .5. A different 5-by-5 window
overlaps at most 20 of the 25 unit spatial pixels, so its RGB patch sum is
at most 60+15(.1)=61.5, including the possibility of zero padding. On every
other image the sum is less than 7.5. Thus this filter has exactly one
positive preactivation in the complete dataset, with strict gate margins.
Its first MaxPool winner is the active site; all other target-channel pools
are identically zero on a parameter neighborhood.

Choose every other first-Conv weight and bias positive, and every downstream
hidden weight and bias positive. Make at least one second-Conv coefficient
on the target input channel center-dominant, with other positive entries
sufficiently small. The active pooled target at site (1,1) then reaches a
strict winner in the second MaxPool. Free positive FC weights propagate it
to all final hidden features. The same path survives deletion of the other
first-Conv channels, which defines the literal-self architecture.

All nonidentical competing positive pooling expressions can be made unequal
by generic small positive choices of the kernels and background pixels.
There are finitely many images and windows, and these equalities are finitely
many proper polynomial zero sets on each fixed preceding branch. Ties that
cannot be removed here are between identical local functions: first-layer
zeros after strict negative ReLUs, or second-layer bias-only entries when the
literal-self input window is identically zero. Their maximum is that same
smooth function. No differentiability of a maximum of two distinct crossing
functions is asserted. The composite network therefore has a smooth local
branch, both full and self, despite these harmless implementation argmax ties.

Let h_n be its 100 final hidden features, w any positive 100-vector, and

    psi_n = w^T h_n,
    kappa = partial psi_target / partial z_target > 0.

Here z_target is the single active preactivation of the selected first filter.
The true mean of this filter over images and first-layer spatial positions is

    m(theta)=u^T theta,
    u_(W1,c,a,b)=mean_(n,s) x_(n,c,s+(a,b)),
    u_(b1)=1,

with zero entries on every other raw coordinate. Padded input entries are
zero. Thus u>=0 and U=sum_j u_j>=1. The direction concerns all spatial sites,
including unselected and inactive ones; it is not an average over winners.

### 2.1 An alternative with initially positive mean

The already-negative mean of the preceding numerical witness is not essential.
Let L=32^2, T=NL, b0=1/10, A=9/10, and tau0=A/(4T). Take the selected first
filter's red center coefficient to be one, all its other 74 coefficients
to be zero, and its bias to be -b0. Every background pixel, in all three
colors and all images, lies in the open interval (b0-2tau0,b0-tau0).
Only the red pixel at the target's interior site equals b0+A=1. Backgrounds
can be chosen distinct and generic inside this interval.

The target preactivation is A; all T-1 others lie in (-2tau0,-tau0).
The exact all-image/spatial mean consequently satisfies

    m* > A/T-2tau0 = A/(2T)>0.                     (2.1)

Downstream positive paths and smooth full/self pool branches are constructed
as above. The target's winning patch has 75 strictly positive entries; put
c_j equal to that patch entry for each kernel coordinate and c_bias=1.
Every c_j>=c_min:=b0-2tau0>0. The gradients now have different positive
scales kappa c_j, handled explicitly in section 3 below. None of the zero
reference kernel entries is frozen; strict gate margins and the stationary
sign allow a full raw-parameter open neighborhood of this reference.

To keep the whole finite trajectory's mean positive as well, reduce the
radius so U r<m*/2 and choose S<m*/(8UM), in addition to section 5's
conditions. Starting within r/4 actually gives an even stronger initial
bound than m0>m*/2. Total mean motion is less than 2UMS<m*/4, so every
intermediate mean exceeds m*/4. The proved terminal decrease therefore goes
from a positive mean to a smaller positive mean. The selected target ReLU
remains active; no crossing or death is being asserted.

This alternative is an analytical nonemptiness argument. The saved full
native float64 Jacobian/routing witness uses the preceding all-ones-kernel
reference; it is not a numerical verification of this second dataset.

## 3. Deriving the stationary mean-field sign with ten genuine classes

At the reference take class coefficients

    a_c=-2/5 for six classes, a_c=3/5 for four classes,
    sum_c a_c=0,
    V_c=a_c w, b_c=-a_c psi_target.

These are initial values of fully independent head coordinates, not permanent
ties. Target logits are all zero. Other images need not have uniform logits.
Let I_k be the event that the target label in task k is one of the six classes,
and xi_k=3/5-I_k. Then I_k are independent Bernoulli(3/5), and E xi_k=0.

Only the target image has a nonzero derivative in the target first-filter
block. Its active input patch consists of ones, so all 75 kernel coordinates
and the bias have the same individual-image derivative kappa. At frozen theta*
their exact batch gradient streams are therefore

    g_(t,j)=-(kappa/16) d_t xi_(task(t)),
    d_t=1{the target is in the current batch},

for each of these 76 coordinates. This statement includes the entire batch
loss and every other image label; their gradients in this block vanish by
strict inactive gates. E g_(t,j)=0 at this reference.

The task-grouped two-point lemma in
[the multiclass obstruction](adam_tenclass_equilibrium.md), section 2, retains
the joint numerator/denominator history. At any stationary phase group the
EMA weights over visits belonging to the same task:

    A_k=sum_(visits in task k) (1-beta1) beta1^lag,
    B_k=sum_(visits in task k) (1-beta2) beta2^lag.

For q=3/5, e=16 epsilon/kappa and

    c(q,e)=(1-q)(2q-1)/(2(q+e)^2)>0,

that lemma gives the sign-reversed quotient bound

    E[q_j^* | complete shuffle schedule] >= c(q,e) sum_k A_k B_k.

The target occurs once every 75-step epoch. At any phase the latest occurrence
has lag at most 149: the extremal gap puts it at the start of the previous
epoch and the end of the current epoch. This remains true across task
boundaries because tasks contain complete epochs. That one occurrence alone
contributes to both weights of the same task, so

    sum_k A_k B_k >= omega_*= (1-beta1)(1-beta2)(beta1 beta2)^149 >0.

Consequently every stationary phase has

    E q_j^* >= delta_* := c(3/5,16epsilon/kappa) omega_* >0,
    u^T barF(theta*) >= d0 := H U delta_* >0.             (3.1)

The parameter update is -eta q, so (3.1) is a downward mean direction.
For the alternative reference of section 2.1, replace kappa in each coordinate
by kappa c_j. Since c(q,e) decreases with e, a single valid lower bound for
all coordinates is

    delta_*=c(3/5,16epsilon/(kappa c_min)) omega_*>0.

The same u projection, all-raw continuity, capacity argument and finite-time
transfer follow. Exact equality of the 76 gradient streams is unnecessary.
The same derivation uses a negative direction if the six/four head signs are
reversed. Thus ReLU or capacity positivity alone does not choose this sign.
It is a genuine stationary Adam bias from unequal gradient magnitudes, not
an expected-CE-gradient signal or a first-step approximation. Task reuse,
shuffle randomness and old moments have not been discarded.

The stationary quotient is locally Lipschitz in the raw parameters: the RMS
norm inequality and epsilon>0 give the bound in
[taskwise averaging](adam_task_averaging.md), section 3. Therefore (3.1)
persists on a sufficiently small full-dimensional raw-parameter ball, even
when target predictions cease to be exactly uniform or head rows cease to
be repeated/proportional. Equality of the 76 gradients is not required off
the reference. Equal parameter values are never imposed during training.

## 4. Connecting the sign to the original full and literal-self capacity

At the reference let Hf be the N-by-100 feature matrix, and let D be the
N-by-P_hidden matrix whose rows are the gradients of psi_n with respect to
all hidden raw weights and biases. For image-major, class-minor indexing,
the full all-raw logit NTK is exactly

    K=(Hf Hf^T+11^T) tensor I_10 + (D D^T) tensor (a a^T).  (4.1)

The first two terms include the independently trainable output weights and
biases. The hidden term includes every hidden layer. Put Hf'=D_u Hf and
D'=D_u D. Then

    K'=(Hf' Hf^T+Hf Hf'^T) tensor I_10
           +(D' D^T+D D'^T) tensor (a a^T),
    tau=tr K'=20 <Hf,Hf'> + 2||a||^2 <D,D'> >0.            (4.2)

Here Hf,D,Hf',D' have nonnegative entries, and Hf' is nonzero on the target
image. To justify this despite the negative target bias, differentiate
the fixed-route forward and backward equations: input activations and
downstream weights are nonnegative, u is nonnegative, and downstream gates
are fixed. First-layer sensitivities do not change along u. A downstream
weight sensitivity differentiates into its nonnegative downstream sensitivity
times a nonnegative change of its input activation. Hidden bias sensitivities
are unchanged along u. This proves D'>=0 without asserting that a polynomial
evaluated at a negative bias has all positive monomials. On other images the
whole output is locally independent of this target-filter block, so Hf'=D'=0.

Equation (4.2) does not claim that K' is positive semidefinite. Define finite
nonnegative bounds

    R=tr K >= ||K||op,
    T=20||Hf||F||Hf'||F+2||a||^2||D||F||D'||F >= ||K'||_*.

For the original capacity C_lambda=(1/2)log det(I+K/lambda),

    D_u C_lambda
       = (1/2)tr[(lambda I+K)^(-1) K']
       >= tau/(2lambda) - R T/(2lambda^2)>0              (4.3)

whenever lambda>RT/tau. This follows from
||(lambda I+K)^(-1)-I/lambda||op<=||K||op/lambda^2 and
the operator/nuclear trace inequality. Additive normalization constants in
the capacity do not affect its derivative.

Repeat (4.1)-(4.3) after actually deleting the other first-Conv channels and
the corresponding second-Conv inputs, retaining the existing biases and
head. The target path and positive downstream weights remain, so tau_self>0.
Choose one finite lambda larger than both thresholds. Strict signs persist
on a raw-parameter neighborhood by continuity, even off the special head
factorization. This compares the actual full-network mean drift to the
original full/self directions; it does not claim that the full capacity
value decreases under every simultaneous all-parameter update.

An all-data finite envelope can be used without forming the 12000-by-12000
NTK: evaluate an auxiliary positive network at the all-ones input, replace
the negative first bias by its absolute value, and replace each MaxPool by
the sum of its four entries. Let h_env and D_env be its features and hidden
psi-gradient. Positive path expansions bound h_n<=h_env and D_n<=D_env
coordinatewise for every original image and every chosen max route. Therefore

    ||Hf||F<=sqrt(N)||h_env||2, ||D||F<=sqrt(N)||D_env||2.

Only the target contributes to Hf' and D'. These quantities give finite
upper bounds R_+,T_+ and the alternative sufficient lambda>R_+ T_+/tau.
The envelope is solely a bound; training uses the original MaxPool model.

## 5. From the fixed-state sign to moving ordinary Adam at constant rate

Take a closed supnorm ball of radius r around theta* contained in the common
smooth full/self branch, and small enough that throughout it

    u^T barF(theta)>=d:=d0/2>0,
    D_u C_full(theta)>0, D_u C_self(theta)>0.

This ball is obtained from the separately derived reference margins and
continuity. On a slightly larger compact neighborhood all phase gradients
are uniformly bounded and Lipschitz for every possible label assignment
and shuffle. Extend them outside that neighborhood by bounded Lipschitz
cutoffs when applying the averaging estimates.

Standard globally bias-corrected Adam satisfies ||q_t||infinity<=K_A=73.
Let M=H K_A and choose

    ||theta_0-theta*||infinity<=r/4,
    0<S<r/(4M), 0<eta<S, K=ceil(S/eta).

In K tasks the total raw movement is at most

    M K eta <= M(S+eta) < 2MS < r/2.

Thus every intermediate step has distance less than 3r/4 from theta*, for
every label and shuffle history. A first-exit argument applies this bound
while inside the region and rules out exit. It requires no claimed stability
of the infinite-time process. The auxiliary extension and the original
CNN coincide throughout the whole finite window; nothing is projected.

For the linear observable m=u^T theta, the finite version of
[observable averaging](adam_observable_averaging.md) gives at task boundaries

    m(theta_k)-m(theta_0)
       =-eta sum_(j<k) u^T barF(theta_j)+R_k,

and, with probability at least 1-alpha, simultaneously for k<=K,

    |R_k| <= E_eta :=
       U[C_init eta zeta/(1-zeta)
              +C_move H K eta^2/(1-zeta)]
       +2 U_m eta +K eta omega_m(M eta)
       +2 U_m eta sqrt(K/alpha).                         (5.1)

Here U=||u||1, zeta=max(beta1,sqrt(beta2)), C_init and C_move are the
finite local tracking constants from the taskwise theorem; U_m is the
scalar mean-reward Poisson bound, and omega_m its x log(1/x) modulus.
There is no Taylor-Hessian error for a linear observable. The constants
include the within-task H factor, initial missing history, all old moments,
and global bias correction. The scalar martingale maximal estimate controls
the signed cumulative error without assuming independent optimizer steps.

For fixed S, alpha and geometry, E_eta tends to zero as eta tends to zero:
K eta is bounded, K eta^2 tends to zero, omega_m(M eta) tends to zero, and
eta sqrt(K) tends to zero. Select one fixed positive eta small enough that
E_eta<=dS/2. Then

    m(theta_K)-m(theta_0)
       <=-d K eta+dS/2 <=-dS/2<0

on the stated probability event. The learning rate is constant across all
updates of this run. All other raw parameters train as usual. Because motion
is pathwise bounded, |m(theta_K)-m(theta_0)|<=U M K eta; consequently, if
desired, choosing alpha<d/(d+4UM) also yields a strictly negative
unconditional expected terminal displacement from the same probability
and bounded-motion estimates. This is a finite-window expectation, not a
stationary nonzero velocity or an infinite-time expectation statement.

No normal-rank condition, equilibrium projection, C3 field, or inverse-RMS
moment is needed for this short-clock theorem. The necessary noncircular
inputs are smooth local gradients and the explicit sign (3.1), with the
capacity connection supplied separately by (4.3).

## 6. ReLU's precise role and the unresolved target

Here ReLU's zero branch supplies the exact one-image gradient support, and
its linear positive branch supplies smooth positive-path derivatives.
MaxPool only requires a locally smooth composite route; identical-zero
inactive ties do not transmit gradients. Shared weight changes still change
the preactivations at inactive sites, and those sites are included in u and
m. The proof does not claim that ReLU always sinks, that a unit must die,
or that another activation would fail. No other activation is tested.

The achieved statement is a special but open, all-raw, native ten-class
finite constant-rate example. The general dataset, broadly active filters,
ordinary initialization, prescribed ridge, eta=.001, the actual finite run
length, and any infinite-time lower mean level remain outside this theorem.
This progress must not be recorded as completion of the standard RL-CIFAR
driving-source problem.
