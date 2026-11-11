# Interpretation guardrails and primary literature (2026-11-11)

This file records equations and source distinctions, not empirical conclusions from the new run. The remembered observation may be CondA (user clarification); its original run and aggregation remain unidentified. The new run is random-label MNIST. No universality of 0.1 is assumed.

For unit i on the fixed N images, p_i=N^{-1} sum_x 1[z_i(x)>0], ignoring the separately defined kink. For 0<=a<=1,

g_i(x)=a+(1-a)1[z_i(x)>0],
mean_x g=a+(1-a)p_i,
mean_x g^2=a^2+(1-a^2)p_i,
var_x g=(1-a)^2 p_i(1-p_i).

The first two means in our tables are then averaged equally over units within a specified layer; seeds are not mixed into this definition. At a=.1, mean g close to.1 indicates negative-branch occupancy close to100%. All inputs weakly responsive and a small subset strongly responsive are different Jacobians. The distribution over units and input images must be retained. These identities are algebraic and are not three independent observations.

The normalized per-unit derivative participation used by the older neff analysis is

n_eff(p,a)=[a+(1-a)p]^2/[a^2+(1-a^2)p].

For 0<a<1, its minimum over p is 4a/(1+a)^2, at p=a/(1+a); it returns to1 at both p=0 and p=1. Thus for a=.1 it CANNOT fall below.33058, even though the unit can become entirely affine on the data. The ELU calibration midpoint .119 from neff_pred_0917 cannot be transported literally to a=.1 leaky. At a=0,p=0, define participation0; the positive-a limit is discontinuous as a measurement convention, not evidence of a network phase transition.

With constant positive K and Gaussian z~N(m,s^2), r=m/s, the simplified shift-force term is proportional to

E[phi_a(z) phi'_a(z)] = s[(1-a^2) normal_pdf(r)+r{a^2+(1-a^2) normal_cdf(r)}].

Changing a changes this sign boundary as well as the derivative floor, nonlinear contrast1-a, features, logits, and the future optimizer trajectory. It is not an intervention on one scalar alone. The zero of this expression is NOT a proven training fixed point. V13 already distinguishes m,s,r and opening; no claim that only the fall of m constitutes the mechanism is intended.

Adam's exact rescaling identity for c>0, if the entire gradient history at a fixed coordinate is scaled, is

Delta theta_c=-eta*c*mhat/(c*sqrt(vhat)+eps)=-eta*mhat/(sqrt(vhat)+eps/c).

Only when eps is negligible and history/directions actually scale does c cancel. With stale history, the next numerators are beta1*m+(1-beta1)c*g and beta2*v+(1-beta2)c^2*g^2, so cancellation does not hold. A change of a generally changes gradient directions and different examples/units unequally. SGD instead gives Delta theta_c=c*Delta theta for an otherwise identical instantaneous gradient. These algebraic controls are not a trained SGD comparison, and the experiment's conclusions remain conditional on Adam.

The measured useful response is delta f approximately J_true delta theta. Let e=softmax(f)-one_hot(y). The predicted full-data cross-entropy decrease is -<e,J_true delta theta>/N. We also record the realized f_after-f_before, actual CE decrease, its error alignment, and hidden-only JVP. This tests whether motion serves the NEW labels; a large parameter step or a high scalar mean derivative alone is insufficient. In surrogate-B arms, J_true is still the natural forward Jacobian; the backward-rule Jacobian is different.

In the fixed-kernel, squared-loss gradient-flow model, error mode k decays as e_k(t)=e_k(0)exp(-eta*lambda_k*t). Finite-budget success therefore depends on eta*B*lambda_k and on the error's projections, not a universal derivative mean. This is a diagnostic analogy, not a fitted dynamical model for the current finite-width Adam/CE network. Persistent failure at24k is not a proof of impossibility; recovery after6k demonstrates a budget limitation only for that branch. Positive/negative branch occupancy does not by itself establish total network linearity or effective capacity.

Primary sources actually opened during this investigation:

| Source | Relevant established result | Difference from this experiment |
|---|---|---|
| [Kingma & Ba, Adam (ICLR2015), Algorithm1 and section2](https://arxiv.org/pdf/1412.6980) | Adaptive first/second moments and rescaling discussion. | Fixed-history scaling differs from suddenly changing a; epsilon retained in our algebra and measurement. |
| [Jacot et al., NTK (NeurIPS2018), section5](https://arxiv.org/pdf/1806.07572) | Fixed-kernel least-squares dynamics decompose into exponentially decaying modes. | Not a guarantee for finite100-wide Adam networks or CE; spectra/projections can change during training. |
| [Dohare et al., Nature2024, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/) | Dead/constant units, weight growth and representation diversity accompany plasticity loss; original study includes online PMNIST. | PMNIST is not random-label MNIST; unit death and low derivative mean are distinct. |
| [Lyle et al., preprint2024 sections3 and appendixE.8](https://arxiv.org/html/2402.18762v1) ([published PMLR2025](https://proceedings.mlr.press/v274/lyle25a.html)) | Several independently acting mechanisms; unit linearization and empirical tangent-kernel degeneracies are analyzed. | Supports separating mechanisms, not a particular0.1 threshold or capacity diagnosis for our box. |
| [Horuz et al., The Resurrection of the ReLU (2025), section3–4](https://arxiv.org/html/2505.22074v1) | Natural ReLU forward with surrogate backward; surrogate choice has different results. | Their supervised generalization experiments differ from our continual adaptation and same-state a control. New B uses existing bwdswap logic, not a novel technique. |
| [Lillo & Cheney, Activation Function Design Sustains Plasticity (2026 revision), section3, appendixB](https://arxiv.org/html/2509.22562v2) | Negative-slope sweeps in a4-layer CNN show a setting-dependent favorable intermediate range and poor extremes. | Their effective NEGATIVE-branch slope, class-incremental replay metrics and tuned learning rates differ from our whole-input mean, random-label memorization and fixed lr. Their range is not imported as a prediction or threshold. |
| [Lu et al., Dying ReLU and Initialization (2020)](https://arxiv.org/pdf/1903.06733) | Theoretical dying-ReLU behavior under specified initialization/depth conditions. | Not a proof of an absorbing transition along these continual positive-leak trajectories. |

Relevant local corrections checked: grad_floor_0911 concerns gradient-noise distribution in PMNIST (not an a-floor intervention); long_horizon_0910 is PMNIST; leaky_group_ablation_0914 is CondA32-input scalar regression, not MNIST. neff_pred_0917 gives BOX_SPECIFIC, not a universal threshold. eps_rejudge_0917 distinguishes expm1-backward float32 zeros from analytic exp and Adam epsilon; respdyn_ee_0917 float64 native-exp collapse excludes exact float32 zero as necessary for the observed loss. head0_mlp_1108 clones S at EACH boundary into a separate one-task branch (spec section3), so its H0A recovery is not an every-task preventive trajectory. The new H likewise resets ONCE from task20. Capacity claims in V14 remain conditional on the measured task, budget and later corrections.
