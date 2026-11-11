# Why an upper tail can rise while the body sinks

This derivation is independent of the numerical result. It is not a theorem that shrinking every readout speeds every training problem.

For class logits ell=Va+b and softmax p, one unit receives D_i=sum_c p_c V_ci-V_yi. Activity growth reduces CE iff D_i<0. If the target class is the largest coefficient in that readout column, then

    D_i = -sum_(c != y) p_c (V_yi - V_ci) <= 0.

This increase request exists even though other inputs request decreased activity in the very same unit. A shared weight update can increase its response to one subset and decrease another. Uniform readout shrink preserves column class ordering; it does not create a rule that all units should rise.

Write u=1/C. D_i=V_i^T(p-u)+V_i^T(u-y) exactly. The first term is the uniform-target loss gradient; the second uses the persistent input-label association. Changing V also changes p. A pure positive multiplier of the backward map at fixed p cannot change the sign; a changed probability vector can. At low centered logits with all class-dependent biases scaled consistently, the first term is O(c^2) and the second O(c), but mean-compensated Q is not necessarily in this asymptotic regime. The experiment directly evaluates the actual terms and probabilities.

Fixed readout size also changes how soon a correctly directed activity increase makes the classifier confident. For the conditional scalar logistic example with a correct-class margin m=v a+r and v>0, the increase demand is

    -dL/da = v / (1+exp(v a+r)).

After v->c v, it is c v/(1+exp(c v a+r)). It is smaller as c->0. It can nevertheless exceed the original demand at sufficiently large a because the original classifier is already confident and its residual is exponentially small. This is a nonmonotone conditional comparison, not an immediate universal amplification law. In a multiclass network, other units, output bias, errors and Adam change, so actual matched-state probes and persistent-label controls are needed.

For a fixed unit and its initially deep input set B_i, define m_Bi=w_i^T mean_(n in B_i)(x_n)+beta_i. Then for every input, exactly

    z_ni = m_Bi + w_i^T (x_n - mean_Bi(x)).

Consequently a negative body motion does not prevent a positive top motion: the second, input-specific term can grow faster than the first falls. The matched-body intervention fixes only m_Bi(t), allowing the input-specific term to learn. It is an external bias force and is not a natural parameter update; success identifies possible selective growth under that prescribed sinking trajectory, not the fraction of Q's natural effect mediated by mean motion.

Adam's numerator is linear in past gradients, but its denominator is not. With gradients split into uniform and label parts, m_t=beta1^t*m_start + m_uniform,t + m_label,t (up to arithmetic rounding). In a fixed observed denominator, each source has a defined contribution to the next parameter update. Deleting one source and recomputing future Adam states is a different intervention. Native ELU is monotone but expm1 autograd can round its derivative to zero in float32; receivers can still move through shared weights driven by other inputs. Finite activation changes are evaluated directly, not replaced by phi' dz across a branch change.
