# Loading-push path diagnostics

No model parameters or optimizer states are updated. The tar archive is read in
memory; the original input bundle and previous raw arrays remain unmodified.

Run in the experiment checkout with the project's Python environment:

```bash
python analysis/chargpt_loadsign_1023/measure.py gate
python analysis/chargpt_loadsign_1023/measure.py run
python analysis/chargpt_loadsign_1023/report.py
```

The gate verifies the bundle SHA256 and eight states against both the previous
loading push and autograd. All 700 states are subsequently compared against the
previous per-unit arrays (including exact sign fractions and alive masks).
Source, probe, dtype, and readout definitions are reused from the previous run.

For final pre-LN residual t, q=(t-mean(t))/sigma and b=gamma*(e@vL), the
transpose LN Jacobian is

    J_LN^T(e@vL) = [b-mean(b)-q*mean(q*b)]/sigma.

The `direct` diagnostic retains the first two terms. It keeps the original
forward LN, sigma, and logits. This is a change to the local derivative, not an
experiment that removes LN from the network. `exact` retains all terms.
`loaderror` changes only e to softmax(h@vL.T)-uniform; it does not change the
forward representation or tied token embeddings. `absK` changes only the sign
of the token weight K=y_uniform@ybar_task.

The observed own-weight push is obtained by multiplying the pullback by the
FFN down columns, activation derivative, and K, then averaging all tokens.
No derivative of the upstream representation with respect to its own parameters
is included. Saved representations may differ across tasks and enter the fixed
coefficient K; this is distinct from movement caused by upstream updates.

All main s values use the same alive set defined on the current-task probe.
Units alive there but inactive on the uniform gradient probe may have exactly
zero push. These remain in the main denominator and are recorded separately.
The cancellation ratio excludes 0/0 and reports its missing count.

ReLUcap is a training-time weight constraint in the original implementation;
its fixed-state activation and derivative are ordinary ReLU. No activation
clipping is introduced by this diagnostic.

Results are descriptive over 10 saved seeds. A local diagnostic intervention
cannot establish the historical learning process that created the state.
