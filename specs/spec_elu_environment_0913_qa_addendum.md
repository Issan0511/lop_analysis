# Additional technical validation: ELU environment 0913
Before full training/outcome inspection. Original spec and all its tolerances remain unchanged.

Initial random-label float32 CPU-autograd versus GPU analytic Adam probes failed the original parameter tolerance (<2e-5), despite gradient maxabs<2.3e-8: parameter maxabs2.50e-5 for centered synthetic inputs and4.54e-5 for uniform[0,1] inputs. An allclass0-label probe passed but is only supplemental. These failures are retained in numerical_preflight.json.

Code review found that the new initializer divided by sqrt(fanin), whereas historical pmnist_0905.init_params multiplied by the precomputed reciprocal. The initializer now uses the historical expression exactly. With corrected initialization, the uniform[0,1] random-labelled synthetic probe satisfies the ORIGINAL tolerance: parameter maxabs1.44e-5. No tolerance is relaxed and no easier-label probe replaces random-labelled validation.

Additional validation gates before full training:
- Same GPU gradients supplied to CPU torch.optim.Adam must reproduce the batched GPU update within1e-6.
- A doubled-learning-rate mutation with those same gradients must differ by>5e-4.
- Graph/eager25-step equivalence is tested with random labels, active norm projection and Adam stepcounter initially73; counter must advance by25, with maxabs<5e-5 and clamp norm relative error<2e-5.
- Historical pmnist_0905.init_params is imported read-only and compared for seeds0,1,2 with exact tensor equality for all6parameters, including multiplication order. Source SHA is saved.

All data, architecture, optimizer, intervention,50-task protocol, endpoints, windows, seed count and decision thresholds are unchanged. Float32 CPU/GPU training trajectories are not promised bitwise equality. Full run begins only after coordinator commits and pushes this addendum and implementation.
