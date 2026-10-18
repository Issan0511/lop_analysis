# F51 cap matched-fresh follow-up

Post hoc follow-up requested after original results. All values below are percentage points or percent. Five tasks are averaged per seed; n=10 paired seeds.

| Activation | Fresh ref | Fresh cap | Continual ref | Continual cap |
|---|---:|---:|---:|---:|
| ELU | 65.354 | 65.298 | 64.613 | 62.613 |
| ReLU | 65.027 | 64.991 | 43.661 | 40.059 |

| Activation | Contrast | Mean | Simultaneous interval |
|---|---|---:|---|
| ELU | Df | -0.056 | [-0.084, -0.028] |
| ELU | Dc | -1.999 | [-2.469, -1.529] |
| ELU | I | -1.943 | [-2.416, -1.471] |
| ReLU | Df | -0.036 | [-0.073, +0.002] |
| ReLU | Dc | -3.601 | [-7.486, +0.283] |
| ReLU | I | -3.565 | [-7.450, +0.319] |

Df = fresh cap − fresh ref. Dc = continual cap − continual ref. I = Dc − Df; negative I denotes extra harm associated with history. Non-significance does not establish equality. These reused seeds/results are not independent replications. Cap radii remain fixed at the original task-1 end; this is not a test of all cap strengths.

Both activations passed exact reproduction of the original task-29 no-cap fresh control for every seed, as well as input/stream/initial-state matching and graph/eager checks.
