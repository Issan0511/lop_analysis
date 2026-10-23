# ELU reserve freezing 0913 results

Primary preregistered pilot verdict: **MIXED**.
Positive excess harm means freezing the designated group impaired prospective learning more than its current-contribution/W2-norm matched control.
The primary candidate is the intersection of top25 actual near-zero occupancy and bottom25 centered pattern norm. Broad shallow-only results remain separate.

| Activation | Checkpoint | Seed | n target | Target−matched AUC CE | Shallow−matched AUC CE |
|---|---:|---:|---:|---:|---:|
| ELU1 | 20 | 0 | 2 | +0.002567 | +0.001287 |
| ELU1 | 100 | 0 | 4 | +0.004910 | +0.025773 |
| ELU1 | 20 | 1 | 1 | +0.001781 | -0.003232 |
| ELU1 | 100 | 1 | 9 | -0.002854 | +0.013409 |
| ELU1 | 20 | 2 | 1 | +0.003037 | -0.000798 |
| ELU1 | 100 | 2 | 5 | +0.004420 | +0.016694 |
| LR | 20 | 0 | 1 | +0.003985 | +0.008809 |
| LR | 100 | 0 | 4 | -0.005411 | +0.014422 |
| LR | 20 | 1 | 1 | -0.002672 | +0.016509 |
| LR | 100 | 1 | 6 | +0.001173 | +0.015488 |
| LR | 20 | 2 | 2 | -0.011853 | +0.017966 |
| LR | 100 | 2 | 9 | +0.001031 | +0.020915 |

AUC is measured on512 test images disjoint from the512 used for cohort selection and current contribution matching. Final metrics use the9488 non-selection test images.
All freeze interventions begin with identical functions/Adam/task RNG states. Frozen IDs never change; Adam moments continue but actual W1-row and bias updates are blocked.
Limitations: small3-seed directional pilot; finite10-task horizon; relative-quantile cohort; matching quality is reported in cohort JSON, not assumed; cohort sizes may differ across activations; null effects do not imply saturation is harmless.
