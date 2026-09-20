# A5 implementation

CPU-only initial-geometry measurement for native float32 host parameters, evaluated
in float64. No optimizer, label draw or training is used. Each seed's eight L1
input predictions are written before initializing/reading W responses. Layer2
predictions are explicitly conditional on measured a1. Random bias is primary;
both-bias-zero, bias-adjusted r and per-unit Gaussian proxies are report-only.

The current authorization covers implementation and tests, not scientific seeds.
Only test seeds100–101 are used by checks. A later production run uses20–39 after
rechecking local seed metadata. An overlap aborts and requires a new preregistration;
the runner never substitutes a seed silently.

```bash
python -m analysis.initgeom_cifar_0920.checks --out results/_checks_initgeom_cifar_0920/attempt
# After a separate production GO:
python -m analysis.initgeom_cifar_0920.run --mode production --production-go --checks CHECKS.json --out results/initgeom_cifar_0920/run
python -m analysis.initgeom_cifar_0920.report --src results/initgeom_cifar_0920/run --out results/initgeom_cifar_0920/report
```

`STOP` inside the run directory stops at the next seed boundary. Remove it and use
`--resume`; source/input/config and completed file checksums must match. Already
written predictions and responses are immutable. Raw arrays and initial parameters
are included in per-seed manifests. Test/partial/foreign results cannot produce
production scientific labels.

`stats.py` uses log-domain binomial PMFs and Poisson-binomial DP for the two central
order statistics; the band is conservative under that approximate model. Each
layer has its own eight-condition family. The raw/gamma100 alias never increases
the family or sample size. `geometry.py` propagates transform and forward roundoff
to signs and keeps ambiguous units in the denominator. A centered, highly skewed
1200-point counterexample prevents hardcoding centered f=0.

Synthetic fixtures independently reconstruct initialization, use scalar fsum dots,
host double forward, permutations, exact integer thresholds, analytic normal
populations and exhaustive discrete order-statistic distributions. Full test-seed
checks exercise the entire pipeline twice, including seed-boundary resume,
immutable predictions, corruption and completeness guards, cost and peak RSS.
