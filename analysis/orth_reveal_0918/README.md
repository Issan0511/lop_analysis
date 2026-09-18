# orth_reveal_0918

Task A's new increment perpendicular to its starting weight is tested for input invisibility, exposure at Task B, and subsequent selective counteraction. Ordinary SGD CondA, leaky slopes .1/.7, prior ages20/100/500 tasks,10 seeds. Existing Task A trajectories from `switch_force_0918` are immutable inputs. `run.py` creates the new B and continue-A trajectories; full W/b/v/c all evolve normally.

- Primary protocol: `specs/spec_orth_reveal_0918.md` (46a0664).
- Main runner: dede695; preregistered primary statistics in `report.py` (38463d0).
- `verify_external.py` replays all12 trajectories with the original environment and uninstrumented production SGD; bitwise snapshots required. Its autograd test checks initial B gradients independently.
- `report.py` independently reconstructs decomposition, projections, receiver counteraction, direction loss, and nullspace cancellation from raw arrays using NumPy. Statistical unit is seed; primary18 comparisons use Bonferroni intervals.
- `supplement.py` is a post-primary descriptive heterogeneity check: a pooled reduction in B visibility need not exclude individual units with increases. No new confirmatory test.
- Retained-state follow-up protocol: `specs/spec_orth_reveal_retained_followup_0918.md` (59dc999), explicitly registered after the primary outcomes. `retained.py` (2fcd953) tests actual A-input-null weight, which differs from the new increment perpendicular to starting w. Its12 adjusted secondary contrasts are a separate family and do not replace primary findings.

Run with Python3, NumPy, PyTorch, SciPy and Matplotlib:

```bash
python3 analysis/orth_reveal_0918/run.py --preflight
python3 analysis/orth_reveal_0918/run.py
python3 analysis/orth_reveal_0918/verify_external.py
python3 analysis/orth_reveal_0918/report.py
python3 analysis/orth_reveal_0918/supplement.py
python3 analysis/orth_reveal_0918/retained.py
```

This run used host Python/PyTorch/NumPy; SciPy was provided by `/tmp/switch_force_report_deps_0918` through PYTHONPATH. Raw inputs must exist at the absolute data paths in `run.py`; their SHA256 hashes are in `results/orth_reveal_0918/provenance.json`. Raw outputs and backup scripts live at `/home/issan/Projects/obsidian-research-data/orth_reveal_0918/`. Scripts should not overwrite an archived run when reused: choose a new output folder and record new provenance.

`G` pools squared raw projections within each seed, averaging all15 possible B flips before its RMS ratio. `C+`/`C-` use fixed signs of the inherited field at B start; these are not the signs of full preactivation z. `K` follows the pooled coefficient of a fixed vector, not an independently identifiable physical parcel. Actual-source positive/negative attribution uses each sampled unit's z at each update. Source and receiver groups are deliberately different.

Whole weights are primary; centered weights are explicitly secondary. Source attribution includes actual floating-point updates. No Adam, long-horizon causal inference, or continuous rotation of the reference w is inferred from this two-task experiment.
