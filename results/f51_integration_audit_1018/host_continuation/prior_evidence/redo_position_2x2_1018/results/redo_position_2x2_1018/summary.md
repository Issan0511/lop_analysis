# ReDo × position — implementation and attempted launch

Status: **BLOCKED_CUDA**. No evaluation seeds were trained and no performance
label was assigned. The registered paper-derived prediction is POSITION_CARRIES.

The real production `run` command was invoked after committing the source. It
exited with code 2 at admission, before loading CIFAR or creating training
checkpoints. `launch_preflight.json` records the command, source hashes,
environment, audit/check hashes, registered protocol and complete blockers:

- `torch.cuda.is_available()` is false; no NVIDIA device nodes exist and
  `nvidia-smi` cannot contact a driver.
- The independent G0 audit has `all_pass`, `host_reproduction_pass` and
  `mutation_checks_pass` false. Static review is not counted as a CUDA replay.

CPU synthetic validation passed 7 check groups and detected all 14 specified
mutation negatives. It executes actual inherited eager host steps and the
unchanged inherited recycler; the float32 physical-bias search is checked
against independent exact count/tie oracles. Readout checks cover all four
labels, overlap, seven paired intervals, seed joins, strict/inclusive boundaries,
missing/duplicate rows and divergence. These checks do not constitute a
production run or evidence for the predicted performance effect.

Run the documented full CLI after CUDA and independent G0 are available.
The implementation does not reduce R, seed count, task count, updates, or switch
the registered GPU protocol to CPU.
